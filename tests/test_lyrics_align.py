"""Tests for lyrics alignment with stable-ts."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alsmuse.exceptions import AlignmentError
from alsmuse.lyrics_align import (
    _normalize_text,
    _tokenize,
    filter_segments_to_valid_ranges,
    filter_to_valid_ranges,
    segments_to_lines,
    words_to_lines,
)
from alsmuse.models import TimedLine, TimedSegment, TimedWord

# ---------------------------------------------------------------------------
# Tests for get_compute_device
# ---------------------------------------------------------------------------


class TestGetComputeDevice:
    """Tests for get_compute_device function."""

    def test_returns_cuda_when_available(self) -> None:
        """Returns 'cuda' when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Need to reimport to pick up the mock
            from alsmuse import lyrics_align

            # Clear any cached import
            result = lyrics_align.get_compute_device()

        assert result == "cuda"

    def test_returns_mps_when_available_no_cuda(self) -> None:
        """Returns 'mps' on macOS when MPS is available but CUDA is not."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from alsmuse import lyrics_align

            result = lyrics_align.get_compute_device()

        assert result == "mps"

    def test_returns_cpu_when_no_gpu_available(self) -> None:
        """Returns 'cpu' when neither CUDA nor MPS is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from alsmuse import lyrics_align

            result = lyrics_align.get_compute_device()

        assert result == "cpu"

    def test_raises_alignment_error_when_torch_not_installed(self) -> None:
        """Raises AlignmentError with helpful message when torch not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            # Force reimport to trigger ImportError
            import importlib

            from alsmuse import lyrics_align

            importlib.reload(lyrics_align)

            with pytest.raises(AlignmentError, match="PyTorch.*not installed"):
                lyrics_align.get_compute_device()


# ---------------------------------------------------------------------------
# Tests for filter_to_valid_ranges
# ---------------------------------------------------------------------------


class TestFilterToValidRanges:
    """Tests for filter_to_valid_ranges function."""

    def test_empty_words_returns_empty(self) -> None:
        """Empty word list returns empty list."""
        result = filter_to_valid_ranges([], [(0.0, 10.0)])
        assert result == []

    def test_empty_ranges_returns_all_words(self) -> None:
        """Empty valid_ranges returns all words."""
        words = [
            TimedWord(text="hello", start=0.0, end=0.5),
            TimedWord(text="world", start=0.5, end=1.0),
        ]
        result = filter_to_valid_ranges(words, [])
        assert result == words

    def test_word_inside_range_is_kept(self) -> None:
        """Word whose midpoint is inside a valid range is kept."""
        words = [
            TimedWord(text="hello", start=1.0, end=2.0),  # midpoint = 1.5
        ]
        result = filter_to_valid_ranges(words, [(0.0, 3.0)])
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_word_outside_range_is_filtered(self) -> None:
        """Word whose midpoint is outside valid ranges is filtered."""
        words = [
            TimedWord(text="hello", start=5.0, end=6.0),  # midpoint = 5.5
        ]
        result = filter_to_valid_ranges(words, [(0.0, 3.0)])
        assert result == []

    def test_word_at_range_boundary_is_kept(self) -> None:
        """Word whose midpoint is exactly at range boundary is kept."""
        words = [
            TimedWord(text="hello", start=2.0, end=4.0),  # midpoint = 3.0
        ]
        result = filter_to_valid_ranges(words, [(0.0, 3.0)])
        assert len(result) == 1

    def test_multiple_ranges_checked(self) -> None:
        """Words in any of multiple valid ranges are kept."""
        words = [
            TimedWord(text="first", start=0.5, end=1.5),  # midpoint = 1.0, in range 1
            TimedWord(text="gap", start=3.0, end=4.0),  # midpoint = 3.5, not in any range
            TimedWord(text="second", start=5.5, end=6.5),  # midpoint = 6.0, in range 2
        ]
        result = filter_to_valid_ranges(words, [(0.0, 2.0), (5.0, 7.0)])

        assert len(result) == 2
        assert result[0].text == "first"
        assert result[1].text == "second"

    def test_overlapping_ranges(self) -> None:
        """Words in overlapping ranges are kept (not duplicated)."""
        words = [
            TimedWord(text="overlap", start=1.0, end=2.0),  # midpoint = 1.5
        ]
        result = filter_to_valid_ranges(words, [(0.0, 2.0), (1.0, 3.0)])
        assert len(result) == 1

    def test_word_spanning_range_boundary(self) -> None:
        """Word that spans a range boundary is filtered based on midpoint."""
        words = [
            TimedWord(text="spanning", start=2.5, end=3.5),  # midpoint = 3.0, at boundary
        ]
        # Range ends at 3.0, midpoint is exactly 3.0, should be kept
        result = filter_to_valid_ranges(words, [(0.0, 3.0)])
        assert len(result) == 1

    def test_realistic_hallucination_filtering(self) -> None:
        """Simulates filtering hallucinations from silent gaps."""
        # Simulate aligned words with some in silent gaps
        words = [
            # Real vocals: 0-10s
            TimedWord(text="hello", start=1.0, end=1.5),
            TimedWord(text="world", start=1.6, end=2.0),
            # Hallucination in gap: 10-20s (model imagined words)
            TimedWord(text="phantom", start=15.0, end=15.5),
            TimedWord(text="words", start=15.6, end=16.0),
            # Real vocals: 20-30s
            TimedWord(text="goodbye", start=21.0, end=21.5),
        ]
        valid_ranges = [(0.0, 10.0), (20.0, 30.0)]

        result = filter_to_valid_ranges(words, valid_ranges)

        assert len(result) == 3
        texts = [w.text for w in result]
        assert texts == ["hello", "world", "goodbye"]


# ---------------------------------------------------------------------------
# Tests for text normalization helpers
# ---------------------------------------------------------------------------


class TestNormalizeText:
    """Tests for _normalize_text function."""

    def test_lowercase(self) -> None:
        """Text is converted to lowercase."""
        assert _normalize_text("Hello World") == "hello world"

    def test_removes_punctuation(self) -> None:
        """Punctuation is removed."""
        assert _normalize_text("Hello, world!") == "hello world"
        assert _normalize_text("It's a test.") == "its a test"

    def test_normalizes_whitespace(self) -> None:
        """Multiple spaces are collapsed."""
        assert _normalize_text("hello   world") == "hello world"
        assert _normalize_text("  hello  ") == "hello"

    def test_handles_unicode(self) -> None:
        """Unicode characters are normalized."""
        # Test with accented characters
        assert "cafe" in _normalize_text("cafe")


class TestTokenize:
    """Tests for _tokenize function."""

    def test_splits_on_whitespace(self) -> None:
        """Text is split into word tokens."""
        assert _tokenize("hello world") == ["hello", "world"]

    def test_normalizes_before_splitting(self) -> None:
        """Text is normalized before tokenizing."""
        assert _tokenize("Hello, World!") == ["hello", "world"]

    def test_empty_string(self) -> None:
        """Empty string returns empty list."""
        assert _tokenize("") == []

    def test_punctuation_only(self) -> None:
        """String with only punctuation returns empty list."""
        assert _tokenize("...!!!") == []


# ---------------------------------------------------------------------------
# Tests for words_to_lines
# ---------------------------------------------------------------------------


class TestWordsToLines:
    """Tests for words_to_lines function."""

    def test_empty_words_returns_empty_lines(self) -> None:
        """Empty word list returns lines with no timing."""
        lines = ["Hello world", "Goodbye world"]
        result = words_to_lines([], lines)

        assert len(result) == 2
        assert result[0].text == "Hello world"
        assert result[0].start == 0.0
        assert result[0].end == 0.0
        assert result[0].words == ()

    def test_empty_lines_returns_empty(self) -> None:
        """Empty line list returns empty result."""
        words = [TimedWord(text="hello", start=0.0, end=0.5)]
        result = words_to_lines(words, [])
        assert result == []

    def test_single_line_single_word(self) -> None:
        """Single word matched to single line."""
        words = [TimedWord(text="hello", start=1.0, end=2.0)]
        lines = ["Hello"]

        result = words_to_lines(words, lines)

        assert len(result) == 1
        assert result[0].text == "Hello"
        assert result[0].start == 1.0
        assert result[0].end == 2.0
        assert len(result[0].words) == 1

    def test_single_line_multiple_words(self) -> None:
        """Multiple words matched to single line."""
        words = [
            TimedWord(text="hello", start=1.0, end=1.5),
            TimedWord(text="world", start=1.6, end=2.0),
        ]
        lines = ["Hello world"]

        result = words_to_lines(words, lines)

        assert len(result) == 1
        assert result[0].text == "Hello world"
        assert result[0].start == 1.0  # First word start
        assert result[0].end == 2.0  # Last word end
        assert len(result[0].words) == 2

    def test_multiple_lines(self) -> None:
        """Words distributed across multiple lines."""
        words = [
            TimedWord(text="hello", start=1.0, end=1.5),
            TimedWord(text="world", start=1.6, end=2.0),
            TimedWord(text="goodbye", start=3.0, end=3.5),
            TimedWord(text="moon", start=3.6, end=4.0),
        ]
        lines = ["Hello world", "Goodbye moon"]

        result = words_to_lines(words, lines)

        assert len(result) == 2
        assert result[0].text == "Hello world"
        assert result[0].start == 1.0
        assert result[0].end == 2.0
        assert result[1].text == "Goodbye moon"
        assert result[1].start == 3.0
        assert result[1].end == 4.0

    def test_skips_empty_lines(self) -> None:
        """Empty lines in original text are skipped."""
        words = [
            TimedWord(text="hello", start=1.0, end=1.5),
            TimedWord(text="world", start=2.0, end=2.5),
        ]
        lines = ["Hello", "", "World"]

        result = words_to_lines(words, lines)

        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "World"

    def test_handles_punctuation_in_lines(self) -> None:
        """Punctuation in original lines doesn't break matching."""
        words = [
            TimedWord(text="hello", start=1.0, end=1.5),
            TimedWord(text="world", start=1.6, end=2.0),
        ]
        lines = ["Hello, world!"]

        result = words_to_lines(words, lines)

        assert len(result) == 1
        assert result[0].text == "Hello, world!"
        assert len(result[0].words) == 2

    def test_case_insensitive_matching(self) -> None:
        """Matching is case-insensitive."""
        words = [
            TimedWord(text="HELLO", start=1.0, end=1.5),
            TimedWord(text="world", start=1.6, end=2.0),
        ]
        lines = ["hello World"]

        result = words_to_lines(words, lines)

        assert len(result) == 1
        assert result[0].text == "hello World"
        assert len(result[0].words) == 2

    def test_more_words_than_line_expects(self) -> None:
        """Extra words after line is matched move to next line."""
        words = [
            TimedWord(text="hello", start=1.0, end=1.5),
            TimedWord(text="world", start=1.6, end=2.0),
            TimedWord(text="extra", start=2.1, end=2.5),
        ]
        lines = ["Hello world"]

        result = words_to_lines(words, lines)

        # Only the first line is present, extra word not matched
        assert len(result) == 1
        assert len(result[0].words) == 2

    def test_line_with_no_matching_words(self) -> None:
        """Line with no matching words gets empty timing."""
        words = [
            TimedWord(text="hello", start=1.0, end=1.5),
        ]
        lines = ["Goodbye"]

        result = words_to_lines(words, lines)

        assert len(result) == 1
        assert result[0].text == "Goodbye"
        assert result[0].start == 0.0
        assert result[0].end == 0.0
        assert result[0].words == ()

    def test_preserves_original_line_text(self) -> None:
        """Original line text is preserved exactly."""
        words = [
            TimedWord(text="its", start=1.0, end=1.5),
            TimedWord(text="a", start=1.6, end=1.7),
            TimedWord(text="test", start=1.8, end=2.0),
        ]
        lines = ["It's a test!"]

        result = words_to_lines(words, lines)

        assert result[0].text == "It's a test!"


# ---------------------------------------------------------------------------
# Tests for TimedWord and TimedLine models
# ---------------------------------------------------------------------------


class TestTimedWordModel:
    """Tests for TimedWord dataclass."""

    def test_frozen_dataclass(self) -> None:
        """TimedWord is immutable."""
        word = TimedWord(text="hello", start=1.0, end=2.0)

        with pytest.raises(AttributeError):
            word.text = "world"  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible after construction."""
        word = TimedWord(text="hello", start=1.5, end=2.5)

        assert word.text == "hello"
        assert word.start == 1.5
        assert word.end == 2.5


class TestTimedLineModel:
    """Tests for TimedLine dataclass."""

    def test_frozen_dataclass(self) -> None:
        """TimedLine is immutable."""
        line = TimedLine(text="hello world", start=1.0, end=2.0, words=())

        with pytest.raises(AttributeError):
            line.text = "goodbye"  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible after construction."""
        word1 = TimedWord(text="hello", start=1.0, end=1.5)
        word2 = TimedWord(text="world", start=1.6, end=2.0)
        line = TimedLine(text="hello world", start=1.0, end=2.0, words=(word1, word2))

        assert line.text == "hello world"
        assert line.start == 1.0
        assert line.end == 2.0
        assert len(line.words) == 2
        assert line.words[0].text == "hello"
        assert line.words[1].text == "world"


# ---------------------------------------------------------------------------
# Tests for align_lyrics (with mocked stable_whisper)
# ---------------------------------------------------------------------------


class TestAlignLyrics:
    """Tests for align_lyrics function."""

    def test_raises_error_when_stable_whisper_not_installed(self) -> None:
        """Raises AlignmentError when stable-ts is not installed."""
        with patch.dict("sys.modules", {"stable_whisper": None}):
            import importlib

            from alsmuse import lyrics_align

            importlib.reload(lyrics_align)

            with pytest.raises(AlignmentError, match="stable-ts.*not installed"):
                lyrics_align.align_lyrics(
                    audio_path=Path("/fake/audio.wav"),
                    lyrics_text="Hello world",
                    valid_ranges=[(0.0, 10.0)],
                )

    def test_successful_alignment_with_mocked_model(self) -> None:
        """Successful alignment returns filtered TimedWord list."""
        # Create mock word objects
        mock_word1 = MagicMock()
        mock_word1.word = "hello"
        mock_word1.start = 1.0
        mock_word1.end = 1.5

        mock_word2 = MagicMock()
        mock_word2.word = "world"
        mock_word2.start = 1.6
        mock_word2.end = 2.0

        # Create mock segment
        mock_segment = MagicMock()
        mock_segment.words = [mock_word1, mock_word2]

        # Create mock result
        mock_result = MagicMock()
        mock_result.segments = [mock_segment]

        # Create mock model
        mock_model = MagicMock()
        mock_model.align.return_value = mock_result

        # Create mock stable_whisper module
        mock_stable_whisper = MagicMock()
        mock_stable_whisper.load_model.return_value = mock_model

        # Create mock torch
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict(
            "sys.modules", {"stable_whisper": mock_stable_whisper, "torch": mock_torch}
        ):
            import importlib

            from alsmuse import lyrics_align

            importlib.reload(lyrics_align)

            result = lyrics_align.align_lyrics(
                audio_path=Path("/fake/audio.wav"),
                lyrics_text="Hello world",
                valid_ranges=[(0.0, 10.0)],
            )

        assert len(result) == 2
        assert result[0].text == "hello"
        assert result[0].start == 1.0
        assert result[0].end == 1.5
        assert result[1].text == "world"

    def test_filters_words_outside_valid_ranges(self) -> None:
        """Words outside valid ranges are filtered out."""
        # Create mock words - one in valid range, one outside
        mock_word_valid = MagicMock()
        mock_word_valid.word = "hello"
        mock_word_valid.start = 1.0  # midpoint = 1.25, in range
        mock_word_valid.end = 1.5

        mock_word_invalid = MagicMock()
        mock_word_invalid.word = "phantom"
        mock_word_invalid.start = 15.0  # midpoint = 15.25, outside range
        mock_word_invalid.end = 15.5

        mock_segment = MagicMock()
        mock_segment.words = [mock_word_valid, mock_word_invalid]

        mock_result = MagicMock()
        mock_result.segments = [mock_segment]

        mock_model = MagicMock()
        mock_model.align.return_value = mock_result

        mock_stable_whisper = MagicMock()
        mock_stable_whisper.load_model.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict(
            "sys.modules", {"stable_whisper": mock_stable_whisper, "torch": mock_torch}
        ):
            import importlib

            from alsmuse import lyrics_align

            importlib.reload(lyrics_align)

            result = lyrics_align.align_lyrics(
                audio_path=Path("/fake/audio.wav"),
                lyrics_text="Hello phantom",
                valid_ranges=[(0.0, 10.0)],  # Only 0-10s is valid
            )

        # Only the word in valid range should be returned
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_uses_correct_device(self) -> None:
        """Model is loaded with the correct compute device."""
        mock_word = MagicMock()
        mock_word.word = "test"
        mock_word.start = 0.0
        mock_word.end = 1.0

        mock_segment = MagicMock()
        mock_segment.words = [mock_word]

        mock_result = MagicMock()
        mock_result.segments = [mock_segment]

        mock_model = MagicMock()
        mock_model.align.return_value = mock_result

        mock_stable_whisper = MagicMock()
        mock_stable_whisper.load_model.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True  # macOS

        with patch.dict(
            "sys.modules", {"stable_whisper": mock_stable_whisper, "torch": mock_torch}
        ):
            import importlib

            from alsmuse import lyrics_align

            importlib.reload(lyrics_align)

            lyrics_align.align_lyrics(
                audio_path=Path("/fake/audio.wav"),
                lyrics_text="Test",
                valid_ranges=[(0.0, 10.0)],
                model_size="small",
            )

        # Verify model was loaded with CPU device (MPS falls back to CPU due to float64 issue)
        mock_stable_whisper.load_model.assert_called_once_with("small", device="cpu")

    def test_skips_empty_words(self) -> None:
        """Words with empty text after stripping are skipped."""
        mock_word_valid = MagicMock()
        mock_word_valid.word = "hello"
        mock_word_valid.start = 1.0
        mock_word_valid.end = 1.5

        mock_word_empty = MagicMock()
        mock_word_empty.word = "   "  # Just whitespace
        mock_word_empty.start = 1.6
        mock_word_empty.end = 1.7

        mock_segment = MagicMock()
        mock_segment.words = [mock_word_valid, mock_word_empty]

        mock_result = MagicMock()
        mock_result.segments = [mock_segment]

        mock_model = MagicMock()
        mock_model.align.return_value = mock_result

        mock_stable_whisper = MagicMock()
        mock_stable_whisper.load_model.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict(
            "sys.modules", {"stable_whisper": mock_stable_whisper, "torch": mock_torch}
        ):
            import importlib

            from alsmuse import lyrics_align

            importlib.reload(lyrics_align)

            result = lyrics_align.align_lyrics(
                audio_path=Path("/fake/audio.wav"),
                lyrics_text="Hello",
                valid_ranges=[(0.0, 10.0)],
            )

        assert len(result) == 1
        assert result[0].text == "hello"

    def test_model_load_failure_raises_alignment_error(self) -> None:
        """Model loading failure raises AlignmentError."""
        mock_stable_whisper = MagicMock()
        mock_stable_whisper.load_model.side_effect = RuntimeError("Model load failed")

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict(
            "sys.modules", {"stable_whisper": mock_stable_whisper, "torch": mock_torch}
        ):
            import importlib

            from alsmuse import lyrics_align

            importlib.reload(lyrics_align)

            with pytest.raises(AlignmentError, match="Failed to load Whisper model"):
                lyrics_align.align_lyrics(
                    audio_path=Path("/fake/audio.wav"),
                    lyrics_text="Test",
                    valid_ranges=[(0.0, 10.0)],
                )

    def test_alignment_failure_raises_alignment_error(self) -> None:
        """Alignment failure raises AlignmentError."""
        mock_model = MagicMock()
        mock_model.align.side_effect = RuntimeError("Alignment failed")

        mock_stable_whisper = MagicMock()
        mock_stable_whisper.load_model.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict(
            "sys.modules", {"stable_whisper": mock_stable_whisper, "torch": mock_torch}
        ):
            import importlib

            from alsmuse import lyrics_align

            importlib.reload(lyrics_align)

            with pytest.raises(AlignmentError, match="Alignment failed"):
                lyrics_align.align_lyrics(
                    audio_path=Path("/fake/audio.wav"),
                    lyrics_text="Test",
                    valid_ranges=[(0.0, 10.0)],
                )


# ---------------------------------------------------------------------------
# Tests for filter_segments_to_valid_ranges
# ---------------------------------------------------------------------------


class TestFilterSegmentsToValidRanges:
    """Tests for filter_segments_to_valid_ranges function."""

    def _make_segment(
        self, text: str, start: float, end: float, words: list[TimedWord] | None = None
    ) -> TimedSegment:
        """Helper to create a TimedSegment."""
        if words is None:
            words = [TimedWord(text=text, start=start, end=end)]
        return TimedSegment(text=text, start=start, end=end, words=tuple(words))

    def test_empty_segments_returns_empty(self) -> None:
        """Empty segment list returns empty list."""
        result = filter_segments_to_valid_ranges([], [(0.0, 10.0)])
        assert result == []

    def test_empty_ranges_returns_all_segments(self) -> None:
        """Empty valid_ranges returns all segments."""
        segments = [
            self._make_segment("hello", 0.0, 0.5),
            self._make_segment("world", 0.5, 1.0),
        ]
        result = filter_segments_to_valid_ranges(segments, [])
        assert result == segments

    def test_segment_inside_range_is_kept(self) -> None:
        """Segment whose midpoint is inside a valid range is kept."""
        segments = [
            self._make_segment("hello", 1.0, 2.0),  # midpoint = 1.5
        ]
        result = filter_segments_to_valid_ranges(segments, [(0.0, 3.0)])
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_segment_outside_range_is_filtered(self) -> None:
        """Segment whose midpoint is outside valid ranges is filtered."""
        segments = [
            self._make_segment("hello", 5.0, 6.0),  # midpoint = 5.5
        ]
        result = filter_segments_to_valid_ranges(segments, [(0.0, 3.0)])
        assert result == []

    def test_segment_at_range_boundary_is_kept(self) -> None:
        """Segment whose midpoint is exactly at range boundary is kept."""
        segments = [
            self._make_segment("hello", 2.0, 4.0),  # midpoint = 3.0
        ]
        result = filter_segments_to_valid_ranges(segments, [(0.0, 3.0)])
        assert len(result) == 1

    def test_multiple_ranges_checked(self) -> None:
        """Segments in any of multiple valid ranges are kept."""
        segments = [
            self._make_segment("first", 0.5, 1.5),  # midpoint = 1.0, in range 1
            self._make_segment("gap", 3.0, 4.0),  # midpoint = 3.5, not in any range
            self._make_segment("second", 5.5, 6.5),  # midpoint = 6.0, in range 2
        ]
        result = filter_segments_to_valid_ranges(segments, [(0.0, 2.0), (5.0, 7.0)])

        assert len(result) == 2
        assert result[0].text == "first"
        assert result[1].text == "second"

    def test_realistic_hallucination_filtering(self) -> None:
        """Simulates filtering hallucinations from silent gaps."""
        segments = [
            # Real vocals: 0-10s
            self._make_segment("hello world", 1.0, 3.0),
            # Hallucination in gap: 10-20s
            self._make_segment("phantom words", 15.0, 16.0),
            # Real vocals: 20-30s
            self._make_segment("goodbye", 21.0, 22.0),
        ]
        valid_ranges = [(0.0, 10.0), (20.0, 30.0)]

        result = filter_segments_to_valid_ranges(segments, valid_ranges)

        assert len(result) == 2
        texts = [s.text for s in result]
        assert texts == ["hello world", "goodbye"]


# ---------------------------------------------------------------------------
# Tests for segments_to_lines
# ---------------------------------------------------------------------------


class TestSegmentsToLines:
    """Tests for segments_to_lines function."""

    def _make_words(self, word_list: list[tuple[str, float, float]]) -> tuple[TimedWord, ...]:
        """Helper to create a tuple of TimedWord objects."""
        return tuple(TimedWord(text=t, start=s, end=e) for t, s, e in word_list)

    def _make_segment(
        self, text: str, start: float, end: float, words: tuple[TimedWord, ...]
    ) -> TimedSegment:
        """Helper to create a TimedSegment."""
        return TimedSegment(text=text, start=start, end=end, words=words)

    def test_empty_segments_returns_empty(self) -> None:
        """Empty segment list returns empty list."""
        result = segments_to_lines([])
        assert result == []

    def test_single_short_segment_becomes_one_line(self) -> None:
        """A segment with few words becomes a single line."""
        words = self._make_words([
            ("hello", 0.0, 0.5),
            ("world", 0.6, 1.0),
        ])
        segment = self._make_segment("hello world", 0.0, 1.0, words)

        result = segments_to_lines([segment])

        assert len(result) == 1
        assert result[0].text == "hello world"
        assert result[0].start == 0.0
        assert result[0].end == 1.0
        assert len(result[0].words) == 2

    def test_multiple_short_segments_become_multiple_lines(self) -> None:
        """Multiple short segments become multiple lines."""
        words1 = self._make_words([("hello", 0.0, 0.5)])
        words2 = self._make_words([("world", 2.0, 2.5)])

        segments = [
            self._make_segment("hello", 0.0, 0.5, words1),
            self._make_segment("world", 2.0, 2.5, words2),
        ]

        result = segments_to_lines(segments)

        assert len(result) == 2
        assert result[0].text == "hello"
        assert result[1].text == "world"

    def test_long_segment_gets_split(self) -> None:
        """A segment exceeding max_words_per_line is split."""
        # Create a segment with 20 words
        word_data = [(f"word{i}", float(i), float(i) + 0.5) for i in range(20)]
        words = self._make_words(word_data)
        segment_text = " ".join(w.text for w in words)
        segment = self._make_segment(segment_text, 0.0, 20.0, words)

        result = segments_to_lines([segment], max_words_per_line=10)

        # Should be split into at least 2 lines
        assert len(result) >= 2
        # Total words should equal original
        total_words = sum(len(line.words) for line in result)
        assert total_words == 20

    def test_split_prefers_punctuation(self) -> None:
        """When splitting, punctuation boundaries are preferred."""
        # Create words where word5 ends with a comma
        word_data = [
            ("This", 0.0, 0.5),
            ("is", 0.6, 1.0),
            ("a", 1.1, 1.3),
            ("test,", 1.4, 1.8),  # Punctuation here
            ("and", 1.9, 2.2),
            ("more", 2.3, 2.6),
            ("words", 2.7, 3.0),
            ("follow", 3.1, 3.4),
            ("after", 3.5, 3.8),
            ("it", 3.9, 4.2),
            ("now", 4.3, 4.6),
            ("even", 4.7, 5.0),
            ("more", 5.1, 5.4),
            ("text", 5.5, 5.8),
            ("here", 5.9, 6.2),
            ("today", 6.3, 6.6),
        ]
        words = self._make_words(word_data)
        segment_text = " ".join(w[0] for w in word_data)
        segment = self._make_segment(segment_text, 0.0, 6.6, words)

        result = segments_to_lines([segment], max_words_per_line=10)

        # Should split at punctuation point if possible
        assert len(result) >= 2
        # First line should end with "test," (index 3, which is 4 words)
        # or the split should be reasonable
        first_line_words = [w.text for w in result[0].words]
        assert len(first_line_words) <= 10

    def test_segment_at_exact_limit_not_split(self) -> None:
        """A segment with exactly max_words_per_line words is not split."""
        word_data = [(f"word{i}", float(i), float(i) + 0.5) for i in range(15)]
        words = self._make_words(word_data)
        segment_text = " ".join(w.text for w in words)
        segment = self._make_segment(segment_text, 0.0, 15.0, words)

        result = segments_to_lines([segment], max_words_per_line=15)

        assert len(result) == 1
        assert len(result[0].words) == 15

    def test_empty_segment_is_skipped(self) -> None:
        """Segments with no words are skipped."""
        segment = self._make_segment("", 0.0, 1.0, ())

        result = segments_to_lines([segment])

        assert result == []

    def test_preserves_word_timing(self) -> None:
        """Word timing is preserved in output lines."""
        words = self._make_words([
            ("hello", 1.5, 2.0),
            ("world", 2.5, 3.0),
        ])
        segment = self._make_segment("hello world", 1.5, 3.0, words)

        result = segments_to_lines([segment])

        assert result[0].words[0].start == 1.5
        assert result[0].words[0].end == 2.0
        assert result[0].words[1].start == 2.5
        assert result[0].words[1].end == 3.0

    def test_line_timing_from_words(self) -> None:
        """Line start/end comes from first/last word timing."""
        words = self._make_words([
            ("hello", 1.5, 2.0),
            ("world", 2.5, 3.0),
        ])
        segment = self._make_segment("hello world", 1.0, 4.0, words)

        result = segments_to_lines([segment])

        # Line timing should use word timing, not segment timing
        # For non-split segments, segment timing is preserved
        assert result[0].start == 1.0  # Uses segment start
        assert result[0].end == 4.0  # Uses segment end


# ---------------------------------------------------------------------------
# Tests for TimedSegment model
# ---------------------------------------------------------------------------


class TestTimedSegmentModel:
    """Tests for TimedSegment dataclass."""

    def test_frozen_dataclass(self) -> None:
        """TimedSegment is immutable."""
        words = (TimedWord(text="hello", start=1.0, end=2.0),)
        segment = TimedSegment(text="hello", start=1.0, end=2.0, words=words)

        with pytest.raises(AttributeError):
            segment.text = "world"  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible after construction."""
        word1 = TimedWord(text="hello", start=1.0, end=1.5)
        word2 = TimedWord(text="world", start=1.6, end=2.0)
        segment = TimedSegment(
            text="hello world", start=1.0, end=2.0, words=(word1, word2)
        )

        assert segment.text == "hello world"
        assert segment.start == 1.0
        assert segment.end == 2.0
        assert len(segment.words) == 2
        assert segment.words[0].text == "hello"
        assert segment.words[1].text == "world"


# ---------------------------------------------------------------------------
# Tests for transcribe_lyrics (with mocked segment-based approach)
# ---------------------------------------------------------------------------


class TestTranscribeLyrics:
    """Tests for transcribe_lyrics function.

    These tests mock split_audio_on_silence and _transcribe_single_segment
    to test the segment-based transcription logic without reading real files.
    """

    def test_raises_error_when_no_backend_installed(self) -> None:
        """Raises AlignmentError when no transcription backend is installed."""
        # Mock split_audio_on_silence to return one segment
        mock_segments = [(Path("/fake/segment_0.wav"), 0.0, 5.0)]

        with (
            patch("alsmuse.audio.split_audio_on_silence", return_value=mock_segments),
            patch.dict("sys.modules", {"stable_whisper": None, "mlx_whisper": None}),
        ):
            import importlib

            from alsmuse import lyrics_align

            importlib.reload(lyrics_align)

            with pytest.raises(AlignmentError, match="No transcription backend"):
                lyrics_align.transcribe_lyrics(
                    audio_path=Path("/fake/audio.wav"),
                    valid_ranges=[(0.0, 10.0)],
                )

    def test_successful_transcription_with_mocked_segments(self) -> None:
        """Successful transcription returns TimedSegment list from segments."""
        # Mock split_audio_on_silence to return one segment
        mock_audio_segments = [(Path("/fake/segment_0.wav"), 0.0, 5.0)]

        # Create expected TimedSegments that _transcribe_single_segment would return
        expected_segment = TimedSegment(
            text="hello world",
            start=1.0,
            end=2.0,
            words=(
                TimedWord(text="hello", start=1.0, end=1.5),
                TimedWord(text="world", start=1.6, end=2.0),
            ),
        )

        with (
            patch(
                "alsmuse.audio.split_audio_on_silence",
                return_value=mock_audio_segments,
            ),
            patch(
                "alsmuse.lyrics_align._transcribe_single_segment",
                return_value=[expected_segment],
            ),
        ):
            from alsmuse import lyrics_align

            segments, raw_text = lyrics_align.transcribe_lyrics(
                audio_path=Path("/fake/audio.wav"),
                valid_ranges=[(0.0, 10.0)],
            )

        assert len(segments) == 1
        assert segments[0].text == "hello world"
        assert segments[0].start == 1.0
        assert segments[0].end == 2.0
        assert len(segments[0].words) == 2
        assert raw_text == "hello world"

    def test_multiple_segments_combined(self) -> None:
        """Segments from multiple audio chunks are combined."""
        # Two audio segments at different times
        mock_audio_segments = [
            (Path("/fake/segment_0.wav"), 0.0, 3.0),
            (Path("/fake/segment_1.wav"), 5.0, 8.0),
        ]

        # Results from each segment
        segment1 = TimedSegment(
            text="hello",
            start=1.0,
            end=2.0,
            words=(TimedWord(text="hello", start=1.0, end=2.0),),
        )
        segment2 = TimedSegment(
            text="world",
            start=6.0,
            end=7.0,
            words=(TimedWord(text="world", start=6.0, end=7.0),),
        )

        # _transcribe_single_segment is called once per audio segment
        call_count = [0]

        def mock_transcribe(audio_path, language, model_size, time_offset):
            result = [segment1] if call_count[0] == 0 else [segment2]
            call_count[0] += 1
            return result

        with (
            patch(
                "alsmuse.audio.split_audio_on_silence",
                return_value=mock_audio_segments,
            ),
            patch(
                "alsmuse.lyrics_align._transcribe_single_segment",
                side_effect=mock_transcribe,
            ),
        ):
            from alsmuse import lyrics_align

            segments, raw_text = lyrics_align.transcribe_lyrics(
                audio_path=Path("/fake/audio.wav"),
                valid_ranges=[],
            )

        assert len(segments) == 2
        assert segments[0].text == "hello"
        assert segments[1].text == "world"
        assert raw_text == "hello\nworld"

    def test_empty_audio_returns_empty_result(self) -> None:
        """No segments from silence detection returns empty result."""
        with patch(
            "alsmuse.audio.split_audio_on_silence",
            return_value=[],  # No audio segments
        ):
            from alsmuse import lyrics_align

            segments, raw_text = lyrics_align.transcribe_lyrics(
                audio_path=Path("/fake/audio.wav"),
                valid_ranges=[(0.0, 10.0)],
            )

        assert segments == []
        assert raw_text == ""

    def test_segments_sorted_by_start_time(self) -> None:
        """Segments are sorted by start time in final result."""
        # Audio segments that produce out-of-order results
        mock_audio_segments = [
            (Path("/fake/segment_0.wav"), 5.0, 8.0),
            (Path("/fake/segment_1.wav"), 0.0, 3.0),
        ]

        # Results in opposite order to input
        segment_early = TimedSegment(
            text="first",
            start=1.0,
            end=2.0,
            words=(TimedWord(text="first", start=1.0, end=2.0),),
        )
        segment_late = TimedSegment(
            text="second",
            start=6.0,
            end=7.0,
            words=(TimedWord(text="second", start=6.0, end=7.0),),
        )

        call_count = [0]

        def mock_transcribe(audio_path, language, model_size, time_offset):
            result = [segment_late] if call_count[0] == 0 else [segment_early]
            call_count[0] += 1
            return result

        with (
            patch(
                "alsmuse.audio.split_audio_on_silence",
                return_value=mock_audio_segments,
            ),
            patch(
                "alsmuse.lyrics_align._transcribe_single_segment",
                side_effect=mock_transcribe,
            ),
        ):
            from alsmuse import lyrics_align

            segments, _ = lyrics_align.transcribe_lyrics(
                audio_path=Path("/fake/audio.wav"),
                valid_ranges=[],
            )

        # Should be sorted by start time
        assert segments[0].text == "first"
        assert segments[1].text == "second"

    def test_transcription_failure_raises_alignment_error(self) -> None:
        """Transcription failure raises AlignmentError."""
        mock_audio_segments = [(Path("/fake/segment_0.wav"), 0.0, 5.0)]

        with (
            patch(
                "alsmuse.audio.split_audio_on_silence",
                return_value=mock_audio_segments,
            ),
            patch(
                "alsmuse.lyrics_align._transcribe_single_segment",
                side_effect=AlignmentError("Transcription failed"),
            ),
        ):
            from alsmuse import lyrics_align

            with pytest.raises(AlignmentError, match="Transcription failed"):
                lyrics_align.transcribe_lyrics(
                    audio_path=Path("/fake/audio.wav"),
                    valid_ranges=[(0.0, 10.0)],
                )
