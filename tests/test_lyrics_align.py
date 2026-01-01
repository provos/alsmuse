"""Tests for lyrics alignment with stable-ts."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alsmuse.exceptions import AlignmentError
from alsmuse.lyrics_align import (
    _normalize_text,
    _tokenize,
    filter_to_valid_ranges,
    words_to_lines,
)
from alsmuse.models import TimedLine, TimedWord

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

        # Verify model was loaded with MPS device
        mock_stable_whisper.load_model.assert_called_once_with("small", device="mps")

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
