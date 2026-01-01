"""Tests for lyrics parsing and distribution."""

from pathlib import Path

import pytest

from alsmuse.lyrics import (
    _apply_lyrics,
    detect_lyrics_format,
    distribute_lyrics,
    parse_enhanced_lrc,
    parse_lrc_lyrics,
    parse_lyrics_file,
    parse_lyrics_file_auto,
    parse_simple_timed_lyrics,
)
from alsmuse.models import LyricsFormat, Phrase


class TestParseLyricsFile:
    """Tests for parse_lyrics_file function."""

    def test_parse_simple_file(self, tmp_path: Path) -> None:
        """Parse a file with one section."""
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("[VERSE1]\nLine one\nLine two\n")

        result = parse_lyrics_file(lyrics_file)

        assert result == {"VERSE1": ["Line one", "Line two"]}

    def test_parse_multiple_sections(self, tmp_path: Path) -> None:
        """Parse a file with multiple sections."""
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text(
            "[VERSE1]\nFirst verse line\n\n"
            "[CHORUS]\nChorus line one\nChorus line two\n\n"
            "[VERSE2]\nSecond verse line\n"
        )

        result = parse_lyrics_file(lyrics_file)

        assert result == {
            "VERSE1": ["First verse line"],
            "CHORUS": ["Chorus line one", "Chorus line two"],
            "VERSE2": ["Second verse line"],
        }

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        """Parse an empty file."""
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("")

        result = parse_lyrics_file(lyrics_file)

        assert result == {}

    def test_parse_section_name_case_insensitive(self, tmp_path: Path) -> None:
        """Section names are converted to uppercase."""
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("[verse1]\nLine one\n[Chorus]\nLine two\n")

        result = parse_lyrics_file(lyrics_file)

        assert "VERSE1" in result
        assert "CHORUS" in result

    def test_parse_ignores_empty_lines_within_section(self, tmp_path: Path) -> None:
        """Empty lines within sections are ignored."""
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("[VERSE1]\nLine one\n\nLine two\n")

        result = parse_lyrics_file(lyrics_file)

        assert result == {"VERSE1": ["Line one", "Line two"]}

    def test_parse_ignores_lines_before_first_section(self, tmp_path: Path) -> None:
        """Lines before the first section header are ignored."""
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("Title\nArtist\n\n[VERSE1]\nActual lyric\n")

        result = parse_lyrics_file(lyrics_file)

        assert result == {"VERSE1": ["Actual lyric"]}

    def test_parse_strips_whitespace(self, tmp_path: Path) -> None:
        """Whitespace is stripped from lines."""
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("[VERSE1]\n  Line with spaces  \n")

        result = parse_lyrics_file(lyrics_file)

        assert result == {"VERSE1": ["Line with spaces"]}

    def test_parse_file_not_found(self, tmp_path: Path) -> None:
        """Raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_lyrics_file(tmp_path / "nonexistent.txt")


class TestApplyLyrics:
    """Tests for _apply_lyrics helper function."""

    def test_apply_no_lyrics(self) -> None:
        """Empty lyrics returns phrases unchanged."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="V1", is_section_start=True),
            Phrase(start_beats=8, end_beats=16, section_name="...", is_section_start=False),
        ]

        result = _apply_lyrics(phrases, [])

        assert len(result) == 2
        assert result[0].lyric == ""
        assert result[1].lyric == ""

    def test_apply_no_phrases(self) -> None:
        """Empty phrases returns empty list."""
        result = _apply_lyrics([], ["Some lyric"])

        assert result == []

    def test_apply_equal_phrases_and_lyrics(self) -> None:
        """One lyric per phrase when counts match."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="V1", is_section_start=True),
            Phrase(start_beats=8, end_beats=16, section_name="...", is_section_start=False),
        ]
        lyrics = ["Line one", "Line two"]

        result = _apply_lyrics(phrases, lyrics)

        assert result[0].lyric == "Line one"
        assert result[1].lyric == "Line two"

    def test_apply_more_phrases_than_lyrics(self) -> None:
        """Lyrics distributed evenly when more phrases."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="V1", is_section_start=True),
            Phrase(start_beats=8, end_beats=16, section_name="...", is_section_start=False),
            Phrase(start_beats=16, end_beats=24, section_name="...", is_section_start=False),
            Phrase(start_beats=24, end_beats=32, section_name="...", is_section_start=False),
        ]
        lyrics = ["Line one", "Line two"]

        result = _apply_lyrics(phrases, lyrics)

        # 4 phrases, 2 lyrics => step = 2, so lyrics at indices 0 and 2
        assert result[0].lyric == "Line one"
        assert result[1].lyric == ""
        assert result[2].lyric == "Line two"
        assert result[3].lyric == ""

    def test_apply_more_lyrics_than_phrases(self) -> None:
        """Extra lyrics are dropped when more lyrics than phrases."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="V1", is_section_start=True),
            Phrase(start_beats=8, end_beats=16, section_name="...", is_section_start=False),
        ]
        lyrics = ["Line one", "Line two", "Line three", "Line four"]

        result = _apply_lyrics(phrases, lyrics)

        # Step = 2 // 4 = 0, clamped to 1, so all phrases get a lyric
        # But we only have 2 phrases, so only first 2 lyrics used
        assert result[0].lyric == "Line one"
        assert result[1].lyric == "Line two"

    def test_apply_preserves_phrase_attributes(self) -> None:
        """Phrase attributes other than lyric are preserved."""
        from alsmuse.models import TrackEvent

        event = TrackEvent(beat=0, track_name="Bass", event_type="enter", category="bass")
        phrases = [
            Phrase(
                start_beats=10,
                end_beats=18,
                section_name="VERSE",
                is_section_start=True,
                events=(event,),
            ),
        ]

        result = _apply_lyrics(phrases, ["Some lyric"])

        assert result[0].start_beats == 10
        assert result[0].end_beats == 18
        assert result[0].section_name == "VERSE"
        assert result[0].is_section_start is True
        assert result[0].events == (event,)
        assert result[0].lyric == "Some lyric"


class TestDistributeLyrics:
    """Tests for distribute_lyrics function."""

    def test_distribute_empty_phrases(self) -> None:
        """Empty phrases returns empty list."""
        result = distribute_lyrics([], {"VERSE1": ["Line"]})

        assert result == []

    def test_distribute_empty_lyrics(self) -> None:
        """Empty lyrics dict leaves phrases without lyrics."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="V1", is_section_start=True),
        ]

        result = distribute_lyrics(phrases, {})

        assert len(result) == 1
        assert result[0].lyric == ""

    def test_distribute_single_section(self) -> None:
        """Distribute lyrics within a single section."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="VERSE1", is_section_start=True),
            Phrase(start_beats=8, end_beats=16, section_name="...", is_section_start=False),
            Phrase(start_beats=16, end_beats=24, section_name="...", is_section_start=False),
            Phrase(start_beats=24, end_beats=32, section_name="...", is_section_start=False),
        ]
        section_lyrics = {"VERSE1": ["Line one", "Line two"]}

        result = distribute_lyrics(phrases, section_lyrics)

        assert result[0].lyric == "Line one"
        assert result[1].lyric == ""
        assert result[2].lyric == "Line two"
        assert result[3].lyric == ""

    def test_distribute_multiple_sections(self) -> None:
        """Distribute lyrics across multiple sections."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="VERSE1", is_section_start=True),
            Phrase(start_beats=8, end_beats=16, section_name="...", is_section_start=False),
            Phrase(start_beats=16, end_beats=24, section_name="CHORUS", is_section_start=True),
            Phrase(start_beats=24, end_beats=32, section_name="...", is_section_start=False),
        ]
        section_lyrics = {
            "VERSE1": ["Verse line"],
            "CHORUS": ["Chorus line"],
        }

        result = distribute_lyrics(phrases, section_lyrics)

        assert result[0].lyric == "Verse line"
        assert result[1].lyric == ""
        assert result[2].lyric == "Chorus line"
        assert result[3].lyric == ""

    def test_distribute_missing_section_lyrics(self) -> None:
        """Sections without lyrics get empty strings."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="VERSE1", is_section_start=True),
            Phrase(start_beats=8, end_beats=16, section_name="INTRO", is_section_start=True),
        ]
        section_lyrics = {"VERSE1": ["Only verse has lyrics"]}

        result = distribute_lyrics(phrases, section_lyrics)

        assert result[0].lyric == "Only verse has lyrics"
        assert result[1].lyric == ""

    def test_distribute_case_sensitive_section_matching(self) -> None:
        """Section names must match case for lyrics mapping."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="VERSE1", is_section_start=True),
        ]
        # parse_lyrics_file uppercases section names, so this tests matching
        section_lyrics = {"VERSE1": ["Line"]}

        result = distribute_lyrics(phrases, section_lyrics)

        assert result[0].lyric == "Line"

    def test_distribute_three_sections(self) -> None:
        """Distribute lyrics across three sections."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="INTRO", is_section_start=True),
            Phrase(start_beats=8, end_beats=16, section_name="VERSE1", is_section_start=True),
            Phrase(start_beats=16, end_beats=24, section_name="...", is_section_start=False),
            Phrase(start_beats=24, end_beats=32, section_name="CHORUS", is_section_start=True),
        ]
        section_lyrics = {
            "INTRO": [],  # No lyrics for intro
            "VERSE1": ["Verse lyric 1", "Verse lyric 2"],
            "CHORUS": ["Chorus lyric"],
        }

        result = distribute_lyrics(phrases, section_lyrics)

        assert result[0].lyric == ""  # INTRO has no lyrics
        assert result[1].lyric == "Verse lyric 1"
        assert result[2].lyric == "Verse lyric 2"
        assert result[3].lyric == "Chorus lyric"


class TestDetectLyricsFormat:
    """Tests for detect_lyrics_format function."""

    def test_detect_plain_text(self) -> None:
        """Plain text without timestamps is detected as PLAIN."""
        content = "[VERSE1]\nHello world\nGoodbye world\n"
        assert detect_lyrics_format(content) == LyricsFormat.PLAIN

    def test_detect_plain_text_no_sections(self) -> None:
        """Plain text without sections is detected as PLAIN."""
        content = "Hello world\nGoodbye world\n"
        assert detect_lyrics_format(content) == LyricsFormat.PLAIN

    def test_detect_lrc_format(self) -> None:
        """Standard LRC format is detected."""
        content = "[00:12.34]First line\n[00:15.67]Second line\n"
        assert detect_lyrics_format(content) == LyricsFormat.LRC

    def test_detect_lrc_with_metadata(self) -> None:
        """LRC format with metadata tags is detected as LRC."""
        content = "[ar:Artist Name]\n[ti:Song Title]\n[00:12.34]First line\n"
        assert detect_lyrics_format(content) == LyricsFormat.LRC

    def test_detect_lrc_only_metadata_is_plain(self) -> None:
        """File with only metadata tags (no timed lines) is PLAIN."""
        content = "[ar:Artist Name]\n[ti:Song Title]\n"
        assert detect_lyrics_format(content) == LyricsFormat.PLAIN

    def test_detect_lrc_with_milliseconds(self) -> None:
        """LRC format with milliseconds (3 digits) is detected."""
        content = "[00:12.345]First line\n[00:15.678]Second line\n"
        assert detect_lyrics_format(content) == LyricsFormat.LRC

    def test_detect_simple_timed_format(self) -> None:
        """Simple timed format is detected."""
        content = "0:12.34 First line\n0:15.67 Second line\n"
        assert detect_lyrics_format(content) == LyricsFormat.SIMPLE_TIMED

    def test_detect_simple_timed_with_double_digit_minutes(self) -> None:
        """Simple timed format with mm:ss.xx is detected."""
        content = "01:12.34 First line\n12:15.67 Second line\n"
        assert detect_lyrics_format(content) == LyricsFormat.SIMPLE_TIMED

    def test_detect_enhanced_lrc(self) -> None:
        """Enhanced LRC with word timestamps is detected."""
        content = "[00:12.34]<00:12.34>Hello <00:12.80>world\n"
        assert detect_lyrics_format(content) == LyricsFormat.LRC_ENHANCED

    def test_detect_enhanced_lrc_multiple_lines(self) -> None:
        """Enhanced LRC across multiple lines is detected."""
        content = (
            "[00:12.34]<00:12.34>Hello <00:12.80>world\n"
            "[00:15.67]<00:15.67>How <00:15.90>are <00:16.20>you\n"
        )
        assert detect_lyrics_format(content) == LyricsFormat.LRC_ENHANCED

    def test_detect_empty_content(self) -> None:
        """Empty content is detected as PLAIN."""
        assert detect_lyrics_format("") == LyricsFormat.PLAIN
        assert detect_lyrics_format("   \n  \n  ") == LyricsFormat.PLAIN

    def test_detect_mixed_formats_prefers_lrc(self) -> None:
        """When LRC timestamps appear, format is detected as LRC even with other patterns."""
        # This file has both LRC timestamps and some plain text lines
        content = "[ar:Artist]\n[00:12.34]First line\nSome plain text\n[00:15.67]Second line\n"
        assert detect_lyrics_format(content) == LyricsFormat.LRC


class TestParseLrcLyrics:
    """Tests for parse_lrc_lyrics function."""

    def test_parse_basic_lrc(self) -> None:
        """Parse basic LRC with two lines."""
        content = "[00:12.34]First line\n[00:15.67]Second line\n"
        result = parse_lrc_lyrics(content)

        assert len(result) == 2
        assert result[0].text == "First line"
        assert result[0].start == pytest.approx(12.34)
        assert result[1].text == "Second line"
        assert result[1].start == pytest.approx(15.67)

    def test_parse_lrc_with_milliseconds(self) -> None:
        """Parse LRC with millisecond precision."""
        content = "[00:12.345]First line\n[00:15.678]Second line\n"
        result = parse_lrc_lyrics(content)

        assert len(result) == 2
        assert result[0].start == pytest.approx(12.345)
        assert result[1].start == pytest.approx(15.678)

    def test_parse_lrc_ignores_metadata(self) -> None:
        """Metadata tags are ignored."""
        content = "[ar:Artist Name]\n[ti:Song Title]\n[al:Album Name]\n[00:12.34]First line\n"
        result = parse_lrc_lyrics(content)

        assert len(result) == 1
        assert result[0].text == "First line"

    def test_parse_lrc_ignores_empty_timestamps(self) -> None:
        """Empty timestamps (no text) are skipped."""
        content = "[00:00.00]\n[00:12.34]First line\n[00:45.00]\n[00:47.00]Second line\n"
        result = parse_lrc_lyrics(content)

        assert len(result) == 2
        assert result[0].text == "First line"
        assert result[1].text == "Second line"

    def test_parse_lrc_multiple_timestamps_same_line(self) -> None:
        """Multiple timestamps for repeated lyrics create separate entries."""
        content = "[00:12.34][01:30.00]Repeated chorus line\n"
        result = parse_lrc_lyrics(content)

        assert len(result) == 2
        assert result[0].text == "Repeated chorus line"
        assert result[0].start == pytest.approx(12.34)
        assert result[1].text == "Repeated chorus line"
        assert result[1].start == pytest.approx(90.0)

    def test_parse_lrc_results_sorted_by_time(self) -> None:
        """Results are sorted by timestamp."""
        content = "[01:30.00]Later line\n[00:12.34]Earlier line\n"
        result = parse_lrc_lyrics(content)

        assert len(result) == 2
        assert result[0].text == "Earlier line"
        assert result[1].text == "Later line"

    def test_parse_lrc_with_minutes(self) -> None:
        """Parse LRC with multi-minute timestamps."""
        content = "[03:45.67]Three minutes in\n"
        result = parse_lrc_lyrics(content)

        assert len(result) == 1
        assert result[0].start == pytest.approx(3 * 60 + 45.67)

    def test_parse_lrc_empty_words_tuple(self) -> None:
        """Standard LRC has no word-level timing."""
        content = "[00:12.34]First line\n"
        result = parse_lrc_lyrics(content)

        assert result[0].words == ()

    def test_parse_lrc_empty_content(self) -> None:
        """Empty content returns empty list."""
        assert parse_lrc_lyrics("") == []
        assert parse_lrc_lyrics("   \n  \n  ") == []


class TestParseSimpleTimedLyrics:
    """Tests for parse_simple_timed_lyrics function."""

    def test_parse_basic_simple_timed(self) -> None:
        """Parse basic simple timed format."""
        content = "0:12.34 First line\n0:15.67 Second line\n"
        result = parse_simple_timed_lyrics(content)

        assert len(result) == 2
        assert result[0].text == "First line"
        assert result[0].start == pytest.approx(12.34)
        assert result[1].text == "Second line"
        assert result[1].start == pytest.approx(15.67)

    def test_parse_simple_timed_double_digit_minutes(self) -> None:
        """Parse with double-digit minutes."""
        content = "01:12.34 First line\n12:15.67 Second line\n"
        result = parse_simple_timed_lyrics(content)

        assert len(result) == 2
        assert result[0].start == pytest.approx(72.34)
        assert result[1].start == pytest.approx(12 * 60 + 15.67)

    def test_parse_simple_timed_with_milliseconds(self) -> None:
        """Parse with millisecond precision."""
        content = "0:12.345 First line\n0:15.678 Second line\n"
        result = parse_simple_timed_lyrics(content)

        assert len(result) == 2
        assert result[0].start == pytest.approx(12.345)
        assert result[1].start == pytest.approx(15.678)

    def test_parse_simple_timed_results_sorted(self) -> None:
        """Results are sorted by timestamp."""
        content = "1:30.00 Later line\n0:12.34 Earlier line\n"
        result = parse_simple_timed_lyrics(content)

        assert len(result) == 2
        assert result[0].text == "Earlier line"
        assert result[1].text == "Later line"

    def test_parse_simple_timed_empty_words_tuple(self) -> None:
        """Simple timed has no word-level timing."""
        content = "0:12.34 First line\n"
        result = parse_simple_timed_lyrics(content)

        assert result[0].words == ()

    def test_parse_simple_timed_skips_invalid_lines(self) -> None:
        """Lines without valid timestamps are skipped."""
        content = "0:12.34 Valid line\nNo timestamp here\n0:15.67 Another valid\n"
        result = parse_simple_timed_lyrics(content)

        assert len(result) == 2
        assert result[0].text == "Valid line"
        assert result[1].text == "Another valid"

    def test_parse_simple_timed_empty_content(self) -> None:
        """Empty content returns empty list."""
        assert parse_simple_timed_lyrics("") == []


class TestParseEnhancedLrc:
    """Tests for parse_enhanced_lrc function."""

    def test_parse_basic_enhanced_lrc(self) -> None:
        """Parse basic enhanced LRC with word timestamps."""
        content = "[00:12.34]<00:12.34>Hello <00:12.80>world\n"
        result = parse_enhanced_lrc(content)

        assert len(result) == 1
        assert result[0].text == "Hello world"
        assert result[0].start == pytest.approx(12.34)
        assert len(result[0].words) == 2
        assert result[0].words[0].text == "Hello"
        assert result[0].words[0].start == pytest.approx(12.34)
        assert result[0].words[1].text == "world"
        assert result[0].words[1].start == pytest.approx(12.80)

    def test_parse_enhanced_lrc_multiple_lines(self) -> None:
        """Parse multiple lines of enhanced LRC."""
        content = (
            "[00:12.34]<00:12.34>Hello <00:12.80>world\n"
            "[00:15.67]<00:15.67>How <00:15.90>are <00:16.20>you\n"
        )
        result = parse_enhanced_lrc(content)

        assert len(result) == 2
        assert result[0].text == "Hello world"
        assert result[1].text == "How are you"
        assert len(result[1].words) == 3

    def test_parse_enhanced_lrc_word_end_times(self) -> None:
        """Word end times are set to next word's start."""
        content = "[00:12.00]<00:12.00>First <00:12.50>Second <00:13.00>Third\n"
        result = parse_enhanced_lrc(content)

        assert len(result[0].words) == 3
        # First word ends when second starts
        assert result[0].words[0].end == pytest.approx(12.50)
        # Second word ends when third starts
        assert result[0].words[1].end == pytest.approx(13.00)
        # Last word gets estimated end time
        assert result[0].words[2].end == pytest.approx(13.50)  # start + 0.5s

    def test_parse_enhanced_lrc_ignores_metadata(self) -> None:
        """Metadata tags are ignored."""
        content = "[ar:Artist]\n[00:12.34]<00:12.34>Hello <00:12.80>world\n"
        result = parse_enhanced_lrc(content)

        assert len(result) == 1
        assert result[0].text == "Hello world"

    def test_parse_enhanced_lrc_with_milliseconds(self) -> None:
        """Parse with millisecond precision."""
        content = "[00:12.345]<00:12.345>Hello <00:12.800>world\n"
        result = parse_enhanced_lrc(content)

        assert len(result) == 1
        assert result[0].start == pytest.approx(12.345)
        assert result[0].words[0].start == pytest.approx(12.345)
        assert result[0].words[1].start == pytest.approx(12.800)

    def test_parse_enhanced_lrc_line_timing_from_words(self) -> None:
        """Line start/end come from first/last word."""
        content = "[00:12.00]<00:12.34>Hello <00:12.80>world\n"
        result = parse_enhanced_lrc(content)

        # Line start is first word's start
        assert result[0].start == pytest.approx(12.34)
        # Line end is last word's end
        assert result[0].end == pytest.approx(13.30)  # 12.80 + 0.5

    def test_parse_enhanced_lrc_empty_content(self) -> None:
        """Empty content returns empty list."""
        assert parse_enhanced_lrc("") == []

    def test_parse_enhanced_lrc_sorted_by_time(self) -> None:
        """Results are sorted by timestamp."""
        content = (
            "[00:30.00]<00:30.00>Later <00:30.50>line\n[00:12.00]<00:12.00>Earlier <00:12.50>line\n"
        )
        result = parse_enhanced_lrc(content)

        assert len(result) == 2
        assert result[0].text == "Earlier line"
        assert result[1].text == "Later line"


class TestParseLyricsFileAuto:
    """Tests for parse_lyrics_file_auto function."""

    def test_auto_parse_plain_text(self, tmp_path: Path) -> None:
        """Plain text is parsed as sections."""
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("[VERSE1]\nHello world\nGoodbye world\n")

        timed, sections = parse_lyrics_file_auto(lyrics_file)

        assert timed is None
        assert sections is not None
        assert sections == {"VERSE1": ["Hello world", "Goodbye world"]}

    def test_auto_parse_lrc(self, tmp_path: Path) -> None:
        """LRC format is parsed as timed lines."""
        lyrics_file = tmp_path / "lyrics.lrc"
        lyrics_file.write_text("[00:12.34]First line\n[00:15.67]Second line\n")

        timed, sections = parse_lyrics_file_auto(lyrics_file)

        assert timed is not None
        assert sections is None
        assert len(timed) == 2
        assert timed[0].text == "First line"

    def test_auto_parse_simple_timed(self, tmp_path: Path) -> None:
        """Simple timed format is parsed as timed lines."""
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("0:12.34 First line\n0:15.67 Second line\n")

        timed, sections = parse_lyrics_file_auto(lyrics_file)

        assert timed is not None
        assert sections is None
        assert len(timed) == 2

    def test_auto_parse_enhanced_lrc(self, tmp_path: Path) -> None:
        """Enhanced LRC is parsed with word timing."""
        lyrics_file = tmp_path / "lyrics.lrc"
        lyrics_file.write_text("[00:12.34]<00:12.34>Hello <00:12.80>world\n")

        timed, sections = parse_lyrics_file_auto(lyrics_file)

        assert timed is not None
        assert sections is None
        assert len(timed) == 1
        assert len(timed[0].words) == 2

    def test_auto_parse_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns None for timed and empty dict for sections."""
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("")

        timed, sections = parse_lyrics_file_auto(lyrics_file)

        assert timed is None
        assert sections == {}

    def test_auto_parse_file_not_found(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_lyrics_file_auto(tmp_path / "nonexistent.txt")

    def test_auto_parse_lrc_with_metadata(self, tmp_path: Path) -> None:
        """LRC with metadata is parsed correctly."""
        lyrics_file = tmp_path / "lyrics.lrc"
        lyrics_file.write_text(
            "[ar:Artist Name]\n[ti:Song Title]\n[00:12.34]First line\n[00:15.67]Second line\n"
        )

        timed, sections = parse_lyrics_file_auto(lyrics_file)

        assert timed is not None
        assert len(timed) == 2
        # Metadata is ignored
        assert timed[0].text == "First line"


class TestTimestampParsing:
    """Tests for timestamp edge cases."""

    def test_lrc_zero_timestamp(self) -> None:
        """Zero timestamp parses correctly."""
        content = "[00:00.00]Start of song\n"
        result = parse_lrc_lyrics(content)

        assert len(result) == 1
        assert result[0].start == pytest.approx(0.0)

    def test_lrc_large_timestamp(self) -> None:
        """Large timestamps (> 1 hour) parse correctly."""
        content = "[99:59.99]End of long song\n"
        result = parse_lrc_lyrics(content)

        assert len(result) == 1
        expected = 99 * 60 + 59.99
        assert result[0].start == pytest.approx(expected)

    def test_simple_timed_single_digit_minute(self) -> None:
        """Single digit minutes parse correctly."""
        content = "0:00.00 Start\n1:00.00 One minute\n9:59.99 Almost ten\n"
        result = parse_simple_timed_lyrics(content)

        assert len(result) == 3
        assert result[0].start == pytest.approx(0.0)
        assert result[1].start == pytest.approx(60.0)
        assert result[2].start == pytest.approx(9 * 60 + 59.99)

    def test_centiseconds_vs_milliseconds(self) -> None:
        """Both 2-digit and 3-digit fractional seconds work."""
        content_2digit = "[00:12.34]Two digits\n"
        content_3digit = "[00:12.340]Three digits\n"

        result_2 = parse_lrc_lyrics(content_2digit)
        result_3 = parse_lrc_lyrics(content_3digit)

        # 0.34 seconds = 340 milliseconds
        assert result_2[0].start == pytest.approx(12.34)
        assert result_3[0].start == pytest.approx(12.340)
