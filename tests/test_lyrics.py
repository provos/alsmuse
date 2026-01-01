"""Tests for lyrics parsing and distribution."""

from pathlib import Path

import pytest

from alsmuse.lyrics import _apply_lyrics, distribute_lyrics, parse_lyrics_file
from alsmuse.models import Phrase


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
