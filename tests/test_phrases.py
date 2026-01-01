"""Tests for phrase subdivision functionality.

These are high-level functional tests that test the subdivide_sections
function's observable behavior, not implementation details.
"""

import pytest

from alsmuse.models import Phrase, Section
from alsmuse.phrases import subdivide_sections


class TestSubdivideSections:
    """Tests for the subdivide_sections function."""

    def test_single_section_exact_multiple(self) -> None:
        """A section that is an exact multiple of phrase length divides evenly."""
        sections = [Section(name="INTRO", start_beats=0, end_beats=16)]

        phrases = subdivide_sections(sections, beats_per_phrase=8)

        assert len(phrases) == 2
        assert phrases[0].start_beats == 0
        assert phrases[0].end_beats == 8
        assert phrases[1].start_beats == 8
        assert phrases[1].end_beats == 16

    def test_single_section_partial_phrase(self) -> None:
        """A section that doesn't divide evenly creates a shorter final phrase."""
        sections = [Section(name="INTRO", start_beats=0, end_beats=12)]

        phrases = subdivide_sections(sections, beats_per_phrase=8)

        assert len(phrases) == 2
        assert phrases[0].end_beats == 8
        assert phrases[1].start_beats == 8
        assert phrases[1].end_beats == 12  # Partial phrase

    def test_section_name_on_first_phrase_only(self) -> None:
        """Only the first phrase of a section carries the section name."""
        sections = [Section(name="VERSE1", start_beats=0, end_beats=24)]

        phrases = subdivide_sections(sections, beats_per_phrase=8)

        assert phrases[0].section_name == "VERSE1"
        assert phrases[0].is_section_start is True
        assert phrases[1].section_name == "..."
        assert phrases[1].is_section_start is False
        assert phrases[2].section_name == "..."
        assert phrases[2].is_section_start is False

    def test_multiple_sections(self) -> None:
        """Multiple sections are subdivided independently."""
        sections = [
            Section(name="INTRO", start_beats=0, end_beats=16),
            Section(name="VERSE1", start_beats=16, end_beats=48),
        ]

        phrases = subdivide_sections(sections, beats_per_phrase=8)

        assert len(phrases) == 6  # 2 + 4

        # First section phrases
        assert phrases[0].section_name == "INTRO"
        assert phrases[0].is_section_start is True
        assert phrases[1].section_name == "..."

        # Second section phrases
        assert phrases[2].section_name == "VERSE1"
        assert phrases[2].is_section_start is True
        assert phrases[2].start_beats == 16
        assert phrases[3].section_name == "..."
        assert phrases[4].section_name == "..."
        assert phrases[5].section_name == "..."

    def test_empty_sections_list(self) -> None:
        """Empty input produces empty output."""
        phrases = subdivide_sections([], beats_per_phrase=8)
        assert phrases == []

    def test_section_smaller_than_phrase(self) -> None:
        """A section shorter than phrase length becomes a single phrase."""
        sections = [Section(name="BREAK", start_beats=0, end_beats=4)]

        phrases = subdivide_sections(sections, beats_per_phrase=8)

        assert len(phrases) == 1
        assert phrases[0].section_name == "BREAK"
        assert phrases[0].start_beats == 0
        assert phrases[0].end_beats == 4
        assert phrases[0].is_section_start is True

    def test_phrases_have_empty_events(self) -> None:
        """Newly created phrases have empty events tuple."""
        sections = [Section(name="INTRO", start_beats=0, end_beats=8)]

        phrases = subdivide_sections(sections, beats_per_phrase=8)

        assert phrases[0].events == ()

    def test_phrases_have_empty_lyric(self) -> None:
        """Newly created phrases have empty lyric string."""
        sections = [Section(name="INTRO", start_beats=0, end_beats=8)]

        phrases = subdivide_sections(sections, beats_per_phrase=8)

        assert phrases[0].lyric == ""

    def test_non_zero_start_section(self) -> None:
        """Sections don't have to start at beat 0."""
        sections = [Section(name="VERSE1", start_beats=28, end_beats=60)]

        phrases = subdivide_sections(sections, beats_per_phrase=8)

        assert len(phrases) == 4
        assert phrases[0].start_beats == 28
        assert phrases[0].end_beats == 36
        assert phrases[1].start_beats == 36
        assert phrases[1].end_beats == 44
        assert phrases[2].start_beats == 44
        assert phrases[2].end_beats == 52
        assert phrases[3].start_beats == 52
        assert phrases[3].end_beats == 60

    def test_custom_beats_per_phrase(self) -> None:
        """Different beat lengths create different subdivisions."""
        sections = [Section(name="INTRO", start_beats=0, end_beats=16)]

        phrases_4beat = subdivide_sections(sections, beats_per_phrase=4)
        phrases_16beat = subdivide_sections(sections, beats_per_phrase=16)

        assert len(phrases_4beat) == 4
        assert len(phrases_16beat) == 1

    def test_zero_beats_per_phrase_raises(self) -> None:
        """Zero beats per phrase is invalid."""
        sections = [Section(name="INTRO", start_beats=0, end_beats=16)]

        with pytest.raises(ValueError, match="positive"):
            subdivide_sections(sections, beats_per_phrase=0)

    def test_negative_beats_per_phrase_raises(self) -> None:
        """Negative beats per phrase is invalid."""
        sections = [Section(name="INTRO", start_beats=0, end_beats=16)]

        with pytest.raises(ValueError, match="positive"):
            subdivide_sections(sections, beats_per_phrase=-4)

    def test_phrase_time_conversion(self) -> None:
        """Phrase time methods convert beats to seconds correctly."""
        sections = [Section(name="INTRO", start_beats=0, end_beats=8)]

        phrases = subdivide_sections(sections, beats_per_phrase=8)
        phrase = phrases[0]

        # At 120 BPM: 1 beat = 0.5 seconds
        # 8 beats = 4 seconds
        bpm = 120.0
        assert phrase.start_time(bpm) == 0.0
        assert phrase.duration_seconds(bpm) == 4.0

    def test_phrase_time_at_144_bpm(self) -> None:
        """At 144 BPM, 8 beats should be ~3.3 seconds."""
        sections = [Section(name="INTRO", start_beats=0, end_beats=8)]

        phrases = subdivide_sections(sections, beats_per_phrase=8)
        phrase = phrases[0]

        # At 144 BPM: 8 beats = 8 * 60 / 144 = 3.333... seconds
        bpm = 144.0
        expected_duration = 8 * 60 / 144  # ~3.333
        assert abs(phrase.duration_seconds(bpm) - expected_duration) < 0.001

    def test_continuous_coverage(self) -> None:
        """Phrases should cover the entire time range without gaps."""
        sections = [
            Section(name="INTRO", start_beats=0, end_beats=16),
            Section(name="VERSE1", start_beats=16, end_beats=48),
            Section(name="CHORUS", start_beats=48, end_beats=80),
        ]

        phrases = subdivide_sections(sections, beats_per_phrase=8)

        # Check no gaps between phrases
        for i in range(len(phrases) - 1):
            assert phrases[i].end_beats == phrases[i + 1].start_beats

        # Check full coverage
        assert phrases[0].start_beats == 0
        assert phrases[-1].end_beats == 80

    def test_returns_phrase_instances(self) -> None:
        """Function returns proper Phrase dataclass instances."""
        sections = [Section(name="INTRO", start_beats=0, end_beats=8)]

        phrases = subdivide_sections(sections, beats_per_phrase=8)

        assert all(isinstance(p, Phrase) for p in phrases)
