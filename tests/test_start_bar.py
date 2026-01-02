"""Tests for start bar detection and time offset functionality."""

import gzip

import pytest

from alsmuse.analyze import detect_suggested_start_bar
from alsmuse.config import MuseConfig
from alsmuse.formatter import format_av_table, format_phrase_table
from alsmuse.models import Clip, LiveSet, Phrase, Section, Tempo, TimeContext, Track
from alsmuse.parser import parse_als_file


class TestDetectSuggestedStartBar:
    """Tests for detect_suggested_start_bar function."""

    def test_clips_start_at_zero(self):
        """Returns 0 when clips start at beat 0."""
        live_set = LiveSet(
            tempo=Tempo(bpm=120.0, time_signature=(4, 4)),
            tracks=(
                Track(
                    name="Track1",
                    track_type="midi",
                    clips=(Clip(name="Clip1", start_beats=0.0, end_beats=16.0),),
                ),
            ),
        )
        assert detect_suggested_start_bar(live_set) == 0

    def test_clips_start_at_bar_8(self):
        """Returns 8 when first clip starts at bar 8 (beat 32 in 4/4)."""
        live_set = LiveSet(
            tempo=Tempo(bpm=120.0, time_signature=(4, 4)),
            tracks=(
                Track(
                    name="Track1",
                    track_type="midi",
                    clips=(Clip(name="Clip1", start_beats=32.0, end_beats=48.0),),
                ),
            ),
        )
        assert detect_suggested_start_bar(live_set) == 8

    def test_multiple_tracks_finds_earliest(self):
        """Finds the earliest clip across all tracks."""
        live_set = LiveSet(
            tempo=Tempo(bpm=120.0, time_signature=(4, 4)),
            tracks=(
                Track(
                    name="Track1",
                    track_type="midi",
                    clips=(Clip(name="Clip1", start_beats=64.0, end_beats=80.0),),
                ),
                Track(
                    name="Track2",
                    track_type="audio",
                    clips=(Clip(name="Clip2", start_beats=32.0, end_beats=48.0),),
                ),
            ),
        )
        assert detect_suggested_start_bar(live_set) == 8

    def test_empty_project_returns_zero(self):
        """Returns 0 for project with no clips."""
        live_set = LiveSet(
            tempo=Tempo(bpm=120.0, time_signature=(4, 4)),
            tracks=(
                Track(
                    name="Track1",
                    track_type="midi",
                    clips=(),
                ),
            ),
        )
        assert detect_suggested_start_bar(live_set) == 0

    def test_no_tracks_returns_zero(self):
        """Returns 0 for project with no tracks."""
        live_set = LiveSet(
            tempo=Tempo(bpm=120.0, time_signature=(4, 4)),
            tracks=(),
        )
        assert detect_suggested_start_bar(live_set) == 0

    def test_3_4_time_signature(self):
        """Correctly handles 3/4 time signature."""
        live_set = LiveSet(
            tempo=Tempo(bpm=120.0, time_signature=(3, 4)),
            tracks=(
                Track(
                    name="Track1",
                    track_type="midi",
                    clips=(
                        # 24 beats = 8 bars in 3/4
                        Clip(name="Clip1", start_beats=24.0, end_beats=36.0),
                    ),
                ),
            ),
        )
        assert detect_suggested_start_bar(live_set) == 8

    def test_partial_bar_floor_division(self):
        """Floors to bar number when clip doesn't start on bar boundary."""
        live_set = LiveSet(
            tempo=Tempo(bpm=120.0, time_signature=(4, 4)),
            tracks=(
                Track(
                    name="Track1",
                    track_type="midi",
                    clips=(
                        # 34 beats = bar 8 + 2 beats, should return bar 8
                        Clip(name="Clip1", start_beats=34.0, end_beats=50.0),
                    ),
                ),
            ),
        )
        assert detect_suggested_start_bar(live_set) == 8


class TestFormatPhraseTableWithOffset:
    """Tests for format_phrase_table with TimeContext offset."""

    def test_no_offset_shows_absolute_time(self):
        """Without offset, times start at 0:00."""
        phrases = [
            Phrase(
                start_beats=0.0,
                end_beats=8.0,
                section_name="INTRO",
                is_section_start=True,
            ),
            Phrase(
                start_beats=8.0,
                end_beats=16.0,
                section_name="...",
                is_section_start=False,
            ),
        ]
        time_ctx = TimeContext(bpm=120.0)
        result = format_phrase_table(phrases, time_ctx, show_events=False)
        assert "| 0:00.0 |" in result
        assert "| 0:04.0 |" in result

    def test_offset_adjusts_times(self):
        """With offset, times are relative to the offset."""
        phrases = [
            Phrase(
                start_beats=32.0,  # Bar 8 at 120 BPM = 16 seconds
                end_beats=40.0,
                section_name="INTRO",
                is_section_start=True,
            ),
            Phrase(
                start_beats=40.0,  # Bar 10 = 20 seconds
                end_beats=48.0,
                section_name="...",
                is_section_start=False,
            ),
        ]
        # Offset of 32 beats (bar 8) should make first phrase start at 0:00
        time_ctx = TimeContext(bpm=120.0, start_offset_beats=32.0)
        result = format_phrase_table(phrases, time_ctx, show_events=False)
        assert "| 0:00.0 |" in result
        assert "| 0:04.0 |" in result

    def test_offset_can_produce_negative_times(self):
        """Offset larger than phrase start produces negative times."""
        phrases = [
            Phrase(
                start_beats=0.0,
                end_beats=8.0,
                section_name="INTRO",
                is_section_start=True,
            ),
        ]
        # Offset of 32 beats when phrase starts at 0 should show -16 seconds
        time_ctx = TimeContext(bpm=120.0, start_offset_beats=32.0)
        result = format_phrase_table(phrases, time_ctx, show_events=False)
        # Should contain negative time
        assert "-" in result.split("\n")[2]  # First data row


class TestFormatAvTableWithOffset:
    """Tests for format_av_table with TimeContext offset."""

    def test_no_offset_shows_absolute_time(self):
        """Without offset, times start at 0:00."""
        sections = [
            Section(name="INTRO", start_beats=0.0, end_beats=16.0),
            Section(name="VERSE", start_beats=16.0, end_beats=32.0),
        ]
        time_ctx = TimeContext(bpm=120.0)
        result = format_av_table(sections, time_ctx)
        assert "| 0:00.0 | INTRO |" in result
        assert "| 0:08.0 | VERSE |" in result

    def test_offset_adjusts_times(self):
        """With offset, times are relative to the offset."""
        sections = [
            Section(name="INTRO", start_beats=32.0, end_beats=48.0),
            Section(name="VERSE", start_beats=48.0, end_beats=64.0),
        ]
        time_ctx = TimeContext(bpm=120.0, start_offset_beats=32.0)
        result = format_av_table(sections, time_ctx)
        assert "| 0:00.0 | INTRO |" in result
        assert "| 0:08.0 | VERSE |" in result


class TestIntegrationWithRealAlsFile:
    """Integration tests with realistic ALS file scenarios."""

    @pytest.fixture
    def als_file_starting_at_bar_8(self, tmp_path):
        """Create an ALS file where content starts at bar 8."""
        xml_content = """\
<?xml version="1.0" encoding="UTF-8"?>
<Ableton>
  <LiveSet>
    <MainTrack>
      <DeviceChain>
        <Mixer>
          <Tempo>
            <Manual Value="120"/>
          </Tempo>
        </Mixer>
      </DeviceChain>
    </MainTrack>
    <Tracks>
      <MidiTrack Id="0">
        <Name>
          <EffectiveName Value="STRUCTURE"/>
          <UserName Value=""/>
        </Name>
        <DeviceChain>
          <Mixer><Speaker><Manual Value="true"/></Speaker></Mixer>
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events>
                  <MidiClip Id="0" Time="32">
                    <Name Value="INTRO"/>
                    <CurrentEnd Value="48"/>
                    <Notes><KeyTracks/></Notes>
                  </MidiClip>
                  <MidiClip Id="1" Time="48">
                    <Name Value="VERSE"/>
                    <CurrentEnd Value="80"/>
                    <Notes><KeyTracks/></Notes>
                  </MidiClip>
                </Events>
              </ArrangerAutomation>
            </ClipTimeable>
          </MainSequencer>
        </DeviceChain>
      </MidiTrack>
    </Tracks>
  </LiveSet>
</Ableton>
"""
        als_file = tmp_path / "test.als"
        with gzip.open(als_file, "wt", encoding="utf-8") as f:
            f.write(xml_content)
        return als_file

    def test_detect_start_bar_from_als(self, als_file_starting_at_bar_8):
        """Detects bar 8 as the start bar from a real ALS file."""
        live_set = parse_als_file(als_file_starting_at_bar_8)
        suggested = detect_suggested_start_bar(live_set)
        assert suggested == 8


class TestMuseConfigStartBar:
    """Tests for start_bar in MuseConfig."""

    def test_default_start_bar_is_none(self):
        """Default start_bar is None."""
        config = MuseConfig()
        assert config.start_bar is None

    def test_start_bar_can_be_set(self):
        """start_bar can be set to a value."""
        config = MuseConfig(start_bar=8)
        assert config.start_bar == 8

    def test_to_dict_excludes_none_start_bar(self):
        """to_dict excludes start_bar when None."""
        config = MuseConfig()
        result = config.to_dict()
        assert "start_bar" not in result

    def test_to_dict_includes_start_bar_when_set(self):
        """to_dict includes start_bar when set."""
        config = MuseConfig(start_bar=8)
        result = config.to_dict()
        assert result["start_bar"] == 8

    def test_from_dict_loads_start_bar(self):
        """from_dict loads start_bar from data."""
        data = {
            "version": 1,
            "vocal_tracks": [],
            "category_overrides": {},
            "start_bar": 16,
        }
        config = MuseConfig.from_dict(data)
        assert config.start_bar == 16

    def test_from_dict_missing_start_bar_is_none(self):
        """from_dict returns None for missing start_bar."""
        data = {
            "version": 1,
            "vocal_tracks": [],
            "category_overrides": {},
        }
        config = MuseConfig.from_dict(data)
        assert config.start_bar is None

    def test_roundtrip_with_start_bar(self):
        """Config with start_bar survives to_dict/from_dict roundtrip."""
        original = MuseConfig(
            vocal_tracks=["Vocals"],
            category_overrides={"Track1": "drums"},
            start_bar=8,
        )
        data = original.to_dict()
        restored = MuseConfig.from_dict(data)
        assert restored.start_bar == 8
        assert restored.vocal_tracks == ["Vocals"]
        assert restored.category_overrides == {"Track1": "drums"}
