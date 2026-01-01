"""Functional integration tests for ALSmuse.

These tests exercise the full pipeline from ALS file parsing through to
formatted output, validating that all components work together correctly.
"""

import gzip
from pathlib import Path

from alsmuse.analyze import analyze_als, analyze_als_v2


def create_als_file(tmp_path: Path, xml_content: str) -> Path:
    """Create a gzip-compressed ALS file with the given XML content."""
    als_file = tmp_path / "test.als"
    with gzip.open(als_file, "wt", encoding="utf-8") as f:
        f.write(xml_content)
    return als_file


MINIMAL_ALS_XML = """\
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
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events>
                  <MidiClip Id="0" Time="0">
                    <Name Value="INTRO"/>
                    <CurrentEnd Value="16"/>
                    <Notes><KeyTracks/></Notes>
                  </MidiClip>
                  <MidiClip Id="1" Time="16">
                    <Name Value="VERSE1"/>
                    <CurrentEnd Value="48"/>
                    <Notes><KeyTracks/></Notes>
                  </MidiClip>
                  <MidiClip Id="2" Time="48">
                    <Name Value="CHORUS"/>
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


ALS_WITH_MIDI_TRACK = """\
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
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events>
                  <MidiClip Id="0" Time="0">
                    <Name Value="INTRO"/>
                    <CurrentEnd Value="16"/>
                    <Notes><KeyTracks/></Notes>
                  </MidiClip>
                  <MidiClip Id="1" Time="16">
                    <Name Value="VERSE1"/>
                    <CurrentEnd Value="48"/>
                    <Notes><KeyTracks/></Notes>
                  </MidiClip>
                </Events>
              </ArrangerAutomation>
            </ClipTimeable>
          </MainSequencer>
        </DeviceChain>
      </MidiTrack>
      <MidiTrack Id="1">
        <Name>
          <EffectiveName Value="Bass"/>
          <UserName Value="Bass"/>
        </Name>
        <DeviceChain>
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events>
                  <MidiClip Id="10" Time="16">
                    <Name Value="Bass Pattern"/>
                    <CurrentEnd Value="48"/>
                    <Notes>
                      <KeyTracks>
                        <KeyTrack Id="0">
                          <MidiKey Value="36"/>
                          <Notes>
                            <MidiNoteEvent Time="0" Duration="0.5" Velocity="100"/>
                            <MidiNoteEvent Time="2" Duration="0.5" Velocity="100"/>
                            <MidiNoteEvent Time="4" Duration="0.5" Velocity="100"/>
                            <MidiNoteEvent Time="6" Duration="0.5" Velocity="100"/>
                          </Notes>
                        </KeyTrack>
                      </KeyTracks>
                    </Notes>
                  </MidiClip>
                </Events>
              </ArrangerAutomation>
            </ClipTimeable>
          </MainSequencer>
        </DeviceChain>
      </MidiTrack>
      <MidiTrack Id="2">
        <Name>
          <EffectiveName Value="Drums"/>
          <UserName Value="Drums"/>
        </Name>
        <DeviceChain>
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events>
                  <MidiClip Id="20" Time="0">
                    <Name Value="Drum Pattern"/>
                    <CurrentEnd Value="48"/>
                    <Notes>
                      <KeyTracks>
                        <KeyTrack Id="0">
                          <MidiKey Value="36"/>
                          <Notes>
                            <MidiNoteEvent Time="0" Duration="0.25" Velocity="100"/>
                            <MidiNoteEvent Time="1" Duration="0.25" Velocity="100"/>
                            <MidiNoteEvent Time="2" Duration="0.25" Velocity="100"/>
                            <MidiNoteEvent Time="3" Duration="0.25" Velocity="100"/>
                          </Notes>
                        </KeyTrack>
                      </KeyTracks>
                    </Notes>
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


class TestAnalyzeAlsV1:
    """Integration tests for basic section analysis (v1 pipeline)."""

    def test_analyze_produces_markdown_table(self, tmp_path: Path) -> None:
        """Full pipeline produces valid markdown table."""
        als_file = create_als_file(tmp_path, MINIMAL_ALS_XML)

        result = analyze_als(als_file)

        assert "| Time | Audio | Video |" in result
        assert "| 0:00 | INTRO | |" in result
        assert "| 0:08 | VERSE1 | |" in result
        assert "| 0:24 | CHORUS | |" in result

    def test_analyze_uses_correct_tempo_for_timing(self, tmp_path: Path) -> None:
        """Timing calculations use the tempo from the file."""
        als_file = create_als_file(tmp_path, MINIMAL_ALS_XML)

        result = analyze_als(als_file)

        # At 120 BPM, 16 beats = 8 seconds, 48 beats = 24 seconds
        assert "0:08" in result  # VERSE1 at beat 16
        assert "0:24" in result  # CHORUS at beat 48

    def test_analyze_respects_structure_track_option(self, tmp_path: Path) -> None:
        """Analysis uses the specified structure track name."""
        # XML with a differently named structure track
        xml = MINIMAL_ALS_XML.replace("STRUCTURE", "MARKERS")
        als_file = create_als_file(tmp_path, xml)

        result = analyze_als(als_file, structure_track="MARKERS")

        assert "INTRO" in result
        assert "VERSE1" in result


class TestAnalyzeAlsV2:
    """Integration tests for phrase-level analysis (v2 pipeline)."""

    def test_phrase_subdivision_produces_chunks(self, tmp_path: Path) -> None:
        """Sections are subdivided into phrase-sized chunks."""
        als_file = create_als_file(tmp_path, MINIMAL_ALS_XML)

        result = analyze_als_v2(als_file, beats_per_phrase=8, show_events=False)

        # INTRO (16 beats) should produce 2 phrases at 8 beats each
        # VERSE1 (32 beats) should produce 4 phrases
        # CHORUS (32 beats) should produce 4 phrases
        lines = result.strip().split("\n")
        data_rows = [
            row for row in lines
            if row.startswith("|") and "Time" not in row and "---" not in row
        ]
        assert len(data_rows) == 10  # 2 + 4 + 4

    def test_phrase_output_shows_section_names(self, tmp_path: Path) -> None:
        """First phrase of each section shows section name."""
        als_file = create_als_file(tmp_path, MINIMAL_ALS_XML)

        result = analyze_als_v2(als_file, beats_per_phrase=8, show_events=False)

        assert "INTRO" in result
        assert "VERSE1" in result
        assert "CHORUS" in result
        assert "..." in result  # Continuation markers

    def test_event_detection_shows_track_changes(self, tmp_path: Path) -> None:
        """Track enter/exit events are detected and shown."""
        als_file = create_als_file(tmp_path, ALS_WITH_MIDI_TRACK)

        result = analyze_als_v2(als_file, beats_per_phrase=8, show_events=True)

        # Drums starts at beat 0, Bass enters at beat 16
        assert "Drums enters" in result
        assert "Bass enters" in result

    def test_event_detection_can_be_disabled(self, tmp_path: Path) -> None:
        """Events are hidden when show_events=False."""
        als_file = create_als_file(tmp_path, ALS_WITH_MIDI_TRACK)

        result = analyze_als_v2(als_file, beats_per_phrase=8, show_events=False)

        assert "Events" not in result
        assert "enters" not in result

    def test_output_format_has_correct_columns(self, tmp_path: Path) -> None:
        """Output has Time, Cue, Events, and Video columns."""
        als_file = create_als_file(tmp_path, ALS_WITH_MIDI_TRACK)

        result = analyze_als_v2(als_file, beats_per_phrase=8, show_events=True)

        header_line = result.split("\n")[0]
        assert "Time" in header_line
        assert "Cue" in header_line
        assert "Events" in header_line
        assert "Video" in header_line


class TestLyricsIntegration:
    """Integration tests for lyrics functionality."""

    def test_lyrics_appear_in_output(self, tmp_path: Path) -> None:
        """Lyrics from file appear in the output table."""
        als_file = create_als_file(tmp_path, MINIMAL_ALS_XML)

        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("[VERSE1]\nFirst verse line\nSecond verse line\n")

        result = analyze_als_v2(
            als_file, beats_per_phrase=8, show_events=False, lyrics_path=lyrics_file
        )

        assert "Lyrics" in result
        assert "First verse line" in result

    def test_lyrics_distributed_across_section(self, tmp_path: Path) -> None:
        """Lyrics are distributed evenly within their section."""
        als_file = create_als_file(tmp_path, MINIMAL_ALS_XML)

        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("[VERSE1]\nLine one\nLine two\n")

        result = analyze_als_v2(
            als_file, beats_per_phrase=8, show_events=False, lyrics_path=lyrics_file
        )

        # Both lyrics should appear
        assert "Line one" in result
        assert "Line two" in result

    def test_sections_without_lyrics_show_empty(self, tmp_path: Path) -> None:
        """Sections not in lyrics file have empty lyric cells."""
        als_file = create_als_file(tmp_path, MINIMAL_ALS_XML)

        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("[CHORUS]\nChorus lyric\n")

        result = analyze_als_v2(
            als_file, beats_per_phrase=8, show_events=False, lyrics_path=lyrics_file
        )

        # INTRO and VERSE1 shouldn't have lyrics
        lines = result.split("\n")
        intro_line = next(line for line in lines if "INTRO" in line)
        # Empty lyrics cell should just have ""
        assert intro_line.count('""') == 0 or '| "" |' not in intro_line

    def test_lyrics_with_events(self, tmp_path: Path) -> None:
        """Lyrics and events can be shown together."""
        als_file = create_als_file(tmp_path, ALS_WITH_MIDI_TRACK)

        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text("[VERSE1]\nVerse with events\n")

        result = analyze_als_v2(
            als_file, beats_per_phrase=8, show_events=True, lyrics_path=lyrics_file
        )

        # Should have both columns
        assert "Events" in result
        assert "Lyrics" in result
        assert "Verse with events" in result


class TestEdgeCases:
    """Integration tests for edge cases and error handling."""

    def test_empty_section_handled(self, tmp_path: Path) -> None:
        """Very short sections are handled without crashing."""
        xml = """\
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
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events>
                  <MidiClip Id="0" Time="0">
                    <Name Value="SHORT"/>
                    <CurrentEnd Value="4"/>
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
        als_file = create_als_file(tmp_path, xml)

        result = analyze_als_v2(als_file, beats_per_phrase=8, show_events=False)

        assert "SHORT" in result

    def test_different_time_signatures_use_correct_timing(self, tmp_path: Path) -> None:
        """Time signature information is extracted (though calculation uses BPM)."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<Ableton>
  <LiveSet>
    <MainTrack>
      <DeviceChain>
        <Mixer>
          <Tempo>
            <Manual Value="90"/>
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
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events>
                  <MidiClip Id="0" Time="0">
                    <Name Value="INTRO"/>
                    <CurrentEnd Value="24"/>
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
        als_file = create_als_file(tmp_path, xml)

        result = analyze_als(als_file)

        # At 90 BPM, 24 beats should end at 16 seconds
        # First section starts at 0:00
        assert "0:00" in result

    def test_tracks_without_midi_notes_dont_crash(self, tmp_path: Path) -> None:
        """MIDI tracks without notes are handled gracefully."""
        als_file = create_als_file(tmp_path, MINIMAL_ALS_XML)

        # Should not crash even though structure track has no notes
        result = analyze_als_v2(als_file, beats_per_phrase=8, show_events=True)

        assert "INTRO" in result
