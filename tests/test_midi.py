"""Comprehensive tests for MIDI note extraction and loop handling.

These tests validate the refactored midi.py module, including:
- Loop info extraction from XML
- Raw note extraction with StartRelative adjustment
- Loop expansion for repeated patterns
- Full extract_midi_notes orchestration
"""

import xml.etree.ElementTree as ET

import pytest

from alsmuse.midi import (
    _expand_looped_notes,
    _extract_loop_info,
    _extract_raw_notes,
    _LoopInfo,
    extract_midi_notes,
)
from alsmuse.models import MidiNote

# =============================================================================
# Helper Functions for Test Data Creation
# =============================================================================


def create_midi_clip_xml(
    time: float = 0.0,
    current_end: float = 8.0,
    loop_on: bool = False,
    loop_start: float = 0.0,
    loop_end: float = 4.0,
    start_relative: float = 0.0,
    notes: list[tuple[float, float, int, int]] | None = None,
    use_keytracks: bool = True,
) -> ET.Element:
    """Create a MidiClip XML element for testing.

    Args:
        time: Clip start time on timeline (Time attribute)
        current_end: Clip end time on timeline
        loop_on: Whether looping is enabled
        loop_start: Loop region start in clip-relative beats
        loop_end: Loop region end in clip-relative beats
        start_relative: Offset into clip where playback starts
        notes: List of (time, duration, velocity, pitch) tuples
        use_keytracks: If True, use KeyTracks structure; else use flat MidiNoteEvent

    Returns:
        MidiClip Element ready for testing
    """
    xml_parts = [f'<MidiClip Time="{time}">']
    xml_parts.append(f'  <CurrentEnd Value="{current_end}"/>')

    # Loop element
    xml_parts.append("  <Loop>")
    xml_parts.append(f'    <LoopOn Value="{str(loop_on).lower()}"/>')
    xml_parts.append(f'    <LoopStart Value="{loop_start}"/>')
    xml_parts.append(f'    <LoopEnd Value="{loop_end}"/>')
    xml_parts.append(f'    <StartRelative Value="{start_relative}"/>')
    xml_parts.append("  </Loop>")

    # Notes
    xml_parts.append("  <Notes>")
    if notes:
        if use_keytracks:
            # Group notes by pitch for KeyTrack structure
            notes_by_pitch: dict[int, list[tuple[float, float, int]]] = {}
            for note_time, duration, velocity, pitch in notes:
                if pitch not in notes_by_pitch:
                    notes_by_pitch[pitch] = []
                notes_by_pitch[pitch].append((note_time, duration, velocity))

            xml_parts.append("    <KeyTracks>")
            for pitch, pitch_notes in notes_by_pitch.items():
                xml_parts.append(
                    f'      <KeyTrack Id="{pitch}">',
                )
                xml_parts.append(f'        <MidiKey Value="{pitch}"/>')
                xml_parts.append("        <Notes>")
                for note_time, duration, velocity in pitch_notes:
                    xml_parts.append(
                        f'          <MidiNoteEvent Time="{note_time}" '
                        f'Duration="{duration}" Velocity="{velocity}"/>'
                    )
                xml_parts.append("        </Notes>")
                xml_parts.append("      </KeyTrack>")
            xml_parts.append("    </KeyTracks>")
        else:
            # Flat MidiNoteEvent structure (fallback path)
            for note_time, duration, velocity, _pitch in notes:
                xml_parts.append(
                    f'    <MidiNoteEvent Time="{note_time}" '
                    f'Duration="{duration}" Velocity="{velocity}"/>'
                )
    else:
        xml_parts.append("    <KeyTracks/>")
    xml_parts.append("  </Notes>")

    xml_parts.append("</MidiClip>")

    return ET.fromstring("\n".join(xml_parts))


# =============================================================================
# Tests for _LoopInfo Dataclass
# =============================================================================


class TestLoopInfoDataclass:
    """Tests for the _LoopInfo frozen dataclass."""

    def test_loop_info_creation(self) -> None:
        """_LoopInfo stores all expected attributes."""
        info = _LoopInfo(
            is_on=True,
            start=0.0,
            end=4.0,
            start_relative=2.0,
            length=4.0,
        )

        assert info.is_on is True
        assert info.start == 0.0
        assert info.end == 4.0
        assert info.start_relative == 2.0
        assert info.length == 4.0

    def test_loop_info_is_frozen(self) -> None:
        """_LoopInfo is immutable."""
        info = _LoopInfo(is_on=False, start=0.0, end=4.0, start_relative=0.0, length=4.0)

        with pytest.raises(AttributeError):
            info.is_on = True  # type: ignore[misc]


# =============================================================================
# Tests for _extract_loop_info
# =============================================================================


class TestExtractLoopInfo:
    """Tests for loop info extraction from XML elements."""

    def test_extracts_all_loop_settings(self) -> None:
        """Extracts all loop parameters from well-formed XML."""
        clip_elem = create_midi_clip_xml(
            loop_on=True,
            loop_start=2.0,
            loop_end=6.0,
            start_relative=1.0,
        )

        info = _extract_loop_info(clip_elem)

        assert info.is_on is True
        assert info.start == 2.0
        assert info.end == 6.0
        assert info.start_relative == 1.0
        assert info.length == 4.0  # end - start

    def test_loop_off_returns_false(self) -> None:
        """LoopOn=false is correctly detected."""
        clip_elem = create_midi_clip_xml(loop_on=False)

        info = _extract_loop_info(clip_elem)

        assert info.is_on is False

    def test_missing_loop_element_returns_defaults(self) -> None:
        """Missing Loop element returns safe defaults."""
        clip_elem = ET.fromstring('<MidiClip Time="0"><CurrentEnd Value="8"/></MidiClip>')

        info = _extract_loop_info(clip_elem)

        assert info.is_on is False
        assert info.start == 0.0
        assert info.end == 0.0
        assert info.start_relative == 0.0
        assert info.length == 0.0

    def test_missing_individual_elements_returns_defaults(self) -> None:
        """Missing individual Loop children return defaults."""
        xml_str = """
        <MidiClip Time="0">
            <CurrentEnd Value="8"/>
            <Loop>
                <LoopOn Value="true"/>
            </Loop>
        </MidiClip>
        """
        clip_elem = ET.fromstring(xml_str)

        info = _extract_loop_info(clip_elem)

        assert info.is_on is True
        assert info.start == 0.0
        assert info.end == 0.0
        assert info.start_relative == 0.0

    def test_loop_length_calculated_correctly(self) -> None:
        """Loop length is calculated as end - start."""
        clip_elem = create_midi_clip_xml(loop_start=4.0, loop_end=12.0)

        info = _extract_loop_info(clip_elem)

        assert info.length == 8.0


# =============================================================================
# Tests for _extract_raw_notes
# =============================================================================


class TestExtractRawNotes:
    """Tests for raw note extraction from XML elements."""

    def test_extracts_notes_from_keytracks(self) -> None:
        """Extracts notes from KeyTracks structure."""
        notes_data = [
            (0.0, 0.5, 100, 36),
            (1.0, 0.5, 80, 38),
            (2.0, 0.5, 90, 36),
        ]
        clip_elem = create_midi_clip_xml(notes=notes_data)

        notes = _extract_raw_notes(clip_elem, start_relative=0.0)

        assert len(notes) == 3
        # Notes should have correct times, pitches, velocities
        times = sorted(n.time for n in notes)
        assert times == [0.0, 1.0, 2.0]

    def test_adjusts_times_by_start_relative(self) -> None:
        """Note times are adjusted by StartRelative offset."""
        notes_data = [
            (4.0, 0.5, 100, 60),  # Raw time is 4.0
            (6.0, 0.5, 100, 60),  # Raw time is 6.0
        ]
        clip_elem = create_midi_clip_xml(notes=notes_data)

        notes = _extract_raw_notes(clip_elem, start_relative=2.0)

        # Times should be adjusted: 4.0 - 2.0 = 2.0, 6.0 - 2.0 = 4.0
        times = sorted(n.time for n in notes)
        assert times == [2.0, 4.0]

    def test_skips_notes_before_start_relative(self) -> None:
        """Notes with adjusted time < 0 are skipped."""
        notes_data = [
            (0.0, 0.5, 100, 60),  # Before start_relative=2.0
            (1.0, 0.5, 100, 60),  # Before start_relative=2.0
            (4.0, 0.5, 100, 60),  # After start_relative=2.0, adjusted time = 2.0
        ]
        clip_elem = create_midi_clip_xml(notes=notes_data)

        notes = _extract_raw_notes(clip_elem, start_relative=2.0)

        assert len(notes) == 1
        assert notes[0].time == 2.0

    def test_skips_disabled_notes(self) -> None:
        """Notes with IsEnabled=false are skipped."""
        xml_str = """
        <MidiClip Time="0">
            <CurrentEnd Value="8"/>
            <Loop>
                <LoopOn Value="false"/>
                <StartRelative Value="0"/>
            </Loop>
            <Notes>
                <KeyTracks>
                    <KeyTrack Id="0">
                        <MidiKey Value="60"/>
                        <Notes>
                            <MidiNoteEvent Time="0" Duration="1.0" Velocity="100"/>
                            <MidiNoteEvent Time="2" Duration="1.0" Velocity="100" IsEnabled="false"/>
                            <MidiNoteEvent Time="4" Duration="1.0" Velocity="100"/>
                        </Notes>
                    </KeyTrack>
                </KeyTracks>
            </Notes>
        </MidiClip>
        """
        clip_elem = ET.fromstring(xml_str)

        notes = _extract_raw_notes(clip_elem, start_relative=0.0)

        assert len(notes) == 2
        times = sorted(n.time for n in notes)
        assert times == [0.0, 4.0]

    def test_extracts_pitch_from_midikey(self) -> None:
        """Pitch is correctly extracted from MidiKey element."""
        notes_data = [
            (0.0, 0.5, 100, 36),  # Kick
            (1.0, 0.5, 80, 38),  # Snare
            (2.0, 0.5, 90, 42),  # Hi-hat
        ]
        clip_elem = create_midi_clip_xml(notes=notes_data)

        notes = _extract_raw_notes(clip_elem, start_relative=0.0)

        pitches = {n.pitch for n in notes}
        assert pitches == {36, 38, 42}

    def test_extracts_velocity(self) -> None:
        """Velocity is correctly extracted."""
        notes_data = [
            (0.0, 0.5, 127, 60),
            (1.0, 0.5, 64, 60),
            (2.0, 0.5, 32, 60),
        ]
        clip_elem = create_midi_clip_xml(notes=notes_data)

        notes = _extract_raw_notes(clip_elem, start_relative=0.0)

        velocities = sorted(n.velocity for n in notes)
        assert velocities == [32, 64, 127]

    def test_extracts_duration(self) -> None:
        """Duration is correctly extracted."""
        notes_data = [
            (0.0, 0.25, 100, 60),
            (1.0, 0.5, 100, 60),
            (2.0, 2.0, 100, 60),
        ]
        clip_elem = create_midi_clip_xml(notes=notes_data)

        notes = _extract_raw_notes(clip_elem, start_relative=0.0)

        durations = sorted(n.duration for n in notes)
        assert durations == [0.25, 0.5, 2.0]

    def test_fallback_extracts_midinoteevents_directly(self) -> None:
        """Fallback path extracts MidiNoteEvent without KeyTracks."""
        notes_data = [
            (0.0, 0.5, 100, 60),
            (2.0, 0.5, 80, 60),
        ]
        clip_elem = create_midi_clip_xml(notes=notes_data, use_keytracks=False)

        notes = _extract_raw_notes(clip_elem, start_relative=0.0)

        assert len(notes) == 2
        # Fallback sets pitch to 0 since there's no MidiKey
        assert all(n.pitch == 0 for n in notes)

    def test_empty_clip_returns_empty_list(self) -> None:
        """Clip without notes returns empty list."""
        clip_elem = create_midi_clip_xml(notes=None)

        notes = _extract_raw_notes(clip_elem, start_relative=0.0)

        assert notes == []


# =============================================================================
# Tests for _expand_looped_notes
# =============================================================================


class TestExpandLoopedNotes:
    """Tests for loop expansion of notes."""

    def test_expands_notes_to_fill_clip_duration(self) -> None:
        """Notes within loop region are repeated to fill clip."""
        base_notes = [
            MidiNote(time=0.0, duration=0.5, velocity=100, pitch=60),
            MidiNote(time=1.0, duration=0.5, velocity=100, pitch=62),
        ]
        loop_info = _LoopInfo(
            is_on=True,
            start=0.0,
            end=4.0,
            start_relative=0.0,
            length=4.0,
        )
        clip_length = 16.0  # 4 repetitions of 4-beat loop

        expanded = _expand_looped_notes(base_notes, loop_info, clip_length)

        # 2 notes per loop * 4 repetitions = 8 notes
        assert len(expanded) == 8

    def test_correct_timing_for_expanded_notes(self) -> None:
        """Expanded notes have correct timing offsets."""
        base_notes = [
            MidiNote(time=0.0, duration=0.5, velocity=100, pitch=60),
        ]
        loop_info = _LoopInfo(
            is_on=True,
            start=0.0,
            end=4.0,
            start_relative=0.0,
            length=4.0,
        )
        clip_length = 12.0  # 3 repetitions

        expanded = _expand_looped_notes(base_notes, loop_info, clip_length)

        times = [n.time for n in expanded]
        assert times == [0.0, 4.0, 8.0]

    def test_non_loopable_notes_play_once(self) -> None:
        """Notes outside loop region play only once."""
        base_notes = [
            MidiNote(time=0.0, duration=0.5, velocity=100, pitch=60),  # In loop
            MidiNote(time=6.0, duration=0.5, velocity=100, pitch=62),  # Outside loop
        ]
        loop_info = _LoopInfo(
            is_on=True,
            start=0.0,
            end=4.0,
            start_relative=0.0,
            length=4.0,
        )
        clip_length = 16.0

        expanded = _expand_looped_notes(base_notes, loop_info, clip_length)

        # 4 repetitions of note at 0.0 + 1 non-looped note at 6.0 = 5 notes
        assert len(expanded) == 5
        # The non-looped note should appear once at time 6.0
        times_at_6 = [n for n in expanded if n.time == 6.0]
        assert len(times_at_6) == 1

    def test_preserves_note_attributes_during_expansion(self) -> None:
        """Expanded notes preserve duration, velocity, and pitch."""
        base_notes = [
            MidiNote(time=1.0, duration=0.75, velocity=85, pitch=48),
        ]
        loop_info = _LoopInfo(
            is_on=True,
            start=0.0,
            end=4.0,
            start_relative=0.0,
            length=4.0,
        )
        clip_length = 8.0

        expanded = _expand_looped_notes(base_notes, loop_info, clip_length)

        for note in expanded:
            assert note.duration == 0.75
            assert note.velocity == 85
            assert note.pitch == 48

    def test_does_not_expand_beyond_clip_length(self) -> None:
        """Notes are not created beyond clip_length."""
        base_notes = [
            MidiNote(time=0.0, duration=0.5, velocity=100, pitch=60),
            MidiNote(time=3.5, duration=0.5, velocity=100, pitch=60),
        ]
        loop_info = _LoopInfo(
            is_on=True,
            start=0.0,
            end=4.0,
            start_relative=0.0,
            length=4.0,
        )
        clip_length = 6.0  # Not a full second loop

        expanded = _expand_looped_notes(base_notes, loop_info, clip_length)

        # First repetition: 0.0, 3.5 (both within clip_length=6)
        # Second repetition: 4.0 (within), 7.5 (beyond clip_length) - skip 7.5
        times = sorted(n.time for n in expanded)
        assert times == [0.0, 3.5, 4.0]
        assert all(n.time < clip_length for n in expanded)

    def test_handles_empty_base_notes(self) -> None:
        """Empty base notes returns empty list."""
        loop_info = _LoopInfo(
            is_on=True,
            start=0.0,
            end=4.0,
            start_relative=0.0,
            length=4.0,
        )

        expanded = _expand_looped_notes([], loop_info, clip_length=16.0)

        assert expanded == []


# =============================================================================
# Tests for extract_midi_notes (main function)
# =============================================================================


class TestExtractMidiNotes:
    """Tests for the main extract_midi_notes orchestration function."""

    def test_extracts_notes_without_looping(self) -> None:
        """Basic extraction without loop expansion."""
        notes_data = [
            (0.0, 0.5, 100, 36),
            (1.0, 0.5, 80, 38),
            (2.0, 0.5, 90, 36),
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=8.0,
            loop_on=False,
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        assert len(notes) == 3
        assert isinstance(notes, tuple)

    def test_returns_notes_sorted_by_time(self) -> None:
        """Notes are returned sorted by time."""
        notes_data = [
            (4.0, 0.5, 100, 60),
            (0.0, 0.5, 100, 60),
            (2.0, 0.5, 100, 60),
        ]
        clip_elem = create_midi_clip_xml(notes=notes_data, loop_on=False)

        notes = extract_midi_notes(clip_elem)

        times = [n.time for n in notes]
        assert times == [0.0, 2.0, 4.0]

    def test_applies_start_relative_adjustment(self) -> None:
        """StartRelative offset is applied to note times."""
        notes_data = [
            (4.0, 0.5, 100, 60),
            (6.0, 0.5, 100, 60),
        ]
        clip_elem = create_midi_clip_xml(
            notes=notes_data,
            loop_on=False,
            start_relative=2.0,
        )

        notes = extract_midi_notes(clip_elem)

        times = [n.time for n in notes]
        assert times == [2.0, 4.0]

    def test_expands_looped_notes_when_clip_longer_than_loop(self) -> None:
        """Notes are expanded when loop is on and clip is longer than loop."""
        notes_data = [
            (0.0, 0.5, 100, 60),
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=16.0,  # Clip is 16 beats
            loop_on=True,
            loop_start=0.0,
            loop_end=4.0,  # Loop is 4 beats
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # 16-beat clip / 4-beat loop = 4 repetitions
        assert len(notes) == 4
        times = [n.time for n in notes]
        assert times == [0.0, 4.0, 8.0, 12.0]

    def test_no_expansion_when_loop_off(self) -> None:
        """No expansion when LoopOn is false."""
        notes_data = [
            (0.0, 0.5, 100, 60),
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=16.0,
            loop_on=False,  # Loop is OFF
            loop_start=0.0,
            loop_end=4.0,
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        assert len(notes) == 1

    def test_no_expansion_when_clip_shorter_than_loop(self) -> None:
        """No expansion when clip is shorter than loop region."""
        notes_data = [
            (0.0, 0.5, 100, 60),
            (2.0, 0.5, 100, 60),
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=4.0,  # Clip is 4 beats (same as loop)
            loop_on=True,
            loop_start=0.0,
            loop_end=4.0,  # Loop is also 4 beats
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # No expansion needed - clip matches loop length
        assert len(notes) == 2

    def test_empty_clip_returns_empty_tuple(self) -> None:
        """Empty clip returns empty tuple."""
        clip_elem = create_midi_clip_xml(notes=None)

        notes = extract_midi_notes(clip_elem)

        assert notes == ()

    def test_calculates_clip_length_correctly(self) -> None:
        """Clip length is calculated from Time and CurrentEnd."""
        notes_data = [
            (0.0, 0.5, 100, 60),
        ]
        # Clip starts at beat 8, ends at beat 24 = 16 beats
        clip_elem = create_midi_clip_xml(
            time=8.0,
            current_end=24.0,
            loop_on=True,
            loop_start=0.0,
            loop_end=4.0,
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # 16-beat clip / 4-beat loop = 4 repetitions
        assert len(notes) == 4


# =============================================================================
# Integration Tests: Real-World Scenarios
# =============================================================================


class TestRealWorldScenarios:
    """Tests simulating real Ableton Live Set scenarios."""

    def test_drum_rack_with_multiple_pads(self) -> None:
        """Drum rack pattern with kick, snare, and hi-hat."""
        # Typical 4-beat drum pattern
        notes_data = [
            # Kick on 1 and 3
            (0.0, 0.25, 100, 36),
            (2.0, 0.25, 100, 36),
            # Snare on 2 and 4
            (1.0, 0.25, 80, 38),
            (3.0, 0.25, 80, 38),
            # Hi-hat on every eighth note
            (0.0, 0.125, 60, 42),
            (0.5, 0.125, 50, 42),
            (1.0, 0.125, 60, 42),
            (1.5, 0.125, 50, 42),
            (2.0, 0.125, 60, 42),
            (2.5, 0.125, 50, 42),
            (3.0, 0.125, 60, 42),
            (3.5, 0.125, 50, 42),
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=16.0,  # 4 bars
            loop_on=True,
            loop_start=0.0,
            loop_end=4.0,  # 1-bar loop
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # 12 notes per bar * 4 bars = 48 notes
        assert len(notes) == 48

        # Verify kick pattern
        kicks = [n for n in notes if n.pitch == 36]
        assert len(kicks) == 8  # 2 per bar * 4 bars

        # Verify snare pattern
        snares = [n for n in notes if n.pitch == 38]
        assert len(snares) == 8  # 2 per bar * 4 bars

        # Verify hi-hat pattern
        hats = [n for n in notes if n.pitch == 42]
        assert len(hats) == 32  # 8 per bar * 4 bars

    def test_bass_pattern_with_offset_start(self) -> None:
        """Bass pattern starting mid-clip with StartRelative."""
        notes_data = [
            # Notes in "internal" coordinates
            (4.0, 1.0, 100, 36),  # Will become 2.0 after adjustment
            (6.0, 1.0, 100, 36),  # Will become 4.0 after adjustment
            (8.0, 1.0, 100, 36),  # Will become 6.0 after adjustment
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=8.0,
            loop_on=False,
            start_relative=2.0,
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        times = [n.time for n in notes]
        assert times == [2.0, 4.0, 6.0]

    def test_sustained_pad_chord(self) -> None:
        """Sustained chord (multiple pitches, long duration)."""
        notes_data = [
            # C major chord sustained for 4 beats
            (0.0, 4.0, 80, 60),  # C
            (0.0, 4.0, 80, 64),  # E
            (0.0, 4.0, 80, 67),  # G
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=16.0,
            loop_on=True,
            loop_start=0.0,
            loop_end=4.0,
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # 3 notes per loop * 4 repetitions = 12 notes
        assert len(notes) == 12

        # All notes should have 4-beat duration
        assert all(n.duration == 4.0 for n in notes)

    def test_one_shot_sample_no_loop(self) -> None:
        """One-shot sample (no looping, plays once)."""
        notes_data = [
            (0.0, 2.0, 127, 60),
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=32.0,  # Long clip
            loop_on=False,  # No looping
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # Should play once regardless of clip length
        assert len(notes) == 1

    def test_complex_arpeggio_pattern(self) -> None:
        """Complex arpeggio with varying velocities."""
        notes_data = [
            (0.0, 0.25, 100, 60),
            (0.25, 0.25, 80, 64),
            (0.5, 0.25, 100, 67),
            (0.75, 0.25, 80, 72),
            (1.0, 0.25, 100, 67),
            (1.25, 0.25, 80, 64),
            (1.5, 0.25, 100, 60),
            (1.75, 0.25, 80, 55),
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=8.0,
            loop_on=True,
            loop_start=0.0,
            loop_end=2.0,  # 2-beat loop
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # 8 notes per loop * 4 repetitions = 32 notes
        assert len(notes) == 32

        # Verify velocity pattern is preserved
        velocities = [n.velocity for n in notes[:8]]
        expected = [100, 80, 100, 80, 100, 80, 100, 80]
        assert velocities == expected


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_length_loop(self) -> None:
        """Zero-length loop doesn't cause issues."""
        notes_data = [(0.0, 0.5, 100, 60)]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=8.0,
            loop_on=True,
            loop_start=0.0,
            loop_end=0.0,  # Zero-length loop
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # Should not expand (loop_length = 0)
        assert len(notes) == 1

    def test_negative_start_relative_notes_filtered(self) -> None:
        """Notes that would have negative time after adjustment are filtered."""
        notes_data = [
            (0.0, 0.5, 100, 60),
            (1.0, 0.5, 100, 60),
            (2.0, 0.5, 100, 60),
        ]
        clip_elem = create_midi_clip_xml(
            start_relative=10.0,  # Very large offset
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # All notes would have negative times, so all filtered
        assert len(notes) == 0

    def test_very_short_clip(self) -> None:
        """Very short clip duration is handled."""
        notes_data = [(0.0, 0.125, 100, 60)]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=0.25,  # Quarter beat
            loop_on=True,
            loop_start=0.0,
            loop_end=0.125,
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # Very short loop, short clip - 2 repetitions
        assert len(notes) == 2

    def test_fractional_loop_boundaries(self) -> None:
        """Fractional beat loop boundaries work correctly."""
        notes_data = [
            (0.0, 0.5, 100, 60),
            (0.75, 0.25, 100, 60),
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=6.0,  # 6 beats
            loop_on=True,
            loop_start=0.0,
            loop_end=1.5,  # 1.5 beat loop
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # 6 / 1.5 = 4 repetitions
        assert len(notes) == 8  # 2 notes * 4 repetitions

    def test_missing_current_end_element(self) -> None:
        """Missing CurrentEnd uses clip Time as fallback (zero length)."""
        xml_str = """
        <MidiClip Time="8">
            <Loop>
                <LoopOn Value="false"/>
                <StartRelative Value="0"/>
            </Loop>
            <Notes>
                <KeyTracks>
                    <KeyTrack Id="0">
                        <MidiKey Value="60"/>
                        <Notes>
                            <MidiNoteEvent Time="0" Duration="1.0" Velocity="100"/>
                        </Notes>
                    </KeyTrack>
                </KeyTracks>
            </Notes>
        </MidiClip>
        """
        clip_elem = ET.fromstring(xml_str)

        notes = extract_midi_notes(clip_elem)

        # Should still extract the note
        assert len(notes) == 1

    def test_all_notes_disabled(self) -> None:
        """Clip with all disabled notes returns empty."""
        xml_str = """
        <MidiClip Time="0">
            <CurrentEnd Value="8"/>
            <Loop>
                <LoopOn Value="false"/>
                <StartRelative Value="0"/>
            </Loop>
            <Notes>
                <KeyTracks>
                    <KeyTrack Id="0">
                        <MidiKey Value="60"/>
                        <Notes>
                            <MidiNoteEvent Time="0" Duration="1.0" Velocity="100" IsEnabled="false"/>
                            <MidiNoteEvent Time="2" Duration="1.0" Velocity="100" IsEnabled="false"/>
                        </Notes>
                    </KeyTrack>
                </KeyTracks>
            </Notes>
        </MidiClip>
        """
        clip_elem = ET.fromstring(xml_str)

        notes = extract_midi_notes(clip_elem)

        assert notes == ()

    def test_notes_exactly_at_loop_boundary(self) -> None:
        """Notes exactly at loop end are treated as non-loopable.

        A note at exactly the loop end position (time == loop_length) is
        considered outside the loop region (which is 0 <= time < loop_length).
        This means:
        - Notes inside the loop (0 <= time < loop_length) repeat with each iteration
        - Notes at or after loop_length play once at their original position

        In this test, we have overlapping notes at time 4.0 because:
        1. The note at 0.0 loops to 0, 4, 8, 12
        2. The note at 4.0 plays once as a non-looped note

        This results in two notes at time 4.0 (one from loop, one non-looped).
        """
        notes_data = [
            (0.0, 0.5, 100, 60),
            (4.0, 0.5, 100, 60),  # Exactly at loop end - will not loop
        ]
        clip_elem = create_midi_clip_xml(
            time=0.0,
            current_end=16.0,
            loop_on=True,
            loop_start=0.0,
            loop_end=4.0,  # Loop ends at 4.0
            notes=notes_data,
        )

        notes = extract_midi_notes(clip_elem)

        # Note at 0.0 loops to: 0, 4, 8, 12 (4 notes)
        # Note at 4.0 is non-loopable, plays once at 4.0 (1 note)
        # Total: 5 notes
        assert len(notes) == 5

        # Two notes at time 4.0: one from loop iteration, one non-looped
        notes_at_4 = [n for n in notes if n.time == 4.0]
        assert len(notes_at_4) == 2

        # Verify the looped note appears at expected times
        looped_times = [n.time for n in notes if n.time in [0.0, 4.0, 8.0, 12.0]]
        assert len(looped_times) == 5  # 4 from loop + 1 non-looped at 4.0
