"""Tests for MIDI event detection functionality.

These are high-level functional tests that test the observable behavior
of the event detection pipeline, not implementation details.
"""

import pytest

from alsmuse.events import (
    categorize_track,
    detect_track_events,
    merge_events_into_phrases,
)
from alsmuse.midi import detect_midi_activity, extract_midi_notes
from alsmuse.models import Clip, MidiClipContent, MidiNote, Phrase, TrackEvent


class TestCategorizeTrack:
    """Tests for track categorization based on name."""

    def test_drums_keywords(self) -> None:
        """Tracks with drum-related keywords are categorized as drums."""
        assert categorize_track("Drum Kit") == "drums"
        assert categorize_track("KICK") == "drums"
        assert categorize_track("snare track") == "drums"
        assert categorize_track("Hi-Hat") == "drums"
        assert categorize_track("Percussion") == "drums"

    def test_bass_keywords(self) -> None:
        """Tracks with bass-related keywords are categorized as bass."""
        assert categorize_track("Bass Line") == "bass"
        assert categorize_track("BASS") == "bass"
        assert categorize_track("Sub Bass") == "bass"

    def test_vocals_keywords(self) -> None:
        """Tracks with vocal-related keywords are categorized as vocals."""
        assert categorize_track("Main Vocal") == "vocals"
        assert categorize_track("VOX") == "vocals"
        assert categorize_track("Verse 1") == "vocals"
        assert categorize_track("Chorus Vocal") == "vocals"
        assert categorize_track("Harmony") == "vocals"

    def test_other_categories(self) -> None:
        """Other known keywords categorize correctly."""
        assert categorize_track("Lead Synth") == "lead"
        assert categorize_track("Guitar") == "guitar"
        assert categorize_track("Piano") == "keys"
        assert categorize_track("Pad Layer") == "pad"
        assert categorize_track("FX Riser") == "fx"

    def test_unknown_track_returns_other(self) -> None:
        """Tracks without matching keywords return 'other'."""
        assert categorize_track("Track 7") == "other"
        assert categorize_track("") == "other"
        assert categorize_track("My Cool Sound") == "other"

    def test_case_insensitivity(self) -> None:
        """Categorization is case-insensitive."""
        assert categorize_track("DRUMS") == "drums"
        assert categorize_track("drums") == "drums"
        assert categorize_track("Drums") == "drums"
        assert categorize_track("DrUmS") == "drums"


class TestMidiNote:
    """Tests for MidiNote dataclass."""

    def test_midi_note_creation(self) -> None:
        """MidiNote stores all expected attributes."""
        note = MidiNote(time=0.0, duration=1.0, velocity=100, pitch=60)

        assert note.time == 0.0
        assert note.duration == 1.0
        assert note.velocity == 100
        assert note.pitch == 60

    def test_midi_note_is_immutable(self) -> None:
        """MidiNote is frozen and cannot be modified."""
        note = MidiNote(time=0.0, duration=1.0, velocity=100, pitch=60)

        with pytest.raises(AttributeError):
            note.time = 2.0  # type: ignore[misc]


class TestMidiClipContent:
    """Tests for MidiClipContent range queries."""

    def test_has_notes_in_range_note_starts_in_range(self) -> None:
        """Detects notes that start within the range."""
        clip = Clip(name="test", start_beats=0, end_beats=8)
        notes = (MidiNote(time=2.0, duration=1.0, velocity=100, pitch=60),)
        content = MidiClipContent(clip=clip, notes=notes)

        assert content.has_notes_in_range(1.0, 4.0) is True

    def test_has_notes_in_range_note_ends_in_range(self) -> None:
        """Detects notes that end within the range."""
        clip = Clip(name="test", start_beats=0, end_beats=8)
        notes = (MidiNote(time=0.0, duration=2.0, velocity=100, pitch=60),)
        content = MidiClipContent(clip=clip, notes=notes)

        assert content.has_notes_in_range(1.0, 4.0) is True

    def test_has_notes_in_range_note_spans_range(self) -> None:
        """Detects notes that completely span the range."""
        clip = Clip(name="test", start_beats=0, end_beats=8)
        notes = (MidiNote(time=0.0, duration=8.0, velocity=100, pitch=60),)
        content = MidiClipContent(clip=clip, notes=notes)

        assert content.has_notes_in_range(2.0, 4.0) is True

    def test_has_notes_in_range_note_before_range(self) -> None:
        """Does not detect notes that end before the range."""
        clip = Clip(name="test", start_beats=0, end_beats=8)
        notes = (MidiNote(time=0.0, duration=1.0, velocity=100, pitch=60),)
        content = MidiClipContent(clip=clip, notes=notes)

        assert content.has_notes_in_range(2.0, 4.0) is False

    def test_has_notes_in_range_note_after_range(self) -> None:
        """Does not detect notes that start after the range."""
        clip = Clip(name="test", start_beats=0, end_beats=8)
        notes = (MidiNote(time=5.0, duration=1.0, velocity=100, pitch=60),)
        content = MidiClipContent(clip=clip, notes=notes)

        assert content.has_notes_in_range(2.0, 4.0) is False

    def test_has_notes_in_range_empty_notes(self) -> None:
        """Empty notes tuple returns False."""
        clip = Clip(name="test", start_beats=0, end_beats=8)
        content = MidiClipContent(clip=clip, notes=())

        assert content.has_notes_in_range(0.0, 8.0) is False

    def test_note_density_calculation(self) -> None:
        """Note density is calculated as notes per beat."""
        clip = Clip(name="test", start_beats=0, end_beats=8)
        notes = (
            MidiNote(time=0.0, duration=1.0, velocity=100, pitch=60),
            MidiNote(time=2.0, duration=1.0, velocity=100, pitch=60),
            MidiNote(time=4.0, duration=1.0, velocity=100, pitch=60),
            MidiNote(time=6.0, duration=1.0, velocity=100, pitch=60),
        )
        content = MidiClipContent(clip=clip, notes=notes)

        assert content.note_density() == 0.5  # 4 notes / 8 beats

    def test_note_density_empty(self) -> None:
        """Empty notes returns zero density."""
        clip = Clip(name="test", start_beats=0, end_beats=8)
        content = MidiClipContent(clip=clip, notes=())

        assert content.note_density() == 0.0


class TestDetectMidiActivity:
    """Tests for MIDI activity detection."""

    def test_detects_activity_in_window(self) -> None:
        """Detects activity when notes fall within a window."""
        clip = Clip(name="test", start_beats=0, end_beats=16)
        notes = (MidiNote(time=2.0, duration=1.0, velocity=100, pitch=60),)
        content = MidiClipContent(clip=clip, notes=notes)

        activity = detect_midi_activity([content], resolution_beats=8.0)

        assert len(activity) == 2
        assert activity[0] == (0.0, True)  # Notes at beat 2
        assert activity[1] == (8.0, False)  # No notes in 8-16

    def test_detects_no_activity_in_empty_clip(self) -> None:
        """No activity when clip has no notes."""
        clip = Clip(name="test", start_beats=0, end_beats=16)
        content = MidiClipContent(clip=clip, notes=())

        activity = detect_midi_activity([content], resolution_beats=8.0)

        assert len(activity) == 2
        assert activity[0] == (0.0, False)
        assert activity[1] == (8.0, False)

    def test_empty_contents_returns_empty(self) -> None:
        """Empty clip contents returns empty activity."""
        activity = detect_midi_activity([], resolution_beats=8.0)
        assert activity == []

    def test_multiple_clips_detected(self) -> None:
        """Activity is detected across multiple clips."""
        clip1 = Clip(name="test1", start_beats=0, end_beats=8)
        clip2 = Clip(name="test2", start_beats=8, end_beats=16)
        notes1 = (MidiNote(time=2.0, duration=1.0, velocity=100, pitch=60),)
        notes2 = (MidiNote(time=2.0, duration=1.0, velocity=100, pitch=60),)

        content1 = MidiClipContent(clip=clip1, notes=notes1)
        content2 = MidiClipContent(clip=clip2, notes=notes2)

        activity = detect_midi_activity([content1, content2], resolution_beats=8.0)

        assert len(activity) == 2
        assert activity[0] == (0.0, True)  # Notes in first clip
        assert activity[1] == (8.0, True)  # Notes in second clip


class TestDetectTrackEvents:
    """Tests for track event detection from activity."""

    def test_enter_event_on_activity_start(self) -> None:
        """Generates enter event when activity starts."""
        activity = [(0.0, False), (8.0, True), (16.0, True)]

        events = detect_track_events("Drums", activity, "drums")

        assert len(events) == 1
        assert events[0].beat == 8.0
        assert events[0].event_type == "enter"
        assert events[0].category == "drums"

    def test_exit_event_on_activity_stop(self) -> None:
        """Generates exit event when activity stops."""
        activity = [(0.0, True), (8.0, True), (16.0, False)]

        events = detect_track_events("Bass", activity, "bass")

        # First activity from start generates an enter, then exit when stops
        assert len(events) == 2
        assert events[0].beat == 0.0
        assert events[0].event_type == "enter"
        assert events[1].beat == 16.0
        assert events[1].event_type == "exit"
        assert events[1].category == "bass"

    def test_enter_and_exit_events(self) -> None:
        """Generates both enter and exit for activity island."""
        activity = [(0.0, False), (8.0, True), (16.0, False)]

        events = detect_track_events("Lead", activity, "lead")

        assert len(events) == 2
        assert events[0].event_type == "enter"
        assert events[0].beat == 8.0
        assert events[1].event_type == "exit"
        assert events[1].beat == 16.0

    def test_no_events_for_constant_inactive(self) -> None:
        """No events when activity is constant False (never plays)."""
        activity_all_false = [(0.0, False), (8.0, False), (16.0, False)]

        events_false = detect_track_events("Test", activity_all_false, "other")

        assert len(events_false) == 0

    def test_single_enter_for_constant_active(self) -> None:
        """Single enter event when track plays from start to end."""
        activity_all_true = [(0.0, True), (8.0, True), (16.0, True)]

        events_true = detect_track_events("Test", activity_all_true, "other")

        # Track enters at start and never exits
        assert len(events_true) == 1
        assert events_true[0].event_type == "enter"
        assert events_true[0].beat == 0.0

    def test_empty_activity_returns_empty(self) -> None:
        """Empty activity returns no events."""
        events = detect_track_events("Test", [], "other")
        assert events == []


class TestMergeEventsIntoPhrases:
    """Tests for merging events into phrase structures."""

    def test_events_assigned_to_correct_phrase(self) -> None:
        """Events are assigned to the phrase that contains them."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="INTRO", is_section_start=True),
            Phrase(start_beats=8, end_beats=16, section_name="...", is_section_start=False),
        ]
        events = [
            TrackEvent(beat=4.0, track_name="Drums", event_type="enter", category="drums"),
            TrackEvent(beat=12.0, track_name="Bass", event_type="enter", category="bass"),
        ]

        result = merge_events_into_phrases(phrases, events)

        assert len(result[0].events) == 1
        assert result[0].events[0].category == "drums"
        assert len(result[1].events) == 1
        assert result[1].events[0].category == "bass"

    def test_deduplicates_by_category(self) -> None:
        """Multiple events of same category in phrase are deduplicated."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="INTRO", is_section_start=True),
        ]
        events = [
            TrackEvent(beat=2.0, track_name="Kick", event_type="enter", category="drums"),
            TrackEvent(beat=4.0, track_name="Snare", event_type="enter", category="drums"),
        ]

        result = merge_events_into_phrases(phrases, events)

        # Should only have one drums enter event
        assert len(result[0].events) == 1
        assert result[0].events[0].category == "drums"
        assert result[0].events[0].event_type == "enter"

    def test_different_event_types_not_deduplicated(self) -> None:
        """Enter and exit for same category are both kept."""
        phrases = [
            Phrase(start_beats=0, end_beats=8, section_name="INTRO", is_section_start=True),
        ]
        events = [
            TrackEvent(beat=2.0, track_name="Drums", event_type="enter", category="drums"),
            TrackEvent(beat=6.0, track_name="Drums", event_type="exit", category="drums"),
        ]

        result = merge_events_into_phrases(phrases, events)

        assert len(result[0].events) == 2
        event_types = {e.event_type for e in result[0].events}
        assert event_types == {"enter", "exit"}

    def test_preserves_phrase_attributes(self) -> None:
        """All phrase attributes are preserved after merge."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=8,
                section_name="INTRO",
                is_section_start=True,
                lyric="test lyric",
            ),
        ]

        result = merge_events_into_phrases(phrases, [])

        assert result[0].start_beats == 0
        assert result[0].end_beats == 8
        assert result[0].section_name == "INTRO"
        assert result[0].is_section_start is True
        assert result[0].lyric == "test lyric"

    def test_empty_phrases_returns_empty(self) -> None:
        """Empty phrases list returns empty."""
        result = merge_events_into_phrases([], [])
        assert result == []


class TestExtractMidiNotesFromXML:
    """Tests for MIDI note extraction from XML elements."""

    def test_extracts_notes_from_keytrack_structure(self) -> None:
        """Extracts notes from standard KeyTrack XML structure."""
        import xml.etree.ElementTree as ET

        xml_str = """
        <MidiClip>
            <Notes>
                <KeyTracks>
                    <KeyTrack Id="0">
                        <MidiKey Value="36"/>
                        <Notes>
                            <MidiNoteEvent Time="0" Duration="0.5" Velocity="97"/>
                            <MidiNoteEvent Time="2" Duration="0.5" Velocity="100"/>
                        </Notes>
                    </KeyTrack>
                </KeyTracks>
            </Notes>
        </MidiClip>
        """
        clip_elem = ET.fromstring(xml_str)

        notes = extract_midi_notes(clip_elem)

        assert len(notes) == 2
        assert notes[0].time == 0.0
        assert notes[0].pitch == 36
        assert notes[0].velocity == 97
        assert notes[1].time == 2.0

    def test_extracts_notes_from_multiple_keytracks(self) -> None:
        """Extracts notes from multiple KeyTrack elements (drum rack)."""
        import xml.etree.ElementTree as ET

        xml_str = """
        <MidiClip>
            <Notes>
                <KeyTracks>
                    <KeyTrack Id="0">
                        <MidiKey Value="36"/>
                        <Notes>
                            <MidiNoteEvent Time="0" Duration="0.5" Velocity="100"/>
                        </Notes>
                    </KeyTrack>
                    <KeyTrack Id="1">
                        <MidiKey Value="38"/>
                        <Notes>
                            <MidiNoteEvent Time="1" Duration="0.5" Velocity="80"/>
                        </Notes>
                    </KeyTrack>
                </KeyTracks>
            </Notes>
        </MidiClip>
        """
        clip_elem = ET.fromstring(xml_str)

        notes = extract_midi_notes(clip_elem)

        assert len(notes) == 2
        pitches = {n.pitch for n in notes}
        assert pitches == {36, 38}

    def test_skips_disabled_notes(self) -> None:
        """Disabled notes are not extracted."""
        import xml.etree.ElementTree as ET

        xml_str = """
        <MidiClip>
            <Notes>
                <KeyTracks>
                    <KeyTrack Id="0">
                        <MidiKey Value="60"/>
                        <Notes>
                            <MidiNoteEvent Time="0" Duration="1.0" Velocity="100"/>
                            <MidiNoteEvent Time="2" Duration="1.0" Velocity="100"
                                           IsEnabled="false"/>
                        </Notes>
                    </KeyTrack>
                </KeyTracks>
            </Notes>
        </MidiClip>
        """
        clip_elem = ET.fromstring(xml_str)

        notes = extract_midi_notes(clip_elem)

        assert len(notes) == 1
        assert notes[0].time == 0.0

    def test_returns_sorted_by_time(self) -> None:
        """Notes are returned sorted by time."""
        import xml.etree.ElementTree as ET

        xml_str = """
        <MidiClip>
            <Notes>
                <KeyTracks>
                    <KeyTrack Id="0">
                        <MidiKey Value="60"/>
                        <Notes>
                            <MidiNoteEvent Time="4" Duration="1.0" Velocity="100"/>
                            <MidiNoteEvent Time="0" Duration="1.0" Velocity="100"/>
                            <MidiNoteEvent Time="2" Duration="1.0" Velocity="100"/>
                        </Notes>
                    </KeyTrack>
                </KeyTracks>
            </Notes>
        </MidiClip>
        """
        clip_elem = ET.fromstring(xml_str)

        notes = extract_midi_notes(clip_elem)

        assert notes[0].time == 0.0
        assert notes[1].time == 2.0
        assert notes[2].time == 4.0

    def test_empty_clip_returns_empty_tuple(self) -> None:
        """Clip without notes returns empty tuple."""
        import xml.etree.ElementTree as ET

        xml_str = """
        <MidiClip>
            <Notes>
                <KeyTracks>
                </KeyTracks>
            </Notes>
        </MidiClip>
        """
        clip_elem = ET.fromstring(xml_str)

        notes = extract_midi_notes(clip_elem)

        assert notes == ()


class TestStroboscopeAvoidance:
    """Tests verifying that range queries avoid the stroboscope problem.

    The stroboscope problem occurs when point sampling at fixed intervals
    (e.g., beat 0, 8, 16) misses activity between sample points.
    """

    def test_detects_off_beat_notes(self) -> None:
        """Notes on off-beats are still detected within their window."""
        clip = Clip(name="test", start_beats=0, end_beats=8)
        # Note at beat 3.5 - between standard sample points
        notes = (MidiNote(time=3.5, duration=0.5, velocity=100, pitch=60),)
        content = MidiClipContent(clip=clip, notes=notes)

        activity = detect_midi_activity([content], resolution_beats=8.0)

        assert activity[0] == (0.0, True)  # Should detect the off-beat note

    def test_detects_short_fills(self) -> None:
        """Short fills between sample points are detected."""
        clip = Clip(name="test", start_beats=0, end_beats=16)
        # Quick fill at beats 7-8 (just before a sample point)
        notes = (
            MidiNote(time=7.0, duration=0.25, velocity=100, pitch=60),
            MidiNote(time=7.25, duration=0.25, velocity=100, pitch=62),
            MidiNote(time=7.5, duration=0.25, velocity=100, pitch=64),
            MidiNote(time=7.75, duration=0.25, velocity=100, pitch=66),
        )
        content = MidiClipContent(clip=clip, notes=notes)

        activity = detect_midi_activity([content], resolution_beats=8.0)

        assert activity[0] == (0.0, True)  # Fill is detected in first window

    def test_long_sustained_note_spans_windows(self) -> None:
        """Long sustained notes are detected in all windows they span."""
        clip = Clip(name="test", start_beats=0, end_beats=24)
        # Long note from beat 2 to beat 18 (spans windows 0-8, 8-16, and into 16-24)
        notes = (MidiNote(time=2.0, duration=16.0, velocity=100, pitch=60),)
        content = MidiClipContent(clip=clip, notes=notes)

        activity = detect_midi_activity([content], resolution_beats=8.0)

        assert len(activity) == 3
        assert activity[0] == (0.0, True)  # Note starts here
        assert activity[1] == (8.0, True)  # Note still playing
        assert activity[2] == (16.0, True)  # Note still playing (ends at 18)
