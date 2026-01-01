"""MIDI analysis for ALSmuse.

This module provides functions to extract MIDI notes from ALS XML elements
and detect track activity using range-based queries.
"""

from dataclasses import dataclass
from xml.etree.ElementTree import Element

from .models import Clip, MidiClipContent, MidiNote


@dataclass(frozen=True)
class _LoopInfo:
    """Internal helper to hold loop settings."""

    is_on: bool
    start: float
    end: float
    start_relative: float
    length: float


def _extract_loop_info(clip_element: Element) -> _LoopInfo:
    """Extract loop settings from a clip element."""
    loop_on = False
    loop_start = 0.0
    loop_end = 0.0
    start_relative = 0.0

    loop_elem = clip_element.find("Loop")
    if loop_elem is not None:
        loop_on_elem = loop_elem.find("LoopOn")
        loop_on = loop_on_elem is not None and loop_on_elem.get("Value") == "true"

        loop_start_elem = loop_elem.find("LoopStart")
        if loop_start_elem is not None:
            loop_start = float(loop_start_elem.get("Value", "0"))

        loop_end_elem = loop_elem.find("LoopEnd")
        if loop_end_elem is not None:
            loop_end = float(loop_end_elem.get("Value", "0"))

        start_rel_elem = loop_elem.find("StartRelative")
        if start_rel_elem is not None:
            start_relative = float(start_rel_elem.get("Value", "0"))

    return _LoopInfo(
        is_on=loop_on,
        start=loop_start,
        end=loop_end,
        start_relative=start_relative,
        length=loop_end - loop_start,
    )


def _extract_raw_notes(clip_element: Element, start_relative: float) -> list[MidiNote]:
    """Extract raw notes from clip using both strategies."""
    base_notes: list[MidiNote] = []

    # Strategy 1: KeyTracks structure (Drum Racks, most MIDI clips)
    for key_track in clip_element.findall(".//KeyTrack"):
        midi_key_elem = key_track.find("MidiKey")
        pitch = int(midi_key_elem.get("Value", "0")) if midi_key_elem is not None else 0

        for note_event in key_track.findall(".//MidiNoteEvent"):
            if note_event.get("IsEnabled") == "false":
                continue

            raw_time = float(note_event.get("Time", "0"))
            adjusted_time = raw_time - start_relative

            # Skip notes that are before the clip's start offset
            if adjusted_time < 0:
                continue

            duration_val = note_event.get("Duration", "0")
            velocity_val = note_event.get("Velocity", "100")

            base_notes.append(
                MidiNote(
                    time=adjusted_time,
                    duration=float(duration_val),
                    velocity=int(float(velocity_val)),
                    pitch=pitch,
                )
            )

    # Strategy 2: Fallback - find any MidiNoteEvent not already captured
    if not base_notes:
        for note_event in clip_element.findall(".//MidiNoteEvent"):
            if note_event.get("IsEnabled") == "false":
                continue

            raw_time = float(note_event.get("Time", "0"))
            adjusted_time = raw_time - start_relative

            # Skip notes that are before the clip's start offset
            if adjusted_time < 0:
                continue

            duration_val = note_event.get("Duration", "0")
            velocity_val = note_event.get("Velocity", "100")

            base_notes.append(
                MidiNote(
                    time=adjusted_time,
                    duration=float(duration_val),
                    velocity=int(float(velocity_val)),
                    pitch=0,  # Unknown pitch in fallback
                )
            )

    return base_notes


def _expand_looped_notes(
    base_notes: list[MidiNote],
    loop_info: _LoopInfo,
    clip_length: float,
) -> list[MidiNote]:
    """Expand notes based on loop settings."""
    expanded_notes: list[MidiNote] = []

    # Only notes within the loop region (time >= 0 and time < loop_length) repeat
    # Note: time is already adjusted by start_relative, so 0 is the start of the loop
    loopable_notes = [n for n in base_notes if 0 <= n.time < loop_info.length]

    # Notes outside the loop region play once at their original time
    non_loopable_notes = [n for n in base_notes if n.time >= loop_info.length]

    # Repeat loopable notes to fill the clip duration
    repetition = 0
    while repetition * loop_info.length < clip_length:
        offset = repetition * loop_info.length
        for note in loopable_notes:
            new_time = note.time + offset
            if new_time < clip_length:
                expanded_notes.append(
                    MidiNote(
                        time=new_time,
                        duration=note.duration,
                        velocity=note.velocity,
                        pitch=note.pitch,
                    )
                )
        repetition += 1

    # Add non-loopable notes (they play once at their original position)
    expanded_notes.extend(non_loopable_notes)
    return expanded_notes


def extract_midi_notes(clip_element: Element) -> tuple[MidiNote, ...]:
    """Extract all MIDI notes from a clip element.

    Handles both Drum Rack (KeyTracks) and standard MIDI track structures.
    Note times are adjusted by StartRelative offset and are relative to clip start.

    The StartRelative value in the Loop element indicates the offset into the
    clip's internal content where playback begins. Notes before this offset
    are not played, and all note times must be adjusted by subtracting this value.

    When LoopOn is true and the clip length exceeds the loop region, notes within
    the loop region are repeated to fill the clip duration.

    Args:
        clip_element: A MidiClip XML element.

    Returns:
        Tuple of MidiNote objects sorted by time.
    """
    # 1. Extract loop settings
    loop_info = _extract_loop_info(clip_element)

    # 2. Get clip length
    clip_time = float(clip_element.get("Time", "0"))
    current_end_elem = clip_element.find("CurrentEnd")
    clip_end = (
        float(current_end_elem.get("Value", str(clip_time)))
        if current_end_elem is not None
        else clip_time
    )
    clip_length = clip_end - clip_time

    # 3. Extract raw notes (adjusted by start_relative)
    base_notes = _extract_raw_notes(clip_element, loop_info.start_relative)

    # 4. Expand if looping is enabled and relevant
    if loop_info.is_on and loop_info.length > 0 and clip_length > loop_info.length:
        expanded_notes = _expand_looped_notes(base_notes, loop_info, clip_length)
        return tuple(sorted(expanded_notes, key=lambda n: n.time))

    return tuple(sorted(base_notes, key=lambda n: n.time))


def extract_midi_clip_contents(
    track_element: Element,
    clips: tuple[Clip, ...],
) -> list[MidiClipContent]:
    """Extract MIDI clip contents with notes from a track element.

    Matches clip elements in the XML to the provided Clip objects by
    start time, then extracts MIDI notes from each.

    Args:
        track_element: A MidiTrack XML element.
        clips: Tuple of Clip objects from parser.

    Returns:
        List of MidiClipContent objects with notes.
    """
    # Build a map of clip start times to Clip objects
    clip_map: dict[float, Clip] = {clip.start_beats: clip for clip in clips}

    contents: list[MidiClipContent] = []

    # Path to arrangement clips
    events_path = "DeviceChain/MainSequencer/ClipTimeable/ArrangerAutomation/Events"
    events_elem = track_element.find(events_path)

    if events_elem is None:
        return contents

    for clip_elem in events_elem.findall("MidiClip"):
        time_str = clip_elem.get("Time")
        if time_str is None:
            continue

        try:
            start_beats = float(time_str)
        except ValueError:
            continue

        # Find the corresponding Clip object
        clip = clip_map.get(start_beats)
        if clip is None:
            continue

        notes = extract_midi_notes(clip_elem)
        contents.append(MidiClipContent(clip=clip, notes=notes))

    return contents


def check_activity_in_range(
    clip_contents: list[MidiClipContent],
    start_beats: float,
    end_beats: float,
) -> bool:
    """Check if any MIDI notes are active in the given beat range.

    Examines all clips in the provided list and checks if any note
    overlaps with the specified range. This is the building block
    for phrase-aligned event detection.

    Args:
        clip_contents: List of MidiClipContent for a track.
        start_beats: Start of the range to check (absolute beats).
        end_beats: End of the range to check (absolute beats).

    Returns:
        True if any note is active in the range, False otherwise.
    """
    for content in clip_contents:
        # Check if clip overlaps with the range
        if content.clip.start_beats < end_beats and content.clip.end_beats > start_beats:
            # Convert to clip-relative coordinates
            relative_start = max(0.0, start_beats - content.clip.start_beats)
            relative_end = min(
                content.clip.end_beats - content.clip.start_beats,
                end_beats - content.clip.start_beats,
            )
            if content.has_notes_in_range(relative_start, relative_end):
                return True
    return False


def detect_midi_activity(
    clip_contents: list[MidiClipContent],
    resolution_beats: float = 8.0,
) -> list[tuple[float, bool]]:
    """Detect track activity using range queries.

    IMPORTANT: Uses range queries to avoid the "stroboscope" problem.
    Point sampling at beat 0, 8, 16... would miss activity between
    sample points (e.g., a 1-bar drum fill or off-beat bass line).

    Args:
        clip_contents: List of MidiClipContent for a track.
        resolution_beats: Size of each detection window in beats.

    Returns:
        List of (window_start_beat, is_active) tuples.
    """
    if not clip_contents:
        return []

    start = min(c.clip.start_beats for c in clip_contents)
    end = max(c.clip.end_beats for c in clip_contents)

    activity: list[tuple[float, bool]] = []
    beat = start

    while beat < end:
        window_start = beat
        window_end = beat + resolution_beats

        # Check if ANY note activity occurs within this window
        is_active = False
        for content in clip_contents:
            # Check if window overlaps with clip
            if content.clip.start_beats < window_end and content.clip.end_beats > window_start:
                # Convert to clip-relative coordinates
                relative_start = max(0.0, window_start - content.clip.start_beats)
                relative_end = min(
                    content.clip.end_beats - content.clip.start_beats,
                    window_end - content.clip.start_beats,
                )
                if content.has_notes_in_range(relative_start, relative_end):
                    is_active = True
                    break

        activity.append((beat, is_active))
        beat += resolution_beats

    return activity
