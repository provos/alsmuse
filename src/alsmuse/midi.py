"""MIDI analysis for ALSmuse.

This module provides functions to extract MIDI notes from ALS XML elements
and detect track activity using range-based queries.
"""

from xml.etree.ElementTree import Element

from .models import Clip, MidiClipContent, MidiNote


def extract_midi_notes(clip_element: Element) -> tuple[MidiNote, ...]:
    """Extract all MIDI notes from a clip element.

    Handles both Drum Rack (KeyTracks) and standard MIDI track structures.
    Note times are relative to clip start.

    Args:
        clip_element: A MidiClip XML element.

    Returns:
        Tuple of MidiNote objects sorted by time.
    """
    notes: list[MidiNote] = []

    # Strategy 1: KeyTracks structure (Drum Racks, most MIDI clips)
    for key_track in clip_element.findall(".//KeyTrack"):
        midi_key_elem = key_track.find("MidiKey")
        pitch = int(midi_key_elem.get("Value", "0")) if midi_key_elem is not None else 0

        for note_event in key_track.findall(".//MidiNoteEvent"):
            if note_event.get("IsEnabled") == "false":
                continue

            time_val = note_event.get("Time", "0")
            duration_val = note_event.get("Duration", "0")
            velocity_val = note_event.get("Velocity", "100")

            notes.append(
                MidiNote(
                    time=float(time_val),
                    duration=float(duration_val),
                    velocity=int(float(velocity_val)),
                    pitch=pitch,
                )
            )

    # Strategy 2: Fallback - find any MidiNoteEvent not already captured
    # This handles edge cases in XML structure variations
    if not notes:
        for note_event in clip_element.findall(".//MidiNoteEvent"):
            if note_event.get("IsEnabled") == "false":
                continue

            time_val = note_event.get("Time", "0")
            duration_val = note_event.get("Duration", "0")
            velocity_val = note_event.get("Velocity", "100")

            notes.append(
                MidiNote(
                    time=float(time_val),
                    duration=float(duration_val),
                    velocity=int(float(velocity_val)),
                    pitch=0,  # Unknown pitch in fallback
                )
            )

    return tuple(sorted(notes, key=lambda n: n.time))


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
