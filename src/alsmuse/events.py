"""Track event detection for ALSmuse.

This module provides track categorization based on name heuristics and
event detection from MIDI activity patterns.
"""

from .models import MidiClipContent, Phrase, TrackEvent

# Track category keywords for classification
TRACK_CATEGORIES: dict[str, list[str]] = {
    "drums": ["kick", "snare", "drum", "hat", "cymbal", "perc", "tom"],
    "bass": ["bass", "sub"],
    "vocals": ["vocal", "vox", "verse", "chorus", "main", "double", "harmony"],
    "lead": ["lead", "solo", "melody"],
    "guitar": ["guitar", "gtr"],
    "keys": ["piano", "keys", "organ", "synth"],
    "pad": ["pad", "strings", "atmosphere"],
    "fx": ["fx", "riser", "downlifter", "reverse", "sweep", "impact"],
}


def categorize_track(track_name: str) -> str:
    """Determine track category from name using keyword matching.

    Uses heuristic keyword matching against the track name to classify
    it into a category like "drums", "bass", "vocals", etc.

    Args:
        track_name: Name of the track to categorize.

    Returns:
        Category string, or "other" if no keywords match.

    Examples:
        >>> categorize_track("Drum Kit")
        'drums'
        >>> categorize_track("Bass Line")
        'bass'
        >>> categorize_track("Track 7")
        'other'
    """
    name_lower = track_name.lower()
    for category, keywords in TRACK_CATEGORIES.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return "other"


def detect_track_events(
    track_name: str,
    activity: list[tuple[float, bool]],
    category: str,
) -> list[TrackEvent]:
    """Convert activity samples to enter/exit events.

    Analyzes the activity pattern to detect when a track starts playing
    (enter) or stops playing (exit).

    Args:
        track_name: Name of the track.
        activity: List of (beat, is_active) tuples from activity detection.
        category: The track category (e.g., "drums", "bass").

    Returns:
        List of TrackEvent objects for state changes.
    """
    events: list[TrackEvent] = []
    was_active = False

    for beat, is_active in activity:
        if is_active and not was_active:
            events.append(
                TrackEvent(
                    beat=beat,
                    track_name=track_name,
                    event_type="enter",
                    category=category,
                )
            )
        elif not is_active and was_active:
            events.append(
                TrackEvent(
                    beat=beat,
                    track_name=track_name,
                    event_type="exit",
                    category=category,
                )
            )
        was_active = is_active

    return events


def detect_events_from_clip_contents(
    track_name: str,
    clip_contents: list[MidiClipContent],
    resolution_beats: float = 8.0,
) -> list[TrackEvent]:
    """Detect track events from MIDI clip contents.

    Combines activity detection with event generation for a track.

    Args:
        track_name: Name of the track.
        clip_contents: List of MidiClipContent for the track.
        resolution_beats: Size of each detection window in beats.

    Returns:
        List of TrackEvent objects for state changes.
    """
    from .midi import detect_midi_activity

    if not clip_contents:
        return []

    category = categorize_track(track_name)
    activity = detect_midi_activity(clip_contents, resolution_beats)
    return detect_track_events(track_name, activity, category)


def merge_events_into_phrases(
    phrases: list[Phrase],
    events: list[TrackEvent],
) -> list[Phrase]:
    """Attach events to the phrases they occur in.

    Events are grouped by category for cleaner output:
    "Drums enters, Bass enters" instead of "KICK enters, SNARE enters, Bass enters".

    Args:
        phrases: List of Phrase objects to attach events to.
        events: List of TrackEvent objects to distribute.

    Returns:
        New list of Phrase objects with events attached.
    """
    if not phrases:
        return phrases

    # Sort events by beat
    sorted_events = sorted(events, key=lambda e: e.beat)

    result: list[Phrase] = []
    event_idx = 0

    for phrase in phrases:
        phrase_events: list[TrackEvent] = []

        # Collect events that fall within this phrase
        while event_idx < len(sorted_events) and sorted_events[event_idx].beat < phrase.end_beats:
            if sorted_events[event_idx].beat >= phrase.start_beats:
                phrase_events.append(sorted_events[event_idx])
            event_idx += 1

        # Deduplicate by category and event type
        seen_categories: dict[tuple[str, str], TrackEvent] = {}
        for event in phrase_events:
            key = (event.category, event.event_type)
            if key not in seen_categories:
                seen_categories[key] = event

        result.append(
            Phrase(
                start_beats=phrase.start_beats,
                end_beats=phrase.end_beats,
                section_name=phrase.section_name,
                is_section_start=phrase.is_section_start,
                events=tuple(seen_categories.values()),
                lyric=phrase.lyric,
            )
        )

    return result
