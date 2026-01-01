"""Application layer for ALSmuse.

This module provides the main analysis pipeline that orchestrates
parsing, extraction, and formatting of Ableton Live Set files.
"""

from pathlib import Path

from .events import (
    detect_events_from_clip_contents_phrase_aligned,
    merge_events_into_phrases,
)
from .extractors import StructureTrackExtractor, fill_gaps
from .formatter import format_av_table, format_phrase_table
from .lyrics import distribute_lyrics, parse_lyrics_file
from .midi import extract_midi_clip_contents
from .models import MidiClipContent, Phrase, TrackEvent
from .parser import (
    extract_track_clips,
    extract_track_name,
    get_track_elements,
    parse_als_file,
    parse_als_xml,
)
from .phrases import subdivide_sections


def analyze_als(
    als_path: Path,
    structure_track: str = "STRUCTURE",
) -> str:
    """Main analysis pipeline.

    1. Parse ALS file
    2. Extract sections using StructureTrackExtractor
    3. Fill gaps with transitions
    4. Format output as markdown A/V table

    Args:
        als_path: Path to the .als file
        structure_track: Name of the structure track (case-insensitive)

    Returns:
        Markdown formatted A/V table string

    Raises:
        ParseError: If the file cannot be parsed
        TrackNotFoundError: If the structure track is not found
    """
    live_set = parse_als_file(als_path)

    extractor = StructureTrackExtractor(structure_track)
    sections = extractor.extract(live_set)
    sections = fill_gaps(sections)

    return format_av_table(sections, live_set.tempo.bpm)


def analyze_als_v2(
    als_path: Path,
    structure_track: str = "STRUCTURE",
    beats_per_phrase: int = 8,
    show_events: bool = True,
    lyrics_path: Path | None = None,
) -> str:
    """Analysis pipeline with phrase subdivision and event detection.

    Extended analysis pipeline that divides sections into phrase-sized
    chunks for more detailed A/V scripts with per-phrase timing and
    track enter/exit events.

    1. Parse ALS file
    2. Extract sections using StructureTrackExtractor
    3. Fill gaps with transitions
    4. Subdivide sections into phrases
    5. Optionally detect MIDI events and merge into phrases
    6. Optionally parse and distribute lyrics
    7. Format output as markdown phrase table

    Args:
        als_path: Path to the .als file
        structure_track: Name of the structure track (case-insensitive)
        beats_per_phrase: Number of beats per phrase (default 8 = 2 bars in 4/4)
        show_events: Whether to detect and show track events (default True)
        lyrics_path: Optional path to lyrics file with [SECTION] headers

    Returns:
        Markdown formatted phrase table string

    Raises:
        ParseError: If the file cannot be parsed
        TrackNotFoundError: If the structure track is not found
    """
    live_set = parse_als_file(als_path)

    extractor = StructureTrackExtractor(structure_track)
    sections = extractor.extract(live_set)
    sections = fill_gaps(sections)

    phrases = subdivide_sections(sections, beats_per_phrase)

    if show_events:
        events = detect_track_events_from_als_phrase_aligned(als_path, phrases)
        phrases = merge_events_into_phrases(phrases, events)

    # Parse and distribute lyrics if provided
    show_lyrics = False
    if lyrics_path is not None:
        section_lyrics = parse_lyrics_file(lyrics_path)
        phrases = distribute_lyrics(phrases, section_lyrics)
        show_lyrics = True

    return format_phrase_table(
        phrases, live_set.tempo.bpm, show_events=show_events, show_lyrics=show_lyrics
    )


def detect_track_events_from_als_phrase_aligned(
    als_path: Path,
    phrases: list[Phrase],
) -> list[TrackEvent]:
    """Detect track events using phrase-aligned boundaries.

    Analyzes all MIDI tracks in the ALS file and generates enter/exit
    events based on activity within each phrase's time boundaries.
    This approach prevents false events at section boundaries caused
    by the old global grid detection.

    Args:
        als_path: Path to the .als file
        phrases: List of Phrase objects defining the time boundaries.

    Returns:
        List of TrackEvent objects for all tracks.
    """
    root = parse_als_xml(als_path)
    track_elements = get_track_elements(root)

    all_events: list[TrackEvent] = []

    for track_elem, track_type in track_elements:
        if track_type != "midi":
            continue

        track_name = extract_track_name(track_elem)
        clips = extract_track_clips(track_elem, track_type)

        if not clips:
            continue

        clip_contents = extract_midi_clip_contents(track_elem, clips)

        if not clip_contents:
            continue

        events = detect_events_from_clip_contents_phrase_aligned(
            track_name, clip_contents, phrases
        )
        all_events.extend(events)

    return all_events


def extract_track_clip_contents(als_path: Path) -> dict[str, list[MidiClipContent]]:
    """Extract MIDI clip contents for all tracks in an ALS file.

    This is a helper function that can be used for testing or advanced
    analysis workflows.

    Args:
        als_path: Path to the .als file

    Returns:
        Dictionary mapping track names to their MidiClipContent lists.
    """
    root = parse_als_xml(als_path)
    track_elements = get_track_elements(root)

    result: dict[str, list[MidiClipContent]] = {}

    for track_elem, track_type in track_elements:
        if track_type != "midi":
            continue

        track_name = extract_track_name(track_elem)
        clips = extract_track_clips(track_elem, track_type)

        if not clips:
            continue

        clip_contents = extract_midi_clip_contents(track_elem, clips)

        if clip_contents:
            result[track_name] = clip_contents

    return result
