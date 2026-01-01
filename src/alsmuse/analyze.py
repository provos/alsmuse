"""Application layer for ALSmuse.

This module provides the main analysis pipeline that orchestrates
parsing, extraction, and formatting of Ableton Live Set files.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import click

from .events import (
    detect_events_from_clip_contents_phrase_aligned,
    merge_events_into_phrases,
)
from .exceptions import AlignmentError
from .extractors import StructureTrackExtractor, fill_gaps
from .formatter import format_av_table, format_phrase_table
from .lyrics import distribute_lyrics, distribute_timed_lyrics, parse_lyrics_file
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

logger = logging.getLogger(__name__)


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
    align_vocals: bool = False,
    vocal_tracks: tuple[str, ...] | None = None,
    use_all_vocals: bool = False,
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
    6. Optionally parse and distribute lyrics (with optional forced alignment)
    7. Format output as markdown phrase table

    Args:
        als_path: Path to the .als file
        structure_track: Name of the structure track (case-insensitive)
        beats_per_phrase: Number of beats per phrase (default 8 = 2 bars in 4/4)
        show_events: Whether to detect and show track events (default True)
        lyrics_path: Optional path to lyrics file with [SECTION] headers
        align_vocals: If True, use forced alignment for precise lyrics timing
        vocal_tracks: Specific vocal track names to use for alignment
        use_all_vocals: If True, use all detected vocal tracks without prompting

    Returns:
        Markdown formatted phrase table string

    Raises:
        ParseError: If the file cannot be parsed
        TrackNotFoundError: If the structure track is not found
    """
    live_set = parse_als_file(als_path)
    bpm = live_set.tempo.bpm

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
        if align_vocals:
            try:
                phrases = align_and_distribute_lyrics(
                    als_path=als_path,
                    lyrics_path=lyrics_path,
                    phrases=phrases,
                    bpm=bpm,
                    vocal_tracks=vocal_tracks,
                    use_all_vocals=use_all_vocals,
                )
                show_lyrics = True
            except AlignmentError as e:
                # Log warning and fall back to heuristic distribution
                click.echo(
                    f"Alignment failed: {e}. Falling back to heuristic distribution.",
                    err=True,
                )
                section_lyrics = parse_lyrics_file(lyrics_path)
                phrases = distribute_lyrics(phrases, section_lyrics)
                show_lyrics = True
        else:
            # Existing heuristic distribution
            section_lyrics = parse_lyrics_file(lyrics_path)
            phrases = distribute_lyrics(phrases, section_lyrics)
            show_lyrics = True

    return format_phrase_table(
        phrases, bpm, show_events=show_events, show_lyrics=show_lyrics
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


def align_and_distribute_lyrics(
    als_path: Path,
    lyrics_path: Path,
    phrases: list[Phrase],
    bpm: float,
    vocal_tracks: tuple[str, ...] | None = None,
    use_all_vocals: bool = False,
) -> list[Phrase]:
    """Full alignment pipeline: extract audio, align lyrics, distribute to phrases.

    This orchestration function combines audio extraction, forced alignment,
    and lyrics distribution into a single workflow.

    Steps:
    1. Extract audio clips from ALS file
    2. Select vocal tracks (using explicit selection, auto-detection, or prompting)
    3. Combine vocal clips to a temporary audio file
    4. Run forced alignment with stable-ts
    5. Distribute timed lyrics to phrases
    6. Clean up temporary file

    Args:
        als_path: Path to the .als file
        lyrics_path: Path to the lyrics file with [SECTION] headers
        phrases: List of phrases to annotate with lyrics
        bpm: Tempo in beats per minute
        vocal_tracks: Specific vocal track names to use (None for auto-detect)
        use_all_vocals: If True, use all detected vocal tracks without prompting

    Returns:
        Phrases with lyric fields populated from forced alignment.

    Raises:
        AlignmentError: If alignment fails for any reason (no vocal tracks,
            audio extraction failure, alignment model failure, etc.)
    """
    from .audio import (
        combine_clips_to_audio,
        extract_audio_clips,
        select_vocal_tracks,
    )
    from .lyrics_align import align_lyrics, words_to_lines

    # Step 1: Extract all audio clips from ALS
    try:
        all_clips = extract_audio_clips(als_path, bpm)
    except Exception as e:
        raise AlignmentError(f"Failed to extract audio clips: {e}") from e

    if not all_clips:
        raise AlignmentError("No audio clips found in ALS file")

    # Step 2: Select vocal tracks
    selected_clips = select_vocal_tracks(
        all_clips,
        explicit_tracks=vocal_tracks,
        use_all=use_all_vocals,
    )

    if not selected_clips:
        raise AlignmentError(
            "No vocal tracks found. Use --vocal-track to specify tracks explicitly."
        )

    # Step 3: Combine vocals into single audio file
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="alsmuse_vocals_"
        ) as tmp:
            combined_path = Path(tmp.name)
    except OSError as e:
        raise AlignmentError(f"Failed to create temporary file: {e}") from e

    try:
        try:
            _, valid_ranges = combine_clips_to_audio(selected_clips, combined_path)
        except Exception as e:
            raise AlignmentError(f"Failed to combine audio clips: {e}") from e

        # Step 4: Read lyrics as plain text and run alignment
        all_lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()

        if not all_lyrics_text:
            raise AlignmentError("Lyrics file is empty or contains no text")

        # Run forced alignment
        timed_words = align_lyrics(
            combined_path,
            all_lyrics_text,
            valid_ranges=valid_ranges,
        )

        # Step 5: Reconstruct lines and distribute to phrases
        original_lines = [
            line.strip()
            for line in all_lyrics_text.split("\n")
            if line.strip()
        ]
        timed_lines = words_to_lines(timed_words, original_lines)

        return distribute_timed_lyrics(phrases, timed_lines, bpm)

    finally:
        # Cleanup temp file
        if combined_path.exists():
            try:
                combined_path.unlink()
            except OSError:
                logger.warning("Failed to clean up temporary file: %s", combined_path)
