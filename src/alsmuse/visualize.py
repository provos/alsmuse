"""Visualization pipeline orchestration for ALSmuse.

This module provides the main visualization pipeline that orchestrates
parsing, event detection, lyrics processing, and video generation.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import click

from .audio import (
    combine_clips_to_audio,
    extract_audio_clips,
    select_vocal_tracks_with_config,
)
from .category_review import prompt_category_review
from .config import MuseConfig, load_config, save_config
from .events import (
    categorize_all_tracks,
    categorize_track,
    detect_events_from_clip_contents_phrase_aligned,
    detect_fill_events,
    get_available_categories,
    merge_events_into_phrases,
)
from .exceptions import AlignmentError
from .extractors import StructureTrackExtractor, fill_gaps
from .lyrics import (
    distribute_lyrics,
    distribute_timed_lyrics,
    parse_lyrics_file,
    parse_lyrics_file_auto,
)
from .lyrics_align import (
    align_lyrics,
    segments_to_lines,
    transcribe_lyrics,
    validate_timed_lines,
    words_to_lines,
)
from .midi import extract_midi_clip_contents
from .models import LiveSet, MidiClipContent, Phrase, Section, TrackEvent
from .parser import (
    extract_track_clips,
    extract_track_name,
    get_track_elements,
    parse_als_file,
    parse_als_xml,
)
from .phrases import subdivide_sections
from .visualizer import generate_visualizer

logger = logging.getLogger(__name__)


def _get_all_track_names(live_set: LiveSet) -> list[str]:
    """Get all effectively enabled track names from a LiveSet.

    Returns only tracks that are enabled and not in disabled groups.

    Args:
        live_set: Pre-parsed LiveSet containing track/group enabled state.

    Returns:
        List of enabled track names (both MIDI and audio tracks).
    """
    return [t.name for t in live_set.tracks if live_set.is_track_effectively_enabled(t)]


def _detect_track_events_from_als_phrase_aligned(
    als_path: Path,
    phrases: list[Phrase],
    category_overrides: dict[str, str] | None = None,
    sections: list[Section] | None = None,
) -> list[TrackEvent]:
    """Detect track events using phrase-aligned boundaries.

    Analyzes all MIDI tracks in the ALS file and generates enter/exit
    events based on activity within each phrase's time boundaries.

    Args:
        als_path: Path to the .als file
        phrases: List of Phrase objects defining the time boundaries.
        category_overrides: Optional mapping of track names to categories.
        sections: Optional list of Section objects for drum fill detection.

    Returns:
        List of TrackEvent objects for all tracks.
    """
    live_set = parse_als_file(als_path)

    enabled_track_names = {
        t.name for t in live_set.tracks if live_set.is_track_effectively_enabled(t)
    }

    root = parse_als_xml(als_path)
    track_elements = get_track_elements(root)

    all_events: list[TrackEvent] = []
    drum_clip_contents: list[MidiClipContent] = []

    for track_elem, track_type in track_elements:
        if track_type != "midi":
            continue

        track_name = extract_track_name(track_elem)

        if track_name not in enabled_track_names:
            continue

        clips = extract_track_clips(track_elem, track_type)

        if not clips:
            continue

        clip_contents = extract_midi_clip_contents(track_elem, clips)

        if not clip_contents:
            continue

        # Determine track category
        if category_overrides and track_name in category_overrides:
            category = category_overrides[track_name]
        else:
            category = categorize_track(track_name)

        if category == "drums":
            drum_clip_contents.extend(clip_contents)

        events = detect_events_from_clip_contents_phrase_aligned(
            track_name, clip_contents, phrases, category_overrides
        )
        all_events.extend(events)

    if sections and drum_clip_contents:
        fill_events = detect_fill_events(drum_clip_contents, sections)
        all_events.extend(fill_events)

    return all_events


def _align_and_distribute_lyrics(
    als_path: Path,
    lyrics_path: Path,
    phrases: list[Phrase],
    bpm: float,
    vocal_tracks: tuple[str, ...] | None = None,
    use_all_vocals: bool = False,
    language: str = "en",
    model_size: str = "base",
) -> list[Phrase]:
    """Alignment pipeline: extract audio, align lyrics, distribute to phrases."""
    try:
        all_clips = extract_audio_clips(als_path, bpm)
    except Exception as e:
        raise AlignmentError(f"Failed to extract audio clips: {e}") from e

    if not all_clips:
        raise AlignmentError("No audio clips found in ALS file")

    selected_clips, _ = select_vocal_tracks_with_config(
        all_clips,
        als_path,
        explicit_tracks=vocal_tracks,
        use_all=use_all_vocals,
    )

    if not selected_clips:
        raise AlignmentError(
            "No vocal tracks found. Use --vocal-track to specify tracks explicitly."
        )

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="alsmuse_vocals_"
        ) as tmp:
            combined_path = Path(tmp.name)
    except OSError as e:
        raise AlignmentError(f"Failed to create temporary file: {e}") from e

    try:
        _, valid_ranges = combine_clips_to_audio(selected_clips, combined_path, bpm)

        all_lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()

        if not all_lyrics_text:
            raise AlignmentError("Lyrics file is empty or contains no text")

        timed_words = align_lyrics(
            combined_path,
            all_lyrics_text,
            valid_ranges=valid_ranges,
            language=language,
            model_size=model_size,
        )

        original_lines = [line.strip() for line in all_lyrics_text.split("\n") if line.strip()]
        timed_lines = words_to_lines(timed_words, original_lines)

        alignment_warnings = validate_timed_lines(timed_lines)
        for warning in alignment_warnings:
            click.echo(warning, err=True)

        return distribute_timed_lyrics(phrases, timed_lines, bpm)

    finally:
        if combined_path.exists():
            try:
                combined_path.unlink()
            except OSError:
                logger.warning("Failed to clean up temporary file: %s", combined_path)


def _transcribe_and_distribute_lyrics(
    als_path: Path,
    phrases: list[Phrase],
    bpm: float,
    vocal_tracks: tuple[str, ...] | None = None,
    use_all_vocals: bool = False,
    language: str = "en",
    model_size: str = "base",
) -> list[Phrase]:
    """Transcription pipeline: extract audio, transcribe, distribute."""
    try:
        all_clips = extract_audio_clips(als_path, bpm)
    except Exception as e:
        raise AlignmentError(f"Failed to extract audio clips: {e}") from e

    if not all_clips:
        raise AlignmentError("No audio clips found in ALS file")

    selected_clips, _ = select_vocal_tracks_with_config(
        all_clips,
        als_path,
        explicit_tracks=vocal_tracks,
        use_all=use_all_vocals,
    )

    if not selected_clips:
        raise AlignmentError(
            "No vocal tracks found. Use --vocal-track to specify tracks explicitly."
        )

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="alsmuse_vocals_"
        ) as tmp:
            combined_path = Path(tmp.name)
    except OSError as e:
        raise AlignmentError(f"Failed to create temporary file: {e}") from e

    try:
        _, valid_ranges = combine_clips_to_audio(selected_clips, combined_path, bpm)

        segments, raw_text = transcribe_lyrics(
            combined_path,
            language=language,
            model_size=model_size,
        )

        timed_lines = segments_to_lines(segments)

        transcription_warnings = validate_timed_lines(timed_lines)
        for warning in transcription_warnings:
            click.echo(warning, err=True)

        return distribute_timed_lyrics(phrases, timed_lines, bpm)

    finally:
        if combined_path.exists():
            try:
                combined_path.unlink()
            except OSError:
                logger.warning("Failed to clean up temporary file: %s", combined_path)


def visualize_als(
    als_path: Path,
    output_path: Path,
    audio_path: Path | None = None,
    structure_track: str = "STRUCTURE",
    beats_per_phrase: int = 8,
    lyrics_path: Path | None = None,
    vocal_tracks: tuple[str, ...] | None = None,
    use_all_vocals: bool = False,
    transcribe: bool = False,
    language: str = "en",
    model_size: str = "base",
    interactive: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Generate a lyrics/cues visualization video from an ALS file.

    This orchestration function combines:
    1. Parse ALS file
    2. Load config if exists (vocal tracks, category overrides)
    3. Extract sections using StructureTrackExtractor
    4. Fill gaps with transitions
    5. Subdivide sections into phrases
    6. Optionally run interactive category review
    7. Detect MIDI events and merge into phrases
    8. Optionally transcribe or align lyrics
    9. Generate video visualization

    Args:
        als_path: Path to the .als file
        output_path: Path to write the output MP4 file
        audio_path: Optional audio file to mux into the video
        structure_track: Name of the structure track (case-insensitive)
        beats_per_phrase: Number of beats per phrase (default 8 = 2 bars in 4/4)
        lyrics_path: Optional path to lyrics file
        vocal_tracks: Specific vocal track names to use for alignment
        use_all_vocals: If True, use all detected vocal tracks without prompting
        transcribe: If True, transcribe lyrics from vocal audio using ASR
        language: Language code for transcription/alignment (default: "en")
        model_size: Whisper model size (default: "base")
        interactive: If True and TTY available, prompt for category review
        progress_callback: Optional callback for progress updates

    Returns:
        Path to the generated video file

    Raises:
        ParseError: If the file cannot be parsed
        TrackNotFoundError: If the structure track is not found
        RuntimeError: If video generation fails
    """
    live_set = parse_als_file(als_path)
    bpm = live_set.tempo.bpm

    # Load existing config
    config = load_config(als_path)
    category_overrides = config.category_overrides if config else {}

    extractor = StructureTrackExtractor(structure_track)
    sections = extractor.extract(live_set)
    sections = fill_gaps(sections)

    phrases = subdivide_sections(sections, beats_per_phrase)

    # Calculate total beats from the last phrase
    total_beats = phrases[-1].end_beats if phrases else 0.0

    # Interactive category review (if TTY available and enabled)
    if interactive and sys.stdin.isatty():
        track_names = _get_all_track_names(live_set)
        current_categories = categorize_all_tracks(track_names, category_overrides)
        available_categories = get_available_categories()

        new_overrides = prompt_category_review(
            track_names, current_categories, available_categories
        )

        if new_overrides:
            category_overrides = {**category_overrides, **new_overrides}

            updated_config = MuseConfig(
                vocal_tracks=config.vocal_tracks if config else [],
                category_overrides=category_overrides,
            )
            save_config(als_path, updated_config)

    # Detect events and merge into phrases
    events = _detect_track_events_from_als_phrase_aligned(
        als_path, phrases, category_overrides, sections
    )
    phrases = merge_events_into_phrases(phrases, events)

    # Handle transcription mode (ASR from vocal audio)
    if transcribe:
        try:
            phrases = _transcribe_and_distribute_lyrics(
                als_path=als_path,
                phrases=phrases,
                bpm=bpm,
                vocal_tracks=vocal_tracks,
                use_all_vocals=use_all_vocals,
                language=language,
                model_size=model_size,
            )
        except AlignmentError as e:
            click.echo(f"Transcription failed: {e}", err=True)

    # Parse and distribute lyrics if provided (and not in transcribe mode)
    elif lyrics_path is not None:
        timed_lines, section_lyrics = parse_lyrics_file_auto(lyrics_path)

        if timed_lines is not None:
            phrases = distribute_timed_lyrics(phrases, timed_lines, bpm)
        else:
            # Plain lyrics - try alignment
            try:
                phrases = _align_and_distribute_lyrics(
                    als_path=als_path,
                    lyrics_path=lyrics_path,
                    phrases=phrases,
                    bpm=bpm,
                    vocal_tracks=vocal_tracks,
                    use_all_vocals=use_all_vocals,
                    language=language,
                    model_size=model_size,
                )
            except AlignmentError as e:
                click.echo(
                    f"Alignment failed: {e}. Falling back to heuristic distribution.",
                    err=True,
                )
                if section_lyrics is None:
                    section_lyrics = parse_lyrics_file(lyrics_path)
                phrases = distribute_lyrics(phrases, section_lyrics)

    # Generate the video
    return generate_visualizer(
        phrases=phrases,
        bpm=bpm,
        total_beats=total_beats,
        output_path=output_path,
        audio_path=audio_path,
        progress_callback=progress_callback,
    )
