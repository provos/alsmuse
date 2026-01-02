"""Application layer for ALSmuse.

This module provides the main analysis pipeline that orchestrates
parsing, extraction, and formatting of Ableton Live Set files.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import click
from tqdm import tqdm

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
from .formatter import format_phrase_table
from .lyrics import (
    distribute_lyrics,
    distribute_timed_lyrics,
    format_segments_as_lrc,
    format_timed_lines_as_lrc,
    parse_lyrics_file,
    parse_lyrics_file_auto,
)
from .lyrics_align import (
    align_lyrics,
    refine_lines_with_bournemouth,
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

logger = logging.getLogger(__name__)


def analyze_als(
    als_path: Path,
    structure_track: str = "STRUCTURE",
    beats_per_phrase: int = 8,
    show_events: bool = True,
    lyrics_path: Path | None = None,
    align_vocals: bool = False,
    vocal_tracks: tuple[str, ...] | None = None,
    use_all_vocals: bool = False,
    save_vocals_path: Path | None = None,
    transcribe: bool = False,
    save_lyrics_path: Path | None = None,
    language: str = "en",
    model_size: str = "base",
    interactive: bool = True,
) -> str:
    """Analysis pipeline with phrase subdivision and event detection.

    Extended analysis pipeline that divides sections into phrase-sized
    chunks for more detailed A/V scripts with per-phrase timing and
    track enter/exit events.

    1. Parse ALS file
    2. Load config if exists (vocal tracks, category overrides)
    3. Extract sections using StructureTrackExtractor
    4. Fill gaps with transitions
    5. Subdivide sections into phrases
    6. Optionally run interactive category review (if TTY and interactive=True)
    7. Optionally detect MIDI events and merge into phrases
    8. Optionally transcribe lyrics from vocal audio (if transcribe=True)
    9. Optionally parse and distribute lyrics (with optional forced alignment)
    10. Format output as markdown phrase table

    Args:
        als_path: Path to the .als file
        structure_track: Name of the structure track (case-insensitive)
        beats_per_phrase: Number of beats per phrase (default 8 = 2 bars in 4/4)
        show_events: Whether to detect and show track events (default True)
        lyrics_path: Optional path to lyrics file
        align_vocals: If True, use forced alignment for precise lyrics timing
        vocal_tracks: Specific vocal track names to use for alignment
        use_all_vocals: If True, use all detected vocal tracks without prompting
        save_vocals_path: If provided, save combined vocals to this path for validation
        transcribe: If True, transcribe lyrics from vocal audio using ASR
        save_lyrics_path: If provided, save lyrics to this path (LRC format with timestamps)
        language: Language code for transcription/alignment (default: "en")
        model_size: Whisper model size (default: "base")
        interactive: If True and TTY available, prompt for category review

    Returns:
        Markdown formatted phrase table string

    Raises:
        ParseError: If the file cannot be parsed
        TrackNotFoundError: If the structure track is not found
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

    # Interactive category review (if TTY available and enabled)
    if show_events and interactive and sys.stdin.isatty():
        track_names = get_all_track_names(live_set)
        current_categories = categorize_all_tracks(track_names, category_overrides)
        available_categories = get_available_categories()

        new_overrides = prompt_category_review(
            track_names, current_categories, available_categories
        )

        if new_overrides:
            # Merge new overrides with existing
            category_overrides = {**category_overrides, **new_overrides}

            # Save to config
            updated_config = MuseConfig(
                vocal_tracks=config.vocal_tracks if config else [],
                category_overrides=category_overrides,
            )
            save_config(als_path, updated_config)

    if show_events:
        events = detect_track_events_from_als_phrase_aligned(
            als_path, phrases, category_overrides, sections
        )
        phrases = merge_events_into_phrases(phrases, events)

    # Handle transcription mode (ASR from vocal audio)
    show_lyrics = False
    if transcribe:
        try:
            phrases = transcribe_and_distribute_lyrics(
                als_path=als_path,
                phrases=phrases,
                bpm=bpm,
                vocal_tracks=vocal_tracks,
                use_all_vocals=use_all_vocals,
                save_vocals_path=save_vocals_path,
                save_lyrics_path=save_lyrics_path,
                language=language,
                model_size=model_size,
            )
            show_lyrics = True
        except AlignmentError as e:
            # Log error and continue without lyrics
            click.echo(f"Transcription failed: {e}", err=True)

    # Parse and distribute lyrics if provided (and not in transcribe mode)
    elif lyrics_path is not None:
        # Try to parse as timed lyrics first (auto-detection)
        timed_lines, section_lyrics = parse_lyrics_file_auto(lyrics_path)

        if timed_lines is not None:
            # Lyrics have timestamps - use directly, no alignment needed
            phrases = distribute_timed_lyrics(phrases, timed_lines, bpm)
            show_lyrics = True
        elif align_vocals:
            # Plain lyrics with alignment requested
            try:
                phrases = align_and_distribute_lyrics(
                    als_path=als_path,
                    lyrics_path=lyrics_path,
                    phrases=phrases,
                    bpm=bpm,
                    vocal_tracks=vocal_tracks,
                    use_all_vocals=use_all_vocals,
                    save_vocals_path=save_vocals_path,
                    save_lyrics_path=save_lyrics_path,
                    language=language,
                    model_size=model_size,
                )
                show_lyrics = True
            except AlignmentError as e:
                # Log warning and fall back to heuristic distribution
                click.echo(
                    f"Alignment failed: {e}. Falling back to heuristic distribution.",
                    err=True,
                )
                if section_lyrics is None:
                    section_lyrics = parse_lyrics_file(lyrics_path)
                phrases = distribute_lyrics(phrases, section_lyrics)
                show_lyrics = True
        else:
            # Plain lyrics, heuristic distribution
            if section_lyrics is None:
                section_lyrics = parse_lyrics_file(lyrics_path)
            phrases = distribute_lyrics(phrases, section_lyrics)
            show_lyrics = True

    return format_phrase_table(phrases, bpm, show_events=show_events, show_lyrics=show_lyrics)


def detect_track_events_from_als_phrase_aligned(
    als_path: Path,
    phrases: list[Phrase],
    category_overrides: dict[str, str] | None = None,
    sections: list[Section] | None = None,
) -> list[TrackEvent]:
    """Detect track events using phrase-aligned boundaries.

    Analyzes all MIDI tracks in the ALS file and generates enter/exit
    events based on activity within each phrase's time boundaries.
    This approach prevents false events at section boundaries caused
    by the old global grid detection.

    Also detects drum fills when sections are provided.

    Tracks that are disabled (muted) or belong to disabled groups are
    skipped and will not generate any events.

    Args:
        als_path: Path to the .als file
        phrases: List of Phrase objects defining the time boundaries.
        category_overrides: Optional mapping of track names to categories.
            If provided, these override auto-categorization.
        sections: Optional list of Section objects for drum fill detection.
            If provided, drum fills will be detected before section changes.

    Returns:
        List of TrackEvent objects for all tracks.
    """
    # Parse the LiveSet to get track enabled state and group information
    live_set = parse_als_file(als_path)

    # Build lookup of effectively enabled tracks by name
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

        # Skip disabled tracks or tracks in disabled groups
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

        # Collect drum track clip contents for fill detection
        if category == "drums":
            drum_clip_contents.extend(clip_contents)

        events = detect_events_from_clip_contents_phrase_aligned(
            track_name, clip_contents, phrases, category_overrides
        )
        all_events.extend(events)

    # Detect drum fills if sections are provided
    if sections and drum_clip_contents:
        fill_events = detect_fill_events(drum_clip_contents, sections)
        all_events.extend(fill_events)

    return all_events


def extract_track_clip_contents(
    live_set: LiveSet,
    als_path: Path,
) -> dict[str, list[MidiClipContent]]:
    """Extract MIDI clip contents for all enabled tracks in an ALS file.

    This is a debugging/testing helper function for advanced analysis workflows.

    Tracks that are disabled (muted) or belong to disabled groups are
    skipped.

    Args:
        live_set: Pre-parsed LiveSet containing track/group enabled state.
        als_path: Path to the .als file

    Returns:
        Dictionary mapping track names to their MidiClipContent lists.
    """
    # Build lookup of effectively enabled tracks by name
    enabled_track_names = {
        t.name for t in live_set.tracks if live_set.is_track_effectively_enabled(t)
    }

    root = parse_als_xml(als_path)
    track_elements = get_track_elements(root)

    result: dict[str, list[MidiClipContent]] = {}

    for track_elem, track_type in track_elements:
        if track_type != "midi":
            continue

        track_name = extract_track_name(track_elem)

        # Skip disabled tracks or tracks in disabled groups
        if track_name not in enabled_track_names:
            continue

        clips = extract_track_clips(track_elem, track_type)

        if not clips:
            continue

        clip_contents = extract_midi_clip_contents(track_elem, clips)

        if clip_contents:
            result[track_name] = clip_contents

    return result


def get_all_track_names(live_set: LiveSet) -> list[str]:
    """Get all effectively enabled track names from an ALS file.

    Returns only tracks that are enabled and not in disabled groups.

    Args:
        live_set: Pre-parsed LiveSet containing track/group enabled state.

    Returns:
        List of enabled track names (both MIDI and audio tracks).
    """
    return [t.name for t in live_set.tracks if live_set.is_track_effectively_enabled(t)]


def align_and_distribute_lyrics(
    als_path: Path,
    lyrics_path: Path,
    phrases: list[Phrase],
    bpm: float,
    vocal_tracks: tuple[str, ...] | None = None,
    use_all_vocals: bool = False,
    save_vocals_path: Path | None = None,
    save_lyrics_path: Path | None = None,
    language: str = "en",
    model_size: str = "base",
) -> list[Phrase]:
    """Full alignment pipeline: extract audio, align lyrics, distribute to phrases.

    This orchestration function combines audio extraction, forced alignment,
    and lyrics distribution into a single workflow.

    Steps:
    1. Extract audio clips from ALS file
    2. Select vocal tracks (using explicit selection, auto-detection, or prompting)
    3. Combine vocal clips to a single audio file
    4. Run forced alignment with stable-ts
    5. Refine with Bournemouth aligner if available (better for repeated lyrics)
    6. Save aligned lyrics if requested
    7. Distribute timed lyrics to phrases

    Args:
        als_path: Path to the .als file
        lyrics_path: Path to the lyrics file
        phrases: List of phrases to annotate with lyrics
        bpm: Tempo in beats per minute
        vocal_tracks: Specific vocal track names to use (None for auto-detect)
        use_all_vocals: If True, use all detected vocal tracks without prompting
        save_vocals_path: If provided, save combined vocals to this path for validation
        save_lyrics_path: If provided, save aligned lyrics in LRC format
        language: Language code for alignment model (default: "en")
        model_size: Whisper model size (default: "base")

    Returns:
        Phrases with lyric fields populated from forced alignment.

    Raises:
        AlignmentError: If alignment fails for any reason (no vocal tracks,
            audio extraction failure, alignment model failure, etc.)
    """
    # Step 1: Extract all audio clips from ALS
    try:
        all_clips = extract_audio_clips(als_path, bpm)
    except Exception as e:
        raise AlignmentError(f"Failed to extract audio clips: {e}") from e

    if not all_clips:
        raise AlignmentError("No audio clips found in ALS file")

    # Step 2: Select vocal tracks (with config integration)
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

    # Step 3: Combine vocals into single audio file
    is_temp_file = save_vocals_path is None
    if save_vocals_path is not None:
        combined_path = save_vocals_path
    else:
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix="alsmuse_vocals_"
            ) as tmp:
                combined_path = Path(tmp.name)
        except OSError as e:
            raise AlignmentError(f"Failed to create temporary file: {e}") from e

    try:
        try:
            _, valid_ranges = combine_clips_to_audio(selected_clips, combined_path, bpm)
        except Exception as e:
            raise AlignmentError(f"Failed to combine audio clips: {e}") from e

        if save_vocals_path is not None:
            logger.info("Saved combined vocals to: %s", combined_path)

        # Step 4: Read lyrics as plain text and run alignment
        all_lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()

        if not all_lyrics_text:
            raise AlignmentError("Lyrics file is empty or contains no text")

        # Run forced alignment
        timed_words = align_lyrics(
            combined_path,
            all_lyrics_text,
            valid_ranges=valid_ranges,
            language=language,
            model_size=model_size,
        )

        # Step 5: Reconstruct lines and distribute to phrases
        original_lines = [line.strip() for line in all_lyrics_text.split("\n") if line.strip()]
        timed_lines = words_to_lines(timed_words, original_lines)

        # Step 5b: Refine with Bournemouth if available (handles repeated lyrics better)
        pbar: tqdm[None] | None = None

        def update_progress(current: int, total: int, text: str) -> None:
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=total, desc="Refining alignment", unit="line")
            pbar.update(1)
            pbar.set_postfix_str(text[:30])

        timed_lines, refinement_messages = refine_lines_with_bournemouth(
            combined_path,
            timed_lines,
            language=f"{language}-us" if language == "en" else language,
            progress_callback=update_progress,
            valid_ranges=valid_ranges,
        )
        if pbar is not None:
            pbar.close()
        for msg in refinement_messages:
            click.echo(msg, err=True)

        # Validate alignment and warn about issues
        alignment_warnings = validate_timed_lines(timed_lines)
        for warning in alignment_warnings:
            click.echo(warning, err=True)

        # Step 6: Save aligned lyrics if requested (in LRC format)
        if save_lyrics_path is not None:
            try:
                lrc_content = format_timed_lines_as_lrc(timed_lines)
                save_lyrics_path.write_text(lrc_content, encoding="utf-8")
                logger.info("Saved aligned lyrics to: %s", save_lyrics_path)
            except OSError as e:
                logger.warning("Failed to save lyrics: %s", e)

        return distribute_timed_lyrics(phrases, timed_lines, bpm)

    finally:
        # Only cleanup if it was a temp file
        if is_temp_file and combined_path.exists():
            try:
                combined_path.unlink()
            except OSError:
                logger.warning("Failed to clean up temporary file: %s", combined_path)


def transcribe_and_distribute_lyrics(
    als_path: Path,
    phrases: list[Phrase],
    bpm: float,
    vocal_tracks: tuple[str, ...] | None = None,
    use_all_vocals: bool = False,
    save_vocals_path: Path | None = None,
    save_lyrics_path: Path | None = None,
    language: str = "en",
    model_size: str = "base",
) -> list[Phrase]:
    """Full transcription pipeline: extract audio, transcribe, distribute.

    Steps:
    1. Extract audio clips from ALS file
    2. Select vocal tracks
    3. Combine vocal clips to a single audio file
    4. Run ASR transcription with stable-ts (preserving segments)
    5. Filter segments to valid audio ranges (remove hallucinations)
    6. Convert segments to lines (splitting only if too long)
    7. Distribute timed lyrics to phrases

    Args:
        als_path: Path to the .als file
        phrases: List of phrases to annotate with lyrics
        bpm: Tempo in beats per minute
        vocal_tracks: Specific vocal track names to use (None for auto-detect)
        use_all_vocals: If True, use all detected vocal tracks without prompting
        save_vocals_path: If provided, save combined vocals to this path
        save_lyrics_path: If provided, save transcribed lyrics to this path
        language: Language code for transcription (default: "en")
        model_size: Whisper model size (default: "base")

    Returns:
        Phrases with lyric fields populated from transcription.

    Raises:
        AlignmentError: If transcription fails for any reason (no vocal tracks,
            audio extraction failure, transcription model failure, etc.)
    """
    # Step 1: Extract all audio clips from ALS
    try:
        all_clips = extract_audio_clips(als_path, bpm)
    except Exception as e:
        raise AlignmentError(f"Failed to extract audio clips: {e}") from e

    if not all_clips:
        raise AlignmentError("No audio clips found in ALS file")

    # Step 2: Select vocal tracks (with config integration)
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

    # Step 3: Combine vocals into single audio file
    is_temp_file = save_vocals_path is None
    if save_vocals_path is not None:
        combined_path = save_vocals_path
    else:
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix="alsmuse_vocals_"
            ) as tmp:
                combined_path = Path(tmp.name)
        except OSError as e:
            raise AlignmentError(f"Failed to create temporary file: {e}") from e

    try:
        try:
            _, valid_ranges = combine_clips_to_audio(selected_clips, combined_path, bpm)
        except Exception as e:
            raise AlignmentError(f"Failed to combine audio clips: {e}") from e

        if save_vocals_path is not None:
            logger.info("Saved combined vocals to: %s", combined_path)

        # Step 4 & 5: Transcribe and filter to valid ranges
        segments, raw_text = transcribe_lyrics(
            combined_path,
            language=language,
            model_size=model_size,
        )

        # Step 6: Save transcription if requested (in LRC format with timestamps)
        if save_lyrics_path is not None:
            try:
                lrc_content = format_segments_as_lrc(segments)
                save_lyrics_path.write_text(lrc_content, encoding="utf-8")
                logger.info("Saved transcribed lyrics to: %s", save_lyrics_path)
            except OSError as e:
                logger.warning("Failed to save lyrics: %s", e)

        # Step 7: Convert segments to lines
        timed_lines = segments_to_lines(segments)

        # Step 7b: Refine with Bournemouth if available (handles repeated lyrics better)
        pbar: tqdm[None] | None = None

        def update_progress(current: int, total: int, text: str) -> None:
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=total, desc="Refining alignment", unit="line")
            pbar.update(1)
            pbar.set_postfix_str(text[:30])

        timed_lines, refinement_messages = refine_lines_with_bournemouth(
            combined_path,
            timed_lines,
            language=f"{language}-us" if language == "en" else language,
            progress_callback=update_progress,
            valid_ranges=valid_ranges,
        )
        if pbar is not None:
            pbar.close()
        for msg in refinement_messages:
            click.echo(msg, err=True)

        # Validate transcription and warn about issues
        transcription_warnings = validate_timed_lines(timed_lines)
        for warning in transcription_warnings:
            click.echo(warning, err=True)

        # Step 8: Distribute to phrases
        return distribute_timed_lyrics(phrases, timed_lines, bpm)

    finally:
        # Only cleanup if it was a temp file
        if is_temp_file and combined_path.exists():
            try:
                combined_path.unlink()
            except OSError:
                logger.warning("Failed to clean up temporary file: %s", combined_path)
