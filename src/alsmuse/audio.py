"""Audio extraction for ALSmuse.

This module provides functionality to extract audio clip references from
Ableton Live Set files, resolving file paths using a relative-first strategy
that handles projects moved between computers.

It also provides vocal track identification, audio combination, and
interactive track selection for the lyrics alignment feature.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element

from .models import AudioClipRef
from .parser import extract_track_name, get_track_elements, parse_als_xml

logger = logging.getLogger(__name__)

# Keywords used to identify vocal tracks by name
VOCAL_KEYWORDS: list[str] = [
    "vocal",
    "vox",
    "voice",
    "lead vocal",
    "main vocal",
    "verse",
    "chorus",
    "bridge",
    "harmony",
    "double",
    "backing",
    "singer",
    "rap",
    "spoken",
]


def resolve_audio_path(
    als_path: Path,
    relative_path: str,
    absolute_path: str,
) -> Path | None:
    """Resolve audio file path, prioritizing relative paths.

    ALS files store both absolute and relative paths for audio samples.
    Absolute paths break when projects move between computers, so we
    prioritize relative paths for reliable path resolution.

    Resolution order:
    1. Relative path from ALS file's parent directory
    2. Relative path from ALS file's grandparent (project root)
    3. Absolute path as fallback (may be stale)

    Args:
        als_path: Path to the .als file
        relative_path: Value from <RelativePath> element
        absolute_path: Value from <Path> element

    Returns:
        Resolved Path if file exists, None otherwise.
    """
    als_dir = als_path.parent

    # Strategy 1: Relative to ALS directory
    if relative_path:
        candidate = als_dir / relative_path
        if candidate.exists():
            return candidate.resolve()

        # Strategy 2: Relative to project root (parent of ALS)
        # Common structure: Project/Project.als + Project/Samples/...
        project_root = als_dir.parent
        candidate = project_root / relative_path
        if candidate.exists():
            return candidate.resolve()

    # Strategy 3: Try the stored absolute path (often stale)
    if absolute_path:
        candidate = Path(absolute_path)
        if candidate.exists():
            return candidate.resolve()

    return None


def beats_to_seconds(beats: float, bpm: float) -> float:
    """Convert beat position to seconds.

    Args:
        beats: Position in beats.
        bpm: Tempo in beats per minute.

    Returns:
        Position in seconds.
    """
    return beats * 60.0 / bpm


def extract_audio_clips(
    als_path: Path,
    bpm: float,
) -> list[AudioClipRef]:
    """Extract all audio clip references from an ALS file.

    Parses AudioClip elements and resolves their file paths.
    Skips clips with unresolvable paths (logs warning).

    Args:
        als_path: Path to the .als file
        bpm: Tempo for beat-to-seconds conversion

    Returns:
        List of AudioClipRef with resolved paths and timing.
    """
    root = parse_als_xml(als_path)
    track_elements = get_track_elements(root)

    clips: list[AudioClipRef] = []

    for track_elem, track_type in track_elements:
        if track_type != "audio":
            continue

        track_name = extract_track_name(track_elem)
        track_clips = _extract_audio_clips_from_track(
            track_elem, track_name, als_path, bpm
        )
        clips.extend(track_clips)

    return clips


def _extract_audio_clips_from_track(
    track_element: Element,
    track_name: str,
    als_path: Path,
    bpm: float,
) -> list[AudioClipRef]:
    """Extract audio clips from a single track element.

    Args:
        track_element: An AudioTrack XML element.
        track_name: Name of the track.
        als_path: Path to the ALS file for path resolution.
        bpm: Tempo for beat-to-seconds conversion.

    Returns:
        List of AudioClipRef objects from this track.
    """
    clips: list[AudioClipRef] = []

    # Path to arrangement clips for audio tracks
    # Note: Audio tracks use Sample/ArrangerAutomation, not ClipTimeable
    events_path = "DeviceChain/MainSequencer/Sample/ArrangerAutomation/Events"
    events_elem = track_element.find(events_path)

    if events_elem is None:
        return clips

    for clip_elem in events_elem.findall("AudioClip"):
        clip = _parse_audio_clip_element(clip_elem, track_name, als_path, bpm)
        if clip is not None:
            clips.append(clip)

    return clips


def _parse_audio_clip_element(
    clip_elem: Element,
    track_name: str,
    als_path: Path,
    bpm: float,
) -> AudioClipRef | None:
    """Parse a single AudioClip element into an AudioClipRef.

    Args:
        clip_elem: An AudioClip XML element.
        track_name: Name of the containing track.
        als_path: Path to the ALS file for path resolution.
        bpm: Tempo for beat-to-seconds conversion.

    Returns:
        AudioClipRef if the clip can be parsed and its path resolved,
        None otherwise.
    """
    # Get start position from Time attribute
    time_str = clip_elem.get("Time")
    if time_str is None:
        return None

    try:
        start_beats = float(time_str)
    except ValueError:
        return None

    # Get end position from CurrentEnd element
    current_end_elem = clip_elem.find("CurrentEnd")
    if current_end_elem is None:
        return None

    end_str = current_end_elem.get("Value")
    if end_str is None:
        return None

    try:
        end_beats = float(end_str)
    except ValueError:
        return None

    # Extract file path from SampleRef/FileRef
    file_path = _extract_file_path(clip_elem, als_path)
    if file_path is None:
        clip_name = clip_elem.find("Name")
        clip_name_str = (
            clip_name.get("Value", "<unknown>") if clip_name is not None else "<unknown>"
        )
        logger.warning(
            "Could not resolve audio path for clip '%s' in track '%s'",
            clip_name_str,
            track_name,
        )
        return None

    # Convert beats to seconds
    start_seconds = beats_to_seconds(start_beats, bpm)
    end_seconds = beats_to_seconds(end_beats, bpm)

    return AudioClipRef(
        track_name=track_name,
        file_path=file_path,
        start_beats=start_beats,
        end_beats=end_beats,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _extract_file_path(clip_elem: Element, als_path: Path) -> Path | None:
    """Extract and resolve file path from an AudioClip element.

    Looks for SampleRef/FileRef structure and extracts both
    RelativePath and Path elements for resolution.

    Args:
        clip_elem: An AudioClip XML element.
        als_path: Path to the ALS file for path resolution.

    Returns:
        Resolved Path if found and exists, None otherwise.
    """
    # Find SampleRef element
    sample_ref = clip_elem.find("SampleRef")
    if sample_ref is None:
        return None

    file_ref = sample_ref.find("FileRef")
    if file_ref is None:
        return None

    # Extract relative path
    relative_path_elem = file_ref.find("RelativePath")
    relative_path = ""
    if relative_path_elem is not None:
        relative_path = relative_path_elem.get("Value", "")

    # Extract absolute path
    path_elem = file_ref.find("Path")
    absolute_path = ""
    if path_elem is not None:
        absolute_path = path_elem.get("Value", "")

    return resolve_audio_path(als_path, relative_path, absolute_path)


# ---------------------------------------------------------------------------
# Vocal Track Identification (Phase 2)
# ---------------------------------------------------------------------------


def is_vocal_track(track_name: str) -> bool:
    """Check if a track name suggests vocal content.

    Uses keyword matching against VOCAL_KEYWORDS to identify tracks
    that likely contain vocal audio.

    Args:
        track_name: Name of the track to check.

    Returns:
        True if track name contains any vocal keyword.
    """
    name_lower = track_name.lower()
    return any(keyword in name_lower for keyword in VOCAL_KEYWORDS)


def find_vocal_clips(
    clips: list[AudioClipRef],
    explicit_tracks: tuple[str, ...] | None = None,
) -> list[AudioClipRef]:
    """Filter audio clips to those likely containing vocals.

    Args:
        clips: All audio clips from ALS.
        explicit_tracks: If provided, match these track names exactly
            (case-insensitive). If None, use keyword-based auto-detection.

    Returns:
        Filtered list of vocal clips, sorted by start time.
    """
    if explicit_tracks:
        explicit_lower = {t.lower() for t in explicit_tracks}
        matching_clips = [
            c for c in clips if c.track_name.lower() in explicit_lower
        ]
    else:
        matching_clips = [c for c in clips if is_vocal_track(c.track_name)]

    return sorted(matching_clips, key=lambda c: c.start_beats)


def get_unique_vocal_track_names(clips: list[AudioClipRef]) -> list[str]:
    """Get unique track names from clips, preserving order of first occurrence.

    Args:
        clips: List of audio clips to extract track names from.

    Returns:
        List of unique track names in order of first occurrence.
    """
    seen: set[str] = set()
    result: list[str] = []
    for clip in clips:
        if clip.track_name not in seen:
            seen.add(clip.track_name)
            result.append(clip.track_name)
    return result


# ---------------------------------------------------------------------------
# Audio Combination (Phase 2)
# ---------------------------------------------------------------------------


def _get_target_sample_rate(clips: list[AudioClipRef]) -> int:
    """Determine the target sample rate (most common among clips).

    Args:
        clips: List of audio clips to analyze.

    Returns:
        The most common sample rate among the clips.
    """
    from collections import Counter

    import soundfile as sf  # type: ignore[import-untyped]

    rates: list[int] = []
    for clip in clips:
        try:
            info = sf.info(str(clip.file_path))
            rates.append(info.samplerate)
        except Exception:
            continue

    if not rates:
        return 44100  # Default fallback

    # Return most common sample rate
    counter = Counter(rates)
    return counter.most_common(1)[0][0]


def _resample_audio(
    audio: Any,
    orig_sr: int,
    target_sr: int,
) -> Any:
    """Resample audio using torchaudio (fast, GPU-capable).

    Args:
        audio: Audio data as numpy array (samples,) or (samples, channels).
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio as numpy array.
    """
    import torch  # type: ignore[import-untyped]
    import torchaudio  # type: ignore[import-untyped]

    # Convert to torch tensor
    # torchaudio expects (channels, samples) format
    tensor = (
        torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
        if audio.ndim == 1
        else torch.from_numpy(audio).T  # (channels, samples)
    )

    # Resample
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    resampled = resampler(tensor)

    # Convert back to numpy in original format
    if audio.ndim == 1:
        return resampled.squeeze(0).numpy()
    else:
        return resampled.T.numpy()


def combine_clips_to_audio(
    clips: list[AudioClipRef],
    output_path: Path,
) -> tuple[Path, list[tuple[float, float]]]:
    """Combine audio clips into a single file, preserving timeline positions.

    Creates a silent buffer and mixes each clip at its correct position.
    Returns both the combined audio path AND the valid time ranges where
    actual audio exists (for hallucination filtering).

    Uses only numpy + soundfile (no ffmpeg, no pydub).
    Supports: WAV, AIFF, FLAC (common Ableton formats).
    Does NOT support: MP3.

    Args:
        clips: Audio clips to combine. Must not be empty.
        output_path: Where to write the combined audio.

    Returns:
        Tuple of:
        - Path to combined audio file
        - List of (start_seconds, end_seconds) tuples for valid audio ranges

    Raises:
        ValueError: If clips list is empty.
        RuntimeError: If soundfile is not installed.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(
            "soundfile is required for audio combination. "
            "Install with: pip install 'alsmuse[align]'"
        ) from e

    if not clips:
        raise ValueError("No clips to combine")

    # Determine target sample rate (most common among clips)
    sample_rate = _get_target_sample_rate(clips)

    # Load first clip to determine channel count
    first_audio, first_sr = sf.read(str(clips[0].file_path), dtype="float32")
    if first_sr != sample_rate:
        first_audio = _resample_audio(first_audio, first_sr, sample_rate)
    channels = first_audio.shape[1] if first_audio.ndim > 1 else 1

    # Calculate total duration in samples
    total_seconds = max(c.end_seconds for c in clips)
    total_samples = int(total_seconds * sample_rate)

    # Create silent buffer
    if channels > 1:
        combined = np.zeros((total_samples, channels), dtype=np.float32)
    else:
        combined = np.zeros(total_samples, dtype=np.float32)

    # Track valid audio ranges for hallucination filtering
    valid_ranges: list[tuple[float, float]] = []

    for clip in clips:
        audio, sr = sf.read(str(clip.file_path), dtype="float32")

        # Resample if needed (uses torchaudio for speed)
        if sr != sample_rate:
            audio = _resample_audio(audio, sr, sample_rate)

        # Handle mono/stereo mismatch
        if channels > 1 and audio.ndim == 1:
            # Convert mono to stereo by duplicating
            audio = np.column_stack([audio, audio])
        elif channels == 1 and audio.ndim > 1:
            # Convert stereo to mono by averaging
            audio = audio.mean(axis=1)

        # Calculate position in samples
        start_sample = int(clip.start_seconds * sample_rate)
        clip_samples = min(len(audio), total_samples - start_sample)

        # Ensure we don't exceed buffer bounds
        if start_sample < total_samples and clip_samples > 0:
            # Mix into combined buffer (additive for overlapping clips)
            combined[start_sample : start_sample + clip_samples] += audio[:clip_samples]

        # Record valid range
        valid_ranges.append((clip.start_seconds, clip.end_seconds))

    # Normalize to prevent clipping if peaks exceed 1.0
    peak = np.abs(combined).max()
    if peak > 1.0:
        combined /= peak

    # Export as WAV
    sf.write(str(output_path), combined, sample_rate)

    return output_path, valid_ranges


# ---------------------------------------------------------------------------
# Interactive Track Selection (Phase 2)
# ---------------------------------------------------------------------------


def prompt_track_selection(
    track_names: list[str],
    auto_select_single: bool = True,
) -> list[str]:
    """Interactively prompt user to select vocal tracks.

    Args:
        track_names: List of detected vocal track names.
        auto_select_single: If True and only one track, skip prompt.

    Returns:
        List of selected track names.

    Raises:
        click.Abort: If user cancels selection.
        RuntimeError: If questionary is not installed.
    """
    import click

    try:
        import questionary
    except ImportError as e:
        raise RuntimeError(
            "questionary is required for interactive track selection. "
            "Install with: pip install 'alsmuse[align]'"
        ) from e

    if not track_names:
        return []

    # Single track: use it automatically
    if len(track_names) == 1 and auto_select_single:
        return track_names

    # Multiple tracks: interactive selection
    selected = questionary.checkbox(
        "Select vocal tracks to include:",
        choices=[
            questionary.Choice(name, checked=True)  # Pre-select all
            for name in track_names
        ],
    ).ask()

    if selected is None:
        # User cancelled (Ctrl+C)
        raise click.Abort()

    # Cast to list[str] since questionary returns Any
    return list(selected)


def select_vocal_tracks(
    all_clips: list[AudioClipRef],
    explicit_tracks: tuple[str, ...] | None,
    use_all: bool,
) -> list[AudioClipRef]:
    """Select vocal tracks for alignment.

    Handles three modes:
    1. Explicit tracks specified via CLI options
    2. use_all=True: Use all detected vocal tracks without prompting
    3. Interactive: Prompt user when multiple tracks found (if TTY available)

    Args:
        all_clips: All audio clips from ALS.
        explicit_tracks: Tracks specified via --vocal-track options.
        use_all: If True, use all detected vocal tracks without prompting.

    Returns:
        Filtered list of clips from selected tracks, sorted by start time.
    """
    import click

    # Explicit selection via CLI
    if explicit_tracks:
        explicit_lower = {t.lower() for t in explicit_tracks}
        return sorted(
            [c for c in all_clips if c.track_name.lower() in explicit_lower],
            key=lambda c: c.start_beats,
        )

    # Find all potential vocal tracks
    vocal_clips = [c for c in all_clips if is_vocal_track(c.track_name)]
    track_names = get_unique_vocal_track_names(vocal_clips)

    if not track_names:
        return []

    # Use all without prompting
    if use_all:
        return sorted(vocal_clips, key=lambda c: c.start_beats)

    # Single track: use automatically
    if len(track_names) == 1:
        return sorted(vocal_clips, key=lambda c: c.start_beats)

    # Interactive selection (if TTY available)
    if sys.stdin.isatty():
        selected_names = prompt_track_selection(track_names)
        selected_set = set(selected_names)
        return sorted(
            [c for c in vocal_clips if c.track_name in selected_set],
            key=lambda c: c.start_beats,
        )

    # Non-TTY: use all and warn
    click.echo(
        f"Multiple vocal tracks found: {', '.join(track_names)}. "
        "Using all. Use --vocal-track to select specific tracks.",
        err=True,
    )
    return sorted(vocal_clips, key=lambda c: c.start_beats)


# ---------------------------------------------------------------------------
# Dependency Validation (Phase 4)
# ---------------------------------------------------------------------------


def check_alignment_dependencies() -> list[str]:
    """Check that all required dependencies for alignment are available.

    Validates that the optional packages stable-ts and soundfile are installed.
    This should be called at the CLI level before attempting alignment to
    provide helpful error messages to users.

    Returns:
        List of error messages describing missing dependencies.
        Empty list if all dependencies are satisfied.
    """
    errors: list[str] = []

    # Check stable-ts
    try:
        import stable_whisper  # type: ignore[import-not-found,import-untyped]  # noqa: F401
    except ImportError:
        errors.append(
            "stable-ts not installed. Run: pip install 'alsmuse[align]'"
        )

    # Check soundfile
    try:
        import soundfile  # type: ignore[import-not-found,import-untyped]  # noqa: F401
    except ImportError:
        errors.append(
            "soundfile not installed. Run: pip install 'alsmuse[align]'"
        )

    return errors
