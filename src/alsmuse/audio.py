"""Audio extraction for ALSmuse.

This module provides functionality to extract audio clip references from
Ableton Live Set files, resolving file paths using a relative-first strategy
that handles projects moved between computers.

It also provides vocal track identification, audio combination, and
interactive track selection for the lyrics alignment feature.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element

import click
import numpy as np
import questionary
import soundfile as sf  # type: ignore[import-untyped]
import torch  # type: ignore[import-untyped]
import torchaudio  # type: ignore[import-untyped]

from .config import MuseConfig, load_config, save_config
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
        track_clips = _extract_audio_clips_from_track(track_elem, track_name, als_path, bpm)
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


def _beats_to_file_seconds(
    beat_time: float,
    warp_markers: list[tuple[float, float]],
    bpm: float,
) -> float:
    """Convert clip-internal beat position to file seconds using warp markers.

    Warp markers define a mapping from BeatTime (clip-internal beats) to
    SecTime (actual position in the audio file). This function interpolates
    between markers to find the file position for a given beat.

    Args:
        beat_time: Beat position in clip-internal time (e.g., LoopStart).
        warp_markers: List of (beat_time, sec_time) tuples, sorted by beat_time.
        bpm: Project tempo, used as fallback if no warp markers exist.

    Returns:
        Position in seconds within the audio file.
    """
    if not warp_markers:
        # No warp markers - fall back to BPM conversion
        return beats_to_seconds(beat_time, bpm)

    # If beat_time is before first marker, extrapolate backwards
    if beat_time <= warp_markers[0][0]:
        if len(warp_markers) >= 2:
            # Use the slope between first two markers
            beat1, sec1 = warp_markers[0]
            beat2, sec2 = warp_markers[1]
            if beat2 != beat1:
                slope = (sec2 - sec1) / (beat2 - beat1)
                return sec1 + slope * (beat_time - beat1)
        # Single marker or same beat position - assume 1:1 mapping at project tempo
        beat1, sec1 = warp_markers[0]
        return sec1 + beats_to_seconds(beat_time - beat1, bpm)

    # If beat_time is after last marker, extrapolate forwards
    if beat_time >= warp_markers[-1][0]:
        if len(warp_markers) >= 2:
            # Use the slope between last two markers
            beat1, sec1 = warp_markers[-2]
            beat2, sec2 = warp_markers[-1]
            if beat2 != beat1:
                slope = (sec2 - sec1) / (beat2 - beat1)
                return sec2 + slope * (beat_time - beat2)
        # Single marker - extrapolate at project tempo
        beat1, sec1 = warp_markers[-1]
        return sec1 + beats_to_seconds(beat_time - beat1, bpm)

    # Find the two markers surrounding beat_time and interpolate
    for i in range(len(warp_markers) - 1):
        beat1, sec1 = warp_markers[i]
        beat2, sec2 = warp_markers[i + 1]
        if beat1 <= beat_time <= beat2:
            if beat2 == beat1:
                return sec1
            # Linear interpolation
            t = (beat_time - beat1) / (beat2 - beat1)
            return sec1 + t * (sec2 - sec1)

    # Fallback (shouldn't reach here)
    return beats_to_seconds(beat_time, bpm)


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

    # Extract warp markers for beat-to-file-time conversion
    # Warp markers map BeatTime (clip-internal beats) to SecTime (file seconds)
    warp_markers: list[tuple[float, float]] = []  # (beat_time, sec_time)
    warp_markers_elem = clip_elem.find(".//WarpMarkers")
    if warp_markers_elem is not None:
        for marker in warp_markers_elem.findall("WarpMarker"):
            sec_time_str = marker.get("SecTime")
            beat_time_str = marker.get("BeatTime")
            if sec_time_str is not None and beat_time_str is not None:
                with contextlib.suppress(ValueError):
                    warp_markers.append((float(beat_time_str), float(sec_time_str)))
        # Sort by beat time for interpolation
        warp_markers.sort(key=lambda x: x[0])

    # Extract sample start/end offsets from Loop element
    # These indicate which portion of the audio file to play
    sample_start_seconds: float | None = None
    sample_end_seconds: float | None = None

    loop_elem = clip_elem.find("Loop")
    if loop_elem is not None:
        loop_start_elem = loop_elem.find("LoopStart")
        loop_end_elem = loop_elem.find("LoopEnd")

        loop_start_beats: float | None = None
        loop_end_beats: float | None = None

        if loop_start_elem is not None:
            with contextlib.suppress(ValueError):
                loop_start_beats = float(loop_start_elem.get("Value", "0"))

        if loop_end_elem is not None:
            with contextlib.suppress(ValueError):
                loop_end_beats = float(loop_end_elem.get("Value", "0"))

        # Convert beats to file seconds using warp markers
        if loop_start_beats is not None:
            sample_start_seconds = _beats_to_file_seconds(loop_start_beats, warp_markers, bpm)
        if loop_end_beats is not None:
            sample_end_seconds = _beats_to_file_seconds(loop_end_beats, warp_markers, bpm)

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
        sample_start_seconds=sample_start_seconds,
        sample_end_seconds=sample_end_seconds,
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
        matching_clips = [c for c in clips if c.track_name.lower() in explicit_lower]
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


def _apply_compression(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 200.0,
) -> np.ndarray:
    """Apply dynamic range compression to audio.

    Uses an envelope follower with attack/release to compute gain reduction,
    then applies makeup gain to bring the overall level up.

    Args:
        audio: Audio samples as numpy array (mono or stereo).
        sample_rate: Sample rate of the audio.
        threshold_db: Level (in dB) above which compression starts.
        ratio: Compression ratio (e.g., 4.0 means 4:1 compression).
        attack_ms: Attack time in milliseconds (how fast compression engages).
        release_ms: Release time in milliseconds (how fast compression releases).

    Returns:
        Compressed audio as numpy array (same shape as input).
    """
    if len(audio) == 0:
        return audio

    # Convert to mono for envelope detection, preserve original for processing
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio

    # Convert threshold to linear
    threshold_linear = 10 ** (threshold_db / 20)

    # Calculate attack and release coefficients
    # These determine how fast the envelope follower responds
    attack_samples = int(attack_ms * sample_rate / 1000)
    release_samples = int(release_ms * sample_rate / 1000)

    # Coefficients for exponential smoothing
    attack_coef = np.exp(-1.0 / max(attack_samples, 1))
    release_coef = np.exp(-1.0 / max(release_samples, 1))

    # Get absolute values for envelope detection
    abs_audio = np.abs(mono)

    # Compute envelope using attack/release follower
    envelope = np.zeros(len(mono), dtype=np.float32)
    env_value = 0.0

    for i in range(len(mono)):
        input_level = abs_audio[i]
        if input_level > env_value:
            # Attack: envelope rises toward input
            env_value = attack_coef * env_value + (1 - attack_coef) * input_level
        else:
            # Release: envelope falls toward input
            env_value = release_coef * env_value + (1 - release_coef) * input_level
        envelope[i] = env_value

    # Compute gain reduction based on envelope
    # For levels above threshold, apply compression ratio
    gain = np.ones(len(mono), dtype=np.float32)

    above_threshold = envelope > threshold_linear
    if np.any(above_threshold):
        # Convert to dB for compression calculation
        # Use maximum to avoid log10(0) warning
        envelope_db = 20 * np.log10(np.maximum(envelope, 1e-10))

        # Calculate gain reduction in dB
        # For signal X dB above threshold, output is threshold + (X / ratio)
        overshoot_db = np.maximum(0, envelope_db - threshold_db)
        gain_reduction_db = overshoot_db * (1 - 1 / ratio)

        # Convert back to linear gain
        gain = (10 ** (-gain_reduction_db / 20)).astype(np.float32)

    # Apply gain to audio
    compressed = audio * gain[:, np.newaxis] if audio.ndim > 1 else audio * gain

    # Calculate makeup gain to bring level up
    # Target peak at -3 dB
    target_peak_db = -3.0
    target_peak_linear = 10 ** (target_peak_db / 20)

    current_peak = np.abs(compressed).max()
    if current_peak > 1e-10:
        makeup_gain = target_peak_linear / current_peak
        compressed = compressed * makeup_gain

    return compressed.astype(np.float32)


def _get_target_sample_rate(clips: list[AudioClipRef]) -> int:
    """Determine the target sample rate (most common among clips).

    Args:
        clips: List of audio clips to analyze.

    Returns:
        The most common sample rate among the clips.
    """
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
    bpm: float = 120.0,
) -> tuple[Path, list[tuple[float, float]]]:
    """Combine audio clips into a single file, preserving timeline positions.

    Creates a silent buffer and mixes each clip at its correct position.
    Returns both the combined audio path AND the valid time ranges where
    actual audio exists (for hallucination filtering).

    Each clip's sample_start_beats and sample_end_beats are used to extract
    only the relevant portion of the audio file (Ableton's Loop region).

    Uses only numpy + soundfile (no ffmpeg, no pydub).
    Supports: WAV, AIFF, FLAC (common Ableton formats).
    Does NOT support: MP3.

    Args:
        clips: Audio clips to combine. Must not be empty.
        output_path: Where to write the combined audio.
        bpm: Beats per minute, used for converting sample offsets.

    Returns:
        Tuple of:
        - Path to combined audio file
        - List of (start_seconds, end_seconds) tuples for valid audio ranges

    Raises:
        ValueError: If clips list is empty.
        RuntimeError: If soundfile is not installed.
    """
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
            # Update sr to reflect the resampled audio's sample rate
            sr = sample_rate

        # Extract the correct portion of the audio file using sample offsets
        # These are already converted to file seconds using warp markers
        if clip.sample_start_seconds is not None and clip.sample_end_seconds is not None:
            sample_start_seconds = clip.sample_start_seconds
            sample_end_seconds = clip.sample_end_seconds

            sample_start_idx = int(sample_start_seconds * sr)
            sample_end_idx = int(sample_end_seconds * sr)

            # Clamp to valid range
            sample_start_idx = max(0, min(sample_start_idx, len(audio)))
            sample_end_idx = max(sample_start_idx, min(sample_end_idx, len(audio)))

            # Extract the portion
            audio = audio[sample_start_idx:sample_end_idx]

        # Handle mono/stereo mismatch
        if channels > 1 and audio.ndim == 1:
            # Convert mono to stereo by duplicating
            audio = np.column_stack([audio, audio])
        elif channels == 1 and audio.ndim > 1:
            # Convert stereo to mono by averaging
            audio = audio.mean(axis=1)

        # Calculate position in samples (timeline position)
        start_sample = int(clip.start_seconds * sample_rate)
        clip_samples = min(len(audio), total_samples - start_sample)

        # Ensure we don't exceed buffer bounds
        if start_sample < total_samples and clip_samples > 0:
            # Mix into combined buffer (additive for overlapping clips)
            combined[start_sample : start_sample + clip_samples] += audio[:clip_samples]

        # Detect non-silent regions within this clip and add with timeline offset
        # This catches silences within clips (pauses between phrases)
        clip_ranges = _detect_non_silent_ranges_from_array(
            audio[:clip_samples],
            sample_rate,
            min_silence_duration=2.0,  # 2s silence = gap between phrases
            silence_threshold_db=-40.0,
            time_offset=clip.start_seconds,
        )
        valid_ranges.extend(clip_ranges)

    # Apply compression to increase volume and limit dynamic range
    # This helps with transcription by making quiet passages more audible
    combined = _apply_compression(combined, sample_rate)

    # Export as WAV
    sf.write(str(output_path), combined, sample_rate)

    return output_path, valid_ranges


def split_audio_on_silence(
    audio_path: Path,
    output_dir: Path,
    min_silence_duration: float = 1.0,
    silence_threshold_db: float = -40.0,
    min_segment_duration: float = 0.5,
) -> list[tuple[Path, float, float]]:
    """Split audio file into segments based on silence detection.

    Detects silent regions and splits the audio into separate files,
    one per non-silent segment. This is ideal for transcription since
    Whisper hallucinates during silence.

    Args:
        audio_path: Path to the audio file to split.
        output_dir: Directory to write segment files.
        min_silence_duration: Minimum silence duration (seconds) to trigger split.
        silence_threshold_db: Audio level below this (in dB) is considered silence.
        min_segment_duration: Minimum segment duration to keep (seconds).

    Returns:
        List of (segment_path, original_start_time, original_end_time) tuples.
        Times are in seconds relative to the original audio file.
    """
    # Read audio
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")

    # Convert to mono for silence detection
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio

    # Convert threshold from dB to linear amplitude
    silence_threshold = 10 ** (silence_threshold_db / 20)

    # Calculate RMS energy in short windows
    window_size = int(0.05 * sample_rate)  # 50ms windows
    hop_size = window_size // 2

    # Pad audio to ensure we can process all samples
    padded = np.pad(mono, (0, window_size))

    # Calculate RMS for each window
    num_windows = (len(padded) - window_size) // hop_size + 1
    rms = np.zeros(num_windows)
    for i in range(num_windows):
        start = i * hop_size
        window = padded[start : start + window_size]
        rms[i] = np.sqrt(np.mean(window**2))

    # Find silent regions (RMS below threshold)
    is_silent = rms < silence_threshold

    # Convert min_silence_duration to windows
    min_silence_windows = int(min_silence_duration * sample_rate / hop_size)

    # Find contiguous silent regions
    silent_regions: list[tuple[int, int]] = []
    in_silence = False
    silence_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not silent and in_silence:
            if i - silence_start >= min_silence_windows:
                silent_regions.append((silence_start, i))
            in_silence = False

    # Handle trailing silence
    if in_silence and len(is_silent) - silence_start >= min_silence_windows:
        silent_regions.append((silence_start, len(is_silent)))

    # Convert window indices to sample indices
    def window_to_sample(w: int) -> int:
        return w * hop_size

    # Build segment boundaries (non-silent regions)
    segments: list[tuple[int, int]] = []
    prev_end = 0

    for silence_start_w, silence_end_w in silent_regions:
        segment_start = prev_end
        segment_end = window_to_sample(silence_start_w)

        if segment_end > segment_start:
            segments.append((segment_start, segment_end))

        prev_end = window_to_sample(silence_end_w)

    # Add final segment if there's audio after last silence
    if prev_end < len(mono):
        segments.append((prev_end, len(mono)))

    # If no silence detected, use entire file as one segment
    if not segments:
        segments = [(0, len(mono))]

    # Filter out segments that are too short
    min_samples = int(min_segment_duration * sample_rate)
    segments = [(s, e) for s, e in segments if e - s >= min_samples]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write each segment to a separate file
    result: list[tuple[Path, float, float]] = []
    for i, (start_sample, end_sample) in enumerate(segments):
        segment_audio = audio[start_sample:end_sample]
        segment_path = output_dir / f"segment_{i:03d}.wav"

        sf.write(str(segment_path), segment_audio, sample_rate)

        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        result.append((segment_path, start_time, end_time))

    return result


def _detect_non_silent_ranges_from_array(
    audio: np.ndarray,
    sample_rate: int,
    min_silence_duration: float = 0.5,
    silence_threshold_db: float = -40.0,
    time_offset: float = 0.0,
) -> list[tuple[float, float]]:
    """Detect non-silent time ranges in an audio array.

    Args:
        audio: Audio samples as numpy array (mono or stereo).
        sample_rate: Sample rate of the audio.
        min_silence_duration: Minimum silence duration (seconds) to count as a gap.
        silence_threshold_db: Audio level below this (in dB) is considered silence.
        time_offset: Offset to add to all returned times (for timeline positioning).

    Returns:
        List of (start_seconds, end_seconds) tuples for non-silent regions.
    """
    # Convert to mono for silence detection
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio

    # Handle empty audio
    if len(mono) == 0:
        return []

    # Convert threshold from dB to linear amplitude
    silence_threshold = 10 ** (silence_threshold_db / 20)

    # Calculate RMS energy in short windows
    window_size = int(0.05 * sample_rate)  # 50ms windows
    hop_size = window_size // 2

    # Ensure window_size is at least 1
    if window_size < 1:
        return [(time_offset, time_offset + len(mono) / sample_rate)]

    # Pad audio to ensure we can process all samples
    padded = np.pad(mono, (0, window_size))

    # Calculate RMS for each window
    num_windows = (len(padded) - window_size) // hop_size + 1
    rms = np.zeros(num_windows)
    for i in range(num_windows):
        start = i * hop_size
        window = padded[start : start + window_size]
        rms[i] = np.sqrt(np.mean(window**2))

    # Find silent regions (RMS below threshold)
    is_silent = rms < silence_threshold

    # Convert min_silence_duration to windows
    min_silence_windows = int(min_silence_duration * sample_rate / hop_size)

    # Find contiguous silent regions
    silent_regions: list[tuple[int, int]] = []
    in_silence = False
    silence_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not silent and in_silence:
            if i - silence_start >= min_silence_windows:
                silent_regions.append((silence_start, i))
            in_silence = False

    # Handle trailing silence
    if in_silence and len(is_silent) - silence_start >= min_silence_windows:
        silent_regions.append((silence_start, len(is_silent)))

    # Convert window indices to sample indices
    def window_to_sample(w: int) -> int:
        return w * hop_size

    # Build segment boundaries (non-silent regions)
    segments: list[tuple[int, int]] = []
    prev_end = 0

    for silence_start_w, silence_end_w in silent_regions:
        segment_start = prev_end
        segment_end = window_to_sample(silence_start_w)

        if segment_end > segment_start:
            segments.append((segment_start, segment_end))

        prev_end = window_to_sample(silence_end_w)

    # Add final segment if there's audio after last silence
    if prev_end < len(mono):
        segments.append((prev_end, len(mono)))

    # If no silence detected, use entire file as one segment
    if not segments:
        segments = [(0, len(mono))]

    # Convert to seconds with offset (no filtering - all non-silent regions are valid)
    result: list[tuple[float, float]] = []
    for start_sample, end_sample in segments:
        result.append(
            (time_offset + start_sample / sample_rate, time_offset + end_sample / sample_rate)
        )

    return result


def detect_non_silent_ranges(
    audio_path: Path,
    min_silence_duration: float = 0.5,
    silence_threshold_db: float = -40.0,
) -> list[tuple[float, float]]:
    """Detect non-silent time ranges in an audio file.

    Wrapper around _detect_non_silent_ranges_from_array that reads from a file.

    Args:
        audio_path: Path to the audio file to analyze.
        min_silence_duration: Minimum silence duration (seconds) to count as a gap.
        silence_threshold_db: Audio level below this (in dB) is considered silence.

    Returns:
        List of (start_seconds, end_seconds) tuples for non-silent regions.
    """
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    return _detect_non_silent_ranges_from_array(
        audio,
        sample_rate,
        min_silence_duration=min_silence_duration,
        silence_threshold_db=silence_threshold_db,
        time_offset=0.0,
    )


# ---------------------------------------------------------------------------
# Interactive Track Selection (Phase 2)
# ---------------------------------------------------------------------------


def prompt_track_selection(
    track_names: list[str],
    auto_select_single: bool = True,
    default_tracks: list[str] | None = None,
) -> list[str]:
    """Interactively prompt user to select vocal tracks.

    Args:
        track_names: List of detected vocal track names.
        auto_select_single: If True and only one track, skip prompt.
        default_tracks: Tracks to pre-select. If None, all tracks are pre-selected.

    Returns:
        List of selected track names.

    Raises:
        click.Abort: If user cancels selection.
        RuntimeError: If questionary is not installed.
    """
    if not track_names:
        return []

    # Single track: use it automatically
    if len(track_names) == 1 and auto_select_single:
        return track_names

    # Determine which tracks to pre-select
    if default_tracks is not None:
        default_set = {t.lower() for t in default_tracks}
        choices = [
            questionary.Choice(name, checked=name.lower() in default_set)
            for name in track_names
        ]
    else:
        # No defaults specified: pre-select all
        choices = [questionary.Choice(name, checked=True) for name in track_names]

    # Multiple tracks: interactive selection
    selected = questionary.checkbox(
        "Select vocal tracks to include for lyrics alignment:",
        choices=choices,
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
    config_tracks: list[str] | None = None,
    category_overrides: dict[str, str] | None = None,
) -> list[AudioClipRef]:
    """Select vocal tracks for alignment.

    Handles selection modes:
    1. Explicit tracks specified via CLI options (highest priority, no prompt)
    2. use_all=True: Use all detected vocal tracks without prompting
    3. Interactive: Prompt user when multiple tracks found (if TTY available),
       using config_tracks as defaults if available
    4. Non-TTY with config: Use config tracks
    5. Non-TTY without config: Use all and warn

    Args:
        all_clips: All audio clips from ALS.
        explicit_tracks: Tracks specified via --vocal-track options.
        use_all: If True, use all detected vocal tracks without prompting.
        config_tracks: Tracks from .muse config file (used as defaults in prompt).
        category_overrides: Category overrides from .muse config file.
            Tracks categorized as "vocals" are included in detection.

    Returns:
        Filtered list of clips from selected tracks, sorted by start time.
    """
    # Explicit selection via CLI (highest priority)
    if explicit_tracks:
        explicit_lower = {t.lower() for t in explicit_tracks}
        return sorted(
            [c for c in all_clips if c.track_name.lower() in explicit_lower],
            key=lambda c: c.start_beats,
        )

    # Find all potential vocal tracks
    # Include tracks detected by keywords OR categorized as "vocals" in overrides
    def is_vocal(clip: AudioClipRef) -> bool:
        if is_vocal_track(clip.track_name):
            return True
        return bool(
            category_overrides and category_overrides.get(clip.track_name) == "vocals"
        )

    vocal_clips = [c for c in all_clips if is_vocal(c)]
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
    # Use config_tracks as defaults if available
    if sys.stdin.isatty():
        selected_names = prompt_track_selection(track_names, default_tracks=config_tracks)
        selected_set = set(selected_names)
        return sorted(
            [c for c in vocal_clips if c.track_name in selected_set],
            key=lambda c: c.start_beats,
        )

    # Non-TTY: use config if available, otherwise use all and warn
    if config_tracks:
        config_lower = {t.lower() for t in config_tracks}
        matching_clips = [c for c in vocal_clips if c.track_name.lower() in config_lower]
        if matching_clips:
            return sorted(matching_clips, key=lambda c: c.start_beats)

    click.echo(
        f"Multiple vocal tracks found: {', '.join(track_names)}. "
        "Using all. Use --vocal-track to select specific tracks.",
        err=True,
    )
    return sorted(vocal_clips, key=lambda c: c.start_beats)


def select_vocal_tracks_with_config(
    all_clips: list[AudioClipRef],
    als_path: Path,
    explicit_tracks: tuple[str, ...] | None,
    use_all: bool,
) -> tuple[list[AudioClipRef], list[str]]:
    """Select vocal tracks with config file integration.

    This is the main entry point for vocal track selection. It:
    1. Loads config from .muse file if it exists
    2. Selects tracks using the priority order in select_vocal_tracks
    3. If interactive selection occurred, saves the selection to config

    Args:
        all_clips: All audio clips from ALS.
        als_path: Path to the ALS file (for config file).
        explicit_tracks: Tracks specified via --vocal-track options.
        use_all: If True, use all detected vocal tracks without prompting.

    Returns:
        Tuple of (selected clips, selected track names).
    """
    # Load existing config
    config = load_config(als_path)
    config_tracks = config.vocal_tracks if config else None
    category_overrides = config.category_overrides if config else None

    # Get initial selection
    selected_clips = select_vocal_tracks(
        all_clips, explicit_tracks, use_all, config_tracks, category_overrides
    )

    # Get unique track names from selected clips
    selected_names = get_unique_vocal_track_names(selected_clips)

    # Save selection to config if interactive selection occurred
    # (no explicit CLI tracks and TTY available)
    if not explicit_tracks and not use_all and sys.stdin.isatty() and selected_names:
        # Update or create config with vocal tracks
        if config:
            config.vocal_tracks = selected_names
        else:
            config = MuseConfig(vocal_tracks=selected_names)
        save_config(als_path, config)

    return selected_clips, selected_names
