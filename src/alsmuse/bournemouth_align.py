"""Word-level alignment refinement using Bournemouth Forced Aligner.

This module provides phoneme-level forced alignment using the Bournemouth
aligner, which uses CTC Viterbi decoding to align text to audio. Unlike
stable-ts which stretches word boundaries to fill silence gaps, Bournemouth
provides accurate phoneme boundaries that can handle repeated lyrics.

Features:
    - Sequential gap-filling: processes entire transcript chronologically
    - Fixes lines with missing timestamps (0.0/0.0) from stable-ts
    - Handles repeated phrases (e.g., "Make My Heart Bleed" x3)
    - Confidence tracking to detect potential lyrics mismatches
    - Works with segments up to 30s (optimal: 10s)

Requirements:
    - Optional dependency: pip install 'alsmuse[align-bournemouth]'
    - System dependency: espeak-ng (brew install espeak-ng on macOS)
    - Environment variable: PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib

Key Functions:
    - refine_alignment_with_bournemouth: Refine word timestamps using sequential gap-filling.
    - is_bournemouth_available: Check if Bournemouth is installed and configured.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Callable
from ctypes.util import find_library
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf  # type: ignore[import-untyped]

from .exceptions import AlignmentError
from .models import TimedLine, TimedWord

if TYPE_CHECKING:
    import torch

# Type alias for progress callback: (current_line, total_lines, line_text) -> None
ProgressCallback = Callable[[int, int, str], None]

# Maximum segment duration for Bournemouth (10s is optimal, >10s gets slow)
MAX_SEGMENT_DURATION = 10.0


def _find_espeak_library() -> str | None:
    """Auto-detect the espeak-ng library path.

    Tries multiple detection methods in order of reliability:
    1. Check if PHONEMIZER_ESPEAK_LIBRARY env var is already set and valid
    2. Use ctypes.util.find_library (cross-platform)
    3. On macOS: query Homebrew for install location
    4. On Linux: parse ldconfig output
    5. Check common installation paths

    Returns:
        Path to libespeak-ng library, or None if not found.
    """
    # 1. Check existing environment variable
    env_path = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. Try ctypes.util.find_library (cross-platform)
    lib_name = find_library("espeak-ng")
    if lib_name:
        # On some systems this returns full path, on others just the name
        if Path(lib_name).exists():
            return lib_name
        # On macOS/Linux, find_library may return just "libespeak-ng.1.dylib"
        # Check common lib directories
        for lib_dir in ["/usr/lib", "/usr/local/lib", "/opt/homebrew/lib"]:
            full_path = Path(lib_dir) / lib_name
            if full_path.exists():
                return str(full_path)

    system = platform.system()

    # 3. macOS: Query Homebrew
    if system == "Darwin" and shutil.which("brew"):
        try:
            result = subprocess.run(
                ["brew", "--prefix", "espeak-ng"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                prefix = result.stdout.strip()
                lib_path = Path(prefix) / "lib" / "libespeak-ng.dylib"
                if lib_path.exists():
                    return str(lib_path)
        except (subprocess.TimeoutExpired, OSError):
            pass

    # 4. Linux: Parse ldconfig
    if system == "Linux" and shutil.which("ldconfig"):
        try:
            result = subprocess.run(
                ["ldconfig", "-p"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "libespeak-ng" in line and "=>" in line:
                        # Format: "libespeak-ng.so.1 (libc6,x86-64) => /usr/lib/..."
                        path = line.split("=>")[-1].strip()
                        if Path(path).exists():
                            return path
        except (subprocess.TimeoutExpired, OSError):
            pass

    # 5. Fall back to common paths
    common_paths = [
        # macOS ARM (Homebrew)
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib",
        # macOS Intel (Homebrew)
        "/usr/local/lib/libespeak-ng.dylib",
        "/usr/local/opt/espeak-ng/lib/libespeak-ng.dylib",
        # Linux (Debian/Ubuntu)
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
        # Linux (generic)
        "/usr/lib/libespeak-ng.so.1",
        "/usr/lib/libespeak-ng.so",
    ]
    for path in common_paths:
        if Path(path).exists():
            return path

    return None


def _get_bournemouth_device() -> str:
    """Get the optimal device for Bournemouth aligner.

    Note: MPS (Metal) doesn't work reliably with Bournemouth due to
    tensor type mismatches between model weights and input. Fall back
    to CPU which is still fast (0.2s per 10s audio).

    Returns:
        Device string: "cuda" or "cpu".
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    # MPS doesn't work with Bournemouth - model weights stay on CPU
    # while input moves to MPS, causing tensor type mismatch
    return "cpu"


def is_bournemouth_available() -> bool:
    """Check if Bournemouth aligner is available.

    Verifies both the Python package and espeak-ng are installed.
    Auto-detects and sets the PHONEMIZER_ESPEAK_LIBRARY environment
    variable if espeak-ng is found.

    Returns:
        True if Bournemouth can be used, False otherwise.
    """
    try:
        from bournemouth_aligner import (  # type: ignore[import-untyped]  # noqa: PLC0415, F401, I001
            PhonemeTimestampAligner,
        )
    except ImportError:
        return False

    # Auto-detect espeak-ng library
    espeak_lib = _find_espeak_library()
    if espeak_lib:
        # Set env var so phonemizer can find it
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib
        return True

    return False


def _load_and_resample_audio(
    audio_path: Path,
    target_sample_rate: int = 16000,
) -> tuple["torch.Tensor", int]:  # noqa: UP037
    """Load audio and resample to target sample rate.

    Uses soundfile to avoid torchcodec/FFmpeg issues.

    Args:
        audio_path: Path to audio file.
        target_sample_rate: Target sample rate (Bournemouth requires 16kHz).

    Returns:
        Tuple of (audio_tensor, sample_rate).
    """
    import torch  # noqa: PLC0415
    import torchaudio.transforms as T  # type: ignore[import-untyped]  # noqa: PLC0415

    # Load with soundfile (avoids torchcodec issues)
    audio_np, sample_rate = sf.read(str(audio_path), dtype="float32")

    # Convert to mono if stereo
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    # Convert to torch tensor with batch dimension [1, samples]
    audio_wav = torch.from_numpy(audio_np).unsqueeze(0)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        audio_wav = resampler(audio_wav)
        sample_rate = target_sample_rate

    return audio_wav, sample_rate


def _extract_audio_segment(
    audio_wav: "torch.Tensor",  # noqa: UP037
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> "torch.Tensor":  # noqa: UP037
    """Extract a segment from the audio tensor.

    Args:
        audio_wav: Full audio tensor [1, samples].
        sample_rate: Audio sample rate.
        start_sec: Segment start in seconds.
        end_sec: Segment end in seconds.

    Returns:
        Audio segment tensor [1, segment_samples].
    """
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(audio_wav.shape[1], end_sample)

    return audio_wav[:, start_sample:end_sample]


def _align_segment_with_bournemouth(
    aligner,  # PhonemeTimestampAligner
    audio_segment: "torch.Tensor",  # noqa: UP037
    text: str,
    segment_start: float,
) -> tuple[list[TimedWord], float, list[tuple[str, float]]]:
    """Align text to audio segment using Bournemouth.

    Args:
        aligner: Initialized PhonemeTimestampAligner.
        audio_segment: Audio tensor for the segment.
        text: Text to align.
        segment_start: Start time of segment (for offset).

    Returns:
        Tuple of:
            - List of TimedWord with absolute timestamps
            - Average confidence score (0.0-1.0)
            - List of (word, confidence) for low-confidence words
    """
    try:
        result = aligner.process_sentence(
            text,
            audio_segment,
            do_groups=False,
            debug=False,
        )
    except Exception as e:
        raise AlignmentError(f"Bournemouth alignment failed: {e}") from e

    # Extract words from result
    words: list[TimedWord] = []
    confidences: list[float] = []
    low_confidence_words: list[tuple[str, float]] = []

    if isinstance(result, dict) and "segments" in result and result["segments"]:
        seg_result = result["segments"][0]
        words_ts = seg_result.get("words_ts", [])

        for w in words_ts:
            word_text = w.get("word", "")
            # Skip silence markers
            if word_text.startswith("<") or not word_text.strip():
                continue

            # Convert from segment-relative milliseconds to absolute seconds
            start_ms = w.get("start_ms", 0)
            end_ms = w.get("end_ms", 0)
            confidence = w.get("confidence", 1.0)

            words.append(
                TimedWord(
                    text=word_text,
                    start=segment_start + start_ms / 1000,
                    end=segment_start + end_ms / 1000,
                )
            )
            confidences.append(confidence)

            # Track low confidence words (threshold: 0.5)
            if confidence < 0.5:
                low_confidence_words.append((word_text, confidence))

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return words, avg_confidence, low_confidence_words


def _find_next_valid_timestamp(lines: list[TimedLine], start_idx: int) -> float | None:
    """Find the next line with a valid timestamp after start_idx.

    Args:
        lines: List of TimedLine objects.
        start_idx: Index to start searching from (exclusive).

    Returns:
        Start timestamp of next valid line, or None if no valid line found.
    """
    for i in range(start_idx + 1, len(lines)):
        if lines[i].start > 0.0 or lines[i].end > 0.0:
            return lines[i].start
    return None


def _find_valid_range_for_time(
    timestamp: float,
    valid_ranges: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """Find the valid range that contains or is nearest after the given timestamp.

    Args:
        timestamp: Time in seconds.
        valid_ranges: List of (start, end) tuples where audio exists.

    Returns:
        The valid range containing or nearest after the timestamp, or None.
    """
    for start, end in valid_ranges:
        if start <= timestamp <= end:
            return (start, end)
        if start > timestamp:
            # Next valid range after timestamp
            return (start, end)
    return None


def _constrain_to_valid_ranges(
    seg_start: float,
    seg_end: float,
    valid_ranges: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """Constrain a search window to valid audio regions.

    If the segment spans multiple valid ranges, returns a window from
    the start of the first overlapping range to the end of the last
    overlapping range. This allows searching across multiple vocal clips.

    Args:
        seg_start: Desired start time.
        seg_end: Desired end time.
        valid_ranges: List of (start, end) tuples where audio exists (must be sorted).

    Returns:
        Constrained (start, end) tuple spanning all overlapping ranges, or None if no overlap.
    """
    first_start: float | None = None
    last_end: float | None = None

    for range_start, range_end in valid_ranges:
        # Check for overlap with segment
        if seg_start < range_end and seg_end > range_start:
            # Track the constrained boundaries
            if first_start is None:
                first_start = max(seg_start, range_start)
            last_end = min(seg_end, range_end)

    if first_start is not None and last_end is not None and last_end > first_start:
        return (first_start, last_end)
    return None


def refine_alignment_with_bournemouth(
    audio_path: Path,
    lines: list[TimedLine],
    language: str = "en-us",
    confidence_threshold: float = 0.5,
    progress_callback: ProgressCallback | None = None,
    valid_ranges: list[tuple[float, float]] | None = None,
) -> tuple[list[TimedLine], list[str]]:
    """Refine word timestamps using Bournemouth forced aligner with sequential gap-filling.

    Processes the entire transcript chronologically, using the end of each aligned
    line as the starting point for the next. This approach:
    - Fixes lines with missing timestamps (0.0/0.0) that stable-ts failed to align
    - Handles repeated phrases by processing them in sequence
    - Provides more accurate word boundaries through phoneme-level alignment
    - Skips silent regions when valid_ranges is provided

    For lines with valid timestamps, uses those as hints but still refines timing.
    For lines with 0.0/0.0 timestamps, searches from the current position to the
    next valid timestamp (or end of audio), limited to 10 seconds max.

    Args:
        audio_path: Path to the vocal audio file.
        lines: List of TimedLine with approximate timestamps.
        language: Language code for phonemization (default: "en-us").
        confidence_threshold: Warn about words below this confidence (default: 0.5).
        progress_callback: Optional callback for progress updates.
            Called with (current_line, total_lines, line_text) after each line.
        valid_ranges: Optional list of (start, end) tuples indicating where
            audio actually exists. If provided, search windows are constrained
            to these ranges, skipping silent gaps.

    Returns:
        Tuple of:
            - List of TimedLine with refined word timestamps
            - List of warning messages about low-confidence alignments

    Raises:
        AlignmentError: If Bournemouth is not available or alignment fails.
    """
    if not is_bournemouth_available():
        raise AlignmentError(
            "Bournemouth aligner is not available. "
            "Install with: pip install 'alsmuse[align-bournemouth]' "
            "and install espeak-ng: brew install espeak-ng (macOS)"
        )

    from bournemouth_aligner import (  # type: ignore[import-untyped]  # noqa: PLC0415, I001
        PhonemeTimestampAligner,
    )

    # Load and resample audio
    audio_wav, sample_rate = _load_and_resample_audio(audio_path)
    audio_duration = audio_wav.shape[1] / sample_rate

    # Sort valid_ranges if provided
    sorted_ranges = sorted(valid_ranges, key=lambda r: r[0]) if valid_ranges else None

    # Initialize aligner - 10s is optimal for speed and accuracy
    device = _get_bournemouth_device()
    aligner = PhonemeTimestampAligner(
        preset=language,
        duration_max=int(MAX_SEGMENT_DURATION),
        device=device,
    )

    refined_lines: list[TimedLine] = []
    warnings: list[str] = []
    total_lines = len(lines)

    # Track current position in audio (where we expect the next line to start)
    current_position = 0.0

    for i, line in enumerate(lines):
        # Report progress
        if progress_callback:
            progress_callback(i + 1, total_lines, line.text[:40])

        # Skip empty lines
        if not line.text.strip():
            refined_lines.append(line)
            continue

        # Determine search window for this line
        has_valid_timestamp = line.start > 0.0 or line.end > 0.0

        padding = 0.5  # 500ms padding
        if has_valid_timestamp:
            # Line has timestamps from stable-ts - use them as hints with padding
            seg_start = max(0, line.start - padding)
            seg_end = min(audio_duration, line.end + padding)
        else:
            # Line has no valid timestamp (0.0/0.0) - this is what we want to fix!
            # Search from current position, limited to MAX_SEGMENT_DURATION
            seg_start = current_position

            # Find next valid timestamp as upper bound
            next_valid = _find_next_valid_timestamp(lines, i)
            if next_valid is not None:
                seg_end = min(next_valid + padding, seg_start + MAX_SEGMENT_DURATION)
            else:
                seg_end = min(audio_duration, seg_start + MAX_SEGMENT_DURATION)

        # Clamp to audio bounds
        seg_start = max(0, seg_start)
        seg_end = min(audio_duration, seg_end)

        # Constrain to valid audio ranges if provided
        if sorted_ranges:
            constrained = _constrain_to_valid_ranges(seg_start, seg_end, sorted_ranges)
            if constrained is None:
                # No valid audio in this window - find next valid range
                next_range = _find_valid_range_for_time(seg_start, sorted_ranges)
                if next_range is not None:
                    # Jump to next valid range
                    seg_start = next_range[0]
                    seg_end = min(next_range[1], seg_start + MAX_SEGMENT_DURATION)
                else:
                    # No more valid ranges - skip this line
                    refined_lines.append(line)
                    continue
            else:
                seg_start, seg_end = constrained

        # Hard limit on segment duration for speed
        if seg_end - seg_start > MAX_SEGMENT_DURATION:
            seg_end = seg_start + MAX_SEGMENT_DURATION

        # Skip very short segments
        if seg_end - seg_start < 0.1:
            refined_lines.append(line)
            continue

        audio_segment = _extract_audio_segment(audio_wav, sample_rate, seg_start, seg_end)

        try:
            # Align this line's text within the search window
            refined_words, avg_confidence, low_conf_words = _align_segment_with_bournemouth(
                aligner,
                audio_segment,
                line.text,
                seg_start,
            )

            if refined_words:
                # Create refined line with new timestamps
                refined_line = TimedLine(
                    text=line.text,
                    start=refined_words[0].start,
                    end=refined_words[-1].end,
                    words=tuple(refined_words),
                )
                refined_lines.append(refined_line)

                # Update current position for next line
                current_position = refined_line.end

                # Track low confidence warnings
                if low_conf_words:
                    word_list = ", ".join(f"'{w}' ({c:.2f})" for w, c in low_conf_words)
                    warnings.append(
                        f"Low confidence alignment in line '{line.text[:30]}...': {word_list}"
                    )

                # Warn if overall line confidence is very low
                if avg_confidence < confidence_threshold:
                    warnings.append(
                        f"Line may have lyrics mismatch (confidence: {avg_confidence:.2f}): "
                        f"'{line.text[:50]}...'"
                    )
            else:
                # No words aligned, keep original
                refined_lines.append(line)
                # Still advance position based on original if valid
                if has_valid_timestamp:
                    current_position = line.end

        except AlignmentError as e:
            # Alignment failed for this line, keep original
            refined_lines.append(line)
            warnings.append(f"Alignment failed for '{line.text[:30]}...': {e}")
            # Advance position if we have valid timestamps
            if has_valid_timestamp:
                current_position = line.end

    return refined_lines, warnings


def align_repeated_section(
    audio_path: Path,
    text: str,
    start_sec: float,
    end_sec: float,
    language: str = "en-us",
) -> list[TimedWord]:
    """Align a section with repeated phrases using Bournemouth.

    This function is designed for sections where stable-ts fails due to
    repeated lyrics. It takes a time range and the full text (including
    repetitions) and uses Bournemouth to align all instances.

    Note: Bournemouth works best with segments under 10 seconds.
    Segments 10-30s work but are slower. Segments over 30s may fail
    or produce poor results due to excessive Viterbi possibilities.

    Example:
        # Align "Make My Heart Bleed" repeated 3 times
        words = align_repeated_section(
            audio_path,
            "Make My Heart Bleed. Make My Heart Bleed. Make My Heart Bleed.",
            start_sec=196.0,
            end_sec=206.0,  # Keep segments under 10s for best results
        )

    Args:
        audio_path: Path to the vocal audio file.
        text: Full text including all repetitions.
        start_sec: Start of the section in seconds.
        end_sec: End of the section in seconds.
        language: Language code for phonemization.

    Returns:
        List of TimedWord with absolute timestamps.

    Raises:
        AlignmentError: If alignment fails or segment is too long.
    """
    if not is_bournemouth_available():
        raise AlignmentError(
            "Bournemouth aligner is not available. "
            "Install with: pip install 'alsmuse[align-bournemouth]'"
        )

    segment_duration = end_sec - start_sec
    if segment_duration > 30:
        raise AlignmentError(
            f"Segment duration ({segment_duration:.1f}s) exceeds 30s maximum. "
            "Bournemouth works best with segments under 10 seconds. "
            "Please split into smaller segments."
        )

    from bournemouth_aligner import (  # type: ignore[import-untyped]  # noqa: PLC0415, I001
        PhonemeTimestampAligner,
    )

    # Load and resample audio
    audio_wav, sample_rate = _load_and_resample_audio(audio_path)

    # Extract the section
    audio_segment = _extract_audio_segment(audio_wav, sample_rate, start_sec, end_sec)

    # Initialize aligner with duration_max based on segment length
    # Optimal: 10s, acceptable: up to 30s
    device = _get_bournemouth_device()
    duration_max = min(int(segment_duration) + 1, 30)
    aligner = PhonemeTimestampAligner(
        preset=language,
        duration_max=duration_max,
        device=device,
    )

    # Align and return just the words (ignore confidence info for this API)
    words, _, _ = _align_segment_with_bournemouth(aligner, audio_segment, text, start_sec)
    return words
