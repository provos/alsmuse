"""Lyrics alignment and transcription using stable-ts.

This module provides functionality to align or transcribe lyrics using
the stable-ts (stable_whisper) library built on OpenAI's Whisper model.

Features:
    - **Forced alignment**: Align provided lyrics text to audio, producing
      word-level timestamps that match when each word is sung.
    - **ASR transcription**: Automatically transcribe lyrics from vocal audio
      when no lyrics file is provided.
    - **Hallucination filtering**: Remove words/segments that fall outside
      known valid audio ranges (e.g., during silent gaps in the vocal track).
    - **Segment-based line breaking**: Use Whisper's natural phrase boundaries
      for transcription, splitting only long segments for readability.

The stable-ts dependency is optional - import errors are handled gracefully
with helpful error messages. Install with: pip install 'alsmuse[align]'

Key Functions:
    - align_lyrics: Force-align lyrics text to audio with word-level timing.
    - transcribe_lyrics: Transcribe lyrics from audio using Whisper ASR.
    - segments_to_lines: Convert transcribed segments to displayable lines.
    - filter_to_valid_ranges: Remove hallucinated words outside vocal regions.
    - filter_segments_to_valid_ranges: Remove hallucinated segments.
    - words_to_lines: Reconstruct original line structure from word timestamps.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unicodedata
from pathlib import Path

from .audio import split_audio_on_silence
from .exceptions import AlignmentError
from .models import TimedLine, TimedSegment, TimedWord


def get_compute_device() -> str:
    """Detect the optimal compute device for model inference.

    PyTorch-based models (including Whisper) default to CPU if CUDA is not found.
    They do NOT automatically use MPS on macOS. This function explicitly detects
    and selects the optimal device for best performance.

    Returns:
        "cuda" - NVIDIA GPU (Linux/Windows)
        "mps"  - Apple Metal Performance Shaders (macOS)
        "cpu"  - Fallback for all other systems

    Raises:
        AlignmentError: If PyTorch (torch) is not installed.
    """
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as e:
        raise AlignmentError(
            "PyTorch (torch) is not installed. "
            "Install alignment dependencies with: pip install 'alsmuse[align]'"
        ) from e

    if torch.cuda.is_available():
        return "cuda"
    # CRITICAL: Explicit check for Metal Performance Shaders on macOS
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def filter_to_valid_ranges(
    words: list[TimedWord],
    valid_ranges: list[tuple[float, float]],
) -> list[TimedWord]:
    """Remove words that fall outside known audio regions.

    A word is kept if its midpoint falls within any valid range.
    This filters out Whisper hallucinations during silent gaps.

    Args:
        words: All words from alignment.
        valid_ranges: Time ranges where real audio exists, as
            (start_seconds, end_seconds) tuples.

    Returns:
        Filtered list with hallucinations removed.
    """
    if not valid_ranges:
        return words

    def is_in_valid_range(word: TimedWord) -> bool:
        midpoint = (word.start + word.end) / 2
        return any(start <= midpoint <= end for start, end in valid_ranges)

    return [w for w in words if is_in_valid_range(w)]


def _normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Converts to lowercase, removes punctuation, and normalizes unicode.

    Args:
        text: Text to normalize.

    Returns:
        Normalized text suitable for comparison.
    """
    # Normalize unicode (e.g., compose accented characters)
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def _tokenize(text: str) -> list[str]:
    """Split text into normalized word tokens.

    Args:
        text: Text to tokenize.

    Returns:
        List of normalized word tokens.
    """
    normalized = _normalize_text(text)
    return normalized.split()


def words_to_lines(
    words: list[TimedWord],
    original_lines: list[str],
) -> list[TimedLine]:
    """Reconstruct line structure from word timestamps.

    Matches timed words back to original lyric lines to preserve
    the user's line breaks.

    Algorithm:
    1. Normalize both word sequences (lowercase, strip punctuation)
    2. For each original line, consume matching words
    3. Line timing = first word start to last word end
    4. If words run out, remaining lines get empty timing

    Args:
        words: Timed words from alignment.
        original_lines: Original lyric lines from user's file.

    Returns:
        List of TimedLine with word-level detail.
    """
    if not words:
        # No words, return empty lines
        return [
            TimedLine(text=line, start=0.0, end=0.0, words=())
            for line in original_lines
            if line.strip()
        ]

    result: list[TimedLine] = []
    word_idx = 0

    for line in original_lines:
        if not line.strip():
            # Skip empty lines
            continue

        # Tokenize the original line
        line_tokens = _tokenize(line)
        if not line_tokens:
            # Line had only punctuation, skip
            continue

        # Collect words that match this line's tokens
        line_words: list[TimedWord] = []
        tokens_matched = 0

        while word_idx < len(words) and tokens_matched < len(line_tokens):
            word = words[word_idx]
            word_normalized = _normalize_text(word.text)

            if not word_normalized:
                # Empty word (punctuation only), skip
                word_idx += 1
                continue

            # Check if this word matches the expected token
            expected_token = line_tokens[tokens_matched]

            if word_normalized == expected_token:
                line_words.append(word)
                tokens_matched += 1
                word_idx += 1
            else:
                # Words may be split differently by the model
                # Try partial matching: does the word start with expected token?
                if word_normalized.startswith(expected_token):
                    line_words.append(word)
                    tokens_matched += 1
                    word_idx += 1
                elif expected_token.startswith(word_normalized):
                    # Token is longer than word, consume word and continue
                    line_words.append(word)
                    word_idx += 1
                else:
                    # No match, break and move to next line
                    break

        # Create the timed line
        if line_words:
            result.append(
                TimedLine(
                    text=line,
                    start=line_words[0].start,
                    end=line_words[-1].end,
                    words=tuple(line_words),
                )
            )
        else:
            # No words matched for this line
            result.append(
                TimedLine(
                    text=line,
                    start=0.0,
                    end=0.0,
                    words=(),
                )
            )

    return result


def align_lyrics(
    audio_path: Path,
    lyrics_text: str,
    valid_ranges: list[tuple[float, float]],
    language: str = "en",
    model_size: str = "base",
) -> list[TimedWord]:
    """Force-align lyrics to audio, filtering hallucinations.

    Uses stable-ts for alignment, then filters results to only include
    words that fall within known valid audio ranges.

    Args:
        audio_path: Path to combined vocal audio.
        lyrics_text: Plain text lyrics to align.
        valid_ranges: List of (start, end) tuples where real audio exists.
        language: Language code for alignment model (default: "en").
        model_size: Whisper model size ("tiny", "base", "small", "medium").

    Returns:
        List of TimedWord with hallucinations removed.

    Raises:
        AlignmentError: If stable-ts is not installed or alignment fails.
    """
    try:
        import stable_whisper  # type: ignore[import-not-found,import-untyped]  # noqa: PLC0415
        import torch  # noqa: F401,PLC0415  # type: ignore[import-not-found,import-untyped]
    except ImportError as e:
        raise AlignmentError(
            "stable-ts is not installed. "
            "Install alignment dependencies with: pip install 'alsmuse[align]'"
        ) from e

    # Select optimal compute device (GPU/MPS/CPU)
    # Note: torch import above ensures PyTorch is available for get_compute_device()
    device = get_compute_device()

    # MPS doesn't support float64 which stable-ts requires internally.
    # Fall back to CPU for reliable operation on macOS.
    # CPU is still fast for alignment (uses all cores).
    if device == "mps":
        device = "cpu"

    try:
        model = stable_whisper.load_model(model_size, device=device)
    except Exception as e:
        raise AlignmentError(f"Failed to load Whisper model '{model_size}': {e}") from e

    try:
        result = model.align(str(audio_path), lyrics_text, language=language)
    except Exception as e:
        raise AlignmentError(f"Alignment failed: {e}") from e

    # Extract all words with timestamps
    all_words: list[TimedWord] = []
    for segment in result.segments:
        for word in segment.words:
            word_text = word.word.strip()
            if word_text:  # Skip empty words
                all_words.append(
                    TimedWord(
                        text=word_text,
                        start=word.start,
                        end=word.end,
                    )
                )

    # Filter to valid ranges only
    filtered_words = filter_to_valid_ranges(all_words, valid_ranges)

    return filtered_words


def filter_segments_to_valid_ranges(
    segments: list[TimedSegment],
    valid_ranges: list[tuple[float, float]],
) -> list[TimedSegment]:
    """Remove segments that fall outside known audio regions.

    A segment is kept if its midpoint falls within any valid range.
    This filters out Whisper hallucinations during silent gaps.

    Args:
        segments: All segments from transcription.
        valid_ranges: Time ranges where real audio exists, as
            (start_seconds, end_seconds) tuples.

    Returns:
        Filtered list with hallucinated segments removed.
    """
    if not valid_ranges:
        return segments

    def is_in_valid_range(segment: TimedSegment) -> bool:
        midpoint = (segment.start + segment.end) / 2
        return any(start <= midpoint <= end for start, end in valid_ranges)

    return [s for s in segments if is_in_valid_range(s)]


def _transcribe_with_mlx_whisper(
    audio_path: Path,
    language: str,
    model_size: str,
    time_offset: float = 0.0,
) -> list[TimedSegment]:
    """Transcribe using mlx-whisper (fast on Apple Silicon).

    Args:
        audio_path: Path to audio file.
        language: Language code.
        model_size: Whisper model size.
        time_offset: Offset to add to all timestamps (for segment-based transcription).

    Returns:
        List of TimedSegment from transcription.

    Raises:
        ImportError: If mlx-whisper is not installed.
        AlignmentError: If transcription fails.
    """
    import mlx_whisper  # type: ignore[import-not-found,import-untyped]  # noqa: PLC0415

    # Map model size to HuggingFace repo
    model_map = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large": "mlx-community/whisper-large-v3-mlx",
    }
    model_repo = model_map.get(model_size, model_map["base"])

    try:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model_repo,
            language=language,
            word_timestamps=True,
            verbose=True,  # Show transcription progress
        )
    except Exception as e:
        raise AlignmentError(f"MLX transcription failed: {e}") from e

    # Build segments from mlx-whisper result, applying time offset
    all_segments: list[TimedSegment] = []
    for segment in result.get("segments", []):
        words: list[TimedWord] = []
        for word_info in segment.get("words", []):
            word_text = word_info.get("word", "").strip()
            if word_text:
                words.append(
                    TimedWord(
                        text=word_text,
                        start=word_info.get("start", 0.0) + time_offset,
                        end=word_info.get("end", 0.0) + time_offset,
                    )
                )

        if words:
            segment_text = segment.get("text", "").strip()
            all_segments.append(
                TimedSegment(
                    text=segment_text,
                    start=segment.get("start", 0.0) + time_offset,
                    end=segment.get("end", 0.0) + time_offset,
                    words=tuple(words),
                )
            )

    return all_segments


def _transcribe_with_stable_ts(
    audio_path: Path,
    language: str,
    model_size: str,
    time_offset: float = 0.0,
) -> list[TimedSegment]:
    """Transcribe using stable-ts (cross-platform, slower on Mac).

    Args:
        audio_path: Path to audio file.
        language: Language code.
        model_size: Whisper model size.
        time_offset: Offset to add to all timestamps (for segment-based transcription).

    Returns:
        List of TimedSegment from transcription.

    Raises:
        ImportError: If stable-ts is not installed.
        AlignmentError: If transcription fails.
    """
    import stable_whisper  # type: ignore[import-not-found,import-untyped]  # noqa: PLC0415
    import torch  # noqa: F401,PLC0415  # type: ignore[import-not-found,import-untyped]

    # Select optimal compute device (GPU/MPS/CPU)
    device = get_compute_device()

    # MPS doesn't support float64 which stable-ts requires internally.
    if device == "mps":
        device = "cpu"

    try:
        model = stable_whisper.load_model(model_size, device=device)
    except Exception as e:
        raise AlignmentError(f"Failed to load Whisper model '{model_size}': {e}") from e

    try:
        result = model.transcribe(str(audio_path), language=language)
    except Exception as e:
        raise AlignmentError(f"Transcription failed: {e}") from e

    # Build segments from Whisper result, applying time offset
    all_segments: list[TimedSegment] = []
    for segment in result.segments:
        words: list[TimedWord] = []
        for word in segment.words:
            word_text = word.word.strip()
            if word_text:
                words.append(
                    TimedWord(
                        text=word_text,
                        start=word.start + time_offset,
                        end=word.end + time_offset,
                    )
                )

        if words:
            segment_text = segment.text.strip()
            all_segments.append(
                TimedSegment(
                    text=segment_text,
                    start=segment.start + time_offset,
                    end=segment.end + time_offset,
                    words=tuple(words),
                )
            )

    return all_segments


def _transcribe_single_segment(
    audio_path: Path,
    language: str,
    model_size: str,
    time_offset: float,
) -> list[TimedSegment]:
    """Transcribe a single audio segment, trying mlx-whisper first.

    Args:
        audio_path: Path to audio file.
        language: Language code.
        model_size: Whisper model size.
        time_offset: Offset to add to all timestamps.

    Returns:
        List of TimedSegment from transcription.

    Raises:
        AlignmentError: If no transcription backend is available.
    """
    try:
        return _transcribe_with_mlx_whisper(audio_path, language, model_size, time_offset)
    except ImportError:
        try:
            return _transcribe_with_stable_ts(audio_path, language, model_size, time_offset)
        except ImportError as e:
            raise AlignmentError(
                "No transcription backend available. Install one of:\n"
                "  - pip install 'alsmuse[align-mlx]'  (fast, Apple Silicon only)\n"
                "  - pip install 'alsmuse[align]'      (cross-platform)"
            ) from e


def transcribe_lyrics(
    audio_path: Path,
    valid_ranges: list[tuple[float, float]],  # Not used with segment-based approach
    language: str = "en",
    model_size: str = "base",
    segment_dir: Path | None = None,
) -> tuple[list[TimedSegment], str]:
    """Transcribe lyrics from audio using Whisper ASR.

    Splits audio on silence and transcribes each segment separately to
    avoid Whisper hallucinations during silent sections. Uses mlx-whisper
    on Apple Silicon for fast transcription, falling back to stable-ts.

    Args:
        audio_path: Path to combined vocal audio.
        valid_ranges: Not used (kept for API compatibility). Silence detection
            now handles filtering.
        language: Language code for transcription model.
        model_size: Whisper model size.
        segment_dir: Optional directory to save audio segments (for debugging).
            If None, uses a temp directory that is cleaned up afterward.

    Returns:
        Tuple of:
        - List of TimedSegment preserving Whisper's phrase boundaries.
        - Raw transcription text (for saving to file).

    Raises:
        AlignmentError: If no transcription backend is available.
    """
    # Create temp directory for segments if not provided
    cleanup_dir = segment_dir is None
    if segment_dir is None:
        segment_dir = Path(tempfile.mkdtemp(prefix="alsmuse_segments_"))

    try:
        # Split audio on silence - returns list of (path, start_time, end_time)
        audio_segments = split_audio_on_silence(
            audio_path,
            segment_dir,
            min_silence_duration=1.0,  # 1 second of silence to split
            silence_threshold_db=-40.0,  # -40dB threshold
            min_segment_duration=0.5,  # Keep segments >= 0.5s
        )

        if not audio_segments:
            return [], ""

        # Transcribe each segment separately
        all_segments: list[TimedSegment] = []
        for seg_path, start_time, _end_time in audio_segments:
            # Transcribe this segment with time offset
            seg_result = _transcribe_single_segment(
                seg_path, language, model_size, time_offset=start_time
            )
            all_segments.extend(seg_result)

        # Sort by start time (should already be sorted, but ensure)
        all_segments.sort(key=lambda s: s.start)

        # Build raw text for saving
        raw_text = "\n".join(s.text for s in all_segments)

        return all_segments, raw_text

    finally:
        # Cleanup temp directory
        if cleanup_dir and segment_dir.exists():
            shutil.rmtree(segment_dir, ignore_errors=True)


def _split_segment_at_punctuation(
    segment: TimedSegment,
    max_words_per_line: int,
) -> list[TimedLine]:
    """Split a segment that exceeds max_words_per_line.

    Prefers splitting at punctuation boundaries, otherwise splits
    at approximately the midpoint.

    Args:
        segment: Segment to split.
        max_words_per_line: Maximum words per resulting line.

    Returns:
        List of TimedLine objects.
    """
    words = list(segment.words)
    lines: list[TimedLine] = []

    while len(words) > max_words_per_line:
        # Find punctuation boundaries in the first max_words_per_line words
        best_split_idx = -1
        for i in range(min(max_words_per_line, len(words)) - 1, 0, -1):
            word_text = words[i].text
            # Check if word ends with punctuation that suggests a phrase break
            if word_text and word_text[-1] in ".,;:!?":
                best_split_idx = i + 1
                break

        # If no punctuation found, split at midpoint
        if best_split_idx == -1:
            best_split_idx = min(max_words_per_line, len(words) // 2)
            if best_split_idx == 0:
                best_split_idx = 1  # At least take one word

        # Create line from first portion
        line_words = tuple(words[:best_split_idx])
        line_text = " ".join(w.text for w in line_words)
        lines.append(
            TimedLine(
                text=line_text,
                start=line_words[0].start,
                end=line_words[-1].end,
                words=line_words,
            )
        )

        # Continue with remaining words
        words = words[best_split_idx:]

    # Handle remaining words
    if words:
        line_words = tuple(words)
        line_text = " ".join(w.text for w in line_words)
        lines.append(
            TimedLine(
                text=line_text,
                start=line_words[0].start,
                end=line_words[-1].end,
                words=line_words,
            )
        )

    return lines


def segments_to_lines(
    segments: list[TimedSegment],
    max_words_per_line: int = 15,
) -> list[TimedLine]:
    """Convert transcribed segments to lines, splitting long segments.

    Uses Whisper's segment boundaries as the primary line breaks.
    Only applies heuristic splitting to segments that exceed max_words_per_line.

    Algorithm:
    1. For each segment:
       a. If word count <= max_words_per_line: create single line from segment
       b. If word count > max_words_per_line: split on punctuation or mid-point
    2. Preserve word-level timing from original segment

    Args:
        segments: Transcribed segments from Whisper.
        max_words_per_line: Only split segments exceeding this word count.

    Returns:
        List of TimedLine, typically one per segment unless segment was split.
    """
    lines: list[TimedLine] = []

    for segment in segments:
        word_count = len(segment.words)

        if word_count == 0:
            # Empty segment, skip
            continue

        if word_count <= max_words_per_line:
            # Segment fits in one line, create directly
            lines.append(
                TimedLine(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    words=segment.words,
                )
            )
        else:
            # Segment too long, split it
            split_lines = _split_segment_at_punctuation(segment, max_words_per_line)
            lines.extend(split_lines)

    return lines
