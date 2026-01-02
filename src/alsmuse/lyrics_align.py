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
    - **Bournemouth refinement**: Optional phoneme-level alignment refinement
      for repeated lyrics that stable-ts struggles with.

For faster transcription on Apple Silicon, install the optional mlx-whisper
backend: pip install 'alsmuse[align-mlx]'

For improved alignment of repeated lyrics, install Bournemouth aligner:
pip install 'alsmuse[align-bournemouth]' and brew install espeak-ng (macOS)

Key Functions:
    - align_lyrics: Force-align lyrics text to audio with word-level timing.
    - transcribe_lyrics: Transcribe lyrics from audio using Whisper ASR.
    - segments_to_lines: Convert transcribed segments to displayable lines.
    - clip_words_to_valid_ranges: Filter hallucinations and clip stretched boundaries.
    - filter_segments_to_valid_ranges: Remove hallucinated segments.
    - words_to_lines: Reconstruct original line structure from word timestamps.
    - refine_lines_with_bournemouth: Refine word timestamps using Bournemouth.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unicodedata
from collections.abc import Callable
from pathlib import Path

import stable_whisper  # type: ignore[import-untyped]

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


def clip_words_to_valid_ranges(
    words: list[TimedWord],
    valid_ranges: list[tuple[float, float]],
) -> list[TimedWord]:
    """Filter words and clip their boundaries to valid audio regions.

    stable-ts forced alignment stretches word boundaries to fill silence
    gaps. A word like "Deep" sung at 52.5s might get assigned start=50.2s
    to fill the silence after the previous word. This function:

    1. Finds the valid range containing the word's end time (which is more
       accurate than start for stretched words)
    2. Clips word boundaries to that valid range
    3. Filters out words that don't overlap any valid range

    The end time is used for matching because stable-ts stretches words
    backwards (adjusting start earlier) but keeps end times accurate.

    Args:
        words: All words from alignment.
        valid_ranges: Time ranges where real audio exists, as
            (start_seconds, end_seconds) tuples.

    Returns:
        Filtered list with word boundaries clipped to valid audio regions.
    """
    if not valid_ranges:
        return words

    # Sort ranges by start time for efficient lookup
    sorted_ranges = sorted(valid_ranges, key=lambda r: r[0])

    def find_range_for_word(word: TimedWord) -> tuple[float, float] | None:
        """Find the valid range that contains the word's end time.

        For stretched words, the end time is accurate (when the word actually
        ends being sung), while start may be stretched backwards into silence.
        """
        # First try: find range containing the end time
        for start, end in sorted_ranges:
            if start <= word.end <= end:
                return (start, end)

        # Fallback: find range containing the midpoint (for normal words)
        midpoint = (word.start + word.end) / 2
        for start, end in sorted_ranges:
            if start <= midpoint <= end:
                return (start, end)

        return None

    result: list[TimedWord] = []
    for word in words:
        # Find the valid range for this word
        containing_range = find_range_for_word(word)
        if containing_range is None:
            # Word doesn't overlap any valid range, skip it
            continue

        range_start, range_end = containing_range

        # Clip word boundaries to the valid range
        new_start = max(word.start, range_start)
        new_end = min(word.end, range_end)

        # Only create new TimedWord if boundaries changed
        if new_start != word.start or new_end != word.end:
            result.append(
                TimedWord(
                    text=word.text,
                    start=new_start,
                    end=new_end,
                )
            )
        else:
            result.append(word)

    return result


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


def validate_timed_lines(
    timed_lines: list[TimedLine],
    min_gap_seconds: float = 0.3,
) -> list[str]:
    """Validate timed lines and return warning messages for issues.

    Checks for:
    1. Lines with start=0.0 that couldn't be aligned
    2. Lines with impossibly close timing (within min_gap_seconds)

    Args:
        timed_lines: List of timed lines to validate.
        min_gap_seconds: Minimum expected gap between line starts.
            Lines closer than this are flagged as suspicious.

    Returns:
        List of warning messages (empty if no issues found).
    """
    warnings: list[str] = []

    # Check for unaligned lines (start=0.0)
    unaligned = [line for line in timed_lines if line.start == 0.0 and line.text.strip()]
    if unaligned:
        warnings.append(f"Warning: {len(unaligned)} line(s) could not be aligned (time=0:00):")
        for line in unaligned[:5]:  # Show first 5
            truncated = line.text[:50] + "..." if len(line.text) > 50 else line.text
            warnings.append(f"  - {truncated}")
        if len(unaligned) > 5:
            warnings.append(f"  ... and {len(unaligned) - 5} more")

    # Check for impossibly close timing
    # Build list of (timestamp, line_text) for non-zero lines, sorted by time
    timed = [(line.start, line.text) for line in timed_lines if line.start > 0.0]
    timed.sort(key=lambda x: x[0])

    close_groups: list[list[tuple[float, str]]] = []
    current_group: list[tuple[float, str]] = []

    for start, text in timed:
        if not current_group:
            current_group.append((start, text))
        else:
            prev_start = current_group[-1][0]
            if start - prev_start < min_gap_seconds:
                current_group.append((start, text))
            else:
                if len(current_group) > 1:
                    close_groups.append(current_group)
                current_group = [(start, text)]

    # Don't forget the last group
    if len(current_group) > 1:
        close_groups.append(current_group)

    if close_groups:
        total_affected = sum(len(g) for g in close_groups)
        warnings.append(
            f"Warning: {total_affected} lines have suspiciously close timing "
            f"(within {min_gap_seconds}s):"
        )
        for group in close_groups[:3]:  # Show first 3 groups
            for start, text in group:
                mins, secs = divmod(start, 60)
                truncated = text[:40] + "..." if len(text) > 40 else text
                warnings.append(f"  [{int(mins):02d}:{secs:05.2f}] {truncated}")
            if group != close_groups[-1] and close_groups.index(group) < 2:
                warnings.append("")  # Blank line between groups
        if len(close_groups) > 3:
            remaining = sum(len(g) for g in close_groups[3:])
            warnings.append(f"  ... and {remaining} more lines in other groups")

    return warnings


def align_lyrics(
    audio_path: Path,
    lyrics_text: str,
    valid_ranges: list[tuple[float, float]],
    language: str = "en",
    model_size: str = "base",
) -> list[TimedWord]:
    """Force-align lyrics to audio, filtering hallucinations.

    Uses stable-ts for alignment, then:
    1. Filters out words that fall outside known valid audio ranges
    2. Clips word boundaries to valid ranges to fix stretched timestamps

    stable-ts stretches word boundaries to fill silence gaps. A word sung
    at 52.5s might get start=50.2s to fill the gap after the previous word.
    This function clips such stretched boundaries to when audio actually exists.

    Args:
        audio_path: Path to combined vocal audio.
        lyrics_text: Plain text lyrics to align.
        valid_ranges: List of (start, end) tuples where real audio exists.
        language: Language code for alignment model (default: "en").
        model_size: Whisper model size ("tiny", "base", "small", "medium").

    Returns:
        List of TimedWord with hallucinations removed and boundaries clipped.

    Raises:
        AlignmentError: If alignment fails.
    """
    # Select optimal compute device (GPU/MPS/CPU)
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

    # Filter hallucinations and clip stretched word boundaries to valid ranges
    filtered_words = clip_words_to_valid_ranges(all_words, valid_ranges)

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
        AlignmentError: If transcription fails.
    """
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
    """
    try:
        # Try mlx-whisper first (faster on Apple Silicon)
        return _transcribe_with_mlx_whisper(audio_path, language, model_size, time_offset)
    except ImportError:
        # Fall back to stable-ts (always available)
        return _transcribe_with_stable_ts(audio_path, language, model_size, time_offset)


def transcribe_lyrics(
    audio_path: Path,
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


def refine_lines_with_bournemouth(
    audio_path: Path,
    lines: list[TimedLine],
    language: str = "en-us",
    confidence_threshold: float = 0.5,
    progress_callback: Callable[[int, int, str], None] | None = None,
    valid_ranges: list[tuple[float, float]] | None = None,
) -> tuple[list[TimedLine], list[str]]:
    """Refine word timestamps using Bournemouth forced aligner with sequential gap-filling.

    Processes the entire transcript chronologically using Bournemouth's phoneme-level
    CTC Viterbi alignment. This approach:
    - Fixes lines with missing timestamps (0.0/0.0) that stable-ts failed to align
    - Handles repeated phrases by processing them in sequence
    - Provides more accurate word boundaries
    - Detects potential lyrics mismatches via confidence scores
    - Skips silent regions when valid_ranges is provided

    Processing is fast (~0.2s per 10s of audio on CPU) so all lines are refined.

    Args:
        audio_path: Path to the vocal audio file.
        lines: List of TimedLine from initial alignment.
        language: Language code for phonemization (default: "en-us").
        confidence_threshold: Warn about lines below this confidence (default: 0.5).
        progress_callback: Optional callback for progress updates.
            Called with (current_line, total_lines, line_text) after each line.
        valid_ranges: Optional list of (start, end) tuples indicating where
            audio actually exists. If provided, search windows are constrained
            to these ranges, skipping silent gaps.

    Returns:
        Tuple of:
        - List of TimedLine with refined timestamps.
        - List of info/warning messages about the refinement.

    Note:
        If Bournemouth is not available, returns the original lines unchanged.
    """
    # Import here to avoid circular imports and make it optional
    try:
        from .bournemouth_align import (  # noqa: PLC0415
            is_bournemouth_available,
            refine_alignment_with_bournemouth,
        )
    except ImportError:
        return lines, ["Bournemouth aligner module not available"]

    if not is_bournemouth_available():
        return lines, ["Bournemouth aligner not installed or espeak-ng not found"]

    messages: list[str] = []
    messages.append(f"Refining {len(lines)} lines with Bournemouth (sequential gap-filling)...")

    try:
        refined_lines, warnings = refine_alignment_with_bournemouth(
            audio_path,
            lines,
            language=language,
            confidence_threshold=confidence_threshold,
            progress_callback=progress_callback,
            valid_ranges=valid_ranges,
        )

        # Count how many lines were actually modified
        modified_count = sum(
            1
            for orig, refined in zip(lines, refined_lines, strict=False)
            if orig.start != refined.start or orig.end != refined.end
        )

        # Count how many previously unaligned lines now have timestamps
        fixed_unaligned = sum(
            1
            for orig, refined in zip(lines, refined_lines, strict=False)
            if (orig.start == 0.0 and orig.end == 0.0)
            and (refined.start > 0.0 or refined.end > 0.0)
        )

        messages.append(f"Refined {modified_count} lines ({fixed_unaligned} previously unaligned)")

        # Add any warnings from the alignment
        messages.extend(warnings)

        return refined_lines, messages

    except Exception as e:
        messages.append(f"Bournemouth refinement failed: {e}")
        return lines, messages
