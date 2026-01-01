"""Lyrics alignment using forced alignment with stable-ts.

This module provides functionality to align lyrics text to audio using
the stable-ts (stable_whisper) library. It includes hallucination filtering
to remove words that fall outside known valid audio ranges.

The stable-ts dependency is optional - import errors are handled gracefully
with helpful error messages.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .exceptions import AlignmentError
from .models import TimedLine, TimedWord


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
        import torch  # type: ignore[import-not-found]
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
        import stable_whisper  # type: ignore[import-not-found]
    except ImportError as e:
        raise AlignmentError(
            "stable-ts is not installed. "
            "Install alignment dependencies with: pip install 'alsmuse[align]'"
        ) from e

    # Select optimal compute device (GPU/MPS/CPU)
    device = get_compute_device()

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
