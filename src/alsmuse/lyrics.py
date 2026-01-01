"""Lyrics parsing and distribution for ALSmuse.

This module provides functionality for parsing lyrics files with section
headers and distributing lyrics across phrases within each section.

Lyrics Distribution Modes:
    - **Heuristic distribution**: Distributes lyrics evenly across phrases
      within each section based on section structure.
    - **Time-based distribution**: Uses precise timestamps (from forced
      alignment or pre-timestamped files) to place lyrics in the correct
      phrases based on when they are sung.

Supported Lyrics Formats:
    - **Plain text**: Optional [SECTION] headers, uses heuristic distribution.
    - **LRC format**: Standard karaoke format [mm:ss.xx]text, line-level timing.
    - **Simple timed**: m:ss.xx text at line start, easy to create manually.
    - **Enhanced LRC**: [mm:ss.xx]<mm:ss.xx>word <mm:ss.xx>word, word-level timing.

When timestamped lyrics (LRC, simple timed, or enhanced LRC) are provided,
the timestamps are used directly without requiring forced alignment. This
is useful when you already have pre-timed lyrics from karaoke software,
other transcription tools, or manual annotation.

Key Functions:
    - detect_lyrics_format: Auto-detect format from file content.
    - parse_lyrics_file_auto: Parse with auto-detection, returning appropriate type.
    - distribute_lyrics: Heuristic distribution for plain text lyrics.
    - distribute_timed_lyrics: Time-based distribution for timestamped lyrics.
"""

import logging
import re
from pathlib import Path

from .models import LyricsFormat, Phrase, TimedLine, TimedSegment, TimedWord

logger = logging.getLogger(__name__)

# Regex patterns for format detection and parsing

# LRC timestamp pattern: [mm:ss.xx] or [mm:ss.xxx]
# Matches: [00:12.34], [01:23.456], etc.
LRC_TIMESTAMP_PATTERN = re.compile(r"^\[(\d{2}):(\d{2})\.(\d{2,3})\]")

# LRC metadata pattern: [tag:value]
# Matches: [ar:Artist], [ti:Title], [al:Album], etc.
LRC_METADATA_PATTERN = re.compile(r"^\[([a-z]{2,}):.*\]$", re.IGNORECASE)

# Simple timed pattern: m:ss.xx or mm:ss.xx at line start followed by space
# Matches: 0:12.34 text, 01:23.45 text, 12:34.56 text
SIMPLE_TIMED_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})\.(\d{2,3})\s+(.+)$")

# Enhanced LRC word timestamp: <mm:ss.xx> or <mm:ss.xxx>
# Matches: <00:12.34>, <01:23.456>, etc.
ENHANCED_WORD_PATTERN = re.compile(r"<(\d{2}):(\d{2})\.(\d{2,3})>")


def _parse_timestamp_components(
    minutes: str, seconds: str, centis: str
) -> float:
    """Convert timestamp components to seconds.

    Args:
        minutes: Minutes as string (e.g., "01", "12").
        seconds: Seconds as string (e.g., "23", "59").
        centis: Centiseconds or milliseconds as string (e.g., "34", "567").

    Returns:
        Time in seconds as float.
    """
    mins = int(minutes)
    secs = int(seconds)
    # Handle both centiseconds (2 digits) and milliseconds (3 digits)
    frac = int(centis) / 100.0 if len(centis) == 2 else int(centis) / 1000.0
    return mins * 60.0 + secs + frac


def detect_lyrics_format(content: str) -> LyricsFormat:
    """Detect the format of lyrics content.

    Analyzes the content to determine which lyrics format is being used.
    The detection is based on characteristic patterns of each format.

    Detection order:
    1. Enhanced LRC: Contains both [mm:ss.xx] and <mm:ss.xx> patterns
    2. Standard LRC: Lines start with [mm:ss.xx] pattern
    3. Simple timed: Lines start with m:ss.xx or mm:ss.xx pattern
    4. Plain: Default when no timestamp patterns are detected

    Args:
        content: Raw lyrics file content.

    Returns:
        LyricsFormat enum indicating the detected format.

    Examples:
        >>> detect_lyrics_format("[00:12.34]Hello world")
        LyricsFormat.LRC
        >>> detect_lyrics_format("0:12.34 Hello world")
        LyricsFormat.SIMPLE_TIMED
        >>> detect_lyrics_format("[00:12.34]<00:12.34>Hello <00:12.80>world")
        LyricsFormat.LRC_ENHANCED
        >>> detect_lyrics_format("Hello world")
        LyricsFormat.PLAIN
    """
    lines = content.strip().split("\n")

    has_lrc_timestamp = False
    has_word_timestamp = False
    has_simple_timestamp = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip LRC metadata tags like [ar:Artist], [ti:Title]
        if LRC_METADATA_PATTERN.match(line):
            continue

        # Check for LRC timestamp at start of line
        if LRC_TIMESTAMP_PATTERN.match(line):
            has_lrc_timestamp = True
            # Check for enhanced LRC word timestamps within the line
            if ENHANCED_WORD_PATTERN.search(line):
                has_word_timestamp = True

        # Check for simple timed format
        elif SIMPLE_TIMED_PATTERN.match(line):
            has_simple_timestamp = True

    # Determine format based on detected patterns
    if has_lrc_timestamp and has_word_timestamp:
        return LyricsFormat.LRC_ENHANCED
    elif has_lrc_timestamp:
        return LyricsFormat.LRC
    elif has_simple_timestamp:
        return LyricsFormat.SIMPLE_TIMED
    else:
        return LyricsFormat.PLAIN


def parse_lrc_lyrics(content: str) -> list[TimedLine]:
    """Parse LRC format lyrics into TimedLine objects.

    Handles standard LRC format with line-level timestamps.
    Multiple timestamps on the same line (for repeated lyrics)
    generate separate TimedLine objects.

    Metadata tags like [ar:Artist], [ti:Title], etc. are ignored.
    Empty timestamps (e.g., [00:00.00] with no text) are skipped.

    Args:
        content: LRC format lyrics content.

    Returns:
        List of TimedLine with timestamps in seconds, sorted by start time.

    Examples:
        >>> lines = parse_lrc_lyrics("[00:12.34]First line\\n[00:15.67]Second line")
        >>> lines[0].text
        'First line'
        >>> lines[0].start
        12.34
    """
    result: list[TimedLine] = []

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Skip metadata tags
        if LRC_METADATA_PATTERN.match(line):
            continue

        # Extract all timestamps from the line
        timestamps: list[float] = []
        remaining = line

        while True:
            match = LRC_TIMESTAMP_PATTERN.match(remaining)
            if not match:
                break
            timestamp = _parse_timestamp_components(
                match.group(1), match.group(2), match.group(3)
            )
            timestamps.append(timestamp)
            remaining = remaining[match.end():]

        # Get the text after all timestamps
        text = remaining.strip()

        # Skip empty lines (timestamps with no text)
        if not text:
            continue

        # Create a TimedLine for each timestamp (handles repeated lyrics)
        for ts in timestamps:
            # For line-level timing, we don't have word timing
            # Set end to start (will be refined when distributed to phrases)
            result.append(
                TimedLine(
                    text=text,
                    start=ts,
                    end=ts,  # End time unknown for line-level LRC
                    words=(),  # No word-level timing
                )
            )

    # Sort by start time
    result.sort(key=lambda line: line.start)
    return result


def parse_simple_timed_lyrics(content: str) -> list[TimedLine]:
    """Parse simple timed format lyrics.

    Parses lyrics where each line starts with a timestamp in the format
    m:ss.xx or mm:ss.xx followed by a space and the lyrics text.

    Args:
        content: Simple timed lyrics content.

    Returns:
        List of TimedLine with timestamps in seconds, sorted by start time.

    Examples:
        >>> lines = parse_simple_timed_lyrics("0:12.34 First line\\n0:15.67 Second line")
        >>> lines[0].text
        'First line'
        >>> lines[0].start
        12.34
    """
    result: list[TimedLine] = []

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        match = SIMPLE_TIMED_PATTERN.match(line)
        if not match:
            # Line doesn't match simple timed format, skip with warning
            logger.warning("Skipping line without valid timestamp: %s", line[:50])
            continue

        timestamp = _parse_timestamp_components(
            match.group(1), match.group(2), match.group(3)
        )
        text = match.group(4).strip()

        if not text:
            continue

        result.append(
            TimedLine(
                text=text,
                start=timestamp,
                end=timestamp,  # End time unknown for line-level timing
                words=(),  # No word-level timing
            )
        )

    # Sort by start time
    result.sort(key=lambda line: line.start)
    return result


def parse_enhanced_lrc(content: str) -> list[TimedLine]:
    """Parse enhanced LRC with word-level timestamps.

    Parses LRC format that includes word-level timing using inline
    <mm:ss.xx> markers before each word.

    Format: [mm:ss.xx]<mm:ss.xx>word <mm:ss.xx>word ...

    Args:
        content: Enhanced LRC content.

    Returns:
        List of TimedLine with word-level TimedWord objects.

    Examples:
        >>> lines = parse_enhanced_lrc("[00:12.34]<00:12.34>Hello <00:12.80>world")
        >>> lines[0].text
        'Hello world'
        >>> lines[0].words[0].text
        'Hello'
        >>> lines[0].words[0].start
        12.34
    """
    result: list[TimedLine] = []

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Skip metadata tags
        if LRC_METADATA_PATTERN.match(line):
            continue

        # Extract line timestamp
        line_match = LRC_TIMESTAMP_PATTERN.match(line)
        if not line_match:
            continue

        line_start = _parse_timestamp_components(
            line_match.group(1), line_match.group(2), line_match.group(3)
        )
        remaining = line[line_match.end():]

        # Parse word timestamps and words
        words: list[TimedWord] = []
        word_timestamps: list[tuple[float, str]] = []

        # Split by word timestamp markers
        parts = ENHANCED_WORD_PATTERN.split(remaining)

        # parts alternates: [text_before, min, sec, frac, text_after, min, sec, frac, ...]
        # First element is text before first timestamp (usually empty)
        # Then groups of 4: (min, sec, frac, text_after_timestamp)

        i = 0
        while i < len(parts):
            if i == 0:
                # Text before first timestamp (skip if empty)
                if parts[i].strip():
                    # Text without timestamp at start - use line start time
                    word_timestamps.append((line_start, parts[i].strip()))
                i += 1
            elif i + 3 < len(parts):
                # We have a full timestamp group
                ts = _parse_timestamp_components(parts[i], parts[i + 1], parts[i + 2])
                text_part = parts[i + 3].strip() if i + 3 < len(parts) else ""
                if text_part:
                    # The text after timestamp may contain multiple words if there's
                    # no timestamp for intermediate words. We take all text until
                    # the next timestamp as belonging to this word/phrase.
                    word_timestamps.append((ts, text_part))
                i += 4
            else:
                break

        # Create TimedWord objects
        for idx, (ts, word_text) in enumerate(word_timestamps):
            # For end time, use next word's start or estimate for last word
            end_ts = (
                word_timestamps[idx + 1][0]
                if idx + 1 < len(word_timestamps)
                else ts + 0.5
            )

            words.append(
                TimedWord(
                    text=word_text,
                    start=ts,
                    end=end_ts,
                )
            )

        # Skip if no words parsed
        if not words:
            continue

        # Reconstruct full text
        full_text = " ".join(w.text for w in words)

        result.append(
            TimedLine(
                text=full_text,
                start=words[0].start,
                end=words[-1].end,
                words=tuple(words),
            )
        )

    # Sort by start time
    result.sort(key=lambda line: line.start)
    return result


def _format_timestamp_lrc(seconds: float) -> str:
    """Format seconds as LRC timestamp [mm:ss.xx]."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02d}:{secs:05.2f}]"


def format_segments_as_lrc(segments: list[TimedSegment]) -> str:
    """Format transcribed segments as LRC lyrics.

    Converts TimedSegment objects to LRC format with timestamps,
    allowing the transcription to be saved and reloaded without
    re-transcribing.

    Args:
        segments: List of TimedSegment objects from transcription.

    Returns:
        LRC-formatted string with timestamps.

    Example:
        >>> from alsmuse.models import TimedSegment, TimedWord
        >>> segments = [
        ...     TimedSegment(
        ...         text="Hello world",
        ...         start=1.5,
        ...         end=3.0,
        ...         words=(TimedWord("Hello", 1.5, 2.0), TimedWord("world", 2.5, 3.0)),
        ...     ),
        ... ]
        >>> print(format_segments_as_lrc(segments))
        [00:01.50]Hello world
    """
    lines: list[str] = []
    for segment in segments:
        timestamp = _format_timestamp_lrc(segment.start)
        lines.append(f"{timestamp}{segment.text}")

    return "\n".join(lines)


def format_timed_lines_as_lrc(timed_lines: list[TimedLine]) -> str:
    """Format aligned lyrics as LRC.

    Converts TimedLine objects to LRC format with timestamps,
    allowing aligned lyrics to be saved and reloaded without
    re-aligning.

    Args:
        timed_lines: List of TimedLine objects from alignment.

    Returns:
        LRC-formatted string with timestamps.

    Example:
        >>> lines = [
        ...     TimedLine(text="Hello world", start=1.5, end=3.0, words=()),
        ... ]
        >>> print(format_timed_lines_as_lrc(lines))
        [00:01.50]Hello world
    """
    lines: list[str] = []
    for timed_line in timed_lines:
        timestamp = _format_timestamp_lrc(timed_line.start)
        lines.append(f"{timestamp}{timed_line.text}")

    return "\n".join(lines)


def parse_lyrics_file_auto(
    path: Path,
) -> tuple[list[TimedLine] | None, dict[str, list[str]] | None]:
    """Parse lyrics file with auto-format detection.

    Automatically detects the lyrics format and parses accordingly.
    Returns timed lines for timestamped formats, or section lyrics
    for plain text format.

    Args:
        path: Path to lyrics file.

    Returns:
        Tuple of:
        - List of TimedLine if timestamps detected, None otherwise.
        - Section lyrics dict if plain text with sections, None otherwise.

        At least one will be non-None. If timestamps are detected,
        section headers are ignored and timed lines are returned.

    Raises:
        FileNotFoundError: If the lyrics file does not exist.

    Examples:
        >>> timed, sections = parse_lyrics_file_auto(Path("song.lrc"))
        >>> if timed is not None:
        ...     print("Got timestamped lyrics")
        >>> if sections is not None:
        ...     print("Got section-based lyrics")
    """
    content = path.read_text(encoding="utf-8")

    if not content.strip():
        # Empty file - return empty section dict
        return None, {}

    fmt = detect_lyrics_format(content)

    if fmt == LyricsFormat.LRC_ENHANCED:
        timed_lines = parse_enhanced_lrc(content)
        return timed_lines, None

    elif fmt == LyricsFormat.LRC:
        timed_lines = parse_lrc_lyrics(content)
        return timed_lines, None

    elif fmt == LyricsFormat.SIMPLE_TIMED:
        timed_lines = parse_simple_timed_lyrics(content)
        return timed_lines, None

    else:  # LyricsFormat.PLAIN
        section_lyrics = parse_lyrics_file(path)
        return None, section_lyrics


def parse_lyrics_file(path: Path) -> dict[str, list[str]]:
    """Parse a lyrics file with section headers.

    Reads a lyrics file where sections are marked with bracketed headers
    like [VERSE1], [CHORUS], etc. Lines following a header belong to that
    section until the next header is encountered.

    Args:
        path: Path to the lyrics file.

    Returns:
        Dictionary mapping section names (uppercase) to lists of lyric lines.

    Example:
        Given a file containing:
        ```
        [VERSE1]
        Walking down the street
        Feeling the beat

        [CHORUS]
        This is the chorus line
        ```

        Returns:
        ```
        {
            "VERSE1": ["Walking down the street", "Feeling the beat"],
            "CHORUS": ["This is the chorus line"],
        }
        ```

    Raises:
        FileNotFoundError: If the lyrics file does not exist.
    """
    lyrics: dict[str, list[str]] = {}
    current_section: str | None = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Check for section header
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].upper()
                lyrics[current_section] = []
            elif line and current_section is not None:
                lyrics[current_section].append(line)

    return lyrics


def distribute_lyrics(
    phrases: list[Phrase],
    section_lyrics: dict[str, list[str]],
) -> list[Phrase]:
    """Distribute lyrics across phrases within each section.

    Processes phrases in section groups and distributes lyrics evenly
    across the phrases in each section using a line-count heuristic.

    Args:
        phrases: List of phrases to annotate with lyrics.
        section_lyrics: Dictionary mapping section names to lyric lines.

    Returns:
        New list of phrases with lyric attributes populated.

    Example:
        If a section has 4 phrases and 2 lyric lines, lines go in phrases 1 and 3.
        If a section has 4 phrases and 4 lyric lines, one line goes in each phrase.
    """
    if not phrases:
        return []

    result: list[Phrase] = []
    current_section: str | None = None
    section_phrases: list[Phrase] = []

    for phrase in phrases:
        if phrase.is_section_start:
            # Process previous section if it exists
            if section_phrases and current_section is not None:
                lyrics = section_lyrics.get(current_section, [])
                result.extend(_apply_lyrics(section_phrases, lyrics))

            # Start new section
            current_section = phrase.section_name
            section_phrases = [phrase]
        else:
            section_phrases.append(phrase)

    # Process final section
    if section_phrases and current_section is not None:
        lyrics = section_lyrics.get(current_section, [])
        result.extend(_apply_lyrics(section_phrases, lyrics))

    return result


def _apply_lyrics(phrases: list[Phrase], lyrics: list[str]) -> list[Phrase]:
    """Apply lyrics to phrases with even distribution.

    Distributes lyric lines evenly across the given phrases. If there are
    more phrases than lyrics, lyrics are spaced out. If there are more
    lyrics than phrases, extra lyrics are skipped.

    Args:
        phrases: List of phrases to annotate.
        lyrics: List of lyric lines to distribute.

    Returns:
        New list of phrases with lyric attributes set.
    """
    if not lyrics:
        return phrases

    if not phrases:
        return []

    # Calculate step size for even distribution
    step = max(1, len(phrases) // len(lyrics))
    result: list[Phrase] = []
    lyric_idx = 0

    for i, phrase in enumerate(phrases):
        lyric = ""
        if i % step == 0 and lyric_idx < len(lyrics):
            lyric = lyrics[lyric_idx]
            lyric_idx += 1

        result.append(
            Phrase(
                start_beats=phrase.start_beats,
                end_beats=phrase.end_beats,
                section_name=phrase.section_name,
                is_section_start=phrase.is_section_start,
                events=phrase.events,
                lyric=lyric,
            )
        )

    return result


def distribute_timed_lyrics(
    phrases: list[Phrase],
    timed_lines: list[TimedLine],
    bpm: float,
) -> list[Phrase]:
    """Assign timed lyrics to phrases based on timestamp overlap.

    Each phrase gets the lyrics whose timing falls primarily within
    that phrase's time window. Lines are assigned to phrases based on
    their midpoint timestamp.

    Args:
        phrases: Phrase list with timing (start_beats, end_beats).
        timed_lines: Lines with precise timestamps from forced alignment.
        bpm: Tempo in beats per minute for converting phrase beats to seconds.

    Returns:
        New list of Phrases with lyric fields populated from alignment.
        Multiple lines in the same phrase are joined with " / ".
    """
    if not phrases:
        return []

    if not timed_lines:
        return phrases

    result: list[Phrase] = []
    line_idx = 0

    for phrase in phrases:
        # Convert phrase boundaries from beats to seconds
        phrase_start_sec = phrase.start_beats * 60.0 / bpm
        phrase_end_sec = phrase.end_beats * 60.0 / bpm

        # Collect lines that fall within this phrase
        phrase_lyrics: list[str] = []

        while line_idx < len(timed_lines):
            line = timed_lines[line_idx]

            # Skip lines with no timing (empty words)
            if line.start == 0.0 and line.end == 0.0 and not line.words:
                line_idx += 1
                continue

            line_midpoint = (line.start + line.end) / 2

            if line_midpoint < phrase_start_sec:
                # Line is before this phrase, skip to next line
                line_idx += 1
            elif line_midpoint <= phrase_end_sec:
                # Line falls within phrase
                phrase_lyrics.append(line.text)
                line_idx += 1
            else:
                # Line is after this phrase, stop collecting
                break

        result.append(
            Phrase(
                start_beats=phrase.start_beats,
                end_beats=phrase.end_beats,
                section_name=phrase.section_name,
                is_section_start=phrase.is_section_start,
                events=phrase.events,
                lyric=" / ".join(phrase_lyrics) if phrase_lyrics else "",
            )
        )

    return result
