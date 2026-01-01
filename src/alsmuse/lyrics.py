"""Lyrics parsing and distribution for ALSmuse.

This module provides functionality for parsing lyrics files with section
headers and distributing lyrics across phrases within each section.

It includes both heuristic distribution (based on section structure) and
time-based distribution (using forced alignment timestamps).
"""

from pathlib import Path

from .models import Phrase, TimedLine


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
