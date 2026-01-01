"""Phrase subdivision for ALSmuse.

This module provides functionality for splitting sections into
phrase-sized chunks for detailed A/V script generation.
"""

from .models import Phrase, Section


def subdivide_sections(
    sections: list[Section],
    beats_per_phrase: int = 8,
) -> list[Phrase]:
    """Split sections into phrase-sized chunks.

    Divides each section into fixed-length phrases. The first phrase
    of each section carries the section name and is marked as the
    section start; subsequent phrases within the same section use
    "..." as the section name.

    Args:
        sections: List of sections to subdivide.
        beats_per_phrase: Target phrase length in beats (default 8 = 2 bars in 4/4).

    Returns:
        List of Phrases covering the same time range as the input sections.

    Examples:
        >>> sections = [
        ...     Section(name="INTRO", start_beats=0, end_beats=16),
        ...     Section(name="VERSE1", start_beats=16, end_beats=48),
        ... ]
        >>> phrases = subdivide_sections(sections, beats_per_phrase=8)
        >>> len(phrases)
        6
        >>> phrases[0].section_name
        'INTRO'
        >>> phrases[1].section_name
        '...'
        >>> phrases[2].section_name
        'VERSE1'
    """
    if beats_per_phrase <= 0:
        raise ValueError("beats_per_phrase must be positive")

    phrases: list[Phrase] = []

    for section in sections:
        current_beat = section.start_beats
        is_first = True

        while current_beat < section.end_beats:
            end_beat = min(current_beat + beats_per_phrase, section.end_beats)

            phrases.append(
                Phrase(
                    start_beats=current_beat,
                    end_beats=end_beat,
                    section_name=section.name if is_first else "...",
                    is_section_start=is_first,
                    events=(),
                    lyric="",
                )
            )

            current_beat = end_beat
            is_first = False

    return phrases
