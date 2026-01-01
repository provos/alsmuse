"""Output formatting for ALSmuse.

Pure functions for converting sections to various output formats.
These functions have no side effects and are easily testable.
"""

from .models import Phrase, Section, TrackEvent


def format_time(seconds: float) -> str:
    """Format seconds as M:SS or MM:SS.

    Converts a time in seconds to a human-readable format with
    minutes and seconds, rounded to the nearest second.

    Args:
        seconds: Time in seconds to format.

    Returns:
        Formatted time string in M:SS or MM:SS format.

    Examples:
        >>> format_time(11.67)
        '0:11'
        >>> format_time(65.5)
        '1:05'
        >>> format_time(125.3)
        '2:05'
    """
    total_seconds = round(seconds)
    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"


def format_av_table(sections: list[Section], bpm: float) -> str:
    """Format sections as a markdown A/V table.

    Creates a markdown table with Time, Audio, and Video columns.
    The Video column is left empty for users to fill in later.

    Args:
        sections: List of sections to format.
        bpm: Beats per minute for time conversion.

    Returns:
        Markdown formatted table as a string.

    Example:
        >>> sections = [
        ...     Section(name="INTRO", start_beats=0, end_beats=16),
        ...     Section(name="VERSE1", start_beats=16, end_beats=32),
        ... ]
        >>> print(format_av_table(sections, bpm=120))
        | Time | Audio | Video |
        |------|-------|-------|
        | 0:00 | INTRO | |
        | 0:08 | VERSE1 | |
    """
    lines = [
        "| Time | Audio | Video |",
        "|------|-------|-------|",
    ]

    for section in sections:
        time_str = format_time(section.start_time(bpm))
        lines.append(f"| {time_str} | {section.name} | |")

    return "\n".join(lines)


def format_events(events: tuple[TrackEvent, ...]) -> str:
    """Format track events as a compact string.

    Converts a tuple of TrackEvent objects into a human-readable
    description of what instruments are entering or exiting.

    Args:
        events: Tuple of TrackEvent objects to format.

    Returns:
        Comma-separated string of event descriptions.

    Examples:
        >>> from alsmuse.models import TrackEvent
        >>> events = (
        ...     TrackEvent(beat=0, track_name="Drums", event_type="enter", category="drums"),
        ...     TrackEvent(beat=0, track_name="Bass", event_type="enter", category="bass"),
        ... )
        >>> format_events(events)
        'Drums enters, Bass enters'
    """
    if not events:
        return ""

    parts = []
    for event in events:
        verb = "enters" if event.event_type == "enter" else "exits"
        parts.append(f"{event.category.title()} {verb}")

    return ", ".join(parts)


def format_phrase_table(
    phrases: list[Phrase],
    bpm: float,
    show_events: bool = True,
    show_lyrics: bool = False,
) -> str:
    """Format phrases as a markdown A/V table.

    Creates a markdown table with Time, Cue, and optionally Events and Lyrics
    columns. The Video column is left empty for users to fill in later.

    Args:
        phrases: List of phrases to format.
        bpm: Beats per minute for time conversion.
        show_events: Whether to include the Events column.
        show_lyrics: Whether to include the Lyrics column.

    Returns:
        Markdown formatted table as a string.

    Examples:
        >>> from alsmuse.models import Phrase
        >>> phrases = [
        ...     Phrase(start_beats=0, end_beats=8, section_name="INTRO",
        ...            is_section_start=True),
        ...     Phrase(start_beats=8, end_beats=16, section_name="...",
        ...            is_section_start=False),
        ... ]
        >>> print(format_phrase_table(phrases, bpm=120, show_events=False))
        | Time | Cue | Video |
        |------|-----|-------|
        | 0:00 | INTRO | |
        | 0:04 | ... | |
    """
    # Build header based on options
    headers = ["Time", "Cue"]
    if show_events:
        headers.append("Events")
    if show_lyrics:
        headers.append("Lyrics")
    headers.append("Video")

    # Create header row and separator
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("------" for _ in headers) + "|",
    ]

    for phrase in phrases:
        time_str = format_time(phrase.start_time(bpm))
        cue = phrase.section_name

        row = [time_str, cue]

        if show_events:
            events_str = format_events(phrase.events)
            row.append(events_str)

        if show_lyrics:
            row.append(f'"{phrase.lyric}"' if phrase.lyric else "")

        row.append("")  # Empty Video column

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
