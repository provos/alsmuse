"""Output formatting for ALSmuse.

Pure functions for converting sections to various output formats.
These functions have no side effects and are easily testable.
"""

from .models import Section


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
