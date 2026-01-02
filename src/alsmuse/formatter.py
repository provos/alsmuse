"""Output formatting for ALSmuse.

Pure functions for converting sections to various output formats.
These functions have no side effects and are easily testable.
"""

from .models import Phrase, Section, TimeContext, TrackEvent


def format_time(seconds: float) -> str:
    """Format seconds as M:SS.s or MM:SS.s.

    Converts a time in seconds to a human-readable format with
    minutes and seconds, showing one decimal place.

    Args:
        seconds: Time in seconds to format.

    Returns:
        Formatted time string in M:SS.s or MM:SS.s format.

    Examples:
        >>> format_time(11.67)
        '0:11.7'
        >>> format_time(65.5)
        '1:05.5'
        >>> format_time(125.3)
        '2:05.3'
    """
    minutes = int(seconds // 60)
    remaining_seconds = seconds - (minutes * 60)
    return f"{minutes}:{remaining_seconds:04.1f}"


def format_av_table(sections: list[Section], time_ctx: TimeContext) -> str:
    """Format sections as a markdown A/V table.

    Creates a markdown table with Time, Audio, and Video columns.
    The Video column is left empty for users to fill in later.

    Args:
        sections: List of sections to format.
        time_ctx: TimeContext for time conversion (includes BPM and offset).

    Returns:
        Markdown formatted table as a string.

    Example:
        >>> sections = [
        ...     Section(name="INTRO", start_beats=0, end_beats=16),
        ...     Section(name="VERSE1", start_beats=16, end_beats=32),
        ... ]
        >>> time_ctx = TimeContext(bpm=120)
        >>> print(format_av_table(sections, time_ctx))
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
        time_str = format_time(time_ctx.beats_to_display_seconds(section.start_beats))
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
        if event.event_type == "fill":
            if event.fill_context:
                parts.append(f"Drum fill {event.fill_context}")
            else:
                parts.append("Drum fill")
        else:
            verb = "enters" if event.event_type == "enter" else "exits"
            parts.append(f"{event.category.title()} {verb}")

    return ", ".join(parts)


def format_phrase_table(
    phrases: list[Phrase],
    time_ctx: TimeContext,
    show_events: bool = True,
    show_lyrics: bool = False,
) -> str:
    """Format phrases as a markdown A/V table.

    Creates a markdown table with Time, Audio, and Video columns.
    The Audio column combines section cue, events, and lyrics using <br> tags.
    The Video column is left empty for users to fill in later.

    Args:
        phrases: List of phrases to format.
        time_ctx: TimeContext for time conversion (includes BPM and offset).
        show_events: Whether to include events in the Audio column.
        show_lyrics: Whether to include lyrics in the Audio column.

    Returns:
        Markdown formatted table as a string.

    Examples:
        >>> from alsmuse.models import Phrase, TimeContext
        >>> phrases = [
        ...     Phrase(start_beats=0, end_beats=8, section_name="INTRO",
        ...            is_section_start=True),
        ...     Phrase(start_beats=8, end_beats=16, section_name="...",
        ...            is_section_start=False),
        ... ]
        >>> time_ctx = TimeContext(bpm=120)
        >>> print(format_phrase_table(phrases, time_ctx, show_events=False))
        | Time | Audio | Video |
        |------|-------|-------|
        | 0:00 | INTRO |  |
        | 0:04 | ... |  |
    """
    # Create header row and separator
    lines = [
        "| Time | Audio | Video |",
        "|------|-------|-------|",
    ]

    for phrase in phrases:
        time_str = format_time(time_ctx.beats_to_display_seconds(phrase.start_beats))

        # Build audio content with <br> separators
        audio_parts: list[str] = []

        # Always include the cue/section name
        audio_parts.append(f"**{phrase.section_name}**")

        # Add events if enabled and present
        if show_events:
            events_str = format_events(phrase.events)
            if events_str:
                audio_parts.append(events_str)

        # Add lyrics if enabled and present
        if show_lyrics and phrase.lyric:
            audio_parts.append(f'*"{phrase.lyric}"*')

        audio_cell = "<br>".join(audio_parts)

        lines.append(f"| {time_str} | {audio_cell} |  |")

    return "\n".join(lines)
