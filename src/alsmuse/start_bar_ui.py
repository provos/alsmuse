"""Interactive start bar selection UI for ALSmuse.

This module provides a rich terminal UI for selecting the start bar
of a song, displaying a visual timeline with track clips and allowing
interactive selection with keyboard input.
"""

from __future__ import annotations

import contextlib
import sys
import termios
import tty
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .models import LiveSet

# Priority order for tracks to display (most relevant first)
TRACK_PRIORITY = [
    "structure",
    "drums",
    "bass",
    "vocal",
    "vox",
    "lead",
    "keys",
    "piano",
    "guitar",
    "synth",
    "pad",
]

# Maximum number of tracks to display
MAX_DISPLAY_TRACKS = 8


@dataclass(frozen=True)
class TimelineConfig:
    """Configuration for timeline rendering.

    Attributes:
        bar_start: First bar to display.
        bar_end: Last bar to display (exclusive).
        track_label_width: Width of track name column.
        timeline_width: Width of the timeline portion.
        total_width: Total display width.
    """

    bar_start: int
    bar_end: int
    track_label_width: int
    timeline_width: int
    total_width: int


@dataclass(frozen=True)
class TrackDisplay:
    """A track prepared for timeline display.

    Attributes:
        name: Display name (truncated if necessary).
        original_name: Full track name.
        bar_ranges: List of (start_bar, end_bar) tuples where clips exist.
    """

    name: str
    original_name: str
    bar_ranges: list[tuple[int, int]]


def beats_to_bar(beats: float, time_signature: tuple[int, int] = (4, 4)) -> int:
    """Convert beats to bar number.

    Args:
        beats: Position in beats.
        time_signature: (numerator, denominator) tuple.

    Returns:
        Bar number (0-indexed).
    """
    numerator, denominator = time_signature
    beats_per_bar = numerator * (4 / denominator)
    return int(beats // beats_per_bar)


def get_track_priority(track_name: str) -> int:
    """Get the display priority for a track based on its name.

    Lower values = higher priority (displayed first).

    Args:
        track_name: The track name.

    Returns:
        Priority value (lower = more important).
    """
    lower_name = track_name.lower()

    for i, keyword in enumerate(TRACK_PRIORITY):
        if keyword in lower_name:
            return i

    return len(TRACK_PRIORITY)


def select_display_tracks(
    live_set: LiveSet, max_tracks: int = MAX_DISPLAY_TRACKS
) -> list[TrackDisplay]:
    """Select and prepare tracks for timeline display.

    Selects the most relevant tracks based on naming conventions and
    converts clip positions to bar ranges.

    Args:
        live_set: The parsed LiveSet.
        max_tracks: Maximum number of tracks to display.

    Returns:
        List of TrackDisplay objects, sorted by priority.
    """
    time_sig = live_set.tempo.time_signature
    enabled_tracks = live_set.enabled_tracks()

    # Filter tracks that have clips
    tracks_with_clips = [t for t in enabled_tracks if t.clips]

    if not tracks_with_clips:
        return []

    # Sort by priority
    sorted_tracks = sorted(tracks_with_clips, key=lambda t: get_track_priority(t.name))

    # Take top N tracks
    selected = sorted_tracks[:max_tracks]

    result: list[TrackDisplay] = []
    for track in selected:
        # Convert clips to bar ranges
        bar_ranges: list[tuple[int, int]] = []
        for clip in track.clips:
            start_bar = beats_to_bar(clip.start_beats, time_sig)
            end_bar = beats_to_bar(clip.end_beats, time_sig)
            # Ensure end_bar is at least start_bar + 1
            if end_bar <= start_bar:
                end_bar = start_bar + 1
            bar_ranges.append((start_bar, end_bar))

        # Merge overlapping ranges
        bar_ranges = _merge_bar_ranges(bar_ranges)

        # Truncate name for display (max 12 chars)
        display_name = track.name[:12] if len(track.name) > 12 else track.name

        result.append(
            TrackDisplay(
                name=display_name,
                original_name=track.name,
                bar_ranges=bar_ranges,
            )
        )

    return result


def _merge_bar_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent bar ranges.

    Args:
        ranges: List of (start, end) tuples.

    Returns:
        Merged list of non-overlapping ranges.
    """
    if not ranges:
        return []

    sorted_ranges = sorted(ranges)
    merged: list[tuple[int, int]] = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping or adjacent, merge
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


MAX_DISPLAY_BARS = 32


def compute_timeline_config(
    tracks: list[TrackDisplay],
    suggested_bar: int,
    terminal_width: int = 100,
) -> TimelineConfig:
    """Compute the timeline display configuration.

    Determines the bar range to display and column widths.
    Limits the display to MAX_DISPLAY_BARS (32) bars, centered around
    the suggested start bar when possible.

    Args:
        tracks: List of tracks to display.
        suggested_bar: The suggested/selected start bar.
        terminal_width: Available terminal width.

    Returns:
        TimelineConfig with computed dimensions.
    """
    # Find the extent of all clips
    content_min_bar = float("inf")
    content_max_bar = 0

    for track in tracks:
        for start, end in track.bar_ranges:
            if start < content_min_bar:
                content_min_bar = start
            if end > content_max_bar:
                content_max_bar = end

    # Handle empty tracks
    if content_min_bar == float("inf"):
        content_min_bar = 0

    # Include the suggested bar in the content range
    if suggested_bar < content_min_bar:
        content_min_bar = suggested_bar
    if suggested_bar >= content_max_bar:
        content_max_bar = suggested_bar + 1

    # Calculate the window to display (max 32 bars)
    content_range = content_max_bar - content_min_bar + 4  # Add padding

    if content_range <= MAX_DISPLAY_BARS:
        # Content fits in window, show it all starting from content_min
        min_bar = max(0, int(content_min_bar))
        max_bar = min_bar + min(content_range, MAX_DISPLAY_BARS)
    else:
        # Content is larger than window, center on suggested bar
        # Show suggested bar roughly 1/4 from the left
        min_bar = max(0, suggested_bar - MAX_DISPLAY_BARS // 4)
        max_bar = min_bar + MAX_DISPLAY_BARS

    # Calculate widths
    track_label_width = 12  # Fixed width for track names
    padding = 6  # Margins and borders

    # Available space for timeline
    timeline_width = min(terminal_width - track_label_width - padding, 80)
    timeline_width = max(timeline_width, 40)  # Minimum width

    total_width = track_label_width + timeline_width + padding

    return TimelineConfig(
        bar_start=min_bar,
        bar_end=max_bar,
        track_label_width=track_label_width,
        timeline_width=timeline_width,
        total_width=total_width,
    )


def render_bar_axis(config: TimelineConfig) -> Text:
    """Render the bar number axis.

    Args:
        config: Timeline configuration.

    Returns:
        Rich Text object with the bar axis.
    """
    bar_count = config.bar_end - config.bar_start
    chars_per_bar = config.timeline_width / bar_count if bar_count > 0 else 1

    # Build the axis line
    axis = Text()
    axis.append(" " * config.track_label_width)

    # Determine label spacing (every 4 or 8 bars depending on space)
    if chars_per_bar >= 5:
        label_interval = 4
    elif chars_per_bar >= 2.5:
        label_interval = 8
    else:
        label_interval = 16

    # Build the bar number labels
    position = 0.0
    for bar in range(config.bar_start, config.bar_end):
        bar_width = chars_per_bar
        if bar % label_interval == 0:
            label = str(bar)
            axis.append(label, style="dim")
            remaining = int(bar_width) - len(label)
            if remaining > 0:
                axis.append(" " * remaining)
        else:
            axis.append(" " * int(bar_width))
        position += bar_width

    return axis


def render_track_row(
    track: TrackDisplay,
    config: TimelineConfig,
    selected_bar: int,
) -> Text:
    """Render a single track row with its clip bars.

    Args:
        track: The track to render.
        config: Timeline configuration.
        selected_bar: Currently selected bar for highlighting.

    Returns:
        Rich Text object with the track row.
    """
    bar_count = config.bar_end - config.bar_start
    chars_per_bar = config.timeline_width / bar_count if bar_count > 0 else 1

    row = Text()

    # Track name (right-aligned, padded)
    name = track.name.rjust(config.track_label_width - 1)
    row.append(name, style="bold")
    row.append(" ")

    # Build a character map for the timeline
    timeline_chars: list[tuple[str, str]] = []  # (char, style)

    for bar in range(config.bar_start, config.bar_end):
        # Check if this bar has content
        has_content = any(start <= bar < end for start, end in track.bar_ranges)
        bar_char_count = max(1, int(chars_per_bar))

        if has_content:
            char = "\u2588"  # Full block
            style = "blue"
        else:
            char = "\u2500"  # Horizontal line (thin)
            style = "dim"

        for _ in range(bar_char_count):
            timeline_chars.append((char, style))

    # Truncate or pad to exact width
    while len(timeline_chars) > config.timeline_width:
        timeline_chars.pop()
    while len(timeline_chars) < config.timeline_width:
        timeline_chars.append(("\u2500", "dim"))

    # Add the timeline characters
    for char, style in timeline_chars:
        row.append(char, style=style)

    return row


def render_selection_indicator(config: TimelineConfig, selected_bar: int) -> Text:
    """Render the selection indicator row.

    Args:
        config: Timeline configuration.
        selected_bar: Currently selected bar.

    Returns:
        Rich Text object with the selection indicator.
    """
    bar_count = config.bar_end - config.bar_start
    chars_per_bar = config.timeline_width / bar_count if bar_count > 0 else 1

    row = Text()
    row.append(" " * config.track_label_width)

    # Calculate position of the selected bar
    if config.bar_start <= selected_bar < config.bar_end:
        bar_offset = selected_bar - config.bar_start
        char_position = int(bar_offset * chars_per_bar)

        # Pad to position, add indicator
        if char_position > 0:
            row.append(" " * char_position)
        row.append("\u25b2", style="bold green")  # Up-pointing triangle
    else:
        # Selected bar is outside visible range
        if selected_bar < config.bar_start:
            row.append("\u25c4", style="bold yellow")  # Left-pointing triangle
        else:
            row.append(" " * (config.timeline_width - 1))
            row.append("\u25ba", style="bold yellow")  # Right-pointing triangle

    return row


def render_timeline(
    tracks: list[TrackDisplay],
    config: TimelineConfig,
    selected_bar: int,
    console: Console,
) -> None:
    """Render the complete timeline display.

    Args:
        tracks: List of tracks to display.
        config: Timeline configuration.
        selected_bar: Currently selected bar.
        console: Rich Console to print to.
    """
    # Bar axis
    console.print(render_bar_axis(config))

    # Separator
    separator = Text()
    separator.append(" " * config.track_label_width)
    separator.append("\u2502" * int(config.timeline_width / 4), style="dim")
    console.print(separator)

    # Track rows
    for track in tracks:
        console.print(render_track_row(track, config, selected_bar))

    # Separator
    console.print(separator)

    # Selection indicator
    console.print(render_selection_indicator(config, selected_bar))

    # Start bar label
    label = Text()
    label.append(" " * config.track_label_width)
    bar_count = config.bar_end - config.bar_start
    chars_per_bar = config.timeline_width / bar_count if bar_count > 0 else 1
    if config.bar_start <= selected_bar < config.bar_end:
        bar_offset = selected_bar - config.bar_start
        char_position = int(bar_offset * chars_per_bar)
        if char_position > 0:
            label.append(" " * (char_position - 3 if char_position >= 3 else 0))
        label.append(f"Start: {selected_bar}", style="bold green")
    else:
        label.append(f"Start: {selected_bar}", style="bold yellow")
    console.print(label)


def render_help_text() -> Text:
    """Render the help text for keyboard controls.

    Returns:
        Rich Text with help text.
    """
    help_text = Text()
    help_text.append("[", style="dim")
    help_text.append("\u2190/\u2192", style="bold cyan")
    help_text.append("] Move  [", style="dim")
    help_text.append("Enter", style="bold cyan")
    help_text.append("] Confirm  [", style="dim")
    help_text.append("0-9", style="bold cyan")
    help_text.append("] Type bar  [", style="dim")
    help_text.append("q/Esc", style="bold cyan")
    help_text.append("] Cancel", style="dim")
    return help_text


@contextmanager
def _raw_terminal() -> Generator[None, None, None]:
    """Context manager for raw terminal mode (Unix only).

    Puts the terminal in raw mode so we can read single keystrokes.
    Restores the terminal settings on exit.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _read_key() -> str:
    """Read a single keypress, handling escape sequences for arrow keys.

    Returns:
        String representing the key pressed:
        - 'left' for left arrow
        - 'right' for right arrow
        - 'enter' for Enter key
        - 'esc' for Escape key (without arrow sequence)
        - 'q' for q key
        - digit characters '0'-'9'
        - other single characters as-is
    """
    arrow_keys = {"D": "left", "C": "right", "A": "up", "B": "down"}

    ch = sys.stdin.read(1)

    if ch == "\x1b":  # Escape sequence
        # Check if more characters are available (arrow key)
        ch2 = sys.stdin.read(1)
        if ch2 == "[":
            ch3 = sys.stdin.read(1)
            if ch3 in arrow_keys:
                return arrow_keys[ch3]
        # Just Escape key
        return "esc"
    elif ch in ("\r", "\n"):
        return "enter"
    elif ch == "\x03":  # Ctrl+C
        return "ctrl-c"
    else:
        return ch


def _build_timeline_content(
    tracks: list[TrackDisplay],
    config: TimelineConfig,
    selected_bar: int,
    number_input: str,
) -> list[Text]:
    """Build the timeline display content as a list of Rich Text objects.

    Args:
        tracks: List of tracks to display.
        config: Timeline configuration.
        selected_bar: Currently selected bar.
        number_input: Current number being typed (empty string if none).

    Returns:
        List of Text objects, one per line.
    """
    lines: list[Text] = []

    # Bar axis
    lines.append(render_bar_axis(config))

    # Separator
    separator = Text()
    separator.append(" " * config.track_label_width)
    separator.append("\u2502" * int(config.timeline_width / 4), style="dim")
    lines.append(separator)

    # Track rows
    for track in tracks:
        lines.append(render_track_row(track, config, selected_bar))

    # Separator
    lines.append(Text(separator.plain, style="dim"))

    # Selection indicator
    lines.append(render_selection_indicator(config, selected_bar))

    # Start bar label
    label = Text()
    label.append(" " * config.track_label_width)
    bar_count = config.bar_end - config.bar_start
    chars_per_bar = config.timeline_width / bar_count if bar_count > 0 else 1
    if config.bar_start <= selected_bar < config.bar_end:
        bar_offset = selected_bar - config.bar_start
        char_position = int(bar_offset * chars_per_bar)
        if char_position > 0:
            label.append(" " * (char_position - 3 if char_position >= 3 else 0))
        label.append(f"Start: {selected_bar}", style="bold green")
    else:
        label.append(f"Start: {selected_bar}", style="bold yellow")
    lines.append(label)

    # Help text
    lines.append(Text())  # Blank line
    lines.append(render_help_text())

    # Input prompt
    prompt_text = Text()
    prompt_text.append("> ", style="bold")
    if number_input:
        prompt_text.append(number_input, style="bold cyan")
        prompt_text.append("_", style="blink")
    else:
        prompt_text.append(f"(bar {selected_bar})", style="dim")
    lines.append(prompt_text)

    return lines


def _render_timeline_to_string(
    tracks: list[TrackDisplay],
    config: TimelineConfig,
    selected_bar: int,
    number_input: str,
) -> tuple[str, int]:
    """Render the timeline to a string with ANSI codes.

    Args:
        tracks: List of tracks to display.
        config: Timeline configuration.
        selected_bar: Currently selected bar.
        number_input: Current number being typed.

    Returns:
        Tuple of (rendered string with ANSI codes, number of lines).
    """
    lines = _build_timeline_content(tracks, config, selected_bar, number_input)

    # Create a console that writes to StringIO with forced terminal mode
    output = StringIO()
    capture_console = Console(
        file=output,
        force_terminal=True,
        width=config.total_width + 20,  # Extra width to avoid wrapping
        no_color=False,
    )

    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            capture_console.print(line)
        else:
            # Last line - no newline
            capture_console.print(line, end="")

    return output.getvalue(), len(lines)


def prompt_start_bar(
    live_set: LiveSet,
    suggested: int,
    console: Console | None = None,
) -> int:
    """Prompt user to select start bar with interactive timeline UI.

    Displays a visual timeline and allows the user to adjust the
    start bar selection using keyboard input or by typing a number.

    Args:
        live_set: The parsed LiveSet with tracks and tempo.
        suggested: The auto-detected suggested start bar.
        console: Optional Rich Console for output (uses default if None).

    Returns:
        The selected start bar.
    """
    if console is None:
        console = Console()

    # Prepare display data
    tracks = select_display_tracks(live_set)
    if not tracks:
        # No tracks to display, fall back to simple prompt
        return _simple_prompt(console, suggested)

    try:
        # Check if we can use interactive input
        if not sys.stdin.isatty():
            # Non-interactive, just return suggested
            return suggested

        # Try to use interactive selection with arrow keys
        return _interactive_select(console, tracks, suggested)

    except (OSError, termios.error):
        # Fall back to simple prompt if terminal manipulation fails
        return _simple_prompt(console, suggested)


def _render_indicator_line(config: TimelineConfig, selected_bar: int) -> str:
    """Render just the indicator line as a plain string.

    Args:
        config: Timeline configuration.
        selected_bar: Currently selected bar.

    Returns:
        String for the indicator line (no newline).
    """
    bar_count = config.bar_end - config.bar_start
    chars_per_bar = config.timeline_width / bar_count if bar_count > 0 else 1
    # Use same calculation as render_track_row for consistency
    bar_char_count = max(1, int(chars_per_bar))

    parts = [" " * config.track_label_width]

    if config.bar_start <= selected_bar < config.bar_end:
        bar_offset = selected_bar - config.bar_start
        char_position = bar_offset * bar_char_count
        if char_position > 0:
            parts.append(" " * char_position)
        # Green bold triangle
        parts.append("\x1b[1;32m\u25b2\x1b[0m")
    else:
        if selected_bar < config.bar_start:
            # Yellow bold left triangle
            parts.append("\x1b[1;33m\u25c4\x1b[0m")
        else:
            parts.append(" " * (config.timeline_width - 1))
            parts.append("\x1b[1;33m\u25ba\x1b[0m")

    return "".join(parts)


def _render_label_line(config: TimelineConfig, selected_bar: int) -> str:
    """Render just the label line as a plain string.

    Args:
        config: Timeline configuration.
        selected_bar: Currently selected bar.

    Returns:
        String for the label line (no newline).
    """
    bar_count = config.bar_end - config.bar_start
    chars_per_bar = config.timeline_width / bar_count if bar_count > 0 else 1
    # Use same calculation as render_track_row for consistency
    bar_char_count = max(1, int(chars_per_bar))

    parts = [" " * config.track_label_width]

    if config.bar_start <= selected_bar < config.bar_end:
        bar_offset = selected_bar - config.bar_start
        char_position = bar_offset * bar_char_count
        if char_position > 3:
            parts.append(" " * (char_position - 3))
        # Green bold label
        parts.append(f"\x1b[1;32mStart: {selected_bar}\x1b[0m")
    else:
        # Yellow bold label
        parts.append(f"\x1b[1;33mStart: {selected_bar}\x1b[0m")

    return "".join(parts)


def _interactive_select(
    console: Console,
    tracks: list[TrackDisplay],
    initial_bar: int,
) -> int:
    """Run interactive selection with arrow key navigation.

    Args:
        console: Rich Console for output.
        tracks: Prepared track display data.
        initial_bar: Initial selected bar.

    Returns:
        Selected start bar.
    """
    selected_bar = initial_bar
    number_input = ""  # Buffer for typed numbers
    config = compute_timeline_config(tracks, selected_bar)

    # Print title
    console.print()
    console.print(Panel.fit("Start Bar Selection", style="bold"))
    console.print()

    # Draw initial full display (static parts)
    console.print(render_bar_axis(config))

    separator = Text()
    separator.append(" " * config.track_label_width)
    separator.append("\u2502" * int(config.timeline_width / 4), style="dim")
    console.print(separator)

    for track in tracks:
        console.print(render_track_row(track, config, selected_bar))

    console.print(separator)

    # Dynamic lines - indicator and label will be updated on each keypress
    # Use sys.stdout.write for all dynamic content to avoid Rich interference
    sys.stdout.write(_render_indicator_line(config, selected_bar) + "\n")
    sys.stdout.write(_render_label_line(config, selected_bar) + "\n")
    # Help text (static, using plain ANSI codes)
    help_str = (
        "\x1b[2m[\x1b[0m\x1b[1;36m\u2190/\u2192\x1b[0m\x1b[2m] Move  "
        "[\x1b[0m\x1b[1;36mEnter\x1b[0m\x1b[2m] Confirm  "
        "[\x1b[0m\x1b[1;36m0-9\x1b[0m\x1b[2m] Type bar  "
        "[\x1b[0m\x1b[1;36mq/Esc\x1b[0m\x1b[2m] Cancel\x1b[0m"
    )
    sys.stdout.write(help_str)
    sys.stdout.flush()

    # Cursor is at end of help line
    # indicator is 2 lines up, label is 1 line up
    indicator_offset = 2
    label_offset = 1

    try:
        with _raw_terminal():
            while True:
                key = _read_key()

                if key == "enter":
                    if number_input:
                        with contextlib.suppress(ValueError):
                            selected_bar = int(number_input)
                    break

                elif key in ("esc", "q", "ctrl-c"):
                    selected_bar = 0
                    break

                elif key == "left":
                    number_input = ""
                    if selected_bar > 0:
                        selected_bar -= 1

                elif key == "right":
                    number_input = ""
                    selected_bar += 1

                elif key.isdigit():
                    number_input += key
                    with contextlib.suppress(ValueError):
                        selected_bar = int(number_input)

                elif key in ("\x7f", "\b"):  # Backspace
                    if number_input:
                        number_input = number_input[:-1]
                        if number_input:
                            with contextlib.suppress(ValueError):
                                selected_bar = int(number_input)
                        else:
                            selected_bar = initial_bar

                # Update indicator and label lines only
                # Move up to indicator line (2 lines up), clear and redraw
                sys.stdout.write(f"\x1b[{indicator_offset}A\r\x1b[K")
                sys.stdout.write(_render_indicator_line(config, selected_bar))

                # Move down to label line (1 line down), clear and redraw
                sys.stdout.write("\x1b[1B\r\x1b[K")
                sys.stdout.write(_render_label_line(config, selected_bar))

                # Move back down to help line (1 line down) and go to end
                sys.stdout.write("\x1b[1B\x1b[999C")
                sys.stdout.flush()

    except (OSError, termios.error):
        sys.stdout.write("\n")
        sys.stdout.flush()
        return _simple_prompt(console, initial_bar)

    console.print(f"\n\nSelected start bar: [bold green]{selected_bar}[/bold green]")
    return selected_bar


def _simple_prompt(console: Console, suggested: int) -> int:
    """Simple fallback prompt without timeline visualization.

    Args:
        console: Rich Console for output.
        suggested: Suggested start bar.

    Returns:
        Selected start bar.
    """
    console.print(f"Detected song start: bar [bold]{suggested}[/bold]")

    response = Prompt.ask(
        f"Use bar {suggested} as start? [Y/n/number]",
        console=console,
        default="y",
    )
    response = response.strip().lower()

    if response in ("y", "yes", ""):
        return suggested
    elif response in ("n", "no"):
        return 0
    else:
        try:
            return int(response)
        except ValueError:
            console.print(f"[yellow]Invalid input '{response}', using bar 0[/yellow]")
            return 0


def select_start_bar_interactive(
    live_set: LiveSet,
    suggested: int,
) -> int:
    """Main entry point for interactive start bar selection.

    This function displays a visual timeline with tracks and clips,
    shows the suggested start bar, and allows the user to confirm
    or adjust the selection.

    Args:
        live_set: The parsed LiveSet containing tracks and tempo.
        suggested: The auto-detected suggested start bar.

    Returns:
        The selected start bar (0 or greater).
    """
    console = Console()
    return prompt_start_bar(live_set, suggested, console)
