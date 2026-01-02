"""Interactive track category review for ALSmuse.

This module provides an interactive UI for reviewing and modifying
track categorizations using questionary prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import questionary
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .config import MuseConfig, load_config, save_config
from .events import categorize_all_tracks, get_available_categories

# Sentinel values for menu navigation
_DONE_SENTINEL = "__done__"
_BACK_SENTINEL = "__back__"

# Color mapping for track categories
CATEGORY_COLORS: dict[str, str] = {
    "drums": "red",
    "bass": "blue",
    "vocals": "green",
    "lead": "yellow",
    "keys": "cyan",
    "guitar": "orange1",
    "pad": "magenta",
    "fx": "purple",
    "other": "dim white",
}


def _display_category_table(
    track_names: list[str],
    working_categories: dict[str, str],
    categories_with_tracks: list[str],
    console: Console,
) -> None:
    """Display a Rich table of categories with their tracks."""
    table = Table(title="Track Categories", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Category", style="bold", no_wrap=True)
    table.add_column("Count", justify="right", no_wrap=True)
    table.add_column("Tracks", overflow="ellipsis")

    for i, cat in enumerate(categories_with_tracks, 1):
        tracks = [t for t in track_names if working_categories.get(t) == cat]
        color = CATEGORY_COLORS.get(cat, "white")

        category_text = Text(cat, style=color)
        tracks_text = Text(", ".join(tracks), style=color)
        table.add_row(str(i), category_text, str(len(tracks)), tracks_text)

    console.print()
    console.print(table)


def _display_tracks_table(
    tracks: list[str],
    category: str,
    console: Console,
) -> None:
    """Display a Rich table of tracks within a category."""
    color = CATEGORY_COLORS.get(category, "white")
    table = Table(
        title=f"Tracks in [bold {color}]{category}[/]",
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Track Name", style=color)

    for i, track in enumerate(tracks, 1):
        table.add_row(str(i), track)

    console.print()
    console.print(table)


def _parse_multi_selection(input_str: str, max_value: int) -> list[int] | None:
    """Parse a multi-selection input string into a list of indices.

    Supports:
    - Single numbers: "3"
    - Comma-separated: "1,3,5"
    - Ranges: "1-3" (expands to 1,2,3)
    - Mixed: "1-3,5,7-9"

    Args:
        input_str: The input string to parse.
        max_value: Maximum valid number.

    Returns:
        List of valid 1-indexed numbers, or None if parsing failed.
    """
    result: set[int] = set()

    for part in input_str.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            # Range like "1-3"
            try:
                start, end = part.split("-", 1)
                start_num = int(start.strip())
                end_num = int(end.strip())
                if start_num > end_num:
                    return None
                for n in range(start_num, end_num + 1):
                    if 1 <= n <= max_value:
                        result.add(n)
                    else:
                        return None
            except ValueError:
                return None
        else:
            # Single number
            try:
                num = int(part)
                if 1 <= num <= max_value:
                    result.add(num)
                else:
                    return None
            except ValueError:
                return None

    return sorted(result) if result else None


def _prompt_selection(
    console: Console,
    prompt_text: str,
    max_value: int,
    allow_empty: bool = True,
) -> int | None:
    """Prompt for a numbered selection.

    Args:
        console: Rich console for output.
        prompt_text: The prompt to display.
        max_value: Maximum valid number.
        allow_empty: If True, empty input returns None (back/done).

    Returns:
        Selected number (1-indexed) or None if cancelled/back.
    """
    while True:
        response = Prompt.ask(prompt_text, console=console, default="")
        response = response.strip()

        if not response:
            if allow_empty:
                return None
            console.print("[dim]Please enter a number[/]")
            continue

        try:
            num = int(response)
            if 1 <= num <= max_value:
                return num
            console.print(f"[dim]Enter a number between 1 and {max_value}[/]")
        except ValueError:
            console.print("[dim]Please enter a valid number[/]")


def _prompt_multi_selection(
    console: Console,
    prompt_text: str,
    max_value: int,
) -> list[int] | None:
    """Prompt for multiple numbered selections.

    Args:
        console: Rich console for output.
        prompt_text: The prompt to display.
        max_value: Maximum valid number.

    Returns:
        List of selected numbers (1-indexed) or None if cancelled/back.
    """
    while True:
        response = Prompt.ask(prompt_text, console=console, default="")
        response = response.strip()

        if not response:
            return None

        result = _parse_multi_selection(response, max_value)
        if result:
            return result

        console.print(f"[dim]Enter numbers 1-{max_value} (e.g., 1,3,5 or 1-3)[/]")


def review_categories_interactive(
    track_names: list[str],
    current_categories: dict[str, str],
    available_categories: list[str],
) -> dict[str, str]:
    """Interactive UI for reviewing and modifying track categories.

    Flow:
    1. User selects a category to review
    2. Shows tracks in that category
    3. User can reassign individual tracks to different categories
    4. Repeat until user chooses to finish

    Args:
        track_names: List of all track names.
        current_categories: Current mapping of track names to categories.
        available_categories: List of valid category names.

    Returns:
        Updated mapping of track names to categories (only includes overrides).
    """
    console = Console()

    # Track overrides made by user
    overrides: dict[str, str] = {}

    # Build working copy of categories
    working_categories = dict(current_categories)

    while True:
        # Build category list
        categories_with_tracks = [
            cat
            for cat in available_categories
            if any(working_categories.get(t) == cat for t in track_names)
        ]

        # Display category table
        _display_category_table(track_names, working_categories, categories_with_tracks, console)
        console.print("[dim]Enter category number to review, or press Enter when done[/]")

        cat_choice = _prompt_selection(console, "Category #", len(categories_with_tracks))

        if cat_choice is None:
            break

        selected_category = categories_with_tracks[cat_choice - 1]

        # Get tracks in selected category
        tracks_in_category = [
            t for t in track_names if working_categories.get(t) == selected_category
        ]

        if not tracks_in_category:
            continue

        # Display tracks table
        _display_tracks_table(tracks_in_category, selected_category, console)
        console.print(
            "[dim]Enter track numbers to reassign (e.g., 1,3,5 or 1-3), or Enter to go back[/]"
        )

        track_choices = _prompt_multi_selection(console, "Track #", len(tracks_in_category))

        if track_choices is None:
            continue

        selected_tracks = [tracks_in_category[i - 1] for i in track_choices]

        # Display reassignment options
        if len(selected_tracks) == 1:
            title = f"Move [bold]{selected_tracks[0]}[/] to..."
        else:
            title = f"Move [bold]{len(selected_tracks)} tracks[/] to..."

        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("#", justify="right", style="dim", no_wrap=True)
        table.add_column("Category", style="bold")
        table.add_column("", no_wrap=True)

        for i, cat in enumerate(available_categories, 1):
            color = CATEGORY_COLORS.get(cat, "white")
            category_text = Text(cat, style=color)
            marker = Text("← current", style="dim") if cat == selected_category else Text("")
            table.add_row(str(i), category_text, marker)

        console.print()
        console.print(table)
        console.print("[dim]Enter category number for new assignment, or press Enter to cancel[/]")

        new_cat_choice = _prompt_selection(console, "New category #", len(available_categories))

        if new_cat_choice is None:
            continue

        new_category = available_categories[new_cat_choice - 1]

        if new_category == selected_category:
            continue

        # Update working categories and record overrides for all selected tracks
        color = CATEGORY_COLORS.get(new_category, "white")
        for track in selected_tracks:
            working_categories[track] = new_category
            overrides[track] = new_category

        if len(selected_tracks) == 1:
            console.print(f"Moved [bold]{selected_tracks[0]}[/] → [bold {color}]{new_category}[/]")
        else:
            console.print(
                f"Moved [bold]{len(selected_tracks)} tracks[/] → [bold {color}]{new_category}[/]"
            )

    return overrides


def prompt_category_review(
    track_names: list[str],
    current_categories: dict[str, str],
    available_categories: list[str],
) -> dict[str, str] | None:
    """Prompt user to optionally review track categories.

    Args:
        track_names: List of all track names.
        current_categories: Current mapping of track names to categories.
        available_categories: List of valid category names.

    Returns:
        Updated category overrides if user chose to review, None if skipped.
    """
    # Only prompt if we have a TTY
    if not sys.stdin.isatty():
        return None

    console = Console()

    # Build list of categories that have tracks (sorted alphabetically)
    categories_with_tracks = sorted(
        cat
        for cat in available_categories
        if any(current_categories.get(t) == cat for t in track_names)
    )

    # Display the category table (without row numbers for initial display)
    table = Table(title="Track Categories", show_header=True, header_style="bold")
    table.add_column("Category", style="bold", no_wrap=True)
    table.add_column("Count", justify="right", no_wrap=True)
    table.add_column("Tracks", overflow="ellipsis")

    for cat in categories_with_tracks:
        tracks = [t for t in track_names if current_categories.get(t) == cat]
        color = CATEGORY_COLORS.get(cat, "white")
        category_text = Text(cat, style=color)
        tracks_text = Text(", ".join(tracks), style=color)
        table.add_row(category_text, str(len(tracks)), tracks_text)

    console.print()
    console.print(table)
    console.print()

    should_review = questionary.confirm(
        "Would you like to review and adjust track categories?",
        default=False,
    ).ask()

    if should_review is None or not should_review:
        return None

    return review_categories_interactive(track_names, current_categories, available_categories)


def run_interactive_setup(
    als_path: Path,
    track_names: list[str],
) -> tuple[dict[str, str], list[str]]:
    """Run the full interactive setup flow.

    This combines vocal track selection and category review into a single
    interactive session. It loads any existing config, runs the prompts,
    and saves the updated config.

    Args:
        als_path: Path to the ALS file.
        track_names: List of all track names in the project.

    Returns:
        Tuple of (category_overrides, vocal_tracks).
    """
    # Load existing config
    config = load_config(als_path)
    existing_overrides = config.category_overrides if config else {}
    existing_vocals = config.vocal_tracks if config else []

    # Get current categorizations (applying any existing overrides)
    current_categories = categorize_all_tracks(track_names, existing_overrides)
    available_categories = get_available_categories()

    # Prompt for category review
    new_overrides = prompt_category_review(track_names, current_categories, available_categories)

    # Merge new overrides with existing
    if new_overrides:
        final_overrides = {**existing_overrides, **new_overrides}
    else:
        final_overrides = existing_overrides

    # Save updated config if we have any data
    if final_overrides or existing_vocals:
        updated_config = MuseConfig(
            vocal_tracks=existing_vocals,
            category_overrides=final_overrides,
        )
        save_config(als_path, updated_config)

    return final_overrides, existing_vocals
