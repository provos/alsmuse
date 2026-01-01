"""Interactive track category review for ALSmuse.

This module provides an interactive UI for reviewing and modifying
track categorizations using questionary prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import questionary

# Sentinel values for menu navigation
_DONE_SENTINEL = "__done__"
_BACK_SENTINEL = "__back__"


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
    # Track overrides made by user
    overrides: dict[str, str] = {}

    # Build working copy of categories
    working_categories = dict(current_categories)

    while True:
        # Build category choices with track counts
        category_counts: dict[str, int] = {}
        for cat in available_categories:
            category_counts[cat] = sum(1 for t in track_names if working_categories.get(t) == cat)

        choices = [
            questionary.Choice(
                title=f"{cat} ({category_counts.get(cat, 0)} tracks)",
                value=cat,
            )
            for cat in available_categories
            if category_counts.get(cat, 0) > 0
        ]

        # Add "Done" option
        choices.append(questionary.Choice(title="[Done - save and continue]", value=_DONE_SENTINEL))

        selected_category = questionary.select(
            "Select a category to review (or Done to finish):",
            choices=choices,
        ).ask()

        if selected_category is None or selected_category == _DONE_SENTINEL:
            # User chose to finish (or cancelled)
            break

        # Get tracks in selected category
        tracks_in_category = [
            t for t in track_names if working_categories.get(t) == selected_category
        ]

        if not tracks_in_category:
            click.echo(f"No tracks in category '{selected_category}'")
            continue

        # Show tracks and allow reassignment
        track_choice = questionary.select(
            f"Tracks in '{selected_category}' - select to reassign or go back:",
            choices=[questionary.Choice(title=track, value=track) for track in tracks_in_category]
            + [questionary.Choice(title="[Back to categories]", value=_BACK_SENTINEL)],
        ).ask()

        if track_choice is None or track_choice == _BACK_SENTINEL:
            continue

        # User selected a track to reassign
        new_category = questionary.select(
            f"Move '{track_choice}' to which category?",
            choices=[
                questionary.Choice(
                    title=f"{cat}" + (" (current)" if cat == selected_category else ""),
                    value=cat,
                )
                for cat in available_categories
            ],
        ).ask()

        if new_category is None:
            # User cancelled
            continue

        if new_category != selected_category:
            # Update working categories and record override
            working_categories[track_choice] = new_category
            overrides[track_choice] = new_category
            click.echo(f"Moved '{track_choice}' from '{selected_category}' to '{new_category}'")

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

    # Show current categorization summary
    category_counts: dict[str, int] = {}
    for cat in available_categories:
        count = sum(1 for t in track_names if current_categories.get(t) == cat)
        if count > 0:
            category_counts[cat] = count

    print("\nTrack categories detected:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} tracks")
    print()

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
    from .config import MuseConfig, load_config, save_config
    from .events import categorize_all_tracks, get_available_categories

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
