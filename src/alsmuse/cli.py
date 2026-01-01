"""Command-line interface for ALSmuse."""

import sys
from pathlib import Path

import click

from .analyze import analyze_als_v2
from .exceptions import ParseError, TrackNotFoundError


@click.group()
@click.version_option()
def main() -> None:
    """ALSmuse - Analyze Ableton Live sets for music video planning."""
    pass


@main.command()
@click.argument("als_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--structure-track",
    default="STRUCTURE",
    help="Name of the structure track containing section markers.",
)
@click.option(
    "--phrase-bars",
    type=int,
    default=2,
    help="Bars per phrase for detailed output (default: 2).",
)
@click.option(
    "--show-events/--no-events",
    default=True,
    help="Show track enter/exit events.",
)
@click.option(
    "--lyrics",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to lyrics file with [SECTION] headers.",
)
def analyze(
    als_file: Path,
    structure_track: str,
    phrase_bars: int,
    show_events: bool,
    lyrics: Path | None,
) -> None:
    """Analyze an Ableton Live Set file.

    Outputs phrase-level chunks of N bars each (default: 2 bars).
    Track enter/exit events are detected by default.
    Use --no-events to disable event detection.
    """
    try:
        # Convert bars to beats (assuming 4/4 time)
        beats_per_phrase = phrase_bars * 4
        result = analyze_als_v2(
            als_file,
            structure_track,
            beats_per_phrase,
            show_events=show_events,
            lyrics_path=lyrics,
        )
        click.echo(result)
    except ParseError as e:
        click.echo(f"Error parsing file: {e}", err=True)
        sys.exit(1)
    except TrackNotFoundError as e:
        click.echo(f"Track not found: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
