"""Command-line interface for ALSmuse."""

import sys
from pathlib import Path

import click

from .analyze import analyze_als
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
def analyze(als_file: Path, structure_track: str) -> None:
    """Analyze an Ableton Live Set file."""
    try:
        result = analyze_als(als_file, structure_track)
        click.echo(result)
    except ParseError as e:
        click.echo(f"Error parsing file: {e}", err=True)
        sys.exit(1)
    except TrackNotFoundError as e:
        click.echo(f"Track not found: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
