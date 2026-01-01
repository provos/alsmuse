"""Command-line interface for ALSmuse."""

import sys
from pathlib import Path

import click

from .analyze import analyze_als_v2
from .audio import check_alignment_dependencies
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
@click.option(
    "--align-vocals",
    is_flag=True,
    help="Use forced alignment for precise lyrics timing (requires stable-ts).",
)
@click.option(
    "--vocal-track",
    type=str,
    multiple=True,
    help="Specific vocal track(s) to use. Can be repeated.",
)
@click.option(
    "--all-vocals",
    is_flag=True,
    help="Use all detected vocal tracks without prompting.",
)
def analyze(
    als_file: Path,
    structure_track: str,
    phrase_bars: int,
    show_events: bool,
    lyrics: Path | None,
    align_vocals: bool,
    vocal_track: tuple[str, ...],
    all_vocals: bool,
) -> None:
    """Analyze an Ableton Live Set file.

    Outputs phrase-level chunks of N bars each (default: 2 bars).
    Track enter/exit events are detected by default.
    Use --no-events to disable event detection.

    When --align-vocals is specified, lyrics are aligned to audio using
    forced alignment with stable-ts. This requires the alignment dependencies
    to be installed (pip install 'alsmuse[align]').
    """
    # Validate alignment options
    if align_vocals:
        # Check that lyrics is provided first (before checking dependencies)
        if lyrics is None:
            click.echo(
                "Error: --align-vocals requires --lyrics to be specified.",
                err=True,
            )
            sys.exit(1)

        # Then check for required dependencies
        errors = check_alignment_dependencies()
        if errors:
            for err in errors:
                click.echo(f"Error: {err}", err=True)
            sys.exit(1)

    try:
        # Convert bars to beats (assuming 4/4 time)
        beats_per_phrase = phrase_bars * 4

        # Convert vocal_track tuple to None if empty
        vocal_tracks = vocal_track if vocal_track else None

        result = analyze_als_v2(
            als_file,
            structure_track,
            beats_per_phrase,
            show_events=show_events,
            lyrics_path=lyrics,
            align_vocals=align_vocals,
            vocal_tracks=vocal_tracks,
            use_all_vocals=all_vocals,
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
