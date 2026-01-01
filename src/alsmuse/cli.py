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
    help="Path to lyrics file. Supports plain text with [SECTION] headers, "
    "LRC format, or simple timed format. Timestamped formats bypass alignment.",
)
@click.option(
    "--align-vocals/--no-align-vocals",
    default=None,
    help="Use forced alignment for precise lyrics timing (requires stable-ts). "
    "Enabled by default when --lyrics is specified.",
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
@click.option(
    "--save-vocals",
    type=click.Path(path_type=Path),
    default=None,
    help="Save combined vocals to this path for validation.",
)
@click.option(
    "--transcribe",
    is_flag=True,
    help="Transcribe lyrics from vocal audio using ASR. Cannot be used with --lyrics.",
)
@click.option(
    "--language",
    type=str,
    default="en",
    help="Language code for transcription/alignment (default: en). Examples: en, es, fr, de, ja.",
)
@click.option(
    "--whisper-model",
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    default="base",
    help="Whisper model size for transcription/alignment (default: base).",
)
@click.option(
    "--save-lyrics",
    type=click.Path(path_type=Path),
    default=None,
    help="Save transcribed lyrics to this file for review/editing.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Save the A/V markdown table to this file instead of printing to stdout.",
)
def analyze(
    als_file: Path,
    structure_track: str,
    phrase_bars: int,
    show_events: bool,
    lyrics: Path | None,
    align_vocals: bool | None,
    vocal_track: tuple[str, ...],
    all_vocals: bool,
    save_vocals: Path | None,
    transcribe: bool,
    language: str,
    whisper_model: str,
    save_lyrics: Path | None,
    output: Path | None,
) -> None:
    """Analyze an Ableton Live Set file.

    Outputs phrase-level chunks of N bars each (default: 2 bars).
    Track enter/exit events are detected by default.
    Use --no-events to disable event detection.

    LYRICS SUPPORT:

    The tool supports multiple lyrics input methods:

    1. Timestamped lyrics (LRC, simple timed, enhanced LRC):
       When a lyrics file contains timestamps, they are used directly
       without requiring forced alignment. No audio extraction needed.

       Example formats:
         LRC: [00:12.34]First line
         Simple: 0:12.34 First line
         Enhanced: [00:12.34]<00:12.34>First <00:12.80>line

    2. Plain text lyrics with forced alignment:
       When --lyrics points to a plain text file with [SECTION] headers,
       forced alignment with stable-ts is used (requires alignment deps).

    3. ASR transcription (--transcribe):
       Automatically transcribe lyrics from vocal audio using Whisper.
       Requires alignment dependencies: pip install 'alsmuse[align]'

    MUTUAL EXCLUSION:

    --transcribe and --lyrics cannot be used together. Use one or the other.
    """
    # Validate mutual exclusion: --transcribe and --lyrics cannot be used together
    if transcribe and lyrics is not None:
        click.echo(
            "Error: --transcribe and --lyrics are mutually exclusive. "
            "Use one or the other.",
            err=True,
        )
        sys.exit(1)

    # Default align_vocals to True when lyrics is provided
    if align_vocals is None:
        align_vocals = lyrics is not None

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

    # Validate transcription options
    if transcribe:
        # Check for required dependencies
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
            save_vocals_path=save_vocals,
            transcribe=transcribe,
            save_lyrics_path=save_lyrics,
            language=language,
            model_size=whisper_model,
        )

        if output is not None:
            output.write_text(result)
            click.echo(f"A/V table saved to: {output}")
        else:
            click.echo(result)
    except ParseError as e:
        click.echo(f"Error parsing file: {e}", err=True)
        sys.exit(1)
    except TrackNotFoundError as e:
        click.echo(f"Track not found: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
