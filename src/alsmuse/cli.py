"""Command-line interface for ALSmuse."""

import os
import sys
from pathlib import Path

# Prevent tokenizers parallelism warning when forking for Whisper alignment
# Must be set before importing anything that uses tokenizers (model2vec, transformers, etc.)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import click

from .analyze import analyze_als, detect_suggested_start_bar
from .config import MuseConfig, load_config, save_config
from .exceptions import ParseError, TrackNotFoundError
from .parser import parse_als_file
from .start_bar_ui import select_start_bar_interactive


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
    type=click.Choice(["tiny", "base", "small", "medium", "large", "large-v3"]),
    default="base",
    help="Whisper model size for transcription/alignment (default: base).",
)
@click.option(
    "--save-lyrics",
    type=click.Path(path_type=Path),
    default=None,
    help="Save lyrics to this file in LRC format with timestamps. "
    "Works with --transcribe or --align-vocals. "
    "Can be edited and reused with --lyrics to avoid re-processing.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path. Extension determines format: "
    ".md or no extension outputs markdown A/V table, "
    ".mp4 outputs video visualization.",
)
@click.option(
    "--audio",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Audio file to mux into video (only used for .mp4 output).",
)
@click.option(
    "--start-bar",
    type=int,
    default=None,
    help="Bar number where the song starts (times will be relative to this bar). "
    "If not specified, auto-detects and prompts interactively. Saved to config.",
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
    audio: Path | None,
    start_bar: int | None,
) -> None:
    """Analyze an Ableton Live Set file.

    Outputs phrase-level chunks of N bars each (default: 2 bars).
    Track enter/exit events are detected by default.
    Use --no-events to disable event detection.

    OUTPUT FORMATS:

    The output format is determined by the -o/--output option:

    - No output option: Print markdown A/V table to stdout
    - .md extension: Write markdown A/V table to file
    - .mp4 extension: Generate video visualization

    EXAMPLES:

        # Print A/V table to stdout
        alsmuse analyze song.als

        # Save A/V table to markdown file
        alsmuse analyze song.als -o script.md

        # Generate video visualization
        alsmuse analyze song.als -o output.mp4

        # Generate video with audio track
        alsmuse analyze song.als -o output.mp4 --audio song.wav

        # Generate video with transcribed lyrics
        alsmuse analyze song.als -o output.mp4 --transcribe

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
            "Error: --transcribe and --lyrics are mutually exclusive. Use one or the other.",
            err=True,
        )
        sys.exit(1)

    # Default align_vocals to True when lyrics is provided
    if align_vocals is None:
        align_vocals = lyrics is not None

    # Validate alignment options
    if align_vocals and lyrics is None:
        click.echo(
            "Error: --align-vocals requires --lyrics to be specified.",
            err=True,
        )
        sys.exit(1)

    # Determine if this is video output
    is_video_output = output is not None and output.suffix.lower() == ".mp4"

    # Warn if --audio is used without video output
    if audio is not None and not is_video_output:
        click.echo(
            "Warning: --audio is only used for .mp4 output. Ignoring.",
            err=True,
        )

    try:
        # Load existing config
        config = load_config(als_file)

        # Determine effective start bar:
        # 1. CLI argument takes precedence
        # 2. Fall back to config value
        # 3. Auto-detect and prompt if interactive
        effective_start_bar = start_bar
        start_bar_from_config = config.start_bar if config else None

        if effective_start_bar is None and start_bar_from_config is not None:
            effective_start_bar = start_bar_from_config
            click.echo(f"Using saved start bar: {effective_start_bar}")

        # Auto-detect and prompt if still not set and interactive
        if effective_start_bar is None:
            live_set = parse_als_file(als_file)
            suggested = detect_suggested_start_bar(live_set)

            if sys.stdin.isatty() and suggested > 0:
                # Use rich interactive UI for start bar selection
                effective_start_bar = select_start_bar_interactive(live_set, suggested)

                # Save the chosen start bar to config
                updated_config = MuseConfig(
                    vocal_tracks=config.vocal_tracks if config else [],
                    category_overrides=config.category_overrides if config else {},
                    start_bar=effective_start_bar if effective_start_bar else None,
                )
                save_config(als_file, updated_config)
            elif suggested > 0:
                # Non-interactive but suggested start is non-zero
                click.echo(
                    f"Note: Song appears to start at bar {suggested}. "
                    f"Use --start-bar {suggested} to adjust times.",
                    err=True,
                )

        # Convert bars to beats (assuming 4/4 time)
        beats_per_phrase = phrase_bars * 4

        # Convert vocal_track tuple to None if empty
        vocal_tracks = vocal_track if vocal_track else None

        # Show effective start bar if non-zero
        if effective_start_bar and effective_start_bar > 0:
            click.echo(
                f"Using start bar: {effective_start_bar} (times relative to bar {effective_start_bar})"
            )

        # Define progress callback for video output
        progress_callback = None
        if is_video_output:

            def progress_callback(frame_num: int, total_frames: int) -> None:
                """Display rendering progress."""
                if frame_num % 24 == 0 or frame_num == total_frames:
                    percent = (frame_num / total_frames) * 100
                    click.echo(
                        f"\rRendering: {frame_num}/{total_frames} frames ({percent:.1f}%)",
                        nl=False,
                    )
                if frame_num == total_frames:
                    click.echo()  # Final newline

        # Call unified analyze_als function
        result = analyze_als(
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
            output_path=output,
            audio_path=audio,
            progress_callback=progress_callback,
            start_bar=effective_start_bar,
        )

        # Handle output based on result type
        if isinstance(result, Path):
            # Video was generated
            click.echo(f"Video generated: {result}")
        elif output is not None:
            # Markdown output to file
            output.write_text(result)
            click.echo(f"A/V table saved to: {output}")
        else:
            # Markdown output to stdout
            click.echo(result)

    except ParseError as e:
        click.echo(f"Error parsing file: {e}", err=True)
        sys.exit(1)
    except TrackNotFoundError as e:
        click.echo(f"Track not found: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Error generating video: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
