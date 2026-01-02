# ALSmuse

[![CI](https://github.com/provos/alsmuse/actions/workflows/ci.yml/badge.svg)](https://github.com/provos/alsmuse/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Generate A/V scripts and visualizations from Ableton Live Sets for music video planning.

ALSmuse analyzes Ableton Live Set (.als) files to extract musical structure, track events, and lyrics timing. It produces either markdown tables for A/V (audio/visual) scripts or video visualizations showing song structure with animated lyrics — helping directors and editors sync video scenes with music.

## Example Output

```markdown
| Time | Audio | Video |
|------|-------|-------|
| 0:17 | **VERSE1**<br>Bass enters, Keys enters<br>*"Our Connection / Undefined"* |  |
| 0:22 | **...**<br>*"Can you see me / Only sometimes"* |  |
| 0:35 | **...**<br>Drums enters<br>*"Can you hear it now, Cause it's getting loud"* |  |
| 0:52 | **CHORUS1**<br>Pad enters, Keys exits<br>*"Feel My Heartbeat"* |  |
```

## Features

- **Section Detection**: Extracts song structure (INTRO, VERSE, CHORUS, etc.) from a dedicated STRUCTURE track
- **Phrase-Level Timing**: Breaks sections into customizable phrase lengths (default: 2 bars)
- **Track Events**: Detects when instruments enter and exit
- **Lyrics Integration**: Supports multiple lyrics input methods:
  - Plain text with `[SECTION]` headers
  - LRC format with timestamps
  - Simple timed format (`0:12.34 Lyric line`)
  - ASR transcription from vocal audio
- **Forced Alignment**: Aligns plain text lyrics to audio using Whisper for precise timing
- **LRC Export**: Save aligned or transcribed lyrics in LRC format for reuse
- **Video Visualization**: Generate MP4 videos showing song structure with animated lyrics, optionally muxed with audio

## Installation

### Basic Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### With Lyrics Alignment

For faster lyrics transcription on Apple silicon, install with optional dependencies:

```bash
# Apple Silicon Macs (uses mlx-whisper, much faster)
uv sync --extra align-mlx
# or: pip install -e ".[align-mlx]"
```

## Usage

### Basic Analysis

```bash
# Analyze an Ableton Live Set (prints A/V table to stdout)
alsmuse analyze song.als

# Customize phrase length (default: 2 bars)
alsmuse analyze song.als --phrase-bars 4

# Save A/V table to markdown file
alsmuse analyze song.als -o av_table.md

# Hide track events
alsmuse analyze song.als --no-events
```

### Video Visualization

Generate an MP4 video showing song structure with animated lyrics:

```bash
# Generate video visualization
alsmuse analyze song.als -o output.mp4

# Include audio in the video
alsmuse analyze song.als -o output.mp4 --audio song.wav

# Video with transcribed lyrics
alsmuse analyze song.als -o output.mp4 --transcribe

# Video with provided lyrics
alsmuse analyze song.als -o output.mp4 --lyrics lyrics.txt
```

The output format is determined by the file extension:
- No extension or `.md` → Markdown A/V table
- `.mp4` → Video visualization

### With Lyrics

```bash
# Plain text lyrics with section headers (uses forced alignment)
alsmuse analyze song.als --lyrics lyrics.txt

# Timestamped lyrics (LRC format) - no alignment needed
alsmuse analyze song.als --lyrics song.lrc

# Save aligned lyrics with timestamps for reuse
alsmuse analyze song.als --lyrics lyrics.txt --save-lyrics aligned.lrc
```

**Plain text lyrics format:**
```
[VERSE1]
First verse line
Second verse line

[CHORUS]
Chorus lyrics here
```

### ASR Transcription

Automatically transcribe lyrics from vocal tracks:

```bash
# Transcribe vocals using Whisper
alsmuse analyze song.als --transcribe

# Specify language and model size
alsmuse analyze song.als --transcribe --language es --whisper-model medium

# Save transcription for review (saved as LRC with timestamps)
alsmuse analyze song.als --transcribe --save-lyrics transcribed.lrc
```

### Vocal Track Selection

When multiple vocal tracks are detected:

```bash
# Use all detected vocal tracks
alsmuse analyze song.als --transcribe --all-vocals

# Specify tracks explicitly
alsmuse analyze song.als --transcribe --vocal-track "Lead Vox" --vocal-track "Backing"

# Save combined vocals for validation
alsmuse analyze song.als --transcribe --save-vocals combined.wav
```

## How It Works

1. **Parse ALS File**: Reads the gzipped XML structure of Ableton Live Sets
2. **Extract Sections**: Finds clips on the STRUCTURE track to identify song sections
3. **Detect Events**: Analyzes when tracks become active/inactive across phrases
4. **Process Lyrics**: Aligns or distributes lyrics to the timeline
5. **Format Output**: Generates a markdown A/V table

### Structure Track Convention

ALSmuse expects a MIDI track named `STRUCTURE` (customizable via `--structure-track`) containing clips that define song sections:

```
STRUCTURE track:
[INTRO  ][VERSE1      ][CHORUS    ][VERSE2      ][CHORUS    ][OUTRO]
```

Each clip's name becomes the section name in the output.

### Track Categorization

ALSmuse uses semantic matching to categorize tracks by their names. Track names are compared against categories like Drums, Bass, Vocals, Lead, Guitar, Keys, Pad, and FX to determine what type of instrument each track represents.

When a track enters or exits, the event is labeled by category (e.g., "Bass enters") rather than the specific track name. This keeps the A/V table clean and focused on musical changes rather than implementation details.

Tracks that don't clearly match any category (generic names like "Track 1" or "Audio 2", or ambiguous names) are excluded from event detection. If you don't use descriptive names, the category mapping will likely fail.

## CLI Reference

```
Usage: alsmuse analyze [OPTIONS] ALS_FILE

Options:
  --structure-track TEXT          Name of the structure track (default: STRUCTURE)
  --phrase-bars INTEGER           Bars per phrase (default: 2)
  --show-events / --no-events     Show track enter/exit events (default: on)
  --lyrics PATH                   Path to lyrics file
  --align-vocals / --no-align-vocals
                                  Force alignment for plain text lyrics
  --vocal-track TEXT              Specific vocal track(s) to use (repeatable)
  --all-vocals                    Use all detected vocal tracks
  --save-vocals PATH              Save combined vocals to file
  --transcribe                    Transcribe lyrics from audio
  --language TEXT                 Language code (default: en)
  --whisper-model [tiny|base|small|medium|large]
                                  Whisper model size (default: base)
  --save-lyrics PATH              Save lyrics to file (LRC format with timestamps)
  -o, --output PATH               Output file (.md for A/V table, .mp4 for video)
  --audio PATH                    Audio file to include in video (only for .mp4)
  --start-bar INTEGER             Bar number where the song starts (default: 1)
  --help                          Show this message and exit
```

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Type check
uv run mypy src/
```

## Architecture

```
CLI (cli.py) --> Application (analyze.py) --> Parser + Extractors + Formatter/Visualizer
                                                      |
                                              Domain Models (models.py)
```

- **cli.py**: Click-based command-line interface
- **analyze.py**: Orchestrates the analysis pipeline, routes to markdown or video output
- **parser.py**: Parses ALS files (gzipped XML) into domain models
- **extractors.py**: Section extraction strategies
- **formatter.py**: Markdown table output formatting
- **visualizer.py**: Video generation with animated lyrics
- **audio.py**: Audio extraction and vocal track handling
- **lyrics_align.py**: Lyrics parsing, alignment, and transcription
- **models.py**: Immutable dataclasses for domain objects

## Requirements

- Python 3.12+
- All core dependencies (including stable-ts for transcription) are installed by default
- Optional: mlx-whisper for faster transcription on Apple Silicon (`pip install 'alsmuse[align-mlx]'`)

## License

Apache 2.0
