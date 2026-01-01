# ALSmuse

Generate A/V script tables from Ableton Live Sets for music video planning.

ALSmuse analyzes Ableton Live Set (.als) files to extract musical structure, track events, and lyrics timing. It produces markdown tables that serve as the "Audio" column of an A/V (audio/visual) script, helping directors and editors sync video scenes with music.

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

## Installation

### Basic Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### With Lyrics Alignment

For lyrics alignment and transcription features, install with optional dependencies:

```bash
# Cross-platform (uses stable-ts, slower on CPU)
uv sync --extra align
# or: pip install -e ".[align]"

# Apple Silicon Macs (uses mlx-whisper, much faster)
uv sync --extra align-mlx
# or: pip install -e ".[align-mlx]"
```

## Usage

### Basic Analysis

```bash
# Analyze an Ableton Live Set
alsmuse analyze song.als

# Customize phrase length (default: 2 bars)
alsmuse analyze song.als --phrase-bars 4

# Save output to file
alsmuse analyze song.als -o av_table.md

# Hide track events
alsmuse analyze song.als --no-events
```

### With Lyrics

```bash
# Plain text lyrics with section headers
alsmuse analyze song.als --lyrics lyrics.txt

# Timestamped lyrics (LRC format) - no alignment needed
alsmuse analyze song.als --lyrics song.lrc
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

# Save transcription for review
alsmuse analyze song.als --transcribe --save-lyrics transcribed.txt
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
  --save-lyrics PATH              Save transcribed lyrics to file
  -o, --output PATH               Save A/V table to file
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
CLI (cli.py) --> Application (analyze.py) --> Parser + Extractors + Formatter
                                                      |
                                              Domain Models (models.py)
```

- **cli.py**: Click-based command-line interface
- **analyze.py**: Orchestrates the analysis pipeline
- **parser.py**: Parses ALS files (gzipped XML) into domain models
- **extractors.py**: Section extraction strategies
- **formatter.py**: Markdown table output formatting
- **audio.py**: Audio extraction and vocal track handling
- **lyrics_align.py**: Lyrics parsing, alignment, and transcription
- **models.py**: Immutable dataclasses for domain objects

## Requirements

- Python 3.12+
- For lyrics alignment: stable-ts or mlx-whisper, soundfile, questionary

## License

Apache 2.0
