# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ALSmuse analyzes Ableton Live Set (.als) files to extract musical structure information for music video screenplay planning. It identifies timing, track, clip, and intensity data to help sync video scenes with music.

## Commands

```bash
# Install dependencies
uv sync

# Run the CLI
uv run alsmuse --help
uv run alsmuse analyze path/to/song.als

# Lint
uv run ruff check src/
uv run ruff format src/

# Type check
uv run mypy src/
```

## Architecture

See `DESIGN.md` for full architecture. Summary:

```
CLI (cli.py) → Application (analyze.py) → Parser + Extractors + Formatter
                                                    ↓
                                            Domain Models (models.py)
```

### Modules
- **cli.py**: Click-based CLI, thin wrapper
- **analyze.py**: Orchestrates parsing → extraction → formatting pipeline
- **parser.py**: Parses ALS files (gzipped XML) into domain models
- **extractors.py**: Section extraction strategies (Protocol pattern)
- **formatter.py**: Output formatting (markdown A/V tables)
- **models.py**: Immutable dataclasses (LiveSet, Track, Clip, Section, Tempo)

### Key Dependencies
- **click**: CLI framework
- **rich**: Terminal output formatting
- **librosa**: Audio analysis (future)
- **matplotlib/plotly**: Visualization (future)

## Code Style

- Python 3.12+
- Ruff for linting (line length 100)
- Ruff lint rules: E, F, I, UP, B, SIM

## Research

Use mgrep for web searches when you need external information:

```bash
mgrep --web --answer "<natural english question>"
```

Examples:
- `mgrep --web --answer "Ableton Live ALS XML format structure"`
- `mgrep --web --answer "Python dataclass frozen immutable best practices"`
