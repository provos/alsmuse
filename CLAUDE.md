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

- **src/alsmuse/cli.py**: Click-based CLI with command group structure. Entry point is `main()`.
- **src/alsmuse/__init__.py**: Package init with version.

The project uses:
- **click** for CLI framework
- **rich** for terminal output formatting
- **librosa** for audio analysis
- **matplotlib/plotly** for visualization

## Code Style

- Python 3.12+
- Ruff for linting (line length 100)
- Ruff lint rules: E, F, I, UP, B, SIM
