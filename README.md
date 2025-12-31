# ALSmuse

Analyze Ableton Live sets to understand musical structure for music video screenplay writing.

## Overview

ALSmuse parses Ableton Live Set (.als) files to extract timing, track, and clip information. It helps identify when musical elements rise and fall, making it easier to plan music video scenes that sync with the music.

## Installation

```bash
uv sync
```

## Usage

```bash
uv run alsmuse analyze path/to/your/song.als
```

## Features (Planned)

- Parse Ableton Live Set (.als) files
- Extract track and clip timeline information
- Identify musical sections (intro, verse, chorus, bridge, outro)
- Analyze volume/intensity curves per track
- Generate visual timeline for screenplay planning
- Export timing data in various formats

## Development

```bash
# Install dependencies
uv sync

# Run the CLI
uv run alsmuse --help
```

## License

Apache 2.0
