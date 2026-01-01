# ALSmuse Design Document

## Overview

ALSmuse generates the Audio column of an A/V script (music video treatment) from Ableton Live Set files. This document describes the software architecture.

## Design Principles

1. **Single Responsibility**: Each module does one thing well
2. **Dependency Inversion**: Core logic depends on abstractions, not concrete implementations
3. **Open/Closed**: Extensible for new section detection strategies without modifying existing code
4. **Explicit over Implicit**: No magic; clear data flow

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
│                      (cli.py - Click)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Application Layer                        │
│                     (analyze.py)                            │
│         Orchestrates parsing, extraction, formatting        │
└───────┬─────────────────┬─────────────────┬─────────────────┘
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│   ALS Parser  │ │   Section     │ │   Formatter   │
│  (parser.py)  │ │  Extractors   │ │ (formatter.py)│
│               │ │ (extractor.py)│ │               │
└───────┬───────┘ └───────┬───────┘ └───────────────┘
        │                 │
┌───────▼─────────────────▼───────────────────────────────────┐
│                      Domain Models                          │
│                      (models.py)                            │
│         LiveSet, Track, Clip, Section, Tempo                │
└─────────────────────────────────────────────────────────────┘
```

## Domain Models (`models.py`)

Immutable data classes representing the domain:

```python
@dataclass(frozen=True)
class Tempo:
    """
    Static tempo for the project.

    MVP Limitation: This represents a single global tempo. Files with
    tempo automation (BPM changes mid-song) will have inaccurate
    timestamps after the first tempo change. Future versions can
    extend this to support tempo events.
    """
    bpm: float
    time_signature: tuple[int, int]  # (numerator, denominator)

@dataclass(frozen=True)
class Clip:
    name: str
    start_beats: float
    end_beats: float

@dataclass(frozen=True)
class Track:
    name: str
    track_type: Literal["midi", "audio"]
    clips: tuple[Clip, ...]

@dataclass(frozen=True)
class LiveSet:
    tempo: Tempo
    tracks: tuple[Track, ...]

@dataclass(frozen=True)
class Section:
    name: str
    start_beats: float
    end_beats: float

    def start_time(self, bpm: float) -> float:
        """Convert start position to seconds."""
        return self.start_beats * 60 / bpm

    def end_time(self, bpm: float) -> float:
        """Convert end position to seconds."""
        return self.end_beats * 60 / bpm
```

## ALS Parser (`parser.py`)

Responsible for reading ALS files and producing a `LiveSet`:

```python
def parse_als_file(path: Path) -> LiveSet:
    """Parse an Ableton Live Set file into a LiveSet model."""
    ...

# Internal functions (private)
def _decompress_als(path: Path) -> bytes:
    """Decompress gzipped ALS file."""
    ...

def _extract_tempo(root: Element) -> Tempo:
    """Extract tempo from XML root."""
    ...

def _extract_tracks(root: Element) -> tuple[Track, ...]:
    """Extract all tracks from XML root."""
    ...

def _extract_track_name(track_element: Element) -> str:
    """
    Extract track name with fallback logic.

    Priority:
    1. UserName if non-empty
    2. EffectiveName as fallback

    This handles cases where UserName is "" but EffectiveName
    contains the auto-generated or sample-based name.
    """
    ...

def _extract_clips(track_element: Element) -> tuple[Clip, ...]:
    """Extract clips from a track element."""
    ...
```

**Design Notes:**
- Returns domain models, not XML elements
- Handles both MIDI and Audio tracks uniformly
- No knowledge of what sections are or how they're detected

### Track Name Resolution

ALS files store track names in two locations:
- `Name/UserName[@Value]`: User-assigned name (may be empty "")
- `Name/EffectiveName[@Value]`: Computed name (always populated)

**Resolution Logic:**
```python
def _extract_track_name(track_element: Element) -> str:
    name_elem = track_element.find("Name")
    user_name = name_elem.find("UserName").get("Value", "")
    if user_name:  # Non-empty string
        return user_name
    return name_elem.find("EffectiveName").get("Value", "")
```

## Section Extractors (`extractors.py`)

Strategy pattern for different section detection methods:

```python
from typing import Protocol

class SectionExtractor(Protocol):
    """Protocol for section extraction strategies."""

    def extract(self, live_set: LiveSet) -> list[Section]:
        """Extract sections from a LiveSet."""
        ...

class StructureTrackExtractor:
    """Extract sections from a dedicated structure track."""

    def __init__(self, track_name: str = "STRUCTURE"):
        self.track_name = track_name

    def extract(self, live_set: LiveSet) -> list[Section]:
        ...

class TrackNameInferenceExtractor:
    """Infer sections from track names (future)."""

    def extract(self, live_set: LiveSet) -> list[Section]:
        ...
```

**Design Notes:**
- Protocol allows easy addition of new strategies
- Each extractor is independent and testable
- Extractors handle gap detection and TRANSITION insertion

## Gap Handling

When there are gaps between clips, insert TRANSITION sections:

```python
def fill_gaps(sections: list[Section]) -> list[Section]:
    """Insert TRANSITION sections for gaps between clips."""
    ...
```

This is a pure function that transforms a list of sections.

## Formatter (`formatter.py`)

Converts sections to output format:

```python
def format_av_table(sections: list[Section], bpm: float) -> str:
    """Format sections as a markdown A/V table."""
    ...

def format_time(seconds: float) -> str:
    """Format seconds as M:SS or MM:SS."""
    ...
```

**Design Notes:**
- Pure functions, no side effects
- Easy to add new output formats (JSON, CSV, etc.)

## Application Layer (`analyze.py`)

Orchestrates the pipeline:

```python
def analyze_als(
    als_path: Path,
    structure_track: str = "STRUCTURE",
) -> str:
    """
    Main analysis pipeline.

    1. Parse ALS file
    2. Extract sections using appropriate strategy
    3. Fill gaps with transitions
    4. Format output
    """
    live_set = parse_als_file(als_path)

    extractor = StructureTrackExtractor(structure_track)
    sections = extractor.extract(live_set)
    sections = fill_gaps(sections)

    return format_av_table(sections, live_set.tempo.bpm)
```

## CLI Layer (`cli.py`)

Thin wrapper using Click:

```python
@click.command()
@click.argument("als_file", type=click.Path(exists=True, path_type=Path))
@click.option("--structure-track", default="STRUCTURE", help="Name of structure track")
def analyze(als_file: Path, structure_track: str):
    """Analyze an Ableton Live Set file."""
    result = analyze_als(als_file, structure_track)
    click.echo(result)
```

## Module Structure

```
src/alsmuse/
├── __init__.py
├── cli.py           # Click commands (entry point)
├── analyze.py       # Application orchestration
├── parser.py        # ALS file parsing
├── extractors.py    # Section extraction strategies
├── formatter.py     # Output formatting
└── models.py        # Domain data classes
```

## Error Handling

Define explicit exceptions:

```python
# exceptions.py
class ALSmuseError(Exception):
    """Base exception for ALSmuse."""

class ParseError(ALSmuseError):
    """Error parsing ALS file."""

class TrackNotFoundError(ALSmuseError):
    """Specified track not found in LiveSet."""
```

## Testing Strategy

Each module is independently testable:

- **models.py**: Test time conversion methods
- **parser.py**: Test with sample ALS XML (can use fixtures)
- **extractors.py**: Test with mock LiveSet objects
- **formatter.py**: Test output formatting
- **analyze.py**: Integration tests with real ALS files

## Implementation Notes

### Critical: Track Name Resolution
The `_extract_track_name` function in `parser.py` must implement fallback logic:
1. Check `UserName` first - use if non-empty
2. Fall back to `EffectiveName` if `UserName` is empty ("")

This is essential because some tracks (especially audio tracks derived from samples) have empty `UserName` but meaningful `EffectiveName`.

### MVP Limitation: Static Tempo
The MVP uses a single static BPM value. Files with tempo automation will produce inaccurate timestamps after the first tempo change point. This is an acceptable tradeoff for MVP simplicity.

**Future Enhancement Path:**
```python
@dataclass(frozen=True)
class TempoEvent:
    beat: float
    bpm: float

@dataclass(frozen=True)
class TempoMap:
    events: tuple[TempoEvent, ...]

    def beat_to_seconds(self, beat: float) -> float:
        """Convert beat to seconds, accounting for tempo changes."""
        ...
```

## Future Extensions

The architecture supports these without major changes:

1. **New Extractors**: Add `TrackNameInferenceExtractor`, `ClipActivityExtractor`
2. **New Formatters**: Add `format_json()`, `format_csv()`
3. **Audio Analysis**: Add `audio.py` module that uses librosa with file refs from parser
4. **Tempo Automation**: Extend `Tempo` model to handle tempo changes over time

## Dependencies

- **click**: CLI framework
- **rich**: Terminal output (optional, for pretty printing)
- Standard library: gzip, xml.etree.ElementTree, dataclasses, typing
