# Design: Lyrics Enhancements

This document describes two related enhancements to ALSmuse's lyrics handling:

1. **ASR Transcription**: Automatically transcribe lyrics from vocal audio when no lyrics file is provided
2. **Timestamped Lyrics**: Support lyrics files with embedded timestamps, eliminating the need for alignment

## Overview

Currently, ALSmuse supports:
- Plain text lyrics with optional `[SECTION]` headers (heuristic distribution)
- Forced alignment of plain lyrics to vocal audio (requires stable-ts)

These enhancements add:
- ASR transcription when no lyrics are provided
- Direct use of pre-timestamped lyrics (LRC format and simple timestamp format)

## Feature 1: ASR Transcription

### Motivation

Users may not have lyrics available, or may want to quickly generate a first draft of lyrics from audio. Since stable-ts (Whisper) is already a dependency for alignment, we can leverage its transcription capability.

### CLI Options

```
--transcribe          Transcribe lyrics from vocal audio using ASR.
                      Cannot be used with --lyrics.

--language TEXT       Language code for transcription/alignment (default: en).
                      Examples: en, es, fr, de, ja, ko, zh.

--whisper-model       Whisper model size (default: base).
                      Choices: tiny, base, small, medium, large.

--save-lyrics PATH    Save transcribed lyrics to this file for review/editing.
```

### Validation Rules

- `--transcribe` and `--lyrics` are mutually exclusive (error if both specified)
- `--transcribe` requires alignment dependencies (stable-ts)
- `--save-lyrics` only applies when `--transcribe` is used

### Implementation

#### Line Segmentation Strategy

**Important**: Whisper's transcription returns segment objects that correspond to
natural phrase/sentence boundaries. These segments are far more reliable than
heuristic-based line breaking (gap thresholds, max words, punctuation).

The implementation should:
1. **Use Whisper segments as primary line breaks** - Each segment becomes a line
2. **Only split long segments** - If a segment exceeds a reasonable length (e.g., 15+ words),
   apply heuristics to split it further
3. **Preserve segment timing** - Use segment start/end times for line timing

This approach leverages Whisper's natural language understanding rather than
relying on brittle heuristics.

#### New Model in `models.py`

```python
@dataclass(frozen=True)
class TimedSegment:
    """A transcribed segment with word-level timing.

    Represents a natural phrase/sentence boundary as detected by Whisper.
    """
    text: str
    start: float  # seconds
    end: float    # seconds
    words: tuple[TimedWord, ...]
```

#### New Functions in `lyrics_align.py`

```python
def transcribe_lyrics(
    audio_path: Path,
    valid_ranges: list[tuple[float, float]],
    language: str = "en",
    model_size: str = "base",
) -> tuple[list[TimedSegment], str]:
    """Transcribe lyrics from audio using Whisper ASR.

    Uses stable-ts for transcription, preserving segment boundaries.
    Filters results to only include content within known valid audio
    ranges (hallucination filtering).

    Args:
        audio_path: Path to combined vocal audio.
        valid_ranges: List of (start, end) tuples where real audio exists.
        language: Language code for transcription model.
        model_size: Whisper model size.

    Returns:
        Tuple of:
        - List of TimedSegment preserving Whisper's phrase boundaries.
        - Raw transcription text (for saving to file).

    Raises:
        AlignmentError: If stable-ts is not installed or transcription fails.
    """
    # Uses model.transcribe() instead of model.align()
    # Preserves segment structure from Whisper result
```

```python
def segments_to_lines(
    segments: list[TimedSegment],
    max_words_per_line: int = 15,
) -> list[TimedLine]:
    """Convert transcribed segments to lines, splitting long segments.

    Uses Whisper's segment boundaries as the primary line breaks.
    Only applies heuristic splitting to segments that exceed max_words_per_line.

    Algorithm:
    1. For each segment:
       a. If word count <= max_words_per_line: create single line from segment
       b. If word count > max_words_per_line: split on punctuation or mid-point
    2. Preserve word-level timing from original segment

    Args:
        segments: Transcribed segments from Whisper.
        max_words_per_line: Only split segments exceeding this word count.

    Returns:
        List of TimedLine, typically one per segment unless segment was split.
    """
```

```python
def filter_segments_to_valid_ranges(
    segments: list[TimedSegment],
    valid_ranges: list[tuple[float, float]],
) -> list[TimedSegment]:
    """Remove segments that fall outside known audio regions.

    A segment is kept if its midpoint falls within any valid range.
    This filters out Whisper hallucinations during silent gaps.

    Args:
        segments: All segments from transcription.
        valid_ranges: Time ranges where real audio exists.

    Returns:
        Filtered list with hallucinated segments removed.
    """
```

#### New Function in `analyze.py`

```python
def transcribe_and_distribute_lyrics(
    als_path: Path,
    phrases: list[Phrase],
    bpm: float,
    vocal_tracks: tuple[str, ...] | None = None,
    use_all_vocals: bool = False,
    save_vocals_path: Path | None = None,
    save_lyrics_path: Path | None = None,
    language: str = "en",
    model_size: str = "base",
) -> list[Phrase]:
    """Full transcription pipeline: extract audio, transcribe, distribute.

    Steps:
    1. Extract audio clips from ALS file
    2. Select vocal tracks
    3. Combine vocal clips to a single audio file
    4. Run ASR transcription with stable-ts (preserving segments)
    5. Filter segments to valid audio ranges (remove hallucinations)
    6. Convert segments to lines (splitting only if too long)
    7. Distribute timed lyrics to phrases
    """
```

### Usage Examples

```bash
# Basic transcription
alsmuse analyze song.als --transcribe

# Transcription with specific vocal tracks
alsmuse analyze song.als --transcribe --vocal-track "Lead Vocal"

# Transcription in Spanish with larger model
alsmuse analyze song.als --transcribe --language es --whisper-model medium

# Save transcription for review and editing
alsmuse analyze song.als --transcribe --save-lyrics lyrics.txt
```

---

## Feature 2: Timestamped Lyrics

### Motivation

Users may already have timestamped lyrics from:
- Karaoke software (LRC files)
- Manual annotation
- Other transcription tools
- Music databases

Supporting timestamped input eliminates alignment overhead and allows users to use pre-existing timed lyrics.

### Supported Formats

#### LRC Format (Standard)

```
[00:12.34]First line of lyrics
[00:15.67]Second line of lyrics
[00:45.12]Line after instrumental break
```

Features:
- Timestamps in `[mm:ss.xx]` or `[mm:ss.xxx]` format
- Optional metadata tags: `[ar:Artist]`, `[ti:Title]`, `[al:Album]`
- Multiple timestamps per line for word-level sync (enhanced LRC)

#### Simple Timestamp Format

```
0:12.34 First line of lyrics
0:15.67 Second line of lyrics
0:45.12 Line after instrumental break
```

Features:
- Timestamps at start of line: `m:ss.xx` or `mm:ss.xx`
- More readable for manual editing
- No special markers needed

#### Enhanced LRC (Word-Level)

```
[00:12.34]<00:12.34>First <00:12.80>line <00:13.10>of <00:13.30>lyrics
[00:15.67]<00:15.67>Second <00:16.00>line
```

Features:
- Line timestamps in `[mm:ss.xx]`
- Word timestamps in `<mm:ss.xx>`
- Provides word-level timing without alignment

### Auto-Detection

The lyrics parser will auto-detect the format:
1. If lines start with `[mm:ss` pattern -> LRC format
2. If lines start with `m:ss` or `mm:ss` pattern -> Simple timestamp format
3. If lines contain `<mm:ss` inline -> Enhanced LRC with word timing
4. Otherwise -> Plain text (existing behavior)

### Implementation

#### New Functions in `lyrics.py`

```python
def detect_lyrics_format(content: str) -> LyricsFormat:
    """Detect the format of lyrics content.

    Args:
        content: Raw lyrics file content.

    Returns:
        LyricsFormat enum: PLAIN, LRC, LRC_ENHANCED, SIMPLE_TIMED
    """


def parse_lrc_lyrics(content: str) -> list[TimedLine]:
    """Parse LRC format lyrics into TimedLine objects.

    Handles:
    - Standard LRC: [mm:ss.xx]text
    - Metadata tags: [ar:Artist] etc. (ignored)
    - Multiple timestamps: [00:12.34][00:45.67]repeated line

    Args:
        content: LRC format lyrics content.

    Returns:
        List of TimedLine with timestamps in seconds.
    """


def parse_simple_timed_lyrics(content: str) -> list[TimedLine]:
    """Parse simple timed format lyrics.

    Format: m:ss.xx text or mm:ss.xx text

    Args:
        content: Simple timed lyrics content.

    Returns:
        List of TimedLine with timestamps in seconds.
    """


def parse_enhanced_lrc(content: str) -> list[TimedLine]:
    """Parse enhanced LRC with word-level timestamps.

    Format: [mm:ss.xx]<mm:ss.xx>word <mm:ss.xx>word

    Args:
        content: Enhanced LRC content.

    Returns:
        List of TimedLine with word-level TimedWord objects.
    """


def parse_lyrics_file_auto(path: Path) -> tuple[list[TimedLine] | None, dict[str, list[str]] | None]:
    """Parse lyrics file with auto-format detection.

    Args:
        path: Path to lyrics file.

    Returns:
        Tuple of:
        - List of TimedLine if timestamps detected, None otherwise
        - Section lyrics dict if plain text with sections, None otherwise

    At least one will be non-None. If timestamps are detected,
    section headers are ignored and timed lines are returned.
    """
```

#### New Model

```python
class LyricsFormat(Enum):
    """Format of lyrics file."""
    PLAIN = "plain"              # No timestamps
    LRC = "lrc"                  # Standard LRC [mm:ss.xx]
    LRC_ENHANCED = "lrc_enhanced"  # LRC with word timing <mm:ss.xx>
    SIMPLE_TIMED = "simple_timed"  # m:ss.xx at line start
```

#### Updated Flow in `analyze.py`

```python
def analyze_als_v2(...):
    # ... existing code ...

    show_lyrics = False
    if lyrics_path is not None:
        # Try to parse as timed lyrics first
        timed_lines, section_lyrics = parse_lyrics_file_auto(lyrics_path)

        if timed_lines is not None:
            # Lyrics have timestamps - use directly, no alignment needed
            phrases = distribute_timed_lyrics(phrases, timed_lines, bpm)
            show_lyrics = True
        elif align_vocals:
            # Plain lyrics with alignment requested
            # ... existing alignment code ...
        else:
            # Plain lyrics, heuristic distribution
            phrases = distribute_lyrics(phrases, section_lyrics)
            show_lyrics = True
```

### CLI Behavior

No new CLI options needed. The behavior is automatic:

| Lyrics Format | `--align-vocals` | Behavior |
|---------------|------------------|----------|
| Timestamped (LRC/simple) | ignored | Use timestamps directly |
| Plain text | True (default) | Run forced alignment |
| Plain text | False (`--no-align-vocals`) | Heuristic distribution |

### Usage Examples

```bash
# LRC file - timestamps used directly, no alignment
alsmuse analyze song.als --lyrics song.lrc

# Simple timed format - timestamps used directly
alsmuse analyze song.als --lyrics timed_lyrics.txt

# Plain text - will align (default when --lyrics specified)
alsmuse analyze song.als --lyrics plain_lyrics.txt

# Plain text - skip alignment, use heuristics
alsmuse analyze song.als --lyrics plain_lyrics.txt --no-align-vocals
```

### Example Lyrics Files

#### LRC Format (`song.lrc`)
```
[ar:Artist Name]
[ti:Song Title]
[00:00.00]
[00:12.34]Deep inside me
[00:15.67]Feel my heartbeat
[00:18.90]Today you will
[00:21.23]Make my heart bleed
[00:45.00]
[00:47.34]Verse two starts here
```

#### Simple Timed Format (`lyrics.txt`)
```
0:12.34 Deep inside me
0:15.67 Feel my heartbeat
0:18.90 Today you will
0:21.23 Make my heart bleed

0:47.34 Verse two starts here
```

#### Enhanced LRC with Word Timing
```
[00:12.34]<00:12.34>Deep <00:12.80>inside <00:13.20>me
[00:15.67]<00:15.67>Feel <00:16.00>my <00:16.30>heartbeat
```

---

## Implementation Plan

### Phase 1: Timestamped Lyrics Parsing
1. Add `LyricsFormat` enum to `models.py`
2. Add format detection to `lyrics.py`
3. Add LRC parser to `lyrics.py`
4. Add simple timed parser to `lyrics.py`
5. Add enhanced LRC parser to `lyrics.py`
6. Add `parse_lyrics_file_auto()` to `lyrics.py`
7. Update `analyze_als_v2()` to use auto-detection
8. Write tests for all new parsers

### Phase 2: ASR Transcription
1. Add `TimedSegment` model to `models.py`
2. Add `transcribe_lyrics()` to `lyrics_align.py` (returns segments, not just words)
3. Add `filter_segments_to_valid_ranges()` to `lyrics_align.py`
4. Add `segments_to_lines()` to `lyrics_align.py` (uses segment boundaries, only splits long segments)
5. Add `transcribe_and_distribute_lyrics()` to `analyze.py`
6. Update `analyze_als_v2()` signature for transcribe mode
7. Add CLI options to `cli.py`
8. Add mutual exclusion validation
9. Write tests for transcription functions

### Phase 3: Integration and Polish
1. Update docstrings throughout
2. Add integration tests
3. Update help text

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid LRC timestamp format | Warning, skip line, continue |
| Empty lyrics file | Error: "Lyrics file is empty" |
| `--transcribe` with `--lyrics` | Error: mutually exclusive |
| No vocal tracks for transcription | Error with suggestion to use `--vocal-track` |
| Transcription fails | Error, exit (no fallback) |
| Mixed formats in file | Use detected dominant format |

---

## Future Enhancements (Out of Scope)

1. **LRC export**: Export aligned/transcribed lyrics as LRC file
2. **SRT support**: Parse SubRip subtitle format
3. **ASS/SSA support**: Parse Advanced SubStation Alpha format
4. **Confidence scores**: Show transcription confidence per word
5. **Interactive correction**: TUI for fixing transcription errors
6. **Speaker diarization**: Identify different singers
