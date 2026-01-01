# Design: Automated Lyrics Alignment

## Overview

Automatically generate precise word-level timestamps for lyrics by force-aligning untimed lyrics text against vocal audio extracted from ALS files.

## Design Principles

1. **Relative Paths First**: ALS projects move between computers; relative paths are the only reliable reference.
2. **Silence-Aware Filtering**: We construct the audio, so we know where valid vocals exist—use this to filter hallucinations.
3. **Graceful Degradation**: If alignment fails, fall back to heuristic distribution.
4. **Optional Heavy Dependencies**: `stable-ts` and `soundfile` are optional; core functionality works without them.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                       │
│                    alsmuse analyze --align-vocals                            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                           Application Layer                                  │
│                      analyze.py (orchestration)                              │
└───────┬─────────────────┬─────────────────┬─────────────────┬───────────────┘
        │                 │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│    Parser     │ │    Audio      │ │  Lyrics       │ │   Alignment   │
│  (parser.py)  │ │  (audio.py)   │ │ (lyrics.py)   │ │(lyrics_align) │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
```

## Domain Models

### New Models (`models.py`)

```python
@dataclass(frozen=True)
class AudioClipRef:
    """Reference to an audio file with timeline position.

    Attributes:
        track_name: Name of the containing track.
        file_path: Resolved path to the audio file.
        start_beats: Start position on timeline in beats.
        end_beats: End position on timeline in beats.
        start_seconds: Start position in seconds (computed from BPM).
        end_seconds: End position in seconds (computed from BPM).
    """
    track_name: str
    file_path: Path
    start_beats: float
    end_beats: float
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class TimedWord:
    """A word with start and end timestamps in seconds."""
    text: str
    start: float  # seconds
    end: float    # seconds


@dataclass(frozen=True)
class TimedLine:
    """A line of lyrics with timing derived from word timestamps."""
    text: str
    start: float  # seconds (from first word)
    end: float    # seconds (from last word)
    words: tuple[TimedWord, ...]
```

## Audio Extraction (`audio.py`)

### Path Resolution Strategy

ALS files store both absolute and relative paths:

```xml
<SampleRef>
  <FileRef>
    <RelativePathType Value="3"/>
    <RelativePath Value="Samples/Recorded/Verse Vox.wav"/>
    <Path Value="/Users/olduser/Music/Project/Samples/Recorded/Verse Vox.wav"/>
  </FileRef>
</SampleRef>
```

**Critical**: Absolute paths break when projects move. Always try relative path first.

```python
def resolve_audio_path(
    als_path: Path,
    relative_path: str,
    absolute_path: str,
) -> Path | None:
    """
    Resolve audio file path, prioritizing relative paths.

    Resolution order:
    1. Relative path from ALS file's parent directory
    2. Relative path from ALS file's grandparent (project root)
    3. Absolute path as fallback (may be stale)

    Args:
        als_path: Path to the .als file
        relative_path: Value from <RelativePath> element
        absolute_path: Value from <Path> element

    Returns:
        Resolved Path if file exists, None otherwise.
    """
    als_dir = als_path.parent

    # Strategy 1: Relative to ALS directory
    candidate = als_dir / relative_path
    if candidate.exists():
        return candidate

    # Strategy 2: Relative to project root (parent of ALS)
    # Common structure: Project/Project.als + Project/Samples/...
    project_root = als_dir.parent
    candidate = project_root / relative_path
    if candidate.exists():
        return candidate

    # Strategy 3: Try the stored absolute path (often stale)
    if absolute_path:
        candidate = Path(absolute_path)
        if candidate.exists():
            return candidate

    return None
```

### Audio Clip Extraction

```python
def extract_audio_clips(
    als_path: Path,
    bpm: float,
) -> list[AudioClipRef]:
    """
    Extract all audio clip references from an ALS file.

    Parses AudioClip elements and resolves their file paths.
    Skips clips with unresolvable paths (logs warning).

    Args:
        als_path: Path to the .als file
        bpm: Tempo for beat-to-seconds conversion

    Returns:
        List of AudioClipRef with resolved paths and timing.
    """
```

XML structure to parse:

```xml
<AudioTrack Id="5">
  <Name>
    <EffectiveName Value="Lead Vocal"/>
  </Name>
  <DeviceChain>
    <MainSequencer>
      <ClipTimeable>
        <ArrangerAutomation>
          <Events>
            <AudioClip Id="10" Time="64">
              <CurrentEnd Value="128"/>
              <SampleRef>
                <FileRef>
                  <RelativePath Value="Samples/Vocals/Lead.wav"/>
                  <Path Value="/old/path/Lead.wav"/>
                </FileRef>
              </SampleRef>
            </AudioClip>
          </Events>
        </ArrangerAutomation>
      </ClipTimeable>
    </MainSequencer>
  </DeviceChain>
</AudioTrack>
```

### Vocal Track Identification

```python
VOCAL_KEYWORDS: list[str] = [
    "vocal", "vox", "voice",
    "lead vocal", "main vocal",
    "verse", "chorus", "bridge",  # Often vocal sections
    "harmony", "double", "backing",
    "singer", "rap", "spoken",
]

def find_vocal_clips(
    clips: list[AudioClipRef],
    explicit_track: str | None = None,
) -> list[AudioClipRef]:
    """
    Filter audio clips to those likely containing vocals.

    Args:
        clips: All audio clips from ALS
        explicit_track: If provided, match this track name exactly

    Returns:
        Filtered list of vocal clips, sorted by start time.
    """
    if explicit_track:
        return sorted(
            [c for c in clips if c.track_name.lower() == explicit_track.lower()],
            key=lambda c: c.start_beats
        )

    return sorted(
        [c for c in clips if is_vocal_track(c.track_name)],
        key=lambda c: c.start_beats
    )


def is_vocal_track(track_name: str) -> bool:
    """Check if track name suggests vocal content."""
    name_lower = track_name.lower()
    return any(kw in name_lower for kw in VOCAL_KEYWORDS)


def get_unique_vocal_track_names(clips: list[AudioClipRef]) -> list[str]:
    """Get unique track names from vocal clips, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for clip in clips:
        if clip.track_name not in seen:
            seen.add(clip.track_name)
            result.append(clip.track_name)
    return result
```

### Interactive Track Selection

When multiple vocal tracks are detected, the user should be able to choose which
tracks to include in the alignment. This supports both interactive and non-interactive
workflows.

**CLI Options:**

```bash
# Non-interactive: explicit track selection (repeatable)
alsmuse analyze song.als --lyrics lyrics.txt --align-vocals \
    --vocal-track "Lead Vocal" --vocal-track "Verse Vox"

# Interactive: prompt when multiple tracks found (default)
alsmuse analyze song.als --lyrics lyrics.txt --align-vocals

# Skip prompt, use all detected vocal tracks
alsmuse analyze song.als --lyrics lyrics.txt --align-vocals --all-vocals
```

**Interactive Prompt (using questionary):**

```python
import questionary


def prompt_track_selection(
    track_names: list[str],
    auto_select_single: bool = True,
) -> list[str]:
    """
    Interactively prompt user to select vocal tracks.

    Args:
        track_names: List of detected vocal track names.
        auto_select_single: If True and only one track, skip prompt.

    Returns:
        List of selected track names.

    Raises:
        click.Abort: If user cancels selection.
    """
    if not track_names:
        return []

    # Single track: use it automatically
    if len(track_names) == 1 and auto_select_single:
        return track_names

    # Multiple tracks: interactive selection
    selected = questionary.checkbox(
        "Select vocal tracks to include:",
        choices=[
            questionary.Choice(name, checked=True)  # Pre-select all
            for name in track_names
        ],
    ).ask()

    if selected is None:
        # User cancelled (Ctrl+C)
        raise click.Abort()

    return selected
```

**Non-Interactive Fallback:**

When stdin is not a TTY (e.g., in scripts or CI), fall back to using all detected
tracks or require explicit `--vocal-track` options:

```python
import sys


def select_vocal_tracks(
    all_clips: list[AudioClipRef],
    explicit_tracks: tuple[str, ...] | None,
    use_all: bool,
) -> list[AudioClipRef]:
    """
    Select vocal tracks for alignment.

    Args:
        all_clips: All audio clips from ALS.
        explicit_tracks: Tracks specified via --vocal-track options.
        use_all: If True, use all detected vocal tracks without prompting.

    Returns:
        Filtered list of clips from selected tracks.
    """
    # Explicit selection via CLI
    if explicit_tracks:
        return [
            c for c in all_clips
            if c.track_name.lower() in [t.lower() for t in explicit_tracks]
        ]

    # Find all potential vocal tracks
    vocal_clips = [c for c in all_clips if is_vocal_track(c.track_name)]
    track_names = get_unique_vocal_track_names(vocal_clips)

    if not track_names:
        return []

    # Use all without prompting
    if use_all:
        return vocal_clips

    # Single track: use automatically
    if len(track_names) == 1:
        return vocal_clips

    # Interactive selection (if TTY available)
    if sys.stdin.isatty():
        selected_names = prompt_track_selection(track_names)
        return [c for c in vocal_clips if c.track_name in selected_names]

    # Non-TTY: use all and warn
    click.echo(
        f"Multiple vocal tracks found: {', '.join(track_names)}. "
        "Using all. Use --vocal-track to select specific tracks.",
        err=True,
    )
    return vocal_clips
```

### Audio Combination

```python
import numpy as np
import soundfile as sf


def combine_clips_to_audio(
    clips: list[AudioClipRef],
    output_path: Path,
) -> tuple[Path, list[tuple[float, float]]]:
    """
    Combine audio clips into a single file, preserving timeline positions.

    Creates a silent buffer and mixes each clip at its correct position.
    Returns both the combined audio path AND the valid time ranges where
    actual audio exists (for hallucination filtering).

    Uses only numpy + soundfile (no ffmpeg, no pydub).
    Supports: WAV, AIFF, FLAC (common Ableton formats).
    Does NOT support: MP3

    Args:
        clips: Audio clips to combine (must be sorted by start time)
        output_path: Where to write the combined audio

    Returns:
        Tuple of:
        - Path to combined audio file
        - List of (start_seconds, end_seconds) tuples for valid audio ranges

    Raises:
        ValueError: If clips list is empty or sample rates don't match.
        RuntimeError: If audio format is not supported.
    """
    if not clips:
        raise ValueError("No clips to combine")

    # Load first clip to get sample rate and channel count
    first_audio, sample_rate = sf.read(clips[0].file_path, dtype="float32")
    channels = first_audio.shape[1] if first_audio.ndim > 1 else 1

    # Calculate total duration in samples
    total_seconds = max(c.end_seconds for c in clips)
    total_samples = int(total_seconds * sample_rate)

    # Create silent buffer
    if channels > 1:
        combined = np.zeros((total_samples, channels), dtype=np.float32)
    else:
        combined = np.zeros(total_samples, dtype=np.float32)

    # Track valid audio ranges for hallucination filtering
    valid_ranges: list[tuple[float, float]] = []

    for clip in clips:
        audio, sr = sf.read(clip.file_path, dtype="float32")

        # Require matching sample rates (resampling adds complexity)
        if sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {clip.file_path} is {sr}Hz, "
                f"expected {sample_rate}Hz. Resample audio to match."
            )

        # Handle mono/stereo mismatch
        if channels > 1 and audio.ndim == 1:
            # Convert mono to stereo by duplicating
            audio = np.column_stack([audio, audio])
        elif channels == 1 and audio.ndim > 1:
            # Convert stereo to mono by averaging
            audio = audio.mean(axis=1)

        # Calculate position in samples
        start_sample = int(clip.start_seconds * sample_rate)
        clip_samples = min(len(audio), total_samples - start_sample)

        # Mix into combined buffer (additive for overlapping clips)
        combined[start_sample : start_sample + clip_samples] += audio[:clip_samples]

        # Record valid range
        valid_ranges.append((clip.start_seconds, clip.end_seconds))

    # Normalize to prevent clipping if peaks exceed 1.0
    peak = np.abs(combined).max()
    if peak > 1.0:
        combined /= peak

    # Export as WAV
    sf.write(output_path, combined, sample_rate)

    return output_path, valid_ranges
```

## Forced Alignment (`lyrics_align.py`)

### Compute Device Selection

PyTorch-based models (including Whisper) default to CPU if CUDA is not found.
They do **not** automatically use MPS on macOS. We must explicitly detect and
select the optimal device for 10-20x speedup on GPU/Neural Engine.

```python
import torch


def get_compute_device() -> str:
    """
    Detect the optimal compute device for model inference.

    Returns:
        "cuda" - NVIDIA GPU (Linux/Windows)
        "mps"  - Apple Metal Performance Shaders (macOS)
        "cpu"  - Fallback for all other systems
    """
    if torch.cuda.is_available():
        return "cuda"
    # CRITICAL: Explicit check for Metal Performance Shaders on macOS
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### Alignment with Hallucination Filtering

```python
def align_lyrics(
    audio_path: Path,
    lyrics_text: str,
    valid_ranges: list[tuple[float, float]],
    language: str = "en",
    model_size: str = "base",
) -> list[TimedWord]:
    """
    Force-align lyrics to audio, filtering hallucinations.

    Uses stable-ts for alignment, then filters results to only include
    words that fall within known valid audio ranges.

    Args:
        audio_path: Path to combined vocal audio
        lyrics_text: Plain text lyrics
        valid_ranges: List of (start, end) tuples where real audio exists
        language: Language code for alignment model
        model_size: Whisper model size ("tiny", "base", "small", "medium")

    Returns:
        List of TimedWord with hallucinations removed.
    """
    import stable_whisper

    # Select optimal compute device (GPU/MPS/CPU)
    device = get_compute_device()
    model = stable_whisper.load_model(model_size, device=device)

    result = model.align(str(audio_path), lyrics_text, language=language)

    # Extract all words with timestamps
    all_words: list[TimedWord] = []
    for segment in result.segments:
        for word in segment.words:
            all_words.append(TimedWord(
                text=word.word.strip(),
                start=word.start,
                end=word.end,
            ))

    # Filter to valid ranges only
    filtered_words = filter_to_valid_ranges(all_words, valid_ranges)

    return filtered_words


def filter_to_valid_ranges(
    words: list[TimedWord],
    valid_ranges: list[tuple[float, float]],
) -> list[TimedWord]:
    """
    Remove words that fall outside known audio regions.

    A word is kept if its midpoint falls within any valid range.
    This filters out Whisper hallucinations during silent gaps.

    Args:
        words: All words from alignment
        valid_ranges: Time ranges where real audio exists

    Returns:
        Filtered list with hallucinations removed.
    """
    def is_in_valid_range(word: TimedWord) -> bool:
        midpoint = (word.start + word.end) / 2
        return any(
            start <= midpoint <= end
            for start, end in valid_ranges
        )

    return [w for w in words if is_in_valid_range(w)]
```

### Word-to-Line Reconstruction

```python
def words_to_lines(
    words: list[TimedWord],
    original_lines: list[str],
) -> list[TimedLine]:
    """
    Reconstruct line structure from word timestamps.

    Matches timed words back to original lyric lines to preserve
    the user's line breaks.

    Algorithm:
    1. Normalize both word sequences (lowercase, strip punctuation)
    2. For each original line, consume matching words
    3. Line timing = first word start to last word end

    Args:
        words: Timed words from alignment
        original_lines: Original lyric lines from user's file

    Returns:
        List of TimedLine with word-level detail.
    """
```

## Integration with Existing Pipeline

### Updated `analyze.py`

```python
def analyze_als_v2(
    als_path: Path,
    structure_track: str = "STRUCTURE",
    beats_per_phrase: int = 8,
    show_events: bool = True,
    lyrics_path: Path | None = None,
    align_vocals: bool = False,
    vocal_track: str | None = None,
) -> str:
    """
    Analysis pipeline with optional vocal alignment.

    When align_vocals=True:
    1. Extract audio clips from ALS
    2. Find/combine vocal clips
    3. Run forced alignment on vocals + lyrics
    4. Map timed lyrics to phrases

    Falls back to heuristic distribution if alignment fails.
    """
    live_set = parse_als_file(als_path)
    bpm = live_set.tempo.bpm

    # ... existing section extraction and phrase subdivision ...

    if lyrics_path is not None:
        if align_vocals:
            try:
                phrases = align_and_distribute_lyrics(
                    als_path=als_path,
                    lyrics_path=lyrics_path,
                    phrases=phrases,
                    bpm=bpm,
                    vocal_track=vocal_track,
                )
            except AlignmentError as e:
                # Log warning, fall back to heuristic
                click.echo(f"Alignment failed: {e}, using heuristic", err=True)
                section_lyrics = parse_lyrics_file(lyrics_path)
                phrases = distribute_lyrics(phrases, section_lyrics)
        else:
            # Existing heuristic distribution
            section_lyrics = parse_lyrics_file(lyrics_path)
            phrases = distribute_lyrics(phrases, section_lyrics)

    # ... rest of pipeline ...


def align_and_distribute_lyrics(
    als_path: Path,
    lyrics_path: Path,
    phrases: list[Phrase],
    bpm: float,
    vocal_track: str | None = None,
) -> list[Phrase]:
    """
    Full alignment pipeline: extract audio, align, distribute.
    """
    from .audio import extract_audio_clips, find_vocal_clips, combine_clips_to_audio
    from .lyrics_align import align_lyrics, words_to_lines
    from .lyrics import parse_lyrics_file
    import tempfile

    # Step 1: Extract and filter audio clips
    all_clips = extract_audio_clips(als_path, bpm)
    vocal_clips = find_vocal_clips(all_clips, explicit_track=vocal_track)

    if not vocal_clips:
        raise AlignmentError("No vocal tracks found")

    # Step 2: Combine vocals into single audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        combined_path, valid_ranges = combine_clips_to_audio(
            vocal_clips,
            Path(tmp.name),
        )

    try:
        # Step 3: Parse and align lyrics
        lyrics_dict = parse_lyrics_file(lyrics_path)
        all_lyrics_text = "\n".join(
            line
            for lines in lyrics_dict.values()
            for line in lines
        )

        timed_words = align_lyrics(
            combined_path,
            all_lyrics_text,
            valid_ranges=valid_ranges,
        )

        # Step 4: Reconstruct lines and distribute to phrases
        original_lines = [
            line
            for lines in lyrics_dict.values()
            for line in lines
        ]
        timed_lines = words_to_lines(timed_words, original_lines)

        return distribute_timed_lyrics(phrases, timed_lines, bpm)

    finally:
        # Cleanup temp file
        Path(combined_path).unlink(missing_ok=True)
```

### Timed Lyrics Distribution

```python
def distribute_timed_lyrics(
    phrases: list[Phrase],
    timed_lines: list[TimedLine],
    bpm: float,
) -> list[Phrase]:
    """
    Assign timed lyrics to phrases based on timestamp overlap.

    Each phrase gets the lyrics whose timing falls primarily within
    that phrase's time window.

    Args:
        phrases: Phrase list with timing
        timed_lines: Lines with precise timestamps
        bpm: For converting phrase beats to seconds

    Returns:
        Phrases with lyric fields populated from alignment.
    """
    result: list[Phrase] = []
    line_idx = 0

    for phrase in phrases:
        phrase_start_sec = phrase.start_beats * 60 / bpm
        phrase_end_sec = phrase.end_beats * 60 / bpm

        # Collect lines that fall within this phrase
        phrase_lyrics: list[str] = []

        while line_idx < len(timed_lines):
            line = timed_lines[line_idx]
            line_midpoint = (line.start + line.end) / 2

            if line_midpoint < phrase_start_sec:
                # Line is before this phrase, skip
                line_idx += 1
            elif line_midpoint <= phrase_end_sec:
                # Line falls within phrase
                phrase_lyrics.append(line.text)
                line_idx += 1
            else:
                # Line is after this phrase, stop
                break

        result.append(Phrase(
            start_beats=phrase.start_beats,
            end_beats=phrase.end_beats,
            section_name=phrase.section_name,
            is_section_start=phrase.is_section_start,
            events=phrase.events,
            lyric=" / ".join(phrase_lyrics) if phrase_lyrics else "",
        ))

    return result
```

## CLI Updates

```python
@main.command()
@click.argument("als_file", type=click.Path(exists=True, path_type=Path))
@click.option("--structure-track", default="STRUCTURE")
@click.option("--phrase-bars", type=int, default=2)
@click.option("--show-events/--no-events", default=True)
@click.option("--lyrics", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--align-vocals",
    is_flag=True,
    help="Use forced alignment for precise lyrics timing (requires stable-ts).",
)
@click.option(
    "--vocal-track",
    type=str,
    multiple=True,
    help="Specific vocal track(s) to use. Can be repeated. Default: auto-detect.",
)
@click.option(
    "--all-vocals",
    is_flag=True,
    help="Use all detected vocal tracks without prompting.",
)
@click.option(
    "--output-lrc",
    type=click.Path(path_type=Path),
    default=None,
    help="Export aligned lyrics to LRC file.",
)
def analyze(...):
    """Analyze an Ableton Live Set file."""
```

## Error Handling

```python
class AlignmentError(Exception):
    """Raised when lyrics alignment fails."""
    pass


class AudioExtractionError(AlignmentError):
    """Raised when audio cannot be extracted from ALS."""
    pass


class NoVocalTracksError(AlignmentError):
    """Raised when no vocal tracks are found."""
    pass


class PathResolutionError(AlignmentError):
    """Raised when audio file paths cannot be resolved."""
    pass
```

## Dependencies

### pyproject.toml

```toml
[project.optional-dependencies]
align = [
    "stable-ts>=2.16.0",
    "soundfile>=0.12.0",
    "questionary>=2.0.0",  # Interactive track selection
]

# Combined with existing
all = [
    "alsmuse[align]",
]
```

### System Requirements

- ~150MB disk space for Whisper "base" model (downloaded on first use)
- ~500MB for "small" model, ~1.5GB for "medium"

No system dependencies required (ffmpeg not needed).

### Supported Audio Formats

| Format | Supported | Notes |
|--------|-----------|-------|
| WAV    | ✅ | Primary format for Ableton recordings |
| AIFF   | ✅ | Common on macOS |
| FLAC   | ✅ | Lossless compressed |
| MP3    | ❌ | Not supported - convert to WAV first |

### Dependency Validation

The CLI should validate dependencies before attempting alignment:

```python
def check_alignment_dependencies() -> list[str]:
    """
    Check that all required dependencies are available.

    Returns:
        List of error messages (empty if all dependencies satisfied).
    """
    errors = []

    # Check optional Python packages
    try:
        import stable_whisper  # noqa: F401
    except ImportError:
        errors.append(
            "stable-ts not installed. Run: pip install 'alsmuse[align]'"
        )

    try:
        import soundfile  # noqa: F401
    except ImportError:
        errors.append(
            "soundfile not installed. Run: pip install 'alsmuse[align]'"
        )

    return errors
```

Usage in CLI:

```python
if align_vocals:
    errors = check_alignment_dependencies()
    if errors:
        for err in errors:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)
```

## Module Structure

```
src/alsmuse/
├── __init__.py
├── cli.py              # CLI entry point (updated)
├── analyze.py          # Orchestration (updated)
├── parser.py           # ALS file parsing
├── extractors.py       # Section extraction
├── phrases.py          # Phrase subdivision
├── midi.py             # MIDI note analysis
├── events.py           # Track event detection
├── lyrics.py           # Lyrics parsing (existing)
├── lyrics_align.py     # NEW: Forced alignment
├── audio.py            # NEW: Audio extraction
├── formatter.py        # Output formatting
├── models.py           # Domain models (extended)
└── exceptions.py       # Exceptions (extended)
```

## Implementation Order

### Phase 1: Audio Extraction

1. Add `AudioClipRef` model
2. Implement `resolve_audio_path()` with relative-first strategy
3. Implement `extract_audio_clips()` to parse ALS XML
4. Write tests with mock ALS files
5. Test path resolution with moved projects

### Phase 2: Vocal Identification & Combination

1. Implement `find_vocal_clips()` with keyword matching
2. Add soundfile dependency (optional)
3. Implement `combine_clips_to_audio()` returning valid ranges
4. Test audio combination with real clips
5. Verify timeline positioning is correct

### Phase 3: Forced Alignment

1. Add stable-ts dependency (optional)
2. Implement `align_lyrics()` wrapper
3. Implement `filter_to_valid_ranges()` for hallucination filtering
4. Implement `words_to_lines()` for line reconstruction
5. Test with clean vocal audio

### Phase 4: Integration

1. Implement `align_and_distribute_lyrics()` orchestration
2. Implement `distribute_timed_lyrics()` for phrase mapping
3. Update CLI with new options
4. Add LRC export functionality
5. Graceful fallback on alignment failure

## Testing Strategy

### Unit Tests

- Path resolution with various project structures
- Hallucination filtering with synthetic valid ranges
- Word-to-line reconstruction

### Integration Tests

- Full pipeline with test ALS + audio files
- Fallback behavior when dependencies missing
- Fallback behavior when alignment fails

### Manual Testing

- Test with example.als, example2.als, example3.als
- Verify alignment quality with real vocals
- Check LRC export format

## Known Limitations

1. **Single Language**: Assumes consistent language throughout; no per-section language detection.

2. **Tempo Changes**: Assumes static BPM; projects with tempo automation will have drift.

3. **Overlapping Vocals**: Multiple simultaneous vocal takes may confuse alignment.

4. **Processing Time**: ~1x realtime on CPU for "base" model; ~3x for "medium".

5. **Model Download**: First run downloads Whisper model (~150MB base, ~500MB small).

6. **Clean Audio Preferred**: Works best with isolated vocals, not full mix.
