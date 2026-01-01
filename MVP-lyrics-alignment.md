# MVP: Automated Lyrics Timing via Forced Alignment

## Goal

Automatically generate precise word-level timestamps for lyrics by aligning untimed lyrics text against vocal audio extracted from ALS files.

## Use Case

User has:
- An ALS project with vocal tracks (bounced stems or recorded audio)
- A plain text lyrics file (untimed)

User wants:
- Lyrics with precise timestamps for each line/word
- Integration with existing phrase-based A/V output

## Technical Approach

### Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   ALS File      │────▶│  Extract Vocal  │────▶│  Continuous     │
│   (gzipped XML) │     │  Audio Refs     │     │  Audio File     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐             │
│   Lyrics File   │────▶│  Forced         │◀────────────┘
│   (plain text)  │     │  Alignment      │
└─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Timed Lyrics   │
                        │  (word-level)   │
                        └─────────────────┘
```

### Step 1: Extract Audio File References from ALS

ALS XML structure for audio references:

```xml
<AudioClip Time="64">
  <SampleRef>
    <FileRef>
      <RelativePath Value="Samples/Recorded/Verse 1 Vox.wav"/>
      <Path Value="/Users/.../Samples/Recorded/Verse 1 Vox.wav"/>
    </FileRef>
  </SampleRef>
</AudioClip>
```

Implementation:
```python
def extract_audio_clips(als_path: Path) -> list[AudioClipRef]:
    """
    Extract audio clip references with their timeline positions.

    Returns:
        List of AudioClipRef(
            track_name="Verse 1 Vox",
            file_path=Path("/Users/.../Verse 1 Vox.wav"),
            start_beats=64.0,
            end_beats=128.0,
        )
    """
```

### Step 2: Identify and Combine Vocal Tracks

Use track name heuristics (already in `events.py`):

```python
VOCAL_KEYWORDS = ["vocal", "vox", "verse", "chorus", "main", "double", "harmony", "lead vocal"]
```

Combine multiple vocal clips into continuous audio:

```python
def combine_vocal_clips(
    clips: list[AudioClipRef],
    bpm: float,
    output_path: Path,
) -> Path:
    """
    Combine vocal clips into a single audio file, preserving timeline positions.

    Uses pydub to:
    1. Create silent base track matching total duration
    2. Overlay each vocal clip at its correct position
    3. Export combined audio
    """
```

### Step 3: Forced Alignment with stable-ts

**Why stable-ts over WhisperX:**
- Simpler API for alignment with existing text
- Direct `model.align(audio, text)` function
- Same underlying Whisper models
- Fewer dependencies

**Installation:**
```bash
pip install stable-ts
```

**Usage:**
```python
import stable_whisper

def align_lyrics_to_audio(
    audio_path: Path,
    lyrics_text: str,
    language: str = "en",
) -> list[TimedWord]:
    """
    Force-align lyrics text to audio.

    Args:
        audio_path: Path to vocal audio file
        lyrics_text: Plain text lyrics (newlines = line breaks)
        language: Language code

    Returns:
        List of TimedWord(text="walking", start=27.5, end=27.9)
    """
    model = stable_whisper.load_model("base")  # or "small", "medium"
    result = model.align(str(audio_path), lyrics_text, language=language)

    words = []
    for segment in result.segments:
        for word in segment.words:
            words.append(TimedWord(
                text=word.word.strip(),
                start=word.start,
                end=word.end,
            ))
    return words
```

### Step 4: Map Timed Words to Sections/Phrases

```python
def distribute_timed_lyrics(
    phrases: list[Phrase],
    timed_words: list[TimedWord],
    bpm: float,
) -> list[Phrase]:
    """
    Assign timed lyrics to phrases based on timestamp overlap.

    Groups consecutive words that fall within each phrase's time window.
    """
```

## New Models

```python
@dataclass(frozen=True)
class AudioClipRef:
    """Reference to an audio file with timeline position."""
    track_name: str
    file_path: Path
    start_beats: float
    end_beats: float

@dataclass(frozen=True)
class TimedWord:
    """A word with start and end timestamps in seconds."""
    text: str
    start: float  # seconds
    end: float    # seconds

@dataclass(frozen=True)
class TimedLine:
    """A line of lyrics with timing."""
    text: str
    start: float
    end: float
    words: tuple[TimedWord, ...]
```

## New Modules

### `audio.py` - Audio Extraction

```python
def extract_audio_clips(als_path: Path) -> list[AudioClipRef]:
    """Extract audio clip references from ALS."""

def find_vocal_tracks(clips: list[AudioClipRef]) -> list[AudioClipRef]:
    """Filter to vocal tracks using name heuristics."""

def combine_clips_to_audio(
    clips: list[AudioClipRef],
    bpm: float,
    output_path: Path,
) -> Path:
    """Combine clips into single audio file."""
```

### `lyrics_align.py` - Forced Alignment

```python
def align_lyrics(
    audio_path: Path,
    lyrics_text: str,
    language: str = "en",
    model_size: str = "base",
) -> list[TimedWord]:
    """Force-align lyrics to audio using stable-ts."""

def words_to_lines(
    words: list[TimedWord],
    original_lines: list[str],
) -> list[TimedLine]:
    """Group words back into original line structure."""
```

## CLI Updates

```bash
# Basic (existing)
alsmuse analyze song.als --lyrics lyrics.txt

# With auto-alignment (new)
alsmuse analyze song.als --lyrics lyrics.txt --align-vocals

# Specify vocal track explicitly
alsmuse analyze song.als --lyrics lyrics.txt --align-vocals --vocal-track "Lead Vocal"

# Output aligned lyrics to LRC file
alsmuse analyze song.als --lyrics lyrics.txt --align-vocals --output-lrc aligned.lrc
```

## Dependencies

New optional dependencies:
```toml
[project.optional-dependencies]
align = [
    "stable-ts>=2.16.0",
    "pydub>=0.25.0",
]
```

Also requires `ffmpeg` for audio format conversion (pydub dependency).

## Example Output

Input lyrics.txt:
```
[VERSE1]
Walking down the empty street
Feeling rhythm in my feet
```

Output with `--align-vocals`:
```
| Time | Cue | Lyrics | Video |
|------|-----|--------|-------|
| 0:27 | VERSE1 | "Walking down the empty street" | |
| 0:30 | ... | "Feeling rhythm in my feet" | |
```

Output with `--output-lrc`:
```
[00:27.50]Walking down the empty street
[00:30.20]Feeling rhythm in my feet
```

## Implementation Order

### Phase 1: Audio Extraction
1. Parse audio clip references from ALS XML
2. Resolve file paths (absolute and relative)
3. Filter to vocal tracks by name
4. Test with example files

### Phase 2: Audio Combination
1. Add pydub dependency
2. Implement clip combination with timeline positioning
3. Handle different audio formats (wav, aif, mp3)
4. Export combined vocal track

### Phase 3: Forced Alignment
1. Add stable-ts dependency
2. Implement alignment wrapper
3. Test with clean vocal audio
4. Handle alignment failures gracefully

### Phase 4: Integration
1. Map timed words to phrases
2. Update formatter for word-level timestamps
3. Add CLI options
4. LRC export option

## Known Limitations

### 1. Vocal Track Identification
Relies on track naming conventions. User may need to specify vocal track explicitly if names don't match heuristics.

### 2. Audio Quality
Forced alignment works best with:
- Clean vocal recordings (minimal backing track bleed)
- Consistent tempo (we assume static BPM)
- Single language per track

### 3. Processing Time
Whisper models require significant processing:
- "base" model: ~1x realtime on CPU
- "small" model: ~0.5x realtime on GPU
- First run downloads model (~150MB for base)

### 4. Audio File Access
Audio files must be accessible at their stored paths. If project was moved, paths may be broken. We'll try:
1. Absolute path from `<Path>`
2. Relative path from ALS file location

### 5. Complex Arrangements
Multiple overlapping vocal takes, harmonies, or doubled vocals may confuse alignment. Best results with a single bounced vocal stem.

## Future Enhancements

1. **GPU acceleration**: Use CUDA if available
2. **Word-level display**: Show individual word timings in output
3. **Confidence scores**: Flag low-confidence alignments
4. **Vocal isolation**: Use Demucs/Spleeter if no clean vocal stem exists
5. **Multi-language**: Detect or specify language per section
