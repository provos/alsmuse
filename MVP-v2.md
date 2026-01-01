# ALSmuse MVP v2: Phrase-Level A/V Scripts

## Goal

Extend MVP v1 to provide phrase-level timing cues (2-4 bar chunks, ~3-5 seconds) with instrumentation change markers.

## Key Concepts

### The 2-Bar Rule

Most pop/electronic music uses 2-bar or 4-bar phrases. At common tempos:

- 1 bar (4 beats) @ 120 BPM = 2 seconds (fast/frenetic)
- 2 bars (8 beats) @ 120 BPM = 4 seconds (standard/groove)
- 4 bars (16 beats) @ 120 BPM = 8 seconds (atmospheric)

**Target**: ~4 second chunks = 2 bars at most tempos

### Delta Events (Not State)

We care about **changes**, not static state:

- ✗ "Bass is playing"
- ✓ "Bass enters" / "Bass exits"

## New Features

### 1. Phrase Subdivision

Automatically break sections into 2-bar chunks.

**Algorithm:**

```python
def subdivide_section(section: Section, bpm: float, bars_per_phrase: int = 2) -> list[Section]:
    """
    Split a section into phrase-sized chunks.

    Args:
        section: The section to subdivide
        bpm: Beats per minute
        bars_per_phrase: Target phrase length (default 2 bars)

    Returns:
        List of sub-sections, each ~4 seconds
    """
    beats_per_bar = 4  # Assuming 4/4 time
    beats_per_phrase = bars_per_phrase * beats_per_bar

    phrases = []
    current_beat = section.start_beats
    phrase_num = 1

    while current_beat < section.end_beats:
        end_beat = min(current_beat + beats_per_phrase, section.end_beats)
        name = f"{section.name}" if phrase_num == 1 else "..."
        phrases.append(Section(name=name, start_beats=current_beat, end_beats=end_beat))
        current_beat = end_beat
        phrase_num += 1

    return phrases
```

**Example Output (VERSE1 = 32 bars at 144 BPM):**

```
| Time | Cue |
|------|-----|
| 0:27 | VERSE1 |
| 0:30 | ... |
| 0:33 | ... |
| 0:37 | ... |
```

### 2. Instrumentation Diff Detection

Identify beats where track activity changes significantly.

**Algorithm:**

```python
def detect_track_events(live_set: LiveSet) -> list[TrackEvent]:
    """
    Find beats where tracks enter or exit.

    Returns list of events like:
        TrackEvent(beat=64, track="Bass", event_type="enter")
        TrackEvent(beat=120, track="Lead Guitar", event_type="exit")
    """
    events = []
    for track in live_set.tracks:
        for clip in track.clips:
            events.append(TrackEvent(
                beat=clip.start_beats,
                track=track.name,
                event_type="enter"
            ))
            events.append(TrackEvent(
                beat=clip.end_beats,
                track=track.name,
                event_type="exit"
            ))
    return sorted(events, key=lambda e: e.beat)
```

**Merge with Phrases:**

```
| Time | Cue | Events |
|------|-----|--------|
| 0:27 | VERSE1 | Bass enters, Drums enter |
| 0:30 | ... | |
| 0:33 | ... | Lead Guitar enters |
| 0:37 | ... | |
```

### 3. Lyrics Distribution (Line-Count Heuristic)

User provides lyrics per section, tool distributes across phrases.

**Input:**

```
--lyrics-file verse1.txt
```

Where `verse1.txt` contains:

```
Walking down the empty street
Feeling rhythm in my feet
Heart begins to race and pound
Lost inside this city sound
```

**Algorithm:**

```python
def distribute_lyrics(phrases: list[Section], lyrics: list[str]) -> list[tuple[Section, str]]:
    """
    Distribute lyrics lines evenly across phrases.

    If 4 lines and 8 phrases: lines go in phrases 1, 3, 5, 7
    If 4 lines and 4 phrases: one line per phrase
    """
    if not lyrics:
        return [(p, "") for p in phrases]

    # Calculate distribution
    step = max(1, len(phrases) // len(lyrics))
    result = []
    lyric_idx = 0

    for i, phrase in enumerate(phrases):
        if i % step == 0 and lyric_idx < len(lyrics):
            result.append((phrase, lyrics[lyric_idx]))
            lyric_idx += 1
        else:
            result.append((phrase, ""))

    return result
```

**Output:**

```
| Time | Cue | Lyrics |
|------|-----|--------|
| 0:27 | VERSE1 | "Walking down the empty street" |
| 0:30 | ... | |
| 0:33 | ... | "Feeling rhythm in my feet" |
| 0:37 | ... | |
```

## New Models

```python
@dataclass(frozen=True)
class TrackEvent:
    """A track entering or exiting at a specific beat."""
    beat: float
    track_name: str
    event_type: Literal["enter", "exit"]

@dataclass(frozen=True)
class Phrase:
    """A subdivision of a section with optional cues."""
    start_beats: float
    end_beats: float
    section_name: str  # Parent section name
    is_section_start: bool
    events: tuple[TrackEvent, ...]
    lyric: str
```

## New Module: `phrases.py`

```python
def subdivide_sections(
    sections: list[Section],
    bpm: float,
    beats_per_phrase: int = 8,  # 2 bars
) -> list[Phrase]:
    """Subdivide sections into phrase-sized chunks."""
    ...

def detect_events(live_set: LiveSet) -> list[TrackEvent]:
    """Find all track enter/exit events."""
    ...

def merge_events_into_phrases(
    phrases: list[Phrase],
    events: list[TrackEvent],
) -> list[Phrase]:
    """Attach events to the phrases they occur in."""
    ...
```

## New CLI Options

```bash
# Basic (MVP v1)
alsmuse analyze song.als

# With phrase subdivision
alsmuse analyze song.als --phrase-bars 2

# With instrumentation events
alsmuse analyze song.als --phrase-bars 2 --show-events

# With lyrics
alsmuse analyze song.als --phrase-bars 2 --lyrics lyrics.txt
```

## Example Full Output

```
| Time | Cue | Events | Lyrics |
|------|-----|--------|--------|
| 0:12 | INTRO | Pad enters | |
| 0:15 | ... | Drums enter | |
| 0:19 | ... | | |
| 0:22 | ... | Bass enters | |
| 0:27 | VERSE1 | Vocals enter | "Walking down..." |
| 0:30 | ... | | |
| 0:33 | ... | Lead Guitar enters | "Feeling rhythm..." |
...
```

## Implementation Order

1. **Phase 1**: Phrase subdivision (`phrases.py`)

   - `subdivide_sections()` function
   - Update formatter for phrase output
   - Add `--phrase-bars` CLI option

2. **Phase 2**: Track event detection

   - `TrackEvent` model
   - `detect_events()` function
   - `merge_events_into_phrases()` function
   - Add `--show-events` CLI option

3. **Phase 3**: Lyrics integration
   - Lyrics file parser (plain text, one line per line)
   - `distribute_lyrics()` function
   - Add `--lyrics` CLI option

## Track Categorization (Future)

For cleaner output, categorize tracks:

```python
TRACK_CATEGORIES = {
    "drums": ["kick", "snare", "drum", "hat", "cymbal"],
    "bass": ["bass"],
    "vocals": ["vocal", "vox", "verse", "chorus", "main", "double"],
    "lead": ["lead", "solo", "guitar lead"],
    "pad": ["pad", "synth pad", "strings"],
    "fx": ["fx", "riser", "downlifter", "reverse", "sweep"],
}
```

Then output: "Drums enter" instead of "KICK enters, SNARE enters, DRUMS enters"
