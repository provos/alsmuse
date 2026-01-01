# ALSmuse v2 Design Document

## Overview

Extend ALSmuse to provide phrase-level A/V scripts with instrumentation change detection, analyzing actual MIDI note content and audio file characteristics.

## Design Principles

1. **Analyze Content, Not Containers**: Clip boundaries are insufficient. We must examine MIDI notes and audio waveforms.
2. **Delta Events**: Report changes ("Bass enters") not state ("Bass is playing").
3. **Phrase-Based Output**: Target 2-4 bar chunks (~3-5 seconds) for video editing.
4. **Layered Analysis**: Start with cheap analysis (MIDI notes), defer expensive analysis (audio transients).

## Architecture Extension

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Application Layer                        │
│                     (analyze.py)                            │
└───────┬─────────────┬─────────────┬─────────────┬───────────┘
        │             │             │             │
┌───────▼───────┐ ┌───▼───┐ ┌───────▼───────┐ ┌───▼───────────┐
│   Parser      │ │Phrases│ │   Events      │ │   Audio       │
│  (parser.py)  │ │       │ │ (events.py)   │ │ (audio.py)    │
└───────────────┘ └───────┘ └───────────────┘ └───────────────┘
```

## New Domain Models

### Extended Models (`models.py`)

```python
@dataclass(frozen=True)
class MidiNote:
    """A single MIDI note event."""
    time: float          # Beats relative to clip start
    duration: float      # Beats
    velocity: int        # 0-127
    pitch: int           # MIDI note number (for drums: specific instrument)

@dataclass(frozen=True)
class MidiClipContent:
    """Content of a MIDI clip including notes."""
    clip: Clip
    notes: tuple[MidiNote, ...]

    def has_notes_in_range(self, start: float, end: float) -> bool:
        """
        Check if any notes are active within the given range (relative to clip).

        A note is active in the range if:
        - It starts within the range, OR
        - It ends within the range, OR
        - It spans the entire range (starts before, ends after)

        This avoids the "stroboscope" problem of point sampling, where
        off-beat notes or fills between sample points would be missed.
        """
        for note in self.notes:
            note_start = note.time
            note_end = note.time + note.duration
            # Check for any overlap between note and range
            if note_start < end and note_end > start:
                return True
        return False

    def note_density(self) -> float:
        """Notes per beat - indicates activity level."""
        if not self.notes:
            return 0.0
        duration = self.clip.end_beats - self.clip.start_beats
        return len(self.notes) / duration if duration > 0 else 0.0

@dataclass(frozen=True)
class TrackEvent:
    """A significant change in track activity."""
    beat: float
    track_name: str
    event_type: Literal["enter", "exit", "accent"]
    category: str  # "drums", "bass", "vocals", etc.

@dataclass(frozen=True)
class Phrase:
    """A time slice with associated events and metadata."""
    start_beats: float
    end_beats: float
    section_name: str
    is_section_start: bool
    events: tuple[TrackEvent, ...]
    lyric: str = ""

    def start_time(self, bpm: float) -> float:
        return self.start_beats * 60 / bpm

    def duration_seconds(self, bpm: float) -> float:
        return (self.end_beats - self.start_beats) * 60 / bpm
```

## MIDI Analysis (`midi.py`)

### Extracting MIDI Notes from ALS

The ALS XML structure for MIDI notes:
```xml
<MidiClip Id="6" Time="28">
  <Notes>
    <KeyTracks>
      <KeyTrack Id="0">
        <MidiKey Value="36"/>  <!-- Kick drum -->
        <Notes>
          <MidiNoteEvent Time="0" Duration="0.5" Velocity="97"/>
          <MidiNoteEvent Time="2" Duration="0.5" Velocity="100"/>
        </Notes>
      </KeyTrack>
    </KeyTracks>
  </Notes>
</MidiClip>
```

### XML Structure Variations

MIDI notes can appear in different locations depending on track type:

**Drum Racks (KeyTracks structure):**
```xml
<MidiClip>
  <Notes>
    <KeyTracks>
      <KeyTrack Id="0">
        <MidiKey Value="36"/>
        <Notes>
          <MidiNoteEvent Time="0" Duration="0.5" Velocity="97"/>
        </Notes>
      </KeyTrack>
    </KeyTracks>
  </Notes>
</MidiClip>
```

**Standard MIDI tracks (direct Notes):**
```xml
<MidiClip>
  <Notes>
    <KeyTracks>
      <KeyTrack Id="0">
        <MidiKey Value="60"/>
        <Notes>
          <MidiNoteEvent Time="0" Duration="1.0" Velocity="100"/>
        </Notes>
      </KeyTrack>
    </KeyTracks>
  </Notes>
</MidiClip>
```

### Implementation

```python
def extract_midi_notes(clip_element: Element) -> tuple[MidiNote, ...]:
    """
    Extract all MIDI notes from a clip element.

    Handles both Drum Rack (KeyTracks) and standard MIDI track structures.
    Uses recursive search to find MidiNoteEvent elements regardless of
    exact XML path.

    Note times are relative to clip start. To get absolute beat position:
        absolute_beat = clip.start_beats + note.time
    """
    notes = []

    # Strategy 1: KeyTracks structure (Drum Racks, most MIDI clips)
    for key_track in clip_element.findall(".//KeyTrack"):
        midi_key_elem = key_track.find("MidiKey")
        pitch = int(midi_key_elem.get("Value", 0)) if midi_key_elem is not None else 0

        for note_event in key_track.findall(".//MidiNoteEvent"):
            if note_event.get("IsEnabled") == "false":
                continue
            notes.append(MidiNote(
                time=float(note_event.get("Time", 0)),
                duration=float(note_event.get("Duration", 0)),
                velocity=int(note_event.get("Velocity", 100)),
                pitch=pitch,
            ))

    # Strategy 2: Fallback - find any MidiNoteEvent not already captured
    # (handles edge cases in XML structure variations)
    if not notes:
        for note_event in clip_element.findall(".//MidiNoteEvent"):
            if note_event.get("IsEnabled") == "false":
                continue
            notes.append(MidiNote(
                time=float(note_event.get("Time", 0)),
                duration=float(note_event.get("Duration", 0)),
                velocity=int(note_event.get("Velocity", 100)),
                pitch=0,  # Unknown pitch in fallback
            ))

    return tuple(sorted(notes, key=lambda n: n.time))

def detect_midi_activity(
    track: Track,
    clip_contents: list[MidiClipContent],
    resolution_beats: float = 8.0,  # Check every 2 bars
) -> list[tuple[float, bool]]:
    """
    Detect track activity using range queries, not point sampling.

    IMPORTANT: Uses range queries to avoid the "stroboscope" problem.
    Point sampling at beat 0, 8, 16... would miss activity between
    sample points (e.g., a 1-bar drum fill or off-beat bass line).

    Returns list of (window_start_beat, is_active) tuples.
    """
    if not clip_contents:
        return []

    start = min(c.clip.start_beats for c in clip_contents)
    end = max(c.clip.end_beats for c in clip_contents)

    activity = []
    beat = start
    while beat < end:
        window_start = beat
        window_end = beat + resolution_beats

        # Check if ANY note activity occurs within this window
        is_active = False
        for content in clip_contents:
            # Check if window overlaps with clip
            if content.clip.start_beats < window_end and content.clip.end_beats > window_start:
                # Convert to clip-relative coordinates
                relative_start = max(0, window_start - content.clip.start_beats)
                relative_end = min(
                    content.clip.end_beats - content.clip.start_beats,
                    window_end - content.clip.start_beats
                )
                if content.has_notes_in_range(relative_start, relative_end):
                    is_active = True
                    break

        activity.append((beat, is_active))
        beat += resolution_beats

    return activity
```

### MIDI-Based Event Detection

```python
def detect_midi_events(
    track_name: str,
    activity: list[tuple[float, bool]],
    category: str,
) -> list[TrackEvent]:
    """
    Convert activity samples to enter/exit events.
    """
    events = []
    was_active = False

    for beat, is_active in activity:
        if is_active and not was_active:
            events.append(TrackEvent(
                beat=beat,
                track_name=track_name,
                event_type="enter",
                category=category,
            ))
        elif not is_active and was_active:
            events.append(TrackEvent(
                beat=beat,
                track_name=track_name,
                event_type="exit",
                category=category,
            ))
        was_active = is_active

    return events
```

## Audio Analysis (`audio.py`)

### Audio File Resolution

ALS stores audio file references:
```xml
<SampleRef>
  <FileRef>
    <RelativePath Value="Samples/Bounce Verse Main.wav"/>
    <Path Value="/Users/.../Bounce Verse Main.wav"/>
  </FileRef>
</SampleRef>
```

### Implementation

```python
from pathlib import Path
import librosa
import numpy as np

@dataclass(frozen=True)
class AudioAnalysis:
    """Analysis results for an audio file."""
    file_path: Path
    duration_seconds: float
    rms_envelope: np.ndarray  # Energy over time
    onset_times: np.ndarray   # Transient positions in seconds

def analyze_audio_file(
    file_path: Path,
    hop_length: int = 512,
) -> AudioAnalysis:
    """
    Analyze audio file for energy and transients.
    """
    y, sr = librosa.load(file_path, sr=None)

    # RMS energy envelope
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    return AudioAnalysis(
        file_path=file_path,
        duration_seconds=len(y) / sr,
        rms_envelope=rms,
        onset_times=onset_times,
    )

def detect_audio_activity(
    analysis: AudioAnalysis,
    clip: Clip,
    bpm: float,
    threshold: float = 0.1,  # RMS threshold for "active"
) -> list[tuple[float, bool]]:
    """
    Determine when audio is meaningfully playing (not silence).
    """
    # Convert clip beats to seconds
    clip_duration = (clip.end_beats - clip.start_beats) * 60 / bpm

    # Sample the RMS envelope
    activity = []
    samples_per_beat = len(analysis.rms_envelope) / (analysis.duration_seconds * bpm / 60)

    for beat_offset in range(int(clip.end_beats - clip.start_beats)):
        sample_idx = int(beat_offset * samples_per_beat)
        if sample_idx < len(analysis.rms_envelope):
            is_active = analysis.rms_envelope[sample_idx] > threshold
            activity.append((clip.start_beats + beat_offset, is_active))

    return activity
```

## Phrase Subdivision (`phrases.py`)

```python
def subdivide_sections(
    sections: list[Section],
    beats_per_phrase: int = 8,  # 2 bars in 4/4
) -> list[Phrase]:
    """
    Split sections into phrase-sized chunks.

    Args:
        sections: List of sections from extractor
        beats_per_phrase: Target phrase length (default 8 = 2 bars)

    Returns:
        List of Phrases covering the same time range
    """
    phrases = []

    for section in sections:
        current_beat = section.start_beats
        is_first = True

        while current_beat < section.end_beats:
            end_beat = min(current_beat + beats_per_phrase, section.end_beats)
            phrases.append(Phrase(
                start_beats=current_beat,
                end_beats=end_beat,
                section_name=section.name if is_first else "...",
                is_section_start=is_first,
                events=(),
                lyric="",
            ))
            current_beat = end_beat
            is_first = False

    return phrases
```

## Event Aggregation (`events.py`)

### Track Categorization

```python
TRACK_CATEGORIES: dict[str, list[str]] = {
    "drums": ["kick", "snare", "drum", "hat", "cymbal", "perc", "tom"],
    "bass": ["bass", "sub"],
    "vocals": ["vocal", "vox", "verse", "chorus", "main", "double", "harmony"],
    "lead": ["lead", "solo", "melody"],
    "guitar": ["guitar", "gtr"],
    "keys": ["piano", "keys", "organ", "synth"],
    "pad": ["pad", "strings", "atmosphere"],
    "fx": ["fx", "riser", "downlifter", "reverse", "sweep", "impact"],
}

def categorize_track(track_name: str) -> str:
    """Determine track category from name."""
    name_lower = track_name.lower()
    for category, keywords in TRACK_CATEGORIES.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return "other"
```

### Merging Events into Phrases

```python
def merge_events_into_phrases(
    phrases: list[Phrase],
    events: list[TrackEvent],
) -> list[Phrase]:
    """
    Attach events to the phrases they occur in.

    Events are grouped by category for cleaner output:
    "Drums enter, Bass enters" instead of "KICK enters, SNARE enters, Bass enters"
    """
    # Sort events by beat
    events = sorted(events, key=lambda e: e.beat)

    result = []
    event_idx = 0

    for phrase in phrases:
        phrase_events = []

        while event_idx < len(events) and events[event_idx].beat < phrase.end_beats:
            if events[event_idx].beat >= phrase.start_beats:
                phrase_events.append(events[event_idx])
            event_idx += 1

        # Deduplicate by category
        seen_categories: dict[str, TrackEvent] = {}
        for event in phrase_events:
            key = (event.category, event.event_type)
            if key not in seen_categories:
                seen_categories[key] = event

        result.append(Phrase(
            start_beats=phrase.start_beats,
            end_beats=phrase.end_beats,
            section_name=phrase.section_name,
            is_section_start=phrase.is_section_start,
            events=tuple(seen_categories.values()),
            lyric=phrase.lyric,
        ))

    return result
```

## Lyrics Integration (`lyrics.py`)

```python
def parse_lyrics_file(path: Path) -> dict[str, list[str]]:
    """
    Parse a lyrics file with section headers.

    Format:
    ```
    [VERSE1]
    First line
    Second line

    [CHORUS]
    Chorus line one
    Chorus line two
    ```

    Returns:
        Dict mapping section names to lists of lyric lines
    """
    lyrics: dict[str, list[str]] = {}
    current_section = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].upper()
                lyrics[current_section] = []
            elif line and current_section:
                lyrics[current_section].append(line)

    return lyrics

def distribute_lyrics(
    phrases: list[Phrase],
    section_lyrics: dict[str, list[str]],
) -> list[Phrase]:
    """
    Distribute lyrics across phrases within each section.

    Uses line-count heuristic: if section has 4 phrases and 2 lyric lines,
    lines go in phrases 1 and 3.
    """
    result = []
    current_section = None
    section_phrases: list[Phrase] = []

    for phrase in phrases:
        if phrase.is_section_start:
            # Process previous section
            if section_phrases and current_section:
                result.extend(_apply_lyrics(section_phrases, section_lyrics.get(current_section, [])))
            current_section = phrase.section_name
            section_phrases = [phrase]
        else:
            section_phrases.append(phrase)

    # Process final section
    if section_phrases and current_section:
        result.extend(_apply_lyrics(section_phrases, section_lyrics.get(current_section, [])))

    return result

def _apply_lyrics(phrases: list[Phrase], lyrics: list[str]) -> list[Phrase]:
    """Apply lyrics to phrases with even distribution."""
    if not lyrics:
        return phrases

    step = max(1, len(phrases) // len(lyrics))
    result = []
    lyric_idx = 0

    for i, phrase in enumerate(phrases):
        lyric = ""
        if i % step == 0 and lyric_idx < len(lyrics):
            lyric = lyrics[lyric_idx]
            lyric_idx += 1

        result.append(Phrase(
            start_beats=phrase.start_beats,
            end_beats=phrase.end_beats,
            section_name=phrase.section_name,
            is_section_start=phrase.is_section_start,
            events=phrase.events,
            lyric=lyric,
        ))

    return result
```

## Extended Formatter (`formatter.py`)

```python
def format_phrase_table(
    phrases: list[Phrase],
    bpm: float,
    show_events: bool = True,
    show_lyrics: bool = False,
) -> str:
    """
    Format phrases as markdown A/V table.
    """
    # Determine columns
    headers = ["Time", "Cue"]
    if show_events:
        headers.append("Events")
    if show_lyrics:
        headers.append("Lyrics")
    headers.append("Video")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("------" for _ in headers) + "|",
    ]

    for phrase in phrases:
        time_str = format_time(phrase.start_time(bpm))
        cue = phrase.section_name

        row = [time_str, cue]

        if show_events:
            events_str = format_events(phrase.events)
            row.append(events_str)

        if show_lyrics:
            row.append(f'"{phrase.lyric}"' if phrase.lyric else "")

        row.append("")  # Empty Video column

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

def format_events(events: tuple[TrackEvent, ...]) -> str:
    """Format events as compact string."""
    if not events:
        return ""

    parts = []
    for event in events:
        verb = "enters" if event.event_type == "enter" else "exits"
        parts.append(f"{event.category.title()} {verb}")

    return ", ".join(parts)
```

## CLI Updates (`cli.py`)

```python
@main.command()
@click.argument("als_file", type=click.Path(exists=True, path_type=Path))
@click.option("--structure-track", default="STRUCTURE", help="Name of structure track")
@click.option("--phrase-bars", default=2, type=int, help="Bars per phrase (default: 2)")
@click.option("--show-events/--no-events", default=True, help="Show track enter/exit events")
@click.option("--lyrics", type=click.Path(exists=True, path_type=Path), help="Lyrics file")
@click.option("--analyze-audio/--no-audio", default=False, help="Analyze audio files (slow)")
def analyze(
    als_file: Path,
    structure_track: str,
    phrase_bars: int,
    show_events: bool,
    lyrics: Path | None,
    analyze_audio: bool,
):
    """Analyze an Ableton Live Set file and generate A/V script."""
    ...
```

## Example Output

```
| Time | Cue | Events | Video |
|------|-----|--------|-------|
| 0:12 | INTRO | Pad enters | |
| 0:15 | ... | Drums enters | |
| 0:19 | ... | | |
| 0:22 | ... | Bass enters | |
| 0:27 | VERSE1 | Vocals enters | |
| 0:30 | ... | | |
| 0:33 | ... | Lead enters | |
| 0:37 | ... | | |
| 0:40 | ... | Drums exits | |
```

With lyrics:
```
| Time | Cue | Events | Lyrics | Video |
|------|-----|--------|--------|-------|
| 0:27 | VERSE1 | Vocals enters | "Walking down the street" | |
| 0:30 | ... | | | |
| 0:33 | ... | Lead enters | "Feeling the beat" | |
```

## Implementation Order

### Phase 1: Phrase Subdivision (No new parsing)
1. Add `Phrase` model to `models.py`
2. Create `phrases.py` with `subdivide_sections()`
3. Update `formatter.py` with `format_phrase_table()`
4. Add `--phrase-bars` CLI option

### Phase 2: MIDI Event Detection
1. Add `MidiNote`, `MidiClipContent` to `models.py`
2. Create `midi.py` with note extraction
3. Create `events.py` with track categorization and event detection
4. Add `--show-events` CLI option

### Phase 3: Lyrics Integration
1. Create `lyrics.py` with parser and distribution
2. Add `--lyrics` CLI option

### Phase 4: Audio Analysis (Optional/Future)
1. Create `audio.py` with librosa integration
2. Add `--analyze-audio` CLI option
3. Requires resolving audio file paths from ALS

## Known Limitations

### 1. Pickup Beats Attribution
Events at the very end of a phrase (e.g., a pickup snare fill into a chorus)
will be listed in the **preceding** phrase, not the phrase they lead into.

**Example:**
```
Beat 63.5: Snare fill (pickup into chorus at beat 64)
```
This fill appears in the VERSE1 phrase (beats 56-64), not the CHORUS phrase.

**Rationale:** This is acceptable for an automated tool. The editor can mentally
associate a fill at the end of a phrase with the upcoming section change.

### 2. Phrase Boundary Alignment
Phrases are subdivided mathematically from section start. If a section doesn't
start on a phrase boundary, sub-phrases may not align with musical phrasing.

**Mitigation:** Ensure structure track clips are placed on bar boundaries.

### 3. MIDI-Only Event Detection
Event detection currently relies on MIDI note data. Audio tracks without
corresponding MIDI (e.g., bounced stems) require the `--analyze-audio` option
for activity detection.

### 4. Track Categorization Heuristics
Track categorization uses keyword matching on track names. Tracks with
non-standard names (e.g., "Track 7") will be categorized as "other".

**Mitigation:** Use descriptive track names in Ableton projects.

## Module Structure (v2)

```
src/alsmuse/
├── __init__.py
├── cli.py           # CLI entry point
├── analyze.py       # Orchestration (updated)
├── parser.py        # ALS file parsing (extended for MIDI notes)
├── extractors.py    # Section extraction
├── phrases.py       # NEW: Phrase subdivision
├── midi.py          # NEW: MIDI note analysis
├── events.py        # NEW: Track event detection
├── lyrics.py        # NEW: Lyrics parsing
├── audio.py         # NEW: Audio file analysis (future)
├── formatter.py     # Output formatting (extended)
├── models.py        # Domain models (extended)
└── exceptions.py    # Exceptions
```
