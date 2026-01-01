# Design: Phrase-Aligned Event Detection

## Status: Implemented

Phase-aligned detection is now implemented. However, during testing we discovered a separate bug.

---

# Bug: MIDI Note StartRelative Offset

## Problem

MIDI notes are stored with times relative to the clip's internal content, but clips can start playback from an offset position using `StartRelative`. We weren't accounting for this.

### Example from KICK track

| Clip | Time | StartRelative | First Note Time | Adjusted |
|------|------|---------------|-----------------|----------|
| CHORUS1 | 96-128 | 32 | 48 | 48-32=16 ✓ |
| DROP1 | 128-160 | 0 | 0 | 0-0=0 ✓ |

The CHORUS1 clip has notes at time 48, but with StartRelative=32, the note actually plays at clip-relative time 16 (i.e., at arrangement beat 96+16=112).

### XML Structure

```xml
<MidiClip Time="96">
  <CurrentStart Value="96"/>
  <CurrentEnd Value="128"/>
  <Loop>
    <LoopStart Value="0"/>
    <LoopEnd Value="64"/>
    <StartRelative Value="32"/>  <!-- KEY: Playback starts from beat 32 -->
  </Loop>
  <Notes>
    <KeyTracks>
      <KeyTrack>
        <Notes>
          <MidiNoteEvent Time="48" .../>  <!-- Stored time, NOT playback time -->
        </Notes>
      </KeyTrack>
    </KeyTracks>
  </Notes>
</MidiClip>
```

### Current (Buggy) Behavior

```python
# We extract note.time directly without adjustment
notes.append(MidiNote(time=float(note_event.get("Time", 0)), ...))
```

This causes notes at time 48 to be checked against a 32-beat clip window, finding no overlap.

## Solution

When extracting MIDI notes, adjust times by `StartRelative`:

```python
def extract_midi_notes(clip_element: Element) -> tuple[MidiNote, ...]:
    # Get StartRelative offset
    loop_elem = clip_element.find("Loop")
    start_relative = 0.0
    if loop_elem is not None:
        start_rel_elem = loop_elem.find("StartRelative")
        if start_rel_elem is not None:
            start_relative = float(start_rel_elem.get("Value", "0"))

    # When creating notes, adjust time:
    adjusted_time = float(note_event.get("Time", 0)) - start_relative

    # Only include notes that fall within the playable range
    # (adjusted_time >= 0 and adjusted_time < clip_length)
```

### After Fix

- CHORUS1 clip: note at 48, StartRelative=32 → adjusted time 16 → within 32-beat clip ✓
- DROP1 clip: note at 0, StartRelative=0 → adjusted time 0 → within 32-beat clip ✓

## Implementation

1. Modify `extract_midi_notes()` in `midi.py` to read `StartRelative`
2. Adjust note times before storing
3. Filter out notes with negative adjusted times (before clip start)

**Status: Fixed**

---

# Bug: MIDI Loop Repetition Not Handled

## Problem

When a clip has `LoopOn=true` and the clip length exceeds the loop region, notes should repeat. We currently only extract notes once.

### Example from example3.als DRUMS track

```
Clip at beat 68: 44 beats long, loop=8 beats → repeats 5.5x
```

Notes in beats 0-8 should appear at 0-8, 8-16, 16-24, 24-32, 32-40, 40-44.

### Files Affected

- `example.als`: 3 looping clips (STRUCTURE track)
- `example3.als`: 13 looping clips (Piano loops 2-3x, DRUMS loops 3-5.5x)

## Solution

When extracting notes, check if looping applies and expand notes:

```python
def extract_midi_notes(clip_element: Element) -> tuple[MidiNote, ...]:
    # Get loop settings
    loop_on = False
    loop_start = 0.0
    loop_end = 0.0
    start_relative = 0.0

    loop_elem = clip_element.find("Loop")
    if loop_elem is not None:
        loop_on = loop_elem.find("LoopOn").get("Value") == "true"
        loop_start = float(loop_elem.find("LoopStart").get("Value", "0"))
        loop_end = float(loop_elem.find("LoopEnd").get("Value", "0"))
        start_relative = float(loop_elem.find("StartRelative").get("Value", "0"))

    # Get clip length from CurrentEnd - Time
    clip_start = float(clip_element.get("Time", "0"))
    current_end_elem = clip_element.find("CurrentEnd")
    clip_end = float(current_end_elem.get("Value", "0")) if current_end_elem is not None else clip_start
    clip_length = clip_end - clip_start

    loop_length = loop_end - loop_start

    # Extract base notes (adjusted by start_relative)
    base_notes = [...]  # existing extraction

    # If looping and clip exceeds loop, expand notes
    if loop_on and loop_length > 0 and clip_length > loop_length:
        expanded_notes = []
        repetition = 0
        while repetition * loop_length < clip_length:
            offset = repetition * loop_length
            for note in base_notes:
                new_time = note.time + offset
                if new_time < clip_length:  # Only include if within clip
                    expanded_notes.append(MidiNote(
                        time=new_time,
                        duration=note.duration,
                        velocity=note.velocity,
                        pitch=note.pitch,
                    ))
            repetition += 1
        return tuple(sorted(expanded_notes, key=lambda n: n.time))

    return tuple(sorted(base_notes, key=lambda n: n.time))
```

## Notes Within Loop Region

Only notes within the loop region (LoopStart to LoopEnd) should be repeated. Notes outside this region play once at their original position.

---

# Original Design: Phrase-Aligned Event Detection

## Problem (Now Fixed)

Current event detection uses a **global grid** that doesn't align with section/phrase boundaries, causing misleading event attribution.

### Example

Given:
- VERSE1 ends at beat 60
- CHORUS1 starts at beat 62 (2-beat gap/transition)
- Drums stop at beat 58, resume at beat 62

Current behavior:
```
| Time | Cue     | Events       |
|------|---------|--------------|
| ...  | VERSE1  |              |
| 0:52 | CHORUS1 | Drums exits  |  ← WRONG: drums are playing in chorus!
```

Why this happens:
1. Detection windows are on a global 8-beat grid: 0-8, 8-16, ..., 56-64, 64-72
2. Window 56-64 spans end of VERSE1 and start of CHORUS1
3. Drums are inactive in the 58-62 gap
4. Window 56-64 shows "inactive" → generates "exit" event
5. Event at beat 56 falls into CHORUS1 phrase → attributed to chorus

### Root Cause

```
Timeline:     |----VERSE1----|--gap--|----CHORUS1----|
Beats:        48            60      62              78

Detection:    |---window----|---window----|---window----|
              48           56            64            72
                            ↑
                     This window spans the gap,
                     detects "inactive", generates "exit"
```

## Proposed Solution: Phrase-Aligned Detection

Instead of detecting on a global grid, detect activity **per phrase**.

### New Algorithm

```python
def detect_phrase_activity(
    phrases: list[Phrase],
    clip_contents: list[MidiClipContent],
) -> list[tuple[Phrase, bool]]:
    """
    For each phrase, determine if track has any note activity
    within that phrase's exact time boundaries.

    Returns:
        List of (phrase, is_active) tuples
    """
    result = []
    for phrase in phrases:
        is_active = False
        for content in clip_contents:
            # Check if clip overlaps with phrase
            if content.clip.start_beats < phrase.end_beats and \
               content.clip.end_beats > phrase.start_beats:
                # Convert to clip-relative coordinates
                relative_start = max(0, phrase.start_beats - content.clip.start_beats)
                relative_end = min(
                    content.clip.end_beats - content.clip.start_beats,
                    phrase.end_beats - content.clip.start_beats
                )
                if content.has_notes_in_range(relative_start, relative_end):
                    is_active = True
                    break
        result.append((phrase, is_active))
    return result


def detect_events_from_phrase_activity(
    track_name: str,
    phrase_activity: list[tuple[Phrase, bool]],
    category: str,
) -> list[TrackEvent]:
    """
    Compare adjacent phrases to detect state changes.

    Event is placed at the START of the phrase where the change occurs.
    """
    events = []
    was_active = False

    for phrase, is_active in phrase_activity:
        if is_active and not was_active:
            events.append(TrackEvent(
                beat=phrase.start_beats,
                track_name=track_name,
                event_type="enter",
                category=category,
            ))
        elif not is_active and was_active:
            events.append(TrackEvent(
                beat=phrase.start_beats,
                track_name=track_name,
                event_type="exit",
                category=category,
            ))
        was_active = is_active

    return events
```

### Result with Fix

```
Timeline:     |----VERSE1----|--gap--|----CHORUS1----|
Beats:        48            60      62              78

Phrases:      |--phrase--|--phrase--|--phrase--|--phrase--|
              48        56        60/62      70        78
                                   ↑
                            CHORUS1 phrase checks 62-70 only
                            Drums ARE active → no "exit" event
```

Expected output:
```
| Time | Cue     | Events       |
|------|---------|--------------|
| ...  | VERSE1  |              |
| 0:52 | CHORUS1 |              |  ← Correct: no false "drums exits"
```

## Changes Required

### 1. New function in `midi.py`

```python
def check_activity_in_range(
    clip_contents: list[MidiClipContent],
    start_beats: float,
    end_beats: float,
) -> bool:
    """Check if any MIDI notes are active in the given beat range."""
```

### 2. Modified pipeline in `analyze.py`

Current flow:
```
1. Parse ALS → tracks
2. Extract MIDI notes
3. Detect activity on global grid → events
4. Subdivide sections → phrases
5. Merge events into phrases
```

New flow:
```
1. Parse ALS → tracks
2. Extract MIDI notes
3. Subdivide sections → phrases
4. For each track:
   a. Check activity per phrase
   b. Generate events from phrase transitions
5. Merge events into phrases
```

### 3. Remove from `events.py`

- `detect_track_events()` - no longer needed (replaced by phrase-based detection)

### 4. Modify in `events.py`

- `detect_events_from_clip_contents()` - now takes phrases as parameter

## Edge Cases

### 1. Pickup Beats / Anticipation

A drum fill at the end of VERSE1 (beats 58-60) that leads into CHORUS1:

- With phrase-aligned detection: fill is in the VERSE1 phrase → "Drums enters" in VERSE1
- This is correct: the fill IS in VERSE1, even if it anticipates the chorus

### 2. Notes Spanning Phrase Boundaries

A sustained pad note from beat 58-66 spanning VERSE1 into CHORUS1:

- VERSE1 phrase (56-62): has_notes_in_range returns True
- CHORUS1 phrase (62-70): has_notes_in_range returns True
- Result: Pad is "active" in both → no enter/exit events
- Correct behavior

### 3. Very Short Phrases

If a section is shorter than the default phrase length:

- Single phrase covers the entire section
- Activity detection works normally

## Testing

Add tests for:
1. Event appears in correct phrase when section boundaries don't align to 8-beat grid
2. No false "exit" events when track plays through section boundaries
3. Notes spanning phrase boundaries don't generate spurious events
4. Pickup beats attributed to the phrase they occur in

## Alternative Considered: Snap Grid to Sections

Instead of phrase-aligned detection, we could snap the global grid to section starts.

Rejected because:
- Still has issues within sections (phrases within a section wouldn't align)
- More complex to implement
- Phrase-aligned is conceptually cleaner: "what's happening in THIS phrase?"
