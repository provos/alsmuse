# ALSmuse Brainstorming: Enriching A/V Scripts

## Goal
Generate A/V scripts with enough detail to plan 3-5 second video sequences that sync with the music's emotional and rhythmic content.

## What We Can Extract from ALS Files

### 1. Already Implemented
- Section structure (INTRO, VERSE, CHORUS, etc.)
- Tempo (BPM) and time signature
- Section start/end times

### 2. Track Activity Analysis
Your ALS file has rich track information:

| Track Type | Examples | What It Tells Us |
|------------|----------|------------------|
| Vocals | "Verse Main", "Chorus Main", "Chorus Double" | When vocals are present, harmonies |
| Drums | KICK, SNARE, DRUMS | Rhythmic anchor, energy level |
| Bass | "Bass" | Low-end energy, groove |
| Leads | "Lead Guitar", "Lead" | Melodic focus, solos |
| Pads | "Pad" | Atmosphere, sustained textures |
| FX | "Reverse Leadin", "downlifter" | Transitions, builds, drops |

**Idea**: Track when each instrument group enters/exits to describe energy changes:
- "VERSE1: Drums enter, vocals begin"
- "CHORUS: Full band, doubled vocals"
- "BRIDGE: Stripped to pad + vocals"

### 3. Drum Pattern Changes (Energy Markers)
Your drum tracks have 39 clips (~1.7s each) - these represent pattern changes that could mark:
- Fills before sections
- Energy shifts within sections
- Breakdown/buildup moments

### 4. Audio File References
The ALS contains paths to audio files:
```
Samples/Processed/Bounce/Bounce Verse Main [2025-12-16 112513]-2.wav
```
**Future**: Use librosa to analyze transients, energy curves, vocal timing.

## Lyrics Integration

### Option A: Manual Alignment
```markdown
| Time | Audio | Video |
|------|-------|-------|
| 0:27 | VERSE1: "First line of lyrics" | |
| 0:31 | "Second line of lyrics" | |
```

### Option B: Timestamped Lyrics (LRC format)
```
[00:27.50] First line of lyrics
[00:31.20] Second line of lyrics
```

### Option C: Section-Based Lyrics
User provides lyrics grouped by section, we distribute based on section duration.

## What Music Video Professionals Consider

### Shot Pacing (from research)
- **Quick cuts** (1-3s): High energy, chorus, dance sequences
- **Medium shots** (3-5s): Verses, storytelling, performance
- **Long takes** (5-10s+): Emotional moments, solos, atmosphere

### Beat-Synced Editing
- Cuts on kick drum hits for power
- Cuts on snare for rhythm
- Cuts on musical accents (crashes, bass drops)

### Visual-Audio Relationships
| Audio Element | Typical Visual |
|---------------|----------------|
| Kick drum | Camera push, flash, impact |
| Snare | Cut, whip pan |
| Vocal phrase start | New shot, performer focus |
| Buildup | Faster cuts, movement |
| Drop | Wide shot, explosion of movement |
| Breakdown | Slow motion, intimate shots |

## Proposed Enhanced Output

### Level 1: Section-Based (Current)
```
| Time | Audio | Video |
|------|-------|-------|
| 0:27 | VERSE1 | |
| 0:53 | CHORUS | |
```

### Level 2: With Instrumentation
```
| Time | Audio | Energy | Video |
|------|-------|--------|-------|
| 0:27 | VERSE1 | Medium (drums + bass + vox) | |
| 0:53 | CHORUS | High (full band + doubles) | |
```

### Level 3: With Sub-Beats (for 3-5s sequences)
```
| Time | Audio | Cue | Video |
|------|-------|-----|-------|
| 0:27 | VERSE1 start | Drums enter | |
| 0:30 | | Vocal phrase 1 | |
| 0:34 | | Vocal phrase 2 | |
| 0:38 | | Fill → | |
| 0:40 | VERSE1 cont. | Vocal phrase 3 | |
```

### Level 4: With Lyrics
```
| Time | Audio | Lyrics | Video |
|------|-------|--------|-------|
| 0:27 | VERSE1 | "Walking down the street" | |
| 0:31 | | "Feeling the beat" | |
| 0:35 | | "Heart starts to race" | |
```

## Questions to Explore

1. **Lyrics file format**: What format do your lyrics files use? Plain text with section headers?

2. **Priority**: Which enhancement would be most valuable first?
   - [x] Track activity (which instruments are playing)
   - [x] Lyrics integration
   - [ ] Drum hit detection for cut points
   - [ ] Energy curve analysis

3. **Output granularity**: How fine-grained should the timeline be?
   - Every 4 bars (current section markers)
   - Every bar
   - Every beat
   - Every drum hit

We should try to time them to be around 2 to 5 seconds. using a mixture of techniques.   

4. **Visual suggestions**: Should the tool suggest shot types based on audio characteristics, or just provide timing cues?

It should only provide timing cues.

## Next Steps (Possible)

1. **Track Activity Analyzer**: Detect which tracks have clips at each time point
2. **Lyrics Parser**: Import and align lyrics with sections
3. **Drum Pattern Detector**: Find fills, accents, and energy changes
4. **Energy Curve**: Calculate "intensity" based on active track count
5. **Audio Analysis**: Use librosa for transient detection (requires actual audio files)
