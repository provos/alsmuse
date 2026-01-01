# ALSmuse MVP

## Goal

Generate the Audio column of an A/V script (music video treatment) from an Ableton Live Set file.

## A/V Script Format

A/V (Audio/Visual) scripts use a two-column layout:

| Audio | Video |
|-------|-------|
| Lyrics, sound effects, musical cues | Visual descriptions synced to audio |

The left column contains:
- Lyrics (for music videos)
- Section markers (INTRO, VERSE, CHORUS, etc.)
- Sound effects and musical hits (SFX)
- Timing information

The right column (filled in by the user) describes what happens visually at each moment.

## MVP Scope

### Input
- Ableton Live Set (.als) file - gzipped XML

### Processing
1. Decompress and parse ALS XML
2. Extract global timing info (BPM, time signature)
3. Find the "structure" track (MIDI track with section clips)
4. Extract clip names and positions (in bars/beats)
5. Convert bar positions to timestamps (MM:SS.ms)

### Output
Markdown table with timestamps and section names:

```markdown
| Time | Audio | Video |
|------|-------|-------|
| 0:00 | INTRO | |
| 0:32 | VERSE 1 | |
| 1:04 | CHORUS | |
| 1:36 | VERSE 2 | |
```

## Out of Scope for MVP
- Drum hit detection (MIDI or audio transients)
- Lyrics integration and timestamping
- Audio file analysis (librosa)
- "Hits" detection (crashes, bass drops)

## Future: Section Inference from Track Names

Track names can hint at when sections occur:
- "Chorus Main", "Chorus Double" → active during chorus
- "Verse Main", "Verse Double" → active during verse

By analyzing when clips on these tracks are active, we could infer section boundaries even without a dedicated structure track. This makes the tool more generic for users who don't have a structure track.

## ALS File Structure (from analysis)

### Format
- Gzip-compressed XML
- Root element: `<Ableton>` with version info
- Main content in `<LiveSet><Tracks>`

### Tempo
Located at `LiveSet/MasterTrack/DeviceChain/Mixer/Tempo/Manual[@Value]`
- Example: `<Manual Value="144" />` = 144 BPM

### Tracks
- `<MidiTrack>` elements with `<Name><UserName Value="STRUCTURE"/>`
- Track name in: `Name/UserName[@Value]` or `Name/EffectiveName[@Value]`

### MIDI Clips (sections)
Located in `MidiTrack/DeviceChain/MainSequencer/ClipTimeable/ArrangerAutomation/Events/MidiClip`

Each clip has:
- `Time` attribute: start position in beats
- `CurrentStart`: same as Time
- `CurrentEnd`: end position in beats
- `Name[@Value]`: clip name (e.g., "INTRO", "VERSE1", "CHORUS")

### Audio File References
Located in `AudioTrack/DeviceChain/MainSequencer/.../SampleRef/FileRef`

```xml
<SampleRef>
  <FileRef>
    <RelativePath Value="../../../../Downloads/.../KICK SAMPLE.cm.wav" />
    <Path Value="/Users/provos/Downloads/.../KICK SAMPLE.cm.wav" />
  </FileRef>
</SampleRef>
```

This allows future audio analysis with librosa by resolving these paths.

### Audio Clips
Similar to MIDI clips but in `AudioTrack`:
- `Time` attribute: start position in beats
- `CurrentStart` / `CurrentEnd`: clip boundaries
- `Name[@Value]`: clip name

### Time Conversion
```
time_seconds = beat_position * 60 / bpm
```

Example at 144 BPM:
- Beat 28 → 11.67 seconds (0:11)
- Beat 64 → 26.67 seconds (0:26)

### Example Output (from example.als at 144 BPM)

| Time | Audio | Video |
|------|-------|-------|
| 0:11 | INTRO | |
| 0:26 | VERSE1 | |
| 0:53 | CHORUS | |
| 1:21 | VERSE2 | |
| 1:45 | CHORUS | |
| 2:00 | BRIDGE | |
| 2:25 | SOLO | |
| 2:38 | CHORUS | |
| 2:51 | OUTRO | |

## Design Decisions
1. ~~What is the exact XML structure of an ALS file?~~ ✓ Answered
2. ~~How are MIDI clips positioned in the XML?~~ ✓ Beats from song start
3. ~~How is BPM stored?~~ ✓ In MasterTrack Tempo element
4. ~~Is the structure track always named "STRUCTURE"?~~ → Configurable via CLI (`--structure-track`)
5. ~~How to handle gaps between clips?~~ → Label as "TRANSITION"
6. Tempo automation: Support for generality (iterate through tempo events)

## Generalization for Other Users
- Structure track name configurable (not everyone uses "STRUCTURE")
- Fallback: infer sections from track names containing "verse", "chorus", "bridge", etc.
- Fallback: analyze clip activity patterns across tracks

## Next Steps
1. ~~Obtain a sample ALS file~~ ✓
2. ~~Decompress and examine the XML structure~~ ✓
3. ~~Identify relevant elements for section extraction~~ ✓
4. Implement parser
