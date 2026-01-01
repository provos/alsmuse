# ALSmuse Status & Roadmap

ALSmuse is a CLI tool that generates A/V scripts from Ableton Live projects.

## Current Capabilities (v2)

The current version implements the full "Phrase-Level" analysis pipeline.

### Core Features
-   **ALS Parsing**: Reads Ableton Live 11/12 project files.
-   **Structure Extraction**: Detects song sections (Verse, Chorus) from a "STRUCTURE" track.
-   **Phrase Subdivision**: Breaks sections into 2-bar phrases for video editing rhythm.
-   **Event Detection**: Automatically detects when instruments enter or exit.
    -   groups tracks by category (Drums, Bass, Vocals, etc.).
    -   Ignores silent MIDI clips (analyzes actual note data).
-   **Interactive Review**: CLI interface to manually categorize tracks and save preferences.

### AI & Lyrics
-   **Forced Alignment**: Aligns a plain text lyrics file to the project's vocal tracks with word-level precision.
-   **Transcription**: Automatically transcribes vocals using OpenAI's Whisper model if no lyrics are provided.
-   **Hardware Acceleration**: Uses `mlx-whisper` on Apple Silicon for rapid processing.
-   **Hallucination Filtering**: Intelligently removes Whisper hallucinations during silent passages.

### Output
-   Markdown-formatted table compatible with GitHub, Obsidian, and standard Markdown editors.
-   Columns: Time, Cue (Section), Events, Lyrics, Video (empty for user input).
-   Exports aligned lyrics to `.lrc` format.

## Usage

```bash
# Basic analysis
alsmuse analyze project.als

# With lyrics text file (forced alignment)
alsmuse analyze project.als --lyrics lyrics.txt --align-vocals

# Automatic transcription (no lyrics file)
alsmuse analyze project.als --transcribe

# Specify vocal tracks explicitly
alsmuse analyze project.als --transcribe --vocal-track "Vox Main" --vocal-track "Vox Double"
```

## Roadmap

### Immediate Improvements
-   **Audio Track Activity**: Currently, event detection relies heavily on MIDI. We need to implement RMS-based activity detection for audio tracks (stems) to support "Enter/Exit" events for non-MIDI audio.
-   **Tempo Map Support**: The current implementation assumes a static tempo. Support for variable tempo maps is needed for projects with tempo changes.

### Future Ideas
-   **Visualizer Generation**: Generate a simple video file with the lyrics and cues burned in for easier review.
-   **Premiere/DaVinci Integration**: Export markers or an XML sequence directly to video editing software.
-   **LLM Description**: Use an LLM to generate a textual description of the song structure based on the analysis.