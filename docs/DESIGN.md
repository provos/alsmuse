# ALSmuse Design Document

## Overview

ALSmuse generates A/V scripts (music video treatments) from Ableton Live Set (.als) files. It analyzes the project structure, timing, and content to produce a time-coded table of events, lyrics, and musical cues, aiding video editors and directors.

## Design Principles

1.  **Analyze Content, Not Just Containers**: We examine MIDI notes and audio waveforms, not just clip boundaries.
2.  **Delta Events**: We report *changes* (e.g., "Bass enters"), not static states.
3.  **Phrase-Based Output**: The output is structured in musical phrases (typically 2-4 bars), matching the rhythm of video editing.
4.  **Single Responsibility**: Each module handles a specific domain (parsing, audio, lyrics, formatting).
5.  **Progressive Enhancement**: Basic analysis works with just MIDI; advanced features (transcription) engage only when requested.

## Architecture

The system follows a pipeline architecture, transforming raw ALS XML into domain models, enhancing them with analysis, and formatting the result.

```mermaid
graph TD
    CLI[CLI (cli.py)] --> App[Application Layer (analyze.py)]
    
    subgraph Parsing
        App --> Parser[ALS Parser (parser.py)]
        Parser --> Models[Domain Models]
    end
    
    subgraph Structure
        App --> Extract[Section Extractor (extractors.py)]
        App --> Phrases[Phrase Subdivider (phrases.py)]
    end
    
    subgraph Content Analysis
        App --> Midi[MIDI Analysis (midi.py)]
        App --> Audio[Audio Extraction (audio.py)]
        App --> Events[Event Detection (events.py)]
    end
    
    subgraph Lyrics & AI
        App --> Lyrics[Lyrics Parsing (lyrics.py)]
        App --> Align[Alignment/ASR (lyrics_align.py)]
        Align --> Whisper[Whisper Model (stable-ts/mlx)]
    end
    
    subgraph Interaction
        App --> Review[Category Review (category_review.py)]
        Review --> Config[Config Persistence (config.py)]
    end
    
    subgraph Output
        App --> Format[Formatter (formatter.py)]
    end
```

## Domain Models (`models.py`)

Immutable data classes represent the project state.

-   **LiveSet**: Root object containing tempo and tracks.
-   **Track/Clip**: Basic structural elements.
-   **Section**: High-level structural regions (Verse, Chorus).
-   **Phrase**: Time slices (e.g., 2 bars) subdividing sections.
-   **TrackEvent**: "Enter" or "Exit" events for instruments.
-   **TimedLine/TimedWord**: Lyric segments with precise timestamps.
-   **AudioClipRef**: Reference to an audio file on disk, with time and crop data.

## Module Responsibilities

### 1. Parsing (`parser.py`)
Responsible for reading the gzipped XML of an `.als` file. It handles:
-   Extraction of global tempo.
-   Parsing of MIDI and Audio tracks.
-   Resolution of track names (handling `UserName` vs `EffectiveName`).
-   XML traversal to find clips in the complex DeviceChain hierarchy.

### 2. Structure Extraction (`extractors.py` & `phrases.py`)
-   **Extractors**: Strategy pattern to find song sections. The primary strategy uses a dedicated "STRUCTURE" MIDI track.
-   **Phrases**: Subdivides sections into musical phrases (default 8 beats/2 bars). This creates the grid for the output table.

### 3. Audio & MIDI Analysis (`audio.py` & `midi.py`)
-   **Midi**: Extracts note data from MIDI clips. Used to determine if a track is actually playing notes, not just if a clip exists.
-   **Audio**:
    -   Resolves relative file paths for audio samples (crucial for moved projects).
    -   Combines multiple vocal clips into a single contiguous audio stream for analysis.
    -   Splits audio on silence to aid transcription.

### 4. Event Detection (`events.py`)
Identifies significant changes in instrumentation.
-   **Activity Detection**: Uses MIDI note data (or audio waveforms) to determine if a track is active in a given phrase.
-   **Event Generation**: Compares activity between phrases to generate "Enter" and "Exit" events.
-   **Categorization**: Groups tracks (e.g., "Kick", "Snare" -> "Drums") for cleaner output.

### 5. Lyrics & AI (`lyrics.py` & `lyrics_align.py`)
-   **Lyrics**: Parses plain text lyrics files.
-   **Alignment/Transcription**: Uses OpenAI's Whisper model (via `stable-ts` or `mlx-whisper`) to:
    -   **Force Align**: Match a text file to the vocal audio to get word-level timing.
    -   **Transcribe**: Generate lyrics from scratch if no text is provided.
    -   **Filter Hallucinations**: Removes words detected during silent sections.

### 6. Interactive Review (`category_review.py`)
Allows the user to correct track categorizations via a TTY interface.
-   Prompts user for unknown or ambiguous tracks.
-   Saves overrides to a `.muse` config file alongside the ALS project.

## Data Flow

1.  **Parse**: `.als` -> `LiveSet`
2.  **Structure**: `LiveSet` -> `list[Section]` -> `list[Phrase]`
3.  **Analysis**:
    -   `LiveSet` -> `list[MidiClipContent]` -> `list[TrackEvent]`
    -   `Phrase` + `TrackEvent` -> `Phrase` (with events)
4.  **Lyrics** (Optional):
    -   `LiveSet` -> `Audio` -> `Whisper` -> `list[TimedLine]`
    -   `Phrase` + `list[TimedLine]` -> `Phrase` (with lyrics)
5.  **Format**: `list[Phrase]` -> Markdown Table

## Configuration (`config.py`)
Project-specific settings (like vocal track selection or category overrides) are stored in a `.muse` JSON file in the same directory as the ALS file.

## Dependencies

-   **click**: CLI interface.
-   **stable-ts**: Whisper wrapper for accurate timestamping.
-   **mlx-whisper**: Apple Silicon optimized inference (optional).
-   **soundfile/numpy**: Audio processing.
-   **questionary**: Interactive terminal prompts.
-   **rich**: Formatted terminal output.