"""Domain models for ALSmuse.

Immutable data classes representing the core domain concepts for
analyzing Ableton Live Set files and extracting music structure.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal


class LyricsFormat(Enum):
    """Format of lyrics file.

    Used to determine how to parse lyrics files based on their structure.

    Values:
        PLAIN: No timestamps, plain text with optional section headers.
        LRC: Standard LRC format with [mm:ss.xx] line timestamps.
        LRC_ENHANCED: LRC with word-level <mm:ss.xx> timestamps.
        SIMPLE_TIMED: Simple format with m:ss.xx at line start.
    """

    PLAIN = "plain"
    LRC = "lrc"
    LRC_ENHANCED = "lrc_enhanced"
    SIMPLE_TIMED = "simple_timed"


@dataclass(frozen=True)
class Tempo:
    """Static tempo for the project.

    MVP Limitation: This represents a single global tempo. Files with
    tempo automation (BPM changes mid-song) will have inaccurate
    timestamps after the first tempo change. Future versions can
    extend this to support tempo events.
    """

    bpm: float
    time_signature: tuple[int, int]  # (numerator, denominator)


@dataclass(frozen=True)
class Clip:
    """A clip within a track.

    Clips represent discrete audio or MIDI regions on the timeline.
    Positions are measured in beats from the project start.
    """

    name: str
    start_beats: float
    end_beats: float


@dataclass(frozen=True)
class Track:
    """A track containing clips.

    Tracks organize clips into lanes within the arrangement view.
    Each track has a type indicating whether it contains MIDI or audio data.
    """

    name: str
    track_type: Literal["midi", "audio"]
    clips: tuple[Clip, ...]


@dataclass(frozen=True)
class LiveSet:
    """An Ableton Live Set project.

    The top-level container representing a parsed .als file,
    containing tempo information and all tracks.
    """

    tempo: Tempo
    tracks: tuple[Track, ...]


@dataclass(frozen=True)
class Section:
    """A musical section within the arrangement.

    Sections represent named regions of the song (e.g., "INTRO", "VERSE 1").
    Used for generating the Audio column of an A/V script.
    """

    name: str
    start_beats: float
    end_beats: float

    def start_time(self, bpm: float) -> float:
        """Convert start position to seconds.

        Args:
            bpm: Beats per minute for time conversion.

        Returns:
            Start position in seconds.
        """
        return self.start_beats * 60 / bpm

    def end_time(self, bpm: float) -> float:
        """Convert end position to seconds.

        Args:
            bpm: Beats per minute for time conversion.

        Returns:
            End position in seconds.
        """
        return self.end_beats * 60 / bpm


@dataclass(frozen=True)
class MidiNote:
    """A single MIDI note event.

    Represents a MIDI note with timing, duration, velocity, and pitch.
    Times are relative to the containing clip start.

    Attributes:
        time: Beat position relative to clip start.
        duration: Note duration in beats.
        velocity: MIDI velocity (0-127).
        pitch: MIDI note number (0-127).
    """

    time: float
    duration: float
    velocity: int
    pitch: int


@dataclass(frozen=True)
class MidiClipContent:
    """Content of a MIDI clip including notes.

    Associates a clip with its MIDI note data for activity detection.

    Attributes:
        clip: The parent Clip containing timing information.
        notes: Tuple of MidiNote objects within this clip.
    """

    clip: Clip
    notes: tuple[MidiNote, ...]

    def has_notes_in_range(self, start: float, end: float) -> bool:
        """Check if any notes are active within the given range.

        A note is considered active in the range if it overlaps with the
        range in any way (starts within, ends within, or spans the range).

        This uses range queries rather than point sampling to avoid the
        "stroboscope" problem where off-beat notes would be missed.

        Args:
            start: Start of range in beats (relative to clip).
            end: End of range in beats (relative to clip).

        Returns:
            True if any note is active within the range.
        """
        for note in self.notes:
            note_start = note.time
            note_end = note.time + note.duration
            # Check for any overlap between note and range
            if note_start < end and note_end > start:
                return True
        return False

    def note_density(self) -> float:
        """Calculate notes per beat indicating activity level.

        Returns:
            Notes per beat, or 0.0 if clip has no duration.
        """
        if not self.notes:
            return 0.0
        duration = self.clip.end_beats - self.clip.start_beats
        return len(self.notes) / duration if duration > 0 else 0.0


@dataclass(frozen=True)
class TrackEvent:
    """A significant change in track activity.

    Represents an enter or exit event for a track, indicating when
    an instrument or sound starts or stops playing in the arrangement.

    Attributes:
        beat: The beat position where the event occurs.
        track_name: Name of the track that generated this event.
        event_type: Either "enter" (starts playing) or "exit" (stops playing).
        category: The instrument category (e.g., "drums", "bass", "vocals").
    """

    beat: float
    track_name: str
    event_type: Literal["enter", "exit"]
    category: str


@dataclass(frozen=True)
class Phrase:
    """A time slice with associated events and metadata.

    Phrases represent fixed-duration chunks of the arrangement, typically
    2 bars (8 beats in 4/4 time). They are used for generating detailed
    A/V scripts with per-phrase event information.

    Attributes:
        start_beats: Start position in beats.
        end_beats: End position in beats.
        section_name: Name of the containing section, or "..." for continuation.
        is_section_start: True if this phrase starts a new section.
        events: Tuple of track events occurring within this phrase.
        lyric: Optional lyric text for this phrase.
    """

    start_beats: float
    end_beats: float
    section_name: str
    is_section_start: bool
    events: tuple[TrackEvent, ...] = ()
    lyric: str = ""

    def start_time(self, bpm: float) -> float:
        """Convert start position to seconds.

        Args:
            bpm: Beats per minute for time conversion.

        Returns:
            Start position in seconds.
        """
        return self.start_beats * 60 / bpm

    def duration_seconds(self, bpm: float) -> float:
        """Calculate phrase duration in seconds.

        Args:
            bpm: Beats per minute for time conversion.

        Returns:
            Duration in seconds.
        """
        return (self.end_beats - self.start_beats) * 60 / bpm


@dataclass(frozen=True)
class AudioClipRef:
    """Reference to an audio file with timeline position.

    Used for extracting audio clips from ALS files for vocal alignment.

    Attributes:
        track_name: Name of the containing track.
        file_path: Resolved path to the audio file.
        start_beats: Start position on timeline in beats.
        end_beats: End position on timeline in beats.
        start_seconds: Start position in seconds (computed from BPM).
        end_seconds: End position in seconds (computed from BPM).
        sample_start_beats: Start offset within the audio file (in beats).
            This is the Loop/LoopStart value from Ableton, indicating where
            in the sample to start playing. None means start from beginning.
        sample_end_beats: End offset within the audio file (in beats).
            This is the Loop/LoopEnd value from Ableton. None means play to end.
    """

    track_name: str
    file_path: Path
    start_beats: float
    end_beats: float
    start_seconds: float
    end_seconds: float
    sample_start_beats: float | None = None
    sample_end_beats: float | None = None


@dataclass(frozen=True)
class TimedWord:
    """A word with start and end timestamps in seconds.

    Used for forced alignment of lyrics to audio.

    Attributes:
        text: The word text.
        start: Start timestamp in seconds.
        end: End timestamp in seconds.
    """

    text: str
    start: float  # seconds
    end: float  # seconds


@dataclass(frozen=True)
class TimedLine:
    """A line of lyrics with timing derived from word timestamps.

    Groups words back into the original line structure while
    preserving word-level timing information.

    Attributes:
        text: The full line text.
        start: Start timestamp in seconds (from first word).
        end: End timestamp in seconds (from last word).
        words: Tuple of TimedWord objects in this line.
    """

    text: str
    start: float  # seconds (from first word)
    end: float  # seconds (from last word)
    words: tuple[TimedWord, ...]


@dataclass(frozen=True)
class TimedSegment:
    """A transcribed segment with word-level timing.

    Represents a natural phrase/sentence boundary as detected by Whisper.
    These segments are more reliable than heuristic-based line breaking
    because they leverage Whisper's natural language understanding.

    Attributes:
        text: The full segment text.
        start: Start timestamp in seconds.
        end: End timestamp in seconds.
        words: Tuple of TimedWord objects in this segment.
    """

    text: str
    start: float  # seconds
    end: float  # seconds
    words: tuple[TimedWord, ...]
