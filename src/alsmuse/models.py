"""Domain models for ALSmuse.

Immutable data classes representing the core domain concepts for
analyzing Ableton Live Set files and extracting music structure.
"""

from dataclasses import dataclass
from typing import Literal


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
