"""Section extraction strategies for ALSmuse.

This module provides different strategies for extracting musical sections
from Ableton Live Set files. The Strategy pattern allows for multiple
approaches to section detection.
"""

from typing import Protocol

from .exceptions import TrackNotFoundError
from .models import Clip, LiveSet, Section, Track


class SectionExtractor(Protocol):
    """Protocol for section extraction strategies.

    Implementations of this protocol analyze a LiveSet and produce
    a list of musical sections. Different strategies can detect
    sections in different ways (e.g., from a dedicated structure track,
    by inferring from track names, or through audio analysis).
    """

    def extract(self, live_set: LiveSet) -> list[Section]:
        """Extract sections from a LiveSet.

        Args:
            live_set: The parsed Ableton Live Set to analyze.

        Returns:
            List of sections found in the LiveSet, sorted by start position.
        """
        ...


class StructureTrackExtractor:
    """Extract sections from a dedicated structure track.

    This extractor looks for a specific track by name and converts
    its clips into sections. This is useful when the producer has
    created a dedicated track with clips marking song sections.
    """

    def __init__(self, track_name: str = "STRUCTURE") -> None:
        """Initialize the extractor with a target track name.

        Args:
            track_name: Name of the track to extract sections from.
                       Comparison is case-insensitive.
        """
        self._track_name = track_name

    def extract(self, live_set: LiveSet) -> list[Section]:
        """Extract sections from the structure track.

        Finds the track matching the configured name (case-insensitive)
        and converts each clip into a Section.

        Args:
            live_set: The parsed Ableton Live Set to analyze.

        Returns:
            List of sections sorted by start position.

        Raises:
            TrackNotFoundError: If no track matches the configured name.
        """
        track = self._find_track(live_set)
        sections = [self._clip_to_section(clip) for clip in track.clips]
        return sorted(sections, key=lambda s: s.start_beats)

    def _find_track(self, live_set: LiveSet) -> Track:
        """Find the structure track by name (case-insensitive).

        Args:
            live_set: The LiveSet to search.

        Returns:
            The matching Track.

        Raises:
            TrackNotFoundError: If no track matches the configured name.
        """
        target_lower = self._track_name.lower()
        for track in live_set.tracks:
            if track.name.lower() == target_lower:
                return track
        raise TrackNotFoundError(
            f"Track '{self._track_name}' not found in LiveSet"
        )

    @staticmethod
    def _clip_to_section(clip: Clip) -> Section:
        """Convert a Clip to a Section.

        Args:
            clip: The clip to convert.

        Returns:
            A Section with the same name and position as the clip.
        """
        return Section(
            name=clip.name,
            start_beats=clip.start_beats,
            end_beats=clip.end_beats,
        )


def fill_gaps(sections: list[Section]) -> list[Section]:
    """Insert TRANSITION sections for gaps between clips.

    Takes a list of sections (assumed to be sorted by start_beats)
    and inserts TRANSITION sections wherever there is a gap between
    consecutive sections.

    A gap exists when one section's end_beats is less than the next
    section's start_beats.

    Args:
        sections: List of sections sorted by start position.

    Returns:
        New list with TRANSITION sections inserted in gaps.
        The original sections are preserved; only new transitions are added.
    """
    if not sections:
        return []

    result: list[Section] = []

    for i, section in enumerate(sections):
        # Check if there's a gap before this section
        if i > 0:
            previous = sections[i - 1]
            if previous.end_beats < section.start_beats:
                transition = Section(
                    name="TRANSITION",
                    start_beats=previous.end_beats,
                    end_beats=section.start_beats,
                )
                result.append(transition)

        result.append(section)

    return result
