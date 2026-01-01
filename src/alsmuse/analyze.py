"""Application layer for ALSmuse.

This module provides the main analysis pipeline that orchestrates
parsing, extraction, and formatting of Ableton Live Set files.
"""

from pathlib import Path

from .extractors import StructureTrackExtractor, fill_gaps
from .formatter import format_av_table
from .parser import parse_als_file


def analyze_als(
    als_path: Path,
    structure_track: str = "STRUCTURE",
) -> str:
    """Main analysis pipeline.

    1. Parse ALS file
    2. Extract sections using StructureTrackExtractor
    3. Fill gaps with transitions
    4. Format output as markdown A/V table

    Args:
        als_path: Path to the .als file
        structure_track: Name of the structure track (case-insensitive)

    Returns:
        Markdown formatted A/V table string

    Raises:
        ParseError: If the file cannot be parsed
        TrackNotFoundError: If the structure track is not found
    """
    live_set = parse_als_file(als_path)

    extractor = StructureTrackExtractor(structure_track)
    sections = extractor.extract(live_set)
    sections = fill_gaps(sections)

    return format_av_table(sections, live_set.tempo.bpm)
