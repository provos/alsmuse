"""Drum fill detection for ALSmuse.

This module detects drum fills by analyzing coordinated density spikes
across multiple drum components before section changes. A fill is identified
when two or more drum sub-categories (e.g., toms, cymbals, snare) spike
significantly above their baseline density in the bars leading up to
a section transition.
"""

from dataclasses import dataclass
from statistics import median

from .drum_mapping import DrumSubCategory, get_drum_subcategory
from .models import MidiClipContent, Section

# Detection parameters
DENSITY_SPIKE_THRESHOLD = 3.0  # 3x baseline = spike
MIN_COMPONENTS_FOR_FILL = 2  # need at least 2 drum components spiking
FILL_WINDOW_BEATS = 4.0  # analyze in 1-bar windows
SECTION_PROXIMITY_BEATS = 8.0  # look 2 bars before section changes


@dataclass(frozen=True)
class DrumFill:
    """A detected drum fill event.

    Represents a coordinated burst of activity across multiple drum
    components, typically occurring before a section change.

    Attributes:
        start_beats: Start position in beats.
        end_beats: End position in beats.
        intensity: Normalized intensity (0.0-1.0) based on spike magnitude.
        next_section: Name of the following section, or None.
        components: Tuple of drum sub-categories that spiked.
    """

    start_beats: float
    end_beats: float
    intensity: float  # normalized 0.0-1.0
    next_section: str | None  # e.g., "CHORUS"
    components: tuple[DrumSubCategory, ...]


def calculate_density_by_subcategory(
    clip_contents: list[MidiClipContent],
    start_beats: float,
    end_beats: float,
) -> dict[DrumSubCategory, float]:
    """Calculate note density per drum sub-category in a time range.

    Counts notes from each sub-category that fall within the specified
    range and normalizes by the duration to get notes per beat.

    Args:
        clip_contents: List of MidiClipContent to analyze.
        start_beats: Start of range in beats.
        end_beats: End of range in beats.

    Returns:
        Dictionary mapping each DrumSubCategory to its density (notes per beat).
    """
    counts: dict[DrumSubCategory, int] = {}
    duration = end_beats - start_beats

    if duration <= 0:
        return {}

    for content in clip_contents:
        clip_start = content.clip.start_beats
        clip_end = content.clip.end_beats

        # Skip clips that don't overlap with range
        if clip_end <= start_beats or clip_start >= end_beats:
            continue

        for note in content.notes:
            abs_time = clip_start + note.time
            if start_beats <= abs_time < end_beats:
                subcat = get_drum_subcategory(note.pitch)
                counts[subcat] = counts.get(subcat, 0) + 1

    return {k: v / duration for k, v in counts.items()}


def calculate_baseline_density(
    clip_contents: list[MidiClipContent],
    window_beats: float = 4.0,
) -> dict[DrumSubCategory, float]:
    """Calculate baseline density (median) for each sub-category.

    Samples density across the entire drum track in fixed windows
    and returns the median density for each sub-category. This
    establishes the "normal" playing density against which spikes
    are measured.

    Args:
        clip_contents: List of MidiClipContent to analyze.
        window_beats: Size of each sampling window in beats.

    Returns:
        Dictionary mapping each DrumSubCategory to its median density.
    """
    if not clip_contents:
        return {}

    # Find the range of the clips
    start = min(c.clip.start_beats for c in clip_contents)
    end = max(c.clip.end_beats for c in clip_contents)

    # Collect density samples for each sub-category
    samples: dict[DrumSubCategory, list[float]] = {}

    beat = start
    while beat < end:
        window_end = min(beat + window_beats, end)
        densities = calculate_density_by_subcategory(clip_contents, beat, window_end)

        for subcat, density in densities.items():
            if subcat not in samples:
                samples[subcat] = []
            samples[subcat].append(density)

        beat = window_end

    # Return median for each sub-category
    return {subcat: median(values) if values else 0.0 for subcat, values in samples.items()}


def detect_drum_fills(
    drum_clip_contents: list[MidiClipContent],
    sections: list[Section],
    threshold: float = DENSITY_SPIKE_THRESHOLD,
    min_components: int = MIN_COMPONENTS_FOR_FILL,
) -> list[DrumFill]:
    """Detect drum fills based on coordinated density spikes.

    Algorithm:
    1. Calculate baseline density for each drum sub-category
    2. Look at windows before each section transition
    3. Identify windows where multiple sub-categories spike (>= threshold)
    4. Return DrumFill events for qualifying windows

    Args:
        drum_clip_contents: List of MidiClipContent from drum tracks.
        sections: List of Section objects defining song structure.
        threshold: Multiplier above baseline to count as a spike.
        min_components: Minimum number of sub-categories that must spike.

    Returns:
        List of DrumFill events, sorted by start time.
    """
    if not drum_clip_contents or not sections:
        return []

    # Calculate baseline densities
    baseline = calculate_baseline_density(drum_clip_contents)

    if not baseline:
        return []

    # Get section start times (sorted)
    section_starts = sorted({s.start_beats for s in sections})

    # Map section starts to names
    section_names = {s.start_beats: s.name for s in sections}

    fills: list[DrumFill] = []

    for section_start in section_starts:
        # Look at the 2 bars before each section
        analysis_start = section_start - SECTION_PROXIMITY_BEATS
        analysis_end = section_start

        if analysis_start < 0:
            continue

        # Analyze in sub-windows
        window_start = analysis_start
        while window_start < analysis_end:
            window_end = min(window_start + FILL_WINDOW_BEATS, analysis_end)

            density = calculate_density_by_subcategory(drum_clip_contents, window_start, window_end)

            # Count spiking components
            spiking: list[DrumSubCategory] = []
            spike_ratios: list[float] = []

            for subcat, d in density.items():
                base = baseline.get(subcat, 0.0)
                if base > 0:
                    ratio = d / base
                    if ratio >= threshold:
                        spiking.append(subcat)
                        spike_ratios.append(ratio)

            # Qualify as fill if enough components spike
            if len(spiking) >= min_components:
                next_section = section_names.get(section_start)

                # Calculate intensity (normalized spike magnitude)
                avg_spike = sum(spike_ratios) / len(spike_ratios)
                intensity = min(1.0, avg_spike / (threshold * 2))

                fills.append(
                    DrumFill(
                        start_beats=window_start,
                        end_beats=window_end,
                        intensity=intensity,
                        next_section=next_section,
                        components=tuple(sorted(spiking, key=lambda x: x.value)),
                    )
                )

            window_start = window_end

    return fills
