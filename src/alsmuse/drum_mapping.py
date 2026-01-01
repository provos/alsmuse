"""General MIDI drum pitch mapping for sub-categorization.

This module provides mappings from GM drum pitches to logical sub-categories
for use in drum fill detection. The categories help identify when multiple
drum components are playing in coordination (e.g., toms + cymbals = fill).

Note: Exact categorization accuracy is not critical. The fill detection
algorithm looks for coordinated density spikes across multiple sub-categories.
What matters is that different drum sounds map to different categories, not
that each mapping is perfectly correct. A tom mapped as percussion still
contributes to detecting a fill when it spikes alongside other components.
"""

from enum import Enum


class DrumSubCategory(Enum):
    """Sub-categories for drum tracks.

    These categories group related drum sounds for fill detection.
    When multiple categories spike in density simultaneously, it
    indicates a drum fill rather than regular playing.
    """

    KICK = "kick"
    SNARE = "snare"
    HATS = "hats"
    TOMS = "toms"
    CYMBALS = "cymbals"
    PERCUSSION = "percussion"
    UNKNOWN = "unknown"


# Map GM drum pitches to sub-categories
# Based on General MIDI Level 1 Percussion Key Map
GM_DRUM_MAPPING: dict[int, DrumSubCategory] = {
    # Kick drums (35-36)
    35: DrumSubCategory.KICK,  # Acoustic Bass Drum
    36: DrumSubCategory.KICK,  # Bass Drum 1
    # Snare drums (37-40)
    37: DrumSubCategory.SNARE,  # Side Stick
    38: DrumSubCategory.SNARE,  # Acoustic Snare
    40: DrumSubCategory.SNARE,  # Electric Snare
    # Hi-Hat (42, 44, 46)
    42: DrumSubCategory.HATS,  # Closed Hi-Hat
    44: DrumSubCategory.HATS,  # Pedal Hi-Hat
    46: DrumSubCategory.HATS,  # Open Hi-Hat
    # Toms (41, 43, 45, 47, 48, 50)
    41: DrumSubCategory.TOMS,  # Low Floor Tom
    43: DrumSubCategory.TOMS,  # High Floor Tom
    45: DrumSubCategory.TOMS,  # Low Tom
    47: DrumSubCategory.TOMS,  # Low-Mid Tom
    48: DrumSubCategory.TOMS,  # Hi-Mid Tom
    50: DrumSubCategory.TOMS,  # High Tom
    # Cymbals (49, 51-53, 55, 57, 59)
    49: DrumSubCategory.CYMBALS,  # Crash Cymbal 1
    51: DrumSubCategory.CYMBALS,  # Ride Cymbal 1
    52: DrumSubCategory.CYMBALS,  # Chinese Cymbal
    53: DrumSubCategory.CYMBALS,  # Ride Bell
    55: DrumSubCategory.CYMBALS,  # Splash Cymbal
    57: DrumSubCategory.CYMBALS,  # Crash Cymbal 2
    59: DrumSubCategory.CYMBALS,  # Ride Cymbal 2
    # Percussion (39, 54, 56, 58, 60-81)
    39: DrumSubCategory.PERCUSSION,  # Hand Clap
    54: DrumSubCategory.PERCUSSION,  # Tambourine
    56: DrumSubCategory.PERCUSSION,  # Cowbell
    58: DrumSubCategory.PERCUSSION,  # Vibraslap
    60: DrumSubCategory.PERCUSSION,  # Hi Bongo
    61: DrumSubCategory.PERCUSSION,  # Low Bongo
    62: DrumSubCategory.PERCUSSION,  # Mute Hi Conga
    63: DrumSubCategory.PERCUSSION,  # Open Hi Conga
    64: DrumSubCategory.PERCUSSION,  # Low Conga
    65: DrumSubCategory.PERCUSSION,  # High Timbale
    66: DrumSubCategory.PERCUSSION,  # Low Timbale
    67: DrumSubCategory.PERCUSSION,  # High Agogo
    68: DrumSubCategory.PERCUSSION,  # Low Agogo
    69: DrumSubCategory.PERCUSSION,  # Cabasa
    70: DrumSubCategory.PERCUSSION,  # Maracas
    71: DrumSubCategory.PERCUSSION,  # Short Whistle
    72: DrumSubCategory.PERCUSSION,  # Long Whistle
    73: DrumSubCategory.PERCUSSION,  # Short Guiro
    74: DrumSubCategory.PERCUSSION,  # Long Guiro
    75: DrumSubCategory.PERCUSSION,  # Claves
    76: DrumSubCategory.PERCUSSION,  # Hi Wood Block
    77: DrumSubCategory.PERCUSSION,  # Low Wood Block
    78: DrumSubCategory.PERCUSSION,  # Mute Cuica
    79: DrumSubCategory.PERCUSSION,  # Open Cuica
    80: DrumSubCategory.PERCUSSION,  # Mute Triangle
    81: DrumSubCategory.PERCUSSION,  # Open Triangle
}


def get_drum_subcategory(pitch: int) -> DrumSubCategory:
    """Map a MIDI pitch to its drum sub-category.

    Args:
        pitch: MIDI note number (0-127).

    Returns:
        The DrumSubCategory for the pitch, or UNKNOWN if not mapped.

    Examples:
        >>> get_drum_subcategory(36)
        <DrumSubCategory.KICK: 'kick'>
        >>> get_drum_subcategory(38)
        <DrumSubCategory.SNARE: 'snare'>
        >>> get_drum_subcategory(99)
        <DrumSubCategory.UNKNOWN: 'unknown'>
    """
    return GM_DRUM_MAPPING.get(pitch, DrumSubCategory.UNKNOWN)
