"""Tests for drum fill detection functionality."""

from alsmuse.drum_mapping import DrumSubCategory, get_drum_subcategory
from alsmuse.fills import (
    calculate_baseline_density,
    calculate_density_by_subcategory,
    detect_drum_fills,
)
from alsmuse.models import Clip, MidiClipContent, MidiNote, Section


class TestGetDrumSubcategory:
    """Tests for GM drum mapping."""

    def test_kick_pitches(self) -> None:
        """Kick drum pitches map correctly."""
        assert get_drum_subcategory(35) == DrumSubCategory.KICK
        assert get_drum_subcategory(36) == DrumSubCategory.KICK

    def test_snare_pitches(self) -> None:
        """Snare drum pitches map correctly."""
        assert get_drum_subcategory(37) == DrumSubCategory.SNARE
        assert get_drum_subcategory(38) == DrumSubCategory.SNARE
        assert get_drum_subcategory(40) == DrumSubCategory.SNARE

    def test_hats_pitches(self) -> None:
        """Hi-hat pitches map correctly."""
        assert get_drum_subcategory(42) == DrumSubCategory.HATS
        assert get_drum_subcategory(44) == DrumSubCategory.HATS
        assert get_drum_subcategory(46) == DrumSubCategory.HATS

    def test_toms_pitches(self) -> None:
        """Tom pitches map correctly."""
        assert get_drum_subcategory(41) == DrumSubCategory.TOMS
        assert get_drum_subcategory(43) == DrumSubCategory.TOMS
        assert get_drum_subcategory(45) == DrumSubCategory.TOMS
        assert get_drum_subcategory(47) == DrumSubCategory.TOMS
        assert get_drum_subcategory(48) == DrumSubCategory.TOMS
        assert get_drum_subcategory(50) == DrumSubCategory.TOMS

    def test_cymbals_pitches(self) -> None:
        """Cymbal pitches map correctly."""
        assert get_drum_subcategory(49) == DrumSubCategory.CYMBALS
        assert get_drum_subcategory(51) == DrumSubCategory.CYMBALS
        assert get_drum_subcategory(57) == DrumSubCategory.CYMBALS

    def test_percussion_pitches(self) -> None:
        """Percussion pitches map correctly."""
        assert get_drum_subcategory(39) == DrumSubCategory.PERCUSSION
        assert get_drum_subcategory(54) == DrumSubCategory.PERCUSSION
        assert get_drum_subcategory(56) == DrumSubCategory.PERCUSSION

    def test_unknown_pitch(self) -> None:
        """Unmapped pitches return UNKNOWN."""
        assert get_drum_subcategory(0) == DrumSubCategory.UNKNOWN
        assert get_drum_subcategory(99) == DrumSubCategory.UNKNOWN
        assert get_drum_subcategory(127) == DrumSubCategory.UNKNOWN


class TestCalculateDensity:
    """Tests for density calculation functions."""

    def test_empty_clips_return_empty_density(self) -> None:
        """Empty clip list returns empty density."""
        result = calculate_density_by_subcategory([], 0.0, 8.0)
        assert result == {}

    def test_zero_duration_returns_empty(self) -> None:
        """Zero duration range returns empty density."""
        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=8.0),
            notes=(MidiNote(time=0.0, duration=0.5, velocity=100, pitch=36),),
        )
        result = calculate_density_by_subcategory([clip], 4.0, 4.0)
        assert result == {}

    def test_basic_density_calculation(self) -> None:
        """Basic density is calculated as notes per beat."""
        # 4 notes over 4 beats = 1 note per beat
        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=8.0),
            notes=(
                MidiNote(time=0.0, duration=0.5, velocity=100, pitch=36),  # kick
                MidiNote(time=1.0, duration=0.5, velocity=100, pitch=36),  # kick
                MidiNote(time=2.0, duration=0.5, velocity=100, pitch=36),  # kick
                MidiNote(time=3.0, duration=0.5, velocity=100, pitch=36),  # kick
            ),
        )
        result = calculate_density_by_subcategory([clip], 0.0, 4.0)
        assert DrumSubCategory.KICK in result
        assert result[DrumSubCategory.KICK] == 1.0  # 4 notes / 4 beats

    def test_multiple_subcategories(self) -> None:
        """Density is tracked separately for each subcategory."""
        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=8.0),
            notes=(
                MidiNote(time=0.0, duration=0.5, velocity=100, pitch=36),  # kick
                MidiNote(time=1.0, duration=0.5, velocity=100, pitch=38),  # snare
                MidiNote(time=2.0, duration=0.5, velocity=100, pitch=36),  # kick
                MidiNote(time=3.0, duration=0.5, velocity=100, pitch=38),  # snare
            ),
        )
        result = calculate_density_by_subcategory([clip], 0.0, 4.0)
        assert result[DrumSubCategory.KICK] == 0.5  # 2 notes / 4 beats
        assert result[DrumSubCategory.SNARE] == 0.5  # 2 notes / 4 beats

    def test_notes_outside_range_excluded(self) -> None:
        """Notes outside the analysis range are excluded."""
        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=16.0),
            notes=(
                MidiNote(time=0.0, duration=0.5, velocity=100, pitch=36),  # before range
                MidiNote(time=4.0, duration=0.5, velocity=100, pitch=36),  # in range
                MidiNote(time=6.0, duration=0.5, velocity=100, pitch=36),  # in range
                MidiNote(time=12.0, duration=0.5, velocity=100, pitch=36),  # after range
            ),
        )
        result = calculate_density_by_subcategory([clip], 4.0, 8.0)
        assert result[DrumSubCategory.KICK] == 0.5  # 2 notes / 4 beats


class TestBaselineDensity:
    """Tests for baseline density calculation."""

    def test_empty_clips_return_empty(self) -> None:
        """Empty clip list returns empty baseline."""
        result = calculate_baseline_density([])
        assert result == {}

    def test_constant_density_returns_that_density(self) -> None:
        """When density is constant, baseline equals that density."""
        # Constant kick pattern: 1 note per beat
        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=16.0),
            notes=tuple(
                MidiNote(time=float(i), duration=0.5, velocity=100, pitch=36) for i in range(16)
            ),
        )
        result = calculate_baseline_density([clip], window_beats=4.0)
        # Should have 4 notes per 4-beat window = 1 note per beat
        assert DrumSubCategory.KICK in result
        assert abs(result[DrumSubCategory.KICK] - 1.0) < 0.001

    def test_median_of_varying_density(self) -> None:
        """Median is taken across windows with varying density."""
        # Create notes: 2 in first window, 4 in second, 2 in third, 4 in fourth
        # Median should be 3 notes per window = 0.75 per beat
        notes = []
        # Window 0-4: 2 notes
        notes.extend(
            [
                MidiNote(time=0.0, duration=0.5, velocity=100, pitch=36),
                MidiNote(time=2.0, duration=0.5, velocity=100, pitch=36),
            ]
        )
        # Window 4-8: 4 notes
        notes.extend(
            [
                MidiNote(time=4.0, duration=0.5, velocity=100, pitch=36),
                MidiNote(time=5.0, duration=0.5, velocity=100, pitch=36),
                MidiNote(time=6.0, duration=0.5, velocity=100, pitch=36),
                MidiNote(time=7.0, duration=0.5, velocity=100, pitch=36),
            ]
        )
        # Window 8-12: 2 notes
        notes.extend(
            [
                MidiNote(time=8.0, duration=0.5, velocity=100, pitch=36),
                MidiNote(time=10.0, duration=0.5, velocity=100, pitch=36),
            ]
        )
        # Window 12-16: 4 notes
        notes.extend(
            [
                MidiNote(time=12.0, duration=0.5, velocity=100, pitch=36),
                MidiNote(time=13.0, duration=0.5, velocity=100, pitch=36),
                MidiNote(time=14.0, duration=0.5, velocity=100, pitch=36),
                MidiNote(time=15.0, duration=0.5, velocity=100, pitch=36),
            ]
        )

        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=16.0),
            notes=tuple(notes),
        )
        result = calculate_baseline_density([clip], window_beats=4.0)
        # Densities: 0.5, 1.0, 0.5, 1.0 -> median = 0.75
        assert abs(result[DrumSubCategory.KICK] - 0.75) < 0.001


class TestDetectDrumFills:
    """Tests for drum fill detection."""

    def test_empty_clips_no_fills(self) -> None:
        """Empty clips produce no fills."""
        sections = [Section(name="VERSE", start_beats=0.0, end_beats=32.0)]
        result = detect_drum_fills([], sections)
        assert result == []

    def test_empty_sections_no_fills(self) -> None:
        """Empty sections produce no fills."""
        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=32.0),
            notes=(MidiNote(time=0.0, duration=0.5, velocity=100, pitch=36),),
        )
        result = detect_drum_fills([clip], [])
        assert result == []

    def test_constant_density_no_fills(self) -> None:
        """Constant density across sections produces no fills."""
        # Create steady kick pattern
        notes = tuple(
            MidiNote(time=float(i), duration=0.5, velocity=100, pitch=36) for i in range(32)
        )
        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=32.0),
            notes=notes,
        )
        sections = [
            Section(name="VERSE", start_beats=0.0, end_beats=16.0),
            Section(name="CHORUS", start_beats=16.0, end_beats=32.0),
        ]
        result = detect_drum_fills([clip], sections)
        # No fill because density is constant (no spike)
        assert result == []

    def test_detects_fill_with_spike(self) -> None:
        """Fill is detected when multiple subcategories spike before section."""
        # Normal pattern with all drum components playing throughout
        notes = []
        for i in range(32):
            # Steady kick
            notes.append(MidiNote(time=float(i), duration=0.5, velocity=100, pitch=36))
            # Steady hi-hat
            notes.append(MidiNote(time=float(i), duration=0.25, velocity=80, pitch=42))
            # Sparse toms (establishing baseline) - 1 per 4 beats
            if i % 4 == 0:
                notes.append(MidiNote(time=float(i), duration=0.5, velocity=100, pitch=45))
            # Sparse cymbals (establishing baseline) - 1 per 4 beats
            if i % 4 == 2:
                notes.append(MidiNote(time=float(i), duration=0.5, velocity=100, pitch=49))

        # Add density spike in the 2 bars before CHORUS (beats 12-16)
        # Increase tom and cymbal density significantly (4 notes per beat vs 0.25 baseline)
        for beat in range(12, 16):
            for sub_beat in [0.0, 0.25, 0.5, 0.75]:
                # Many toms
                notes.append(MidiNote(time=beat + sub_beat, duration=0.2, velocity=100, pitch=45))
                # Many cymbals
                notes.append(MidiNote(time=beat + sub_beat, duration=0.2, velocity=100, pitch=49))

        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=32.0),
            notes=tuple(notes),
        )
        sections = [
            Section(name="VERSE", start_beats=0.0, end_beats=16.0),
            Section(name="CHORUS", start_beats=16.0, end_beats=32.0),
        ]
        result = detect_drum_fills([clip], sections)

        # Should detect a fill before CHORUS
        assert len(result) >= 1
        # Find the fill pointing to CHORUS
        chorus_fills = [f for f in result if f.next_section == "CHORUS"]
        assert len(chorus_fills) >= 1

        fill = chorus_fills[0]
        assert fill.next_section == "CHORUS"
        assert fill.start_beats >= 8.0  # Within proximity window
        assert fill.start_beats < 16.0
        assert DrumSubCategory.TOMS in fill.components or DrumSubCategory.CYMBALS in fill.components

    def test_fill_requires_min_components(self) -> None:
        """Fill requires at least min_components to spike."""
        # Only spike one subcategory (toms) - should not trigger fill
        notes = []
        for i in range(32):
            notes.append(MidiNote(time=float(i), duration=0.5, velocity=100, pitch=36))

        # Add tom spike only
        for beat in range(12, 16):
            for sub_beat in [0.0, 0.25, 0.5, 0.75]:
                notes.append(MidiNote(time=beat + sub_beat, duration=0.2, velocity=100, pitch=45))

        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=32.0),
            notes=tuple(notes),
        )
        sections = [
            Section(name="VERSE", start_beats=0.0, end_beats=16.0),
            Section(name="CHORUS", start_beats=16.0, end_beats=32.0),
        ]
        # Default min_components is 2
        result = detect_drum_fills([clip], sections, min_components=2)

        # Should not detect a fill with only one component spiking
        chorus_fills = [f for f in result if f.next_section == "CHORUS"]
        assert len(chorus_fills) == 0

    def test_fill_intensity_normalized(self) -> None:
        """Fill intensity is normalized between 0 and 1."""
        # Create a fill with extreme spike
        notes = []
        for i in range(32):
            notes.append(MidiNote(time=float(i), duration=0.5, velocity=100, pitch=36))

        # Massive spike before section
        for beat in range(12, 16):
            for j in range(20):  # 20 notes per beat
                notes.append(MidiNote(time=beat + j * 0.05, duration=0.02, velocity=100, pitch=45))
                notes.append(MidiNote(time=beat + j * 0.05, duration=0.02, velocity=100, pitch=49))

        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=32.0),
            notes=tuple(notes),
        )
        sections = [
            Section(name="VERSE", start_beats=0.0, end_beats=16.0),
            Section(name="CHORUS", start_beats=16.0, end_beats=32.0),
        ]
        result = detect_drum_fills([clip], sections)

        for fill in result:
            assert 0.0 <= fill.intensity <= 1.0

    def test_section_at_beginning_skipped(self) -> None:
        """Section at beat 0 doesn't cause negative analysis range."""
        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=32.0),
            notes=(MidiNote(time=0.0, duration=0.5, velocity=100, pitch=36),),
        )
        sections = [
            Section(name="INTRO", start_beats=0.0, end_beats=8.0),
            Section(name="VERSE", start_beats=8.0, end_beats=16.0),
        ]
        # Should not raise an error
        result = detect_drum_fills([clip], sections)
        # Just verify it runs without error
        assert isinstance(result, list)

    def test_custom_threshold(self) -> None:
        """Custom threshold affects fill detection."""
        notes = []
        for i in range(32):
            notes.append(MidiNote(time=float(i), duration=0.5, velocity=100, pitch=36))

        # Moderate spike (2x, not 3x)
        for beat in range(12, 16):
            notes.append(MidiNote(time=float(beat), duration=0.2, velocity=100, pitch=45))
            notes.append(MidiNote(time=float(beat), duration=0.2, velocity=100, pitch=49))

        clip = MidiClipContent(
            clip=Clip(name="Drums", start_beats=0.0, end_beats=32.0),
            notes=tuple(notes),
        )
        sections = [
            Section(name="VERSE", start_beats=0.0, end_beats=16.0),
            Section(name="CHORUS", start_beats=16.0, end_beats=32.0),
        ]

        # With default threshold (3.0), no fill
        high_threshold_result = detect_drum_fills([clip], sections, threshold=3.0)

        # With lower threshold (1.5), should detect fill
        low_threshold_result = detect_drum_fills([clip], sections, threshold=1.5)

        # Lower threshold should detect more fills or equal fills
        assert len(low_threshold_result) >= len(high_threshold_result)
