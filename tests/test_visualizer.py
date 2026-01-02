"""Tests for the visualizer module.

These are functional tests that validate the observable behavior of the
visualizer components without mocking implementation details. We test
the frame rendering, state computation, and helper functions.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from alsmuse.models import Phrase, TrackEvent
from alsmuse.visualizer import (
    BACKGROUND_COLOR,
    CATEGORY_COLORS,
    EVENT_FADE_SECONDS,
    FRAME_RATE,
    PROGRESS_BAR_HEIGHT,
    PROGRESS_BAR_WIDTH,
    PROGRESS_BAR_X,
    PROGRESS_BAR_Y,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    FontSet,
    FrameState,
    build_frame_states,
    compute_phrase_times,
    format_event_badge,
    format_timecode,
    generate_visualizer,
    get_category_color,
    render_frame,
)


class TestFormatTimecode:
    """Tests for the format_timecode helper function."""

    def test_zero_seconds(self) -> None:
        """Zero seconds formats as 00:00."""
        assert format_timecode(0) == "00:00"

    def test_less_than_minute(self) -> None:
        """Times under a minute show 00:SS."""
        assert format_timecode(30) == "00:30"
        assert format_timecode(59.9) == "00:59"

    def test_one_minute(self) -> None:
        """One minute formats correctly."""
        assert format_timecode(60) == "01:00"

    def test_minutes_and_seconds(self) -> None:
        """Mixed minutes and seconds format correctly."""
        assert format_timecode(90) == "01:30"
        assert format_timecode(125) == "02:05"
        assert format_timecode(3661) == "61:01"  # Over an hour

    def test_fractional_seconds_truncated(self) -> None:
        """Fractional parts are truncated (not rounded)."""
        assert format_timecode(59.9) == "00:59"
        assert format_timecode(60.5) == "01:00"


class TestGetCategoryColor:
    """Tests for the get_category_color helper function."""

    def test_known_categories_return_defined_colors(self) -> None:
        """Known categories return their defined colors."""
        assert get_category_color("drums") == CATEGORY_COLORS["drums"]
        assert get_category_color("bass") == CATEGORY_COLORS["bass"]
        assert get_category_color("vocals") == CATEGORY_COLORS["vocals"]
        assert get_category_color("keys") == CATEGORY_COLORS["keys"]
        assert get_category_color("guitar") == CATEGORY_COLORS["guitar"]

    def test_unknown_category_returns_other_color(self) -> None:
        """Unknown categories fall back to 'other' color."""
        assert get_category_color("unknown_instrument") == CATEGORY_COLORS["other"]
        assert get_category_color("xyz") == CATEGORY_COLORS["other"]

    def test_case_insensitive(self) -> None:
        """Category matching is case-insensitive."""
        assert get_category_color("DRUMS") == CATEGORY_COLORS["drums"]
        assert get_category_color("Drums") == CATEGORY_COLORS["drums"]
        assert get_category_color("BASS") == CATEGORY_COLORS["bass"]


class TestFormatEventBadge:
    """Tests for the format_event_badge helper function."""

    def test_enter_event_has_plus_prefix(self) -> None:
        """Enter events show + prefix."""
        event = TrackEvent(beat=0, track_name="Kick", event_type="enter", category="drums")
        assert format_event_badge(event) == "+DRUMS"

    def test_exit_event_has_minus_prefix(self) -> None:
        """Exit events show - prefix."""
        event = TrackEvent(beat=8, track_name="Bass Line", event_type="exit", category="bass")
        assert format_event_badge(event) == "-BASS"

    def test_category_is_uppercased(self) -> None:
        """Category name is displayed in uppercase."""
        event = TrackEvent(beat=0, track_name="Piano", event_type="enter", category="keys")
        assert format_event_badge(event) == "+KEYS"


class TestComputePhraseTimes:
    """Tests for the compute_phrase_times helper function."""

    def test_empty_phrases_returns_empty_list(self) -> None:
        """Empty phrase list returns empty list."""
        assert compute_phrase_times([], 120.0) == []

    def test_single_phrase_computes_times(self) -> None:
        """Single phrase computes correct start and end times."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=8,
                section_name="INTRO",
                is_section_start=True,
            )
        ]
        # At 120 BPM: 1 beat = 0.5 seconds, so 8 beats = 4 seconds
        times = compute_phrase_times(phrases, 120.0)

        assert len(times) == 1
        assert times[0][0] == pytest.approx(0.0)
        assert times[0][1] == pytest.approx(4.0)

    def test_multiple_phrases_compute_times(self) -> None:
        """Multiple phrases compute correct sequential times."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=8,
                section_name="INTRO",
                is_section_start=True,
            ),
            Phrase(
                start_beats=8,
                end_beats=16,
                section_name="...",
                is_section_start=False,
            ),
            Phrase(
                start_beats=16,
                end_beats=24,
                section_name="VERSE",
                is_section_start=True,
            ),
        ]
        times = compute_phrase_times(phrases, 120.0)

        assert len(times) == 3
        assert times[0] == pytest.approx((0.0, 4.0))
        assert times[1] == pytest.approx((4.0, 8.0))
        assert times[2] == pytest.approx((8.0, 12.0))


class TestFontSet:
    """Tests for the FontSet class."""

    def test_load_creates_font_set(self) -> None:
        """FontSet.load() creates a valid FontSet instance."""
        fonts = FontSet.load()

        assert fonts is not None
        assert fonts.section is not None
        assert fonts.timecode is not None
        assert fonts.prev_next is not None
        assert fonts.current is not None
        assert fonts.current_bold is not None
        assert fonts.event is not None

    def test_load_with_invalid_path_uses_fallback(self) -> None:
        """Loading with invalid font path falls back to default."""
        fonts = FontSet.load(Path("/nonexistent/font.ttf"))

        # Should still work with fallback
        assert fonts is not None


class TestFrameState:
    """Tests for the FrameState dataclass."""

    def test_frame_state_creation(self) -> None:
        """FrameState can be created with all required fields."""
        state = FrameState(
            current_time=10.5,
            total_time=180.0,
            section_name="CHORUS",
            prev_lyric="Previous line",
            current_lyric="Current line",
            next_lyric="Next line",
            active_events=[],
        )

        assert state.current_time == 10.5
        assert state.total_time == 180.0
        assert state.section_name == "CHORUS"
        assert state.prev_lyric == "Previous line"
        assert state.current_lyric == "Current line"
        assert state.next_lyric == "Next line"
        assert state.active_events == []


class TestBuildFrameStates:
    """Tests for the build_frame_states generator function."""

    def test_empty_phrases_yields_nothing(self) -> None:
        """Empty phrase list yields no frames."""
        states = list(build_frame_states([], 120.0, 0.0))
        assert states == []

    def test_generates_correct_number_of_frames(self) -> None:
        """Generator yields correct number of frames based on duration."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=8,
                section_name="INTRO",
                is_section_start=True,
                lyric="Hello world",
            )
        ]
        # 8 beats at 120 BPM = 4 seconds
        # 4 seconds * 24 fps = 96 frames
        total_beats = 8.0
        states = list(build_frame_states(phrases, 120.0, total_beats))

        assert len(states) == 96

    def test_frame_times_progress_correctly(self) -> None:
        """Frame times increase by 1/frame_rate each frame."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=8,
                section_name="INTRO",
                is_section_start=True,
            )
        ]
        states = list(build_frame_states(phrases, 120.0, 8.0))

        # Check first few frames
        assert states[0].current_time == pytest.approx(0.0)
        assert states[1].current_time == pytest.approx(1 / FRAME_RATE)
        assert states[2].current_time == pytest.approx(2 / FRAME_RATE)

    def test_section_name_reflects_current_phrase(self) -> None:
        """Section name in state reflects the current phrase."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=8,
                section_name="INTRO",
                is_section_start=True,
            ),
            Phrase(
                start_beats=8,
                end_beats=16,
                section_name="VERSE",
                is_section_start=True,
            ),
        ]
        states = list(build_frame_states(phrases, 120.0, 16.0))

        # First phrase (0-4 seconds = frames 0-95)
        assert states[0].section_name == "INTRO"
        assert states[95].section_name == "INTRO"

        # Second phrase (4-8 seconds = frames 96-191)
        assert states[96].section_name == "VERSE"

    def test_lyrics_show_previous_current_next(self) -> None:
        """Frame states include prev/current/next lyrics."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=8,
                section_name="INTRO",
                is_section_start=True,
                lyric="First line",
            ),
            Phrase(
                start_beats=8,
                end_beats=16,
                section_name="...",
                is_section_start=False,
                lyric="Second line",
            ),
            Phrase(
                start_beats=16,
                end_beats=24,
                section_name="...",
                is_section_start=False,
                lyric="Third line",
            ),
        ]
        states = list(build_frame_states(phrases, 120.0, 24.0))

        # During first phrase: no prev, current is first, next is second
        assert states[0].prev_lyric == ""
        assert states[0].current_lyric == "First line"
        assert states[0].next_lyric == "Second line"

        # During second phrase: prev is first, current is second, next is third
        assert states[96].prev_lyric == "First line"
        assert states[96].current_lyric == "Second line"
        assert states[96].next_lyric == "Third line"

        # During third phrase: prev is second, current is third, no next
        assert states[192].prev_lyric == "Second line"
        assert states[192].current_lyric == "Third line"
        assert states[192].next_lyric == ""

    def test_total_time_is_correct(self) -> None:
        """Total time in frame state reflects actual total duration."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=8,
                section_name="INTRO",
                is_section_start=True,
            )
        ]
        # 8 beats at 120 BPM = 4 seconds
        states = list(build_frame_states(phrases, 120.0, 8.0))

        assert all(s.total_time == pytest.approx(4.0) for s in states)


class TestRenderFrame:
    """Tests for the render_frame function."""

    @pytest.fixture
    def fonts(self) -> FontSet:
        """Load fonts for rendering tests."""
        return FontSet.load()

    def test_renders_correct_size_image(self, fonts: FontSet) -> None:
        """Rendered frame has correct dimensions."""
        state = FrameState(
            current_time=0.0,
            total_time=60.0,
            section_name="INTRO",
            prev_lyric="",
            current_lyric="Hello",
            next_lyric="",
            active_events=[],
        )

        img = render_frame(state, fonts)

        assert img.size == (VIDEO_WIDTH, VIDEO_HEIGHT)

    def test_renders_rgb_mode(self, fonts: FontSet) -> None:
        """Rendered frame is in RGB mode."""
        state = FrameState(
            current_time=0.0,
            total_time=60.0,
            section_name="INTRO",
            prev_lyric="",
            current_lyric="",
            next_lyric="",
            active_events=[],
        )

        img = render_frame(state, fonts)

        assert img.mode == "RGB"

    def test_background_is_black(self, fonts: FontSet) -> None:
        """Empty frame has black background."""
        state = FrameState(
            current_time=0.0,
            total_time=60.0,
            section_name="INTRO",
            prev_lyric="",
            current_lyric="",
            next_lyric="",
            active_events=[],
        )

        img = render_frame(state, fonts)

        # Check corner pixels are black
        assert img.getpixel((0, 0)) == BACKGROUND_COLOR
        assert img.getpixel((VIDEO_WIDTH - 1, 0)) == BACKGROUND_COLOR
        assert img.getpixel((0, VIDEO_HEIGHT - 1)) == BACKGROUND_COLOR

    def test_progress_bar_renders_at_bottom(self, fonts: FontSet) -> None:
        """Progress bar renders at expected location."""
        state = FrameState(
            current_time=30.0,  # Half way
            total_time=60.0,
            section_name="INTRO",
            prev_lyric="",
            current_lyric="",
            next_lyric="",
            active_events=[],
        )

        img = render_frame(state, fonts)

        # Check progress bar fill (white) is present at expected location
        # At 50% progress, the fill should extend to about half the bar width
        expected_fill_end = PROGRESS_BAR_X + int(PROGRESS_BAR_WIDTH * 0.5)

        # Check that there's a white pixel in the filled area
        fill_color = img.getpixel((PROGRESS_BAR_X + 10, PROGRESS_BAR_Y + PROGRESS_BAR_HEIGHT // 2))
        assert fill_color == (255, 255, 255)  # White fill

        # Check that unfilled area is grey
        unfilled_x = expected_fill_end + 100
        if unfilled_x < PROGRESS_BAR_X + PROGRESS_BAR_WIDTH:
            unfilled_color = img.getpixel((unfilled_x, PROGRESS_BAR_Y + PROGRESS_BAR_HEIGHT // 2))
            assert unfilled_color == (51, 51, 51)  # Grey background

    def test_renders_without_error_with_events(self, fonts: FontSet) -> None:
        """Frame renders successfully with active events."""
        enter_event = TrackEvent(beat=0, track_name="Kick", event_type="enter", category="drums")
        exit_event = TrackEvent(
            beat=0, track_name="Synth Lead", event_type="exit", category="synth"
        )

        state = FrameState(
            current_time=0.5,
            total_time=60.0,
            section_name="CHORUS",
            prev_lyric="Previous",
            current_lyric="Current lyric here",
            next_lyric="Next",
            active_events=[(enter_event, 0.5), (exit_event, 0.5)],
        )

        img = render_frame(state, fonts)

        # Should render without error
        assert img is not None
        assert img.size == (VIDEO_WIDTH, VIDEO_HEIGHT)

    def test_renders_with_multiline_lyrics(self, fonts: FontSet) -> None:
        """Frame renders successfully with multi-line lyrics."""
        state = FrameState(
            current_time=0.0,
            total_time=60.0,
            section_name="VERSE",
            prev_lyric="",
            current_lyric="Line one\nLine two",
            next_lyric="",
            active_events=[],
        )

        img = render_frame(state, fonts)

        # Should render without error
        assert img is not None


class TestGenerateVisualizer:
    """Tests for the generate_visualizer function."""

    def test_raises_on_empty_phrases(self) -> None:
        """Raises ValueError when phrases list is empty."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="No phrases"):
                generate_visualizer(
                    phrases=[],
                    bpm=120.0,
                    total_beats=0.0,
                    output_path=output_path,
                )
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_raises_on_zero_duration(self) -> None:
        """Raises ValueError when video would have zero frames."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=0,  # Zero duration
                section_name="INTRO",
                is_section_start=True,
            )
        ]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="zero frames"):
                generate_visualizer(
                    phrases=phrases,
                    bpm=120.0,
                    total_beats=0.0,
                    output_path=output_path,
                )
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_progress_callback_is_called(self) -> None:
        """Progress callback is called during rendering."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=2,  # Short phrase for fast test
                section_name="INTRO",
                is_section_start=True,
            )
        ]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = Path(f.name)

        callback_calls: list[tuple[int, int]] = []

        def progress_callback(frame_num: int, total_frames: int) -> None:
            callback_calls.append((frame_num, total_frames))

        try:
            # Skip this test if ffmpeg is not available
            if shutil.which("ffmpeg") is None:
                pytest.skip("ffmpeg not available")

            generate_visualizer(
                phrases=phrases,
                bpm=120.0,
                total_beats=2.0,  # 1 second at 120 BPM
                output_path=output_path,
                progress_callback=progress_callback,
            )

            # Should have been called for each frame
            expected_frames = int(1.0 * FRAME_RATE)  # 24 frames for 1 second
            assert len(callback_calls) == expected_frames

            # Each call should have correct total
            assert all(total == expected_frames for _, total in callback_calls)

            # Frame numbers should increment
            assert [f for f, _ in callback_calls] == list(range(1, expected_frames + 1))

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg not available")
    def test_generates_valid_mp4_file(self) -> None:
        """Generates a valid MP4 file when ffmpeg is available."""
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=2,  # Short for fast test
                section_name="INTRO",
                is_section_start=True,
                lyric="Test lyric",
            )
        ]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = generate_visualizer(
                phrases=phrases,
                bpm=120.0,
                total_beats=2.0,
                output_path=output_path,
            )

            assert result == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

        finally:
            if output_path.exists():
                output_path.unlink()


class TestEventFading:
    """Tests for event fade-out behavior."""

    def test_events_appear_and_fade(self) -> None:
        """Events appear at phrase start and fade over time."""
        event = TrackEvent(beat=0, track_name="Drums", event_type="enter", category="drums")
        phrases = [
            Phrase(
                start_beats=0,
                end_beats=8,
                section_name="INTRO",
                is_section_start=True,
                events=(event,),
            )
        ]

        states = list(build_frame_states(phrases, 120.0, 8.0))

        # Events may not be immediately active in first frame due to timing
        # Check early frames for event presence
        early_states = states[:FRAME_RATE]  # First second
        events_found = any(len(s.active_events) > 0 for s in early_states)
        assert events_found, "Event should appear in early frames"

        # Later frames should have fading events
        # After EVENT_FADE_SECONDS, events should be removed
        fade_frame = int(EVENT_FADE_SECONDS * FRAME_RATE)
        if fade_frame < len(states):
            late_state = states[fade_frame + 10]
            # Events should have faded by now
            assert len(late_state.active_events) == 0


class TestVisualizerConstants:
    """Tests for visualizer constants and configuration."""

    def test_video_dimensions_are_720p(self) -> None:
        """Video dimensions are 720p."""
        assert VIDEO_WIDTH == 1280
        assert VIDEO_HEIGHT == 720

    def test_frame_rate_is_24fps(self) -> None:
        """Frame rate is 24 fps."""
        assert FRAME_RATE == 24

    def test_category_colors_defined_for_all_categories(self) -> None:
        """All expected categories have defined colors."""
        expected_categories = [
            "drums",
            "bass",
            "vocals",
            "keys",
            "guitar",
            "pad",
            "synth",
            "fx",
            "lead",
            "other",
        ]
        for cat in expected_categories:
            assert cat in CATEGORY_COLORS
            color = CATEGORY_COLORS[cat]
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_progress_bar_dimensions(self) -> None:
        """Progress bar dimensions are reasonable."""
        assert PROGRESS_BAR_X > 0
        assert PROGRESS_BAR_Y > 0
        assert PROGRESS_BAR_WIDTH > 0
        assert PROGRESS_BAR_HEIGHT > 0
        assert PROGRESS_BAR_X + PROGRESS_BAR_WIDTH <= VIDEO_WIDTH
        assert PROGRESS_BAR_Y + PROGRESS_BAR_HEIGHT <= VIDEO_HEIGHT
