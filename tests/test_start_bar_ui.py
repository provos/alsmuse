"""Tests for start_bar_ui module."""

from __future__ import annotations

import termios
from io import StringIO
from unittest.mock import patch

from rich.console import Console

from alsmuse.models import Clip, LiveSet, Tempo, Track
from alsmuse.start_bar_ui import (
    TimelineConfig,
    TrackDisplay,
    _merge_bar_ranges,
    beats_to_bar,
    compute_timeline_config,
    get_track_priority,
    prompt_start_bar,
    render_bar_axis,
    render_help_text,
    render_selection_indicator,
    render_timeline,
    render_track_row,
    select_display_tracks,
    select_start_bar_interactive,
)


def make_live_set(
    tracks: list[tuple[str, list[tuple[float, float]]]],
    time_signature: tuple[int, int] = (4, 4),
    bpm: float = 120.0,
) -> LiveSet:
    """Create a LiveSet for testing.

    Args:
        tracks: List of (track_name, [(start_beats, end_beats), ...]) tuples.
        time_signature: Time signature tuple.
        bpm: Tempo in BPM.

    Returns:
        LiveSet with the specified tracks and clips.
    """
    tempo = Tempo(bpm=bpm, time_signature=time_signature)
    track_list: list[Track] = []

    for name, clips in tracks:
        clip_list = tuple(
            Clip(name=f"{name}_clip_{i}", start_beats=start, end_beats=end)
            for i, (start, end) in enumerate(clips)
        )
        track_list.append(Track(name=name, track_type="midi", clips=clip_list, enabled=True))

    return LiveSet(tempo=tempo, tracks=tuple(track_list))


class TestBeatsToBar:
    """Tests for beats_to_bar function."""

    def test_standard_4_4_time(self) -> None:
        """Converts beats to bars correctly in 4/4 time."""
        assert beats_to_bar(0.0) == 0
        assert beats_to_bar(4.0) == 1
        assert beats_to_bar(8.0) == 2
        assert beats_to_bar(7.9) == 1

    def test_3_4_time(self) -> None:
        """Converts beats to bars correctly in 3/4 time."""
        assert beats_to_bar(0.0, (3, 4)) == 0
        assert beats_to_bar(3.0, (3, 4)) == 1
        assert beats_to_bar(6.0, (3, 4)) == 2

    def test_6_8_time(self) -> None:
        """Converts beats to bars correctly in 6/8 time."""
        # 6/8 has 6 eighth notes = 3 quarter notes = 3 beats per bar
        assert beats_to_bar(0.0, (6, 8)) == 0
        assert beats_to_bar(3.0, (6, 8)) == 1
        assert beats_to_bar(6.0, (6, 8)) == 2


class TestGetTrackPriority:
    """Tests for get_track_priority function."""

    def test_structure_has_highest_priority(self) -> None:
        """Structure tracks have highest priority (lowest value)."""
        assert get_track_priority("STRUCTURE") == 0
        assert get_track_priority("Structure Markers") == 0

    def test_drums_have_high_priority(self) -> None:
        """Drum tracks have high priority."""
        assert get_track_priority("Drums") == 1
        assert get_track_priority("Main Drums") == 1

    def test_unknown_tracks_have_lowest_priority(self) -> None:
        """Unknown track names get lowest priority."""
        priority = get_track_priority("Random FX Chain")
        assert priority > get_track_priority("Drums")

    def test_case_insensitive(self) -> None:
        """Priority matching is case-insensitive."""
        assert get_track_priority("DRUMS") == get_track_priority("drums")
        assert get_track_priority("Bass Line") == get_track_priority("bass line")


class TestMergeBarRanges:
    """Tests for _merge_bar_ranges function."""

    def test_empty_list(self) -> None:
        """Returns empty list for empty input."""
        assert _merge_bar_ranges([]) == []

    def test_single_range(self) -> None:
        """Returns single range unchanged."""
        assert _merge_bar_ranges([(0, 4)]) == [(0, 4)]

    def test_non_overlapping_ranges(self) -> None:
        """Returns non-overlapping ranges as-is (sorted)."""
        result = _merge_bar_ranges([(8, 12), (0, 4)])
        assert result == [(0, 4), (8, 12)]

    def test_overlapping_ranges_merged(self) -> None:
        """Overlapping ranges are merged."""
        result = _merge_bar_ranges([(0, 6), (4, 10)])
        assert result == [(0, 10)]

    def test_adjacent_ranges_merged(self) -> None:
        """Adjacent ranges are merged."""
        result = _merge_bar_ranges([(0, 4), (4, 8)])
        assert result == [(0, 8)]

    def test_multiple_merges(self) -> None:
        """Multiple overlapping groups are merged correctly."""
        result = _merge_bar_ranges([(0, 4), (2, 6), (10, 14), (12, 16)])
        assert result == [(0, 6), (10, 16)]


class TestSelectDisplayTracks:
    """Tests for select_display_tracks function."""

    def test_returns_empty_for_no_clips(self) -> None:
        """Returns empty list when no tracks have clips."""
        live_set = make_live_set([("Empty Track", [])])
        result = select_display_tracks(live_set)
        assert result == []

    def test_returns_tracks_sorted_by_priority(self) -> None:
        """Returns tracks sorted by priority (most relevant first)."""
        live_set = make_live_set(
            [
                ("Random FX", [(0, 8)]),
                ("Drums", [(0, 8)]),
                ("STRUCTURE", [(0, 8)]),
            ]
        )
        result = select_display_tracks(live_set)
        assert len(result) == 3
        assert result[0].original_name == "STRUCTURE"
        assert result[1].original_name == "Drums"
        assert result[2].original_name == "Random FX"

    def test_limits_to_max_tracks(self) -> None:
        """Limits result to max_tracks parameter."""
        tracks = [(f"Track {i}", [(0, 8)]) for i in range(20)]
        live_set = make_live_set(tracks)
        result = select_display_tracks(live_set, max_tracks=5)
        assert len(result) == 5

    def test_truncates_long_track_names(self) -> None:
        """Truncates track names longer than 12 characters."""
        live_set = make_live_set([("This Is A Very Long Track Name", [(0, 8)])])
        result = select_display_tracks(live_set)
        assert len(result[0].name) == 12
        assert result[0].original_name == "This Is A Very Long Track Name"

    def test_converts_clips_to_bar_ranges(self) -> None:
        """Converts clip beat positions to bar ranges."""
        # Clips at beats 0-8 and 16-24 = bars 0-2 and 4-6
        live_set = make_live_set([("Track", [(0.0, 8.0), (16.0, 24.0)])])
        result = select_display_tracks(live_set)
        assert result[0].bar_ranges == [(0, 2), (4, 6)]


class TestComputeTimelineConfig:
    """Tests for compute_timeline_config function."""

    def test_includes_all_track_content(self) -> None:
        """Config includes all track content in bar range."""
        tracks = [
            TrackDisplay("Track 1", "Track 1", [(0, 8)]),
            TrackDisplay("Track 2", "Track 2", [(16, 24)]),
        ]
        config = compute_timeline_config(tracks, suggested_bar=0)
        assert config.bar_start == 0
        assert config.bar_end >= 24

    def test_includes_suggested_bar(self) -> None:
        """Config includes the suggested bar in range."""
        tracks = [TrackDisplay("Track", "Track", [(8, 16)])]
        config = compute_timeline_config(tracks, suggested_bar=4)
        assert config.bar_start <= 4 < config.bar_end

    def test_respects_terminal_width(self) -> None:
        """Timeline width respects terminal width constraint."""
        tracks = [TrackDisplay("Track", "Track", [(0, 100)])]
        config = compute_timeline_config(tracks, suggested_bar=0, terminal_width=80)
        assert config.total_width <= 80


class TestRenderFunctions:
    """Tests for rendering functions."""

    def test_render_bar_axis_returns_text(self) -> None:
        """render_bar_axis returns a Rich Text object."""
        config = TimelineConfig(
            bar_start=0,
            bar_end=32,
            track_label_width=12,
            timeline_width=64,
            total_width=80,
        )
        result = render_bar_axis(config)
        assert result is not None
        assert len(str(result)) > 0

    def test_render_track_row_returns_text(self) -> None:
        """render_track_row returns a Rich Text object."""
        config = TimelineConfig(
            bar_start=0,
            bar_end=16,
            track_label_width=12,
            timeline_width=48,
            total_width=64,
        )
        track = TrackDisplay("Drums", "Drums", [(0, 8)])
        result = render_track_row(track, config, selected_bar=0)
        assert result is not None
        assert "Drums" in str(result)

    def test_render_selection_indicator_shows_marker(self) -> None:
        """render_selection_indicator includes selection marker."""
        config = TimelineConfig(
            bar_start=0,
            bar_end=16,
            track_label_width=12,
            timeline_width=48,
            total_width=64,
        )
        result = render_selection_indicator(config, selected_bar=4)
        # Should contain the triangle marker
        result_str = str(result)
        assert "\u25b2" in result_str or "\u25c4" in result_str or "\u25ba" in result_str

    def test_render_timeline_includes_all_elements(self) -> None:
        """render_timeline includes axis, tracks, and selection."""
        tracks = [TrackDisplay("Drums", "Drums", [(0, 8)])]
        config = TimelineConfig(
            bar_start=0,
            bar_end=16,
            track_label_width=12,
            timeline_width=48,
            total_width=64,
        )
        # Use a console with StringIO to capture output
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        render_timeline(tracks, config, selected_bar=0, console=console)
        result = output.getvalue()
        assert "Drums" in result
        assert "Start:" in result

    def test_render_help_text_includes_key_info(self) -> None:
        """render_help_text includes keyboard controls."""
        result = render_help_text()
        # render_help_text returns a Rich Text object, convert to plain string
        plain_text = result.plain
        assert "Enter" in plain_text
        assert "Confirm" in plain_text


class TestPromptStartBar:
    """Tests for prompt_start_bar function."""

    def test_returns_suggested_when_not_tty(self) -> None:
        """Returns suggested bar when not running in TTY."""
        live_set = make_live_set([("Drums", [(0, 8)])])

        with patch("alsmuse.start_bar_ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False

            result = prompt_start_bar(live_set, suggested=8)

        assert result == 8

    def test_falls_back_to_simple_prompt_when_no_tracks(self) -> None:
        """Falls back to simple prompt when no displayable tracks."""
        live_set = make_live_set([])  # No tracks

        with patch("alsmuse.start_bar_ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("alsmuse.start_bar_ui.Prompt.ask") as mock_ask:
                mock_ask.return_value = "y"

                result = prompt_start_bar(live_set, suggested=4)

        assert result == 4

    def test_simple_prompt_accepts_yes(self) -> None:
        """Simple prompt returns suggested bar on 'y' input."""
        live_set = make_live_set([])

        with patch("alsmuse.start_bar_ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("alsmuse.start_bar_ui.Prompt.ask") as mock_ask:
                mock_ask.return_value = "y"

                result = prompt_start_bar(live_set, suggested=12)

        assert result == 12

    def test_simple_prompt_accepts_no(self) -> None:
        """Simple prompt returns 0 on 'n' input."""
        live_set = make_live_set([])

        with patch("alsmuse.start_bar_ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("alsmuse.start_bar_ui.Prompt.ask") as mock_ask:
                mock_ask.return_value = "n"

                result = prompt_start_bar(live_set, suggested=8)

        assert result == 0

    def test_simple_prompt_accepts_number(self) -> None:
        """Simple prompt returns custom bar number."""
        live_set = make_live_set([])

        with patch("alsmuse.start_bar_ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("alsmuse.start_bar_ui.Prompt.ask") as mock_ask:
                mock_ask.return_value = "16"

                result = prompt_start_bar(live_set, suggested=8)

        assert result == 16

    def test_interactive_mode_returns_selected_bar(self) -> None:
        """Interactive mode returns the selected bar."""
        live_set = make_live_set([("Drums", [(0, 32)])])

        with patch("alsmuse.start_bar_ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            # Mock _interactive_select to return a specific value
            with patch("alsmuse.start_bar_ui._interactive_select") as mock_select:
                mock_select.return_value = 8

                result = prompt_start_bar(live_set, suggested=8)

        assert result == 8

    def test_interactive_mode_cancel_returns_zero(self) -> None:
        """Interactive mode returns 0 when user cancels."""
        live_set = make_live_set([("Drums", [(0, 32)])])

        with patch("alsmuse.start_bar_ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            # Mock _interactive_select to return 0 (cancelled)
            with patch("alsmuse.start_bar_ui._interactive_select") as mock_select:
                mock_select.return_value = 0

                result = prompt_start_bar(live_set, suggested=8)

        assert result == 0

    def test_interactive_mode_falls_back_on_terminal_error(self) -> None:
        """Interactive mode falls back to simple prompt on terminal error."""
        live_set = make_live_set([("Drums", [(0, 32)])])

        with patch("alsmuse.start_bar_ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            # Mock _interactive_select to raise an error
            with patch("alsmuse.start_bar_ui._interactive_select") as mock_select:
                mock_select.side_effect = termios.error("terminal error")

                with patch("alsmuse.start_bar_ui._simple_prompt") as mock_simple:
                    mock_simple.return_value = 4

                    result = prompt_start_bar(live_set, suggested=8)

        assert result == 4


class TestSelectStartBarInteractive:
    """Tests for select_start_bar_interactive function."""

    def test_uses_prompt_start_bar(self) -> None:
        """Uses prompt_start_bar internally."""
        live_set = make_live_set([("Drums", [(0, 8)])])

        with patch("alsmuse.start_bar_ui.prompt_start_bar") as mock_prompt:
            mock_prompt.return_value = 12

            result = select_start_bar_interactive(live_set, suggested=8)

        assert result == 12
        mock_prompt.assert_called_once()
