"""Unit tests for ALS file parsing.

Tests for group parsing and track enabled/disabled state detection.
"""

import gzip
from pathlib import Path

import pytest

from alsmuse.models import Group, LiveSet, Track
from alsmuse.parser import parse_als_file


def create_als_file(tmp_path: Path, xml_content: str) -> Path:
    """Create a gzip-compressed ALS file with the given XML content."""
    als_file = tmp_path / "test.als"
    with gzip.open(als_file, "wt", encoding="utf-8") as f:
        f.write(xml_content)
    return als_file


# Base ALS XML template with placeholders for tracks
ALS_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<Ableton>
  <LiveSet>
    <MainTrack>
      <DeviceChain>
        <Mixer>
          <Tempo>
            <Manual Value="120"/>
          </Tempo>
        </Mixer>
      </DeviceChain>
    </MainTrack>
    <Tracks>
{tracks}
    </Tracks>
  </LiveSet>
</Ableton>
"""

# Template for a MIDI track with configurable enabled state and group
MIDI_TRACK_TEMPLATE = """\
      <MidiTrack Id="{id}">
        <Name>
          <EffectiveName Value="{name}"/>
          <UserName Value="{name}"/>
        </Name>
        <TrackGroupId Value="{group_id}"/>
        <DeviceChain>
          <Mixer>
            <Speaker>
              <Manual Value="{enabled}"/>
            </Speaker>
          </Mixer>
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events/>
              </ArrangerAutomation>
            </ClipTimeable>
          </MainSequencer>
        </DeviceChain>
      </MidiTrack>"""

# Template for a Group track with configurable enabled state
GROUP_TRACK_TEMPLATE = """\
      <GroupTrack Id="{id}">
        <Name>
          <EffectiveName Value="{name}"/>
          <UserName Value="{name}"/>
        </Name>
        <TrackGroupId Value="{group_id}"/>
        <DeviceChain>
          <Mixer>
            <Speaker>
              <Manual Value="{enabled}"/>
            </Speaker>
          </Mixer>
        </DeviceChain>
      </GroupTrack>"""


def make_midi_track(
    track_id: int, name: str, enabled: bool = True, group_id: int = -1
) -> str:
    """Create a MIDI track XML string."""
    return MIDI_TRACK_TEMPLATE.format(
        id=track_id,
        name=name,
        enabled="true" if enabled else "false",
        group_id=group_id,
    )


def make_group_track(
    group_id: int, name: str, enabled: bool = True, parent_group_id: int = -1
) -> str:
    """Create a Group track XML string."""
    return GROUP_TRACK_TEMPLATE.format(
        id=group_id,
        name=name,
        enabled="true" if enabled else "false",
        group_id=parent_group_id,
    )


class TestTrackEnabledState:
    """Tests for track enabled/disabled state parsing."""

    def test_track_enabled_by_default(self, tmp_path: Path) -> None:
        """Track without Speaker element is enabled by default."""
        # Minimal track without explicit Speaker element
        xml = ALS_TEMPLATE.format(
            tracks="""\
      <MidiTrack Id="1">
        <Name>
          <EffectiveName Value="Test"/>
          <UserName Value="Test"/>
        </Name>
        <TrackGroupId Value="-1"/>
        <DeviceChain>
          <Mixer/>
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events/>
              </ArrangerAutomation>
            </ClipTimeable>
          </MainSequencer>
        </DeviceChain>
      </MidiTrack>"""
        )
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert len(live_set.tracks) == 1
        assert live_set.tracks[0].enabled is True

    def test_track_enabled_true(self, tmp_path: Path) -> None:
        """Track with Speaker/Manual Value='true' is enabled."""
        xml = ALS_TEMPLATE.format(tracks=make_midi_track(1, "Enabled Track", enabled=True))
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert len(live_set.tracks) == 1
        assert live_set.tracks[0].name == "Enabled Track"
        assert live_set.tracks[0].enabled is True

    def test_track_enabled_false(self, tmp_path: Path) -> None:
        """Track with Speaker/Manual Value='false' is disabled."""
        xml = ALS_TEMPLATE.format(tracks=make_midi_track(1, "Disabled Track", enabled=False))
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert len(live_set.tracks) == 1
        assert live_set.tracks[0].name == "Disabled Track"
        assert live_set.tracks[0].enabled is False

    def test_mixed_enabled_disabled_tracks(self, tmp_path: Path) -> None:
        """Mix of enabled and disabled tracks are parsed correctly."""
        tracks = "\n".join(
            [
                make_midi_track(1, "Track1", enabled=True),
                make_midi_track(2, "Track2", enabled=False),
                make_midi_track(3, "Track3", enabled=True),
                make_midi_track(4, "Track4", enabled=False),
            ]
        )
        xml = ALS_TEMPLATE.format(tracks=tracks)
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert len(live_set.tracks) == 4
        enabled_map = {t.name: t.enabled for t in live_set.tracks}
        assert enabled_map["Track1"] is True
        assert enabled_map["Track2"] is False
        assert enabled_map["Track3"] is True
        assert enabled_map["Track4"] is False


class TestGroupParsing:
    """Tests for group track parsing."""

    def test_no_groups_returns_empty_tuple(self, tmp_path: Path) -> None:
        """ALS file without groups returns empty groups tuple."""
        xml = ALS_TEMPLATE.format(tracks=make_midi_track(1, "Solo Track"))
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert live_set.groups == ()

    def test_single_group_parsed(self, tmp_path: Path) -> None:
        """Single group track is parsed correctly."""
        tracks = "\n".join(
            [
                make_group_track(10, "Drums Group", enabled=True),
                make_midi_track(1, "Kick", group_id=10),
            ]
        )
        xml = ALS_TEMPLATE.format(tracks=tracks)
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert len(live_set.groups) == 1
        assert live_set.groups[0].id == 10
        assert live_set.groups[0].name == "Drums Group"
        assert live_set.groups[0].enabled is True
        assert live_set.groups[0].group_id is None

    def test_disabled_group_parsed(self, tmp_path: Path) -> None:
        """Disabled group track is parsed correctly."""
        tracks = "\n".join(
            [
                make_group_track(10, "Disabled Group", enabled=False),
                make_midi_track(1, "Track In Group", group_id=10),
            ]
        )
        xml = ALS_TEMPLATE.format(tracks=tracks)
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert len(live_set.groups) == 1
        assert live_set.groups[0].enabled is False

    def test_multiple_groups_parsed(self, tmp_path: Path) -> None:
        """Multiple group tracks are parsed correctly."""
        tracks = "\n".join(
            [
                make_group_track(10, "Drums", enabled=True),
                make_group_track(20, "Bass", enabled=False),
                make_group_track(30, "Vocals", enabled=True),
            ]
        )
        xml = ALS_TEMPLATE.format(tracks=tracks)
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert len(live_set.groups) == 3
        group_map = {g.name: g for g in live_set.groups}
        assert group_map["Drums"].enabled is True
        assert group_map["Bass"].enabled is False
        assert group_map["Vocals"].enabled is True

    def test_nested_groups_parsed(self, tmp_path: Path) -> None:
        """Nested groups (group within group) are parsed correctly."""
        tracks = "\n".join(
            [
                make_group_track(10, "Parent Group", enabled=True),
                make_group_track(20, "Child Group", enabled=True, parent_group_id=10),
                make_midi_track(1, "Track In Child", group_id=20),
            ]
        )
        xml = ALS_TEMPLATE.format(tracks=tracks)
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert len(live_set.groups) == 2
        parent = next(g for g in live_set.groups if g.name == "Parent Group")
        child = next(g for g in live_set.groups if g.name == "Child Group")

        assert parent.group_id is None
        assert child.group_id == 10


class TestTrackGroupMembership:
    """Tests for track group membership parsing."""

    def test_track_without_group(self, tmp_path: Path) -> None:
        """Track with TrackGroupId=-1 has no group."""
        xml = ALS_TEMPLATE.format(tracks=make_midi_track(1, "Solo Track", group_id=-1))
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert len(live_set.tracks) == 1
        assert live_set.tracks[0].group_id is None

    def test_track_in_group(self, tmp_path: Path) -> None:
        """Track with TrackGroupId referencing a group is in that group."""
        tracks = "\n".join(
            [
                make_group_track(10, "My Group"),
                make_midi_track(1, "Track In Group", group_id=10),
            ]
        )
        xml = ALS_TEMPLATE.format(tracks=tracks)
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        assert len(live_set.tracks) == 1
        assert live_set.tracks[0].group_id == 10


class TestIsGroupEnabled:
    """Tests for LiveSet.is_group_enabled method."""

    def test_none_group_id_returns_true(self) -> None:
        """is_group_enabled(None) returns True."""
        live_set = LiveSet(
            tempo=None,  # type: ignore
            tracks=(),
            groups=(),
        )
        assert live_set.is_group_enabled(None) is True

    def test_enabled_group_returns_true(self) -> None:
        """Enabled group returns True."""
        live_set = LiveSet(
            tempo=None,  # type: ignore
            tracks=(),
            groups=(Group(id=10, name="Group", enabled=True),),
        )
        assert live_set.is_group_enabled(10) is True

    def test_disabled_group_returns_false(self) -> None:
        """Disabled group returns False."""
        live_set = LiveSet(
            tempo=None,  # type: ignore
            tracks=(),
            groups=(Group(id=10, name="Group", enabled=False),),
        )
        assert live_set.is_group_enabled(10) is False

    def test_missing_group_returns_true(self) -> None:
        """Unknown group ID returns True (assume enabled)."""
        live_set = LiveSet(
            tempo=None,  # type: ignore
            tracks=(),
            groups=(),
        )
        assert live_set.is_group_enabled(999) is True

    def test_nested_group_all_enabled(self) -> None:
        """Nested group with all ancestors enabled returns True."""
        live_set = LiveSet(
            tempo=None,  # type: ignore
            tracks=(),
            groups=(
                Group(id=10, name="Parent", enabled=True),
                Group(id=20, name="Child", enabled=True, group_id=10),
            ),
        )
        assert live_set.is_group_enabled(20) is True

    def test_nested_group_parent_disabled(self) -> None:
        """Nested group with disabled parent returns False."""
        live_set = LiveSet(
            tempo=None,  # type: ignore
            tracks=(),
            groups=(
                Group(id=10, name="Parent", enabled=False),
                Group(id=20, name="Child", enabled=True, group_id=10),
            ),
        )
        assert live_set.is_group_enabled(20) is False

    def test_nested_group_child_disabled(self) -> None:
        """Nested group that is disabled returns False."""
        live_set = LiveSet(
            tempo=None,  # type: ignore
            tracks=(),
            groups=(
                Group(id=10, name="Parent", enabled=True),
                Group(id=20, name="Child", enabled=False, group_id=10),
            ),
        )
        assert live_set.is_group_enabled(20) is False

    def test_deeply_nested_groups(self) -> None:
        """Deeply nested groups (3+ levels) work correctly."""
        live_set = LiveSet(
            tempo=None,  # type: ignore
            tracks=(),
            groups=(
                Group(id=10, name="Level1", enabled=True),
                Group(id=20, name="Level2", enabled=True, group_id=10),
                Group(id=30, name="Level3", enabled=True, group_id=20),
            ),
        )
        assert live_set.is_group_enabled(30) is True

        # Disable middle level
        live_set_disabled = LiveSet(
            tempo=None,  # type: ignore
            tracks=(),
            groups=(
                Group(id=10, name="Level1", enabled=True),
                Group(id=20, name="Level2", enabled=False, group_id=10),
                Group(id=30, name="Level3", enabled=True, group_id=20),
            ),
        )
        assert live_set_disabled.is_group_enabled(30) is False


class TestIsTrackEffectivelyEnabled:
    """Tests for LiveSet.is_track_effectively_enabled method."""

    def test_enabled_track_no_group(self) -> None:
        """Enabled track without group is effectively enabled."""
        track = Track(name="Track", track_type="midi", clips=(), enabled=True, group_id=None)
        live_set = LiveSet(tempo=None, tracks=(track,), groups=())  # type: ignore

        assert live_set.is_track_effectively_enabled(track) is True

    def test_disabled_track_no_group(self) -> None:
        """Disabled track without group is not effectively enabled."""
        track = Track(name="Track", track_type="midi", clips=(), enabled=False, group_id=None)
        live_set = LiveSet(tempo=None, tracks=(track,), groups=())  # type: ignore

        assert live_set.is_track_effectively_enabled(track) is False

    def test_enabled_track_in_enabled_group(self) -> None:
        """Enabled track in enabled group is effectively enabled."""
        track = Track(name="Track", track_type="midi", clips=(), enabled=True, group_id=10)
        group = Group(id=10, name="Group", enabled=True)
        live_set = LiveSet(tempo=None, tracks=(track,), groups=(group,))  # type: ignore

        assert live_set.is_track_effectively_enabled(track) is True

    def test_enabled_track_in_disabled_group(self) -> None:
        """Enabled track in disabled group is not effectively enabled."""
        track = Track(name="Track", track_type="midi", clips=(), enabled=True, group_id=10)
        group = Group(id=10, name="Group", enabled=False)
        live_set = LiveSet(tempo=None, tracks=(track,), groups=(group,))  # type: ignore

        assert live_set.is_track_effectively_enabled(track) is False

    def test_disabled_track_in_enabled_group(self) -> None:
        """Disabled track in enabled group is not effectively enabled."""
        track = Track(name="Track", track_type="midi", clips=(), enabled=False, group_id=10)
        group = Group(id=10, name="Group", enabled=True)
        live_set = LiveSet(tempo=None, tracks=(track,), groups=(group,))  # type: ignore

        assert live_set.is_track_effectively_enabled(track) is False

    def test_track_in_nested_disabled_group(self) -> None:
        """Track in nested group with disabled ancestor is not effectively enabled."""
        track = Track(name="Track", track_type="midi", clips=(), enabled=True, group_id=20)
        parent = Group(id=10, name="Parent", enabled=False)
        child = Group(id=20, name="Child", enabled=True, group_id=10)
        live_set = LiveSet(tempo=None, tracks=(track,), groups=(parent, child))  # type: ignore

        assert live_set.is_track_effectively_enabled(track) is False


class TestEnabledTracks:
    """Tests for LiveSet.enabled_tracks method."""

    def test_all_enabled_returns_all(self) -> None:
        """All enabled tracks are returned."""
        tracks = (
            Track(name="Track1", track_type="midi", clips=(), enabled=True),
            Track(name="Track2", track_type="midi", clips=(), enabled=True),
        )
        live_set = LiveSet(tempo=None, tracks=tracks, groups=())  # type: ignore

        result = live_set.enabled_tracks()

        assert len(result) == 2

    def test_filters_disabled_tracks(self) -> None:
        """Disabled tracks are filtered out."""
        tracks = (
            Track(name="Enabled", track_type="midi", clips=(), enabled=True),
            Track(name="Disabled", track_type="midi", clips=(), enabled=False),
        )
        live_set = LiveSet(tempo=None, tracks=tracks, groups=())  # type: ignore

        result = live_set.enabled_tracks()

        assert len(result) == 1
        assert result[0].name == "Enabled"

    def test_filters_tracks_in_disabled_groups(self) -> None:
        """Tracks in disabled groups are filtered out."""
        tracks = (
            Track(name="Ungrouped", track_type="midi", clips=(), enabled=True),
            Track(name="InEnabledGroup", track_type="midi", clips=(), enabled=True, group_id=10),
            Track(name="InDisabledGroup", track_type="midi", clips=(), enabled=True, group_id=20),
        )
        groups = (
            Group(id=10, name="EnabledGroup", enabled=True),
            Group(id=20, name="DisabledGroup", enabled=False),
        )
        live_set = LiveSet(tempo=None, tracks=tracks, groups=groups)  # type: ignore

        result = live_set.enabled_tracks()

        assert len(result) == 2
        names = {t.name for t in result}
        assert names == {"Ungrouped", "InEnabledGroup"}


class TestIntegrationGroupAndEnabled:
    """Integration tests combining group parsing and enabled state detection."""

    def test_realistic_drums_scenario(self, tmp_path: Path) -> None:
        """Realistic scenario: MIDI drums disabled, live drums enabled."""
        tracks = "\n".join(
            [
                # Live Drums group (enabled)
                make_group_track(66, "Live Drums", enabled=True),
                make_midi_track(1, "Kick Audio", group_id=66),
                make_midi_track(2, "Snare Audio", group_id=66),
                # MIDI Drums group (disabled)
                make_group_track(58, "MIDI Drums", enabled=False),
                make_midi_track(3, "Kick MIDI", group_id=58),
                make_midi_track(4, "Snare MIDI", group_id=58),
                # Solo track (not in group)
                make_midi_track(5, "Bass"),
            ]
        )
        xml = ALS_TEMPLATE.format(tracks=tracks)
        als_file = create_als_file(tmp_path, xml)

        live_set = parse_als_file(als_file)

        # Check groups
        assert len(live_set.groups) == 2
        live_drums = next(g for g in live_set.groups if g.name == "Live Drums")
        midi_drums = next(g for g in live_set.groups if g.name == "MIDI Drums")
        assert live_drums.enabled is True
        assert midi_drums.enabled is False

        # Check enabled tracks
        enabled = live_set.enabled_tracks()
        enabled_names = {t.name for t in enabled}

        assert "Kick Audio" in enabled_names
        assert "Snare Audio" in enabled_names
        assert "Bass" in enabled_names
        assert "Kick MIDI" not in enabled_names
        assert "Snare MIDI" not in enabled_names
