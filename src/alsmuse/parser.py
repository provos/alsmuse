"""ALS file parser for ALSmuse.

This module provides functionality to parse Ableton Live Set (.als) files
and extract structured data into domain models.

ALS files are gzip-compressed XML documents containing project data
including tracks, clips, tempo, and routing information.
"""

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ParseError as XMLParseError

from .exceptions import ParseError
from .models import Clip, Group, LiveSet, Tempo, Track


def parse_als_xml(path: Path) -> Element:
    """Parse an Ableton Live Set file and return the XML root element.

    This provides access to the raw XML for advanced analysis like
    MIDI note extraction.

    Args:
        path: Path to the .als file to parse.

    Returns:
        The root XML Element.

    Raises:
        ParseError: If the file cannot be read, decompressed, or parsed.
    """
    xml_bytes = _decompress_als(path)

    try:
        root = ET.fromstring(xml_bytes)
    except XMLParseError as e:
        raise ParseError(f"Invalid XML in ALS file: {e}") from e

    return root


def get_track_elements(root: Element) -> list[tuple[Element, str]]:
    """Get all track elements with their types from a parsed ALS XML root.

    Args:
        root: The root XML element from parse_als_xml.

    Returns:
        List of (track_element, track_type) tuples where track_type is
        "midi" or "audio".

    Raises:
        ParseError: If the Tracks element is missing.
    """
    live_set_elem = root.find("LiveSet")
    if live_set_elem is None:
        raise ParseError("Missing LiveSet element in ALS file")

    tracks_elem = live_set_elem.find("Tracks")
    if tracks_elem is None:
        raise ParseError("Missing Tracks element in ALS file")

    result: list[tuple[Element, str]] = []

    for midi_track in tracks_elem.findall("MidiTrack"):
        result.append((midi_track, "midi"))

    for audio_track in tracks_elem.findall("AudioTrack"):
        result.append((audio_track, "audio"))

    return result


def extract_track_name(track_element: Element) -> str:
    """Extract track name from a track element.

    Public wrapper around _extract_track_name for use in MIDI extraction.

    Args:
        track_element: A MidiTrack or AudioTrack XML element.

    Returns:
        The track name.

    Raises:
        ParseError: If neither UserName nor EffectiveName can be found.
    """
    return _extract_track_name(track_element)


def extract_track_clips(
    track_element: Element,
    track_type: str,
) -> tuple[Clip, ...]:
    """Extract clips from a track element.

    Public wrapper around _extract_clips for use in MIDI extraction.

    Args:
        track_element: A MidiTrack or AudioTrack XML element.
        track_type: Either "midi" or "audio".

    Returns:
        Tuple of Clip objects.
    """
    return _extract_clips(track_element, "midi" if track_type == "midi" else "audio")


def extract_track_enabled(track_element: Element) -> bool:
    """Extract whether a track is enabled (not muted).

    Public wrapper around _extract_track_enabled.

    Args:
        track_element: A MidiTrack, AudioTrack, or GroupTrack XML element.

    Returns:
        True if the track is enabled, False if muted.
    """
    return _extract_track_enabled(track_element)


def extract_track_group_id(track_element: Element) -> int | None:
    """Extract the group ID that a track belongs to.

    Public wrapper around _extract_track_group_id.

    Args:
        track_element: A MidiTrack, AudioTrack, or GroupTrack XML element.

    Returns:
        The group ID, or None if the track is not in a group.
    """
    return _extract_track_group_id(track_element)


def parse_als_file(path: Path) -> LiveSet:
    """Parse an Ableton Live Set file into a LiveSet model.

    Args:
        path: Path to the .als file to parse.

    Returns:
        A LiveSet containing the extracted tempo and tracks.

    Raises:
        ParseError: If the file cannot be read, decompressed, or parsed.
    """
    xml_bytes = _decompress_als(path)

    try:
        root = ET.fromstring(xml_bytes)
    except XMLParseError as e:
        raise ParseError(f"Invalid XML in ALS file: {e}") from e

    live_set_elem = root.find("LiveSet")
    if live_set_elem is None:
        raise ParseError("Missing LiveSet element in ALS file")

    tempo = _extract_tempo(live_set_elem)
    tracks = _extract_tracks(live_set_elem)
    groups = _extract_groups(live_set_elem)

    return LiveSet(tempo=tempo, tracks=tracks, groups=groups)


def _decompress_als(path: Path) -> bytes:
    """Decompress a gzipped ALS file.

    Args:
        path: Path to the .als file.

    Returns:
        The decompressed XML content as bytes.

    Raises:
        ParseError: If the file cannot be read or decompressed.
    """
    try:
        with gzip.open(path, "rb") as f:
            return f.read()
    except FileNotFoundError as e:
        raise ParseError(f"ALS file not found: {path}") from e
    except gzip.BadGzipFile as e:
        raise ParseError(f"ALS file is not valid gzip: {path}") from e
    except OSError as e:
        raise ParseError(f"Error reading ALS file: {e}") from e


def _extract_tempo(root: Element) -> Tempo:
    """Extract tempo and time signature from the LiveSet element.

    The tempo is located at MainTrack/DeviceChain/Mixer/Tempo/Manual[@Value].
    The time signature is extracted from the first clip's TimeSignature element,
    defaulting to 4/4 if not found.

    Args:
        root: The LiveSet XML element.

    Returns:
        A Tempo object with BPM and time signature.

    Raises:
        ParseError: If tempo information cannot be extracted.
    """
    # Extract BPM from MainTrack/DeviceChain/Mixer/Tempo/Manual
    tempo_elem = root.find("MainTrack/DeviceChain/Mixer/Tempo/Manual")
    if tempo_elem is None:
        raise ParseError("Could not find Tempo element in ALS file")

    bpm_str = tempo_elem.get("Value")
    if bpm_str is None:
        raise ParseError("Tempo element missing Value attribute")

    try:
        bpm = float(bpm_str)
    except ValueError as e:
        raise ParseError(f"Invalid tempo value: {bpm_str}") from e

    # Extract time signature from first clip's TimeSignature element
    # Default to 4/4 if not found
    time_sig_elem = root.find(".//TimeSignatures/RemoteableTimeSignature")
    if time_sig_elem is not None:
        numerator_elem = time_sig_elem.find("Numerator")
        denominator_elem = time_sig_elem.find("Denominator")
        if numerator_elem is not None and denominator_elem is not None:
            try:
                numerator = int(numerator_elem.get("Value", "4"))
                denominator = int(denominator_elem.get("Value", "4"))
            except ValueError:
                numerator, denominator = 4, 4
        else:
            numerator, denominator = 4, 4
    else:
        numerator, denominator = 4, 4

    return Tempo(bpm=bpm, time_signature=(numerator, denominator))


def _extract_track_enabled(track_element: Element) -> bool:
    """Extract whether a track is enabled (not muted).

    The enabled state is determined by the Speaker/Manual value in the Mixer.
    When false, the track output is muted.

    Args:
        track_element: A MidiTrack, AudioTrack, or GroupTrack XML element.

    Returns:
        True if the track is enabled, False if muted.
    """
    speaker_elem = track_element.find("DeviceChain/Mixer/Speaker/Manual")
    if speaker_elem is None:
        return True  # Default to enabled if not found
    return speaker_elem.get("Value", "true").lower() == "true"


def _extract_track_group_id(track_element: Element) -> int | None:
    """Extract the group ID that a track belongs to.

    Args:
        track_element: A MidiTrack, AudioTrack, or GroupTrack XML element.

    Returns:
        The group ID, or None if the track is not in a group (TrackGroupId is -1).
    """
    group_id_elem = track_element.find("TrackGroupId")
    if group_id_elem is None:
        return None
    group_id_str = group_id_elem.get("Value", "-1")
    try:
        group_id = int(group_id_str)
        return None if group_id == -1 else group_id
    except ValueError:
        return None


def _extract_groups(root: Element) -> tuple[Group, ...]:
    """Extract all group tracks from the LiveSet element.

    Args:
        root: The LiveSet XML element.

    Returns:
        A tuple of Group objects.
    """
    tracks_elem = root.find("Tracks")
    if tracks_elem is None:
        return ()

    groups: list[Group] = []

    for group_track in tracks_elem.findall("GroupTrack"):
        group_id_str = group_track.get("Id")
        if group_id_str is None:
            continue
        try:
            group_id = int(group_id_str)
        except ValueError:
            continue

        name = _extract_track_name(group_track)
        enabled = _extract_track_enabled(group_track)
        parent_group_id = _extract_track_group_id(group_track)

        groups.append(
            Group(
                id=group_id,
                name=name,
                enabled=enabled,
                group_id=parent_group_id,
            )
        )

    return tuple(groups)


def _extract_tracks(root: Element) -> tuple[Track, ...]:
    """Extract all MIDI and Audio tracks from the LiveSet element.

    Args:
        root: The LiveSet XML element.

    Returns:
        A tuple of Track objects.

    Raises:
        ParseError: If the Tracks element is missing.
    """
    tracks_elem = root.find("Tracks")
    if tracks_elem is None:
        raise ParseError("Missing Tracks element in ALS file")

    tracks: list[Track] = []

    # Extract MIDI tracks
    for midi_track in tracks_elem.findall("MidiTrack"):
        name = _extract_track_name(midi_track)
        clips = _extract_clips(midi_track, "midi")
        enabled = _extract_track_enabled(midi_track)
        group_id = _extract_track_group_id(midi_track)
        tracks.append(
            Track(
                name=name,
                track_type="midi",
                clips=clips,
                enabled=enabled,
                group_id=group_id,
            )
        )

    # Extract Audio tracks
    for audio_track in tracks_elem.findall("AudioTrack"):
        name = _extract_track_name(audio_track)
        clips = _extract_clips(audio_track, "audio")
        enabled = _extract_track_enabled(audio_track)
        group_id = _extract_track_group_id(audio_track)
        tracks.append(
            Track(
                name=name,
                track_type="audio",
                clips=clips,
                enabled=enabled,
                group_id=group_id,
            )
        )

    return tuple(tracks)


def _extract_track_name(track_element: Element) -> str:
    """Extract track name with fallback logic.

    Priority:
    1. UserName if non-empty
    2. EffectiveName as fallback

    This handles cases where UserName is "" but EffectiveName
    contains the auto-generated or sample-based name.

    Args:
        track_element: The MidiTrack or AudioTrack XML element.

    Returns:
        The track name.

    Raises:
        ParseError: If neither UserName nor EffectiveName can be found.
    """
    name_elem = track_element.find("Name")
    if name_elem is None:
        raise ParseError("Track element missing Name child")

    user_name_elem = name_elem.find("UserName")
    if user_name_elem is not None:
        user_name = user_name_elem.get("Value", "")
        if user_name:  # Non-empty string
            return user_name

    # Fall back to EffectiveName
    effective_name_elem = name_elem.find("EffectiveName")
    if effective_name_elem is not None:
        effective_name = effective_name_elem.get("Value", "")
        if effective_name:
            return effective_name

    raise ParseError("Track has neither UserName nor EffectiveName")


def _extract_clips(
    track_element: Element, track_type: Literal["midi", "audio"]
) -> tuple[Clip, ...]:
    """Extract clips from a track element.

    For MIDI tracks: finds MidiClip elements in
        DeviceChain/MainSequencer/ClipTimeable/ArrangerAutomation/Events
    For Audio tracks: finds AudioClip elements similarly

    Args:
        track_element: The MidiTrack or AudioTrack XML element.
        track_type: Either "midi" or "audio" to determine clip element name.

    Returns:
        A tuple of Clip objects.
    """
    clip_tag = "MidiClip" if track_type == "midi" else "AudioClip"

    # Path to arrangement clips
    events_path = "DeviceChain/MainSequencer/ClipTimeable/ArrangerAutomation/Events"
    events_elem = track_element.find(events_path)

    clips: list[Clip] = []

    if events_elem is not None:
        for clip_elem in events_elem.findall(clip_tag):
            clip = _parse_clip_element(clip_elem)
            if clip is not None:
                clips.append(clip)

    return tuple(clips)


def _parse_clip_element(clip_elem: Element) -> Clip | None:
    """Parse a single clip element into a Clip object.

    Args:
        clip_elem: A MidiClip or AudioClip XML element.

    Returns:
        A Clip object, or None if required attributes are missing.
    """
    # Get start position from Time attribute
    time_str = clip_elem.get("Time")
    if time_str is None:
        return None

    try:
        start_beats = float(time_str)
    except ValueError:
        return None

    # Get end position from CurrentEnd element
    current_end_elem = clip_elem.find("CurrentEnd")
    if current_end_elem is None:
        return None

    end_str = current_end_elem.get("Value")
    if end_str is None:
        return None

    try:
        end_beats = float(end_str)
    except ValueError:
        return None

    # Get clip name
    name_elem = clip_elem.find("Name")
    name = name_elem.get("Value", "") if name_elem is not None else ""

    return Clip(name=name, start_beats=start_beats, end_beats=end_beats)
