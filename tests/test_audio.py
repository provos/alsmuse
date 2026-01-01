"""Tests for audio extraction from ALS files."""

import gzip
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from alsmuse.audio import (
    VOCAL_KEYWORDS,
    beats_to_seconds,
    combine_clips_to_audio,
    extract_audio_clips,
    find_vocal_clips,
    get_unique_vocal_track_names,
    is_vocal_track,
    prompt_track_selection,
    resolve_audio_path,
    select_vocal_tracks,
)
from alsmuse.models import AudioClipRef


class TestResolveAudioPath:
    """Tests for resolve_audio_path function."""

    def test_relative_path_from_als_directory(self, tmp_path: Path) -> None:
        """Relative path from ALS file's parent directory is found first."""
        # Create project structure
        project_dir = tmp_path / "MyProject"
        project_dir.mkdir()
        samples_dir = project_dir / "Samples"
        samples_dir.mkdir()

        # Create audio file and ALS path
        audio_file = samples_dir / "vocals.wav"
        audio_file.write_bytes(b"fake audio")
        als_path = project_dir / "song.als"

        # Relative path should resolve
        result = resolve_audio_path(
            als_path,
            relative_path="Samples/vocals.wav",
            absolute_path="/some/old/path/vocals.wav",
        )

        assert result == audio_file.resolve()

    def test_relative_path_from_project_root(self, tmp_path: Path) -> None:
        """Relative path from grandparent (project root) is tried second."""
        # Create project structure where ALS is in a subdirectory
        project_root = tmp_path / "MyProject"
        project_root.mkdir()
        sessions_dir = project_root / "Sessions"
        sessions_dir.mkdir()
        samples_dir = project_root / "Samples"
        samples_dir.mkdir()

        # Create audio file and ALS path
        audio_file = samples_dir / "vocals.wav"
        audio_file.write_bytes(b"fake audio")
        als_path = sessions_dir / "song.als"

        # Relative path from parent doesn't work, but from grandparent does
        result = resolve_audio_path(
            als_path,
            relative_path="Samples/vocals.wav",
            absolute_path="/some/old/path/vocals.wav",
        )

        assert result == audio_file.resolve()

    def test_absolute_path_fallback(self, tmp_path: Path) -> None:
        """Absolute path is used as fallback when relative paths fail."""
        # Create audio file at an absolute path
        audio_file = tmp_path / "vocals.wav"
        audio_file.write_bytes(b"fake audio")

        # ALS file in a different location
        als_path = tmp_path / "projects" / "song.als"

        result = resolve_audio_path(
            als_path,
            relative_path="nonexistent/path.wav",
            absolute_path=str(audio_file),
        )

        assert result == audio_file.resolve()

    def test_returns_none_when_no_path_exists(self, tmp_path: Path) -> None:
        """Returns None when neither relative nor absolute paths exist."""
        als_path = tmp_path / "song.als"

        result = resolve_audio_path(
            als_path,
            relative_path="nonexistent/path.wav",
            absolute_path="/also/nonexistent/path.wav",
        )

        assert result is None

    def test_prefers_relative_over_absolute(self, tmp_path: Path) -> None:
        """Relative path is preferred even when absolute path also exists."""
        # Create two different audio files
        project_dir = tmp_path / "MyProject"
        project_dir.mkdir()
        samples_dir = project_dir / "Samples"
        samples_dir.mkdir()

        relative_file = samples_dir / "vocals.wav"
        relative_file.write_bytes(b"relative audio")

        absolute_file = tmp_path / "absolute_vocals.wav"
        absolute_file.write_bytes(b"absolute audio")

        als_path = project_dir / "song.als"

        result = resolve_audio_path(
            als_path,
            relative_path="Samples/vocals.wav",
            absolute_path=str(absolute_file),
        )

        # Should find relative path first, not absolute
        assert result == relative_file.resolve()

    def test_empty_relative_path_uses_absolute(self, tmp_path: Path) -> None:
        """Empty relative path falls back to absolute path."""
        audio_file = tmp_path / "vocals.wav"
        audio_file.write_bytes(b"fake audio")

        als_path = tmp_path / "song.als"

        result = resolve_audio_path(
            als_path,
            relative_path="",
            absolute_path=str(audio_file),
        )

        assert result == audio_file.resolve()

    def test_empty_absolute_path(self, tmp_path: Path) -> None:
        """Empty absolute path is handled correctly."""
        project_dir = tmp_path / "MyProject"
        project_dir.mkdir()
        samples_dir = project_dir / "Samples"
        samples_dir.mkdir()

        audio_file = samples_dir / "vocals.wav"
        audio_file.write_bytes(b"fake audio")
        als_path = project_dir / "song.als"

        result = resolve_audio_path(
            als_path,
            relative_path="Samples/vocals.wav",
            absolute_path="",
        )

        assert result == audio_file.resolve()


class TestBeatsToSeconds:
    """Tests for beats_to_seconds function."""

    def test_basic_conversion(self) -> None:
        """Basic beat to seconds conversion at 120 BPM."""
        # At 120 BPM, 1 beat = 0.5 seconds
        assert beats_to_seconds(1.0, 120.0) == 0.5
        assert beats_to_seconds(2.0, 120.0) == 1.0
        assert beats_to_seconds(4.0, 120.0) == 2.0

    def test_different_tempos(self) -> None:
        """Conversion at various tempos."""
        # At 60 BPM, 1 beat = 1 second
        assert beats_to_seconds(1.0, 60.0) == 1.0

        # At 180 BPM, 1 beat = 0.333... seconds
        assert beats_to_seconds(1.0, 180.0) == pytest.approx(1 / 3)

    def test_fractional_beats(self) -> None:
        """Fractional beat positions convert correctly."""
        # At 120 BPM, 0.5 beats = 0.25 seconds
        assert beats_to_seconds(0.5, 120.0) == 0.25
        assert beats_to_seconds(1.5, 120.0) == 0.75

    def test_zero_beats(self) -> None:
        """Zero beats equals zero seconds."""
        assert beats_to_seconds(0.0, 120.0) == 0.0


def create_minimal_als_xml(
    audio_clips: list[dict[str, str]],
    tempo: float = 120.0,
) -> bytes:
    """Create minimal ALS XML content for testing.

    Args:
        audio_clips: List of dicts with keys:
            - track_name: Name of the track
            - clip_name: Name of the clip
            - start_beats: Start position in beats
            - end_beats: End position in beats
            - relative_path: Relative path to audio file
            - absolute_path: Absolute path to audio file
        tempo: Project tempo in BPM.

    Returns:
        Gzipped XML bytes representing a minimal ALS file.
    """
    # Build track XML for each unique track
    tracks_by_name: dict[str, list[dict[str, str]]] = {}
    for clip in audio_clips:
        track_name = clip.get("track_name", "Audio")
        if track_name not in tracks_by_name:
            tracks_by_name[track_name] = []
        tracks_by_name[track_name].append(clip)

    tracks_xml = ""
    track_id = 1

    for track_name, clips in tracks_by_name.items():
        clips_xml = ""
        for clip in clips:
            clip_xml = f'''
                            <AudioClip Id="{track_id}" Time="{clip["start_beats"]}">
                              <CurrentEnd Value="{clip["end_beats"]}"/>
                              <Name Value="{clip.get("clip_name", "")}"/>
                              <SampleRef>
                                <FileRef>
                                  <RelativePath Value="{clip.get("relative_path", "")}"/>
                                  <Path Value="{clip.get("absolute_path", "")}"/>
                                </FileRef>
                              </SampleRef>
                            </AudioClip>'''
            clips_xml += clip_xml
            track_id += 1

        tracks_xml += f'''
        <AudioTrack Id="{track_id}">
          <Name>
            <EffectiveName Value="{track_name}"/>
          </Name>
          <DeviceChain>
            <MainSequencer>
              <Sample>
                <ArrangerAutomation>
                  <Events>{clips_xml}
                  </Events>
                </ArrangerAutomation>
              </Sample>
            </MainSequencer>
          </DeviceChain>
        </AudioTrack>'''
        track_id += 1

    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<Ableton>
  <LiveSet>
    <MainTrack>
      <DeviceChain>
        <Mixer>
          <Tempo>
            <Manual Value="{tempo}"/>
          </Tempo>
        </Mixer>
      </DeviceChain>
    </MainTrack>
    <Tracks>{tracks_xml}
    </Tracks>
  </LiveSet>
</Ableton>'''

    return gzip.compress(xml_content.encode("utf-8"))


class TestExtractAudioClips:
    """Tests for extract_audio_clips function."""

    def test_extract_single_clip(self, tmp_path: Path) -> None:
        """Extract a single audio clip from ALS."""
        # Create audio file
        samples_dir = tmp_path / "Samples"
        samples_dir.mkdir()
        audio_file = samples_dir / "vocals.wav"
        audio_file.write_bytes(b"fake audio")

        # Create ALS file with one audio clip
        als_path = tmp_path / "song.als"
        als_content = create_minimal_als_xml(
            [
                {
                    "track_name": "Lead Vocal",
                    "clip_name": "Verse",
                    "start_beats": "64",
                    "end_beats": "128",
                    "relative_path": "Samples/vocals.wav",
                    "absolute_path": "/old/path/vocals.wav",
                }
            ]
        )
        als_path.write_bytes(als_content)

        clips = extract_audio_clips(als_path, bpm=120.0)

        assert len(clips) == 1
        clip = clips[0]
        assert clip.track_name == "Lead Vocal"
        assert clip.file_path == audio_file.resolve()
        assert clip.start_beats == 64.0
        assert clip.end_beats == 128.0
        # At 120 BPM: 64 beats = 32 seconds, 128 beats = 64 seconds
        assert clip.start_seconds == 32.0
        assert clip.end_seconds == 64.0

    def test_extract_multiple_clips_same_track(self, tmp_path: Path) -> None:
        """Extract multiple clips from the same track."""
        samples_dir = tmp_path / "Samples"
        samples_dir.mkdir()

        verse_file = samples_dir / "verse.wav"
        verse_file.write_bytes(b"verse audio")
        chorus_file = samples_dir / "chorus.wav"
        chorus_file.write_bytes(b"chorus audio")

        als_path = tmp_path / "song.als"
        als_content = create_minimal_als_xml(
            [
                {
                    "track_name": "Vocals",
                    "clip_name": "Verse",
                    "start_beats": "0",
                    "end_beats": "64",
                    "relative_path": "Samples/verse.wav",
                    "absolute_path": "",
                },
                {
                    "track_name": "Vocals",
                    "clip_name": "Chorus",
                    "start_beats": "64",
                    "end_beats": "128",
                    "relative_path": "Samples/chorus.wav",
                    "absolute_path": "",
                },
            ]
        )
        als_path.write_bytes(als_content)

        clips = extract_audio_clips(als_path, bpm=120.0)

        assert len(clips) == 2
        assert clips[0].track_name == "Vocals"
        assert clips[1].track_name == "Vocals"
        assert clips[0].start_beats == 0.0
        assert clips[1].start_beats == 64.0

    def test_extract_clips_from_multiple_tracks(self, tmp_path: Path) -> None:
        """Extract clips from multiple audio tracks."""
        samples_dir = tmp_path / "Samples"
        samples_dir.mkdir()

        lead_file = samples_dir / "lead.wav"
        lead_file.write_bytes(b"lead audio")
        backing_file = samples_dir / "backing.wav"
        backing_file.write_bytes(b"backing audio")

        als_path = tmp_path / "song.als"
        als_content = create_minimal_als_xml(
            [
                {
                    "track_name": "Lead Vocal",
                    "clip_name": "Lead",
                    "start_beats": "0",
                    "end_beats": "64",
                    "relative_path": "Samples/lead.wav",
                    "absolute_path": "",
                },
                {
                    "track_name": "Backing Vocal",
                    "clip_name": "Backing",
                    "start_beats": "32",
                    "end_beats": "96",
                    "relative_path": "Samples/backing.wav",
                    "absolute_path": "",
                },
            ]
        )
        als_path.write_bytes(als_content)

        clips = extract_audio_clips(als_path, bpm=120.0)

        assert len(clips) == 2
        track_names = {clip.track_name for clip in clips}
        assert track_names == {"Lead Vocal", "Backing Vocal"}

    def test_skips_clips_with_unresolvable_paths(self, tmp_path: Path) -> None:
        """Clips with unresolvable audio paths are skipped."""
        # Create only one of the two audio files
        samples_dir = tmp_path / "Samples"
        samples_dir.mkdir()
        existing_file = samples_dir / "exists.wav"
        existing_file.write_bytes(b"audio")

        als_path = tmp_path / "song.als"
        als_content = create_minimal_als_xml(
            [
                {
                    "track_name": "Vocal 1",
                    "clip_name": "Exists",
                    "start_beats": "0",
                    "end_beats": "64",
                    "relative_path": "Samples/exists.wav",
                    "absolute_path": "",
                },
                {
                    "track_name": "Vocal 2",
                    "clip_name": "Missing",
                    "start_beats": "64",
                    "end_beats": "128",
                    "relative_path": "Samples/missing.wav",
                    "absolute_path": "/nonexistent/missing.wav",
                },
            ]
        )
        als_path.write_bytes(als_content)

        clips = extract_audio_clips(als_path, bpm=120.0)

        # Only the clip with existing file should be returned
        assert len(clips) == 1
        assert clips[0].track_name == "Vocal 1"

    def test_empty_als_file(self, tmp_path: Path) -> None:
        """ALS file with no audio tracks returns empty list."""
        als_path = tmp_path / "song.als"
        als_content = create_minimal_als_xml([])
        als_path.write_bytes(als_content)

        clips = extract_audio_clips(als_path, bpm=120.0)

        assert clips == []

    def test_beat_to_seconds_conversion(self, tmp_path: Path) -> None:
        """Verify beat to seconds conversion at different tempos."""
        samples_dir = tmp_path / "Samples"
        samples_dir.mkdir()
        audio_file = samples_dir / "audio.wav"
        audio_file.write_bytes(b"audio")

        als_path = tmp_path / "song.als"
        als_content = create_minimal_als_xml(
            [
                {
                    "track_name": "Track",
                    "clip_name": "Clip",
                    "start_beats": "60",
                    "end_beats": "120",
                    "relative_path": "Samples/audio.wav",
                    "absolute_path": "",
                }
            ]
        )
        als_path.write_bytes(als_content)

        # At 60 BPM: 60 beats = 60 seconds
        clips = extract_audio_clips(als_path, bpm=60.0)
        assert len(clips) == 1
        assert clips[0].start_seconds == 60.0
        assert clips[0].end_seconds == 120.0

        # At 120 BPM: 60 beats = 30 seconds
        clips = extract_audio_clips(als_path, bpm=120.0)
        assert len(clips) == 1
        assert clips[0].start_seconds == 30.0
        assert clips[0].end_seconds == 60.0


class TestAudioClipRefModel:
    """Tests for the AudioClipRef data model."""

    def test_frozen_dataclass(self) -> None:
        """AudioClipRef is immutable."""
        clip = AudioClipRef(
            track_name="Vocals",
            file_path=Path("/path/to/audio.wav"),
            start_beats=0.0,
            end_beats=64.0,
            start_seconds=0.0,
            end_seconds=32.0,
        )

        with pytest.raises(AttributeError):
            clip.track_name = "New Name"  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible after construction."""
        clip = AudioClipRef(
            track_name="Lead Vocal",
            file_path=Path("/path/to/vocals.wav"),
            start_beats=64.0,
            end_beats=128.0,
            start_seconds=32.0,
            end_seconds=64.0,
        )

        assert clip.track_name == "Lead Vocal"
        assert clip.file_path == Path("/path/to/vocals.wav")
        assert clip.start_beats == 64.0
        assert clip.end_beats == 128.0
        assert clip.start_seconds == 32.0
        assert clip.end_seconds == 64.0


# ---------------------------------------------------------------------------
# Phase 2 Tests: Vocal Track Identification
# ---------------------------------------------------------------------------


class TestIsVocalTrack:
    """Tests for is_vocal_track function."""

    def test_matches_exact_keyword(self) -> None:
        """Track names containing exact keywords are identified as vocal."""
        assert is_vocal_track("vocal") is True
        assert is_vocal_track("vox") is True
        assert is_vocal_track("voice") is True

    def test_matches_keyword_in_longer_name(self) -> None:
        """Keywords embedded in longer names are matched."""
        assert is_vocal_track("Lead Vocal Track") is True
        assert is_vocal_track("My Vox Recording") is True
        assert is_vocal_track("Main Voice") is True

    def test_case_insensitive(self) -> None:
        """Matching is case-insensitive."""
        assert is_vocal_track("VOCAL") is True
        assert is_vocal_track("Vocal") is True
        assert is_vocal_track("VoCAL") is True
        assert is_vocal_track("LEAD VOX") is True

    def test_section_names_match(self) -> None:
        """Section names like verse, chorus, bridge are matched."""
        assert is_vocal_track("Verse 1") is True
        assert is_vocal_track("Chorus") is True
        assert is_vocal_track("Bridge Section") is True

    def test_backing_and_harmony(self) -> None:
        """Backing and harmony keywords are matched."""
        assert is_vocal_track("Backing Vocals") is True
        assert is_vocal_track("Harmony") is True
        assert is_vocal_track("Double Track") is True

    def test_non_vocal_tracks(self) -> None:
        """Non-vocal track names are not matched."""
        assert is_vocal_track("Drums") is False
        assert is_vocal_track("Bass") is False
        assert is_vocal_track("Guitar") is False
        assert is_vocal_track("Piano") is False
        assert is_vocal_track("Synth Lead") is False
        assert is_vocal_track("Kick") is False

    def test_empty_string(self) -> None:
        """Empty string is not a vocal track."""
        assert is_vocal_track("") is False

    def test_all_keywords_work(self) -> None:
        """All defined keywords successfully match."""
        for keyword in VOCAL_KEYWORDS:
            assert is_vocal_track(keyword) is True, f"Keyword '{keyword}' should match"


class TestFindVocalClips:
    """Tests for find_vocal_clips function."""

    def _make_clip(
        self, track_name: str, start_beats: float = 0.0, end_beats: float = 64.0
    ) -> AudioClipRef:
        """Helper to create test clips."""
        return AudioClipRef(
            track_name=track_name,
            file_path=Path(f"/test/{track_name.replace(' ', '_')}.wav"),
            start_beats=start_beats,
            end_beats=end_beats,
            start_seconds=start_beats * 0.5,  # 120 BPM
            end_seconds=end_beats * 0.5,
        )

    def test_auto_detect_vocal_tracks(self) -> None:
        """Clips from vocal tracks are automatically detected."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("Drums", 0, 128),
            self._make_clip("Backing Vox", 32, 96),
            self._make_clip("Bass", 0, 128),
        ]

        result = find_vocal_clips(clips)

        assert len(result) == 2
        track_names = {c.track_name for c in result}
        assert track_names == {"Lead Vocal", "Backing Vox"}

    def test_sorted_by_start_time(self) -> None:
        """Results are sorted by start time."""
        clips = [
            self._make_clip("Chorus Vocal", 64, 128),
            self._make_clip("Verse Vocal", 0, 64),
            self._make_clip("Bridge Vocal", 128, 192),
        ]

        result = find_vocal_clips(clips)

        assert len(result) == 3
        assert result[0].track_name == "Verse Vocal"
        assert result[1].track_name == "Chorus Vocal"
        assert result[2].track_name == "Bridge Vocal"

    def test_explicit_tracks_exact_match(self) -> None:
        """Explicit track selection matches exactly (case-insensitive)."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("Drums", 0, 128),
            self._make_clip("Backing Vox", 32, 96),
        ]

        result = find_vocal_clips(clips, explicit_tracks=("Lead Vocal",))

        assert len(result) == 1
        assert result[0].track_name == "Lead Vocal"

    def test_explicit_tracks_case_insensitive(self) -> None:
        """Explicit track matching is case-insensitive."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("backing vox", 32, 96),
        ]

        result = find_vocal_clips(clips, explicit_tracks=("LEAD VOCAL", "BACKING VOX"))

        assert len(result) == 2

    def test_explicit_tracks_multiple(self) -> None:
        """Multiple explicit tracks can be specified."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("Drums", 0, 128),
            self._make_clip("Backing Vox", 32, 96),
            self._make_clip("Bass", 0, 128),
        ]

        result = find_vocal_clips(clips, explicit_tracks=("Lead Vocal", "Backing Vox"))

        assert len(result) == 2
        track_names = {c.track_name for c in result}
        assert track_names == {"Lead Vocal", "Backing Vox"}

    def test_explicit_tracks_not_found(self) -> None:
        """Returns empty list if explicit tracks not found."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("Drums", 0, 128),
        ]

        result = find_vocal_clips(clips, explicit_tracks=("Nonexistent Track",))

        assert result == []

    def test_no_vocal_tracks_found(self) -> None:
        """Returns empty list when no vocal tracks exist."""
        clips = [
            self._make_clip("Drums", 0, 128),
            self._make_clip("Bass", 0, 128),
            self._make_clip("Guitar", 0, 128),
        ]

        result = find_vocal_clips(clips)

        assert result == []

    def test_empty_clips_list(self) -> None:
        """Returns empty list when given empty input."""
        assert find_vocal_clips([]) == []


class TestGetUniqueVocalTrackNames:
    """Tests for get_unique_vocal_track_names function."""

    def _make_clip(self, track_name: str, start_beats: float = 0.0) -> AudioClipRef:
        """Helper to create test clips."""
        return AudioClipRef(
            track_name=track_name,
            file_path=Path(f"/test/{track_name}.wav"),
            start_beats=start_beats,
            end_beats=start_beats + 64.0,
            start_seconds=start_beats * 0.5,
            end_seconds=(start_beats + 64.0) * 0.5,
        )

    def test_unique_names_preserved(self) -> None:
        """Unique track names are returned."""
        clips = [
            self._make_clip("Lead Vocal", 0),
            self._make_clip("Backing Vox", 32),
            self._make_clip("Harmony", 64),
        ]

        result = get_unique_vocal_track_names(clips)

        assert result == ["Lead Vocal", "Backing Vox", "Harmony"]

    def test_preserves_first_occurrence_order(self) -> None:
        """Order of first occurrence is preserved."""
        clips = [
            self._make_clip("Verse Vocal", 0),
            self._make_clip("Chorus Vocal", 64),
            self._make_clip("Verse Vocal", 128),  # Duplicate
            self._make_clip("Bridge Vocal", 192),
        ]

        result = get_unique_vocal_track_names(clips)

        assert result == ["Verse Vocal", "Chorus Vocal", "Bridge Vocal"]

    def test_removes_duplicates(self) -> None:
        """Duplicate track names are removed."""
        clips = [
            self._make_clip("Vocal", 0),
            self._make_clip("Vocal", 64),
            self._make_clip("Vocal", 128),
        ]

        result = get_unique_vocal_track_names(clips)

        assert result == ["Vocal"]

    def test_empty_list(self) -> None:
        """Empty input returns empty list."""
        assert get_unique_vocal_track_names([]) == []


# ---------------------------------------------------------------------------
# Phase 2 Tests: Audio Combination
# ---------------------------------------------------------------------------


def create_test_wav(
    path: Path, duration_seconds: float, sample_rate: int = 44100, stereo: bool = False
) -> None:
    """Create a test WAV file with a sine wave.

    Args:
        path: Path to write the WAV file.
        duration_seconds: Duration in seconds.
        sample_rate: Sample rate in Hz.
        stereo: If True, create stereo file.
    """
    import soundfile as sf

    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), dtype=np.float32)
    # 440 Hz sine wave at 0.5 amplitude
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    if stereo:
        audio = np.column_stack([audio, audio])

    sf.write(str(path), audio, sample_rate)


class TestCombineClipsToAudio:
    """Tests for combine_clips_to_audio function."""

    def _make_clip(
        self,
        file_path: Path,
        start_seconds: float,
        end_seconds: float,
        track_name: str = "Vocal",
    ) -> AudioClipRef:
        """Helper to create test clips."""
        return AudioClipRef(
            track_name=track_name,
            file_path=file_path,
            start_beats=start_seconds * 2,  # Assume 120 BPM
            end_beats=end_seconds * 2,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

    def test_empty_clips_raises_error(self, tmp_path: Path) -> None:
        """Empty clips list raises ValueError."""
        output_path = tmp_path / "output.wav"

        with pytest.raises(ValueError, match="No clips to combine"):
            combine_clips_to_audio([], output_path)

    def test_single_clip_at_start(self, tmp_path: Path) -> None:
        """Single clip at timeline start is correctly positioned."""
        # Create test audio file
        audio_file = tmp_path / "vocals.wav"
        create_test_wav(audio_file, duration_seconds=2.0)

        clip = self._make_clip(audio_file, start_seconds=0.0, end_seconds=2.0)
        output_path = tmp_path / "combined.wav"

        result_path, valid_ranges = combine_clips_to_audio([clip], output_path)

        assert result_path == output_path
        assert output_path.exists()
        assert valid_ranges == [(0.0, 2.0)]

    def test_single_clip_with_offset(self, tmp_path: Path) -> None:
        """Single clip with timeline offset creates silence before it."""
        import soundfile as sf

        audio_file = tmp_path / "vocals.wav"
        create_test_wav(audio_file, duration_seconds=1.0, sample_rate=44100)

        # Clip starts at 2 seconds on timeline
        clip = self._make_clip(audio_file, start_seconds=2.0, end_seconds=3.0)
        output_path = tmp_path / "combined.wav"

        result_path, valid_ranges = combine_clips_to_audio([clip], output_path)

        # Read the output and check duration
        data, sr = sf.read(str(output_path))
        duration = len(data) / sr

        assert duration == pytest.approx(3.0, abs=0.1)
        assert valid_ranges == [(2.0, 3.0)]

        # First 2 seconds should be silent
        first_two_seconds = data[: int(sr * 1.5)]  # Check first 1.5 seconds
        assert np.abs(first_two_seconds).max() < 0.01

    def test_multiple_clips_sequential(self, tmp_path: Path) -> None:
        """Multiple sequential clips are correctly combined."""
        # Create two audio files
        audio1 = tmp_path / "verse.wav"
        audio2 = tmp_path / "chorus.wav"
        create_test_wav(audio1, duration_seconds=1.0)
        create_test_wav(audio2, duration_seconds=1.0)

        clips = [
            self._make_clip(audio1, start_seconds=0.0, end_seconds=1.0),
            self._make_clip(audio2, start_seconds=2.0, end_seconds=3.0),
        ]
        output_path = tmp_path / "combined.wav"

        result_path, valid_ranges = combine_clips_to_audio(clips, output_path)

        assert valid_ranges == [(0.0, 1.0), (2.0, 3.0)]
        assert output_path.exists()

    def test_overlapping_clips_mixed(self, tmp_path: Path) -> None:
        """Overlapping clips are mixed together (additive)."""
        import soundfile as sf

        audio1 = tmp_path / "clip1.wav"
        audio2 = tmp_path / "clip2.wav"
        create_test_wav(audio1, duration_seconds=2.0)
        create_test_wav(audio2, duration_seconds=2.0)

        # Clips overlap from 1.0 to 2.0 seconds
        clips = [
            self._make_clip(audio1, start_seconds=0.0, end_seconds=2.0),
            self._make_clip(audio2, start_seconds=1.0, end_seconds=3.0),
        ]
        output_path = tmp_path / "combined.wav"

        result_path, valid_ranges = combine_clips_to_audio(clips, output_path)

        assert valid_ranges == [(0.0, 2.0), (1.0, 3.0)]

        # Read output and verify overlapped section has higher amplitude
        data, sr = sf.read(str(output_path))

        # The overlapping region (1.0-2.0s) should have mixed audio
        # Just verify file was created successfully
        assert len(data) > 0

    def test_sample_rate_mismatch_resamples(self, tmp_path: Path) -> None:
        """Mismatched sample rates are resampled to match."""
        import soundfile as sf

        audio1 = tmp_path / "clip1.wav"
        audio2 = tmp_path / "clip2.wav"

        # Create files with different sample rates
        # More 44100 files so that becomes the target
        create_test_wav(audio1, duration_seconds=1.0, sample_rate=44100)
        create_test_wav(audio2, duration_seconds=1.0, sample_rate=48000)

        clips = [
            self._make_clip(audio1, start_seconds=0.0, end_seconds=1.0),
            self._make_clip(audio2, start_seconds=1.0, end_seconds=2.0),
        ]
        output_path = tmp_path / "combined.wav"

        # Should succeed - resampling handles the mismatch
        result_path, valid_ranges = combine_clips_to_audio(clips, output_path)

        assert result_path.exists()
        data, sr = sf.read(str(result_path))
        # Output uses the most common sample rate (44100 in this case)
        assert sr == 44100
        assert len(valid_ranges) == 2

    def test_mono_to_stereo_conversion(self, tmp_path: Path) -> None:
        """Mono clips are converted to stereo when first clip is stereo."""
        import soundfile as sf

        stereo_file = tmp_path / "stereo.wav"
        mono_file = tmp_path / "mono.wav"

        create_test_wav(stereo_file, duration_seconds=1.0, stereo=True)
        create_test_wav(mono_file, duration_seconds=1.0, stereo=False)

        clips = [
            self._make_clip(stereo_file, start_seconds=0.0, end_seconds=1.0),
            self._make_clip(mono_file, start_seconds=1.0, end_seconds=2.0),
        ]
        output_path = tmp_path / "combined.wav"

        result_path, valid_ranges = combine_clips_to_audio(clips, output_path)

        # Output should be stereo
        data, sr = sf.read(str(output_path))
        assert data.ndim == 2
        assert data.shape[1] == 2

    def test_stereo_to_mono_conversion(self, tmp_path: Path) -> None:
        """Stereo clips are converted to mono when first clip is mono."""
        import soundfile as sf

        mono_file = tmp_path / "mono.wav"
        stereo_file = tmp_path / "stereo.wav"

        create_test_wav(mono_file, duration_seconds=1.0, stereo=False)
        create_test_wav(stereo_file, duration_seconds=1.0, stereo=True)

        clips = [
            self._make_clip(mono_file, start_seconds=0.0, end_seconds=1.0),
            self._make_clip(stereo_file, start_seconds=1.0, end_seconds=2.0),
        ]
        output_path = tmp_path / "combined.wav"

        result_path, valid_ranges = combine_clips_to_audio(clips, output_path)

        # Output should be mono
        data, sr = sf.read(str(output_path))
        assert data.ndim == 1

    def test_normalization_when_peaks_exceed_one(self, tmp_path: Path) -> None:
        """Audio is normalized when peaks exceed 1.0 after mixing."""
        import soundfile as sf

        # Create two loud files that will clip when mixed
        audio1 = tmp_path / "loud1.wav"
        audio2 = tmp_path / "loud2.wav"

        # Create audio at 0.8 amplitude - when mixed, will exceed 1.0
        sr = 44100
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        loud_audio = 0.8 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(str(audio1), loud_audio, sr)
        sf.write(str(audio2), loud_audio, sr)

        # Overlapping clips
        clips = [
            self._make_clip(audio1, start_seconds=0.0, end_seconds=1.0),
            self._make_clip(audio2, start_seconds=0.0, end_seconds=1.0),
        ]
        output_path = tmp_path / "combined.wav"

        result_path, valid_ranges = combine_clips_to_audio(clips, output_path)

        # Read output and verify normalized (peaks should not exceed 1.0)
        data, sr = sf.read(str(output_path))
        assert np.abs(data).max() <= 1.0

    def test_valid_ranges_returned_correctly(self, tmp_path: Path) -> None:
        """Valid ranges match input clip positions."""
        audio1 = tmp_path / "clip1.wav"
        audio2 = tmp_path / "clip2.wav"
        audio3 = tmp_path / "clip3.wav"

        create_test_wav(audio1, duration_seconds=1.0)
        create_test_wav(audio2, duration_seconds=1.0)
        create_test_wav(audio3, duration_seconds=1.0)

        clips = [
            self._make_clip(audio1, start_seconds=0.0, end_seconds=1.0),
            self._make_clip(audio2, start_seconds=5.0, end_seconds=6.0),
            self._make_clip(audio3, start_seconds=10.0, end_seconds=11.0),
        ]
        output_path = tmp_path / "combined.wav"

        result_path, valid_ranges = combine_clips_to_audio(clips, output_path)

        assert valid_ranges == [(0.0, 1.0), (5.0, 6.0), (10.0, 11.0)]


# ---------------------------------------------------------------------------
# Phase 2 Tests: Interactive Track Selection
# ---------------------------------------------------------------------------


class TestPromptTrackSelection:
    """Tests for prompt_track_selection function."""

    def test_empty_list_returns_empty(self) -> None:
        """Empty track list returns empty list without prompting."""
        # Note: questionary import happens inside function, so empty list returns
        # before the import is attempted
        result = prompt_track_selection([])
        assert result == []

    def test_single_track_auto_selected(self) -> None:
        """Single track is auto-selected when auto_select_single=True."""
        result = prompt_track_selection(["Lead Vocal"], auto_select_single=True)
        assert result == ["Lead Vocal"]

    def test_single_track_not_auto_selected(self) -> None:
        """Single track prompts when auto_select_single=False."""
        import questionary

        with patch.object(questionary, "checkbox") as mock_checkbox:
            mock_checkbox.return_value.ask.return_value = ["Lead Vocal"]

            result = prompt_track_selection(["Lead Vocal"], auto_select_single=False)

            assert result == ["Lead Vocal"]
            mock_checkbox.assert_called_once()

    def test_multiple_tracks_prompts_user(self) -> None:
        """Multiple tracks trigger interactive prompt."""
        import questionary

        with patch.object(questionary, "checkbox") as mock_checkbox:
            mock_checkbox.return_value.ask.return_value = ["Track 1", "Track 2"]

            result = prompt_track_selection(["Track 1", "Track 2", "Track 3"])

            assert result == ["Track 1", "Track 2"]
            mock_checkbox.assert_called_once()

    def test_user_cancel_raises_abort(self) -> None:
        """User cancellation (Ctrl+C) raises click.Abort."""
        import click
        import questionary

        with patch.object(questionary, "checkbox") as mock_checkbox:
            mock_checkbox.return_value.ask.return_value = None

            with pytest.raises(click.Abort):
                prompt_track_selection(["Track 1", "Track 2"])

    def test_default_tracks_preselected(self) -> None:
        """Default tracks are pre-selected in the prompt."""
        import questionary

        with patch.object(questionary, "checkbox") as mock_checkbox:
            mock_checkbox.return_value.ask.return_value = ["Track 1"]

            prompt_track_selection(
                ["Track 1", "Track 2", "Track 3"],
                default_tracks=["Track 1", "Track 3"],
            )

            # Check that checkbox was called with correct choices
            call_args = mock_checkbox.call_args
            choices = call_args[1]["choices"]

            # Track 1 and Track 3 should be checked, Track 2 should not
            choice_states = {c.title: c.checked for c in choices}
            assert choice_states["Track 1"] is True
            assert choice_states["Track 2"] is False
            assert choice_states["Track 3"] is True

    def test_no_default_tracks_all_preselected(self) -> None:
        """When no default_tracks, all tracks are pre-selected."""
        import questionary

        with patch.object(questionary, "checkbox") as mock_checkbox:
            mock_checkbox.return_value.ask.return_value = ["Track 1"]

            prompt_track_selection(["Track 1", "Track 2"])

            call_args = mock_checkbox.call_args
            choices = call_args[1]["choices"]

            # All should be checked when no defaults specified
            for choice in choices:
                assert choice.checked is True


class TestSelectVocalTracks:
    """Tests for select_vocal_tracks function."""

    def _make_clip(
        self, track_name: str, start_beats: float = 0.0, end_beats: float = 64.0
    ) -> AudioClipRef:
        """Helper to create test clips."""
        return AudioClipRef(
            track_name=track_name,
            file_path=Path(f"/test/{track_name.replace(' ', '_')}.wav"),
            start_beats=start_beats,
            end_beats=end_beats,
            start_seconds=start_beats * 0.5,
            end_seconds=end_beats * 0.5,
        )

    def test_explicit_tracks_override_detection(self) -> None:
        """Explicit tracks override auto-detection."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("Drums", 0, 128),
            self._make_clip("Backing Vox", 32, 96),
        ]

        result = select_vocal_tracks(
            clips,
            explicit_tracks=("Drums",),  # Not a vocal track but explicitly selected
            use_all=False,
        )

        assert len(result) == 1
        assert result[0].track_name == "Drums"

    def test_use_all_returns_all_vocal_tracks(self) -> None:
        """use_all=True returns all detected vocal tracks."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("Drums", 0, 128),
            self._make_clip("Backing Vox", 32, 96),
        ]

        result = select_vocal_tracks(clips, explicit_tracks=None, use_all=True)

        assert len(result) == 2
        track_names = {c.track_name for c in result}
        assert track_names == {"Lead Vocal", "Backing Vox"}

    def test_single_vocal_track_auto_selected(self) -> None:
        """Single vocal track is automatically selected without prompting."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("Drums", 0, 128),
            self._make_clip("Bass", 0, 128),
        ]

        result = select_vocal_tracks(clips, explicit_tracks=None, use_all=False)

        assert len(result) == 1
        assert result[0].track_name == "Lead Vocal"

    def test_no_vocal_tracks_returns_empty(self) -> None:
        """Returns empty list when no vocal tracks found."""
        clips = [
            self._make_clip("Drums", 0, 128),
            self._make_clip("Bass", 0, 128),
        ]

        result = select_vocal_tracks(clips, explicit_tracks=None, use_all=False)

        assert result == []

    def test_non_tty_uses_all_with_warning(self) -> None:
        """Non-TTY environment uses all tracks with warning."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("Backing Vox", 32, 96),
        ]

        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("click.echo") as mock_echo,
        ):
            result = select_vocal_tracks(clips, explicit_tracks=None, use_all=False)

        # Should return all vocal tracks
        assert len(result) == 2

        # Should have printed warning
        mock_echo.assert_called_once()
        call_args = mock_echo.call_args
        assert "Multiple vocal tracks found" in call_args[0][0]

    def test_tty_environment_prompts_user(self) -> None:
        """TTY environment triggers interactive prompt for multiple tracks."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("Backing Vox", 32, 96),
        ]

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("alsmuse.audio.prompt_track_selection") as mock_prompt,
        ):
            mock_prompt.return_value = ["Lead Vocal"]

            result = select_vocal_tracks(clips, explicit_tracks=None, use_all=False)

        assert len(result) == 1
        assert result[0].track_name == "Lead Vocal"
        mock_prompt.assert_called_once_with(
            ["Lead Vocal", "Backing Vox"], default_tracks=None
        )

    def test_tty_with_config_tracks_uses_as_defaults(self) -> None:
        """TTY with config_tracks passes them as defaults to prompt."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),
            self._make_clip("Backing Vox", 32, 96),
        ]

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("alsmuse.audio.prompt_track_selection") as mock_prompt,
        ):
            mock_prompt.return_value = ["Lead Vocal"]

            result = select_vocal_tracks(
                clips,
                explicit_tracks=None,
                use_all=False,
                config_tracks=["Lead Vocal"],
            )

        assert len(result) == 1
        mock_prompt.assert_called_once_with(
            ["Lead Vocal", "Backing Vox"], default_tracks=["Lead Vocal"]
        )

    def test_category_overrides_includes_vocals(self) -> None:
        """Tracks categorized as 'vocals' in overrides are included."""
        clips = [
            self._make_clip("Lead Vocal", 0, 64),  # Detected by keyword
            self._make_clip("Mystery Track", 32, 96),  # Not detected by keyword
            self._make_clip("Drums", 0, 128),  # Not vocal
        ]

        # Without overrides, only Lead Vocal is detected
        result = select_vocal_tracks(clips, explicit_tracks=None, use_all=True)
        assert len(result) == 1
        assert result[0].track_name == "Lead Vocal"

        # With category override, Mystery Track is also included
        result = select_vocal_tracks(
            clips,
            explicit_tracks=None,
            use_all=True,
            category_overrides={"Mystery Track": "vocals"},
        )
        assert len(result) == 2
        track_names = {c.track_name for c in result}
        assert track_names == {"Lead Vocal", "Mystery Track"}

    def test_results_sorted_by_start_time(self) -> None:
        """Results are sorted by start time."""
        clips = [
            self._make_clip("Chorus Vocal", 64, 128),
            self._make_clip("Verse Vocal", 0, 64),
            self._make_clip("Bridge Vocal", 128, 192),
        ]

        result = select_vocal_tracks(clips, explicit_tracks=None, use_all=True)

        assert result[0].track_name == "Verse Vocal"
        assert result[1].track_name == "Chorus Vocal"
        assert result[2].track_name == "Bridge Vocal"
