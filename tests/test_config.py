"""Tests for config module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from alsmuse.config import (
    CONFIG_VERSION,
    MuseConfig,
    get_config_path,
    load_config,
    save_config,
    update_category_overrides,
    update_vocal_tracks,
)

if TYPE_CHECKING:
    pass


class TestMuseConfig:
    """Tests for MuseConfig dataclass."""

    def test_default_values(self) -> None:
        """MuseConfig has sensible defaults."""
        config = MuseConfig()
        assert config.version == CONFIG_VERSION
        assert config.vocal_tracks == []
        assert config.category_overrides == {}

    def test_to_dict(self) -> None:
        """to_dict produces correct JSON structure."""
        config = MuseConfig(
            vocal_tracks=["Lead Vox", "Backing"],
            category_overrides={"My Cool Sound": "lead"},
        )
        data = config.to_dict()

        assert data["version"] == CONFIG_VERSION
        assert data["vocal_tracks"] == ["Lead Vox", "Backing"]
        assert data["category_overrides"] == {"My Cool Sound": "lead"}

    def test_from_dict(self) -> None:
        """from_dict creates MuseConfig from dictionary."""
        data = {
            "version": 1,
            "vocal_tracks": ["Vocals"],
            "category_overrides": {"Track 7": "drums"},
        }
        config = MuseConfig.from_dict(data)

        assert config.version == 1
        assert config.vocal_tracks == ["Vocals"]
        assert config.category_overrides == {"Track 7": "drums"}

    def test_from_dict_missing_fields(self) -> None:
        """from_dict handles missing fields gracefully."""
        config = MuseConfig.from_dict({})

        assert config.version == CONFIG_VERSION
        assert config.vocal_tracks == []
        assert config.category_overrides == {}


class TestGetConfigPath:
    """Tests for get_config_path function."""

    def test_returns_muse_extension(self, tmp_path: Path) -> None:
        """Config path uses .muse extension."""
        als_path = tmp_path / "song.als"
        config_path = get_config_path(als_path)

        assert config_path == tmp_path / "song.als.muse"

    def test_preserves_directory(self, tmp_path: Path) -> None:
        """Config path is in same directory as ALS file."""
        als_path = tmp_path / "subdir" / "project.als"
        config_path = get_config_path(als_path)

        assert config_path.parent == als_path.parent


class TestSaveAndLoadConfig:
    """Tests for save_config and load_config functions."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """save_config creates the config file."""
        als_path = tmp_path / "song.als"
        config = MuseConfig(vocal_tracks=["Lead"])

        result = save_config(als_path, config)

        assert result is True
        config_path = get_config_path(als_path)
        assert config_path.exists()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        """save_config writes valid JSON."""
        als_path = tmp_path / "song.als"
        config = MuseConfig(
            vocal_tracks=["Lead", "Backing"],
            category_overrides={"Bass": "bass"},
        )

        save_config(als_path, config)

        config_path = get_config_path(als_path)
        data = json.loads(config_path.read_text())

        assert data["version"] == CONFIG_VERSION
        assert data["vocal_tracks"] == ["Lead", "Backing"]
        assert data["category_overrides"] == {"Bass": "bass"}

    def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
        """load_config returns None when file doesn't exist."""
        als_path = tmp_path / "song.als"

        result = load_config(als_path)

        assert result is None

    def test_load_reads_saved_config(self, tmp_path: Path) -> None:
        """load_config reads config saved by save_config."""
        als_path = tmp_path / "song.als"
        original = MuseConfig(
            vocal_tracks=["Vox"],
            category_overrides={"Keys": "keys"},
        )

        save_config(als_path, original)
        loaded = load_config(als_path)

        assert loaded is not None
        assert loaded.vocal_tracks == ["Vox"]
        assert loaded.category_overrides == {"Keys": "keys"}

    def test_load_handles_corrupt_json(self, tmp_path: Path) -> None:
        """load_config returns None for corrupt JSON."""
        als_path = tmp_path / "song.als"
        config_path = get_config_path(als_path)
        config_path.write_text("not valid json")

        result = load_config(als_path)

        assert result is None

    def test_load_handles_invalid_structure(self, tmp_path: Path) -> None:
        """load_config handles unexpected JSON structure."""
        als_path = tmp_path / "song.als"
        config_path = get_config_path(als_path)
        config_path.write_text('["not", "an", "object"]')

        # Should not raise, returns None for invalid structure
        result = load_config(als_path)
        assert result is None


class TestUpdateFunctions:
    """Tests for update_vocal_tracks and update_category_overrides."""

    def test_update_vocal_tracks_creates_config(self, tmp_path: Path) -> None:
        """update_vocal_tracks creates config if it doesn't exist."""
        als_path = tmp_path / "song.als"

        result = update_vocal_tracks(als_path, ["Lead", "Harmony"])

        assert result.vocal_tracks == ["Lead", "Harmony"]
        assert load_config(als_path) is not None

    def test_update_vocal_tracks_preserves_overrides(self, tmp_path: Path) -> None:
        """update_vocal_tracks preserves existing category overrides."""
        als_path = tmp_path / "song.als"
        save_config(
            als_path,
            MuseConfig(
                vocal_tracks=["Old"],
                category_overrides={"Bass": "bass"},
            ),
        )

        update_vocal_tracks(als_path, ["New"])

        loaded = load_config(als_path)
        assert loaded is not None
        assert loaded.vocal_tracks == ["New"]
        assert loaded.category_overrides == {"Bass": "bass"}

    def test_update_category_overrides_creates_config(self, tmp_path: Path) -> None:
        """update_category_overrides creates config if it doesn't exist."""
        als_path = tmp_path / "song.als"

        result = update_category_overrides(als_path, {"Track 1": "drums"})

        assert result.category_overrides == {"Track 1": "drums"}
        assert load_config(als_path) is not None

    def test_update_category_overrides_preserves_vocals(self, tmp_path: Path) -> None:
        """update_category_overrides preserves existing vocal tracks."""
        als_path = tmp_path / "song.als"
        save_config(
            als_path,
            MuseConfig(
                vocal_tracks=["Lead"],
                category_overrides={"Old": "bass"},
            ),
        )

        update_category_overrides(als_path, {"New": "lead"})

        loaded = load_config(als_path)
        assert loaded is not None
        assert loaded.vocal_tracks == ["Lead"]
        assert loaded.category_overrides == {"New": "lead"}
