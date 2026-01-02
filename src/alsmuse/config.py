"""Configuration file management for ALSmuse.

This module provides loading and saving of .muse configuration files
that store track category overrides and vocal track selections.
The config file is a JSON file stored alongside the ALS file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Current config file version
CONFIG_VERSION = 1


@dataclass
class MuseConfig:
    """Configuration stored in a .muse file.

    Attributes:
        version: Config file format version.
        vocal_tracks: List of track names selected as vocal tracks.
        category_overrides: Mapping of track names to category overrides.
            The category values should match keys in TRACK_CATEGORIES
            (e.g., "drums", "bass", "vocals", "lead", "guitar", "keys", "pad", "fx")
            or "other" to exclude from event detection.
        start_bar: Bar number where the song starts (for time offset).
            If None, times are shown from bar 0.
    """

    version: int = CONFIG_VERSION
    vocal_tracks: list[str] = field(default_factory=list)
    category_overrides: dict[str, str] = field(default_factory=dict)
    start_bar: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "version": self.version,
            "vocal_tracks": self.vocal_tracks,
            "category_overrides": self.category_overrides,
        }
        if self.start_bar is not None:
            result["start_bar"] = self.start_bar
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MuseConfig:
        """Create from dictionary (parsed JSON).

        Args:
            data: Dictionary from parsed JSON.

        Returns:
            MuseConfig instance.

        Raises:
            TypeError: If data is not a dictionary.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        return cls(
            version=data.get("version", CONFIG_VERSION),
            vocal_tracks=data.get("vocal_tracks", []),
            category_overrides=data.get("category_overrides", {}),
            start_bar=data.get("start_bar"),
        )


def get_config_path(als_path: Path) -> Path:
    """Get the config file path for an ALS file.

    The config file is named {als_filename}.muse and lives alongside the ALS file.

    Args:
        als_path: Path to the .als file.

    Returns:
        Path to the corresponding .muse config file.
    """
    return als_path.parent / f"{als_path.name}.muse"


def load_config(als_path: Path) -> MuseConfig | None:
    """Load config from .muse file if it exists.

    Args:
        als_path: Path to the .als file.

    Returns:
        MuseConfig if file exists and is valid, None otherwise.
    """
    config_path = get_config_path(als_path)

    if not config_path.exists():
        logger.debug("No config file found at %s", config_path)
        return None

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        config = MuseConfig.from_dict(data)
        logger.info("Loaded config from %s", config_path)
        return config
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to parse config file %s: %s", config_path, e)
        return None
    except OSError as e:
        logger.warning("Failed to read config file %s: %s", config_path, e)
        return None


def save_config(als_path: Path, config: MuseConfig) -> bool:
    """Save config to .muse file.

    Args:
        als_path: Path to the .als file.
        config: Configuration to save.

    Returns:
        True if saved successfully, False otherwise.
    """
    config_path = get_config_path(als_path)

    try:
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info("Saved config to %s", config_path)
        return True
    except OSError as e:
        logger.warning("Failed to write config file %s: %s", config_path, e)
        return False


def update_vocal_tracks(als_path: Path, vocal_tracks: list[str]) -> MuseConfig:
    """Update vocal tracks in config, creating if necessary.

    Args:
        als_path: Path to the .als file.
        vocal_tracks: List of vocal track names.

    Returns:
        Updated config.
    """
    config = load_config(als_path) or MuseConfig()
    config.vocal_tracks = vocal_tracks
    save_config(als_path, config)
    return config


def update_category_overrides(als_path: Path, category_overrides: dict[str, str]) -> MuseConfig:
    """Update category overrides in config, creating if necessary.

    Args:
        als_path: Path to the .als file.
        category_overrides: Mapping of track names to categories.

    Returns:
        Updated config.
    """
    config = load_config(als_path) or MuseConfig()
    config.category_overrides = category_overrides
    save_config(als_path, config)
    return config
