"""Track event detection for ALSmuse.

This module provides track categorization based on name heuristics and
event detection from MIDI activity patterns.

Track Categorization:
    When model2vec is available (pip install 'alsmuse[smart]'), track names
    are categorized using semantic embeddings for better matching. Falls back
    to keyword matching if embeddings aren't available or confidence is low.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

import numpy as np
from model2vec import StaticModel

from .midi import check_activity_in_range
from .models import MidiClipContent, Phrase, TrackEvent

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Track category keywords for classification
# These are used both for keyword matching AND to generate category embeddings
# Keywords should be lowercase - matching is case-insensitive
TRACK_CATEGORIES: dict[str, list[str]] = {
    "drums": [
        "kick",
        "snare",
        "drum",
        "hi-hat",
        "hat",
        "cymbal",
        "percussion",
        "perc",
        "tom",
        "clap",
        "shaker",
    ],
    "bass": ["bass", "sub bass", "sub", "low end", "808"],
    "vocals": [
        "vocal",
        "vocals",
        "vox",
        "voice",
        "singing",
        "lead vocal",
        "backing vocal",
        "verse",
        "chorus",
        "main",
        "double",
        "harmony",
    ],
    "lead": ["lead", "solo", "melody", "top line"],
    "guitar": ["guitar", "acoustic guitar", "electric guitar", "gtr"],
    "keys": [
        "piano",
        "keys",
        "keyboard",
        "organ",
        "synth",
        "synthesizer",
        "rhodes",
        "wurlitzer",
    ],
    "pad": ["pad", "strings", "atmosphere", "ambient", "texture"],
    "fx": [
        "fx",
        "effects",
        "riser",
        "downlifter",
        "reverse",
        "sweep",
        "impact",
        "transition",
    ],
}

# Similarity threshold for embedding-based matching
# Below this threshold, return "other"
EMBEDDING_SIMILARITY_THRESHOLD = 0.28

# Minimum margin between best and second-best category
# If the best category isn't clearly better, return "other"
EMBEDDING_MARGIN_THRESHOLD = 0.05

# Generic track name patterns that should return "other"
# These match common default track names like "Track 1", "Audio 2", "MIDI 3"
# Uses regex to match pattern + optional number at end
GENERIC_TRACK_PATTERN = re.compile(
    r"^(track|audio|midi|aux|bus|return|master|group|channel|send)(\s*\d+)?$",
    re.IGNORECASE,
)

# Module-level cache for model and embeddings (lazy loaded)
_embedding_model: StaticModel | None = None
_category_centroids: dict[str, NDArray[np.float32]] | None = None


def _get_embedding_model() -> StaticModel:
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.debug("Loading Model2Vec embedding model...")
        _embedding_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
        logger.debug("Model2Vec loaded successfully")
    return _embedding_model


def _get_category_centroids() -> dict[str, NDArray[np.float32]]:
    """Compute and cache category centroid embeddings."""
    global _category_centroids
    if _category_centroids is None:
        model = _get_embedding_model()
        _category_centroids = {}
        for category, keywords in TRACK_CATEGORIES.items():
            # Embed all keywords for this category
            embeddings = model.encode(keywords)
            # Compute centroid (mean of all keyword embeddings)
            centroid = np.mean(embeddings, axis=0).astype(np.float32)
            # Normalize for cosine similarity
            centroid = centroid / np.linalg.norm(centroid)
            _category_centroids[category] = centroid

        logger.debug("Computed centroids for %d categories", len(_category_centroids))

    return _category_centroids


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def categorize_track_embedding(
    track_name: str,
    debug: bool = False,
) -> tuple[str, float, float]:
    """Categorize track using semantic embeddings.

    Uses Model2Vec to embed the track name and find the most similar
    category based on cosine similarity to category centroids.

    Args:
        track_name: Name of the track to categorize.
        debug: If True, print similarity scores for all categories.

    Returns:
        Tuple of (category, similarity_score, margin). The margin is the
        difference between best and second-best scores.
        Returns ("other", 0.0, 0.0) if embeddings aren't available.
    """
    model = _get_embedding_model()
    centroids = _get_category_centroids()

    if model is None or centroids is None:
        return ("other", 0.0, 0.0)

    # Embed the track name
    track_embedding = model.encode([track_name])[0].astype(np.float32)

    # Find most similar category
    all_scores: list[tuple[str, float]] = []

    for category, centroid in centroids.items():
        similarity = _cosine_similarity(track_embedding, centroid)
        all_scores.append((category, similarity))

    # Sort by similarity (descending)
    all_scores.sort(key=lambda x: x[1], reverse=True)

    best_category, best_similarity = all_scores[0]
    second_similarity = all_scores[1][1] if len(all_scores) > 1 else 0.0
    margin = best_similarity - second_similarity

    if debug or os.environ.get("ALSMUSE_DEBUG_CATEGORIZE"):
        scores_str = ", ".join(f"{cat}={sim:.3f}" for cat, sim in all_scores)
        print(
            f"[categorize] '{track_name}' -> {best_category} "
            f"(score={best_similarity:.3f}, margin={margin:.3f})"
        )
        print(f"             scores: {scores_str}")

    return (best_category, best_similarity, margin)


def categorize_track(track_name: str, debug: bool = False) -> str:
    """Determine track category from name using semantic embeddings.

    Uses Model2Vec embeddings to find the most semantically similar category
    for the track name. Returns "other" if:
    - The track name matches a generic pattern (e.g., "Track 1", "Audio 2")
    - The similarity score is below the threshold

    Args:
        track_name: Name of the track to categorize.
        debug: If True, print debug information about the matching process.

    Returns:
        Category string, or "other" if no confident match is found.

    Examples:
        >>> categorize_track("Drum Kit")
        'drums'
        >>> categorize_track("Bass Line")
        'bass'
        >>> categorize_track("Track 7")
        'other'
    """
    debug = debug or bool(os.environ.get("ALSMUSE_DEBUG_CATEGORIZE"))

    # Check for generic track names that should always be "other"
    if GENERIC_TRACK_PATTERN.match(track_name.strip()):
        if debug:
            print(f"[categorize] '{track_name}' -> other (generic track name)")
        return "other"

    # Use embedding-based categorization
    category, similarity, margin = categorize_track_embedding(track_name, debug=debug)

    # Check both threshold and margin
    if similarity >= EMBEDDING_SIMILARITY_THRESHOLD and margin >= EMBEDDING_MARGIN_THRESHOLD:
        if debug:
            print(
                f"[categorize] '{track_name}' -> {category} "
                f"(score={similarity:.3f}, margin={margin:.3f})"
            )
        return category

    if debug:
        if similarity < EMBEDDING_SIMILARITY_THRESHOLD:
            print(
                f"[categorize] '{track_name}' -> other (below threshold: "
                f"{similarity:.3f} < {EMBEDDING_SIMILARITY_THRESHOLD})"
            )
        else:
            print(
                f"[categorize] '{track_name}' -> other (margin too small: "
                f"{margin:.3f} < {EMBEDDING_MARGIN_THRESHOLD})"
            )

    return "other"


def detect_track_events(
    track_name: str,
    activity: list[tuple[float, bool]],
    category: str,
) -> list[TrackEvent]:
    """Convert activity samples to enter/exit events.

    Analyzes the activity pattern to detect when a track starts playing
    (enter) or stops playing (exit).

    Args:
        track_name: Name of the track.
        activity: List of (beat, is_active) tuples from activity detection.
        category: The track category (e.g., "drums", "bass").

    Returns:
        List of TrackEvent objects for state changes.
    """
    events: list[TrackEvent] = []
    was_active = False

    for beat, is_active in activity:
        if is_active and not was_active:
            events.append(
                TrackEvent(
                    beat=beat,
                    track_name=track_name,
                    event_type="enter",
                    category=category,
                )
            )
        elif not is_active and was_active:
            events.append(
                TrackEvent(
                    beat=beat,
                    track_name=track_name,
                    event_type="exit",
                    category=category,
                )
            )
        was_active = is_active

    return events


def detect_events_from_clip_contents(
    track_name: str,
    clip_contents: list[MidiClipContent],
    resolution_beats: float = 8.0,
) -> list[TrackEvent]:
    """Detect track events from MIDI clip contents.

    Combines activity detection with event generation for a track.

    Args:
        track_name: Name of the track.
        clip_contents: List of MidiClipContent for the track.
        resolution_beats: Size of each detection window in beats.

    Returns:
        List of TrackEvent objects for state changes.
    """
    from .midi import detect_midi_activity

    if not clip_contents:
        return []

    category = categorize_track(track_name)
    activity = detect_midi_activity(clip_contents, resolution_beats)
    return detect_track_events(track_name, activity, category)


def detect_phrase_activity(
    phrases: list[Phrase],
    clip_contents: list[MidiClipContent],
) -> list[tuple[Phrase, bool]]:
    """For each phrase, determine if track has any note activity.

    Uses the phrase's exact time boundaries to check for MIDI note
    activity, enabling phrase-aligned event detection.

    Args:
        phrases: List of Phrase objects to check activity for.
        clip_contents: List of MidiClipContent for a track.

    Returns:
        List of (phrase, is_active) tuples.
    """
    result: list[tuple[Phrase, bool]] = []
    for phrase in phrases:
        is_active = check_activity_in_range(clip_contents, phrase.start_beats, phrase.end_beats)
        result.append((phrase, is_active))
    return result


def detect_events_from_phrase_activity(
    track_name: str,
    phrase_activity: list[tuple[Phrase, bool]],
    category: str,
) -> list[TrackEvent]:
    """Compare adjacent phrases to detect state changes.

    Events are placed at the START of the phrase where the change occurs.
    This approach aligns events with musical structure rather than an
    arbitrary global grid.

    Args:
        track_name: Name of the track.
        phrase_activity: List of (phrase, is_active) tuples from detect_phrase_activity.
        category: The track category (e.g., "drums", "bass").

    Returns:
        List of TrackEvent objects for state changes.
    """
    events: list[TrackEvent] = []
    was_active = False

    for phrase, is_active in phrase_activity:
        if is_active and not was_active:
            events.append(
                TrackEvent(
                    beat=phrase.start_beats,
                    track_name=track_name,
                    event_type="enter",
                    category=category,
                )
            )
        elif not is_active and was_active:
            events.append(
                TrackEvent(
                    beat=phrase.start_beats,
                    track_name=track_name,
                    event_type="exit",
                    category=category,
                )
            )
        was_active = is_active

    return events


def detect_events_from_clip_contents_phrase_aligned(
    track_name: str,
    clip_contents: list[MidiClipContent],
    phrases: list[Phrase],
) -> list[TrackEvent]:
    """Detect track events using phrase-aligned boundaries.

    Combines phrase activity detection with event generation.
    This is the phrase-aligned replacement for detect_events_from_clip_contents.

    Args:
        track_name: Name of the track.
        clip_contents: List of MidiClipContent for the track.
        phrases: List of Phrase objects defining the time boundaries.

    Returns:
        List of TrackEvent objects for state changes.
    """
    if not clip_contents or not phrases:
        return []

    category = categorize_track(track_name)
    phrase_activity = detect_phrase_activity(phrases, clip_contents)
    return detect_events_from_phrase_activity(track_name, phrase_activity, category)


def merge_events_into_phrases(
    phrases: list[Phrase],
    events: list[TrackEvent],
) -> list[Phrase]:
    """Attach events to the phrases they occur in.

    Events are grouped by category for cleaner output:
    "Drums enters, Bass enters" instead of "KICK enters, SNARE enters, Bass enters".

    Args:
        phrases: List of Phrase objects to attach events to.
        events: List of TrackEvent objects to distribute.

    Returns:
        New list of Phrase objects with events attached.
    """
    if not phrases:
        return phrases

    # Sort events by beat
    sorted_events = sorted(events, key=lambda e: e.beat)

    result: list[Phrase] = []
    event_idx = 0

    for phrase in phrases:
        phrase_events: list[TrackEvent] = []

        # Collect events that fall within this phrase
        while event_idx < len(sorted_events) and sorted_events[event_idx].beat < phrase.end_beats:
            if sorted_events[event_idx].beat >= phrase.start_beats:
                phrase_events.append(sorted_events[event_idx])
            event_idx += 1

        # Deduplicate by category and event type
        seen_categories: dict[tuple[str, str], TrackEvent] = {}
        for event in phrase_events:
            key = (event.category, event.event_type)
            if key not in seen_categories:
                seen_categories[key] = event

        result.append(
            Phrase(
                start_beats=phrase.start_beats,
                end_beats=phrase.end_beats,
                section_name=phrase.section_name,
                is_section_start=phrase.is_section_start,
                events=tuple(seen_categories.values()),
                lyric=phrase.lyric,
            )
        )

    return result
