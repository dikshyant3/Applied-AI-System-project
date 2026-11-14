"""Deterministic like, dislike, and skip processing for catalog songs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence, Tuple

from .schemas import PREFERENCE_FIELDS
from .user_profile import UserProfileStore

_CATEGORY_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("favorite_genre", "genre"),
    ("favorite_mood", "mood"),
    ("preferred_detailed_mood", "detailed_mood"),
    ("preferred_release_decade", "release_decade"),
)
_NUMERIC_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("target_energy", "energy"),
    ("target_popularity", "popularity"),
    ("target_instrumentalness", "instrumentalness"),
    ("target_speechiness", "speechiness"),
)
_ACTION_WEIGHT = {"like": 1, "dislike": -1, "skip": 0}


def submit_feedback(store: UserProfileStore, song_id: int, action: str) -> Dict[str, Any]:
    """Validate and store feedback, then recompute preferences from the full history.

    Category preferences are the highest positive like-minus-dislike vote,
    with catalog order breaking ties. Numeric preferences are means over liked
    songs. Skips are recorded but do not influence preference values.
    """
    song = store.song(song_id)
    if not isinstance(action, str) or action not in _ACTION_WEIGHT:
        raise ValueError("action must be 'like', 'dislike', or 'skip'")

    history = [*store.feedback, {"song_id": song_id, "action": action}]
    preferences = _derive_preferences(history, store)
    store.save_feedback(song_id, action, preferences)
    return dict(preferences)


def latest_reactions(store: UserProfileStore) -> Dict[int, str]:
    """Each rated song's most recent action (the history itself is append-only)."""
    reactions: Dict[int, str] = {}
    for event in store.feedback:
        reactions[event["song_id"]] = event["action"]
    return reactions


def clear_feedback(store: UserProfileStore, song_id: int) -> Dict[str, Any]:
    """Forget every reaction to one song (un-like / un-dislike) and recompute preferences.

    Preferences that only came from feedback are dropped when no like or
    dislike remains to support them.
    """
    store.song(song_id)
    history = [event for event in store.feedback if event["song_id"] != song_id]
    if any(event["action"] != "skip" for event in history):
        preferences = _derive_preferences(history, store)
    else:
        derived = {field for field, _ in _CATEGORY_FIELDS + _NUMERIC_FIELDS}
        preferences = {k: v for k, v in store.preferences.items() if k not in derived}
    store.replace_feedback(history, preferences)
    return dict(preferences)


def _derive_preferences(history: Sequence[Mapping[str, Any]],
                        store: UserProfileStore) -> Dict[str, Any]:
    preferences = store.preferences
    non_skip = [event for event in history if event["action"] != "skip"]

    for field, column in _CATEGORY_FIELDS:
        relevant = [event for event in non_skip if column in store.song(event["song_id"])]
        if not relevant:
            continue
        scores: Dict[Any, int] = defaultdict(int)
        for event in relevant:
            value = store.song(event["song_id"])[column]
            scores[value] += _ACTION_WEIGHT[event["action"]]
        best_score = max(scores.values(), default=0)
        if best_score > 0:
            preferences[field] = next(
                store.song(song_id)[column]
                for song_id in _catalog_order(store)
                if scores.get(store.song(song_id)[column]) == best_score
            )
        else:
            preferences.pop(field, None)

    for field, column in _NUMERIC_FIELDS:
        relevant_likes = [
            store.song(event["song_id"])[column]
            for event in history
            if event["action"] == "like" and column in store.song(event["song_id"])
        ]
        had_relevant_feedback = any(
            event["action"] != "skip" and column in store.song(event["song_id"])
            for event in history
        )
        if relevant_likes:
            preferences[field] = sum(relevant_likes) / len(relevant_likes)
        elif had_relevant_feedback:
            preferences.pop(field, None)

    return {name: value for name, value in preferences.items() if name in PREFERENCE_FIELDS}


def _catalog_order(store: UserProfileStore) -> Tuple[int, ...]:
    return store.catalog_ids


__all__ = ["clear_feedback", "latest_reactions", "submit_feedback"]
