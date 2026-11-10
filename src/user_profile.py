"""Local JSON persistence for TuneWise user preferences and feedback."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .schemas import PREFERENCE_FIELDS

PROFILE_VERSION = 1
DEFAULT_PROFILE_PATH = Path.home() / ".tunewise" / "profile.json"


class UserProfileError(ValueError):
    """A saved profile is malformed or contains unsupported data."""


class UserProfileStore:
    """Read and atomically update one user's local JSON profile."""

    def __init__(self, path: str | Path = DEFAULT_PROFILE_PATH,
                 catalog: Sequence[Mapping[str, Any]] = ()):
        self.path = Path(path).expanduser()
        self._songs_by_id: Dict[int, Mapping[str, Any]] = {}
        for song in catalog:
            song_id = song.get("id")
            if isinstance(song_id, bool) or not isinstance(song_id, int):
                raise ValueError(f"catalog song id must be an integer, got {song_id!r}")
            if song_id in self._songs_by_id:
                raise ValueError(f"duplicate catalog song id {song_id}")
            self._songs_by_id[song_id] = song

    @property
    def preferences(self) -> Dict[str, Any]:
        """Return a copy of the validated saved preferences."""
        return dict(self._read()["preferences"])

    @property
    def feedback(self) -> list[Dict[str, Any]]:
        """Return a copy of the validated feedback history."""
        return [dict(event) for event in self._read()["feedback"]]

    @property
    def catalog_ids(self) -> tuple[int, ...]:
        """Catalog order, used as the deterministic tie-break for feedback votes."""
        return tuple(self._songs_by_id)

    def song(self, song_id: int) -> Mapping[str, Any]:
        """Return a catalog row or reject an identifier that is not in this catalog."""
        if isinstance(song_id, bool) or not isinstance(song_id, int):
            raise ValueError("song_id must be an integer")
        try:
            return self._songs_by_id[song_id]
        except KeyError:
            raise ValueError(f"song_id {song_id} is not present in the catalog") from None

    def save_preferences(self, preferences: Mapping[str, Any]) -> None:
        """Merge validated preference values into the profile."""
        if not isinstance(preferences, Mapping):
            raise ValueError("preferences must be a mapping")
        profile = self._read()
        updated = {**profile["preferences"], **preferences}
        self._validate_preferences(updated)
        profile["preferences"] = updated
        self._write(profile)

    def save_feedback(self, song_id: int, action: str,
                      preferences: Mapping[str, Any]) -> None:
        """Persist validated feedback and its deterministically updated preferences."""
        self.song(song_id)
        if action not in ("like", "dislike", "skip"):
            raise ValueError("action must be 'like', 'dislike', or 'skip'")
        if not isinstance(preferences, Mapping):
            raise ValueError("preferences must be a mapping")
        profile = self._read()
        self._validate_preferences(preferences)
        profile["preferences"] = dict(preferences)
        profile["feedback"].append({"song_id": song_id, "action": action})
        self._write(profile)

    def reset(self) -> None:
        """Clear saved preferences and feedback while keeping a valid empty profile."""
        self._write({"version": PROFILE_VERSION, "preferences": {}, "feedback": []})

    def _read(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"version": PROFILE_VERSION, "preferences": {}, "feedback": []}
        try:
            profile = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UserProfileError(f"cannot read profile at {self.path}: {error}") from error
        if (not isinstance(profile, dict) or set(profile) != {"version", "preferences", "feedback"}
                or type(profile.get("version")) is not int
                or profile.get("version") != PROFILE_VERSION):
            raise UserProfileError(f"profile at {self.path} has an unsupported format")
        if not isinstance(profile.get("preferences"), dict) or not isinstance(profile.get("feedback"), list):
            raise UserProfileError(f"profile at {self.path} must contain preferences and feedback")
        self._validate_preferences(profile["preferences"])
        for event in profile["feedback"]:
            if (not isinstance(event, dict) or set(event) != {"song_id", "action"}
                    or event["action"] not in ("like", "dislike", "skip")):
                raise UserProfileError(f"profile at {self.path} contains invalid feedback")
            try:
                self.song(event["song_id"])
            except ValueError as error:
                raise UserProfileError(f"profile at {self.path} contains invalid feedback: {error}") from error
        return profile

    def _validate_preferences(self, preferences: Mapping[str, Any]) -> None:
        unknown = set(preferences) - set(PREFERENCE_FIELDS)
        if unknown:
            raise UserProfileError(f"unsupported preference fields: {sorted(unknown, key=str)}")
        if not self._songs_by_id and preferences:
            raise UserProfileError("a catalog is required to validate saved preferences")
        for name, value in preferences.items():
            if value is None:
                raise UserProfileError(f"{name} cannot be null in a saved profile")
            if name in ("favorite_genre", "favorite_mood", "preferred_detailed_mood"):
                column = {
                    "favorite_genre": "genre",
                    "favorite_mood": "mood",
                    "preferred_detailed_mood": "detailed_mood",
                }[name]
                allowed = {song[column] for song in self._songs_by_id.values() if column in song}
                if not isinstance(value, str) or value not in allowed:
                    raise UserProfileError(f"{name} must match a value in the catalog")
            elif name == "preferred_release_decade":
                allowed_decades = {song.get("release_decade") for song in self._songs_by_id.values()}
                if isinstance(value, bool) or not isinstance(value, int) or value not in allowed_decades:
                    raise UserProfileError("preferred_release_decade must match a catalog decade")
            else:
                bounds = {
                    "target_energy": (0.0, 1.0),
                    "target_popularity": (0.0, 100.0),
                    "target_instrumentalness": (0.0, 1.0),
                    "target_speechiness": (0.0, 1.0),
                }
                minimum, maximum = bounds[name]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not minimum <= value <= maximum):
                    raise UserProfileError(f"{name} must be a number between {minimum} and {maximum}")

    def _write(self, profile: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as temporary:
                temporary_path = temporary.name
                json.dump(profile, temporary, ensure_ascii=False, indent=2, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass


__all__ = ["DEFAULT_PROFILE_PATH", "UserProfileError", "UserProfileStore"]
