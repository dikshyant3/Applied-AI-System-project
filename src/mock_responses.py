"""
Builders for planner (LLM) responses, for use with ``MockLLMClient``.

They produce exactly the JSON shape the planning agent asks the LLM for, so
tests and offline demos can script a planner without calling a real model:

    response = planner_response(
        genre=label("lofi", evidence="lofi"),
        mood=label("chill", evidence="calm"),
        energy=level("low", reason="calm music is low energy"),
        playlist={"playlist_size": requirement(5, "five songs")},
    )
    client = MockLLMClient([response])
"""

from typing import Any, Dict, Optional, Sequence


def label(value: Any, source: str = "explicit", evidence: Optional[str] = None,
          reason: Optional[str] = None) -> Dict[str, Any]:
    """A label preference (genre, mood, detailed mood, decade) or a strategy."""
    return {"value": value, "source": source, "evidence": evidence, "reason": reason}


def level(level_name: str, source: str = "inferred", evidence: Optional[str] = None,
          reason: Optional[str] = None) -> Dict[str, Any]:
    """A numeric preference given as a qualitative level ("low", "high", ...)."""
    return {"level": level_name, "number": None, "source": source, "evidence": evidence, "reason": reason}


def requirement(value: Any, evidence: str) -> Dict[str, Any]:
    """A playlist requirement (size, unique artists, progression) or a duration."""
    return {"value": value, "evidence": evidence}


def constraint(constraint_type: str, evidence: str, values: Optional[list] = None,
               minimum: Optional[float] = None, maximum: Optional[float] = None) -> Dict[str, Any]:
    """A hard constraint: set types use ``values``, range types use ``minimum``/``maximum``."""
    return {"type": constraint_type, "values": values, "min": minimum, "max": maximum, "evidence": evidence}


def planner_response(
    genre: Dict[str, Any],
    mood: Dict[str, Any],
    energy: Dict[str, Any],
    *,
    detailed_mood: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
    playlist: Optional[Dict[str, Any]] = None,
    constraints: Sequence[Dict[str, Any]] = (),
    duration: Optional[Dict[str, Any]] = None,
    unsupported: Sequence[Dict[str, str]] = (),
) -> Dict[str, Any]:
    """A complete planner response; anything not given is null/empty."""
    return {
        "preferences": {
            "favorite_genre": genre,
            "favorite_mood": mood,
            "target_energy": energy,
            "target_popularity": None,
            "preferred_release_decade": None,
            "preferred_detailed_mood": detailed_mood,
            "target_instrumentalness": None,
            "target_speechiness": None,
        },
        "strategy": strategy,
        "playlist": {"playlist_size": None, "unique_artists": None, "energy_progression": None, **(playlist or {})},
        "hard_constraints": list(constraints),
        "duration_minutes": duration,
        "unsupported_requests": list(unsupported),
    }
