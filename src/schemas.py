"""
Structured recommendation plans for TuneWise AI (Phase 1).

A plan is the contract between a (future) planning agent and the existing
VibeMatch recommender. It separates three kinds of information:

* Soft preferences (``preferences``): scoring signals passed to the existing
  recommender. Each one records whether the user stated it (``explicit``) or
  the planner filled it in (``inferred``). Preferences never exclude songs.
* Hard constraints (``hard_constraints``): explicit user requirements that a
  song must satisfy. The existing scorer has no filtering, so these are NOT
  passed to ``recommend_songs``; a later playlist builder/validator enforces
  them. They can never be inferred.
* Playlist requirements (``playlist``): size (passed to the recommender as
  ``k``), unique artists, and energy progression (enforced after ranking).

Unspecified optional preferences are omitted from the recommender input so the
recommender's own defaults apply unchanged.

Plans are built only through ``validate_plan`` / ``parse_plan``. Validation
collects every problem instead of stopping at the first one, and unsupported
requests are reported as errors rather than being dropped or weakened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .recommender import SCORING_STRATEGIES


EXPLICIT = "explicit"
INFERRED = "inferred"
# Previously saved user preferences (merged in by application code, never by the LLM).
SAVED = "saved"
PREFERENCE_SOURCES = (EXPLICIT, INFERRED, SAVED)

# Derived from the recommender so the schema can never accept a strategy the
# scorer does not implement.
SUPPORTED_STRATEGIES: Tuple[str, ...] = tuple(SCORING_STRATEGIES)

# The recommender reads these with user_prefs[...] and raises KeyError if absent.
REQUIRED_PREFERENCES = ("favorite_genre", "favorite_mood", "target_energy")
# The recommender reads these with user_prefs.get(...) and its own defaults.
OPTIONAL_PREFERENCES = (
    "target_popularity",
    "preferred_release_decade",
    "preferred_detailed_mood",
    "target_instrumentalness",
    "target_speechiness",
)
PREFERENCE_FIELDS = REQUIRED_PREFERENCES + OPTIONAL_PREFERENCES

# Preference name -> catalog column it is compared against.
_LABEL_PREFERENCES = {
    "favorite_genre": "genre",
    "favorite_mood": "mood",
    "preferred_detailed_mood": "detailed_mood",
}
# Preference name -> inclusive (min, max), matching the catalog value ranges.
_NUMERIC_PREFERENCES = {
    "target_energy": (0.0, 1.0),
    "target_popularity": (0, 100),
    "target_instrumentalness": (0.0, 1.0),
    "target_speechiness": (0.0, 1.0),
}
_DECADE_RANGE = (1900, 2100)

ENERGY_PROGRESSIONS = ("ascending", "descending")
PLAYLIST_FIELDS = ("playlist_size", "unique_artists", "energy_progression")

# Constraint type -> catalog column. "Set" constraints list allowed values.
_SET_CONSTRAINTS = {
    "genre": "genre",
    "mood": "mood",
    "detailed_mood": "detailed_mood",
    "release_decade": "release_decade",
    "exclude_artist": "artist",
}
# Range constraints: type -> (catalog column, lower bound, upper bound).
_RANGE_CONSTRAINTS = {
    "energy_range": ("energy", 0.0, 1.0),
    "popularity_range": ("popularity", 0, 100),
    "instrumentalness_range": ("instrumentalness", 0.0, 1.0),
    "speechiness_range": ("speechiness", 0.0, 1.0),
}
SUPPORTED_CONSTRAINT_TYPES: Tuple[str, ...] = tuple(_SET_CONSTRAINTS) + tuple(
    _RANGE_CONSTRAINTS
)

_NO_DURATION = (
    "the song catalog has no duration metadata, so playlist length in time "
    "cannot be planned or verified"
)
# Requests that are recognizably about metadata the catalog does not have.
UNSUPPORTED_FIELD_REASONS = {
    "target_duration_minutes": _NO_DURATION,
    "duration": _NO_DURATION,
    "max_duration": _NO_DURATION,
    "duration_minutes": _NO_DURATION,
}

# Preference -> hard constraint type it must agree with.
PREFERENCE_CONSTRAINT_PAIRS = {
    "favorite_genre": "genre",
    "favorite_mood": "mood",
    "preferred_detailed_mood": "detailed_mood",
    "preferred_release_decade": "release_decade",
    "target_energy": "energy_range",
    "target_popularity": "popularity_range",
    "target_instrumentalness": "instrumentalness_range",
    "target_speechiness": "speechiness_range",
}

_TOP_LEVEL_FIELDS = ("preferences", "strategy", "playlist", "hard_constraints")
_HARDNESS_KEYS = ("priority", "strictness", "hard", "required")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Preference:
    """A soft scoring preference and where it came from."""

    value: Any
    source: str = EXPLICIT


@dataclass(frozen=True)
class HardConstraint:
    """A requirement every recommended song must satisfy."""

    type: str
    values: Tuple[Any, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    @property
    def column(self) -> str:
        if self.type in _SET_CONSTRAINTS:
            return _SET_CONSTRAINTS[self.type]
        return _RANGE_CONSTRAINTS[self.type][0]

    def is_satisfied_by(self, song: Mapping[str, Any]) -> bool:
        value = song[self.column]
        if self.type == "exclude_artist":
            return _casefold(value) not in {_casefold(v) for v in self.values}
        if self.type in _SET_CONSTRAINTS:
            return value in self.values
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True

    def to_plan_data(self) -> Dict[str, Any]:
        """The raw plan form of this constraint (what ``validate_plan`` accepts)."""
        if self.values:
            return {"type": self.type, "values": list(self.values)}
        data: Dict[str, Any] = {"type": self.type}
        if self.minimum is not None:
            data["min"] = self.minimum
        if self.maximum is not None:
            data["max"] = self.maximum
        return data


@dataclass(frozen=True)
class PlaylistSpec:
    """Playlist-level requirements. ``size=None`` means the recommender default."""

    size: Optional[int] = None
    unique_artists: bool = False
    energy_progression: Optional[str] = None


@dataclass(frozen=True)
class RecommendationPlan:
    """A validated plan. Build it with ``parse_plan`` or ``validate_plan``."""

    preferences: Mapping[str, Preference]
    strategy: Optional[str] = None
    playlist: PlaylistSpec = field(default_factory=PlaylistSpec)
    hard_constraints: Tuple[HardConstraint, ...] = ()

    def explicit_preferences(self) -> Dict[str, Any]:
        return self._values_with_source(EXPLICIT)

    def inferred_preferences(self) -> Dict[str, Any]:
        return self._values_with_source(INFERRED)

    def saved_preferences(self) -> Dict[str, Any]:
        return self._values_with_source(SAVED)

    def _values_with_source(self, source: str) -> Dict[str, Any]:
        return {
            name: preference.value
            for name, preference in self.preferences.items()
            if preference.source == source
        }

    def constraint(self, constraint_type: str) -> Optional[HardConstraint]:
        for constraint in self.hard_constraints:
            if constraint.type == constraint_type:
                return constraint
        return None


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class ValidationResult:
    plan: Optional[RecommendationPlan]
    errors: List[ValidationIssue]
    warnings: List[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        return not self.errors


class PlanValidationError(ValueError):
    """Raised by ``parse_plan`` with every validation error found."""

    def __init__(self, errors: Sequence[ValidationIssue]):
        self.errors = list(errors)
        details = "\n".join(f"  - {issue}" for issue in self.errors)
        super().__init__(f"Invalid recommendation plan:\n{details}")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def validate_plan(
    data: Any,
    catalog: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    check_capacity: bool = True,
) -> ValidationResult:
    """Validate raw plan data (e.g. parsed JSON).

    When ``catalog`` (the output of ``load_songs``) is given, the plan is also
    checked against the songs that actually exist: hard-constraint values must
    exist and the hard requirements must be satisfiable together.

    ``check_capacity=False`` skips only the "enough songs / enough distinct
    artists" checks, for callers (the playlist validator) that report
    infeasibility themselves as a structured result.
    """
    return _PlanValidator(catalog, check_capacity).run(data)


def parse_plan(
    data: Any, catalog: Optional[Sequence[Mapping[str, Any]]] = None
) -> RecommendationPlan:
    """Validate raw plan data and return the plan, or raise PlanValidationError."""
    result = validate_plan(data, catalog)
    if not result.is_valid:
        raise PlanValidationError(result.errors)
    assert result.plan is not None
    return result.plan


def plan_to_data(plan: RecommendationPlan) -> Dict[str, Any]:
    """Raw plan data for a validated plan (the inverse of ``validate_plan``)."""
    return {
        "preferences": {
            name: {"value": preference.value, "source": preference.source}
            for name, preference in plan.preferences.items()
        },
        "strategy": plan.strategy,
        "playlist": {
            "playlist_size": plan.playlist.size,
            "unique_artists": plan.playlist.unique_artists,
            "energy_progression": plan.playlist.energy_progression,
        },
        "hard_constraints": [constraint.to_plan_data() for constraint in plan.hard_constraints],
    }


def to_recommender_kwargs(plan: RecommendationPlan) -> Dict[str, Any]:
    """Convert a validated plan into keyword arguments for ``recommend_songs``.

    Usage: ``recommend_songs(songs=songs, **to_recommender_kwargs(plan))``.

    Only values the plan actually specifies are included, so the recommender's
    defaults (k=5, mode="genre-first", popularity 50, instrumentalness 0.5,
    speechiness 0.5, no decade/detailed mood) apply to everything else.

    Hard constraints, ``unique_artists`` and ``energy_progression`` have no
    counterpart in ``recommend_songs`` and are NOT represented here; they must
    be enforced on the ranked results by the playlist builder/validator.
    """
    missing = [name for name in REQUIRED_PREFERENCES if name not in plan.preferences]
    if missing:
        raise ValueError(
            f"Plan is missing preferences required by the recommender: {missing}"
        )

    kwargs: Dict[str, Any] = {
        "user_prefs": {
            name: plan.preferences[name].value
            for name in PREFERENCE_FIELDS
            if name in plan.preferences
        }
    }
    if plan.playlist.size is not None:
        kwargs["k"] = plan.playlist.size
    if plan.strategy is not None:
        kwargs["mode"] = plan.strategy
    return kwargs


# --------------------------------------------------------------------------
# Validation internals
# --------------------------------------------------------------------------


def _casefold(value: Any) -> Any:
    return value.casefold() if isinstance(value, str) else value


def is_number(value: Any) -> bool:
    """An int or float, but not a bool (True/False are ints in Python)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_finite_number(value: Any) -> bool:
    return is_number(value) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _describe(constraint: HardConstraint) -> str:
    if constraint.type in _SET_CONSTRAINTS:
        return f"{constraint.type} {list(constraint.values)}"
    return f"{constraint.type} [min={constraint.minimum}, max={constraint.maximum}]"


class _PlanValidator:
    """Collects every error and warning for one plan, then builds the plan."""

    def __init__(self, catalog: Optional[Sequence[Mapping[str, Any]]], check_capacity: bool = True):
        self.catalog = list(catalog) if catalog is not None else None
        self.check_capacity = check_capacity
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []

    def error(self, path: str, message: str) -> None:
        self.errors.append(ValidationIssue(path, message))

    def warn(self, path: str, message: str) -> None:
        self.warnings.append(ValidationIssue(path, message))

    # ---- entry point -----------------------------------------------------

    def run(self, data: Any) -> ValidationResult:
        if not isinstance(data, Mapping):
            self.error("plan", f"must be an object, got {_type_name(data)}")
            return ValidationResult(None, self.errors, self.warnings)

        self._reject_unknown_keys("plan", data, _TOP_LEVEL_FIELDS)

        # Each section returns None when it is invalid.
        preferences = self._parse_preferences(data.get("preferences"))
        strategy = self._parse_strategy(data.get("strategy"))
        playlist = self._parse_playlist(data.get("playlist"))
        constraints = self._parse_constraints(data.get("hard_constraints"))

        # Cross-section checks only run on sections that parsed cleanly.
        if preferences is not None and constraints is not None:
            self._check_preferences_against_constraints(preferences, constraints)
        if playlist is not None:
            self._check_playlist_consistency(playlist)
        if self.catalog is not None and preferences is not None:
            self._check_preferences_against_catalog(preferences)
        if self.catalog is not None and playlist is not None and constraints is not None:
            self._check_constraint_values_exist(constraints)
            if self.check_capacity:
                self._check_capacity(playlist, constraints)

        if self.errors:
            return ValidationResult(None, self.errors, self.warnings)

        plan = RecommendationPlan(
            preferences=MappingProxyType(dict(preferences)),
            strategy=strategy,
            playlist=playlist,
            hard_constraints=tuple(constraints),
        )
        return ValidationResult(plan, self.errors, self.warnings)

    def _reject_unknown_keys(self, path: str, data: Mapping, allowed: Sequence[str]) -> None:
        for key in data:
            if key in allowed:
                continue
            key_path = str(key) if path == "plan" else f"{path}.{key}"
            if key in UNSUPPORTED_FIELD_REASONS:
                self.error(key_path, f"unsupported: {UNSUPPORTED_FIELD_REASONS[key]}")
            else:
                self.error(key_path, f"unknown field; supported fields are: {', '.join(allowed)}")

    # ---- single-value checks (return the cleaned value, or None on error) --

    def _label(self, path: str, value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            self.error(path, f"must be a non-empty string, got {value!r}")
            return None
        return value.strip().lower()

    def _number(self, path: str, value: Any, low: float, high: float) -> Optional[float]:
        if not is_number(value):
            self.error(path, f"must be a number, got {_type_name(value)} {value!r}")
            return None
        if not math.isfinite(value) or not low <= value <= high:
            self.error(path, f"must be between {low} and {high}, got {value!r}")
            return None
        return value

    def _decade(self, path: str, value: Any) -> Optional[int]:
        if not _is_int(value):
            self.error(path, f"must be an integer decade such as 2010, got {value!r}")
            return None
        low, high = _DECADE_RANGE
        if value % 10 != 0 or not low <= value <= high:
            self.error(path, f"must be a decade (multiple of 10) between {low} and {high}, got {value!r}")
            return None
        return value

    # ---- preferences -----------------------------------------------------

    def _parse_preferences(self, raw: Any) -> Optional[Dict[str, Preference]]:
        path = "preferences"
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            self.error(path, f"must be an object, got {_type_name(raw)}")
            return None

        self._reject_unknown_keys(path, raw, PREFERENCE_FIELDS)

        parsed: Dict[str, Preference] = {}
        invalid_names = set()
        for name in PREFERENCE_FIELDS:
            if name not in raw:
                continue
            is_valid, preference = self._parse_preference(name, raw[name])
            if not is_valid:
                invalid_names.add(name)
            elif preference is not None:
                parsed[name] = preference

        all_valid = not invalid_names
        for name in REQUIRED_PREFERENCES:
            if name not in parsed and name not in invalid_names:
                all_valid = False
                self.error(
                    f"{path}.{name}",
                    "is required by the recommender; supply it explicitly or as an inferred default",
                )
        return parsed if all_valid else None

    def _parse_preference(self, name: str, raw: Any) -> Tuple[bool, Optional[Preference]]:
        """Return (is_valid, preference). A null value is valid and means "omitted"."""
        path = f"preferences.{name}"
        if isinstance(raw, Mapping):
            is_valid, value, source = self._unwrap_preference(path, raw)
            if not is_valid:
                return False, None
        else:
            value, source = raw, EXPLICIT  # a bare value is explicit

        if value is None:
            return True, None

        checked = self._check_preference_value(name, path, value)
        if checked is None:
            return False, None
        return True, Preference(checked, source)

    def _unwrap_preference(self, path: str, raw: Mapping) -> Tuple[bool, Any, Optional[str]]:
        """Read a {"value": ..., "source": ...} envelope: (is_valid, value, source)."""
        has_unknown_keys = False
        for key in raw:
            if key in ("value", "source"):
                continue
            has_unknown_keys = True
            if key in _HARDNESS_KEYS:
                self.error(
                    f"{path}.{key}",
                    "preferences are soft scoring signals; express hard requirements in hard_constraints",
                )
            else:
                self.error(f"{path}.{key}", "unknown field; expected 'value' and 'source'")

        if "value" not in raw:
            self.error(path, "is missing 'value'")
            return False, None, None

        source = raw.get("source", EXPLICIT)
        if source not in PREFERENCE_SOURCES:
            self.error(f"{path}.source", f"must be one of {', '.join(PREFERENCE_SOURCES)}, got {source!r}")
            return False, None, None

        if has_unknown_keys:
            return False, None, None
        return True, raw["value"], source

    def _check_preference_value(self, name: str, path: str, value: Any) -> Any:
        if name in _LABEL_PREFERENCES:
            return self._label(path, value)
        if name in _NUMERIC_PREFERENCES:
            low, high = _NUMERIC_PREFERENCES[name]
            return self._number(path, value, low, high)
        return self._decade(path, value)  # preferred_release_decade

    # ---- strategy --------------------------------------------------------

    def _parse_strategy(self, raw: Any) -> Optional[str]:
        if raw is None:
            return None
        if not isinstance(raw, str):
            self.error("strategy", f"must be a string, got {_type_name(raw)}")
            return None
        normalized = raw.strip().lower()  # same normalization the recommender uses
        if normalized not in SUPPORTED_STRATEGIES:
            self.error(
                "strategy",
                f"unsupported strategy {raw!r}; supported strategies are: {', '.join(SUPPORTED_STRATEGIES)}",
            )
            return None
        return normalized

    # ---- playlist --------------------------------------------------------

    def _parse_playlist(self, raw: Any) -> Optional[PlaylistSpec]:
        path = "playlist"
        if raw is None:
            return PlaylistSpec()
        if not isinstance(raw, Mapping):
            self.error(path, f"must be an object, got {_type_name(raw)}")
            return None

        errors_before = len(self.errors)
        self._reject_unknown_keys(path, raw, PLAYLIST_FIELDS)

        size = raw.get("playlist_size")
        if size is not None and not (_is_int(size) and size >= 1):
            self.error(f"{path}.playlist_size", f"must be a positive integer, got {size!r}")

        unique_artists = raw.get("unique_artists")
        if unique_artists is None:
            unique_artists = False
        if not isinstance(unique_artists, bool):
            self.error(f"{path}.unique_artists", f"must be true or false, got {unique_artists!r}")

        progression = raw.get("energy_progression")
        if progression is not None:
            normalized = progression.strip().lower() if isinstance(progression, str) else None
            if normalized not in ENERGY_PROGRESSIONS:
                self.error(
                    f"{path}.energy_progression",
                    f"must be one of {', '.join(ENERGY_PROGRESSIONS)}, got {progression!r}",
                )
            progression = normalized

        if len(self.errors) > errors_before:
            return None
        return PlaylistSpec(size, unique_artists, progression)

    # ---- hard constraints ------------------------------------------------

    def _parse_constraints(self, raw: Any) -> Optional[List[HardConstraint]]:
        path = "hard_constraints"
        if raw is None:
            return []
        if not isinstance(raw, list):
            self.error(path, f"must be a list, got {_type_name(raw)}")
            return None

        constraints: List[HardConstraint] = []
        first_index_of_type: Dict[str, int] = {}
        all_valid = True
        for index, item in enumerate(raw):
            constraint = self._parse_constraint(f"{path}[{index}]", item)
            if constraint is None:
                all_valid = False
                continue
            if constraint.type in first_index_of_type:
                all_valid = False
                self.error(
                    f"{path}[{index}].type",
                    f"duplicate constraint type {constraint.type!r} (already at "
                    f"index {first_index_of_type[constraint.type]}); combine them into one",
                )
                continue
            first_index_of_type[constraint.type] = index
            constraints.append(constraint)
        return constraints if all_valid else None

    def _parse_constraint(self, path: str, item: Any) -> Optional[HardConstraint]:
        if not isinstance(item, Mapping):
            self.error(path, f"must be an object, got {_type_name(item)}")
            return None

        constraint_type = self._parse_constraint_type(path, item.get("type"))
        if constraint_type is None:
            return None

        errors_before = len(self.errors)
        self._check_constraint_source(path, item.get("source", EXPLICIT))
        if constraint_type in _SET_CONSTRAINTS:
            constraint = self._parse_set_constraint(path, constraint_type, item)
        else:
            constraint = self._parse_range_constraint(path, constraint_type, item)

        if len(self.errors) > errors_before:
            return None
        return constraint

    def _parse_constraint_type(self, path: str, raw_type: Any) -> Optional[str]:
        if not isinstance(raw_type, str) or not raw_type.strip():
            self.error(f"{path}.type", f"must be a non-empty string, got {raw_type!r}")
            return None
        constraint_type = raw_type.strip().lower()
        if constraint_type in UNSUPPORTED_FIELD_REASONS:
            self.error(
                f"{path}.type",
                f"unsupported constraint {raw_type!r}: {UNSUPPORTED_FIELD_REASONS[constraint_type]}",
            )
            return None
        if constraint_type not in SUPPORTED_CONSTRAINT_TYPES:
            self.error(
                f"{path}.type",
                f"unsupported constraint type {raw_type!r}; supported types are: "
                f"{', '.join(SUPPORTED_CONSTRAINT_TYPES)}",
            )
            return None
        return constraint_type

    def _check_constraint_source(self, path: str, source: Any) -> None:
        if source == INFERRED:
            self.error(
                f"{path}.source",
                "hard constraints must come from the user; express inferred values as soft preferences instead",
            )
        elif source != EXPLICIT:
            self.error(f"{path}.source", f"must be {EXPLICIT!r}, got {source!r}")

    def _parse_set_constraint(self, path: str, constraint_type: str, item: Mapping) -> Optional[HardConstraint]:
        self._reject_unknown_keys(path, item, ("type", "source", "values"))
        raw_values = item.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            self.error(f"{path}.values", f"must be a non-empty list, got {raw_values!r}")
            return None

        values: List[Any] = []
        for index, raw in enumerate(raw_values):
            value = self._parse_constraint_value(constraint_type, f"{path}.values[{index}]", raw)
            if value is not None and value not in values:  # drop duplicates, keep order
                values.append(value)
        return HardConstraint(constraint_type, values=tuple(values))

    def _parse_constraint_value(self, constraint_type: str, path: str, raw: Any) -> Any:
        if constraint_type == "release_decade":
            return self._decade(path, raw)
        if constraint_type == "exclude_artist":
            if isinstance(raw, str) and raw.strip():
                return raw.strip()  # artist names keep their case
            self.error(path, f"must be a non-empty string, got {raw!r}")
            return None
        return self._label(path, raw)

    def _parse_range_constraint(self, path: str, constraint_type: str, item: Mapping) -> Optional[HardConstraint]:
        self._reject_unknown_keys(path, item, ("type", "source", "min", "max"))
        _, low, high = _RANGE_CONSTRAINTS[constraint_type]
        minimum = item.get("min")
        maximum = item.get("max")
        if minimum is None and maximum is None:
            self.error(path, "must set at least one of 'min' or 'max'")
            return None
        if minimum is not None:
            minimum = self._number(f"{path}.min", minimum, low, high)
        if maximum is not None:
            maximum = self._number(f"{path}.max", maximum, low, high)
        if minimum is not None and maximum is not None and minimum > maximum:
            self.error(path, f"'min' ({minimum}) must not exceed 'max' ({maximum})")
        return HardConstraint(constraint_type, minimum=minimum, maximum=maximum)

    # ---- cross-section checks -------------------------------------------

    def _check_preferences_against_constraints(
        self, preferences: Mapping[str, Preference], constraints: Sequence[HardConstraint]
    ) -> None:
        constraint_by_type = {constraint.type: constraint for constraint in constraints}
        for name, constraint_type in PREFERENCE_CONSTRAINT_PAIRS.items():
            if name not in preferences or constraint_type not in constraint_by_type:
                continue
            value = preferences[name].value
            constraint = constraint_by_type[constraint_type]
            if not constraint.is_satisfied_by({constraint.column: value}):
                self.error(
                    f"preferences.{name}",
                    f"value {value!r} contradicts hard constraint {_describe(constraint)}",
                )

    def _check_playlist_consistency(self, playlist: PlaylistSpec) -> None:
        if playlist.energy_progression and playlist.size == 1:
            self.error("playlist.energy_progression", "requires a playlist_size of at least 2")

    def _check_preferences_against_catalog(self, preferences: Mapping[str, Preference]) -> None:
        columns = {**_LABEL_PREFERENCES, "preferred_release_decade": "release_decade"}
        for name, column in columns.items():
            if name not in preferences:
                continue
            value = preferences[name].value
            known_values = {song[column] for song in self.catalog}
            if value not in known_values:
                self.warn(
                    f"preferences.{name}",
                    f"{value!r} does not appear in the catalog's {column!r} values, "
                    "so it will never earn a match bonus",
                )

    def _check_constraint_values_exist(self, constraints: Sequence[HardConstraint]) -> None:
        """Every value a set constraint names must exist somewhere in the catalog."""
        for index, constraint in enumerate(constraints):
            if constraint.type not in _SET_CONSTRAINTS:
                continue
            known_values = {_casefold(song[constraint.column]) for song in self.catalog}
            for value in constraint.values:
                if _casefold(value) in known_values:
                    continue
                path = f"hard_constraints[{index}].values"
                message = f"{value!r} does not appear in the catalog's {constraint.column!r} values"
                if constraint.type == "exclude_artist":
                    self.warn(path, message + "; excluding it has no effect")
                else:
                    self.error(path, message + "; this option can never be satisfied")

    def _check_capacity(self, playlist: PlaylistSpec, constraints: Sequence[HardConstraint]) -> None:
        """Enough eligible songs (and distinct artists) must exist for the playlist."""
        candidates = [
            song for song in self.catalog
            if all(constraint.is_satisfied_by(song) for constraint in constraints)
        ]
        if not candidates:
            self.error("hard_constraints", "no song in the catalog satisfies all hard constraints together")
            return
        if playlist.size is None:
            return

        scope = "satisfy the hard constraints" if constraints else "are in the catalog"
        if playlist.size > len(candidates):
            self.error(
                "playlist.playlist_size",
                f"requests {playlist.size} songs but only {len(candidates)} songs {scope}",
            )
        if playlist.unique_artists:
            artist_count = len({_casefold(song["artist"]) for song in candidates})
            if playlist.size > artist_count:
                self.error(
                    "playlist.unique_artists",
                    f"requires {playlist.size} distinct artists but only {artist_count} distinct artists {scope}",
                )
