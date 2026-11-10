"""
Deterministic playlist validation (Phase 4). No LLM is involved.

Every check yields a ``ConstraintCheck`` with a status, the requirement, a
message, machine-readable evidence, and whether the failure is *infeasible*
(cannot be satisfied by any playlist drawn from the available catalog).

Rules
-----
V1  song_metadata        Each song has id/title/artist/genre/mood/energy; energy in [0, 1];
                         artist and title are non-empty strings; id is an integer.
V2  catalog_membership   Each song's id exists in the catalog and its record equals the
                         catalog record exactly (no fabricated or edited songs).
V3  no_duplicate_songs   No id appears twice.
V4  playlist_size        Exactly the requested number of songs. Not applied when the
                         playlist was selected by duration (no song count requested).
                         Infeasible when fewer eligible songs (or, with unique artists,
                         distinct eligible artists) exist than requested.
V5  unique_artists       (if required) No artist appears twice (case-insensitive).
V6  hard_constraint:<t>  (per constraint) Every song satisfies it, and at least one catalog
                         song can (otherwise violated + infeasible, even for an empty playlist).
V7  energy_progression   (if requested) "ascending": every energy >= the previous one AND
                         last > first. "descending": every energy <= the previous one AND
                         last < first. Needs >= 2 songs.
V8  duration             (if requested) Requires a positive numeric ``duration_seconds`` on
                         every catalog song; otherwise UNSUPPORTED (and infeasible). With
                         metadata, the total must be within +/-10% of the target.
                         Infeasible when all eligible songs together are too short.
V9  unsupported_request  Each request the planner could not represent is UNSUPPORTED.

"Eligible" songs are the catalog songs that satisfy all hard constraints.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from .playlist_builder import (
    ASCENDING, DURATION_FIELD, DURATION_TOLERANCE_RATIO, REQUIRED_SONG_FIELDS, Playlist,
    PlaylistRequirements, artist_key, duration_bounds_seconds, has_duration_metadata,
)
from .schemas import is_finite_number

PASSED = "passed"
VIOLATED = "violated"
UNSUPPORTED = "unsupported"



@dataclass(frozen=True)
class ConstraintCheck:
    name: str
    requirement: Any
    status: str
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    infeasible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "requirement": self.requirement,
            "status": self.status,
            "message": self.message,
            "evidence": dict(self.evidence),
            "infeasible": self.infeasible,
        }


@dataclass(frozen=True)
class ValidationReport:
    checks: Tuple[ConstraintCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.status == PASSED for check in self.checks)

    @property
    def violations(self) -> Tuple[ConstraintCheck, ...]:
        return tuple(c for c in self.checks if c.status == VIOLATED)

    @property
    def unsupported(self) -> Tuple[ConstraintCheck, ...]:
        return tuple(c for c in self.checks if c.status == UNSUPPORTED)

    @property
    def infeasible(self) -> bool:
        return any(check.infeasible for check in self.checks)

    @property
    def checked(self) -> Tuple[str, ...]:
        return tuple(check.name for check in self.checks)

    @property
    def outcome(self) -> str:
        """passed | infeasible | failed | unsupported"""
        if self.passed:
            return "passed"
        if self.infeasible:
            return "infeasible"
        if self.violations:
            return "failed"
        return "unsupported"

    def check(self, name: str) -> ConstraintCheck:
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "passed": self.passed,
            "infeasible": self.infeasible,
            "constraints_checked": list(self.checked),
            "violated": [c.name for c in self.violations],
            "unsupported": [c.name for c in self.unsupported],
            "checks": [c.to_dict() for c in self.checks],
        }


def validate_playlist(
    playlist: Union[Playlist, Sequence[Mapping[str, Any]]],
    requirements: PlaylistRequirements,
    catalog: Sequence[Mapping[str, Any]],
) -> ValidationReport:
    songs = playlist.songs if isinstance(playlist, Playlist) else list(playlist)
    eligible = [
        song for song in catalog
        if all(c.is_satisfied_by(song) for c in requirements.hard_constraints)
    ]
    checks: List[ConstraintCheck] = [
        _check_metadata(songs),
        _check_catalog_membership(songs, catalog),
        _check_duplicates(songs),
    ]
    if not (requirements.wants_duration_driven and has_duration_metadata(catalog)):
        checks.append(_check_size(songs, requirements, eligible))
    if requirements.unique_artists:
        checks.append(_check_unique_artists(songs, requirements, eligible))
    for constraint in requirements.hard_constraints:
        checks.append(_check_hard_constraint(songs, constraint, catalog))
    if requirements.energy_progression:
        checks.append(_check_energy_progression(songs, requirements, eligible))
    if requirements.target_duration_minutes is not None:
        checks.append(_check_duration(songs, requirements.target_duration_minutes, catalog, eligible))
    for index, request in enumerate(requirements.unsupported_requests):
        checks.append(ConstraintCheck(
            f"unsupported_request[{index}]", request, UNSUPPORTED,
            "the planner could not represent this request, so it was not applied",
            {"request": request},
        ))
    return ValidationReport(tuple(checks))


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def _check(name: str, requirement: Any, ok: bool, ok_message: str, bad_message: str,
           evidence: Mapping[str, Any], infeasible: bool = False) -> ConstraintCheck:
    """Build a passed/violated check. ``infeasible`` only counts when the check failed."""
    return ConstraintCheck(
        name, requirement, PASSED if ok else VIOLATED,
        ok_message if ok else bad_message,
        MappingProxyType(dict(evidence)), infeasible and not ok,
    )


def _song_ref(position: int, song: Mapping[str, Any]) -> Dict[str, Any]:
    return {"position": position, "id": song.get("id"), "title": song.get("title")}


def _check_metadata(songs: Sequence[Mapping[str, Any]]) -> ConstraintCheck:
    problems = []
    for position, song in enumerate(songs, start=1):
        missing = [f for f in REQUIRED_SONG_FIELDS if f not in song]
        if missing:
            problems.append({**_song_ref(position, song), "problem": f"missing fields {missing}"})
            continue
        if isinstance(song["id"], bool) or not isinstance(song["id"], int):
            problems.append({**_song_ref(position, song), "problem": f"id must be an integer, got {song['id']!r}"})
        if not is_finite_number(song["energy"]) or not 0 <= song["energy"] <= 1:
            problems.append({**_song_ref(position, song), "problem": f"energy must be in [0, 1], got {song['energy']!r}"})
        for text_field in ("title", "artist"):
            if not isinstance(song[text_field], str) or not song[text_field].strip():
                problems.append({**_song_ref(position, song), "problem": f"{text_field} must be a non-empty string"})
    return _check(
        "song_metadata", list(REQUIRED_SONG_FIELDS), not problems,
        "all songs have valid metadata", f"{len(problems)} metadata problem(s)",
        {"problems": problems},
    )


def _check_catalog_membership(songs, catalog) -> ConstraintCheck:
    by_id = {song["id"]: song for song in catalog}
    unknown, modified = [], []
    for position, song in enumerate(songs, start=1):
        original = by_id.get(song.get("id"))
        if original is None:
            unknown.append(_song_ref(position, song))
            continue
        changed = sorted(
            key for key in set(original) | set(song)
            if original.get(key, _MISSING) != song.get(key, _MISSING)
        )
        if changed:
            modified.append({**_song_ref(position, song), "fields": changed})
    return _check(
        "catalog_membership", "every song is an unmodified catalog record", not unknown and not modified,
        "all songs are unmodified catalog records",
        f"{len(unknown)} song(s) not in the catalog, {len(modified)} with altered metadata",
        {"not_in_catalog": unknown, "metadata_differs_from_catalog": modified, "catalog_size": len(catalog)},
    )


def _check_duplicates(songs) -> ConstraintCheck:
    counts = Counter(song.get("id") for song in songs)
    duplicates = {song_id: n for song_id, n in counts.items() if n > 1}
    return _check(
        "no_duplicate_songs", "each song at most once", not duplicates,
        "no song appears twice", f"{len(duplicates)} song id(s) repeated",
        {"duplicate_ids": duplicates},
    )


def _distinct_artists(songs) -> int:
    return len({artist_key(song) for song in songs})


def _check_size(songs, requirements: PlaylistRequirements, eligible) -> ConstraintCheck:
    # With unique artists, the most songs possible is the number of distinct eligible artists.
    if requirements.unique_artists:
        capacity = _distinct_artists(eligible)
        capacity_text = f"{capacity} distinct artists among eligible songs"
    else:
        capacity = len(eligible)
        capacity_text = f"{capacity} eligible songs"

    evidence = {
        "requested": requirements.size,
        "actual": len(songs),
        "eligible_catalog_songs": len(eligible),
    }
    if requirements.unique_artists:
        evidence["eligible_distinct_artists"] = capacity

    impossible = capacity < requirements.size
    bad_message = f"playlist has {len(songs)} of {requirements.size} requested songs"
    if impossible:
        bad_message += f"; the catalog only has {capacity_text}"
    return _check(
        "playlist_size", requirements.size, len(songs) == requirements.size,
        f"playlist has the requested {requirements.size} songs", bad_message,
        evidence, infeasible=impossible,
    )


def _check_unique_artists(songs, requirements, eligible) -> ConstraintCheck:
    positions: Dict[str, List[int]] = {}
    for position, song in enumerate(songs, start=1):
        positions.setdefault(artist_key(song), []).append(position)
    repeated = {artist: pos for artist, pos in positions.items() if len(pos) > 1}
    return _check(
        "unique_artists", True, not repeated,
        "no artist appears twice", f"{len(repeated)} artist(s) repeated",
        {"repeated_artists": repeated, "eligible_distinct_artists": _distinct_artists(eligible)},
        infeasible=_distinct_artists(eligible) < requirements.size,
    )


def _check_hard_constraint(songs, constraint, catalog) -> ConstraintCheck:
    offending = []
    for position, song in enumerate(songs, start=1):
        try:
            ok = constraint.is_satisfied_by(song)
            value = song[constraint.column]
        except (KeyError, TypeError):
            ok, value = False, "<missing or invalid>"
        if not ok:
            offending.append({**_song_ref(position, song), constraint.column: value})
    if constraint.values:
        requirement = {"values": list(constraint.values)}
    else:
        requirement = {"min": constraint.minimum, "max": constraint.maximum}

    satisfying = sum(1 for song in catalog if constraint.is_satisfied_by(song))
    if offending:
        bad_message = f"{len(offending)} song(s) violate {constraint.type}"
    else:
        bad_message = f"no catalog song satisfies {constraint.type}"
    # A constraint no catalog song can meet is violated even if the playlist
    # is empty (an empty playlist must not "pass" vacuously).
    return _check(
        f"hard_constraint:{constraint.type}", requirement, not offending and satisfying > 0,
        f"every song satisfies {constraint.type}", bad_message,
        {"offending_songs": offending, "catalog_songs_satisfying": satisfying},
        infeasible=satisfying == 0,
    )


def _check_energy_progression(songs, requirements, eligible) -> ConstraintCheck:
    direction = requirements.energy_progression
    rising = direction == ASCENDING
    energies = [song.get("energy") for song in songs]
    evidence: Dict[str, Any] = {"energies_in_order": energies}
    # No order can make energy move if every eligible song has the same energy.
    infeasible = len({song["energy"] for song in eligible}) < 2

    too_few = len(energies) < 2
    if too_few or not all(is_finite_number(energy) for energy in energies):
        reason = "needs at least 2 songs" if too_few else "energy values are invalid"
        return _check(
            "energy_progression", direction, False, "",
            f"cannot evaluate {direction} progression: {reason}",
            evidence, infeasible=infeasible or (too_few and len(eligible) < 2),
        )

    breaks = _order_breaks(energies, rising)
    net_change = energies[-1] - energies[0]
    evidence.update({"order_breaks": breaks, "net_change": round(net_change, 6)})

    if rising:
        moves_right_way = net_change > 0
        ok_message = "energies are non-decreasing with a net rise"
    else:
        moves_right_way = net_change < 0
        ok_message = "energies are non-increasing with a net fall"
    problem = "order breaks" if breaks else "no net change in energy"
    return _check(
        "energy_progression", direction, not breaks and moves_right_way,
        ok_message, f"{direction} progression violated: {problem}",
        evidence, infeasible=infeasible,
    )


def _order_breaks(energies: List[float], rising: bool) -> List[Dict[str, Any]]:
    """Each consecutive pair that moves the wrong way (positions are 1-based)."""
    breaks = []
    for i in range(1, len(energies)):
        previous, current = energies[i - 1], energies[i]
        wrong_way = current < previous if rising else current > previous
        if wrong_way:
            breaks.append({"from_position": i, "to_position": i + 1, "from": previous, "to": current})
    return breaks


def _check_duration(songs, target_minutes: float, catalog, eligible) -> ConstraintCheck:
    # Without a duration on every catalog song the requirement cannot be checked at all.
    missing = sum(1 for song in catalog if not has_duration_metadata([song]))
    if not catalog or missing:
        return ConstraintCheck(
            "duration", {"target_minutes": target_minutes}, UNSUPPORTED,
            f"cannot verify a {target_minutes:g}-minute playlist: the catalog has no "
            f"'{DURATION_FIELD}' metadata, and durations are never estimated",
            MappingProxyType({
                "required_field": DURATION_FIELD,
                "catalog_songs_missing_field": missing,
                "catalog_fields": sorted(catalog[0]) if catalog else [],
            }),
            infeasible=True,
        )
    if not has_duration_metadata(songs) and songs:
        return _check("duration", {"target_minutes": target_minutes}, False, "",
                      "playlist songs lack valid duration_seconds", {"songs": len(songs)})
    total = sum(song[DURATION_FIELD] for song in songs) / 60
    low, high = (bound / 60 for bound in duration_bounds_seconds(target_minutes))
    available = sum(song[DURATION_FIELD] for song in eligible) / 60
    return _check(
        "duration", {"target_minutes": target_minutes, "tolerance_ratio": DURATION_TOLERANCE_RATIO},
        low <= total <= high,
        f"total length {total:.1f} min is within {low:.1f}-{high:.1f} min",
        f"total length {total:.1f} min is outside {low:.1f}-{high:.1f} min",
        {"total_minutes": round(total, 2), "allowed_minutes": [round(low, 2), round(high, 2)],
         "eligible_catalog_minutes": round(available, 2)},
        infeasible=available < low,
    )


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()
