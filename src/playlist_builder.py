"""
Deterministic playlist construction from the recommender's ranking (Phase 4).

Input is exactly what ``recommend_songs`` returns: an ordered list of
``(song, score, explanation)`` tuples, best first. The builder never scores,
invents, or edits songs; it only selects and orders them.

Selection rules (applied in this order)
--------------------------------------
S1  Relevance: songs are considered strictly in the recommender's rank order.
S2  No duplicates: a song whose ``id`` was already selected is skipped.
S3  Unique artists (only when required): a song whose artist (case-insensitive,
    trimmed) was already selected is skipped.
S4  Size: selection stops once ``size`` songs are selected.
S5  Shortfall: if the ranking runs out first, the shorter playlist is returned
    with ``shortfall > 0``. No rule is relaxed to fill the gap.

Duration-driven selection (only when a total length is requested, no song
count is requested, and EVERY ranked song has numeric ``duration_seconds``)
--------------------------------------------------------------------------
D1  Walk the ranking in order, applying S2 and S3.
D2  Add a song only if the running total stays <= target * (1 + 10%).
D3  Stop as soon as the total reaches >= target * (1 - 10%).
Without duration metadata, selection falls back to S4 with the default size,
and the validator reports the duration as UNSUPPORTED. Durations are never
estimated.

Ordering rules (applied to the selected songs)
----------------------------------------------
O1  No progression: rank order.
O2  "ascending":  sorted by (energy, rank)  -> energies are non-decreasing.
O3  "descending": sorted by (-energy, rank) -> energies are non-increasing.
Ties are always broken by relevance rank, so output is fully deterministic.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from .schemas import ENERGY_PROGRESSIONS, HardConstraint, RecommendationPlan, is_finite_number

ASCENDING, DESCENDING = ENERGY_PROGRESSIONS  # ("ascending", "descending")
REQUIRED_SONG_FIELDS = ("id", "title", "artist", "genre", "mood", "energy")
DURATION_FIELD = "duration_seconds"
DURATION_TOLERANCE_RATIO = 0.10


def has_duration_metadata(songs: Sequence[Mapping[str, Any]]) -> bool:
    """True only if there are songs and every one has a positive numeric duration."""
    return bool(songs) and all(_has_duration(song) for song in songs)


def _has_duration(song: Mapping[str, Any]) -> bool:
    duration = song.get(DURATION_FIELD)
    return is_finite_number(duration) and duration > 0


def duration_bounds_seconds(target_minutes: float) -> Tuple[float, float]:
    return (
        target_minutes * 60 * (1 - DURATION_TOLERANCE_RATIO),
        target_minutes * 60 * (1 + DURATION_TOLERANCE_RATIO),
    )



class PlaylistInputError(ValueError):
    """The ranked input is malformed (missing or invalid song metadata)."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("Invalid ranked songs:\n" + "\n".join(f"  - {p}" for p in self.problems))


@dataclass(frozen=True)
class PlaylistRequirements:
    """Everything the playlist must satisfy, in one place for builder and validator."""

    size: int
    unique_artists: bool = False
    energy_progression: Optional[str] = None
    hard_constraints: Tuple[HardConstraint, ...] = ()
    target_duration_minutes: Optional[float] = None
    unsupported_requests: Tuple[str, ...] = ()
    size_requested: bool = True
    """False when ``size`` is only the application default."""

    @property
    def wants_duration_driven(self) -> bool:
        return self.target_duration_minutes is not None and not self.size_requested

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 1:
            raise ValueError(f"size must be a positive integer, got {self.size!r}")
        if self.energy_progression not in (None, *ENERGY_PROGRESSIONS):
            raise ValueError(f"unsupported energy_progression {self.energy_progression!r}")

    @classmethod
    def from_plan(
        cls,
        plan: RecommendationPlan,
        default_size: int,
        target_duration_minutes: Optional[float] = None,
        unsupported_requests: Sequence[str] = (),
    ) -> "PlaylistRequirements":
        return cls(
            size=plan.playlist.size or default_size,
            unique_artists=plan.playlist.unique_artists,
            energy_progression=plan.playlist.energy_progression,
            hard_constraints=tuple(plan.hard_constraints),
            target_duration_minutes=target_duration_minutes,
            unsupported_requests=tuple(unsupported_requests),
            size_requested=plan.playlist.size is not None,
        )


@dataclass(frozen=True)
class PlaylistEntry:
    position: int
    """1-based position in the final playlist order."""
    rank: int
    """1-based relevance rank in the recommender's output."""
    song: Mapping[str, Any]
    """Read-only copy of the catalog record, unmodified."""
    score: float
    """Score exactly as returned by recommend_songs."""
    explanation: str
    """Explanation exactly as returned by recommend_songs."""


@dataclass(frozen=True)
class SkippedSong:
    rank: int
    song_id: Any
    title: str
    rule: str
    detail: str


@dataclass(frozen=True)
class Playlist:
    entries: Tuple[PlaylistEntry, ...]
    requested_size: int
    ranked_pool_size: int
    skipped: Tuple[SkippedSong, ...] = field(default=())
    selection_mode: str = "size"
    """"size" (S4) or "duration" (D1-D3)."""

    @property
    def total_duration_seconds(self) -> Optional[float]:
        """Total length, or None when any song lacks duration metadata."""
        if not has_duration_metadata(self.songs):
            return None
        return sum(song[DURATION_FIELD] for song in self.songs)

    @property
    def shortfall(self) -> int:
        return max(0, self.requested_size - len(self.entries))

    @property
    def songs(self) -> List[Mapping[str, Any]]:
        return [entry.song for entry in self.entries]


class _Selected(NamedTuple):
    rank: int
    song: Mapping[str, Any]
    score: float
    explanation: str


def build_playlist(
    ranked: Sequence[Tuple[Mapping[str, Any], float, str]],
    requirements: PlaylistRequirements,
) -> Playlist:
    """Select and order songs from the recommender's ranking (see module rules)."""
    _check_ranked_input(ranked)

    by_duration = requirements.wants_duration_driven and has_duration_metadata([song for song, _, _ in ranked])
    selected, skipped = _select_songs(ranked, requirements, by_duration)
    selected = _order_by_energy(selected, requirements.energy_progression)

    entries = tuple(
        PlaylistEntry(position, item.rank, MappingProxyType(dict(item.song)), item.score, item.explanation)
        for position, item in enumerate(selected, start=1)
    )
    if by_duration:
        return Playlist(entries, len(entries), len(ranked), tuple(skipped), "duration")
    return Playlist(entries, requirements.size, len(ranked), tuple(skipped), "size")


def _select_songs(ranked, requirements: PlaylistRequirements, by_duration: bool):
    """Walk the ranking in order (S1/D1) and keep songs until a stop rule fires."""
    if by_duration:
        min_seconds, max_seconds = duration_bounds_seconds(requirements.target_duration_minutes)
    else:
        min_seconds, max_seconds = 0, None

    selected: List[_Selected] = []
    skipped: List[SkippedSong] = []
    seen_ids = set()
    seen_artists = set()
    total_seconds = 0.0

    for rank, (song, score, explanation) in enumerate(ranked, start=1):
        if by_duration and total_seconds >= min_seconds:  # D3: long enough
            break
        if not by_duration and len(selected) == requirements.size:  # S4: enough songs
            break

        reason = _skip_reason(song, seen_ids, seen_artists, requirements.unique_artists, max_seconds, total_seconds)
        if reason is not None:
            rule, detail = reason
            skipped.append(SkippedSong(rank, song["id"], song["title"], rule, detail))
            continue

        seen_ids.add(song["id"])
        seen_artists.add(artist_key(song))
        if by_duration:
            total_seconds += song[DURATION_FIELD]
        selected.append(_Selected(rank, song, score, explanation))
    return selected, skipped


def _skip_reason(song, seen_ids, seen_artists, unique_artists: bool, max_seconds, total_seconds: float):
    """(rule, detail) if the song must be skipped, else None. Rules are checked in order S2, S3, D2."""
    if song["id"] in seen_ids:
        return "S2 duplicate song", f"id {song['id']} already selected"
    if unique_artists and artist_key(song) in seen_artists:
        return "S3 unique artists", f"artist {song['artist']!r} already selected"
    if max_seconds is not None and total_seconds + song[DURATION_FIELD] > max_seconds:
        return "D2 duration cap", f"{song[DURATION_FIELD]}s would exceed {max_seconds:.0f}s"
    return None


def _order_by_energy(selected: List[_Selected], progression: Optional[str]) -> List[_Selected]:
    """O1-O3: rank order, or energy order with ties broken by rank."""
    if progression == ASCENDING:
        return sorted(selected, key=lambda item: (item.song["energy"], item.rank))
    if progression == DESCENDING:
        return sorted(selected, key=lambda item: (-item.song["energy"], item.rank))
    return selected


def artist_key(song: Mapping[str, Any]) -> str:
    """Case- and space-insensitive artist name, used to compare artists."""
    return str(song.get("artist", "")).strip().casefold()


def _check_ranked_input(ranked: Sequence[Any]) -> None:
    problems = []
    for index, item in enumerate(ranked):
        problems += _ranked_item_problems(f"ranked[{index}]", item)
    if problems:
        raise PlaylistInputError(problems)


def _ranked_item_problems(where: str, item: Any) -> List[str]:
    """Problems with one (song, score, explanation) item; empty if it is valid."""
    if not isinstance(item, (tuple, list)) or len(item) != 3:
        return [f"{where}: expected (song, score, explanation)"]
    song, score, explanation = item
    if not isinstance(song, Mapping):
        return [f"{where}: song must be a mapping"]
    missing = [name for name in REQUIRED_SONG_FIELDS if name not in song]
    if missing:
        return [f"{where}: song is missing {missing}"]

    problems = []
    energy = song["energy"]
    if not (is_finite_number(energy) and 0 <= energy <= 1):
        problems.append(f"{where}: energy must be a number in [0, 1], got {energy!r}")
    if not isinstance(song["artist"], str) or not song["artist"].strip():
        problems.append(f"{where}: artist must be a non-empty string, got {song['artist']!r}")
    if not is_finite_number(score):
        problems.append(f"{where}: score must be a finite number, got {score!r}")
    if not isinstance(explanation, str):
        problems.append(f"{where}: explanation must be a string")
    return problems
