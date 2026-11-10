"""
TuneWise workflow orchestrator (Phases 3-4).

natural-language request
  -> load the real catalog            (recommender.load_songs)
  -> LLM planning agent               (agent.create_recommendation_plan)
  -> validated RecommendationPlan     (schemas, re-checked here)
  -> recommender arguments            (schemas.to_recommender_kwargs)
  -> hard-constraint candidate filter (HardConstraint.is_satisfied_by)
  -> full ranking + explanations      (recommender.recommend_songs / score_song)
  -> playlist selection and ordering  (playlist_builder.build_playlist)
  -> deterministic validation report  (playlist_validator.validate_playlist)

The orchestrator never scores songs itself. Requests that the catalog cannot
satisfy are not errors: they return a playlist plus a report whose outcome is
"infeasible" (or "unsupported"), and no requirement is relaxed.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .agent import RECOMMENDER_DEFAULT_K, LLMClient, PlanningError, PlanningResult, create_recommendation_plan
from .playlist_builder import Playlist, PlaylistInputError, PlaylistRequirements, build_playlist
from .playlist_validator import ValidationReport, validate_playlist
from .recommender import SCORING_STRATEGIES, ScoringStrategy, load_songs, recommend_songs, score_song
from .schemas import RecommendationPlan, to_recommender_kwargs, validate_plan

logger = logging.getLogger("tunewise.orchestrator")

CATALOG_PATH = "data/songs.csv"
EXPLANATION_SEPARATOR = "; "  # how recommend_songs joins its reasons
RECOMMENDER_DEFAULT_MODE = inspect.signature(recommend_songs).parameters["mode"].default


class WorkflowError(Exception):
    """A workflow stage failed. ``stage`` is catalog, planning, recommendation or playlist."""

    def __init__(self, stage: str, message: str, cause: Optional[BaseException] = None):
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage} failed: {message}")


@dataclass(frozen=True)
class RecommendedSong:
    position: int
    """1-based position in the playlist."""
    rank: int
    """1-based relevance rank among eligible songs (recommender order)."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    score: float
    """Final ranking score from recommend_songs (after diversity penalty)."""
    base_score: float
    """Feature score from score_song, before the diversity penalty."""
    explanations: Tuple[str, ...]
    """recommend_songs' own feature-level reasons, one entry per feature."""
    record: Mapping[str, Any]
    """The full, unmodified catalog row."""

    @property
    def explanation(self) -> str:
        return EXPLANATION_SEPARATOR.join(self.explanations)


@dataclass(frozen=True)
class WorkflowResult:
    request: str
    planning: PlanningResult
    strategy: ScoringStrategy
    recommender_kwargs: Mapping[str, Any]
    catalog_size: int
    candidate_count: int
    songs: Tuple[RecommendedSong, ...]
    requirements: PlaylistRequirements
    playlist: Playlist
    validation: ValidationReport

    @property
    def plan(self) -> RecommendationPlan:
        return self.planning.plan

    @property
    def status(self) -> str:
        """passed | infeasible | failed | unsupported (from the validation report)."""
        return self.validation.outcome

    @property
    def fully_satisfied(self) -> bool:
        """True only when every requirement was checked and passed."""
        return self.validation.passed

    def as_recommendation_tuples(self) -> List[Tuple[Dict[str, Any], float, str]]:
        """Same (song, score, explanation) shape recommend_songs returns."""
        return [(dict(song.record), song.score, song.explanation) for song in self.songs]


def run_recommendation_workflow(
    user_request: str,
    saved_preferences: Optional[Mapping[str, Any]] = None,
    *,
    client: Optional[LLMClient] = None,
    catalog_path: str = CATALOG_PATH,
    songs: Optional[Sequence[Dict[str, Any]]] = None,
) -> WorkflowResult:
    """Turn a natural-language request into a validated playlist of catalog songs.

    Raises ``WorkflowError`` (with ``stage``) when any stage fails, and
    ``ValueError`` for an invalid request or invalid saved preferences.
    """
    catalog = _load_catalog(catalog_path) if songs is None else list(songs)
    if not catalog:
        raise WorkflowError("catalog", "the song catalog is empty")

    # 1. Plan: natural language -> validated plan -> recommender arguments.
    planning = _plan_request(user_request, saved_preferences, client, catalog)
    plan = planning.plan
    _recheck_plan(plan, catalog)
    kwargs = to_recommender_kwargs(plan)
    mode = _resolve_mode(kwargs)
    strategy = SCORING_STRATEGIES[mode]
    requirements = PlaylistRequirements.from_plan(
        plan,
        default_size=RECOMMENDER_DEFAULT_K,
        target_duration_minutes=planning.requested_duration_minutes,
        unsupported_requests=[f"{u.request} ({u.reason})" for u in planning.unsupported_requests],
    )

    # 2. Rank: hard constraints narrow the songs; the original recommender ranks them.
    candidates = _eligible_songs(catalog, plan)
    ranked = _rank_all(candidates, kwargs)

    # 3. Build and validate the playlist.
    try:
        playlist = build_playlist(ranked, requirements)
    except PlaylistInputError as error:
        raise WorkflowError("playlist", str(error), error) from error
    validation = validate_playlist(playlist, requirements, catalog)

    songs_out = tuple(
        _to_recommended_song(entry, kwargs["user_prefs"], mode) for entry in playlist.entries
    )
    logger.info(
        "Workflow complete: strategy=%s candidates=%d/%d playlist=%d/%d outcome=%s",
        strategy.name, len(candidates), len(catalog), len(songs_out), requirements.size,
        validation.outcome,
    )
    return WorkflowResult(
        request=user_request,
        planning=planning,
        strategy=strategy,
        recommender_kwargs=MappingProxyType(kwargs),
        catalog_size=len(catalog),
        candidate_count=len(candidates),
        songs=songs_out,
        requirements=requirements,
        playlist=playlist,
        validation=validation,
    )


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _load_catalog(path: str) -> List[Dict[str, Any]]:
    try:
        return load_songs(path)
    except FileNotFoundError as error:
        raise WorkflowError("catalog", f"song catalog not found at {path!r}", error) from error
    except (KeyError, ValueError) as error:
        raise WorkflowError("catalog", f"song catalog at {path!r} is malformed: {error}", error) from error


def _plan_request(user_request, saved_preferences, client, catalog) -> PlanningResult:
    try:
        return create_recommendation_plan(
            user_request, saved_preferences, client=client, catalog=catalog,
            check_capacity=False,  # capacity problems are reported by the playlist validator
        )
    except PlanningError as error:
        logger.warning("Planning failed: %s", type(error).__name__)
        raise WorkflowError("planning", str(error), error) from error


def _eligible_songs(catalog: Sequence[Dict[str, Any]], plan: RecommendationPlan) -> List[Dict[str, Any]]:
    """Catalog songs that satisfy every hard constraint."""
    return [
        song for song in catalog
        if all(constraint.is_satisfied_by(song) for constraint in plan.hard_constraints)
    ]


def _rank_all(candidates: List[Dict[str, Any]], kwargs: Mapping[str, Any]):
    """Rank every eligible song with the original recommender.

    recommend_songs is greedy, so its first k picks equal a call with k; the
    playlist builder walks this full ranking.
    """
    try:
        return recommend_songs(songs=candidates, **{**kwargs, "k": len(candidates)})
    except (KeyError, TypeError, ValueError) as error:
        raise WorkflowError("recommendation", f"recommender error: {error}", error) from error


def _recheck_plan(plan: RecommendationPlan, catalog: Sequence[Mapping[str, Any]]) -> None:
    """Defensive re-validation: the plan must still round-trip through the schema."""
    data = {
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
    result = validate_plan(data, catalog=catalog, check_capacity=False)
    if not result.is_valid:
        details = "; ".join(str(issue) for issue in result.errors)
        raise WorkflowError("planning", f"plan failed re-validation: {details}")


def _resolve_mode(kwargs: Mapping[str, Any]) -> str:
    """The scoring mode recommend_songs will use (its own default if unset)."""
    mode = kwargs.get("mode", RECOMMENDER_DEFAULT_MODE)
    if mode not in SCORING_STRATEGIES:
        raise WorkflowError("planning", f"plan selected unsupported scoring strategy {mode!r}")
    return mode


def _to_recommended_song(entry, user_prefs: Mapping[str, Any], mode: str) -> RecommendedSong:
    song = entry.song
    base_score, _ = score_song(dict(user_prefs), dict(song), mode=mode)
    return RecommendedSong(
        position=entry.position,
        rank=entry.rank,
        id=song["id"],
        title=song["title"],
        artist=song["artist"],
        genre=song["genre"],
        mood=song["mood"],
        energy=song["energy"],
        score=entry.score,
        base_score=base_score,
        explanations=tuple(entry.explanation.split(EXPLANATION_SEPARATOR)),
        record=song,
    )
