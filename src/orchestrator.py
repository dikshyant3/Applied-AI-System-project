"""
TuneWise workflow orchestrator (Phases 3-5).

natural-language request
  -> load the real catalog            (recommender.load_songs)
  -> LLM planning agent               (agent.create_recommendation_plan)
  -> validated RecommendationPlan     (schemas, re-checked here)
  -> recommender arguments            (schemas.to_recommender_kwargs)
  -> hard-constraint candidate filter (HardConstraint.is_satisfied_by)
  -> full ranking + explanations      (recommender.recommend_songs / score_song)
  -> playlist selection and ordering  (playlist_builder.build_playlist)
  -> deterministic validation report  (playlist_validator.validate_playlist)

Optional RAG (``retriever=``, see retrieval.py): before planning, the songs
whose documents are most similar to the request are summarized for the planner
as ``catalog_matches``; after the hard-constraint filter, only the top-k most
similar eligible songs are ranked (falling back to all eligible songs if that
pool is smaller than the playlist). The loop itself is unchanged.

The orchestrator never scores songs itself. Requests that the catalog cannot
satisfy are not errors: they return a playlist plus a report whose outcome is
"infeasible" (or "unsupported"), and no requirement is relaxed.

Agentic loop (Phase 5): Plan -> Act -> Observe -> Revise
--------------------------------------------------------
Each attempt (at most ``max_attempts``, default 3, counted as LLM plan calls):

1. Plan    attempt 1: agent.create_recommendation_plan; later attempts:
           agent.revise_recommendation_plan with the last structured observation
           (or create_recommendation_plan with feedback, if no valid plan exists yet).
2. Check   schema re-validation; a plan identical to an earlier one stops the
           loop ("repeated_plan") instead of re-running the same tools.
3. Act     filter -> rank -> build -> validate (the Phase 3-4 pipeline).
4. Observe observations.observe_validation turns the report into a category:
           passed -> stop (succeeded)
           constraints_unsatisfied / invalid_plan -> revise if budget remains
           impossible_request / unsupported_metadata / api_failure /
           tool_failure -> stop immediately (revising cannot help)

Hard constraints, playlist rules, requested duration, unsupported requests and
explicit preferences are locked: revisions are copied from the previous plan
by code and re-verified. Every attempt is recorded in ``ExecutionTrace``.
The returned playlist is always the last one actually built and validated;
success is reported only when its validation passed.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import agent as _agent
from .agent import (
    RECOMMENDER_DEFAULT_K, LLMClient, PlanningError, PlanningResult, check_request_inputs,
    create_recommendation_plan, plan_snapshot, revisable_fields, revise_recommendation_plan,
)
from .observations import (
    INVALID_PLAN, NO_REVISABLE_FIELDS, REPEATED_PLAN, RETRY_BUDGET_EXHAUSTED, SUCCEEDED,
    AttemptRecord, ExecutionTrace, Observation, ToolCall, observe_invalid_plan,
    observe_planning_error, observe_tool_error, observe_validation,
)
from .playlist_builder import Playlist, PlaylistInputError, PlaylistRequirements, build_playlist
from .playlist_validator import ValidationReport, validate_playlist
from .recommender import SCORING_STRATEGIES, ScoringStrategy, load_songs, recommend_songs, score_song
from .schemas import RecommendationPlan, plan_to_data, to_recommender_kwargs, validate_plan
from .workflow_logging import log_workflow_event

if TYPE_CHECKING:  # retrieval needs numpy; only import it when RAG is used
    from .retrieval import SongRetriever

logger = logging.getLogger("tunewise.orchestrator")

CATALOG_PATH = "data/songs.csv"
EXPLANATION_SEPARATOR = "; "  # how recommend_songs joins its reasons
RECOMMENDER_DEFAULT_MODE = inspect.signature(recommend_songs).parameters["mode"].default
DEFAULT_MAX_ATTEMPTS = 3
MAX_ATTEMPTS_LIMIT = 10  # hard ceiling: the loop can never run unbounded
DEFAULT_RETRIEVAL_K = 60  # eligible songs passed to the recommender when RAG is on


class WorkflowError(Exception):
    """A workflow stage failed. ``stage`` is catalog, planning, recommendation or playlist.

    ``trace`` holds the agentic execution trace when the loop had started.
    """

    def __init__(self, stage: str, message: str, cause: Optional[BaseException] = None,
                 trace: Optional[ExecutionTrace] = None):
        self.stage = stage
        self.cause = cause
        self.trace = trace
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
    trace: Optional[ExecutionTrace] = None
    retrieval: Optional[Mapping[str, Any]] = None
    """RAG summary (embedder, catalog matches, pool size, fallback); None without RAG."""

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

    @property
    def final_status(self) -> str:
        """Agentic-loop outcome (see observations.FINAL_STATUSES)."""
        if self.trace is not None:
            return self.trace.final_status
        return SUCCEEDED if self.validation.passed else self.validation.outcome

    @property
    def attempts_used(self) -> int:
        return len(self.trace.attempts) if self.trace is not None else 1

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
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    request_id: Optional[str] = None,
    retriever: Optional["SongRetriever"] = None,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
) -> WorkflowResult:
    """Turn a natural-language request into a validated playlist of catalog songs.

    Runs the Plan -> Act -> Observe -> Revise loop for at most ``max_attempts``
    plan attempts. Returns the last playlist actually built and validated, with
    ``result.trace`` describing every attempt and ``result.final_status``.

    Raises ``WorkflowError`` (with ``stage`` and ``trace``) when no playlist
    could be built at all, and ``ValueError`` for an invalid request, invalid
    saved preferences or an invalid ``max_attempts``.

    With ``retriever`` (a ``retrieval.SongRetriever`` built over the same
    catalog) the workflow is retrieval-augmented; see the module docstring.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    elif not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    _check_max_attempts(max_attempts)
    check_request_inputs(user_request, saved_preferences)
    catalog = _load_catalog(catalog_path) if songs is None else list(songs)
    if not catalog:
        raise WorkflowError("catalog", "the song catalog is empty")
    started_at = time.perf_counter()
    log_workflow_event(
        request_id, "workflow_started", max_attempts=max_attempts,
        catalog_size=len(catalog),
    )
    try:
        llm = client if client is not None else _agent.create_llm_client()
    except PlanningError as error:
        log_workflow_event(
            request_id, "workflow_finished", final_status="api_failure",
            attempts_used=0, result_attempt=None,
            duration_seconds=round(time.perf_counter() - started_at, 6),
        )
        raise WorkflowError("planning", str(error), error) from error

    retrieval = None
    if retriever is not None:
        if isinstance(retrieval_k, bool) or not isinstance(retrieval_k, int) or retrieval_k < 1:
            raise ValueError(f"retrieval_k must be a positive integer, got {retrieval_k!r}")
        retrieval = _Retrieval.start(retriever, user_request, retrieval_k, request_id)

    loop = _AgentLoop(
        user_request, saved_preferences, llm, catalog, max_attempts, request_id, started_at,
        retrieval,
    )
    return loop.run()


@dataclass(frozen=True)
class _Retrieval:
    """RAG state for one request: the embedded query and the planner's catalog matches."""

    retriever: "SongRetriever"
    query_vector: Any
    k: int
    catalog_matches: Tuple[Mapping[str, Any], ...]

    @classmethod
    def start(cls, retriever: "SongRetriever", request: str, k: int, request_id: str) -> "_Retrieval":
        vector = retriever.query_vector(request)
        hits = retriever.search(vector, k=k)
        matches = tuple(retriever.planner_context(hits))
        log_workflow_event(
            request_id, "tool_executed", attempt=0, tool="retrieve_catalog_matches",
            embedder=retriever.embedder.name, retrieved=len(hits),
            top_similarity=round(hits[0].score, 4) if hits else None,
        )
        return cls(retriever, vector, k, matches)

    def narrow(self, eligible: List[Dict[str, Any]], needed: int):
        """Top-k eligible songs by similarity, or every eligible song if that is too few."""
        hits = self.retriever.search(self.query_vector, k=max(self.k, 4 * needed), among=eligible)
        if len(hits) < needed:
            return eligible, {}, True
        return [hit.song for hit in hits], {hit.song["id"]: hit for hit in hits}, False

    def summary(self, pool: int, fallback: bool) -> Dict[str, Any]:
        return {"embedder": self.retriever.embedder.name, "k": self.k, "pool": pool,
                "fallback_to_all_eligible": fallback, "catalog_matches": list(self.catalog_matches)}


def _check_max_attempts(max_attempts: Any) -> None:
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ValueError(f"max_attempts must be an integer, got {max_attempts!r}")
    if not 1 <= max_attempts <= MAX_ATTEMPTS_LIMIT:
        raise ValueError(f"max_attempts must be between 1 and {MAX_ATTEMPTS_LIMIT}, got {max_attempts}")


@dataclass(frozen=True)
class _Execution:
    """One run of the deterministic tools for one plan."""

    planning: PlanningResult
    strategy: ScoringStrategy
    kwargs: Dict[str, Any]
    candidates: int
    songs: Tuple[RecommendedSong, ...]
    requirements: PlaylistRequirements
    playlist: Playlist
    validation: ValidationReport
    tool_calls: Tuple[ToolCall, ...]
    retrieval: Optional[Dict[str, Any]] = None


class _AgentLoop:
    """Bounded Plan -> Act -> Observe -> Revise loop (see the module docstring)."""

    def __init__(self, request, saved, client, catalog, max_attempts, request_id, started_at,
                 retrieval: Optional[_Retrieval] = None):
        self.retrieval = retrieval
        self.request = request
        self.saved = saved
        self.client = client
        self.catalog = catalog
        self.max_attempts = max_attempts
        self.request_id = request_id
        self.started_at = started_at
        self.attempts: List[AttemptRecord] = []
        self.planning: Optional[PlanningResult] = None  # last valid plan
        self.execution: Optional[_Execution] = None  # last executed attempt
        self.result_attempt: Optional[int] = None
        self.observation: Optional[Observation] = None  # last failure, fed to the LLM
        self.seen: Dict[str, int] = {}  # plan fingerprint -> attempt number
        self.last_error: Optional[BaseException] = None
        self.error_stage = "planning"

    # ---- main loop -------------------------------------------------------

    def run(self) -> WorkflowResult:
        final_status, stop_reason = RETRY_BUDGET_EXHAUSTED, ""
        for number in range(1, self.max_attempts + 1):  # bounded: never loops forever
            stop = self._attempt(number)
            if stop is not None:
                final_status, stop_reason = stop
                break
        else:
            stop_reason = f"retry budget of {self.max_attempts} attempt(s) exhausted without a valid playlist"
            if self.execution is None:
                final_status = INVALID_PLAN
                stop_reason = f"no valid plan was produced in {self.max_attempts} attempt(s)"

        trace = ExecutionTrace(self.max_attempts, tuple(self.attempts), final_status, stop_reason,
                               self.result_attempt)
        logger.info("Agent loop finished: status=%s attempts=%d/%d reason=%s",
                    final_status, len(self.attempts), self.max_attempts, stop_reason)
        log_workflow_event(
            self.request_id, "workflow_finished", final_status=final_status,
            attempts_used=len(self.attempts), result_attempt=self.result_attempt,
            duration_seconds=round(time.perf_counter() - self.started_at, 6),
        )
        if self.execution is None:
            message = str(self.last_error) if self.last_error else stop_reason
            raise WorkflowError(self.error_stage, message, self.last_error, trace)
        return self._result(trace)

    def _attempt(self, number: int) -> Optional[Tuple[str, str]]:
        """Run one attempt. Returns (final_status, reason) to stop, or None to continue."""
        kind = "initial" if self.planning is None else "revision"
        logger.info("Attempt %d/%d (%s)", number, self.max_attempts, kind)
        log_workflow_event(
            self.request_id, "attempt_started", attempt=number, kind=kind,
            max_attempts=self.max_attempts,
        )

        # 1. Plan (or revise).
        try:
            candidate = self._plan(number)
            _recheck_plan(candidate.plan, self.catalog)
            kwargs = to_recommender_kwargs(candidate.plan)
            mode = _resolve_mode(kwargs)
        except PlanningError as error:
            return self._planning_failed(number, kind, error, observe_planning_error(error))
        except WorkflowError as error:  # re-check / strategy failures: the plan is invalid
            return self._planning_failed(number, kind, error, observe_invalid_plan(str(error)))

        log_workflow_event(
            self.request_id, "initial_plan_validated" if self.planning is None
            else "revised_plan_validated", attempt=number,
            strategy=mode, plan=_safe_plan_summary(candidate),
        )
        snapshot = plan_snapshot(candidate)
        fingerprint = _fingerprint(candidate, kwargs)
        if fingerprint in self.seen:
            earlier = self.seen[fingerprint]
            self._record(number, kind, REPEATED_PLAN, candidate, snapshot, fingerprint)
            return REPEATED_PLAN, (f"attempt {number} repeated the plan of attempt {earlier}; "
                                   "re-running identical tools cannot change the result")
        self.seen[fingerprint] = number
        self.planning = candidate

        # 2. Act.
        try:
            execution = _execute_plan(
                candidate, kwargs, mode, self.catalog, self.request_id, number, self.retrieval,
            )
        except WorkflowError as error:
            observation = observe_tool_error(error.stage, error)
            self.last_error, self.error_stage = error, error.stage
            log_workflow_event(
                self.request_id, "tool_execution_failed", attempt=number,
                stage=error.stage, error_type=type(error.cause or error).__name__,
            )
            self._record(number, kind, observation.category, candidate, snapshot, fingerprint,
                         observation=observation)
            return observation.category, f"tool failure in {error.stage}: {error}"
        self.execution, self.result_attempt = execution, number

        # 3. Observe.
        observation = observe_validation(execution.validation)
        outcome = "passed" if observation is None else observation.category
        log_workflow_event(
            self.request_id, "validation_result", attempt=number,
            outcome=execution.validation.outcome,
            checks=[{"name": check.name, "status": check.status,
                     "infeasible": check.infeasible} for check in execution.validation.checks],
        )
        self._record(number, kind, outcome, candidate, snapshot, fingerprint,
                     execution=execution, observation=observation)
        if observation is None:
            return SUCCEEDED, f"playlist passed validation on attempt {number}"
        if not observation.retryable:
            return observation.category, (f"{observation.category} ({', '.join(observation.codes)}); "
                                          "revising soft preferences cannot fix this")
        if not revisable_fields(candidate):
            return NO_REVISABLE_FIELDS, "every plan field is locked, so no revision is possible"
        self.observation = observation
        log_workflow_event(
            self.request_id, "retry_scheduled", attempt=number,
            next_attempt=number + 1, failure_codes=list(observation.codes),
        )
        return None

    def _plan(self, number: int) -> PlanningResult:
        if self.planning is None:
            feedback = self.observation.to_dict() if self.observation is not None else None
            return create_recommendation_plan(
                self.request, self.saved, client=self.client, catalog=self.catalog,
                check_capacity=False,  # capacity problems are reported by the playlist validator
                feedback=feedback,
                catalog_matches=self.retrieval.catalog_matches if self.retrieval else None,
            )
        assert self.observation is not None
        return revise_recommendation_plan(
            self.request, self.planning, self.observation, client=self.client,
            catalog=self.catalog, history=self._history(), attempt=number,
        )

    def _planning_failed(self, number, kind, error, observation) -> Optional[Tuple[str, str]]:
        logger.warning("Attempt %d: %s (%s)", number, observation.category, type(error).__name__)
        self.last_error, self.error_stage = error, "planning"
        self._record(number, kind, observation.category, observation=observation)
        log_workflow_event(
            self.request_id, "plan_rejected", attempt=number,
            category=observation.category, failure_codes=list(observation.codes),
        )
        if not observation.retryable:
            return observation.category, f"{observation.category}: {error}"
        self.observation = observation
        log_workflow_event(
            self.request_id, "retry_scheduled", attempt=number,
            next_attempt=number + 1, failure_codes=list(observation.codes),
        )
        return None

    # ---- bookkeeping -----------------------------------------------------

    def _history(self) -> List[Dict[str, Any]]:
        """Earlier attempts, summarised for the revision prompt."""
        return [
            {
                "attempt": a.number,
                "soft_settings": None if a.plan is None else {
                    "preferences": a.plan["preferences"], "strategy": a.plan["strategy"]},
                "outcome": a.outcome,
                "failure_codes": list(a.observation.codes) if a.observation else [],
            }
            for a in self.attempts
        ]

    def _record(self, number, kind, outcome, planning=None, snapshot=None, fingerprint=None,
                execution: Optional[_Execution] = None, observation=None) -> None:
        self.attempts.append(AttemptRecord(
            number=number,
            kind=kind,
            outcome=outcome,
            plan=snapshot,
            plan_fingerprint=fingerprint,
            rationale=planning.rationale if planning is not None else None,
            changed_fields=planning.revised_fields if planning is not None else (),
            tool_calls=execution.tool_calls if execution is not None else (),
            validation=execution.validation.to_dict() if execution is not None else None,
            observation=observation,
        ))

    def _result(self, trace: ExecutionTrace) -> WorkflowResult:
        execution = self.execution
        assert execution is not None
        return WorkflowResult(
            request=self.request,
            planning=execution.planning,
            strategy=execution.strategy,
            recommender_kwargs=MappingProxyType(execution.kwargs),
            catalog_size=len(self.catalog),
            candidate_count=execution.candidates,
            songs=execution.songs,
            requirements=execution.requirements,
            playlist=execution.playlist,
            validation=execution.validation,
            trace=trace,
            retrieval=execution.retrieval,
        )


def _fingerprint(planning: PlanningResult, kwargs: Mapping[str, Any]) -> str:
    """Everything that determines tool output; equal fingerprints mean equal playlists."""
    return json.dumps({
        "recommender": kwargs,
        "plan": plan_to_data(planning.plan)["playlist"],
        "hard_constraints": plan_to_data(planning.plan)["hard_constraints"],
        "duration": planning.requested_duration_minutes,
    }, sort_keys=True, default=str)


def _safe_plan_summary(planning: PlanningResult) -> Dict[str, Any]:
    """Plan fields needed for debugging, excluding request text and evidence quotes."""
    data = plan_to_data(planning.plan)
    return {
        "preferences": {
            name: {"value": preference.value, "source": preference.source}
            for name, preference in planning.plan.preferences.items()
        },
        "strategy": data.get("strategy"),
        "playlist": data["playlist"],
        "hard_constraints": data["hard_constraints"],
        "target_duration_minutes": planning.requested_duration_minutes,
        "unsupported_request_count": len(planning.unsupported_requests),
    }


def _execute_plan(planning: PlanningResult, kwargs: Dict[str, Any], mode: str,
                  catalog: Sequence[Dict[str, Any]], request_id: str,
                  attempt: int, retrieval: Optional[_Retrieval] = None) -> _Execution:
    """Act: filter -> rank -> build -> validate, recording each tool call."""
    plan = planning.plan
    strategy = SCORING_STRATEGIES[mode]
    requirements = PlaylistRequirements.from_plan(
        plan,
        default_size=RECOMMENDER_DEFAULT_K,
        target_duration_minutes=planning.requested_duration_minutes,
        unsupported_requests=[f"{u.request} ({u.reason})" for u in planning.unsupported_requests],
    )

    # Rank: hard constraints narrow the songs; the original recommender ranks them.
    candidates = _eligible_songs(catalog, plan)
    log_workflow_event(
        request_id, "tool_executed", attempt=attempt, tool="filter_hard_constraints",
        eligible_songs=len(candidates), catalog_size=len(catalog),
    )
    eligible_count = len(candidates)
    hits: Dict[Any, Any] = {}
    retrieval_summary = None
    if retrieval is not None:
        candidates, hits, fallback = retrieval.narrow(candidates, requirements.size)
        retrieval_summary = retrieval.summary(len(candidates), fallback)
        log_workflow_event(
            request_id, "tool_executed", attempt=attempt, tool="retrieve_candidates",
            eligible_songs=eligible_count, pool=len(candidates), fallback=fallback,
        )
    ranked = _rank_all(candidates, kwargs)
    log_workflow_event(
        request_id, "tool_executed", attempt=attempt, tool="recommend_songs",
        strategy=mode, ranked_candidate_count=len(ranked),
    )

    # Build and validate the playlist.
    try:
        playlist = build_playlist(ranked, requirements)
    except PlaylistInputError as error:
        raise WorkflowError("playlist", str(error), error) from error
    log_workflow_event(
        request_id, "tool_executed", attempt=attempt, tool="build_playlist",
        selected_count=len(playlist.entries), selection_mode=playlist.selection_mode,
    )
    validation = validate_playlist(playlist, requirements, catalog)
    log_workflow_event(
        request_id, "tool_executed", attempt=attempt, tool="validate_playlist",
        outcome=validation.outcome, check_count=len(validation.checks),
    )

    songs_out = tuple(
        _to_recommended_song(entry, kwargs["user_prefs"], mode, hits.get(entry.song["id"]))
        for entry in playlist.entries
    )
    tool_calls = (
        ToolCall("filter_hard_constraints", {
            "hard_constraints": len(plan.hard_constraints),
            "eligible_songs": eligible_count, "catalog_size": len(catalog)}),
    ) + ((ToolCall("retrieve_candidates", {
            "pool": len(candidates),
            "fallback_to_all_eligible": retrieval_summary["fallback_to_all_eligible"]}),)
         if retrieval_summary else ()) + (
        ToolCall("recommend_songs", {"mode": mode, "ranked_songs": len(ranked)}),
        ToolCall("build_playlist", {
            "selected_ids": [song.id for song in songs_out],
            "selection_mode": playlist.selection_mode,
            "requested_size": requirements.size,
            "skipped": len(playlist.skipped)}),
        ToolCall("validate_playlist", {
            "outcome": validation.outcome,
            "violated": [c.name for c in validation.violations],
            "unsupported": [c.name for c in validation.unsupported]}),
    )
    logger.info(
        "Tools complete: strategy=%s candidates=%d/%d playlist=%d/%d outcome=%s",
        strategy.name, len(candidates), len(catalog), len(songs_out), requirements.size,
        validation.outcome,
    )
    return _Execution(planning, strategy, kwargs, eligible_count, songs_out, requirements,
                      playlist, validation, tool_calls, retrieval_summary)


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
    data = plan_to_data(plan)
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


def _to_recommended_song(entry, user_prefs: Mapping[str, Any], mode: str, hit=None) -> RecommendedSong:
    song = entry.song
    base_score, _ = score_song(dict(user_prefs), dict(song), mode=mode)
    explanations = tuple(entry.explanation.split(EXPLANATION_SEPARATOR))
    if hit is not None:
        explanations += (f"semantic match to request {hit.score:.2f} (retrieval rank {hit.rank})",)
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
        explanations=explanations,
        record=song,
    )
