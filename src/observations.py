"""
Structured observations and execution traces for the agentic loop (Phase 5).

An ``Observation`` is what the orchestrator tells the LLM (and the caller)
after an attempt fails. It is plain, machine-readable data:

* ``category``  one of ``FAILURE_CATEGORIES`` (below)
* ``retryable`` whether a revised plan could possibly fix it
* ``reasons``   one ``FailureReason`` per problem, each with a stable ``code``

Failure categories
------------------
invalid_plan             The LLM output was malformed, failed schema
                         validation, or tried to change a locked field.
                         Retryable (the LLM may produce a valid plan).
constraints_unsatisfied  The playlist broke a requirement that some other
                         choice of soft preferences / strategy might meet
                         (e.g. total duration outside the allowed window).
                         Retryable.
unsupported_metadata     The request needs data the catalog does not have
                         (an unsupported request, or duration without
                         ``duration_seconds``). Not retryable: no revision
                         can add metadata.
impossible_request       A requirement is provably unsatisfiable by ANY
                         playlist from the songs that satisfy the immutable
                         hard constraints (validator ``infeasible`` flag).
                         Not retryable.
api_failure              The LLM provider failed (unreachable, rate limited,
                         misconfigured, unexpected error). Not retried.
tool_failure             A deterministic tool (recommender, playlist builder)
                         failed or produced invalid output. Not retried.

Final statuses (``ExecutionTrace.final_status``)
-----------------------------------------------
succeeded, retry_budget_exhausted, repeated_plan, no_revisable_fields,
impossible_request, unsupported_metadata, api_failure, tool_failure,
invalid_plan (no valid plan was ever produced).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

INVALID_PLAN = "invalid_plan"
CONSTRAINTS_UNSATISFIED = "constraints_unsatisfied"
UNSUPPORTED_METADATA = "unsupported_metadata"
IMPOSSIBLE_REQUEST = "impossible_request"
API_FAILURE = "api_failure"
TOOL_FAILURE = "tool_failure"
FAILURE_CATEGORIES = (
    INVALID_PLAN, CONSTRAINTS_UNSATISFIED, UNSUPPORTED_METADATA,
    IMPOSSIBLE_REQUEST, API_FAILURE, TOOL_FAILURE,
)
RETRYABLE_CATEGORIES = frozenset({INVALID_PLAN, CONSTRAINTS_UNSATISFIED})

SUCCEEDED = "succeeded"
RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
REPEATED_PLAN = "repeated_plan"
NO_REVISABLE_FIELDS = "no_revisable_fields"
FINAL_STATUSES = (
    SUCCEEDED, RETRY_BUDGET_EXHAUSTED, REPEATED_PLAN, NO_REVISABLE_FIELDS,
    IMPOSSIBLE_REQUEST, UNSUPPORTED_METADATA, API_FAILURE, TOOL_FAILURE, INVALID_PLAN,
)

# Validator checks that only fail when a deterministic tool misbehaved.
_TOOL_INTEGRITY_CHECKS = {
    "song_metadata": "song_metadata_invalid",
    "catalog_membership": "song_not_in_catalog",
    "no_duplicate_songs": "duplicate_song",
}


def _plain(value: Any) -> Any:
    """Convert mappings/tuples (incl. MappingProxyType) into JSON-friendly data."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class FailureReason:
    code: str
    """Stable machine-readable code, e.g. ``duration_out_of_range``."""
    message: str
    check: Optional[str] = None
    """Validator check name, when the reason comes from validation."""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "check": self.check, "message": self.message, "details": _plain(self.details)}


@dataclass(frozen=True)
class Observation:
    category: str
    retryable: bool
    reasons: Tuple[FailureReason, ...]

    def __post_init__(self) -> None:
        if self.category not in FAILURE_CATEGORIES:
            raise ValueError(f"unknown failure category {self.category!r}")

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(reason.code for reason in self.reasons)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "retryable": self.retryable,
            "reasons": [reason.to_dict() for reason in self.reasons],
        }


def _observation(category: str, reasons: Sequence[FailureReason]) -> Observation:
    return Observation(category, category in RETRYABLE_CATEGORIES, tuple(reasons))


# --------------------------------------------------------------------------
# Building observations
# --------------------------------------------------------------------------


def _check_code(check) -> str:
    """Reason code for one non-passing validator check."""
    name = check.name
    if check.status == "unsupported":
        return "unsupported_request" if name.startswith("unsupported_request") else "missing_catalog_metadata"
    if name in _TOOL_INTEGRITY_CHECKS:
        return _TOOL_INTEGRITY_CHECKS[name]
    base = name.split(":", 1)[0]
    if check.infeasible:
        return f"{base}_infeasible"
    return {
        "duration": "duration_out_of_range",
        "playlist_size": "playlist_size_short",
        "energy_progression": "energy_progression_violated",
        "unique_artists": "unique_artists_violated",
        "hard_constraint": "hard_constraint_violated",
    }.get(base, f"{base}_violated")


def observe_validation(report) -> Optional[Observation]:
    """Observation for a ``ValidationReport``; ``None`` when every check passed.

    Category precedence: impossible_request > unsupported_metadata >
    tool_failure > constraints_unsatisfied. Only the last is retryable.
    """
    failing = [check for check in report.checks if check.status != "passed"]
    if not failing:
        return None
    reasons = [
        FailureReason(
            _check_code(check), check.message, check.name,
            {"status": check.status, "requirement": check.requirement,
             "infeasible": check.infeasible, "evidence": check.evidence},
        )
        for check in failing
    ]
    if any(check.status == "violated" and check.infeasible for check in failing):
        category = IMPOSSIBLE_REQUEST
    elif any(check.status == "unsupported" for check in failing):
        category = UNSUPPORTED_METADATA
    elif any(check.name in _TOOL_INTEGRITY_CHECKS for check in failing):
        category = TOOL_FAILURE
    else:
        category = CONSTRAINTS_UNSATISFIED
    return _observation(category, reasons)


def observe_planning_error(error: BaseException) -> Observation:
    """Observation for a planning/revision failure (invalid plan or API failure)."""
    # Imported lazily: agent.py imports this module.
    from .agent import (
        ImmutableFieldError, LLMAPIError, LLMConfigurationError, LLMRateLimitError,
        LLMResponseError, LLMUnavailableError, PlanRejectedError,
    )

    if isinstance(error, LLMResponseError):
        return _observation(INVALID_PLAN, [FailureReason(
            "llm_output_invalid", "the LLM output was malformed or broke the plan rules",
            details={"problems": list(error.problems)})])
    if isinstance(error, PlanRejectedError):
        return _observation(INVALID_PLAN, [FailureReason(
            "plan_schema_invalid", "the plan failed schema validation",
            details={"errors": [str(issue) for issue in error.errors]})])
    if isinstance(error, ImmutableFieldError):
        return _observation(INVALID_PLAN, [FailureReason(
            "immutable_field_changed", str(error), details={"fields": list(error.fields)})])
    if isinstance(error, LLMRateLimitError):
        code = "llm_rate_limited"
    elif isinstance(error, LLMUnavailableError):
        code = "llm_unavailable"
    elif isinstance(error, LLMConfigurationError):
        code = "llm_configuration_error"
    elif isinstance(error, LLMAPIError):
        code = "llm_api_error"
    else:
        code = "planning_error"
    return _observation(API_FAILURE, [FailureReason(code, str(error), details={"error_type": type(error).__name__})])


def observe_invalid_plan(message: str) -> Observation:
    """A plan that passed the agent but failed the orchestrator's own re-checks."""
    return _observation(INVALID_PLAN, [FailureReason("plan_recheck_failed", message)])


def observe_tool_error(stage: str, error: BaseException) -> Observation:
    return _observation(TOOL_FAILURE, [FailureReason(
        f"{stage}_tool_error", str(error), details={"stage": stage, "error_type": type(error).__name__})])


# --------------------------------------------------------------------------
# Execution trace
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    tool: str
    summary: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "summary": _plain(self.summary)}


@dataclass(frozen=True)
class AttemptRecord:
    number: int
    kind: str
    """``initial`` (create a plan) or ``revision`` (revise the last valid plan)."""
    outcome: str
    """``passed``, a failure category, or ``repeated_plan``."""
    plan: Optional[Mapping[str, Any]] = None
    """Snapshot of the plan this attempt produced (None if no valid plan)."""
    plan_fingerprint: Optional[str] = None
    rationale: Optional[str] = None
    """The LLM's stated reason for a revision."""
    changed_fields: Tuple[str, ...] = ()
    tool_calls: Tuple[ToolCall, ...] = ()
    validation: Optional[Mapping[str, Any]] = None
    """``ValidationReport.to_dict()`` when the playlist was built and validated."""
    observation: Optional[Observation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "kind": self.kind,
            "outcome": self.outcome,
            "plan": _plain(self.plan),
            "plan_fingerprint": self.plan_fingerprint,
            "rationale": self.rationale,
            "changed_fields": list(self.changed_fields),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "validation": _plain(self.validation),
            "observation": self.observation.to_dict() if self.observation else None,
        }


@dataclass(frozen=True)
class ExecutionTrace:
    max_attempts: int
    attempts: Tuple[AttemptRecord, ...]
    final_status: str
    stop_reason: str
    result_attempt: Optional[int] = None
    """Which attempt's playlist is returned (None if none was built)."""

    @property
    def initial_plan(self) -> Optional[Mapping[str, Any]]:
        for attempt in self.attempts:
            if attempt.plan is not None:
                return attempt.plan
        return None

    @property
    def revised_plans(self) -> List[Mapping[str, Any]]:
        return [a.plan for a in self.attempts if a.kind == "revision" and a.plan is not None]

    @property
    def succeeded(self) -> bool:
        return self.final_status == SUCCEEDED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "attempts_used": len(self.attempts),
            "final_status": self.final_status,
            "stop_reason": self.stop_reason,
            "result_attempt": self.result_attempt,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


__all__ = [
    "API_FAILURE", "AttemptRecord", "CONSTRAINTS_UNSATISFIED", "ExecutionTrace", "FAILURE_CATEGORIES",
    "FINAL_STATUSES", "FailureReason", "IMPOSSIBLE_REQUEST", "INVALID_PLAN", "NO_REVISABLE_FIELDS",
    "Observation", "REPEATED_PLAN", "RETRYABLE_CATEGORIES", "RETRY_BUDGET_EXHAUSTED", "SUCCEEDED",
    "TOOL_FAILURE", "ToolCall", "UNSUPPORTED_METADATA", "observe_invalid_plan", "observe_planning_error",
    "observe_tool_error", "observe_validation",
]
