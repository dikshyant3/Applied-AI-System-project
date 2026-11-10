"""Phase 5: the agentic Plan -> Act -> Observe -> Revise loop.

Every test is deterministic: the LLM is a ``MockLLMClient`` with scripted
responses, and the catalog is the real ``data/songs.csv``.

Reference scenario (real catalog numbers, see test_success_after_revision):
"5 calm lofi songs ... about 20 minutes" first ranks five short lofi tracks
(12.6 min, outside the 18-22 min window). Revising the *inferred* detailed mood
from "focused" to "cozy" brings in longer tracks (21.3 min) and passes.
With "only lofi" the same request is provably impossible: the five longest
lofi tracks total 15.7 min, so the loop stops without revising.
"""

import json

import pytest

from src.agent import (
    ImmutableFieldError, LLMRateLimitError, LLMUnavailableError, MockLLMClient,
    _assert_locked_fields_preserved, locked_fields, revisable_fields,
)
from src.mock_responses import constraint, label, level, planner_response, requirement
from src.observations import (
    API_FAILURE, CONSTRAINTS_UNSATISFIED, IMPOSSIBLE_REQUEST, INVALID_PLAN, NO_REVISABLE_FIELDS,
    REPEATED_PLAN, RETRY_BUDGET_EXHAUSTED, SUCCEEDED, UNSUPPORTED_METADATA, Observation,
    observe_validation,
)
from src.orchestrator import DEFAULT_MAX_ATTEMPTS, WorkflowError, run_recommendation_workflow
from src.playlist_builder import DURATION_FIELD
from src.recommender import load_songs
from src.schemas import HardConstraint

CALM_LOFI = "I want calm lofi music for studying."
TWENTY_MIN = "Give me 5 calm lofi songs for studying, about 20 minutes long, nothing by Neon Echo"
TWENTY_MIN_ONLY_LOFI = "Give me 5 calm lofi songs for studying, about 20 minutes long, only lofi"
NEON_ECHO = {"type": "exclude_artist", "values": ["Neon Echo"]}


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


def calm_lofi_plan(**overrides):
    response = planner_response(
        genre=label("lofi", evidence="lofi"),
        mood=label("chill", evidence="calm"),
        energy=level("low", evidence="calm", reason="calm music is low energy"),
        detailed_mood=label("focused", source="inferred", evidence="for studying", reason="studying needs focus"),
    )
    response.update(overrides)
    return response


def twenty_minute_plan(constraints=None):
    """Initial plan for TWENTY_MIN: 5 songs, 20 minutes, explicit lofi/chill."""
    if constraints is None:
        constraints = [constraint("exclude_artist", "nothing by Neon Echo", values=["Neon Echo"])]
    return calm_lofi_plan(
        playlist={"playlist_size": requirement(5, "5 calm lofi songs"),
                  "unique_artists": None, "energy_progression": None},
        duration_minutes=requirement(20, "about 20 minutes"),
        hard_constraints=constraints,
    )


def revision(rationale="the playlist was too short; try other soft settings", strategy="omit", **preferences):
    """A revision response. Keyword args are preference name -> label()/level() or None."""
    data = {"preferences": preferences, "rationale": rationale}
    if strategy != "omit":
        data["strategy"] = strategy
    return data


def inferred(value, reason="revised"):
    return label(value, source="inferred", reason=reason)


COZY = revision("Total length was 12.6 min, below 18 min; cozy lofi tracks tend to be longer.",
                preferred_detailed_mood=inferred("cozy", "longer, cozier tracks"))


def run(request, responses, **kwargs):
    client = MockLLMClient(responses)
    return run_recommendation_workflow(request, client=client, **kwargs), client


# --------------------------------------------------------------------------
# 1. Success on the first attempt
# --------------------------------------------------------------------------


def test_success_on_first_attempt():
    result, client = run(CALM_LOFI, [calm_lofi_plan(), COZY])
    assert result.final_status == SUCCEEDED and result.fully_satisfied
    assert len(client.calls) == 1  # no revision call was made
    (attempt,) = result.trace.attempts
    assert attempt.kind == "initial" and attempt.outcome == "passed" and attempt.observation is None
    assert [call.tool for call in attempt.tool_calls] == [
        "filter_hard_constraints", "recommend_songs", "build_playlist", "validate_playlist"]
    assert result.trace.result_attempt == 1


def test_default_budget_is_three_attempts():
    assert DEFAULT_MAX_ATTEMPTS == 3
    result, _ = run(CALM_LOFI, [calm_lofi_plan()])
    assert result.trace.max_attempts == 3


# --------------------------------------------------------------------------
# 2. Success after a revised plan (demonstration scenario)
# --------------------------------------------------------------------------


def test_success_after_revision(songs):
    result, client = run(TWENTY_MIN, [twenty_minute_plan(), COZY])

    assert result.final_status == SUCCEEDED and result.validation.passed
    assert len(client.calls) == 2
    first, second = result.trace.attempts

    # Observe: attempt 1 failed with a machine-readable, retryable reason.
    assert first.outcome == CONSTRAINTS_UNSATISFIED
    assert first.observation.retryable and first.observation.codes == ("duration_out_of_range",)
    (reason,) = first.observation.reasons
    assert reason.check == "duration" and reason.details["evidence"]["total_minutes"] == 12.57

    # Revise: only the inferred detailed mood changed, with the LLM's rationale recorded.
    assert second.kind == "revision" and second.outcome == "passed"
    assert second.changed_fields == ("preferred_detailed_mood",)
    assert "cozy lofi tracks" in second.rationale
    assert first.plan["preferences"]["preferred_detailed_mood"]["value"] == "focused"
    assert second.plan["preferences"]["preferred_detailed_mood"] == {"value": "cozy", "source": "inferred"}

    # Hard requirements are identical in the initial and revised plans.
    for key in ("hard_constraints", "playlist", "target_duration_minutes"):
        assert first.plan[key] == second.plan[key]
    assert second.plan["hard_constraints"] == [NEON_ECHO]

    # The revision prompt carried the structured observation and the field locks.
    prompt = json.loads(client.calls[1]["user_prompt"])
    assert prompt["observation"]["reasons"][0]["code"] == "duration_out_of_range"
    assert "preferred_detailed_mood" in prompt["revisable_fields"]
    assert {"favorite_genre", "favorite_mood", "hard_constraints[0]", "playlist_size"} <= set(prompt["locked_fields"])
    assert prompt["previous_attempts"][0]["failure_codes"] == ["duration_out_of_range"]
    schema_prefs = client.calls[1]["schema"]["properties"]["preferences"]["properties"]
    assert "favorite_genre" not in schema_prefs and "preferred_detailed_mood" in schema_prefs

    # Real catalog songs only, and the playlist really meets the requirements.
    by_id = {song["id"]: song for song in songs}
    assert all(dict(s.record) == by_id[s.id] for s in result.songs)
    total = sum(s.record[DURATION_FIELD] for s in result.songs) / 60
    assert 18 <= total <= 22 and len(result.songs) == 5
    assert "Neon Echo" not in {s.artist for s in result.songs}
    note = result.planning.provenance_for("preferred_detailed_mood").note
    assert "revised in attempt 2" in note


def test_revision_can_change_inferred_strategy():
    response = revision(strategy=inferred("mood-first", "mood matters for studying"),
                        preferred_detailed_mood=inferred("cozy"))
    result, _ = run(TWENTY_MIN, [twenty_minute_plan(), response])
    assert result.trace.attempts[1].changed_fields == ("preferred_detailed_mood", "strategy")
    assert result.plan.strategy == "mood-first"


def test_invalid_initial_plan_is_retried_with_feedback():
    result, client = run(CALM_LOFI, ["{not json", calm_lofi_plan()])
    assert result.final_status == SUCCEEDED
    first, second = result.trace.attempts
    assert first.outcome == INVALID_PLAN and first.observation.codes == ("llm_output_invalid",)
    assert second.kind == "initial"  # no valid plan existed, so planning is retried
    feedback = json.loads(client.calls[1]["user_prompt"])["previous_attempt_rejected"]
    assert feedback["category"] == INVALID_PLAN


# --------------------------------------------------------------------------
# 3. Retry exhaustion (demonstration of a failed scenario)
# --------------------------------------------------------------------------


def test_retry_budget_exhausted_returns_honest_failure():
    responses = [
        twenty_minute_plan(),
        revision(preferred_detailed_mood=None),  # drop the detailed mood: 12.2 min
        revision(target_energy=level("medium", reason="try medium energy")),  # still 12.6 min
        COZY,  # would pass, but the budget is already spent
    ]
    result, client = run(TWENTY_MIN, responses)

    assert result.final_status == RETRY_BUDGET_EXHAUSTED
    assert len(client.calls) == 3 and len(result.trace.attempts) == 3
    assert [a.outcome for a in result.trace.attempts] == [CONSTRAINTS_UNSATISFIED] * 3
    # No fabricated success: the returned playlist is the last one built, and it failed.
    assert not result.fully_satisfied and result.status == "failed"
    assert result.trace.result_attempt == 3
    assert "exhausted" in result.trace.stop_reason
    # Three distinct plans were tried.
    assert len({a.plan_fingerprint for a in result.trace.attempts}) == 3


def test_max_attempts_is_configurable():
    result, client = run(TWENTY_MIN, [twenty_minute_plan(), COZY], max_attempts=1)
    assert result.final_status == RETRY_BUDGET_EXHAUSTED and len(client.calls) == 1

    result, client = run(TWENTY_MIN, [twenty_minute_plan(), COZY], max_attempts=2)
    assert result.final_status == SUCCEEDED and len(client.calls) == 2


@pytest.mark.parametrize("bad", [0, -1, 11, 2.5, True, "3"])
def test_invalid_max_attempts_is_rejected(bad):
    client = MockLLMClient([calm_lofi_plan()])
    with pytest.raises(ValueError):
        run_recommendation_workflow(CALM_LOFI, client=client, max_attempts=bad)
    assert client.calls == []


def test_only_invalid_plans_raise_with_trace():
    with pytest.raises(WorkflowError) as excinfo:
        run(CALM_LOFI, ["{not json"] * 3)
    trace = excinfo.value.trace
    assert trace.final_status == INVALID_PLAN and len(trace.attempts) == 3
    assert trace.result_attempt is None  # no playlist was built, none is returned


# --------------------------------------------------------------------------
# 4. Repeated identical plans
# --------------------------------------------------------------------------


def test_repeated_plan_stops_the_loop():
    # The "revision" restates the current soft values, so the plan is unchanged.
    same = revision("no better idea", preferred_detailed_mood=inferred("focused"),
                    target_energy=level("low", reason="unchanged"))
    result, client = run(TWENTY_MIN, [twenty_minute_plan(), same, COZY])

    assert result.final_status == REPEATED_PLAN
    assert len(client.calls) == 2  # stopped; the third response was never requested
    first, second = result.trace.attempts
    assert second.outcome == REPEATED_PLAN and second.tool_calls == ()  # tools not re-run
    assert first.plan_fingerprint == second.plan_fingerprint
    assert result.trace.result_attempt == 1 and not result.fully_satisfied


def test_repeat_of_an_earlier_plan_is_also_detected():
    responses = [
        twenty_minute_plan(),
        revision(preferred_detailed_mood=None),
        revision(preferred_detailed_mood=inferred("focused")),  # back to attempt 1's plan
    ]
    result, _ = run(TWENTY_MIN, responses)
    assert result.final_status == REPEATED_PLAN
    assert "attempt 1" in result.trace.stop_reason


# --------------------------------------------------------------------------
# 5. Immutable hard constraints
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tampered, message", [
    ({"hard_constraints": [], **COZY}, "hard_constraints: locked"),
    ({"playlist": {"playlist_size": 3}, **COZY}, "playlist: locked"),
    ({"duration_minutes": None, **COZY}, "duration_minutes: locked"),
    (revision(favorite_genre=inferred("ambient")), "preferences.favorite_genre: locked"),
    (revision(favorite_mood=inferred("relaxed")), "preferences.favorite_mood: locked"),
])
def test_revision_cannot_change_locked_fields(tampered, message):
    result, _ = run(TWENTY_MIN, [twenty_minute_plan(), tampered, COZY])

    rejected = result.trace.attempts[1]
    assert rejected.outcome == INVALID_PLAN and rejected.plan is None
    assert rejected.observation.codes == ("llm_output_invalid",)
    assert any(message in problem for problem in rejected.observation.reasons[0].details["problems"])
    # The next (valid) revision succeeds and every plan kept the same hard requirements.
    assert result.final_status == SUCCEEDED
    plans = [a.plan for a in result.trace.attempts if a.plan is not None]
    assert all(p["hard_constraints"] == [NEON_ECHO] for p in plans)
    assert all(p["playlist"]["playlist_size"] == 5 for p in plans)
    assert all(p["target_duration_minutes"] == 20 for p in plans)
    assert all(p["preferences"]["favorite_genre"] == {"value": "lofi", "source": "explicit"} for p in plans)


def test_explicit_strategy_is_locked():
    request = TWENTY_MIN + ", genre matters most"
    initial = twenty_minute_plan()
    initial["strategy"] = label("genre-first", evidence="genre matters most")
    result, client = run(request, [initial, revision(strategy=inferred("mood-first"))])
    assert "strategy" not in json.loads(client.calls[1]["user_prompt"])["revisable_fields"]
    assert any("strategy: locked" in p for p in result.trace.attempts[1].observation.reasons[0].details["problems"])


def test_revision_must_use_inferred_source():
    tampered = revision(preferred_detailed_mood=label("cozy", source="explicit", evidence="calm"))
    result, _ = run(TWENTY_MIN, [twenty_minute_plan(), tampered], max_attempts=2)
    assert result.trace.attempts[1].outcome == INVALID_PLAN


def test_locked_field_guard_rejects_tampered_plans():
    result, _ = run(TWENTY_MIN, [twenty_minute_plan()], max_attempts=1)
    planning = result.planning
    assert set(revisable_fields(planning)).isdisjoint(locked_fields(planning))

    import dataclasses
    loosened = dataclasses.replace(planning.plan, hard_constraints=())
    with pytest.raises(ImmutableFieldError) as excinfo:
        _assert_locked_fields_preserved(planning, loosened)
    assert excinfo.value.fields == ("hard_constraints",)

    resized = dataclasses.replace(planning.plan, playlist=dataclasses.replace(planning.plan.playlist, size=3))
    with pytest.raises(ImmutableFieldError):
        _assert_locked_fields_preserved(planning, resized)

    other = dataclasses.replace(planning.plan, hard_constraints=(HardConstraint("exclude_artist", ("LoRoom",)),))
    with pytest.raises(ImmutableFieldError):
        _assert_locked_fields_preserved(planning, other)


# --------------------------------------------------------------------------
# 6. Unsupported and impossible requests are not retried
# --------------------------------------------------------------------------


def test_impossible_request_is_not_retried():
    initial = twenty_minute_plan([constraint("genre", "only lofi", values=["lofi"])])
    result, client = run(TWENTY_MIN_ONLY_LOFI, [initial, COZY, COZY])

    assert result.final_status == IMPOSSIBLE_REQUEST
    assert len(client.calls) == 1  # provably impossible: no revision requested
    observation = result.trace.attempts[0].observation
    assert not observation.retryable and observation.codes == ("duration_infeasible",)
    evidence = observation.reasons[0].details["evidence"]
    assert evidence["possible_minutes_for_size"][1] < 18  # five longest lofi songs: 15.7 min
    assert not result.fully_satisfied and result.status == "infeasible"


def test_too_many_unique_artists_is_impossible():
    request = "10 songs, only lofi, all different artists"
    initial = calm_lofi_plan(
        playlist={"playlist_size": requirement(10, "10 songs"),
                  "unique_artists": requirement(True, "all different artists"), "energy_progression": None},
        hard_constraints=[constraint("genre", "only lofi", values=["lofi"])],
    )
    result, client = run(request, [initial, COZY])
    assert result.final_status == IMPOSSIBLE_REQUEST and len(client.calls) == 1
    assert "playlist_size_infeasible" in result.trace.attempts[0].observation.codes


def test_unsupported_request_is_not_retried():
    response = calm_lofi_plan(unsupported_requests=[{"request": "in French", "reason": "no language data"}])
    result, client = run("calm lofi for studying, in French", [response, COZY])
    assert result.final_status == UNSUPPORTED_METADATA and len(client.calls) == 1
    assert result.trace.attempts[0].observation.codes == ("unsupported_request",)
    assert len(result.songs) == 5  # the supported part is still delivered, flagged as unsupported


def test_missing_duration_metadata_is_unsupported(songs):
    no_duration = [{k: v for k, v in s.items() if k != DURATION_FIELD} for s in songs]
    response = calm_lofi_plan(duration_minutes=requirement(45, "45 minutes"))
    result, client = run("calm lofi for 45 minutes", [response, COZY], songs=no_duration)
    assert result.final_status == UNSUPPORTED_METADATA and len(client.calls) == 1
    assert result.trace.attempts[0].observation.codes == ("missing_catalog_metadata",)


def test_no_revisable_fields_stops():
    request = "5 lofi chill songs, very low energy, about 20 minutes, nothing by Neon Echo, focused, " \
              "popularity 50, from the 2020s, instrumentalness 50%, speechiness 10%, genre matters most"
    explicit = lambda value, ev: label(value, evidence=ev)  # noqa: E731
    number = lambda n, ev: {"level": None, "number": n, "source": "explicit", "evidence": ev, "reason": None}  # noqa: E731
    initial = twenty_minute_plan()
    initial["preferences"] = {
        "favorite_genre": explicit("lofi", "lofi"),
        "favorite_mood": explicit("chill", "chill"),
        "target_energy": level("very_low", source="explicit", evidence="very low energy"),
        "target_popularity": number(50, "popularity 50"),
        "preferred_release_decade": explicit(2020, "from the 2020s"),
        "preferred_detailed_mood": explicit("focused", "focused"),
        "target_instrumentalness": number(0.5, "instrumentalness 50%"),
        "target_speechiness": number(0.1, "speechiness 10%"),
    }
    initial["strategy"] = explicit("genre-first", "genre matters most")
    initial["playlist"]["playlist_size"] = requirement(5, "5 lofi chill songs")
    initial["duration_minutes"] = requirement(20, "about 20 minutes")
    result, client = run(request, [initial, COZY])
    assert revisable_fields(result.planning) == ()
    # The failure (duration) is fixable in principle, but every field is locked.
    assert result.trace.attempts[0].observation.codes == ("duration_out_of_range",)
    assert result.final_status == NO_REVISABLE_FIELDS and len(client.calls) == 1


# --------------------------------------------------------------------------
# 7. LLM API failures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("error", [LLMRateLimitError("slow down"), LLMUnavailableError("down")])
def test_api_failure_on_initial_plan_is_not_retried(error):
    with pytest.raises(WorkflowError) as excinfo:
        run(CALM_LOFI, [error, calm_lofi_plan()])
    assert excinfo.value.stage == "planning" and excinfo.value.cause is error
    trace = excinfo.value.trace
    assert trace.final_status == API_FAILURE and len(trace.attempts) == 1
    assert trace.attempts[0].observation.category == API_FAILURE


def test_api_failure_during_revision_returns_last_real_playlist():
    result, client = run(TWENTY_MIN, [twenty_minute_plan(), LLMRateLimitError("slow down"), COZY])
    assert result.final_status == API_FAILURE and len(client.calls) == 2
    assert result.trace.attempts[1].observation.codes == ("llm_rate_limited",)
    # The playlist shown is attempt 1's real (failed) playlist, never a fabricated success.
    assert result.trace.result_attempt == 1 and not result.fully_satisfied


# --------------------------------------------------------------------------
# Observations and traces
# --------------------------------------------------------------------------


def test_trace_is_json_serialisable_and_complete():
    result, _ = run(TWENTY_MIN, [twenty_minute_plan(), COZY])
    data = json.loads(json.dumps(result.trace.to_dict()))
    assert data["final_status"] == SUCCEEDED and data["attempts_used"] == 2
    assert result.trace.initial_plan == result.trace.attempts[0].plan
    assert result.trace.revised_plans == [result.trace.attempts[1].plan]
    first = data["attempts"][0]
    assert first["validation"]["outcome"] == "failed"
    assert first["observation"]["category"] == CONSTRAINTS_UNSATISFIED
    assert first["tool_calls"][0]["summary"]["hard_constraints"] == 1


def test_observe_validation_passed_is_none():
    result, _ = run(CALM_LOFI, [calm_lofi_plan()])
    assert observe_validation(result.validation) is None


def test_observation_rejects_unknown_category():
    with pytest.raises(ValueError):
        Observation("made_up", True, ())
