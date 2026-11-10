import json
import logging

import pytest

from src.agent import MockLLMClient
from src.mock_responses import label, level, planner_response
from src.orchestrator import WorkflowError, run_recommendation_workflow


def _plan():
    return planner_response(
        genre=label("lofi", evidence="lofi"),
        mood=label("chill", evidence="calm"),
        energy=level("low", evidence="calm"),
    )


def test_workflow_emits_json_events_with_trace_fields(caplog):
    request = "Private request text: lofi and chill"
    caplog.set_level(logging.INFO, logger="tunewise.workflow")

    result = run_recommendation_workflow(
        request, client=MockLLMClient([_plan()]), request_id="phase6-test",
    )

    events = [json.loads(record.getMessage()) for record in caplog.records
              if record.name == "tunewise.workflow"]
    assert events
    assert all(event["request_id"] == "phase6-test" for event in events)
    by_name = {event["event"]: event for event in events}
    assert by_name["initial_plan_validated"]["strategy"] == "genre-first"
    assert "evidence" not in json.dumps(by_name["initial_plan_validated"])
    assert request not in json.dumps(events)
    assert by_name["workflow_finished"]["final_status"] == result.final_status
    assert by_name["workflow_finished"]["duration_seconds"] >= 0
    assert by_name["validation_result"]["outcome"] == "passed"
    tools = {event["tool"]: event for event in events if event["event"] == "tool_executed"}
    assert {"filter_hard_constraints", "recommend_songs", "build_playlist", "validate_playlist"} <= set(tools)
    assert tools["recommend_songs"]["ranked_candidate_count"] == result.candidate_count


def test_retries_and_final_status_are_logged(caplog):
    invalid = "{not valid json"
    caplog.set_level(logging.INFO, logger="tunewise.workflow")

    with pytest.raises(WorkflowError):
        run_recommendation_workflow(
            "lofi and chill", client=MockLLMClient([invalid, invalid, invalid]),
            request_id="invalid-plan-test",
        )

    events = [json.loads(record.getMessage()) for record in caplog.records
              if record.name == "tunewise.workflow"]
    assert [event["attempt"] for event in events if event["event"] == "attempt_started"] == [1, 2, 3]
    assert events[-1]["event"] == "workflow_finished"
    assert events[-1]["final_status"] == "invalid_plan"


def test_client_configuration_failure_is_logged(monkeypatch, caplog):
    import src.orchestrator as orchestrator
    from src.agent import LLMConfigurationError

    def fail_to_create_client():
        raise LLMConfigurationError("missing test credentials")

    monkeypatch.setattr(orchestrator._agent, "create_llm_client", fail_to_create_client)
    caplog.set_level(logging.INFO, logger="tunewise.workflow")

    with pytest.raises(WorkflowError):
        run_recommendation_workflow("lofi and chill", request_id="config-failure")

    events = [json.loads(record.getMessage()) for record in caplog.records
              if record.name == "tunewise.workflow"]
    assert [event["event"] for event in events] == ["workflow_started", "workflow_finished"]
    assert events[-1]["final_status"] == "api_failure"
    assert events[-1]["attempts_used"] == 0
