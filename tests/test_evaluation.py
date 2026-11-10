import pytest

from evaluation.evaluate import evaluate_scenarios


def test_mock_evaluation_covers_scenarios_and_reports_real_metrics():
    report = evaluate_scenarios(mode="mock")

    assert report["evaluation_mode"] == "deterministic_mock_llm"
    assert report["scenario_count"] == 10
    assert report["metrics"]["plan_validity_rate"] > 0
    assert report["metrics"]["mean_execution_latency_ms"] > 0
    assert report["metrics"]["error_handling_accuracy"] == 1
    assert report["vibematch_comparison"]["scenario_count"] == 4
    assert {
        row["id"]: row["actual_status"] for row in report["scenarios"]
    } == {
        "normal-lofi-study": "succeeded",
        "explicit-overrides-saved-preference": "succeeded",
        "unsupported-language-metadata": "unsupported_metadata",
        "unique-artists": "succeeded",
        "ascending-energy": "succeeded",
        "impossible-small-catalog-subset": "impossible_request",
        "invalid-llm-output": "invalid_plan",
        "retry-budget-exhaustion": "retry_budget_exhausted",
        "successful-plan-revision": "succeeded",
        "llm-api-rate-limit": "api_failure",
    }


def test_unknown_evaluation_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        evaluate_scenarios(mode="unrecognized")
