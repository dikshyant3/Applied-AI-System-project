import pytest

from src.main import main


def test_mock_scenario_runs_full_workflow_offline(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert main(["--mock-scenario", "successful-plan-revision"]) == 0

    out = capsys.readouterr().out
    assert "no API call is made" in out
    assert "Agent loop: SUCCEEDED after 2 of 3 attempt(s)" in out
    assert "Playlist validation: PASSED" in out


def test_mock_scenario_reports_failed_validation(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert main(["--mock-scenario", "retry-budget-exhaustion"]) == 3
    assert "RETRY_BUDGET_EXHAUSTED" in capsys.readouterr().out


def test_list_mock_scenarios(capsys):
    assert main(["--list-mock-scenarios"]) == 0
    assert "successful-plan-revision" in capsys.readouterr().out


def test_unknown_or_conflicting_mock_scenario_is_rejected():
    with pytest.raises(SystemExit):
        main(["--mock-scenario", "does-not-exist"])
    with pytest.raises(SystemExit):
        main(["--mock-scenario", "normal-lofi-study", "--request", "anything"])
