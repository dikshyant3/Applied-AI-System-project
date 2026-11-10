#!/usr/bin/env python3
"""Run deterministic mock-LLM or explicitly requested live TuneWise evaluations."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import (  # noqa: E402
    LLMRateLimitError, LLMUnavailableError, MockLLMClient, create_llm_client, load_env_file,
)
from src.mock_responses import (  # noqa: E402
    constraint, label, level, planner_response, requirement,
)
from src.orchestrator import WorkflowError, run_recommendation_workflow  # noqa: E402
from src.recommender import load_songs, recommend_songs  # noqa: E402
from src.workflow_logging import configure_workflow_logging  # noqa: E402

DEFAULT_SCENARIOS = ROOT / "evaluation" / "scenarios.json"
DEFAULT_CATALOG = ROOT / "data" / "songs.csv"


def _plan_response(spec: Mapping[str, Any]) -> Dict[str, Any]:
    def label_source(field: str) -> str:
        return spec.get(f"{field}_source", "explicit")

    def label_evidence(field: str) -> str | None:
        if f"{field}_evidence" in spec:
            return spec[f"{field}_evidence"]
        return spec[field] if label_source(field) == "explicit" else None

    preferences = {
        "genre": label(spec["genre"], label_source("genre"), label_evidence("genre")),
        "mood": label(spec["mood"], label_source("mood"), label_evidence("mood")),
        "energy": level(spec.get("energy", "low"), spec.get("energy_source", "inferred")),
    }
    if "detailed_mood" in spec:
        preferences["detailed_mood"] = label(
            spec["detailed_mood"], spec.get("detailed_mood_source", "inferred"),
            spec.get("detailed_mood_evidence"), "evaluation scenario",
        )

    playlist: Dict[str, Any] = {}
    if "playlist_size" in spec:
        playlist["playlist_size"] = requirement(
            spec["playlist_size"], spec.get("playlist_size_evidence", f"{spec['playlist_size']} songs"),
        )
    if "unique_artists" in spec:
        playlist["unique_artists"] = requirement(
            spec["unique_artists"], spec.get("unique_artists_evidence", "different artists"),
        )
    if "energy_progression" in spec:
        playlist["energy_progression"] = requirement(
            spec["energy_progression"],
            spec.get("energy_progression_evidence", spec["energy_progression"]),
        )

    constraints = [
        constraint(
            item["type"], item["evidence"], values=item.get("values"),
            minimum=item.get("minimum"), maximum=item.get("maximum"),
        )
        for item in spec.get("constraints", [])
    ]
    duration = None
    if "duration_minutes" in spec:
        duration = requirement(
            spec["duration_minutes"], spec.get("duration_evidence", "about 20 minutes"),
        )
    return planner_response(
        preferences["genre"], preferences["mood"], preferences["energy"],
        detailed_mood=preferences.get("detailed_mood"),
        playlist=playlist, constraints=constraints, duration=duration,
        unsupported=spec.get("unsupported", ()),
    )


def _mock_responses(scenario: Mapping[str, Any]) -> list[Any]:
    responses: list[Any] = []
    for spec in scenario["responses"]:
        response_type = spec["type"]
        if response_type == "plan":
            responses.append(_plan_response(spec))
        elif response_type == "revision":
            responses.append({
                "preferences": spec.get("preferences", {}),
                "rationale": spec.get("rationale", "Evaluation revision."),
            })
        elif response_type == "raw":
            responses.append(spec["value"])
        elif response_type == "api_failure":
            error_type = {
                "rate_limit": LLMRateLimitError,
                "unavailable": LLMUnavailableError,
            }.get(spec["kind"])
            if error_type is None:
                raise ValueError(f"unknown API failure kind {spec['kind']!r}")
            responses.append(error_type(spec.get("message", "simulated API failure")))
        else:
            raise ValueError(f"unknown response type {response_type!r}")
    return responses


def _baseline_metrics(spec: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    started = time.perf_counter()
    recommendations = recommend_songs(
        dict(spec["preferences"]), list(catalog), k=spec.get("k", 5),
        mode=spec.get("strategy", "genre-first"),
    )
    duration_ms = (time.perf_counter() - started) * 1000
    selected = [song for song, _, _ in recommendations]
    distinct_artists = len({song["artist"].casefold() for song in selected})
    checks = []
    if spec.get("unique_artists"):
        checks.append(distinct_artists == len(selected))
    direction = spec.get("energy_progression")
    if direction == "ascending":
        checks.append(all(a["energy"] <= b["energy"] for a, b in zip(selected, selected[1:])))
    elif direction == "descending":
        checks.append(all(a["energy"] >= b["energy"] for a, b in zip(selected, selected[1:])))
    constraint_satisfaction = all(checks) if checks else True
    return {
        "latency_ms": duration_ms,
        "selected_count": len(selected),
        "artist_diversity": distinct_artists / len(selected) if selected else 0.0,
        "constraint_satisfaction": constraint_satisfaction,
    }


def _evaluate_one(scenario: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]],
                  mode: str) -> Dict[str, Any]:
    if mode == "mock":
        client = MockLLMClient(_mock_responses(scenario))
    else:
        load_env_file()
        client = create_llm_client()
    started = time.perf_counter()
    result = None
    error = None
    try:
        result = run_recommendation_workflow(
            scenario["request"],
            scenario.get("saved_preferences"),
            client=client,
            songs=catalog,
            max_attempts=scenario.get("max_attempts", 3),
            request_id=f"evaluation-{scenario['id']}",
        )
    except WorkflowError as caught:
        error = caught
    latency_ms = (time.perf_counter() - started) * 1000
    trace = result.trace if result is not None else error.trace if error is not None else None
    final_status = result.final_status if result is not None else (
        trace.final_status if trace is not None else "workflow_error"
    )
    songs = result.songs if result is not None else ()
    distinct_artists = len({song.artist.casefold() for song in songs})
    attempts_used = len(trace.attempts) if trace is not None else 0
    valid_plan = bool(trace and any(attempt.plan is not None for attempt in trace.attempts))
    constraints_satisfied = bool(result and result.validation.passed)
    expected_status = scenario["expected_status"] if mode == "mock" else None

    row: Dict[str, Any] = {
        "id": scenario["id"],
        "category": scenario["category"],
        "expected_status": expected_status,
        "actual_status": final_status,
        "expected_status_matched": final_status == expected_status if expected_status else None,
        "plan_valid": valid_plan,
        "constraints_satisfied": constraints_satisfied,
        "artist_diversity": distinct_artists / len(songs) if songs else 0.0,
        "attempts": attempts_used,
        "retry_success": attempts_used > 1 and final_status == "succeeded",
        "latency_ms": latency_ms,
    }
    if scenario.get("baseline"):
        row["baseline"] = _baseline_metrics(scenario["baseline"], catalog)
    return row


def evaluate_scenarios(scenarios_path: Path = DEFAULT_SCENARIOS, catalog_path: Path = DEFAULT_CATALOG,
                       mode: str = "mock") -> Dict[str, Any]:
    """Evaluate every scenario and return measured metrics as JSON-ready data."""
    if mode not in ("mock", "live"):
        raise ValueError("mode must be 'mock' or 'live'")
    data = json.loads(scenarios_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("scenarios"), list):
        raise ValueError(f"unsupported scenario file format: {scenarios_path}")
    scenarios = data["scenarios"]
    catalog = load_songs(str(catalog_path))
    rows = [_evaluate_one(scenario, catalog, mode) for scenario in scenarios]
    latencies = [row["latency_ms"] for row in rows]
    retried = [row for row in rows if row["attempts"] > 1]
    matched = [row for row in rows if row["expected_status_matched"] is not None]
    comparable = [row for row in rows if "baseline" in row]
    metrics = {
        "plan_validity_rate": _mean([float(row["plan_valid"]) for row in rows]),
        "constraint_satisfaction_rate": _mean([float(row["constraints_satisfied"]) for row in rows]),
        "mean_artist_diversity": _mean([row["artist_diversity"] for row in rows]),
        "retry_success_rate": (
            _mean([float(row["retry_success"]) for row in retried]) if retried else None
        ),
        "error_handling_accuracy": (
            _mean([float(row["expected_status_matched"]) for row in matched]) if matched else None
        ),
        "mean_execution_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
    }
    comparison = {
        "scenario_count": len(comparable),
        "comparable_scenarios": [row["id"] for row in comparable],
        "vibematch": {
            "mean_artist_diversity": _mean([row["baseline"]["artist_diversity"] for row in comparable]),
            "constraint_satisfaction_rate": _mean([
                float(row["baseline"]["constraint_satisfaction"]) for row in comparable
            ]),
            "mean_execution_latency_ms": _mean([row["baseline"]["latency_ms"] for row in comparable]),
        },
        "tunewise": {
            "mean_artist_diversity": _mean([row["artist_diversity"] for row in comparable]),
            "constraint_satisfaction_rate": _mean([
                float(row["constraints_satisfied"]) for row in comparable
            ]),
            "mean_execution_latency_ms": _mean([row["latency_ms"] for row in comparable]),
        },
    }
    return {
        "evaluation_mode": "deterministic_mock_llm" if mode == "mock" else "live_llm",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_file": str(scenarios_path),
        "scenario_count": len(rows),
        "metrics": metrics,
        "vibematch_comparison": comparison,
        "scenarios": rows,
    }


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    args = parser.parse_args(argv)
    configure_workflow_logging()
    report = evaluate_scenarios(args.scenarios, args.catalog, args.mode)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
