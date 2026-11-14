#!/usr/bin/env python3
"""Export blinded recommendation candidates and score human judgments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate import DEFAULT_SCENARIOS, _mock_responses  # noqa: E402
from src.agent import MockLLMClient  # noqa: E402
from src.orchestrator import WorkflowError, run_recommendation_workflow  # noqa: E402
from src.recommender import load_songs, recommend_songs  # noqa: E402

DEFAULT_CATALOG = ROOT / "data" / "songs.csv"
DEFAULT_CASES = ROOT / "evaluation" / "quality_cases.json"
# Gitignored, so neither the answer key nor in-progress ratings get committed by accident.
DEFAULT_REVIEW_DIR = ROOT / "evaluation" / "human_review"
DEFAULT_REVIEW = DEFAULT_REVIEW_DIR / "quality_review.json"
DEFAULT_ANSWER_KEY = DEFAULT_REVIEW_DIR / "quality_answer_key.json"
SYSTEMS = ("vibematch", "tunewise")
PLAYLIST_LABELS = ("A", "B")
RELEVANCE_SCALE = {
    0: "not relevant",
    1: "poor fit",
    2: "partial fit",
    3: "good fit",
    4: "excellent fit",
}
PLAYLIST_FIT_SCALE = {
    0: "does not satisfy the request",
    1: "mostly misses the request",
    2: "partly satisfies the request",
    3: "satisfies the request with minor issues",
    4: "fully satisfies the request",
}
_CANDIDATE_KEYS = {
    "candidate_id", "title", "artist", "genre", "mood", "detailed_mood", "energy", "relevance",
}


class QualityEvaluationError(ValueError):
    """The review file, judgments, or evaluation cases are invalid."""


def export_review(
    output_path: Path = DEFAULT_REVIEW,
    *,
    answer_key_path: Path | None = None,
    seed: int = 0,
    cases_path: Path = DEFAULT_CASES,
    scenarios_path: Path = DEFAULT_SCENARIOS,
    catalog_path: Path = DEFAULT_CATALOG,
) -> Dict[str, Any]:
    """Write a blinded review file for raters and a separate private answer key.

    Each song appears once per scenario even when both systems chose it, so raters
    cannot spot overlap and every song gets one consistent rating. Scenarios whose
    request is about the playlist as a whole (artist variety, energy flow) also get
    a side-by-side comparison of the two playlists under neutral labels A and B.
    """
    answer_key_path = answer_key_path or _default_answer_key_path(output_path)
    if output_path.resolve() == answer_key_path.resolve():
        raise QualityEvaluationError("review and answer-key output paths must be different")
    cases_data = json.loads(cases_path.read_text(encoding="utf-8"))
    scenarios_data = json.loads(scenarios_path.read_text(encoding="utf-8"))
    if cases_data.get("schema_version") != 2 or not isinstance(cases_data.get("cases"), list):
        raise QualityEvaluationError(f"unsupported quality-case format: {cases_path}")
    if scenarios_data.get("schema_version") != 1 or not isinstance(scenarios_data.get("scenarios"), list):
        raise QualityEvaluationError(f"unsupported workflow-scenario format: {scenarios_path}")

    scenarios = {item["id"]: item for item in scenarios_data["scenarios"]}
    catalog = load_songs(str(catalog_path))
    catalog_by_id = {song["id"]: song for song in catalog}
    rng = random.Random(seed)
    review_cases = []
    answer_key_cases = []
    next_candidate_number = 1
    for case_spec in cases_data["cases"]:
        scenario_id = case_spec.get("scenario_id")
        if scenario_id not in scenarios:
            raise QualityEvaluationError(f"quality case references unknown scenario {scenario_id!r}")
        rankings = _system_rankings(scenarios[scenario_id], catalog)

        song_ids = sorted(set(rankings["vibematch"]) | set(rankings["tunewise"]))
        rng.shuffle(song_ids)
        candidate_by_song = {}
        candidates = []
        for song_id in song_ids:
            candidate_id = f"candidate-{next_candidate_number:03d}"
            next_candidate_number += 1
            candidate_by_song[song_id] = candidate_id
            candidates.append(_candidate(candidate_id, catalog_by_id[song_id]))

        review_case: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "request": scenarios[scenario_id]["request"],
            "candidates": candidates,
        }
        key_case: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "song_ids": {cid: song_id for song_id, cid in candidate_by_song.items()},
            "rankings": {
                system: [candidate_by_song[song_id] for song_id in rankings[system]]
                for system in SYSTEMS
            },
            "shared_song_count": len(set(rankings["vibematch"]) & set(rankings["tunewise"])),
            "playlists": None,
        }
        if case_spec.get("playlist_comparison"):
            order = list(SYSTEMS)
            rng.shuffle(order)
            key_case["playlists"] = dict(zip(PLAYLIST_LABELS, order))
            review_case["playlists"] = {
                playlist_label: {
                    "songs": [
                        _playlist_entry(position, candidate_by_song[song_id], catalog_by_id[song_id])
                        for position, song_id in enumerate(rankings[system], start=1)
                    ],
                    "fit": None,
                }
                for playlist_label, system in key_case["playlists"].items()
            }
        review_cases.append(review_case)
        answer_key_cases.append(key_case)

    review = {
        "schema_version": 3,
        "rating_instructions": {
            "song_relevance_scale": {str(key): value for key, value in RELEVANCE_SCALE.items()},
            "playlist_fit_scale": {str(key): value for key, value in PLAYLIST_FIT_SCALE.items()},
            "guidance": (
                "Step 1: for every candidate, set relevance (0-4) for how well that single song "
                "fits the request. Candidates are shuffled and each song is listed once. "
                "Step 2: only where a case has playlists, rate each playlist's fit (0-4) as a "
                "whole, in the order shown, including artist variety and how energy changes "
                "across positions. Finish step 1 before looking at the playlists. "
                "Use only the request and the displayed metadata (energy runs from 0.0 = very "
                "calm to 1.0 = very intense). Do not infer missing song properties, and do not "
                "edit anything except the null rating fields."
            ),
        },
        "cases": review_cases,
        "judgment_status": "pending",
    }
    answer_key = {
        "schema_version": 2,
        "seed": seed,
        "review_fingerprint": _fingerprint(review_cases),
        "cases": answer_key_cases,
    }
    _write_json(output_path, review)
    _write_json(answer_key_path, answer_key)
    return review


def _system_rankings(scenario: Mapping[str, Any], catalog: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    scenario_id = scenario["id"]
    baseline = scenario.get("baseline")
    if scenario.get("expected_status") != "succeeded" or not baseline:
        raise QualityEvaluationError(
            f"quality case {scenario_id!r} must have a successful mock plan and baseline"
        )
    baseline_tuples = recommend_songs(
        dict(baseline["preferences"]), catalog, k=baseline.get("k", 5),
        mode=baseline.get("strategy", "genre-first"),
    )
    try:
        result = run_recommendation_workflow(
            scenario["request"], scenario.get("saved_preferences"),
            client=MockLLMClient(_mock_responses(scenario)), songs=catalog,
            max_attempts=scenario.get("max_attempts", 3),
            request_id=f"quality-review-{scenario_id}",
        )
    except WorkflowError as error:
        raise QualityEvaluationError(
            f"mock TuneWise workflow failed for {scenario_id!r}: {error}"
        ) from error
    if not result.validation.passed:
        raise QualityEvaluationError(
            f"mock TuneWise workflow did not satisfy constraints for {scenario_id!r}"
        )
    rankings = {
        "vibematch": [song["id"] for song, _, _ in baseline_tuples],
        "tunewise": [song.record["id"] for song in sorted(result.songs, key=lambda s: s.position)],
    }
    for system, ids in rankings.items():
        if not ids or len(set(ids)) != len(ids):
            raise QualityEvaluationError(f"{scenario_id}/{system} returned no or duplicate songs")
    return rankings


def _candidate(candidate_id: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "title": row["title"],
        "artist": row["artist"],
        "genre": row["genre"],
        "mood": row["mood"],
        "detailed_mood": row.get("detailed_mood"),
        "energy": row["energy"],
        "relevance": None,
    }


def _playlist_entry(position: int, candidate_id: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "position": position,
        "candidate_id": candidate_id,
        "title": row["title"],
        "artist": row["artist"],
        "energy": row["energy"],
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _default_answer_key_path(review_path: Path) -> Path:
    if review_path.resolve() == DEFAULT_REVIEW.resolve():
        return DEFAULT_ANSWER_KEY
    return review_path.with_name(f"{review_path.stem}.answer_key.json")


def _blank_cases(cases: Any) -> Any:
    """Copy of the review cases with every rating cleared, for fingerprinting."""
    blank = copy.deepcopy(cases)
    try:
        for case in blank:
            for candidate in case["candidates"]:
                candidate["relevance"] = None
            for playlist in (case.get("playlists") or {}).values():
                playlist["fit"] = None
    except (TypeError, KeyError, AttributeError) as error:
        raise QualityEvaluationError("review file is malformed") from error
    return blank


def _fingerprint(review_cases: Any) -> str:
    canonical = json.dumps(_blank_cases(review_cases), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rating(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in RELEVANCE_SCALE:
        raise QualityEvaluationError(f"{label} needs a human rating from 0 to 4")
    return value


def score_review(
    review_paths: Path | Sequence[Path] = DEFAULT_REVIEW,
    output_path: Path | None = None,
    answer_key_path: Path | None = None,
) -> Dict[str, Any]:
    """Join one or more raters' completed review files with the private answer key."""
    if isinstance(review_paths, (str, Path)):
        review_paths = [Path(review_paths)]
    review_paths = [Path(path) for path in review_paths]
    if not review_paths:
        raise QualityEvaluationError("at least one review file is required")
    if answer_key_path is None:
        if len(review_paths) > 1:
            raise QualityEvaluationError("pass --answer-key when scoring several review files")
        answer_key_path = _default_answer_key_path(review_paths[0])
    answer_key = json.loads(Path(answer_key_path).read_text(encoding="utf-8"))
    if answer_key.get("schema_version") != 2 or not isinstance(answer_key.get("cases"), list):
        raise QualityEvaluationError(f"unsupported answer-key format: {answer_key_path}")

    # ratings[scenario_id][candidate_id] -> one rating per rater (same for playlist fit).
    song_ratings: Dict[str, Dict[str, List[int]]] = {}
    playlist_ratings: Dict[str, Dict[str, List[int]]] = {}
    for review_path in review_paths:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("schema_version") != 3 or not isinstance(review.get("cases"), list):
            raise QualityEvaluationError(f"unsupported review format: {review_path}")
        if _fingerprint(review["cases"]) != answer_key.get("review_fingerprint"):
            raise QualityEvaluationError(
                f"{review_path} does not match the answer key: it comes from a different export "
                "or something other than a rating field was edited"
            )
        for case in review["cases"]:
            scenario_id = case["scenario_id"]
            for candidate in case["candidates"]:
                if set(candidate) != _CANDIDATE_KEYS:
                    raise QualityEvaluationError(f"{scenario_id!r} contains a malformed candidate")
                song_ratings.setdefault(scenario_id, {}).setdefault(
                    candidate["candidate_id"], [],
                ).append(_rating(candidate["relevance"], f"{review_path.name}: {candidate['candidate_id']!r}"))
            for playlist_label, playlist in (case.get("playlists") or {}).items():
                playlist_ratings.setdefault(scenario_id, {}).setdefault(playlist_label, []).append(
                    _rating(playlist["fit"], f"{review_path.name}: {scenario_id!r} playlist {playlist_label}")
                )

    per_scenario = []
    for key_case in answer_key["cases"]:
        scenario_id = key_case["scenario_id"]
        mean_rating = {
            candidate_id: sum(values) / len(values)
            for candidate_id, values in song_ratings[scenario_id].items()
        }
        rankings = key_case["rankings"]
        k = max(len(rankings[system]) for system in SYSTEMS)
        # Pooled ideal: the best ordering of every judged song for this request, so a
        # system is penalized for missing relevant songs the other system found.
        ideal_gains = sorted(mean_rating.values(), reverse=True)[:k]
        row: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "shared_song_count": key_case["shared_song_count"],
        }
        for system in SYSTEMS:
            gains = [mean_rating[candidate_id] for candidate_id in rankings[system]]
            row[system] = {
                "mean_relevance": _round(sum(gains) / len(gains)),
                "ndcg_at_k": _round(_ndcg(gains, ideal_gains)),
                "relevance_by_rank": [_round(gain) for gain in gains],
            }
        if key_case.get("playlists"):
            for playlist_label, system in key_case["playlists"].items():
                fits = playlist_ratings[scenario_id][playlist_label]
                row[system]["playlist_fit"] = _round(sum(fits) / len(fits))
        per_scenario.append(row)

    metrics = {}
    for system in SYSTEMS:
        all_gains = [
            gain for row in per_scenario for gain in row[system]["relevance_by_rank"]
        ]
        metrics[system] = {
            "song_count": len(all_gains),
            "mean_relevance_0_to_4": _round(sum(all_gains) / len(all_gains)),
            "mean_ndcg_at_k": _round(
                sum(row[system]["ndcg_at_k"] for row in per_scenario) / len(per_scenario)
            ),
        }
    playlist_rows = [row for row in per_scenario if "playlist_fit" in row["tunewise"]]
    report = {
        "evaluation_type": "human_judged_relevance",
        "review_files": [str(path) for path in review_paths],
        "rater_count": len(review_paths),
        "judged_song_count": sum(len(ratings) for ratings in song_ratings.values()),
        "relevance_scale": {str(key): value for key, value in RELEVANCE_SCALE.items()},
        "metrics": metrics,
        "paired_ndcg_comparison": _paired(
            [row["tunewise"]["ndcg_at_k"] - row["vibematch"]["ndcg_at_k"] for row in per_scenario],
            "mean_ndcg_delta_tunewise_minus_vibematch",
        ),
        "playlist_fit_comparison": None if not playlist_rows else {
            "vibematch_mean_fit_0_to_4": _round(
                sum(row["vibematch"]["playlist_fit"] for row in playlist_rows) / len(playlist_rows)
            ),
            "tunewise_mean_fit_0_to_4": _round(
                sum(row["tunewise"]["playlist_fit"] for row in playlist_rows) / len(playlist_rows)
            ),
            **_paired(
                [row["tunewise"]["playlist_fit"] - row["vibematch"]["playlist_fit"] for row in playlist_rows],
                "mean_fit_delta_tunewise_minus_vibematch",
            ),
        },
        "rater_agreement": _agreement(song_ratings, playlist_ratings),
        "per_scenario": per_scenario,
        "constraints_and_workflow_quality": (
            "Measured separately by evaluation.evaluate; relevance ratings do not replace "
            "deterministic constraint validation."
        ),
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report


def _paired(differences: Sequence[float], mean_name: str) -> Dict[str, Any]:
    return {
        "scenario_count": len(differences),
        mean_name: _round(sum(differences) / len(differences)),
        "tunewise_wins": sum(delta > 1e-9 for delta in differences),
        "ties": sum(abs(delta) <= 1e-9 for delta in differences),
        "vibematch_wins": sum(delta < -1e-9 for delta in differences),
    }


def _agreement(*rating_groups: Mapping[str, Mapping[str, Sequence[int]]]) -> Dict[str, Any] | None:
    """Pairwise agreement over every rated item; None with a single rater."""
    pairs = [
        abs(a - b)
        for group in rating_groups
        for items in group.values()
        for values in items.values()
        for a, b in itertools.combinations(values, 2)
    ]
    if not pairs:
        return None
    return {
        "rated_item_pairs": len(pairs),
        "exact_agreement_rate": _round(sum(diff == 0 for diff in pairs) / len(pairs)),
        "within_one_point_rate": _round(sum(diff <= 1 for diff in pairs) / len(pairs)),
        "mean_absolute_difference": _round(sum(pairs) / len(pairs)),
    }


def _round(value: float) -> float:
    return round(value, 4)


def _dcg(gains: Sequence[float]) -> float:
    return sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def _ndcg(gains: Sequence[float], ideal_gains: Sequence[float] | None = None) -> float:
    """NDCG against ``ideal_gains`` (defaults to the best reordering of ``gains``).

    Returns 0.0 when nothing judged is relevant, since no ordering found anything useful.
    """
    ideal = sorted(gains if ideal_gains is None else ideal_gains, reverse=True)[:len(gains)]
    ideal_dcg = _dcg(ideal)
    return _dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    export_parser = actions.add_parser("export", help="generate candidates for human relevance review")
    export_parser.add_argument("--output", type=Path, default=DEFAULT_REVIEW)
    export_parser.add_argument("--answer-key", type=Path)
    export_parser.add_argument("--seed", type=int, default=0, help="fixed shuffle seed (default: 0)")
    export_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    export_parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    export_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    score_parser = actions.add_parser("score", help="score one or more completed rating files")
    score_parser.add_argument(
        "--review", type=Path, action="append",
        help="a rater's completed review file; repeat once per rater (default: the exported file)",
    )
    score_parser.add_argument("--answer-key", type=Path)
    score_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "export":
        review = export_review(
            args.output, answer_key_path=args.answer_key, seed=args.seed, cases_path=args.cases,
            scenarios_path=args.scenarios, catalog_path=args.catalog,
        )
        song_count = sum(len(case["candidates"]) for case in review["cases"])
        playlist_count = sum(len(case.get("playlists") or {}) for case in review["cases"])
        print(f"Exported {song_count} songs and {playlist_count} playlists to rate: {args.output}")
        print("Give raters only that file (one copy per rater).")
        print(f"Keep the answer key private: {args.answer_key or _default_answer_key_path(args.output)}")
    else:
        report = score_review(args.review or [DEFAULT_REVIEW], args.output, args.answer_key)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
