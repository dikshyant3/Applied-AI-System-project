import json
import shutil
import subprocess

import pytest

from evaluation.quality import (
    DEFAULT_ANSWER_KEY, DEFAULT_REVIEW, QualityEvaluationError, _ndcg, export_review, score_review,
)


@pytest.fixture
def exported(tmp_path):
    review_path = tmp_path / "review.json"
    answer_key_path = tmp_path / "answer_key.json"
    review = export_review(review_path, answer_key_path=answer_key_path, seed=41)
    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    return review_path, answer_key_path, review, answer_key


def _fill(review, relevance, fit=2):
    filled = json.loads(json.dumps(review))
    for case in filled["cases"]:
        for candidate in case["candidates"]:
            candidate["relevance"] = relevance(candidate) if callable(relevance) else relevance
        for playlist in (case.get("playlists") or {}).values():
            playlist["fit"] = fit
    return filled


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_ndcg_rewards_relevant_songs_at_higher_ranks():
    assert _ndcg([4, 3, 0]) == 1
    assert _ndcg([0, 3, 4]) < 1
    assert _ndcg([0, 0]) == 0


def test_pooled_ndcg_penalizes_missing_relevant_songs():
    # Perfectly ordered, but the other system found better songs.
    assert _ndcg([2, 1], ideal_gains=[4, 3, 2, 1]) < 1
    assert _ndcg([4, 3], ideal_gains=[4, 3, 2, 1]) == 1


def test_export_is_blinded_deduplicated_and_reproducible(exported, tmp_path):
    _, answer_key_path, review, answer_key = exported

    assert answer_key_path.exists()
    assert review["judgment_status"] == "pending"
    assert len(review["cases"]) == 6
    review_json = json.dumps(review).lower()
    assert "vibematch" not in review_json
    assert "tunewise" not in review_json
    for case, key_case in zip(review["cases"], answer_key["cases"]):
        titles = [(c["title"], c["artist"]) for c in case["candidates"]]
        assert len(titles) == len(set(titles)), "each song is shown once per scenario"
        for candidate in case["candidates"]:
            assert candidate["relevance"] is None
            assert isinstance(candidate["energy"], float)
        shown = {c["candidate_id"] for c in case["candidates"]}
        assert shown == set(key_case["song_ids"])
        for system in ("vibematch", "tunewise"):
            assert set(key_case["rankings"][system]) <= shown

    repeated = export_review(
        tmp_path / "repeat.json", answer_key_path=tmp_path / "repeat-key.json", seed=41,
    )
    assert review == repeated


def test_cases_differ_between_systems(exported):
    _, _, _, answer_key = exported
    for key_case in answer_key["cases"]:
        rankings = key_case["rankings"]
        assert rankings["vibematch"] != rankings["tunewise"], key_case["scenario_id"]


def test_set_level_requests_get_blinded_playlist_comparison(exported):
    _, _, review, answer_key = exported
    with_playlists = {
        case["scenario_id"] for case in review["cases"] if case.get("playlists")
    }
    assert with_playlists == {"ascending-energy", "ambient-wind-down", "happy-pop-unique-artists"}
    for case, key_case in zip(review["cases"], answer_key["cases"]):
        if not case.get("playlists"):
            assert key_case["playlists"] is None
            continue
        assert set(case["playlists"]) == {"A", "B"}
        assert sorted(key_case["playlists"].values()) == ["tunewise", "vibematch"]
        for playlist_label, playlist in case["playlists"].items():
            assert playlist["fit"] is None
            system = key_case["playlists"][playlist_label]
            assert [song["candidate_id"] for song in playlist["songs"]] == key_case["rankings"][system]


def test_score_requires_every_rating(exported):
    review_path, answer_key_path, review, _ = exported
    with pytest.raises(QualityEvaluationError, match="rating from 0 to 4"):
        score_review(review_path, answer_key_path=answer_key_path)

    songs_only = _fill(review, 3, fit=None)
    _write(review_path, songs_only)
    with pytest.raises(QualityEvaluationError, match="playlist"):
        score_review(review_path, answer_key_path=answer_key_path)


def test_invalid_rating_is_rejected(exported):
    review_path, answer_key_path, review, _ = exported
    filled = _fill(review, 2)
    filled["cases"][0]["candidates"][0]["relevance"] = 5
    _write(review_path, filled)

    with pytest.raises(QualityEvaluationError, match="rating from 0 to 4"):
        score_review(review_path, answer_key_path=answer_key_path)


def test_edited_song_details_are_rejected(exported):
    review_path, answer_key_path, review, _ = exported
    filled = _fill(review, 2)
    filled["cases"][0]["candidates"][0]["genre"] = "edited"
    _write(review_path, filled)

    with pytest.raises(QualityEvaluationError, match="does not match the answer key"):
        score_review(review_path, answer_key_path=answer_key_path)


def test_report_compares_systems_and_scores_playlists(exported, tmp_path):
    review_path, answer_key_path, review, answer_key = exported
    report_path = tmp_path / "report.json"
    _write(review_path, _fill(review, lambda c: 4 if c["genre"] in {"lofi", "rock"} else 1))

    report = score_review(review_path, report_path, answer_key_path)

    assert report_path.exists()
    assert report["evaluation_type"] == "human_judged_relevance"
    assert report["rater_count"] == 1
    assert report["rater_agreement"] is None
    assert report["judged_song_count"] == sum(len(c["song_ids"]) for c in answer_key["cases"])
    assert set(report["metrics"]) == {"vibematch", "tunewise"}
    assert report["paired_ndcg_comparison"]["scenario_count"] == 6
    assert report["playlist_fit_comparison"]["scenario_count"] == 3
    assert len(report["per_scenario"]) == 6
    rock = next(row for row in report["per_scenario"] if row["scenario_id"] == "rock-workout-exclude-artist")
    # TuneWise picked a non-rock song in this scenario, which these ratings penalize.
    assert rock["tunewise"]["mean_relevance"] < rock["vibematch"]["mean_relevance"]
    assert "constraint validation" in report["constraints_and_workflow_quality"]


def test_multiple_raters_are_averaged_with_agreement(exported, tmp_path):
    review_path, answer_key_path, review, _ = exported
    first = _write(tmp_path / "rater1.json", _fill(review, 4, fit=4))
    second = _write(tmp_path / "rater2.json", _fill(review, 2, fit=2))

    report = score_review([first, second], answer_key_path=answer_key_path)

    assert report["rater_count"] == 2
    assert report["metrics"]["tunewise"]["mean_relevance_0_to_4"] == 3
    assert report["playlist_fit_comparison"]["tunewise_mean_fit_0_to_4"] == 3
    assert report["rater_agreement"]["exact_agreement_rate"] == 0
    assert report["rater_agreement"]["mean_absolute_difference"] == 2

    with pytest.raises(QualityEvaluationError, match="--answer-key"):
        score_review([first, second])


def test_review_and_answer_key_cannot_overwrite_each_other(tmp_path):
    path = tmp_path / "same.json"

    with pytest.raises(QualityEvaluationError, match="paths must be different"):
        export_review(path, answer_key_path=path)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_default_review_files_are_gitignored():
    root = DEFAULT_REVIEW.parents[2]
    for path in (DEFAULT_REVIEW, DEFAULT_ANSWER_KEY, root / "tunewise-blind-answer-key-final.json"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)], cwd=root, check=False,
        )
        assert result.returncode == 0, f"{path} is not gitignored"
