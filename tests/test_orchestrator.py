"""Integration tests: NL request -> mock LLM -> plan -> original recommender."""

import copy
from pathlib import Path

import pytest

from src import agent, main as cli
from src.agent import LLMRateLimitError, LLMResponseError, MockLLMClient, ModelUnavailableError
from src.mock_responses import constraint, label, level, planner_response, requirement
from src.orchestrator import WorkflowError, run_recommendation_workflow
from src.recommender import SCORING_STRATEGIES, load_songs, recommend_songs, score_song

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CALM_LOFI = "I want calm lofi music for studying."


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


def llm_response(**overrides):
    """What a well-behaved planner returns for CALM_LOFI."""
    response = planner_response(
        genre=label("lofi", evidence="lofi"),
        mood=label("chill", evidence="calm"),
        energy=level("low", evidence="calm", reason="calm music is low energy"),
        detailed_mood=label("focused", source="inferred", evidence="for studying", reason="studying needs focus"),
    )
    response.update(overrides)
    return response


def run(response, request=CALM_LOFI, **kwargs):
    return run_recommendation_workflow(request, client=MockLLMClient([response]), **kwargs)


# --------------------------------------------------------------------------
# End-to-end workflow
# --------------------------------------------------------------------------


def test_calm_lofi_request_returns_real_ranked_songs(songs):
    result = run(llm_response())

    assert result.recommender_kwargs == {
        "user_prefs": {
            "favorite_genre": "lofi",
            "favorite_mood": "chill",
            "target_energy": 0.25,
            "preferred_detailed_mood": "focused",
        }
    }
    assert result.strategy is SCORING_STRATEGIES["genre-first"]
    assert len(result.songs) == 5
    assert [s.rank for s in result.songs] == [1, 2, 3, 4, 5]
    assert result.songs[0].genre == "lofi"
    for song in result.songs:
        assert dict(song.record) in songs  # an actual catalog row, unmodified
        assert (song.id, song.title, song.artist, song.genre, song.mood, song.energy) == tuple(
            song.record[k] for k in ("id", "title", "artist", "genre", "mood", "energy")
        )


def test_workflow_output_is_exactly_the_original_recommender_output(songs):
    result = run(llm_response())
    direct = recommend_songs(dict(result.recommender_kwargs["user_prefs"]), songs)
    assert result.as_recommendation_tuples() == direct


def test_explanations_and_scores_come_from_existing_scorer(songs):
    result = run(llm_response())
    prefs = dict(result.recommender_kwargs["user_prefs"])
    for song in result.songs:
        base, reasons = score_song(prefs, dict(song.record))
        assert song.base_score == base
        assert list(song.explanations[: len(reasons)]) == reasons
        assert song.explanations[-1].startswith("diversity penalty")
        assert song.score <= song.base_score
    assert "genre match (+2.0)" in result.songs[0].explanations


@pytest.mark.parametrize("mode", ["genre-first", "mood-first"])
def test_llm_selected_strategy_drives_the_recommender(songs, mode):
    response = llm_response(strategy=label(mode, source="inferred", reason="test"))
    result = run(response)
    assert result.strategy is SCORING_STRATEGIES[mode]
    assert result.recommender_kwargs["mode"] == mode
    direct = recommend_songs(dict(result.recommender_kwargs["user_prefs"]), songs, mode=mode)
    assert result.as_recommendation_tuples() == direct


def test_unsupported_strategy_from_llm_stops_the_workflow():
    response = llm_response(strategy=label("popularity-first", source="inferred"))
    with pytest.raises(WorkflowError) as excinfo:
        run(response)
    assert excinfo.value.stage == "planning"
    assert isinstance(excinfo.value.cause, LLMResponseError)


def test_playlist_size_becomes_k():
    response = llm_response(playlist={"playlist_size": requirement(3, "lofi"), "unique_artists": None, "energy_progression": None})
    assert len(run(response).songs) == 3


def test_hard_constraints_filter_candidates_before_ranking(songs):
    request = "Only songs from the 2020s, nothing by Neon Echo, happy pop"
    response = llm_response(
        preferences={
            **llm_response()["preferences"],
            "favorite_genre": label("pop", evidence="pop"),
            "favorite_mood": label("happy", evidence="happy"),
            "preferred_detailed_mood": None,
        },
        hard_constraints=[
            constraint("release_decade", "Only songs from the 2020s", values=[2020]),
            constraint("exclude_artist", "nothing by Neon Echo", values=["Neon Echo"]),
        ],
    )
    result = run(response, request)

    eligible = [s for s in songs if s["release_decade"] == 2020 and s["artist"] != "Neon Echo"]
    assert result.candidate_count == len(eligible)
    assert all(s.record["release_decade"] == 2020 for s in result.songs)
    assert "Neon Echo" not in {s.artist for s in result.songs}
    direct = recommend_songs(dict(result.recommender_kwargs["user_prefs"]), eligible)
    assert result.as_recommendation_tuples() == direct


def test_playlist_rules_are_now_enforced_and_validated():
    request = "calm lofi, different artists, wind down"
    response = llm_response(playlist={
        "playlist_size": None,
        "unique_artists": requirement(True, "different artists"),
        "energy_progression": requirement("descending", "wind down"),
    })
    result = run(response, request)
    assert result.fully_satisfied and result.status == "passed"
    assert len({s.artist for s in result.songs}) == len(result.songs) == 5
    energies = [s.energy for s in result.songs]
    assert energies == sorted(energies, reverse=True)
    assert {"unique_artists", "energy_progression"} <= set(result.validation.checked)


def test_unsupported_requests_mark_result_not_fully_satisfied():
    response = llm_response(unsupported_requests=[{"request": "in French", "reason": "no language data"}])
    result = run(response, "calm lofi for studying, in French")
    assert not result.fully_satisfied
    assert result.status == "unsupported"
    assert result.validation.check("unsupported_request[0]").status == "unsupported"
    assert len(result.songs) == 5  # the supported part is still delivered


def test_fully_satisfied_for_plain_request():
    assert run(llm_response()).fully_satisfied


def test_saved_preferences_flow_through_to_recommender():
    result = run_recommendation_workflow(
        CALM_LOFI, {"target_popularity": 40}, client=MockLLMClient([llm_response()])
    )
    assert result.recommender_kwargs["user_prefs"]["target_popularity"] == 40


def test_workflow_does_not_mutate_catalog(songs):
    catalog = copy.deepcopy(songs)
    run(llm_response(), songs=catalog)
    assert catalog == songs


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response, cause",
    [
        ("{not json", LLMResponseError),
        (ModelUnavailableError("model gone"), ModelUnavailableError),
        (LLMRateLimitError("slow down"), LLMRateLimitError),
    ],
)
def test_planning_failures_become_workflow_errors(response, cause):
    with pytest.raises(WorkflowError) as excinfo:
        run(response)
    assert excinfo.value.stage == "planning"
    assert isinstance(excinfo.value.cause, cause)


def test_missing_catalog_is_a_catalog_error():
    client = MockLLMClient([llm_response()])
    with pytest.raises(WorkflowError) as excinfo:
        run_recommendation_workflow(CALM_LOFI, client=client, catalog_path="data/missing.csv")
    assert excinfo.value.stage == "catalog"
    assert client.calls == []  # LLM never called


def test_malformed_catalog_is_a_catalog_error(tmp_path):
    bad = tmp_path / "songs.csv"
    bad.write_text("id,title\n1,Only Title\n")
    with pytest.raises(WorkflowError) as excinfo:
        run(llm_response(), catalog_path=str(bad))
    assert excinfo.value.stage == "catalog"


def test_empty_catalog_is_a_catalog_error():
    with pytest.raises(WorkflowError, match="empty"):
        run(llm_response(), songs=[])


def test_invalid_request_is_rejected_before_llm():
    client = MockLLMClient([llm_response()])
    with pytest.raises(ValueError):
        run_recommendation_workflow("  ", client=client)
    assert client.calls == []


# --------------------------------------------------------------------------
# CLI: original behaviour preserved, natural-language path added
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, fixture",
    [
        ([], "original_cli_genre_first.txt"),
        (["--mode", "genre-first"], "original_cli_genre_first.txt"),
        (["--mode", "mood-first"], "original_cli_mood_first.txt"),
    ],
)
def test_original_cli_output_is_unchanged(monkeypatch, capsys, argv, fixture):
    # The fixtures were captured from the 18-song catalog before any TuneWise
    # change. Feeding the CLI those same 18 preserved records must reproduce
    # them byte-for-byte (proves the code path and the original rows are intact).
    originals = [s for s in load_songs("data/songs.csv") if s["id"] <= 18]
    monkeypatch.setattr(cli, "load_songs", lambda path: originals)
    assert cli.main(argv) is None
    assert capsys.readouterr().out == (FIXTURES / fixture).read_text()


def test_cli_natural_language_request(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_env_file", lambda *a, **k: [])
    monkeypatch.setattr(agent, "create_llm_client", lambda config=None: MockLLMClient([llm_response()]))

    assert cli.main(["--request", CALM_LOFI]) == 0
    out = capsys.readouterr().out
    assert "Playlist validation: PASSED" in out
    assert "Scoring mode: Genre-First" in out
    assert "favorite_genre" in out and "[explicit]" in out and "[inferred]" in out
    assert "Library Rain" in out and "Midnight Coding" in out
    assert "genre match (+2.0)" in out


def test_cli_request_failure_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_env_file", lambda *a, **k: [])
    monkeypatch.setattr(agent, "create_llm_client", lambda config=None: MockLLMClient(["oops"]))

    assert cli.main(["--request", CALM_LOFI]) == 1
    assert "planning failed" in capsys.readouterr().err


def test_cli_rejects_mode_with_request(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--request", CALM_LOFI, "--mode", "mood-first"])
    assert excinfo.value.code == 2
