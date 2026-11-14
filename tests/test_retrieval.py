"""RAG layer: retrieval, planner augmentation and candidate narrowing (offline)."""

import json

import pytest

from evaluation.evaluate import _mock_responses
from src.agent import MockLLMClient, build_system_prompt, CatalogVocabulary, MAX_PROMPT_ARTISTS
from src.main import _mock_scenarios
from src.orchestrator import run_recommendation_workflow
from src.recommender import load_songs
from src.retrieval import HashingEmbedder, SongRetriever, song_document


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture(scope="module")
def retriever(songs):
    return SongRetriever(songs, HashingEmbedder())


def _run(scenario_id, songs, retriever, k=60):
    scenario = _mock_scenarios()[scenario_id]
    client = MockLLMClient(_mock_responses(scenario))
    result = run_recommendation_workflow(
        scenario["request"], saved_preferences=scenario.get("saved_preferences"), client=client,
        songs=songs, max_attempts=scenario.get("max_attempts", 3), request_id="t",
        retriever=retriever, retrieval_k=k,
    )
    return result, client


def test_song_document_includes_labels_and_tags():
    doc = song_document({"title": "Rain", "artist": "A", "genre": "lofi", "mood": "chill",
                         "detailed_mood": "cozy", "album": "B", "tags": "rainy day | study"})
    assert "lofi" in doc and "cozy" in doc and "rainy day" in doc and "Album: B" in doc


def test_search_ranks_matching_genre_first(songs, retriever):
    hits = retriever.search("ambient serene", k=5)
    assert len(hits) == 5 and [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert hits[0].song["genre"] == "ambient"
    assert hits[0].score >= hits[-1].score


def test_search_can_be_restricted_to_a_subset(songs, retriever):
    subset = [s for s in songs if s["genre"] == "rock"][:4]
    hits = retriever.search("ambient serene", k=10, among=subset)
    assert {h.song["id"] for h in hits} == {s["id"] for s in subset}


def test_embeddings_are_cached_on_disk(songs, tmp_path):
    first = SongRetriever(songs, HashingEmbedder(), cache_dir=tmp_path)
    assert len(list(tmp_path.glob("embeddings-*.npy"))) == 1
    second = SongRetriever(songs, HashingEmbedder(), cache_dir=tmp_path)
    assert (first.matrix == second.matrix).all()


def test_planner_receives_catalog_matches_as_hints(songs, retriever):
    result, client = _run("normal-lofi-study", songs, retriever)
    first_call = client.calls[0]
    data = json.loads(first_call["user_prompt"])
    assert data["catalog_matches"] and set(data["catalog_matches"][0]) == {
        "genre", "mood", "detailed_mood", "tags", "similarity"}
    assert "CATALOG MATCHES" in first_call["system_prompt"]
    assert result.retrieval["catalog_matches"] == data["catalog_matches"]


def test_without_retriever_the_prompt_is_unchanged(songs):
    scenario = _mock_scenarios()["normal-lofi-study"]
    client = MockLLMClient(_mock_responses(scenario))
    result = run_recommendation_workflow(scenario["request"], client=client, songs=songs, request_id="t")
    assert "catalog_matches" not in json.loads(client.calls[0]["user_prompt"])
    assert "CATALOG MATCHES" not in client.calls[0]["system_prompt"]
    assert result.retrieval is None


def test_rag_narrows_the_ranked_pool_and_explains_matches(songs, retriever):
    result, _ = _run("normal-lofi-study", songs, retriever, k=30)
    assert result.retrieval["pool"] < result.candidate_count  # ranked fewer than eligible
    assert not result.retrieval["fallback_to_all_eligible"]
    assert all(any(r.startswith("semantic match") for r in s.explanations) for s in result.songs)
    tools = [c.tool for c in result.trace.attempts[0].tool_calls]
    assert tools[:2] == ["filter_hard_constraints", "retrieve_candidates"]


def test_rag_falls_back_when_pool_is_too_small(songs, retriever):
    # k is raised to 4x the playlist size, so a tiny eligible set triggers the fallback
    result, _ = _run("impossible-small-catalog-subset", songs, retriever, k=1)
    pool, eligible = result.retrieval["pool"], result.candidate_count
    assert pool == eligible
    assert result.retrieval["fallback_to_all_eligible"] == (eligible < result.requirements.size)


def test_agent_loop_still_revises_with_rag(songs, retriever):
    result, _ = _run("successful-plan-revision", songs, retriever, k=250)
    assert len(result.trace.attempts) >= 2
    assert result.trace.attempts[1].kind == "revision"


def test_invalid_retrieval_k_is_rejected(songs, retriever):
    with pytest.raises(ValueError):
        _run("normal-lofi-study", songs, retriever, k=0)


def test_large_catalogs_do_not_list_every_artist(songs):
    vocab = CatalogVocabulary.from_songs(songs)
    many = CatalogVocabulary(vocab.genres, vocab.moods, vocab.detailed_moods, vocab.release_decades,
                             tuple(f"Artist {i}" for i in range(MAX_PROMPT_ARTISTS + 1)))
    prompt = build_system_prompt(many)
    assert "Artist 7," not in prompt and f"{MAX_PROMPT_ARTISTS + 1} catalog artists" in prompt
    assert "Neon Echo" in build_system_prompt(vocab)


def test_rag_eval_precision_and_baseline(songs, retriever):
    from evaluation.rag_eval import evaluate_retrieval, precision

    accept = {"genres": ["ambient"]}
    assert precision([s for s in songs if s["genre"] == "ambient"][:3], accept) == 1.0
    report = evaluate_retrieval(songs, [{"id": "q", "query": "ambient serene", "accept": accept}],
                                {"lexical": retriever}, k=5)
    assert report["mean_precision"]["lexical"] > report["mean_precision"]["random"]
    assert report["beats_random"] == {"lexical": 1}
