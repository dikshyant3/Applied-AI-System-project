"""Prompt-free For You recommendations, liked-song library and un-liking."""

import pytest

from src.feedback import clear_feedback, latest_reactions, submit_feedback
from src.personalized import recommend_for_profile, starter_picks
from src.recommender import load_songs
from src.retrieval import HashingEmbedder, SongRetriever
from src.user_profile import UserProfileStore


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture
def store(tmp_path, songs):
    return UserProfileStore(tmp_path / "profile.json", songs)


def _by_genre(songs, genre):
    return [s for s in songs if s["genre"] == genre]


def test_cold_start_gives_diverse_popular_starter_picks(songs, store):
    picks = recommend_for_profile(songs, store.preferences, {}, k=8)
    assert len(picks) == 8 and all(p.kind == "starter" for p in picks)
    assert len({p.song["genre"] for p in picks}) == 8
    assert len({p.song["artist"] for p in picks}) == 8


def test_starter_picks_avoid_disliked_genres(songs, store):
    metal = _by_genre(songs, "metal")[0]
    submit_feedback(store, metal["id"], "dislike")
    picks = recommend_for_profile(songs, store.preferences, latest_reactions(store), k=10)
    assert all(p.song["genre"] != "metal" for p in picks)


def test_likes_personalize_and_rated_songs_are_excluded(songs, store):
    lofi = _by_genre(songs, "lofi")[:3]
    for song in lofi:
        submit_feedback(store, song["id"], "like")
    reactions = latest_reactions(store)
    picks = recommend_for_profile(songs, store.preferences, reactions, k=6)
    assert picks and all(p.kind == "personalized" for p in picks)
    assert picks[0].song["genre"] == "lofi"
    assert not {p.song["id"] for p in picks} & set(reactions)


def test_similarity_explains_with_the_closest_liked_song(songs, store):
    liked = _by_genre(songs, "ambient")[:2]
    for song in liked:
        submit_feedback(store, song["id"], "like")
    retriever = SongRetriever(songs, HashingEmbedder())
    picks = recommend_for_profile(songs, store.preferences, latest_reactions(store), k=4,
                                  retriever=retriever)
    titles = {s["title"] for s in liked}
    assert picks[0].reasons[0].startswith("Because you liked")
    assert any(t in picks[0].reasons[0] for t in titles)


def test_clear_feedback_unlikes_and_drops_feedback_only_preferences(songs, store):
    song = _by_genre(songs, "jazz")[0]
    submit_feedback(store, song["id"], "like")
    assert latest_reactions(store) == {song["id"]: "like"}
    assert store.preferences["favorite_genre"] == "jazz"
    clear_feedback(store, song["id"])
    assert latest_reactions(store) == {}
    assert "favorite_genre" not in store.preferences
    assert recommend_for_profile(songs, store.preferences, {}, k=3)[0].kind == "starter"


def test_clear_feedback_keeps_other_likes(songs, store):
    a, b = _by_genre(songs, "rock")[:2]
    submit_feedback(store, a["id"], "like")
    submit_feedback(store, b["id"], "like")
    clear_feedback(store, a["id"])
    assert latest_reactions(store) == {b["id"]: "like"}
    assert store.preferences["favorite_genre"] == "rock"


def test_latest_reaction_wins(songs, store):
    song = songs[0]
    submit_feedback(store, song["id"], "like")
    submit_feedback(store, song["id"], "dislike")
    assert latest_reactions(store)[song["id"]] == "dislike"


def test_invalid_k_and_fully_rated_catalog(songs):
    with pytest.raises(ValueError):
        recommend_for_profile(songs, {}, {}, k=0)
    everything = {s["id"]: "skip" for s in songs}
    assert recommend_for_profile(songs, {}, everything) == []
    assert starter_picks([], 5) == []
