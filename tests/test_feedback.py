import json

import pytest

from src.agent import MockLLMClient, create_recommendation_plan
from src.feedback import submit_feedback
from src.mock_responses import label, level, planner_response
from src.recommender import load_songs
from src.user_profile import UserProfileError, UserProfileStore


@pytest.fixture
def songs():
    return load_songs("data/songs.csv")


def test_feedback_preferences_persist_and_are_inspectable(tmp_path, songs):
    path = tmp_path / "profile.json"
    store = UserProfileStore(path, songs)

    preferences = submit_feedback(store, 2, "like")

    assert preferences["favorite_genre"] == "lofi"
    assert preferences["favorite_mood"] == "chill"
    assert preferences["target_energy"] == songs[1]["energy"]
    reloaded = UserProfileStore(path, songs)
    assert reloaded.preferences == preferences
    assert reloaded.feedback == [{"song_id": 2, "action": "like"}]
    assert json.loads(path.read_text())["version"] == 1


def test_like_dislike_and_skip_have_deterministic_updates(tmp_path, songs):
    store = UserProfileStore(tmp_path / "profile.json", songs)
    assert submit_feedback(store, 2, "like")["favorite_genre"] == "lofi"
    disliked = submit_feedback(store, 2, "dislike")
    assert "favorite_genre" not in disliked and disliked["target_energy"] == songs[1]["energy"]
    assert submit_feedback(store, 1, "skip") == disliked
    assert store.feedback == [
        {"song_id": 2, "action": "like"},
        {"song_id": 2, "action": "dislike"},
        {"song_id": 1, "action": "skip"},
    ]


@pytest.mark.parametrize("song_id", [99999, "2", True])
def test_feedback_rejects_non_catalog_song_ids(tmp_path, songs, song_id):
    store = UserProfileStore(tmp_path / "profile.json", songs)
    with pytest.raises(ValueError, match="song_id"):
        submit_feedback(store, song_id, "like")
    assert not store.feedback


def test_feedback_rejects_unknown_action(tmp_path, songs):
    store = UserProfileStore(tmp_path / "profile.json", songs)
    with pytest.raises(ValueError, match="action"):
        submit_feedback(store, 2, "love")
    with pytest.raises(ValueError, match="action"):
        submit_feedback(store, 2, [])


def test_profile_preferences_are_validated_and_resettable(tmp_path, songs):
    store = UserProfileStore(tmp_path / "profile.json", songs)
    store.save_preferences({"favorite_genre": "lofi"})
    assert store.preferences == {"favorite_genre": "lofi"}
    with pytest.raises(UserProfileError, match="catalog"):
        store.save_preferences({"favorite_mood": "invented"})
    store.reset()
    assert store.preferences == {}
    assert store.feedback == []


def test_explicit_request_preference_overrides_saved_profile(tmp_path, songs):
    store = UserProfileStore(tmp_path / "profile.json", songs)
    store.save_preferences({"favorite_genre": "jazz"})
    response = planner_response(
        genre=label("lofi", evidence="lofi"),
        mood=label("chill", evidence="calm"),
        energy=level("low", evidence="calm"),
    )
    result = create_recommendation_plan(
        "lofi calm music", store.preferences, client=MockLLMClient([response]), catalog=songs,
    )
    assert result.plan.preferences["favorite_genre"].value == "lofi"
    assert result.plan.preferences["favorite_genre"].source == "explicit"


def test_cli_feedback_profile_inspection_and_reset(tmp_path, capsys):
    from src.main import main

    profile_path = str(tmp_path / "cli-profile.json")
    assert main(["--profile-path", profile_path, "--feedback", "2", "like"]) == 0
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["preferences"]["favorite_genre"] == "lofi"
    assert recorded["feedback_count"] == 1

    assert main(["--profile-path", profile_path, "--show-profile"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["preferences"]["favorite_genre"] == "lofi"
    assert inspected["feedback_count"] == 1

    assert main(["--profile-path", profile_path, "--reset-profile"]) == 0
    assert capsys.readouterr().out == "Saved profile reset.\n"
    assert main(["--profile-path", profile_path, "--show-profile"]) == 0
    reset = json.loads(capsys.readouterr().out)
    assert reset == {"feedback_count": 0, "preferences": {}}
