"""Expanded catalog (250 songs: 18 original + 232 synthetic) and its generator."""

import csv
import hashlib
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from src.agent import MockLLMClient
from src.mock_responses import label, level, planner_response, requirement
from src.orchestrator import run_recommendation_workflow
from src.playlist_builder import duration_bounds_seconds
from src.recommender import SCORING_STRATEGIES, load_songs, recommend_songs

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "songs.csv"

_spec = importlib.util.spec_from_file_location("generate_songs", ROOT / "scripts" / "generate_songs.py")
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)


@pytest.fixture(scope="module")
def rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


def test_file_is_exactly_the_seeded_generator_output():
    assert CSV_PATH.read_text(encoding="utf-8") == generator.render_csv(generator.generate_rows())
    assert generator.validate_rows(generator.generate_rows()) == []


def test_original_18_records_are_preserved_byte_for_byte(rows):
    columns = generator.ORIGINAL_COLUMNS
    canonical = "\n".join(",".join(r) for r in [columns, *([row[c] for c in columns] for row in rows[:18])])
    assert hashlib.sha256(canonical.encode()).hexdigest() == generator.ORIGINAL_SHA256


def test_schema_ids_and_missing_values(rows):
    assert list(rows[0]) == generator.COLUMNS
    assert len(rows) == 250
    assert [int(r["id"]) for r in rows] == list(range(1, 251))
    assert all(value.strip() for r in rows for value in r.values())
    assert len({(r["title"].casefold(), r["artist"].casefold()) for r in rows}) == 250


def test_numeric_ranges(songs):
    for s in songs:
        for field in ("energy", "valence", "danceability", "acousticness", "instrumentalness", "speechiness"):
            assert 0 <= s[field] <= 1, (s["id"], field)
        assert 0 <= s["popularity"] <= 100
        assert s["release_decade"] % 10 == 0
        assert 60 <= s["duration_seconds"] <= 900


def test_genre_mood_and_artist_balance(songs):
    genres = Counter(s["genre"] for s in songs)
    moods = Counter(s["mood"] for s in songs)
    artists = Counter(s["artist"] for s in songs)
    assert len(genres) == 20 and max(genres.values()) - min(genres.values()) <= 1
    assert min(moods.values()) >= 5
    assert len(artists) >= 80 and max(artists.values()) <= 6


def test_detailed_moods_fit_their_mood(songs):
    for s in songs:
        assert s["detailed_mood"] in generator.DETAILED_MOODS[s["mood"]], s["id"]


@pytest.mark.parametrize("mode", list(SCORING_STRATEGIES))
def test_recommender_runs_on_full_catalog(songs, mode):
    prefs = {"favorite_genre": "jazz", "favorite_mood": "relaxed", "target_energy": 0.4}
    results = recommend_songs(prefs, songs, k=10, mode=mode)
    assert len(results) == 10
    assert all(song in songs for song, _, _ in results)
    assert all(results[i][1] >= results[i + 1][1] for i in range(9))


def test_duration_playlist_end_to_end_with_synthetic_durations(songs):
    response = planner_response(
        genre=label("jazz", evidence="jazz"),
        mood=label("relaxed", evidence="relaxed"),
        energy=level("low", reason="relaxed"),
        duration=requirement(45, "45 minutes"),
    )
    result = run_recommendation_workflow("45 minutes of relaxed jazz", client=MockLLMClient([response]), songs=songs)
    low, high = duration_bounds_seconds(45)
    assert result.status == "passed", result.validation.to_dict()
    assert result.playlist.selection_mode == "duration"
    assert low <= result.playlist.total_duration_seconds <= high
    assert "playlist_size" not in result.validation.checked
