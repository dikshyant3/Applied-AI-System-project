"""Phase 4: playlist builder, validator, and orchestrator integration."""

import json

import pytest

from src.agent import MockLLMClient
from src.mock_responses import constraint, label, level, planner_response, requirement
from src.orchestrator import run_recommendation_workflow
from src.playlist_builder import (
    DURATION_FIELD,
    PlaylistInputError,
    PlaylistRequirements,
    build_playlist,
    duration_bounds_seconds,
)
from src.playlist_validator import PASSED, UNSUPPORTED, VIOLATED, validate_playlist
from src.recommender import load_songs, recommend_songs
from src.schemas import HardConstraint

PREFS = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.7}


def make_song(song_id, artist, energy, genre="pop", mood="happy", duration=None):
    song = {
        "id": song_id, "title": f"Song {song_id}", "artist": artist, "genre": genre, "mood": mood,
        "energy": energy, "tempo_bpm": 100, "valence": 0.5, "danceability": 0.5,
        "acousticness": 0.5, "popularity": 50, "release_decade": 2020,
        "detailed_mood": "radiant", "instrumentalness": 0.1, "speechiness": 0.05,
    }
    if duration is not None:
        song[DURATION_FIELD] = duration
    return song


@pytest.fixture
def catalog():
    return [
        make_song(1, "A", 0.70), make_song(2, "A", 0.72), make_song(3, "B", 0.40),
        make_song(4, "C", 0.90), make_song(5, "B", 0.65, mood="chill"),
        make_song(6, "D", 0.20, genre="jazz"), make_song(7, "E", 0.55, genre="rock"),
    ]


def rank_all(songs, prefs=PREFS):
    return recommend_songs(prefs, songs, k=len(songs))


def reqs(size=3, **kwargs):
    return PlaylistRequirements(size=size, **kwargs)


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def test_builder_takes_top_ranked_songs_and_preserves_them(catalog):
    ranked = rank_all(catalog)
    playlist = build_playlist(ranked, reqs(3))

    assert [e.rank for e in playlist.entries] == [1, 2, 3]
    assert [e.position for e in playlist.entries] == [1, 2, 3]
    for entry, (song, score, explanation) in zip(playlist.entries, ranked):
        assert dict(entry.song) == song
        assert entry.score == score and entry.explanation == explanation
    with pytest.raises(TypeError):
        playlist.entries[0].song["energy"] = 0.0


def test_default_playlist_equals_original_recommender_top_k(catalog):
    playlist = build_playlist(rank_all(catalog), reqs(4))
    direct = recommend_songs(PREFS, catalog, k=4)
    assert [(dict(e.song), e.score, e.explanation) for e in playlist.entries] == direct


def test_unique_artists_skips_repeats_in_rank_order(catalog):
    ranked = rank_all(catalog)
    playlist = build_playlist(ranked, reqs(4, unique_artists=True))
    artists = [e.song["artist"] for e in playlist.entries]
    assert len(set(artists)) == 4
    assert all(s.rule == "S3 unique artists" for s in playlist.skipped)
    assert [e.rank for e in playlist.entries] == sorted(e.rank for e in playlist.entries)


def test_duplicate_song_ids_are_skipped(catalog):
    ranked = rank_all(catalog)
    ranked = [ranked[0], ranked[0], *ranked[1:]]
    playlist = build_playlist(ranked, reqs(3))
    assert len({e.song["id"] for e in playlist.entries}) == 3
    assert playlist.skipped[0].rule == "S2 duplicate song"


def test_shortfall_never_relaxes_unique_artists(catalog):
    playlist = build_playlist(rank_all(catalog), reqs(7, unique_artists=True))
    assert len(playlist.entries) == 5  # only 5 distinct artists exist
    assert playlist.shortfall == 2
    assert len({e.song["artist"] for e in playlist.entries}) == 5


@pytest.mark.parametrize("direction, reverse", [("ascending", False), ("descending", True)])
def test_energy_progression_ordering(catalog, direction, reverse):
    playlist = build_playlist(rank_all(catalog), reqs(4, energy_progression=direction))
    energies = [e.song["energy"] for e in playlist.entries]
    assert energies == sorted(energies, reverse=reverse)
    assert [e.position for e in playlist.entries] == [1, 2, 3, 4]


def test_energy_ties_are_broken_by_rank():
    tied = [make_song(i, f"artist{i}", 0.5) for i in range(1, 5)]
    ranked = rank_all(tied)
    playlist = build_playlist(ranked, reqs(4, energy_progression="ascending"))
    assert [e.rank for e in playlist.entries] == [1, 2, 3, 4]


def test_builder_is_deterministic(catalog):
    ranked = rank_all(catalog)
    r = reqs(4, unique_artists=True, energy_progression="descending")
    assert build_playlist(ranked, r) == build_playlist(list(ranked), r)


@pytest.mark.parametrize(
    "bad_item, message",
    [
        ((make_song(9, "Z", "high"), 1.0, "x"), "energy must be a number"),
        ((make_song(9, "Z", 1.5), 1.0, "x"), "energy must be a number"),
        (({"id": 9, "title": "t"}, 1.0, "x"), "missing"),
        ((make_song(9, "", 0.5), 1.0, "x"), "artist must be"),
        ((make_song(9, "Z", 0.5), float("nan"), "x"), "score must be"),
        ((make_song(9, "Z", 0.5), 1.0), "expected (song, score, explanation)"),
    ],
)
def test_builder_rejects_invalid_metadata(catalog, bad_item, message):
    with pytest.raises(PlaylistInputError, match=message.replace("(", r"\(").replace(")", r"\)")):
        build_playlist([*rank_all(catalog), bad_item], reqs(3))


@pytest.mark.parametrize("kwargs", [{"size": 0}, {"size": True}, {"size": 3, "energy_progression": "zigzag"}])
def test_requirements_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        PlaylistRequirements(**kwargs)


def test_duration_driven_selection_uses_real_metadata():
    songs = [make_song(i, f"artist{i}", 0.5 + i / 100, duration=180 + i * 10) for i in range(1, 20)]
    r = reqs(5, target_duration_minutes=15, size_requested=False)
    playlist = build_playlist(rank_all(songs), r)
    low, high = duration_bounds_seconds(15)
    assert playlist.selection_mode == "duration"
    assert low <= playlist.total_duration_seconds <= high
    assert validate_playlist(playlist, r, songs).passed


def test_duration_without_metadata_falls_back_to_size(catalog):
    r = reqs(3, target_duration_minutes=15, size_requested=False)
    playlist = build_playlist(rank_all(catalog), r)
    assert playlist.selection_mode == "size" and len(playlist.entries) == 3
    assert playlist.total_duration_seconds is None


def test_explicit_size_wins_over_duration_selection():
    songs = [make_song(i, f"artist{i}", 0.5, duration=200) for i in range(1, 10)]
    playlist = build_playlist(rank_all(songs), reqs(2, target_duration_minutes=30))
    assert playlist.selection_mode == "size" and len(playlist.entries) == 2


# --------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------


def test_successful_playlist_report(catalog):
    r = reqs(4, unique_artists=True, energy_progression="ascending",
             hard_constraints=(HardConstraint("genre", values=("pop", "rock")),))
    eligible = [s for s in catalog if s["genre"] in ("pop", "rock")]
    report = validate_playlist(build_playlist(rank_all(eligible), r), r, catalog)

    assert report.passed and report.outcome == "passed" and not report.infeasible
    assert report.checked == (
        "song_metadata", "catalog_membership", "no_duplicate_songs", "playlist_size",
        "unique_artists", "hard_constraint:genre", "energy_progression",
    )
    data = report.to_dict()
    json.dumps(data)  # serializable
    assert data["violated"] == [] and data["unsupported"] == []


def test_fabricated_song_is_rejected(catalog):
    fake = make_song(99, "Ghost", 0.5)
    report = validate_playlist([catalog[0], fake], reqs(2), catalog)
    check = report.check("catalog_membership")
    assert check.status == VIOLATED
    assert check.evidence["not_in_catalog"] == [{"position": 2, "id": 99, "title": "Song 99"}]


def test_edited_metadata_is_rejected(catalog):
    edited = {**catalog[0], "energy": 0.99}
    check = validate_playlist([edited], reqs(1), catalog).check("catalog_membership")
    assert check.status == VIOLATED
    assert check.evidence["metadata_differs_from_catalog"][0]["fields"] == ["energy"]


def test_duplicate_songs_are_rejected(catalog):
    check = validate_playlist([catalog[0], catalog[0]], reqs(2), catalog).check("no_duplicate_songs")
    assert check.status == VIOLATED and check.evidence["duplicate_ids"] == {1: 2}


def test_duplicate_artists_are_rejected(catalog):
    report = validate_playlist([catalog[0], catalog[1]], reqs(2, unique_artists=True), catalog)
    check = report.check("unique_artists")
    assert check.status == VIOLATED
    assert check.evidence["repeated_artists"] == {"a": [1, 2]}
    assert not check.infeasible  # enough artists exist; this playlist is just wrong
    assert report.outcome == "failed"


def test_insufficient_songs_is_infeasible(catalog):
    r = reqs(3, hard_constraints=(HardConstraint("genre", values=("jazz",)),))
    report = validate_playlist(build_playlist(rank_all([catalog[5]]), r), r, catalog)
    check = report.check("playlist_size")
    assert check.status == VIOLATED and check.infeasible
    assert check.evidence == {"requested": 3, "actual": 1, "eligible_catalog_songs": 1}
    assert report.outcome == "infeasible"


def test_insufficient_distinct_artists_is_infeasible(catalog):
    r = reqs(6, unique_artists=True)
    report = validate_playlist(build_playlist(rank_all(catalog), r), r, catalog)
    check = report.check("playlist_size")
    assert check.infeasible and check.evidence["eligible_distinct_artists"] == 5
    assert report.check("unique_artists").status == PASSED  # no repeats were used to fill


def test_invalid_metadata_is_reported(catalog):
    bad = {**catalog[0], "energy": "loud"}
    report = validate_playlist([bad], reqs(1), catalog)
    assert report.check("song_metadata").status == VIOLATED
    assert "energy" in report.check("song_metadata").evidence["problems"][0]["problem"]


def test_hard_constraint_violation_has_evidence(catalog):
    r = reqs(2, hard_constraints=(HardConstraint("energy_range", minimum=0.6),))
    check = validate_playlist([catalog[0], catalog[2]], r, catalog).check("hard_constraint:energy_range")
    assert check.status == VIOLATED
    assert check.evidence["offending_songs"] == [{"position": 2, "id": 3, "title": "Song 3", "energy": 0.40}]


def test_unsatisfiable_constraints_are_infeasible(catalog):
    r = reqs(2, hard_constraints=(HardConstraint("mood", values=("melancholic",)),))
    report = validate_playlist([], r, catalog)
    assert report.check("hard_constraint:mood").infeasible
    assert report.outcome == "infeasible"


@pytest.mark.parametrize(
    "energies, direction, ok, problem",
    [
        ([0.2, 0.5, 0.5, 0.9], "ascending", True, None),
        ([0.2, 0.6, 0.5, 0.9], "ascending", False, "order breaks"),
        ([0.5, 0.5, 0.5], "ascending", False, "no net change"),
        ([0.9, 0.5, 0.2], "descending", True, None),
        ([0.9, 0.95, 0.2], "descending", False, "order breaks"),
        ([0.5], "ascending", False, "at least 2 songs"),
    ],
)
def test_energy_progression_rules(energies, direction, ok, problem):
    songs = [make_song(i, f"x{i}", e) for i, e in enumerate(energies, start=1)]
    check = validate_playlist(songs, reqs(len(songs), energy_progression=direction), songs).check("energy_progression")
    assert (check.status == PASSED) is ok
    if problem:
        assert problem in check.message


def test_flat_energy_catalog_makes_progression_infeasible():
    songs = [make_song(i, f"x{i}", 0.5) for i in range(1, 4)]
    check = validate_playlist(songs, reqs(3, energy_progression="ascending"), songs).check("energy_progression")
    assert check.infeasible


def test_duration_without_metadata_is_unsupported_not_passed(catalog):
    r = reqs(3, target_duration_minutes=45)
    report = validate_playlist(catalog[:3], r, catalog)
    check = report.check("duration")
    assert check.status == UNSUPPORTED and check.infeasible
    assert check.evidence["required_field"] == DURATION_FIELD
    assert not report.passed and report.outcome == "infeasible"


def test_duration_with_metadata_pass_and_fail():
    songs = [make_song(i, f"x{i}", 0.5, duration=300) for i in range(1, 7)]  # 5 min each
    ok = validate_playlist(songs[:2], reqs(2, target_duration_minutes=10), songs).check("duration")
    long = validate_playlist(songs[:4], reqs(4, target_duration_minutes=10), songs).check("duration")
    assert ok.status == PASSED
    assert long.status == VIOLATED and long.evidence["total_minutes"] == 20.0
    assert not long.infeasible


def test_duration_longer_than_catalog_is_infeasible():
    songs = [make_song(i, f"x{i}", 0.5, duration=120) for i in range(1, 4)]  # 6 min total
    check = validate_playlist(songs, reqs(3, target_duration_minutes=60), songs).check("duration")
    assert check.infeasible and check.evidence["eligible_catalog_minutes"] == 6.0


def test_unsupported_requests_are_reported(catalog):
    report = validate_playlist(catalog[:2], reqs(2, unsupported_requests=("in French",)), catalog)
    assert report.outcome == "unsupported" and not report.passed and not report.infeasible


# --------------------------------------------------------------------------
# Orchestrator integration (mock LLM, real catalog)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


def response(genre="lofi", mood="chill", playlist=None, constraints=(), duration=None):
    return planner_response(
        genre=label(genre, evidence=genre),
        mood=label(mood, evidence=mood),
        energy=level("low", reason="calm"),
        playlist=playlist,
        constraints=constraints,
        duration=duration,
    )


def run(request, llm_response, **kwargs):
    return run_recommendation_workflow(request, client=MockLLMClient([llm_response]), **kwargs)


def test_workflow_successful_playlist(songs):
    request = "5 chill lofi songs, different artists, building up"
    result = run(request, response(playlist={
        "playlist_size": requirement(5, "5 chill lofi songs"),
        "unique_artists": requirement(True, "different artists"),
        "energy_progression": requirement("ascending", "building up"),
    }))
    assert result.status == "passed", result.validation.to_dict()
    assert len(result.songs) == 5
    assert len({s.artist for s in result.songs}) == 5
    energies = [s.energy for s in result.songs]
    assert energies == sorted(energies) and energies[-1] > energies[0]
    assert all(dict(s.record) in songs for s in result.songs)


def test_workflow_impossible_artist_diversity_is_reported_not_relaxed(songs):
    lofi_artists = {s["artist"] for s in songs if s["genre"] == "lofi"}
    size = len(lofi_artists) + 1
    request = f"only lofi, {size} songs, all different artists"
    result = run(request, response(
        playlist={"playlist_size": requirement(size, f"{size} songs"),
                  "unique_artists": requirement(True, "all different artists")},
        constraints=[constraint("genre", "only lofi", values=["lofi"])],
    ))
    assert result.status == "infeasible"
    assert len(result.songs) == len(lofi_artists)  # partial, never padded
    assert all(s.genre == "lofi" for s in result.songs)
    assert len({s.artist for s in result.songs}) == len(result.songs)
    size_check = result.validation.check("playlist_size")
    assert size_check.infeasible and size_check.evidence["eligible_distinct_artists"] == len(lofi_artists)


def test_workflow_unsatisfiable_constraints_return_empty_infeasible_playlist(songs):
    artists = sorted({s["artist"] for s in songs})
    request = "exclude " + ", ".join(artists)
    result = run(request, response(constraints=[
        constraint("exclude_artist", request, values=artists)
    ]))
    assert result.candidate_count == 0 and result.songs == ()
    assert result.status == "infeasible"


def test_workflow_duration_without_metadata_is_unsupported(songs):
    no_duration = [{k: v for k, v in s.items() if k != DURATION_FIELD} for s in songs]
    result = run("calm lofi for 45 minutes", response(duration=requirement(45, "45 minutes")), songs=no_duration)
    check = result.validation.check("duration")
    assert check.status == UNSUPPORTED
    assert result.status == "infeasible" and not result.fully_satisfied
    assert result.planning.provenance_for("target_duration_minutes").evidence == "45 minutes"
    assert len(result.songs) == 5  # default size; nothing pretends to be 45 minutes
