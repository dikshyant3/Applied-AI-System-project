import copy
import dataclasses
import json
from pathlib import Path

import pytest

from src.recommender import SCORING_STRATEGIES, load_songs, recommend_songs, score_song
from src.schemas import (
    EXPLICIT,
    INFERRED,
    SUPPORTED_CONSTRAINT_TYPES,
    SUPPORTED_STRATEGIES,
    HardConstraint,
    PlanValidationError,
    parse_plan,
    to_recommender_kwargs,
    validate_plan,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "plans"

# Mirrors the "High-Energy Pop" profile in src/main.py.
HIGH_ENERGY_POP = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.9,
    "target_popularity": 80,
    "preferred_release_decade": 2020,
    "preferred_detailed_mood": "euphoric",
    "target_instrumentalness": 0.05,
    "target_speechiness": 0.05,
}


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


def minimal(**overrides):
    plan = {
        "preferences": {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.8,
        }
    }
    plan.update(overrides)
    return plan


def error_paths(data, catalog=None):
    return [issue.path for issue in validate_plan(data, catalog).errors]


def error_text(data, catalog=None):
    return "\n".join(str(issue) for issue in validate_plan(data, catalog).errors)


# --------------------------------------------------------------------------
# Valid plans
# --------------------------------------------------------------------------


def test_full_plan_parses_with_provenance(songs):
    data = json.loads((EXAMPLES_DIR / "valid_full_plan.json").read_text())
    plan = parse_plan(data, catalog=songs)

    assert plan.strategy == "genre-first"
    assert plan.playlist.size == 4
    assert plan.playlist.unique_artists is True
    assert plan.playlist.energy_progression == "ascending"
    assert plan.inferred_preferences() == {
        "target_popularity": 80,
        "target_instrumentalness": 0.05,
        "target_speechiness": 0.05,
    }
    assert plan.explicit_preferences()["favorite_genre"] == "pop"
    assert plan.explicit_preferences()["preferred_release_decade"] == 2020
    assert [c.type for c in plan.hard_constraints] == [
        "energy_range",
        "release_decade",
        "exclude_artist",
    ]


def test_bare_values_are_explicit_and_labels_are_normalized():
    plan = parse_plan(
        minimal(preferences={
            "favorite_genre": "  Indie Pop ",
            "favorite_mood": {"value": "HAPPY", "source": "inferred"},
            "target_energy": 1,
        })
    )
    assert plan.preferences["favorite_genre"].value == "indie pop"
    assert plan.preferences["favorite_genre"].source == EXPLICIT
    assert plan.preferences["favorite_mood"].value == "happy"
    assert plan.preferences["favorite_mood"].source == INFERRED
    assert plan.preferences["target_energy"].value == 1


def test_plan_is_immutable():
    plan = parse_plan(minimal())
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.strategy = "mood-first"
    with pytest.raises(TypeError):
        plan.preferences["target_energy"] = 0.1


def test_validation_does_not_mutate_input():
    data = json.loads((EXAMPLES_DIR / "valid_full_plan.json").read_text())
    snapshot = copy.deepcopy(data)
    parse_plan(data)
    assert data == snapshot


def test_boundary_values_are_accepted():
    plan = parse_plan(minimal(preferences={
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.0,
        "target_popularity": 100,
        "target_instrumentalness": 1.0,
        "target_speechiness": 0,
    }))
    assert plan.preferences["target_popularity"].value == 100


# --------------------------------------------------------------------------
# Omitted preferences and default preservation
# --------------------------------------------------------------------------


def test_omitted_optional_preferences_are_not_sent_to_recommender():
    kwargs = to_recommender_kwargs(parse_plan(minimal()))
    assert kwargs == {
        "user_prefs": {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.8,
        }
    }


def test_null_optional_preferences_are_treated_as_omitted():
    data = minimal()
    data["preferences"].update(
        target_popularity=None,
        preferred_release_decade={"value": None, "source": "inferred"},
    )
    assert to_recommender_kwargs(parse_plan(data)) == to_recommender_kwargs(
        parse_plan(minimal())
    )


def test_minimal_plan_reproduces_recommender_defaults(songs):
    raw_prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}
    kwargs = to_recommender_kwargs(parse_plan(minimal()))

    assert recommend_songs(songs=songs, **kwargs) == recommend_songs(raw_prefs, songs)
    _, reasons = score_song(kwargs["user_prefs"], songs[0])
    assert "release decade not specified (+0.0)" in reasons
    assert "detailed mood not specified (+0.0)" in reasons


@pytest.mark.parametrize("field", ["favorite_genre", "favorite_mood", "target_energy"])
def test_required_preferences_must_be_present(field):
    data = minimal()
    del data["preferences"][field]
    assert error_paths(data) == [f"preferences.{field}"]
    assert "required by the recommender" in error_text(data)


@pytest.mark.parametrize("value", [None, {"value": None, "source": "inferred"}])
def test_required_preferences_cannot_be_null(value):
    data = minimal()
    data["preferences"]["target_energy"] = value
    assert error_paths(data) == ["preferences.target_energy"]


def test_missing_preferences_section_reports_all_required_fields():
    assert error_paths({}) == [
        "preferences.favorite_genre",
        "preferences.favorite_mood",
        "preferences.target_energy",
    ]


# --------------------------------------------------------------------------
# Conversion to the recommender's input format
# --------------------------------------------------------------------------


def test_full_conversion_matches_existing_recommender_input(songs):
    data = {
        "preferences": dict(HIGH_ENERGY_POP),
        "strategy": "mood-first",
        "playlist": {"playlist_size": 3},
    }
    kwargs = to_recommender_kwargs(parse_plan(data, catalog=songs))

    assert kwargs == {"user_prefs": HIGH_ENERGY_POP, "k": 3, "mode": "mood-first"}
    assert recommend_songs(songs=songs, **kwargs) == recommend_songs(
        HIGH_ENERGY_POP, songs, k=3, mode="mood-first"
    )


def test_conversion_excludes_requirements_recommender_cannot_take(songs):
    data = json.loads((EXAMPLES_DIR / "valid_full_plan.json").read_text())
    kwargs = to_recommender_kwargs(parse_plan(data, catalog=songs))

    assert set(kwargs) == {"user_prefs", "k", "mode"}
    assert set(kwargs["user_prefs"]) == set(HIGH_ENERGY_POP)
    assert kwargs["mode"] == "genre-first"
    assert len(recommend_songs(songs=songs, **kwargs)) == 4


def test_conversion_rejects_plan_missing_required_preferences():
    plan = parse_plan(minimal())
    broken = dataclasses.replace(plan, preferences={})
    with pytest.raises(ValueError, match="required by the recommender"):
        to_recommender_kwargs(broken)


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def test_supported_strategies_come_from_the_recommender():
    assert SUPPORTED_STRATEGIES == tuple(SCORING_STRATEGIES)
    assert SUPPORTED_STRATEGIES == ("genre-first", "mood-first")


@pytest.mark.parametrize("raw", ["genre-first", "Genre-First", " MOOD-FIRST "])
def test_strategy_names_normalize_like_the_recommender(raw):
    assert parse_plan(minimal(strategy=raw)).strategy == raw.strip().lower()


@pytest.mark.parametrize("raw", ["popularity-first", "genre_first", "", 1, ["genre-first"]])
def test_unsupported_strategies_are_rejected(raw):
    assert error_paths(minimal(strategy=raw)) == ["strategy"]


def test_unsupported_strategy_error_lists_supported_options():
    assert "genre-first, mood-first" in error_text(minimal(strategy="balanced"))


# --------------------------------------------------------------------------
# Invalid preference values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("target_energy", 1.01),
        ("target_energy", -0.1),
        ("target_energy", float("nan")),
        ("target_energy", "0.8"),
        ("target_energy", True),
        ("target_popularity", 101),
        ("target_popularity", -1),
        ("target_instrumentalness", 2),
        ("target_speechiness", -0.5),
        ("preferred_release_decade", 2015),
        ("preferred_release_decade", 1850),
        ("preferred_release_decade", 2020.0),
        ("preferred_release_decade", "2020s"),
        ("favorite_genre", ""),
        ("favorite_genre", "   "),
        ("favorite_mood", 7),
        ("preferred_detailed_mood", ["cozy"]),
    ],
)
def test_invalid_preference_values_are_rejected(field, value):
    data = minimal()
    data["preferences"][field] = value
    assert error_paths(data) == [f"preferences.{field}"]


def test_unknown_preference_field_is_rejected():
    data = minimal()
    data["preferences"]["favourite_genre"] = "rock"
    assert error_paths(data) == ["preferences.favourite_genre"]


def test_preference_cannot_be_marked_hard():
    data = minimal()
    data["preferences"]["target_popularity"] = {"value": 70, "priority": "hard"}
    assert error_paths(data) == ["preferences.target_popularity.priority"]
    assert "hard_constraints" in error_text(data)


def test_preference_source_must_be_known():
    data = minimal()
    data["preferences"]["target_energy"] = {"value": 0.5, "source": "guessed"}
    assert error_paths(data) == ["preferences.target_energy.source"]


def test_preference_envelope_requires_value():
    data = minimal()
    data["preferences"]["target_energy"] = {"source": "inferred"}
    assert error_paths(data) == ["preferences.target_energy"]


@pytest.mark.parametrize("data", [None, [], "pop", 3])
def test_non_object_plan_is_rejected(data):
    assert error_paths(data) == ["plan"]
    with pytest.raises(PlanValidationError):
        parse_plan(data)


def test_unknown_top_level_field_is_rejected():
    assert error_paths(minimal(mood_boost=True)) == ["mood_boost"]


@pytest.mark.parametrize(
    "section, value",
    [("preferences", ["pop"]), ("playlist", 5), ("hard_constraints", {"type": "genre"})],
)
def test_malformed_sections_are_rejected(section, value):
    data = minimal()
    data[section] = value
    assert section in error_paths(data)


# --------------------------------------------------------------------------
# Playlist fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, -3, 2.5, "5", True])
def test_playlist_size_must_be_positive_integer(size):
    assert error_paths(minimal(playlist={"playlist_size": size})) == [
        "playlist.playlist_size"
    ]


def test_unique_artists_must_be_boolean():
    assert error_paths(minimal(playlist={"unique_artists": "yes"})) == [
        "playlist.unique_artists"
    ]


@pytest.mark.parametrize("progression", ["zigzag", "", 1])
def test_energy_progression_must_be_supported(progression):
    assert error_paths(minimal(playlist={"energy_progression": progression})) == [
        "playlist.energy_progression"
    ]


def test_energy_progression_needs_more_than_one_song():
    data = minimal(playlist={"playlist_size": 1, "energy_progression": "ascending"})
    assert error_paths(data) == ["playlist.energy_progression"]


def test_duration_is_rejected_because_catalog_has_no_duration():
    data = json.loads((EXAMPLES_DIR / "invalid_unsupported_duration.json").read_text())
    result = validate_plan(data)
    assert not result.is_valid
    assert [issue.path for issue in result.errors] == [
        "playlist.target_duration_minutes"
    ]
    assert "no duration metadata" in result.errors[0].message


def test_plan_still_rejects_duration_even_though_catalog_now_has_it(songs):
    # The catalog gained synthetic duration_seconds, but a plan never carries a
    # duration field: duration is handled by the playlist builder/validator.
    assert "duration_seconds" in songs[0]
    assert "target_duration_minutes" in error_text(minimal(playlist={"target_duration_minutes": 30}))


# --------------------------------------------------------------------------
# Hard constraints
# --------------------------------------------------------------------------


def test_all_supported_constraint_types_parse(songs):
    data = minimal(hard_constraints=[
        {"type": "genre", "values": ["pop", "k-pop"]},
        {"type": "mood", "values": ["happy", "playful"]},
        {"type": "detailed_mood", "values": ["euphoric", "exuberant"]},
        {"type": "release_decade", "values": [2020]},
        {"type": "exclude_artist", "values": ["Voltline"]},
        {"type": "energy_range", "min": 0.7, "max": 1.0},
        {"type": "popularity_range", "min": 50},
        {"type": "instrumentalness_range", "max": 0.1},
        {"type": "speechiness_range", "max": 0.2},
    ])
    plan = parse_plan(data, catalog=songs)
    assert tuple(c.type for c in plan.hard_constraints) == SUPPORTED_CONSTRAINT_TYPES


@pytest.mark.parametrize("constraint_type", ["tempo_bpm", "language", "explicit"])
def test_unsupported_constraint_types_are_rejected(constraint_type):
    data = minimal(hard_constraints=[{"type": constraint_type, "values": ["x"]}])
    assert error_paths(data) == ["hard_constraints[0].type"]
    assert "supported types are" in error_text(data)


@pytest.mark.parametrize("constraint_type", ["max_duration", "target_duration_minutes"])
def test_duration_constraints_are_rejected_not_dropped(constraint_type):
    data = minimal(hard_constraints=[{"type": constraint_type, "max": 30}])
    result = validate_plan(data)
    assert result.plan is None
    assert "no duration metadata" in result.errors[0].message


def test_hard_constraints_cannot_be_inferred():
    data = minimal(hard_constraints=[
        {"type": "mood", "values": ["happy"], "source": "inferred"}
    ])
    assert error_paths(data) == ["hard_constraints[0].source"]


@pytest.mark.parametrize(
    "constraint, path",
    [
        ({"type": "energy_range"}, "hard_constraints[0]"),
        ({"type": "energy_range", "min": 0.9, "max": 0.2}, "hard_constraints[0]"),
        ({"type": "energy_range", "min": 1.5}, "hard_constraints[0].min"),
        ({"type": "popularity_range", "max": "high"}, "hard_constraints[0].max"),
        ({"type": "energy_range", "values": [0.5]}, "hard_constraints[0].values"),
        ({"type": "genre", "values": []}, "hard_constraints[0].values"),
        ({"type": "genre", "values": "pop"}, "hard_constraints[0].values"),
        ({"type": "genre", "min": 1}, "hard_constraints[0].min"),
        ({"type": "release_decade", "values": [2015]}, "hard_constraints[0].values[0]"),
        ({"type": "exclude_artist", "values": [""]}, "hard_constraints[0].values[0]"),
        ({"values": ["pop"]}, "hard_constraints[0].type"),
        ("genre=pop", "hard_constraints[0]"),
    ],
)
def test_malformed_constraints_are_rejected(constraint, path):
    assert path in error_paths(minimal(hard_constraints=[constraint]))


def test_duplicate_constraint_types_are_rejected():
    data = minimal(hard_constraints=[
        {"type": "genre", "values": ["pop"]},
        {"type": "Genre", "values": ["rock"]},
    ])
    assert error_paths(data) == ["hard_constraints[1].type"]


@pytest.mark.parametrize(
    "preference, value, constraint",
    [
        ("favorite_genre", "rock", {"type": "genre", "values": ["pop"]}),
        ("favorite_mood", "sad", {"type": "mood", "values": ["happy"]}),
        ("target_energy", 0.3, {"type": "energy_range", "min": 0.6}),
        ("target_popularity", 90, {"type": "popularity_range", "max": 60}),
        ("preferred_release_decade", 2000, {"type": "release_decade", "values": [2020]}),
        ("preferred_detailed_mood", "cozy", {"type": "detailed_mood", "values": ["euphoric"]}),
    ],
)
def test_preferences_contradicting_hard_constraints_are_rejected(preference, value, constraint):
    data = minimal(hard_constraints=[constraint])
    data["preferences"][preference] = value
    assert error_paths(data) == [f"preferences.{preference}"]
    assert "contradicts hard constraint" in error_text(data)


def test_constraint_semantics():
    energy = HardConstraint("energy_range", minimum=0.5, maximum=0.8)
    assert energy.is_satisfied_by({"energy": 0.5})
    assert energy.is_satisfied_by({"energy": 0.8})
    assert not energy.is_satisfied_by({"energy": 0.81})

    genre = HardConstraint("genre", values=("pop", "rock"))
    assert genre.is_satisfied_by({"genre": "rock"})
    assert not genre.is_satisfied_by({"genre": "lofi"})

    exclude = HardConstraint("exclude_artist", values=("neon echo",))
    assert not exclude.is_satisfied_by({"artist": "Neon Echo"})
    assert exclude.is_satisfied_by({"artist": "LoRoom"})


# --------------------------------------------------------------------------
# Catalog-aware checks
# --------------------------------------------------------------------------


def test_constraint_value_missing_from_catalog_is_an_error(songs):
    data = minimal(hard_constraints=[{"type": "genre", "values": ["pop", "polka"]}])
    assert validate_plan(data).is_valid  # structurally fine without a catalog
    assert error_paths(data, songs) == ["hard_constraints[0].values"]
    assert "'polka'" in error_text(data, songs)


def test_soft_preference_missing_from_catalog_is_only_a_warning(songs):
    data = minimal()
    data["preferences"]["favorite_genre"] = "polka"
    result = validate_plan(data, songs)
    assert result.is_valid
    assert [w.path for w in result.warnings] == ["preferences.favorite_genre"]


def test_excluding_unknown_artist_is_only_a_warning(songs):
    data = minimal(hard_constraints=[{"type": "exclude_artist", "values": ["Nobody"]}])
    result = validate_plan(data, songs)
    assert result.is_valid
    assert [w.path for w in result.warnings] == ["hard_constraints[0].values"]


def test_playlist_larger_than_catalog_is_rejected(songs):
    n = len(songs)
    assert error_paths(minimal(playlist={"playlist_size": n}), songs) == []
    assert error_paths(minimal(playlist={"playlist_size": n + 1}), songs) == [
        "playlist.playlist_size"
    ]


def test_capacity_checks_can_be_deferred_to_playlist_validator(songs):
    data = minimal(playlist={"playlist_size": len(songs) + 1})
    assert validate_plan(data, songs, check_capacity=False).is_valid


def test_unique_artists_limited_by_distinct_artists(songs):
    distinct = len({song["artist"] for song in songs})
    ok = minimal(playlist={"playlist_size": distinct, "unique_artists": True})
    too_many = minimal(playlist={"playlist_size": distinct + 1, "unique_artists": True})
    assert error_paths(ok, songs) == []
    assert error_paths(too_many, songs) == ["playlist.unique_artists"]


def test_jointly_unsatisfiable_constraints_are_rejected(songs):
    data = minimal(hard_constraints=[
        {"type": "genre", "values": ["classical"]},
        {"type": "energy_range", "min": 0.9},
    ])
    data["preferences"]["favorite_genre"] = "classical"
    data["preferences"]["target_energy"] = 0.95
    assert validate_plan(data).is_valid  # each part is fine on its own
    assert error_paths(data, songs) == ["hard_constraints"]


def test_incompatible_catalog_example(songs):
    data = json.loads(
        (EXAMPLES_DIR / "invalid_incompatible_with_catalog.json").read_text()
    )
    assert validate_plan(data).is_valid
    assert error_paths(data, songs) == [
        "playlist.playlist_size",
        "playlist.unique_artists",
    ]


# --------------------------------------------------------------------------
# Error reporting and example files
# --------------------------------------------------------------------------


def test_all_errors_are_reported_together():
    data = json.loads((EXAMPLES_DIR / "invalid_many_errors.json").read_text())
    with pytest.raises(PlanValidationError) as excinfo:
        parse_plan(data)
    paths = [issue.path for issue in excinfo.value.errors]
    assert paths == [
        "preferences.favourite_decade",
        "preferences.target_energy",
        "preferences.target_popularity.priority",
        "preferences.favorite_mood",
        "strategy",
        "playlist.playlist_size",
        "hard_constraints[1].type",
        "hard_constraints[2]",
        "hard_constraints[3].source",
    ]
    assert "Invalid recommendation plan" in str(excinfo.value)


@pytest.mark.parametrize("path", sorted(EXAMPLES_DIR.glob("*.json")), ids=lambda p: p.name)
def test_example_plans_match_their_labels(path, songs):
    result = validate_plan(json.loads(path.read_text()), catalog=songs)
    assert result.is_valid is path.name.startswith("valid_")
