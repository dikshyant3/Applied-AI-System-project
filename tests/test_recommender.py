from src.recommender import (
    Recommender,
    Song,
    UserProfile,
    load_songs,
    recommend_songs,
    score_song,
)

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


ORIGINAL_SONG_COUNT = 18  # ids 1-18 are the original VibeMatch records


def original_songs():
    """The 18 original records (ids 1-18), unchanged in the expanded catalog."""
    return [song for song in load_songs("data/songs.csv") if song["id"] <= ORIGINAL_SONG_COUNT]


def test_load_songs_converts_advanced_attributes():
    songs = load_songs("data/songs.csv")

    # Expanded from 18 originals to 250 (232 synthetic) by scripts/generate_songs.py.
    assert len(songs) == 250
    assert [song["id"] for song in songs[:ORIGINAL_SONG_COUNT]] == list(range(1, 19))
    for song in songs:
        assert isinstance(song["duration_seconds"], int) and song["duration_seconds"] > 0
        assert isinstance(song["popularity"], int)
        assert 0 <= song["popularity"] <= 100
        assert isinstance(song["release_decade"], int)
        assert isinstance(song["detailed_mood"], str)
        assert isinstance(song["instrumentalness"], float)
        assert 0 <= song["instrumentalness"] <= 1
        assert isinstance(song["speechiness"], float)
        assert 0 <= song["speechiness"] <= 1
        assert isinstance(song["danceability"], float)
        assert isinstance(song["acousticness"], float)


def test_score_song_accepts_existing_preferences():
    preferences = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.9,
    }

    score, reasons = score_song(preferences, load_songs("data/songs.csv")[0])

    assert score > 0
    assert any("not specified" in reason for reason in reasons)


def test_each_advanced_attribute_changes_song_score():
    preferences = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.9,
        "target_popularity": 80,
        "preferred_release_decade": 2020,
        "preferred_detailed_mood": "euphoric",
        "target_instrumentalness": 0.05,
        "target_speechiness": 0.05,
    }
    song = load_songs("data/songs.csv")[0]
    original_score, reasons = score_song(preferences, song)

    changed_attributes = {
        "popularity": 0,
        "release_decade": 2010,
        "detailed_mood": "wistful",
        "instrumentalness": 0.9,
        "speechiness": 0.9,
    }
    for attribute, changed_value in changed_attributes.items():
        changed_song = {**song, attribute: changed_value}
        changed_score, _ = score_song(preferences, changed_song)
        assert changed_score < original_score

    assert any("popularity" in reason for reason in reasons)
    assert any("release decade" in reason for reason in reasons)
    assert any("detailed mood" in reason for reason in reasons)
    assert any("instrumentalness" in reason for reason in reasons)
    assert any("speechiness" in reason for reason in reasons)


def test_recommendations_are_ranked_by_score():
    preferences = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.9,
        "target_popularity": 80,
        "preferred_release_decade": 2020,
        "preferred_detailed_mood": "euphoric",
        "target_instrumentalness": 0.05,
        "target_speechiness": 0.05,
    }
    songs = load_songs("data/songs.csv")

    for mode in ("genre-first", "mood-first"):
        recommendations = recommend_songs(preferences, songs, k=5, mode=mode)

        assert len(recommendations) == 5
        assert all(
            recommendations[index][1] >= recommendations[index + 1][1]
            for index in range(len(recommendations) - 1)
        )


def test_scoring_modes_prioritize_their_named_preference():
    preferences = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.9,
    }
    base_song = load_songs("data/songs.csv")[0]
    genre_match = {**base_song, "genre": "pop", "mood": "curious"}
    mood_match = {**base_song, "genre": "other", "mood": "happy"}

    genre_score, genre_reasons = score_song(
        preferences, genre_match, mode="genre-first"
    )
    mood_only_score, _ = score_song(preferences, mood_match, mode="genre-first")
    genre_first_results = recommend_songs(
        preferences, [genre_match, mood_match], k=2, mode="genre-first"
    )

    assert genre_score - mood_only_score == 1.0
    assert genre_first_results[0][0] == genre_match
    assert "genre match (+2.0)" in genre_reasons
    assert "mood mismatch (+0.0)" in genre_reasons

    mood_first_genre_score, mood_first_genre_reasons = score_song(
        preferences, genre_match, mode="mood-first"
    )
    mood_score, mood_reasons = score_song(
        preferences, mood_match, mode="mood-first"
    )
    mood_first_results = recommend_songs(
        preferences, [genre_match, mood_match], k=2, mode="mood-first"
    )

    assert mood_score - mood_first_genre_score == 1.0
    assert mood_first_results[0][0] == mood_match
    assert "genre match (+1.0)" in mood_first_genre_reasons
    assert "mood mismatch (+0.0)" in mood_first_genre_reasons
    assert "genre mismatch (+0.0)" in mood_reasons
    assert "mood match (+2.0)" in mood_reasons


def test_diversity_penalty_improves_variety_and_ranks_adjusted_scores():
    preferences = {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.4,
        "target_popularity": 55,
        "preferred_release_decade": 2020,
        "preferred_detailed_mood": "cozy",
        "target_instrumentalness": 0.2,
        "target_speechiness": 0.04,
    }
    # This assertion is data-dependent: it was written for the original 18 songs
    # (3 lofi tracks). It runs against exactly those preserved records.
    songs = original_songs()
    base_ranked = sorted(
        (
            (song, score_song(preferences, song)[0])
            for song in songs
        ),
        key=lambda recommendation: recommendation[1],
        reverse=True,
    )
    recommendations = recommend_songs(preferences, songs, k=5)

    baseline_top_three_genres = {
        song["genre"] for song, _ in base_ranked[:3]
    }
    diversified_top_three_genres = {
        song["genre"] for song, _, _ in recommendations[:3]
    }
    assert len(diversified_top_three_genres) > len(baseline_top_three_genres)
    assert all(
        recommendations[index][1] >= recommendations[index + 1][1]
        for index in range(len(recommendations) - 1)
    )
    assert any("artist -1.00" in reason for _, _, reason in recommendations)
    assert any("genre -0.75" in reason for _, _, reason in recommendations)

def test_diversity_penalty_on_expanded_catalog():
    """Full 250-song catalog: penalties apply and pull in another genre within the top 5."""
    preferences = {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.4,
        "target_popularity": 55,
        "preferred_release_decade": 2020,
        "preferred_detailed_mood": "cozy",
        "target_instrumentalness": 0.2,
        "target_speechiness": 0.04,
    }
    songs = load_songs("data/songs.csv")
    base_top5 = sorted(songs, key=lambda song: score_song(preferences, song)[0], reverse=True)[:5]
    recommendations = recommend_songs(preferences, songs, k=5)

    assert len({song["genre"] for song, _, _ in recommendations}) > len({song["genre"] for song in base_top5})
    assert all(
        recommendations[index][1] >= recommendations[index + 1][1]
        for index in range(len(recommendations) - 1)
    )
