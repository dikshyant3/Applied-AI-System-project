import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


@dataclass(frozen=True)
class ScoringStrategy:
    """Weights genre and mood matches while sharing the other scoring rules."""

    name: str
    genre_weight: float
    mood_weight: float

    def score_category_matches(
        self, genre_match: bool, mood_match: bool
    ) -> Tuple[float, float]:
        genre_score = self.genre_weight if genre_match else 0.0
        mood_score = self.mood_weight if mood_match else 0.0
        return genre_score, mood_score


SCORING_STRATEGIES = {
    "genre-first": ScoringStrategy("Genre-First", genre_weight=2.0, mood_weight=1.0),
    "mood-first": ScoringStrategy("Mood-First", genre_weight=1.0, mood_weight=2.0),
}

ARTIST_DIVERSITY_PENALTY = 1.0
GENRE_DIVERSITY_PENALTY = 0.75


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Load song records from a CSV file."""
    path = Path(csv_path)
    if not path.is_absolute() and not path.is_file():
        path = Path(__file__).resolve().parent.parent / path

    with path.open(newline="", encoding="utf-8") as csv_file:
        songs = []
        for row in csv.DictReader(csv_file):
            row["id"] = int(row["id"])
            row["tempo_bpm"] = int(row["tempo_bpm"])
            row["popularity"] = int(row["popularity"])
            row["release_decade"] = int(row["release_decade"])
            if "duration_seconds" in row:  # synthetic metadata, used only by playlist rules
                row["duration_seconds"] = int(row["duration_seconds"])
            for field in (
                "energy",
                "valence",
                "danceability",
                "acousticness",
                "instrumentalness",
                "speechiness",
            ):
                row[field] = float(row[field])
            songs.append(row)
    return songs

def _get_scoring_strategy(mode: str) -> ScoringStrategy:
    normalized_mode = mode.strip().lower()
    try:
        return SCORING_STRATEGIES[normalized_mode]
    except KeyError:
        available_modes = ", ".join(SCORING_STRATEGIES)
        raise ValueError(
            f"Unknown scoring mode {mode!r}. Choose one of: {available_modes}."
        ) from None


def _score_song(
    user_prefs: Dict, song: Dict, strategy: ScoringStrategy
) -> Tuple[float, List[str]]:
    """Score a song against user preferences and explain the score."""
    genre_match = song["genre"] == user_prefs["favorite_genre"]
    mood_match = song["mood"] == user_prefs["favorite_mood"]
    energy_similarity = 1 - abs(song["energy"] - user_prefs["target_energy"])

    genre_score, mood_score = strategy.score_category_matches(
        genre_match, mood_match
    )
    energy_score = energy_similarity

    target_popularity = user_prefs.get("target_popularity", 50)
    preferred_release_decade = user_prefs.get("preferred_release_decade")
    preferred_detailed_mood = user_prefs.get("preferred_detailed_mood")
    target_instrumentalness = user_prefs.get("target_instrumentalness", 0.5)
    target_speechiness = user_prefs.get("target_speechiness", 0.5)

    popularity_score = 0.4 * (1 - abs(song["popularity"] - target_popularity) / 100)
    release_decade_score = (
        0.4 if song["release_decade"] == preferred_release_decade else 0.0
    )
    detailed_mood_score = (
        0.6 if song["detailed_mood"] == preferred_detailed_mood else 0.0
    )
    instrumentalness_score = 0.3 * (
        1 - abs(song["instrumentalness"] - target_instrumentalness)
    )
    speechiness_score = 0.3 * (1 - abs(song["speechiness"] - target_speechiness))

    if preferred_release_decade is None:
        release_decade_result = "not specified"
    elif release_decade_score:
        release_decade_result = "match"
    else:
        release_decade_result = "mismatch"

    if preferred_detailed_mood is None:
        detailed_mood_result = "not specified"
    elif detailed_mood_score:
        detailed_mood_result = "match"
    else:
        detailed_mood_result = "mismatch"

    score = (
        genre_score
        + mood_score
        + energy_score
        + popularity_score
        + release_decade_score
        + detailed_mood_score
        + instrumentalness_score
        + speechiness_score
    )

    reasons = [
        f"genre {'match' if genre_match else 'mismatch'} (+{genre_score:.1f})",
        f"mood {'match' if mood_match else 'mismatch'} (+{mood_score:.1f})",
        f"energy similarity (+{energy_score:.1f})",
        f"popularity similarity (+{popularity_score:.1f})",
        f"release decade {release_decade_result} "
        f"(+{release_decade_score:.1f})",
        f"detailed mood {detailed_mood_result} "
        f"(+{detailed_mood_score:.1f})",
        f"instrumentalness similarity (+{instrumentalness_score:.1f})",
        f"speechiness similarity (+{speechiness_score:.1f})",
    ]
    return score, reasons


def score_song(
    user_prefs: Dict, song: Dict, mode: str = "genre-first"
) -> Tuple[float, List[str]]:
    """Score one song with the selected mode and explain its contributions."""
    return _score_song(user_prefs, song, _get_scoring_strategy(mode))


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    mode: str = "genre-first",
) -> List[Tuple[Dict, float, str]]:
    """Select top songs greedily, applying diversity penalties as the list grows."""
    strategy = _get_scoring_strategy(mode)
    scored_songs = []
    for song in songs:
        score, reasons = _score_song(user_prefs, song, strategy)
        scored_songs.append((song, score, "; ".join(reasons)))

    recommendations = []
    selected_songs = []
    remaining_songs = scored_songs.copy()

    for _ in range(min(k, len(remaining_songs))):
        best_index = 0
        best_score = float("-inf")
        best_reason = ""

        for index, (song, base_score, base_reason) in enumerate(remaining_songs):
            repeats_artist = any(
                selected["artist"] == song["artist"] for selected in selected_songs
            )
            repeats_genre = any(
                selected["genre"] == song["genre"] for selected in selected_songs
            )
            artist_penalty = ARTIST_DIVERSITY_PENALTY if repeats_artist else 0.0
            genre_penalty = GENRE_DIVERSITY_PENALTY if repeats_genre else 0.0
            diversity_penalty = artist_penalty + genre_penalty
            adjusted_score = base_score - diversity_penalty

            if adjusted_score > best_score:
                best_index = index
                best_score = adjusted_score
                if diversity_penalty:
                    diversity_reason = (
                        f"diversity penalty (-{diversity_penalty:.2f}: "
                        f"artist -{artist_penalty:.2f}, genre -{genre_penalty:.2f})"
                    )
                else:
                    diversity_reason = "diversity penalty (+0.00: no artist or genre repeats)"
                best_reason = f"{base_reason}; {diversity_reason}"

        song, _, _ = remaining_songs.pop(best_index)
        recommendations.append((song, best_score, best_reason))
        selected_songs.append(song)

    return recommendations