"""Prompt-free "For You" recommendations from the saved profile.

No LLM is involved. Two signals are combined:

1. **Saved preferences** (derived from likes/dislikes by ``feedback.py``, or
   saved during earlier requests) are scored by the unchanged VibeMatch
   ``recommend_songs``, including its artist/genre diversity penalty.
2. **Similarity to liked songs** (optional, needs a ``SongRetriever``): the
   embedding centroid of liked songs, pushed away from disliked songs, picks
   the candidate pool that VibeMatch ranks. Each pick is explained by the liked
   song it is closest to ("Because you liked ...").

Songs the user already liked, disliked or skipped are never re-recommended.
Without enough profile data (cold start) the result is a diverse set of
popular "starter picks", one per genre, so the user has something to rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from .recommender import recommend_songs
from .schemas import REQUIRED_PREFERENCES

if TYPE_CHECKING:
    from .retrieval import SongRetriever

SIMILAR_POOL = 150  # unrated songs closest to the taste centroid, ranked by VibeMatch
DISLIKE_WEIGHT = 0.5


@dataclass(frozen=True)
class Pick:
    song: Mapping[str, Any]
    score: float
    reasons: tuple
    kind: str  # "personalized" or "starter"


def recommend_for_profile(songs: Sequence[Mapping[str, Any]], preferences: Mapping[str, Any],
                          reactions: Mapping[int, str], k: int = 12,
                          retriever: Optional["SongRetriever"] = None) -> List[Pick]:
    """Up to ``k`` unrated songs for this profile, best first."""
    if k < 1:
        raise ValueError("k must be at least 1")
    unrated = [dict(s) for s in songs if s["id"] not in reactions]
    liked = [s for s in songs if reactions.get(s["id"]) == "like"]
    disliked = [s for s in songs if reactions.get(s["id"]) == "dislike"]
    if not unrated:
        return []
    if not all(field in preferences for field in REQUIRED_PREFERENCES):
        return starter_picks(unrated, k, avoid_genres={s["genre"] for s in disliked})

    pool, closest = unrated, {}
    if retriever is not None and liked:
        pool, closest = _taste_pool(retriever, unrated, liked, disliked)

    ranked = recommend_songs(dict(preferences), pool, k=k)
    picks = []
    for song, score, _explanation in ranked:
        reasons = []
        if song["id"] in closest:
            title, similarity = closest[song["id"]]
            reasons.append(f"Because you liked “{title}” (similarity {similarity:.2f})")
        if song["genre"] == preferences["favorite_genre"]:
            reasons.append(f"your favorite genre: {song['genre']}")
        if song["mood"] == preferences["favorite_mood"]:
            reasons.append(f"your favorite mood: {song['mood']}")
        reasons.append(f"energy {song['energy']:.2f} vs your {float(preferences['target_energy']):.2f}")
        picks.append(Pick(song, round(score, 3), tuple(reasons), "personalized"))
    return picks


def _taste_pool(retriever: "SongRetriever", unrated, liked, disliked):
    """Unrated songs nearest the liked-minus-disliked centroid, and each one's closest liked song."""
    import numpy as np

    def vectors(group):
        return retriever.matrix[[retriever._index[s["id"]] for s in group if s["id"] in retriever._index]]

    liked_vecs = vectors(liked)
    if liked_vecs.size == 0:
        return unrated, {}
    centroid = liked_vecs.mean(axis=0)
    disliked_vecs = vectors(disliked)
    if disliked_vecs.size:
        centroid = centroid - DISLIKE_WEIGHT * disliked_vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return unrated, {}
    hits = retriever.search(centroid / norm, k=SIMILAR_POOL, among=unrated)
    closest: Dict[Any, Any] = {}
    for hit in hits:
        sims = liked_vecs @ retriever.matrix[retriever._index[hit.song["id"]]]
        best = int(np.argmax(sims))
        closest[hit.song["id"]] = (liked[best]["title"], float(sims[best]))
    return [hit.song for hit in hits], closest


def starter_picks(unrated: Sequence[Mapping[str, Any]], k: int,
                  avoid_genres: frozenset = frozenset()) -> List[Pick]:
    """Cold start: the most popular song of each genre (round-robin), to be rated."""
    by_genre: Dict[str, List[Mapping[str, Any]]] = {}
    for song in sorted(unrated, key=lambda s: (-s["popularity"], s["id"])):
        if song["genre"] not in avoid_genres:
            by_genre.setdefault(song["genre"], []).append(song)
    genres = sorted(by_genre, key=lambda g: -by_genre[g][0]["popularity"])
    picks: List[Pick] = []
    seen_artists = set()
    depth = 0
    while len(picks) < k and any(len(by_genre[g]) > depth for g in genres):
        for genre in genres:
            if len(picks) >= k:
                break
            if len(by_genre[genre]) > depth:
                song = by_genre[genre][depth]
                if song["artist"] in seen_artists:
                    continue
                seen_artists.add(song["artist"])
                picks.append(Pick(song, float(song["popularity"]),
                                  (f"popular in {genre} — rate it to personalize",), "starter"))
        depth += 1
    return picks


__all__ = ["Pick", "recommend_for_profile", "starter_picks"]
