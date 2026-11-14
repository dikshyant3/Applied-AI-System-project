"""Retrieval-augmented generation (RAG) layer for TuneWise.

Each catalog song becomes a short text document (title, artist, album, genre,
mood labels and the titles of the Deezer playlists it was found in). Documents
are embedded once; a request is embedded at query time and compared by cosine
similarity. Retrieval is used in two places:

* **Augment the planner.** Before planning, the closest songs' labels and tags
  are given to the LLM as ``catalog_matches`` so it can map free-text vibes
  ("a rainy Sunday in Tokyo") to catalog genres/moods. They are hints only:
  the evidence rules still require explicit preferences to be quoted from the
  user's own words.
* **Narrow the candidates.** After hard constraints are applied, only the
  ``k`` eligible songs most similar to the request are passed to VibeMatch.
  If that pool is too small for the playlist, TuneWise falls back to every
  eligible song. Scoring, playlist rules and validation are unchanged.

The request text is embedded locally; it is never sent anywhere by this module.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

logger = logging.getLogger("tunewise.retrieval")

# Hugging Face sentence-transformers MiniLM (ONNX export run by FastEmbed, no PyTorch).
# Override with TUNEWISE_EMBEDDING_MODEL, e.g. "BAAI/bge-small-en-v1.5".
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
DEFAULT_RETRIEVAL_K = 60
CONTEXT_SONGS = 8  # songs summarized for the planner


def song_document(song: Mapping[str, Any]) -> str:
    """The text a song is retrieved by."""
    parts = [
        f"{song['title']} by {song['artist']}.",
        f"Genre: {song['genre']}. Mood: {song['mood']}, {song['detailed_mood']}.",
    ]
    if song.get("album"):
        parts.append(f"Album: {song['album']}.")
    if song.get("tags"):
        parts.append(f"Found in playlists: {song['tags']}.")
    return " ".join(parts)


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class FastEmbedEmbedder:
    """Local semantic embeddings (ONNX, no API key; the model is downloaded once from Hugging Face)."""

    def __init__(self, model: str = None):
        from fastembed import TextEmbedding  # optional dependency, imported lazily

        model = model or os.environ.get("TUNEWISE_EMBEDDING_MODEL") or DEFAULT_MODEL
        self.name = model
        self._model = TextEmbedding(model)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.passage_embed(list(texts))), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(next(iter(self._model.query_embed(text))), dtype=np.float32))


class HashingEmbedder:
    """Deterministic lexical fallback: hashed word unigrams and bigrams.

    Used in tests and when fastembed is unavailable. It matches shared words
    only (no synonyms), so it is a weaker retriever than FastEmbedEmbedder.
    """

    def __init__(self, dimensions: int = 1024):
        self.name = f"hashing-{dimensions}"
        self.dimensions = dimensions

    def _vector(self, text: str) -> np.ndarray:
        words = re.findall(r"[a-z0-9&]+", text.lower())
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in words + [f"{a} {b}" for a, b in zip(words, words[1:])]:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dimensions] += 1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(np.stack([self._vector(t) for t in texts])) if texts else np.zeros((0, self.dimensions))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text))


@dataclass(frozen=True)
class RetrievalHit:
    song: Mapping[str, Any]
    score: float
    rank: int  # 1-based among the songs searched


class SongRetriever:
    """Cosine-similarity search over song documents."""

    def __init__(self, songs: Sequence[Mapping[str, Any]], embedder: Embedder,
                 cache_dir: Optional[Path] = None):
        self.songs = list(songs)
        self.embedder = embedder
        self._index = {song["id"]: i for i, song in enumerate(self.songs)}
        documents = [song_document(song) for song in self.songs]
        self.matrix = self._load_or_embed(documents, cache_dir)

    def _load_or_embed(self, documents: List[str], cache_dir: Optional[Path]) -> np.ndarray:
        fingerprint = hashlib.sha256(
            ("\n".join(documents) + self.embedder.name).encode("utf-8")).hexdigest()[:16]
        path = None
        if cache_dir is not None:
            safe = re.sub(r"[^A-Za-z0-9.-]+", "_", self.embedder.name)
            path = Path(cache_dir) / f"embeddings-{safe}-{fingerprint}.npy"
            if path.is_file():
                matrix = np.load(path)
                if matrix.shape[0] == len(documents):
                    return matrix
        matrix = self.embedder.embed_documents(documents).astype(np.float32)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
        return matrix

    def query_vector(self, query: str) -> np.ndarray:
        return self.embedder.embed_query(query)

    def search(self, query: Any, k: int = DEFAULT_RETRIEVAL_K,
               among: Optional[Iterable[Mapping[str, Any]]] = None) -> List[RetrievalHit]:
        """Top-``k`` songs for ``query`` (text or a query vector), optionally within ``among``."""
        vector = self.query_vector(query) if isinstance(query, str) else query
        if among is None:
            rows = list(range(len(self.songs)))
        else:
            rows = [self._index[s["id"]] for s in among if s["id"] in self._index]
        if not rows or k <= 0:
            return []
        scores = self.matrix[rows] @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [RetrievalHit(self.songs[rows[i]], float(scores[i]), rank + 1)
                for rank, i in enumerate(order)]

    def planner_context(self, hits: Sequence[RetrievalHit], limit: int = CONTEXT_SONGS) -> List[Dict[str, Any]]:
        """Compact catalog evidence for the planner (labels and tags, no audio data)."""
        return [{
            "genre": hit.song["genre"], "mood": hit.song["mood"],
            "detailed_mood": hit.song["detailed_mood"],
            "tags": hit.song.get("tags", ""), "similarity": round(hit.score, 3),
        } for hit in hits[:limit]]


def build_retriever(songs: Sequence[Mapping[str, Any]], cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
                    prefer_semantic: bool = True) -> SongRetriever:
    """A semantic retriever if fastembed works, otherwise the lexical fallback."""
    if prefer_semantic:
        try:
            return SongRetriever(songs, FastEmbedEmbedder(), cache_dir)
        except Exception as error:  # missing package, no network for the first model download, ...
            logger.warning("Semantic embeddings unavailable (%s); using lexical retrieval", error)
    return SongRetriever(songs, HashingEmbedder(), cache_dir)


def summarize_hits(hits: Sequence[RetrievalHit]) -> Tuple[int, Optional[float]]:
    return len(hits), (round(hits[0].score, 4) if hits else None)
