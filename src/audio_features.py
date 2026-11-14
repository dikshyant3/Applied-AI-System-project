"""Measure audio features from a 30-second MP3 preview.

Spotify no longer offers audio features to new apps, so TuneWise measures
them itself from the Deezer preview with librosa:

* ``measure_preview`` returns *raw* per-track measurements (loudness, onset
  strength, brightness, noisiness, tempo, pulse clarity);
* ``normalize_features`` turns a whole catalog of raw measurements into the
  0-1 fields VibeMatch scores (energy, danceability, acousticness, valence,
  instrumentalness, speechiness) using catalog-relative percentiles plus
  label-based priors.

Measured directly: tempo and energy. Estimated (audio proxy and/or label
prior): danceability, acousticness, valence, instrumentalness, speechiness.
Each catalog row records this in its ``feature_source`` column.
"""

from __future__ import annotations

import io
import math
from typing import Dict, List, Mapping, Sequence

TARGET_SAMPLE_RATE = 22050

FEATURE_SOURCE = (
    "measured: tempo_bpm,energy; "
    "estimated: danceability,acousticness (audio proxy), "
    "valence (mood label + brightness), instrumentalness,speechiness (genre prior)"
)

# Label priors (estimates, documented in the model card).
VALENCE_BY_MOOD = {
    "happy": 0.85, "playful": 0.8, "carefree": 0.78, "energetic": 0.7, "confident": 0.65,
    "romantic": 0.6, "relaxed": 0.6, "chill": 0.55, "dreamy": 0.5, "focused": 0.45,
    "nostalgic": 0.45, "wistful": 0.35, "intense": 0.35, "moody": 0.3, "cathartic": 0.3,
    "melancholic": 0.2,
}
INSTRUMENTALNESS_BY_GENRE = {
    "classical": 0.9, "ambient": 0.85, "lofi": 0.7, "jazz": 0.5, "synthwave": 0.5,
    "electronic": 0.45, "funk": 0.2, "metal": 0.1, "blues": 0.1, "reggae": 0.1,
    "hip-hop": 0.02,
}
SPEECHINESS_BY_GENRE = {"hip-hop": 0.3, "r&b": 0.08, "latin": 0.08, "k-pop": 0.07, "reggae": 0.07}


def strip_id3(data: bytes) -> bytes:
    """Remove a leading ID3v2 tag; libsndfile cannot decode MP3s that start with one."""
    if len(data) >= 10 and data[:3] == b"ID3":
        size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
        footer = 10 if data[5] & 0x10 else 0
        return data[10 + size + footer:]
    return data


def decode_mp3(data: bytes):
    """Decode MP3 bytes to a mono float32 signal at TARGET_SAMPLE_RATE."""
    import numpy as np
    import soundfile as sf
    import librosa

    signal, rate = sf.read(io.BytesIO(strip_id3(data)), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if rate != TARGET_SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr=rate, target_sr=TARGET_SAMPLE_RATE)
    return np.ascontiguousarray(signal, dtype="float32"), TARGET_SAMPLE_RATE


def measure_signal(signal, rate: int) -> Dict[str, float]:
    """Raw, unnormalized measurements of one preview."""
    import numpy as np
    import librosa

    if signal.size < rate * 3:
        raise ValueError("preview is shorter than 3 seconds")
    rms = librosa.feature.rms(y=signal)[0]
    onset_env = librosa.onset.onset_strength(y=signal, sr=rate)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=rate)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=rate)
    return {
        "loudness_db": float(20 * np.log10(np.mean(rms) + 1e-9)),
        "onset_strength": float(np.mean(onset_env)),
        "brightness_hz": float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=rate))),
        "flatness": float(np.mean(librosa.feature.spectral_flatness(y=signal))),
        "tempo_bpm": float(np.atleast_1d(tempo)[0]),
        "pulse_clarity": float(np.mean(pulse)),
    }


def measure_preview(data: bytes) -> Dict[str, float]:
    signal, rate = decode_mp3(data)
    return measure_signal(signal, rate)


# ---------------------------------------------------------------------------
# Catalog-level normalization (pure Python, deterministic, testable)
# ---------------------------------------------------------------------------


def percentiles(values: Sequence[float]) -> List[float]:
    """Rank-based percentile in [0, 1] (ties share the average rank)."""
    n = len(values)
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 / (n - 1)
        i = j + 1
    return ranks


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return round(min(high, max(low, value)), 2)


# Beat trackers often report double time for slow, soft music (lofi at 160 instead of 80).
HALF_TIME_GENRES = {"lofi", "ambient", "classical", "jazz", "r&b", "blues", "folk"}


def correct_tempo(bpm: float, genre: str) -> float:
    """Halve implausibly fast tempos for genres that are almost never that fast."""
    return bpm / 2 if genre in HALF_TIME_GENRES and bpm > 145 else bpm


def _tempo_fit(bpm: float) -> float:
    """How danceable a tempo is on its own (peaks around 118 BPM)."""
    return math.exp(-(((bpm - 118.0) / 30.0) ** 2))


def normalize_features(raw: Sequence[Mapping[str, float]],
                       labels: Sequence[Mapping[str, str]]) -> List[Dict[str, float]]:
    """Map raw measurements (+ genre/mood labels) to VibeMatch's 0-1 fields."""
    if len(raw) != len(labels):
        raise ValueError("raw measurements and labels must have the same length")
    if not raw:
        return []
    loud = percentiles([r["loudness_db"] for r in raw])
    onset = percentiles([r["onset_strength"] for r in raw])
    bright = percentiles([r["brightness_hz"] for r in raw])
    flat = percentiles([r["flatness"] for r in raw])
    pulse = percentiles([r["pulse_clarity"] for r in raw])

    energy = percentiles([0.5 * loud[i] + 0.3 * onset[i] + 0.2 * bright[i] for i in range(len(raw))])
    harsh = percentiles([0.45 * bright[i] + 0.35 * loud[i] + 0.2 * flat[i] for i in range(len(raw))])

    out = []
    for i, (measure, label) in enumerate(zip(raw, labels)):
        genre, mood = label["genre"], label["mood"]
        bpm = correct_tempo(measure["tempo_bpm"], genre)
        out.append({
            "energy": _clip(0.02 + 0.96 * energy[i]),
            "tempo_bpm": int(round(bpm)),
            "danceability": _clip(0.6 * pulse[i] + 0.4 * _tempo_fit(bpm)),
            "acousticness": _clip(1.0 - harsh[i]),
            "valence": _clip(0.75 * VALENCE_BY_MOOD.get(mood, 0.5) + 0.25 * bright[i]),
            "instrumentalness": _clip(INSTRUMENTALNESS_BY_GENRE.get(genre, 0.08)),
            "speechiness": _clip(SPEECHINESS_BY_GENRE.get(genre, 0.05)),
        })
    return out
