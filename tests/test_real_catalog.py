"""Deezer client, audio feature normalization and the real-catalog builder (no network)."""

import importlib.util
import json
from pathlib import Path

import pytest

from src.audio_features import correct_tempo, normalize_features, percentiles, strip_id3
from src.providers.deezer import DeezerClient, DeezerError

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_real_catalog", ROOT / "scripts" / "build_real_catalog.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _client(routes, clock=None):
    clock = clock or FakeClock()
    calls = []

    def fetch(url):
        calls.append(url)
        for key, value in routes.items():
            if key in url:
                result = value.pop(0) if isinstance(value, list) else value
                if isinstance(result, Exception):
                    raise result
                return json.dumps(result).encode()
        raise AssertionError(f"unexpected url {url}")

    client = DeezerClient(fetch=fetch, clock=clock, sleep=clock.sleep)
    return client, calls, clock


# ---- Deezer client ------------------------------------------------------------


def test_quota_error_is_retried():
    quota = {"error": {"code": 4, "message": "Quota limit exceeded"}}
    client, calls, clock = _client({"track/1": [quota, {"id": 1, "preview": "u"}]})
    assert client.track(1)["preview"] == "u"
    assert len(calls) == 2 and 5.0 in clock.slept


def test_other_api_errors_raise():
    client, _, _ = _client({"track/1": {"error": {"code": 800, "message": "no data"}}})
    with pytest.raises(DeezerError, match="800"):
        client.track(1)


def test_network_errors_raise_after_retries():
    client, calls, _ = _client({"track/1": [OSError("down")] * 5})
    with pytest.raises(DeezerError):
        client.track(1)
    assert len(calls) == 5


def test_preview_url_is_cached_then_refreshed():
    client, calls, clock = _client({"track/9": [{"preview": "a"}, {"preview": "b"}]})
    assert client.preview_url(9) == "a"
    assert client.preview_url(9) == "a" and len(calls) == 1
    clock.now += 10 * 60  # past the cache window: preview tokens expire
    assert client.preview_url(9) == "b"


def test_missing_preview_is_none():
    client, _, _ = _client({"track/3": {"preview": ""}})
    assert client.preview_url(3) is None


# ---- audio features -------------------------------------------------------------


def test_strip_id3_removes_tag():
    tag = b"ID3\x04\x00\x00\x00\x00\x00\x05" + b"x" * 5
    assert strip_id3(tag + b"\xff\xfbAUDIO") == b"\xff\xfbAUDIO"
    assert strip_id3(b"\xff\xfbAUDIO") == b"\xff\xfbAUDIO"


def test_percentiles_handle_ties_and_single_values():
    assert percentiles([3, 1, 2]) == [1.0, 0.0, 0.5]
    assert percentiles([1, 1]) == [0.5, 0.5]
    assert percentiles([7]) == [0.5]


def test_correct_tempo_only_halves_slow_genres():
    assert correct_tempo(160, "lofi") == 80
    assert correct_tempo(160, "punk") == 160
    assert correct_tempo(120, "jazz") == 120


def _raw(loud, onset, bright, tempo=120.0):
    return {"loudness_db": loud, "onset_strength": onset, "brightness_hz": bright,
            "flatness": 0.01, "tempo_bpm": tempo, "pulse_clarity": 0.5}


def test_normalize_features_orders_energy_and_stays_in_range():
    raw = [_raw(-30, 0.5, 900), _raw(-10, 2.0, 3000), _raw(-20, 1.0, 1500)]
    labels = [{"genre": "ambient", "mood": "chill"}, {"genre": "metal", "mood": "intense"},
              {"genre": "pop", "mood": "happy"}]
    out = normalize_features(raw, labels)
    assert out[0]["energy"] < out[2]["energy"] < out[1]["energy"]
    assert out[0]["acousticness"] > out[1]["acousticness"]
    assert out[0]["instrumentalness"] > out[2]["instrumentalness"]
    assert out[2]["valence"] > out[1]["valence"]
    for row in out:
        for field in ("energy", "valence", "danceability", "acousticness", "instrumentalness", "speechiness"):
            assert 0.0 <= row[field] <= 1.0


def test_normalize_features_requires_matching_lengths():
    with pytest.raises(ValueError):
        normalize_features([_raw(-10, 1, 1000)], [])


def test_measure_signal_on_synthetic_clicks():
    np = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    from src.audio_features import measure_signal

    rate = 22050
    signal = np.zeros(rate * 10, dtype=np.float32)
    for beat in np.arange(0, 10, 0.5):  # 120 BPM click track
        start = int(beat * rate)
        signal[start:start + 400] = 0.8
    measured = measure_signal(signal, rate)
    assert abs(measured["tempo_bpm"] - 120) < 6
    with pytest.raises(ValueError):
        measure_signal(signal[:rate], rate)


# ---- catalog builder ----------------------------------------------------------


class FakeDeezer:
    def __init__(self):
        self.tracks = {
            "lofi study": [_track(1, "A", "Song A"), _track(2, "A", "Song B"), _track(3, "B", "Song C")],
        }

    def search_playlists(self, query, limit=5):
        return [{"id": hash(query) % 1000, "title": f"{query} playlist", "nb_tracks": 30, "q": query}]

    def playlist_tracks(self, playlist_id, limit=50):
        for query, tracks in self.tracks.items():
            if hash(query) % 1000 == playlist_id:
                return tracks
        return [_track(1, "A", "Song A")]  # every other playlist repeats track 1


def _track(track_id, artist, title):
    return {"id": track_id, "title": title, "readable": True, "preview": "p", "duration": 180,
            "rank": 1000 * track_id, "link": f"https://www.deezer.com/track/{track_id}",
            "artist": {"name": artist}, "album": {"id": 5, "title": "Alb", "cover_big": "c.jpg"}}


def test_collect_candidates_dedupes_caps_artists_and_merges_tags(monkeypatch):
    monkeypatch.setattr(builder, "SEEDS", [
        ("lofi", "focused", "focused", ["lofi study"]),
        ("jazz", "relaxed", "mellow", ["jazz cafe"]),
    ])
    found = builder.collect_candidates(FakeDeezer(), per_seed=10, artist_cap=1)
    by_id = {c["deezer_id"]: c for c in found}
    assert set(by_id) == {1, 3}  # track 2 dropped: artist A already used once
    assert by_id[1]["genre"] == "lofi"  # first seed wins the labels
    assert "jazz cafe playlist" in by_id[1]["tags"]  # later playlists add tags


def test_build_rows_produces_catalog_compatible_rows(tmp_path):
    candidates = [
        {**_cand(1), "genre": "lofi", "mood": "chill", "release_date": "2021-05-01"},
        {**_cand(2), "genre": "rock", "mood": "energetic", "release_date": "", "bpm": 140},
        {**_cand(3), "genre": "pop", "mood": "happy", "release_date": "1987-01-01"},
    ]
    audio = {"1": _raw(-25, 0.6, 1000, 160), "2": _raw(-8, 2.0, 2800), "3": {"error": "no preview"}}
    rows = builder.build_rows(candidates, audio)
    assert [r["id"] for r in rows] == [1, 2]
    assert rows[0]["tempo_bpm"] == 80 and rows[1]["tempo_bpm"] == 140  # halved lofi; Deezer BPM wins
    assert rows[0]["release_decade"] == 2020 and rows[1]["release_decade"] == 2020
    path = tmp_path / "real.csv"
    builder.write_csv(rows, path)

    from src.recommender import load_songs, recommend_songs
    songs = load_songs(str(path))
    assert songs[0]["cover_url"] == "c.jpg" and songs[0]["deezer_id"] == "1"
    prefs = {"favorite_genre": "rock", "favorite_mood": "energetic", "target_energy": 0.9}
    assert recommend_songs(prefs, songs, k=1)[0][0]["genre"] == "rock"


def _cand(track_id):
    return {"deezer_id": track_id, "title": f"T{track_id}", "artist": f"A{track_id}", "album": "Alb",
            "cover_url": "c.jpg", "duration_seconds": 200, "rank": track_id * 10,
            "deezer_link": "l", "detailed_mood": "focused", "tags": ["q", "Playlist"]}
