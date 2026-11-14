"""Build data/real_songs.csv from the public Deezer API.

    python scripts/build_real_catalog.py                 # ~1,000 real tracks
    python scripts/build_real_catalog.py --per-seed 15   # quicker, smaller

Pipeline (all steps are cached under data/cache/, so re-runs are cheap and a
crashed run resumes where it stopped):

1. Seeds: each seed is a (genre, mood, detailed mood) label from TuneWise's
   vocabulary plus search phrases such as "late night jazz".
2. Collect: search Deezer playlists for each phrase and take their tracks.
   A track inherits its seed's labels (weak, playlist-level labels) and the
   titles of the playlists it appeared in become its free-text ``tags``.
3. Details: per-track lookup for release date (-> release_decade).
4. Measure: download the 30 s preview once, measure tempo/energy and audio
   proxies with librosa (src/audio_features.py), then discard the audio.
5. Normalize and write a CSV with the same columns as data/songs.csv plus
   album, tags, cover_url, deezer_id, deezer_link and feature_source.

Preview URLs expire, so they are NOT written; the web page fetches a fresh one.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_features import FEATURE_SOURCE, measure_preview, normalize_features, percentiles  # noqa: E402
from src.providers.deezer import DeezerClient, DeezerError  # noqa: E402

OUTPUT = ROOT / "data" / "real_songs.csv"
CACHE = ROOT / "data" / "cache"

COLUMNS = [
    "id", "title", "artist", "genre", "mood", "energy", "tempo_bpm", "valence", "danceability",
    "acousticness", "popularity", "release_decade", "detailed_mood", "instrumentalness",
    "speechiness", "duration_seconds", "album", "tags", "cover_url", "deezer_id", "deezer_link",
    "feature_source",
]

# (genre, mood, detailed_mood, search phrases). Labels reuse the synthetic catalog's vocabulary.
SEEDS = [
    ("lofi", "focused", "focused", ["lofi study", "lofi beats to study"]),
    ("lofi", "chill", "cozy", ["lofi chill cozy", "rainy lofi"]),
    ("jazz", "relaxed", "mellow", ["jazz cafe", "coffee shop jazz"]),
    ("jazz", "romantic", "nocturnal", ["late night jazz", "jazz night"]),
    ("pop", "happy", "euphoric", ["happy pop hits", "feel good pop"]),
    ("pop", "romantic", "passionate", ["pop love songs"]),
    ("pop", "melancholic", "lonely", ["sad pop songs", "sad songs crying"]),
    ("indie pop", "carefree", "breezy", ["indie pop summer", "indie feel good"]),
    ("indie pop", "wistful", "bittersweet", ["sad indie", "indie melancholy"]),
    ("rock", "energetic", "driving", ["rock workout", "rock gym"]),
    ("rock", "nostalgic", "reflective", ["classic rock 70s 80s"]),
    ("metal", "intense", "fierce", ["metal workout", "heavy metal"]),
    ("punk", "intense", "rebellious", ["punk rock", "pop punk"]),
    ("ambient", "chill", "serene", ["ambient sleep", "ambient relax"]),
    ("ambient", "dreamy", "ethereal", ["ambient dreamy", "space ambient"]),
    ("classical", "focused", "meditative", ["classical study", "classical focus"]),
    ("classical", "melancholic", "sorrowful", ["sad classical piano", "melancholic piano"]),
    ("hip-hop", "confident", "swaggering", ["hip hop bangers", "rap hits"]),
    ("hip-hop", "moody", "brooding", ["dark rap", "chill rap night"]),
    ("r&b", "romantic", "intimate", ["r&b love songs", "slow jams r&b"]),
    ("electronic", "energetic", "electrified", ["edm party", "dance workout"]),
    ("electronic", "focused", "focused", ["deep focus electronic", "chill house"]),
    ("synthwave", "nostalgic", "nocturnal", ["synthwave night drive", "retrowave"]),
    ("folk", "wistful", "homesick", ["folk acoustic", "indie folk"]),
    ("blues", "melancholic", "sorrowful", ["blues", "sad blues"]),
    ("country", "carefree", "sunny", ["country road trip", "country hits"]),
    ("reggae", "relaxed", "laid-back", ["reggae chill", "reggae vibes"]),
    ("k-pop", "playful", "exuberant", ["k-pop hits", "kpop dance"]),
    ("latin", "happy", "sunny", ["latin party", "reggaeton hits"]),
    ("funk", "confident", "swaggering", ["funk groove", "disco funk"]),
]


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def collect_candidates(client: DeezerClient, per_seed: int, artist_cap: int,
                       playlists_per_query: int = 2) -> List[Dict[str, Any]]:
    """Step 1-2: labelled, de-duplicated candidate tracks."""
    chosen: Dict[int, Dict[str, Any]] = {}
    seen_titles = set()
    per_artist: Counter = Counter()
    for genre, mood, detailed, queries in SEEDS:
        taken = 0
        for query in queries:
            try:
                playlists = [p for p in client.search_playlists(query, limit=6) if p.get("nb_tracks", 0) >= 15]
            except DeezerError as error:
                print(f"  ! search '{query}' failed: {error}")
                continue
            for playlist in playlists[:playlists_per_query]:
                try:
                    tracks = client.playlist_tracks(playlist["id"], limit=60)
                except DeezerError as error:
                    print(f"  ! playlist {playlist['id']} failed: {error}")
                    continue
                for track in tracks:
                    track_id = track.get("id")
                    if not track_id or not track.get("readable", True) or not track.get("preview"):
                        continue
                    if track_id in chosen:  # seen under another seed: just add the tag
                        tags = chosen[track_id]["tags"]
                        if playlist["title"] not in tags and len(tags) < 6:
                            tags.append(playlist["title"])
                        continue
                    if taken >= per_seed:
                        continue
                    artist = (track.get("artist") or {}).get("name", "").strip()
                    title = (track.get("title") or "").strip()
                    key = (title.lower(), artist.lower())
                    if not artist or not title or key in seen_titles or per_artist[artist] >= artist_cap:
                        continue
                    album = track.get("album") or {}
                    chosen[track_id] = {
                        "deezer_id": track_id, "title": title, "artist": artist,
                        "album": album.get("title", ""), "album_id": album.get("id"),
                        "cover_url": album.get("cover_big") or album.get("cover_medium") or "",
                        "duration_seconds": int(track.get("duration") or 0),
                        "rank": int(track.get("rank") or 0), "deezer_link": track.get("link", ""),
                        "genre": genre, "mood": mood, "detailed_mood": detailed,
                        "tags": [query, playlist["title"]],
                    }
                    seen_titles.add(key)
                    per_artist[artist] += 1
                    taken += 1
        print(f"  {genre:>10} / {mood:<11} {taken:3d} tracks")
    return list(chosen.values())


def add_details(client: DeezerClient, candidates: List[Dict[str, Any]], cache: Dict[str, Any]) -> None:
    """Step 3: release year (cached per track id)."""
    for index, cand in enumerate(candidates, 1):
        key = str(cand["deezer_id"])
        if key not in cache:
            try:
                detail = client.track(cand["deezer_id"])
                cache[key] = {"release_date": detail.get("release_date") or "", "bpm": detail.get("bpm") or 0}
            except DeezerError as error:
                print(f"  ! track {key}: {error}")
                cache[key] = {"release_date": "", "bpm": 0}
            if index % 100 == 0:
                _save_json(CACHE / "track_details.json", cache)
                print(f"  details {index}/{len(candidates)}")
        cand.update(cache[key])
    _save_json(CACHE / "track_details.json", cache)


def add_audio(client: DeezerClient, candidates: List[Dict[str, Any]], cache: Dict[str, Any],
              workers: int) -> None:
    """Step 4: download each preview once and measure it (cached per track id)."""
    todo = [c for c in candidates if str(c["deezer_id"]) not in cache]
    print(f"  measuring {len(todo)} previews ({len(candidates) - len(todo)} cached)")

    def work(cand):
        url = client.preview_url(cand["deezer_id"])
        if not url:
            raise DeezerError("no preview")
        return measure_preview(client.download(url))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, cand): cand for cand in todo}
        for done, future in enumerate(as_completed(futures), 1):
            cand = futures[future]
            try:
                cache[str(cand["deezer_id"])] = future.result()
            except Exception as error:  # one bad preview must not stop the build
                cache[str(cand["deezer_id"])] = {"error": str(error)[:200]}
            if done % 50 == 0:
                _save_json(CACHE / "audio_features.json", cache)
                print(f"  measured {done}/{len(todo)}")
    _save_json(CACHE / "audio_features.json", cache)


def build_rows(candidates: List[Dict[str, Any]], audio: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Step 5: normalize features and produce catalog rows."""
    usable = [c for c in candidates
              if "error" not in audio.get(str(c["deezer_id"]), {"error": "missing"})
              and c["duration_seconds"] > 0]
    raw = [audio[str(c["deezer_id"])] for c in usable]
    features = normalize_features(raw, usable)
    popularity = percentiles([c["rank"] for c in usable])
    rows = []
    for index, (cand, feats, pop) in enumerate(zip(usable, features, popularity), 1):
        year = int(cand["release_date"][:4]) if cand.get("release_date", "")[:4].isdigit() else 2020
        if cand.get("bpm"):  # Deezer's own BPM, when present, beats our estimate
            feats["tempo_bpm"] = int(round(cand["bpm"]))
        rows.append({
            "id": index, "title": cand["title"], "artist": cand["artist"],
            "genre": cand["genre"], "mood": cand["mood"], "energy": feats["energy"],
            "tempo_bpm": feats["tempo_bpm"], "valence": feats["valence"],
            "danceability": feats["danceability"], "acousticness": feats["acousticness"],
            "popularity": int(round(1 + 98 * pop)), "release_decade": max(1950, year // 10 * 10),
            "detailed_mood": cand["detailed_mood"], "instrumentalness": feats["instrumentalness"],
            "speechiness": feats["speechiness"], "duration_seconds": cand["duration_seconds"],
            "album": cand["album"], "tags": " | ".join(dict.fromkeys(cand["tags"])),
            "cover_url": cand["cover_url"], "deezer_id": cand["deezer_id"],
            "deezer_link": cand["deezer_link"], "feature_source": FEATURE_SOURCE,
        })
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-seed", type=int, default=40, help="max new tracks per seed label")
    parser.add_argument("--artist-cap", type=int, default=3, help="max tracks per artist")
    parser.add_argument("--workers", type=int, default=6, help="parallel preview downloads")
    parser.add_argument("--refresh", action="store_true", help="re-collect candidates from Deezer")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    client = DeezerClient()
    cand_path = CACHE / "candidates.json"
    candidates = None if args.refresh else _load_json(cand_path, None)
    if candidates is None:
        print("Collecting candidates from Deezer playlists...")
        candidates = collect_candidates(client, args.per_seed, args.artist_cap)
        _save_json(cand_path, candidates)
    print(f"{len(candidates)} candidate tracks")

    print("Fetching track details...")
    add_details(client, candidates, _load_json(CACHE / "track_details.json", {}))
    print("Measuring previews...")
    audio = _load_json(CACHE / "audio_features.json", {})
    add_audio(client, candidates, audio, args.workers)

    rows = build_rows(candidates, audio)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} songs to {args.output}")
    print("Genres:", dict(Counter(r["genre"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
