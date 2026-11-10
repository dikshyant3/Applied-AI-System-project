"""
Reproducibly expand data/songs.csv from 18 to 250 songs.

SYNTHETIC DATA NOTICE
---------------------
* Songs with id 1-18 are the original VibeMatch records. Their 15 original
  fields are preserved byte-for-byte (verified against a SHA-256 below).
* Songs with id 19-250 are SYNTHETIC: fictional titles and artists generated
  from word lists, with feature values drawn from hand-written per-genre
  ranges. They are not real songs; any resemblance to real titles or artists
  is coincidental.
* ``duration_seconds`` is SYNTHETIC for ALL 250 songs, including the original
  18 (which never had durations). Values come from per-genre ranges.

Usage
-----
    python scripts/generate_songs.py            # (re)write data/songs.csv and print a summary
    python scripts/generate_songs.py --check    # verify the file matches the generator; no writes

The output depends only on SEED and the original 18 records, so re-running it
produces an identical file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SEED = 20251107
TOTAL_SONGS = 250
ORIGINAL_COUNT = 18
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"

ORIGINAL_COLUMNS = [
    "id", "title", "artist", "genre", "mood", "energy", "tempo_bpm", "valence",
    "danceability", "acousticness", "popularity", "release_decade", "detailed_mood",
    "instrumentalness", "speechiness",
]
COLUMNS = ORIGINAL_COLUMNS + ["duration_seconds"]
# SHA-256 of the original file (header + 18 rows, LF-separated, 15 columns).
ORIGINAL_SHA256 = "91af28a0671fd2db597036df725b925ea81d9abaf31158953c1d954ad64ec654"

# Mood -> compatible detailed moods. The first entries are the original labels.
DETAILED_MOODS: Dict[str, Tuple[str, ...]] = {
    "happy": ("euphoric", "radiant", "sunny"),
    "chill": ("cozy", "serene", "meditative", "hazy"),
    "intense": ("defiant", "triumphant", "fierce"),
    "relaxed": ("intimate", "laid-back", "mellow"),
    "moody": ("nocturnal", "brooding"),
    "focused": ("focused", "determined"),
    "nostalgic": ("homesick", "bittersweet"),
    "romantic": ("tender", "passionate"),
    "confident": ("assertive", "swaggering"),
    "carefree": ("carefree", "breezy"),
    "cathartic": ("rebellious", "explosive"),
    "wistful": ("reflective", "yearning"),
    "playful": ("exuberant", "whimsical"),
    "melancholic": ("sorrowful", "lonely"),
    "energetic": ("driving", "electrified"),
    "dreamy": ("ethereal", "floating"),
}

ALL_DECADES = (1980, 1990, 2000, 2010, 2020)

# Per-genre feature ranges (inclusive). Keys: energy, tempo, valence, dance,
# acoustic, instr, speech, popularity, duration (seconds), moods, decades.
GENRES: Dict[str, dict] = {
    "pop": dict(energy=(.55, .95), tempo=(95, 135), valence=(.5, .95), dance=(.6, .92), acoustic=(.02, .35), instr=(0, .05), speech=(.03, .10), popularity=(55, 95), duration=(165, 235), moods=("happy", "playful", "romantic", "confident", "intense", "carefree", "energetic"), decades=(2000, 2010, 2020)),
    "lofi": dict(energy=(.2, .5), tempo=(65, 90), valence=(.35, .7), dance=(.45, .7), acoustic=(.55, .92), instr=(.1, .6), speech=(.02, .06), popularity=(30, 70), duration=(110, 200), moods=("chill", "focused", "relaxed", "dreamy", "wistful", "melancholic"), decades=(2010, 2020)),
    "rock": dict(energy=(.6, .95), tempo=(100, 165), valence=(.3, .75), dance=(.4, .7), acoustic=(.03, .3), instr=(0, .15), speech=(.03, .08), popularity=(40, 85), duration=(190, 300), moods=("intense", "cathartic", "confident", "nostalgic", "energetic", "moody"), decades=ALL_DECADES),
    "ambient": dict(energy=(.1, .4), tempo=(55, 90), valence=(.3, .7), dance=(.2, .5), acoustic=(.6, .98), instr=(.7, .98), speech=(.02, .05), popularity=(20, 55), duration=(240, 480), moods=("chill", "dreamy", "focused", "wistful", "melancholic"), decades=(1990, 2000, 2010, 2020)),
    "jazz": dict(energy=(.2, .6), tempo=(70, 160), valence=(.4, .8), dance=(.4, .7), acoustic=(.6, .95), instr=(.2, .9), speech=(.03, .07), popularity=(25, 65), duration=(210, 420), moods=("relaxed", "romantic", "nostalgic", "playful", "moody", "chill"), decades=ALL_DECADES),
    "synthwave": dict(energy=(.55, .9), tempo=(90, 125), valence=(.3, .7), dance=(.6, .85), acoustic=(.02, .25), instr=(.3, .85), speech=(.02, .06), popularity=(35, 75), duration=(200, 320), moods=("moody", "nostalgic", "energetic", "dreamy", "confident"), decades=(2010, 2020)),
    "indie pop": dict(energy=(.45, .85), tempo=(95, 130), valence=(.5, .9), dance=(.55, .85), acoustic=(.1, .5), instr=(0, .1), speech=(.03, .07), popularity=(40, 80), duration=(180, 250), moods=("happy", "wistful", "carefree", "playful", "dreamy", "romantic"), decades=(2000, 2010, 2020)),
    "country": dict(energy=(.4, .75), tempo=(80, 130), valence=(.45, .85), dance=(.5, .75), acoustic=(.4, .85), instr=(0, .08), speech=(.03, .06), popularity=(35, 80), duration=(170, 250), moods=("nostalgic", "carefree", "romantic", "melancholic", "happy"), decades=(1990, 2000, 2010, 2020)),
    "classical": dict(energy=(.08, .45), tempo=(55, 130), valence=(.2, .75), dance=(.15, .45), acoustic=(.85, .99), instr=(.8, .98), speech=(.02, .05), popularity=(20, 60), duration=(240, 600), moods=("romantic", "melancholic", "focused", "wistful", "chill", "dreamy"), decades=ALL_DECADES),
    "hip-hop": dict(energy=(.55, .9), tempo=(80, 110), valence=(.35, .8), dance=(.65, .93), acoustic=(.02, .3), instr=(0, .08), speech=(.1, .38), popularity=(50, 92), duration=(170, 260), moods=("confident", "intense", "playful", "moody", "energetic", "nostalgic"), decades=(1990, 2000, 2010, 2020)),
    "reggae": dict(energy=(.45, .75), tempo=(70, 105), valence=(.6, .92), dance=(.65, .88), acoustic=(.15, .5), instr=(0, .15), speech=(.04, .12), popularity=(30, 70), duration=(190, 280), moods=("carefree", "relaxed", "happy", "romantic", "chill"), decades=ALL_DECADES),
    "metal": dict(energy=(.82, .99), tempo=(120, 190), valence=(.15, .5), dance=(.3, .6), acoustic=(0, .08), instr=(0, .3), speech=(.04, .12), popularity=(30, 70), duration=(220, 380), moods=("cathartic", "intense", "moody", "energetic"), decades=ALL_DECADES),
    "folk": dict(energy=(.2, .55), tempo=(70, 120), valence=(.3, .7), dance=(.35, .6), acoustic=(.75, .97), instr=(.02, .4), speech=(.03, .06), popularity=(25, 65), duration=(170, 280), moods=("wistful", "nostalgic", "melancholic", "romantic", "relaxed"), decades=(1990, 2000, 2010, 2020)),
    "k-pop": dict(energy=(.65, .95), tempo=(100, 135), valence=(.6, .95), dance=(.7, .93), acoustic=(.02, .2), instr=(0, .03), speech=(.05, .15), popularity=(60, 95), duration=(180, 230), moods=("playful", "happy", "confident", "energetic", "romantic"), decades=(2010, 2020)),
    "blues": dict(energy=(.25, .6), tempo=(65, 120), valence=(.2, .55), dance=(.35, .6), acoustic=(.5, .9), instr=(.02, .3), speech=(.03, .07), popularity=(20, 60), duration=(200, 330), moods=("melancholic", "wistful", "moody", "relaxed", "nostalgic"), decades=ALL_DECADES),
    "r&b": dict(energy=(.35, .75), tempo=(70, 110), valence=(.35, .8), dance=(.6, .85), acoustic=(.1, .5), instr=(0, .06), speech=(.04, .12), popularity=(45, 88), duration=(180, 260), moods=("romantic", "confident", "moody", "relaxed", "dreamy"), decades=(1990, 2000, 2010, 2020)),
    "electronic": dict(energy=(.6, .97), tempo=(118, 140), valence=(.35, .85), dance=(.7, .95), acoustic=(0, .15), instr=(.4, .9), speech=(.03, .08), popularity=(40, 85), duration=(200, 360), moods=("energetic", "happy", "intense", "dreamy", "focused"), decades=(2000, 2010, 2020)),
    "funk": dict(energy=(.6, .9), tempo=(95, 120), valence=(.65, .95), dance=(.75, .95), acoustic=(.05, .35), instr=(.05, .4), speech=(.04, .10), popularity=(35, 75), duration=(200, 300), moods=("playful", "happy", "confident", "carefree", "energetic"), decades=ALL_DECADES),
    "latin": dict(energy=(.6, .92), tempo=(90, 130), valence=(.65, .95), dance=(.75, .95), acoustic=(.1, .45), instr=(0, .06), speech=(.04, .12), popularity=(45, 90), duration=(180, 250), moods=("happy", "romantic", "playful", "carefree", "energetic"), decades=(2000, 2010, 2020)),
    "punk": dict(energy=(.8, .98), tempo=(150, 200), valence=(.3, .7), dance=(.35, .6), acoustic=(0, .1), instr=(0, .05), speech=(.05, .15), popularity=(30, 70), duration=(110, 190), moods=("cathartic", "intense", "energetic", "playful"), decades=ALL_DECADES),
}

ARTIST_WORDS_A = (
    "Amber", "Cobalt", "Velvet", "Hollow", "Silver", "Saffron", "Lumen", "Copper", "Juniper",
    "Marble", "Indigo", "Static", "Paper", "Crimson", "Willow", "Granite", "Opal", "Hazel",
    "Polar", "Tidal", "Ember", "Cinder", "Lattice", "Signal", "Meridian", "Prism", "Harbor",
    "Fable", "Orchid", "Quartz", "Rustle", "Solace", "Tundra", "Vesper", "Wren", "Zephyr",
)
ARTIST_WORDS_B = (
    "Atlas", "Lanterns", "Collective", "Avenue", "Orchestra", "Parade", "Circuit", "Season",
    "Choir", "Arcade", "Tides", "Engines", "Garden", "Society", "Signals", "Theory", "Assembly",
    "Hours", "Rooms", "Satellites", "Union", "Motel", "Kites", "Foxes", "Canyon", "Static",
)
FIRST_NAMES = (
    "Ines", "Tavi", "Rowan", "Kenji", "Odile", "Marisol", "Bram", "Leilani", "Dario", "Sunniva",
    "Ezra", "Noor", "Callum", "Yara", "Teodor", "Amaya", "Joaquin", "Freya", "Kofi", "Liesel",
)
LAST_NAMES = (
    "Vale", "Okafor", "Lindqvist", "Moreau", "Castell", "Ashby", "Novak", "Reyes", "Halloran",
    "Takeda", "Brandt", "Quill", "Sorensen", "Delacroix", "Mbeki", "Ferreira", "Larkspur",
)
TITLE_WORDS_A = (
    "Paper", "Glass", "Velvet", "Neon", "Quiet", "Hollow", "Silver", "Northern", "Electric",
    "Lantern", "Copper", "Tidal", "Marble", "Saffron", "Static", "Distant", "Crimson", "Amber",
    "Midnight", "Morning", "Faded", "Wandering", "Slow", "Cobalt", "Borrowed", "Second",
    "Folded", "Restless", "Gentle", "Burning", "Orbit", "Window", "Harbor", "Canvas", "Satellite",
)
TITLE_WORDS_B = (
    "Postcards", "Engines", "Harbor", "Echoes", "Gardens", "Signals", "Rooftops", "Letters",
    "Currents", "Lanterns", "Mirrors", "Avenues", "Comets", "Embers", "Tides", "Static", "Waltz",
    "Parade", "Horizon", "Corridor", "Monsoon", "Fireflies", "Compass", "Daydream", "Cassette",
    "Skyline", "Orchard", "Voltage", "Lullaby", "Telegram", "Hourglass", "Driftwood", "Prism",
)


def _original_rows() -> List[List[str]]:
    """Read ids 1-18 from data/songs.csv and verify they are untouched."""
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    if header[: len(ORIGINAL_COLUMNS)] != ORIGINAL_COLUMNS:
        raise SystemExit(f"Unexpected columns in {CSV_PATH}: {header}")
    originals = [row[: len(ORIGINAL_COLUMNS)] for row in body if 1 <= int(row[0]) <= ORIGINAL_COUNT]
    canonical = "\n".join(",".join(r) for r in [ORIGINAL_COLUMNS, *originals])
    if len(originals) != ORIGINAL_COUNT or hashlib.sha256(canonical.encode()).hexdigest() != ORIGINAL_SHA256:
        raise SystemExit("The original 18 records differ from the verified originals; refusing to continue.")
    return originals


def _uniform(rng: random.Random, bounds: Tuple[float, float]) -> float:
    return rng.uniform(*bounds)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _new_unique_name(rng: random.Random, taken: set, make_name) -> str:
    """Call ``make_name`` until it returns a name not used yet; remember it."""
    while True:
        name = make_name(rng)
        if name not in taken:
            taken.add(name)
            return name


def _artist_name(rng: random.Random) -> str:
    if rng.random() < 0.35:
        return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    return f"{rng.choice(ARTIST_WORDS_A)} {rng.choice(ARTIST_WORDS_B)}"


def _title(rng: random.Random) -> str:
    return f"{rng.choice(TITLE_WORDS_A)} {rng.choice(TITLE_WORDS_B)}"


def _genre_targets() -> Dict[str, int]:
    """Spread TOTAL_SONGS as evenly as possible across genres."""
    base, extra = divmod(TOTAL_SONGS, len(GENRES))
    return {genre: base + (1 if index < extra else 0) for index, genre in enumerate(GENRES)}


def generate_rows() -> List[List[str]]:
    rng = random.Random(SEED)
    originals = _original_rows()

    # Synthetic durations for the 18 originals (they never had one).
    rows = [[*row, str(rng.randint(*GENRES[row[GENRE]]["duration"]))] for row in originals]

    existing_per_genre = Counter(row[GENRE] for row in originals)
    titles_taken = {row[TITLE] for row in originals}
    artists_taken = {row[ARTIST] for row in originals}
    original_artists = _original_artists_by_genre(originals)

    next_id = ORIGINAL_COUNT + 1
    for genre, target in _genre_targets().items():
        profile = GENRES[genre]
        needed = target - existing_per_genre.get(genre, 0)
        artists = _artist_slots(rng, needed, original_artists.get(genre, []), artists_taken)

        # Moods: round-robin over a shuffled list for an even spread within the genre.
        moods = list(profile["moods"])
        rng.shuffle(moods)
        for index in range(needed):
            mood = moods[index % len(moods)]
            title = _new_unique_name(rng, titles_taken, _title)
            rows.append(_song_row(rng, next_id, title, artists[index], genre, mood, profile))
            next_id += 1
    return rows


# Column positions in a row.
TITLE, ARTIST, GENRE = 1, 2, 3


def _original_artists_by_genre(originals: Sequence[Sequence[str]]) -> Dict[str, List[str]]:
    """Artists of the original songs, per genre, in first-seen order."""
    by_genre: Dict[str, List[str]] = {}
    for row in originals:
        artists = by_genre.setdefault(row[GENRE], [])
        if row[ARTIST] not in artists:
            artists.append(row[ARTIST])
    return by_genre


def _artist_slots(rng: random.Random, needed: int, original_artists: List[str], artists_taken: set) -> List[str]:
    """One artist name per new song in a genre: original artists may get 0-2 more songs, the rest are new."""
    slots: List[str] = []
    for artist in original_artists:
        slots += [artist] * rng.choice((0, 1, 2))
    while len(slots) < needed:
        artist = _new_unique_name(rng, artists_taken, _artist_name)
        slots += [artist] * rng.choice((1, 2, 2, 3, 3, 4))
    slots = slots[:needed]
    rng.shuffle(slots)
    return slots


def _song_row(rng: random.Random, song_id: int, title: str, artist: str, genre: str, mood: str, profile: dict) -> List[str]:
    """One synthetic song. The random draws happen in column order."""
    return [
        str(song_id), title, artist, genre, mood,
        _fmt(_uniform(rng, profile["energy"])),
        str(rng.randint(*profile["tempo"])),
        _fmt(_uniform(rng, profile["valence"])),
        _fmt(_uniform(rng, profile["dance"])),
        _fmt(_uniform(rng, profile["acoustic"])),
        str(rng.randint(*profile["popularity"])),
        str(rng.choice(profile["decades"])),
        rng.choice(DETAILED_MOODS[mood]),
        _fmt(_uniform(rng, profile["instr"])),
        _fmt(_uniform(rng, profile["speech"])),
        str(rng.randint(*profile["duration"])),
    ]


def render_csv(rows: Sequence[Sequence[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Validation and summary
# --------------------------------------------------------------------------


NUMERIC_RANGES = {
    "energy": (0.0, 1.0), "valence": (0.0, 1.0), "danceability": (0.0, 1.0),
    "acousticness": (0.0, 1.0), "instrumentalness": (0.0, 1.0), "speechiness": (0.0, 1.0),
    "popularity": (0, 100), "tempo_bpm": (40, 220), "duration_seconds": (60, 900),
}


def validate_rows(rows: Sequence[Sequence[str]]) -> List[str]:
    records = [dict(zip(COLUMNS, row)) for row in rows]
    problems: List[str] = []
    if len(records) != TOTAL_SONGS:
        problems.append(f"expected {TOTAL_SONGS} songs, found {len(records)}")
    if [r["id"] for r in records] != [str(i) for i in range(1, len(records) + 1)]:
        problems.append("ids must be unique and run 1..N in order")
    for record in records:
        problems += _record_problems(record)
    problems += _uniqueness_problems(records)
    problems += _balance_problems(records)
    return problems


def _record_problems(r: Dict[str, str]) -> List[str]:
    """Missing values, numeric ranges, decade and mood consistency for one song."""
    problems = []
    for column in COLUMNS:
        if r.get(column, "").strip() == "":
            problems.append(f"id {r['id']}: missing {column}")
    for column, (low, high) in NUMERIC_RANGES.items():
        try:
            value = float(r[column])
        except (KeyError, ValueError):
            problems.append(f"id {r['id']}: {column} is not numeric")
            continue
        if not low <= value <= high:
            problems.append(f"id {r['id']}: {column}={value} outside [{low}, {high}]")
    if int(r["release_decade"]) % 10:
        problems.append(f"id {r['id']}: release_decade must be a decade")
    if r["mood"] in DETAILED_MOODS and r["detailed_mood"] not in DETAILED_MOODS[r["mood"]]:
        problems.append(f"id {r['id']}: detailed_mood {r['detailed_mood']!r} does not fit mood {r['mood']!r}")
    return problems


def _uniqueness_problems(records: Sequence[Dict[str, str]]) -> List[str]:
    problems = []
    if len({(r["title"].casefold(), r["artist"].casefold()) for r in records}) != len(records):
        problems.append("duplicate (title, artist) pairs")
    if len({r["title"].casefold() for r in records}) != len(records):
        problems.append("duplicate titles")
    return problems


def _balance_problems(records: Sequence[Dict[str, str]]) -> List[str]:
    problems = []
    genres = Counter(r["genre"] for r in records)
    if max(genres.values()) - min(genres.values()) > 1:
        problems.append(f"genres are unbalanced: {dict(genres)}")
    moods = Counter(r["mood"] for r in records)
    if min(moods.values()) < 5:
        problems.append(f"some moods have fewer than 5 songs: {dict(moods)}")
    artists = Counter(r["artist"] for r in records)
    if len(artists) < 80:
        problems.append(f"only {len(artists)} distinct artists")
    if max(artists.values()) > 6:
        problems.append(f"an artist has more than 6 songs: {artists.most_common(1)}")
    return problems


def summarize(rows: Sequence[Sequence[str]]) -> str:
    records = [dict(zip(COLUMNS, row)) for row in rows]

    def counts(column: str) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(Counter(r[column] for r in records).items()))

    def buckets(column: str, edges: Sequence[float]) -> str:
        """Counts per [low-high) bucket; the last bucket also includes its upper edge."""
        values = [float(r[column]) for r in records]
        parts = []
        for low, high in zip(edges, edges[1:]):
            is_last = high == edges[-1]
            count = sum(1 for v in values if low <= v < high or (is_last and v == high))
            closing = "]" if is_last else ")"
            parts.append(f"[{low:g}-{high:g}{closing}={count}")
        return ", ".join(parts)

    durations = [int(r["duration_seconds"]) for r in records]
    artists = Counter(r["artist"] for r in records)
    return "\n".join([
        f"songs: {len(records)} (original: {ORIGINAL_COUNT}, synthetic: {len(records) - ORIGINAL_COUNT})",
        f"genres ({len(set(r['genre'] for r in records))}): {counts('genre')}",
        f"moods ({len(set(r['mood'] for r in records))}): {counts('mood')}",
        f"detailed moods: {len(set(r['detailed_mood'] for r in records))} distinct",
        f"release decades: {counts('release_decade')}",
        f"energy: {buckets('energy', [0, .2, .4, .6, .8, 1])}",
        f"popularity: {buckets('popularity', [0, 20, 40, 60, 80, 100])}",
        f"duration_seconds: min={min(durations)} median={statistics.median(durations):g} "
        f"max={max(durations)} total={sum(durations) / 3600:.1f}h",
        f"artists: {len(artists)} distinct; songs per artist min={min(artists.values())} "
        f"max={max(artists.values())}",
    ])


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify data/songs.csv without writing")
    args = parser.parse_args(argv)

    rows = generate_rows()
    problems = validate_rows(rows)
    if problems:
        print("Generated data failed validation:", *problems, sep="\n  - ", file=sys.stderr)
        return 1
    content = render_csv(rows)

    if args.check:
        current = CSV_PATH.read_text(encoding="utf-8")
        if current != content:
            print(f"{CSV_PATH} does not match the generator output", file=sys.stderr)
            return 1
        print(f"{CSV_PATH} matches the generator (seed {SEED}) and passes validation")
    else:
        CSV_PATH.write_text(content, encoding="utf-8")
        print(f"Wrote {len(rows)} songs to {CSV_PATH}")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
