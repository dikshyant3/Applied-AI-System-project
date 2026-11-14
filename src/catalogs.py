"""The two song catalogs TuneWise can run on.

* ``demo`` - data/songs.csv: 250 synthetic songs. Deterministic; used by the
  tests, the mock scenarios and the reproducible evaluation.
* ``real`` - data/real_songs.csv: real tracks from Deezer with album covers,
  30-second previews and audio features measured from those previews
  (built by scripts/build_real_catalog.py).

Saved feedback refers to catalog song IDs, so each catalog has its own profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .user_profile import DEFAULT_PROFILE_PATH

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Catalog:
    name: str
    csv_path: str
    profile_path: Path

    @property
    def exists(self) -> bool:
        return (ROOT / self.csv_path).is_file()


CATALOGS = {
    "demo": Catalog("demo", "data/songs.csv", DEFAULT_PROFILE_PATH),
    "real": Catalog("real", "data/real_songs.csv", DEFAULT_PROFILE_PATH.with_name("profile-real.json")),
}
