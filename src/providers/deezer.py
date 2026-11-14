"""Minimal client for the public Deezer API (https://developers.deezer.com/api).

No API key is needed. Used for two things:

* building the real catalog offline (``scripts/build_real_catalog.py``):
  playlist search, playlist tracks and per-track details;
* the web page: a *fresh* 30-second preview URL for a track. Deezer preview
  URLs carry a signed token that expires after roughly 15 minutes, so they are
  never stored in the catalog; ``preview_url`` fetches one on demand and caches
  it for a few minutes.

Only previews and cover art are used. Audio is streamed by the listener's
browser from Deezer; TuneWise never redistributes or stores audio files.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

API_ROOT = "https://api.deezer.com"
USER_AGENT = "TuneWise/1.0 (educational music recommender)"
PREVIEW_CACHE_SECONDS = 8 * 60  # tokens last ~15 min; refresh well before that
QUOTA_ERROR_CODE = 4  # Deezer: "Quota limit exceeded" (50 requests / 5 s)


class DeezerError(Exception):
    """The Deezer API could not be reached or returned an error."""


Fetcher = Callable[[str], bytes]


def _default_fetch(url: str, timeout: float = 15.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class DeezerClient:
    """Rate-limited JSON client. ``fetch`` is injectable so tests never hit the network."""

    def __init__(self, fetch: Optional[Fetcher] = None, min_interval: float = 0.12,
                 max_retries: int = 4, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self._fetch = fetch or _default_fetch
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_call = float("-inf")
        self._previews: Dict[int, Tuple[float, Optional[str]]] = {}

    # ---- low level -------------------------------------------------------

    def get(self, path: str, **params: Any) -> Dict[str, Any]:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{API_ROOT}/{path.lstrip('/')}" + (f"?{query}" if query else "")
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                payload = json.loads(self._fetch(url))
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempt == self._max_retries:
                    raise DeezerError(f"Deezer request failed: {error}") from error
                self._sleep(1.0 + attempt)
                continue
            except ValueError as error:
                raise DeezerError("Deezer returned invalid JSON") from error
            error = payload.get("error") if isinstance(payload, dict) else None
            if not error:
                return payload
            if error.get("code") == QUOTA_ERROR_CODE and attempt < self._max_retries:
                self._sleep(5.0)  # quota window is 5 seconds
                continue
            raise DeezerError(f"Deezer error {error.get('code')}: {error.get('message')}")
        raise DeezerError("Deezer request failed after retries")  # pragma: no cover

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_call + self._min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._last_call = self._clock()

    # ---- catalog building --------------------------------------------------

    def search_playlists(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        return self.get("search/playlist", q=query, limit=limit).get("data", [])

    def playlist_tracks(self, playlist_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        return self.get(f"playlist/{int(playlist_id)}/tracks", limit=limit).get("data", [])

    def track(self, track_id: int) -> Dict[str, Any]:
        return self.get(f"track/{int(track_id)}")

    # ---- web page ---------------------------------------------------------

    def preview_url(self, track_id: int) -> Optional[str]:
        """A currently valid 30-second preview URL, or None if Deezer has none."""
        track_id = int(track_id)
        cached = self._previews.get(track_id)
        now = self._clock()
        if cached is not None and now - cached[0] < PREVIEW_CACHE_SECONDS:
            return cached[1]
        url = self.track(track_id).get("preview") or None
        self._previews[track_id] = (now, url)
        return url

    def download(self, url: str) -> bytes:
        """Raw bytes of a preview (used only to measure audio features)."""
        try:
            return self._fetch(url)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise DeezerError(f"preview download failed: {error}") from error
