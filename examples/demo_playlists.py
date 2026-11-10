"""
Phase 4 examples: one successful and one unsuccessful playlist (offline, deterministic).

    python examples/demo_playlists.py

The planner is a MockLLMClient with fixed responses; everything after planning
(ranking, playlist building, validation) is the real TuneWise pipeline running
on data/songs.csv.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import MockLLMClient  # noqa: E402
from src.mock_responses import constraint, label, level, planner_response, requirement  # noqa: E402
from src.orchestrator import run_recommendation_workflow  # noqa: E402


def example_response(genre, mood, energy_level, playlist, constraints=()):
    """A planner response where genre and mood are quoted from the request."""
    return planner_response(
        genre=label(genre, evidence=genre),
        mood=label(mood, evidence=mood),
        energy=level(energy_level, reason="from mood"),
        playlist=playlist,
        constraints=constraints,
    )


EXAMPLES = [
    (
        "SUCCESS",
        "6 energetic pop songs, all different artists, building up in energy",
        example_response("pop", "energetic", "high", {
            "playlist_size": requirement(6, "6 energetic pop songs"),
            "unique_artists": requirement(True, "all different artists"),
            "energy_progression": requirement("ascending", "building up in energy"),
        }),
    ),
    (
        "UNSUCCESSFUL",
        "only lofi, 10 chill songs, every one by a different artist",
        example_response("lofi", "chill", "low", {
            "playlist_size": requirement(10, "10 chill songs"),
            "unique_artists": requirement(True, "every one by a different artist"),
        }, constraints=[constraint("genre", "only lofi", values=["lofi"])]),
    ),
]


def main() -> None:
    for title, request, response in EXAMPLES:
        result = run_recommendation_workflow(request, client=MockLLMClient([response]))
        print(f"=== {title}: {request!r}")
        print(f"outcome={result.status}  songs={len(result.songs)}/{result.requirements.size}  "
              f"candidates={result.candidate_count}/{result.catalog_size}")
        for song in result.songs:
            print(f"  {song.position}. [rank {song.rank}] {song.title} - {song.artist} "
                  f"({song.genre}/{song.mood}) energy={song.energy:.2f} score={song.score:.2f}")
        report = result.validation.to_dict()
        print(json.dumps({k: report[k] for k in ("outcome", "passed", "infeasible", "constraints_checked", "violated")}, indent=2))
        for check in result.validation.violations:
            print(f"  violated {check.name}: {check.message}; evidence={dict(check.evidence)}")
        print()


if __name__ == "__main__":
    main()
