"""
Demonstrate natural language -> validated RecommendationPlan -> recommender.

Offline (mock LLM, no network):
    python examples/demo_planning_agent.py

Live (Groq; reads GROQ_API_KEY from the environment or the git-ignored .env):
    python examples/demo_planning_agent.py --live
    python examples/demo_planning_agent.py --live "Upbeat pop for a run, no repeated artists"

The LLM only produces the plan. Songs are ranked by the existing recommender.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import (  # noqa: E402
    MockLLMClient,
    PlanningError,
    create_recommendation_plan,
    load_env_file,
)
from src.recommender import load_songs, recommend_songs  # noqa: E402
from src.schemas import to_recommender_kwargs  # noqa: E402

EXAMPLE_REQUEST = (
    "Recommend five relaxing lofi songs for studying, with low energy and minimal vocals."
)
MOCK_RESPONSE = PROJECT_ROOT / "examples" / "llm_responses" / "lofi_study.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", nargs="?", default=EXAMPLE_REQUEST)
    parser.add_argument("--live", action="store_true", help="call the configured Groq model")
    parser.add_argument("--saved", help="path to a JSON file of saved preferences")
    args = parser.parse_args()

    if args.live:
        load_env_file(PROJECT_ROOT / ".env")
        client = None  # use LLMConfig.from_env()
    else:
        if args.request != EXAMPLE_REQUEST:
            parser.error("custom requests need --live (the mock only knows the example request)")
        client = MockLLMClient([MOCK_RESPONSE.read_text()])
    saved = json.loads(Path(args.saved).read_text()) if args.saved else None

    songs = load_songs("data/songs.csv")
    print(f"Request: {args.request}\n")
    try:
        result = create_recommendation_plan(args.request, saved, client=client, catalog=songs)
    except PlanningError as error:
        print(f"Planning failed ({type(error).__name__}):\n{error}")
        return 1

    print(f"Plan from model: {result.model}\n")
    print(f"{'field':26} {'value':30} {'source':9} evidence / note")
    for entry in result.provenance:
        detail = " | ".join(str(part) for part in (entry.evidence and repr(entry.evidence), entry.note) if part)
        print(f"{entry.field:26} {str(entry.value):30} {entry.source:9} {detail}")

    if result.unsupported_requests:
        print("\nUnsupported (NOT represented in the plan):")
        for item in result.unsupported_requests:
            print(f"  - {item.request}: {item.reason}")
    for warning in result.warnings:
        print(f"Warning: {warning}")

    kwargs = to_recommender_kwargs(result.plan)
    print(f"\nRecommender input: {kwargs}")
    if result.plan.hard_constraints or result.plan.playlist.unique_artists or result.plan.playlist.energy_progression:
        print("Note: hard constraints / playlist rules are not yet enforced on the ranked list (later phase).")
    print("\nTop songs (scored by the existing recommender):")
    for rank, (song, score, _) in enumerate(recommend_songs(songs=songs, **kwargs), start=1):
        print(f"  {rank}. {song['title']} - {song['artist']} ({song['genre']}/{song['mood']}) score={score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
