"""
Phase 5 examples: the Plan -> Act -> Observe -> Revise loop (offline, deterministic).

    python examples/demo_agent_loop.py

The LLM is a MockLLMClient with scripted plan and revision responses; ranking,
playlist building, validation and the loop itself are the real TuneWise code
running on data/songs.csv. The same scenarios are asserted in
tests/test_agent_loop.py.

1. Success after revision: the first playlist is 12.6 min (needs 18-22 min);
   the LLM revises the inferred detailed mood and the second playlist passes.
2. Failure, retry budget exhausted: three distinct plans all miss the window.
3. Failure, impossible request: with "only lofi" no 5 songs can reach 18 min,
   so the loop stops after one attempt without asking for a revision.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import MockLLMClient  # noqa: E402
from src.mock_responses import constraint, label, level, planner_response, requirement  # noqa: E402
from src.orchestrator import run_recommendation_workflow  # noqa: E402

REQUEST = "Give me 5 calm lofi songs for studying, about 20 minutes long, nothing by Neon Echo"
ONLY_LOFI = "Give me 5 calm lofi songs for studying, about 20 minutes long, only lofi"


def initial_plan(hard_constraint):
    return planner_response(
        genre=label("lofi", evidence="lofi"),
        mood=label("chill", evidence="calm"),
        energy=level("low", evidence="calm", reason="calm music is low energy"),
        detailed_mood=label("focused", source="inferred", evidence="for studying", reason="studying needs focus"),
        playlist={"playlist_size": requirement(5, "5 calm lofi songs")},
        duration=requirement(20, "about 20 minutes"),
        constraints=[hard_constraint],
    )


NO_NEON_ECHO = constraint("exclude_artist", "nothing by Neon Echo", values=["Neon Echo"])
LOFI_ONLY = constraint("genre", "only lofi", values=["lofi"])

COZY = {
    "preferences": {"preferred_detailed_mood": label("cozy", source="inferred", reason="longer, cozier tracks")},
    "rationale": "Total length was 12.6 min, below 18 min; cozy lofi tracks tend to be longer.",
}
DROP_MOOD = {"preferences": {"preferred_detailed_mood": None}, "rationale": "Remove the detailed-mood bias."}
MEDIUM_ENERGY = {
    "preferences": {"target_energy": level("medium", reason="widen the energy target")},
    "rationale": "Try a medium energy target to reach different songs.",
}


def show(title, request, responses):
    print("=" * 78)
    print(title)
    print(f"Request: {request}")
    client = MockLLMClient(responses)
    result = run_recommendation_workflow(request, client=client)
    trace = result.trace
    print(f"Final status: {trace.final_status}  ({len(trace.attempts)}/{trace.max_attempts} attempts, "
          f"{len(client.calls)} LLM call(s))")
    print(f"Stop reason: {trace.stop_reason}")
    for attempt in trace.attempts:
        print(f"\n  Attempt {attempt.number} [{attempt.kind}] -> {attempt.outcome}")
        if attempt.changed_fields:
            print(f"    revised fields: {', '.join(attempt.changed_fields)}")
            print(f"    rationale: {attempt.rationale}")
        if attempt.plan is not None:
            prefs = {k: v["value"] for k, v in attempt.plan["preferences"].items()}
            print(f"    soft preferences: {prefs}")
            print(f"    hard constraints: {attempt.plan['hard_constraints']} (locked)")
        if attempt.observation is not None:
            print("    observation: " + json.dumps(
                {"category": attempt.observation.category, "retryable": attempt.observation.retryable,
                 "codes": list(attempt.observation.codes)}))
            for reason in attempt.observation.reasons:
                print(f"      - {reason.code}: {reason.message}")
    print(f"\n  Returned playlist (attempt {trace.result_attempt}, validation {result.validation.outcome}):")
    for song in result.songs:
        print(f"    {song.position}. {song.title} - {song.artist} ({song.genre}, "
              f"{song.record['duration_seconds']}s)")
    print()


if __name__ == "__main__":
    show("1. Success after one revision", REQUEST, [initial_plan(NO_NEON_ECHO), COZY])
    show("2. Failure: retry budget exhausted", REQUEST,
         [initial_plan(NO_NEON_ECHO), DROP_MOOD, MEDIUM_ENERGY, COZY])
    show("3. Failure: provably impossible, not retried", ONLY_LOFI, [initial_plan(LOFI_ONLY), COZY])
