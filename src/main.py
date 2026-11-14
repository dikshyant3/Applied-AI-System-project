"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if not __package__:  # run as "python src/main.py": make the "src" package importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import load_env_file
from src.catalogs import CATALOGS
from src.feedback import submit_feedback
from src.orchestrator import WorkflowError, WorkflowResult, run_recommendation_workflow
from src.recommender import load_songs, recommend_songs
from src.user_profile import DEFAULT_PROFILE_PATH, UserProfileStore
from src.workflow_logging import configure_workflow_logging


def _render_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    """ASCII table. A cell is a string, or a list of strings for a multi-line cell."""
    rows_of_lines = [[cell if isinstance(cell, list) else [cell] for cell in row] for row in rows]

    widths = [len(header) for header in headers]
    for row in rows_of_lines:
        for index, lines in enumerate(row):
            widths[index] = max([widths[index], *(len(line) for line in lines)])

    def format_line(values: Sequence[str]) -> str:
        return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    table_lines = [separator, format_line(headers), separator]
    for row in rows_of_lines:
        height = max(len(lines) for lines in row)
        for line_index in range(height):
            table_lines.append(format_line([
                lines[line_index] if line_index < len(lines) else "" for lines in row
            ]))
    table_lines.append(separator)
    return "\n".join(table_lines)


def _wrap_reasons(explanation: str) -> List[str]:
    """One reason per line, each wrapped to 55 characters."""
    return [
        line
        for reason in explanation.split("; ")
        for line in textwrap.wrap(reason, width=55) or [""]
    ]


def _format_recommendation_table(
    recommendations: List[Tuple[Dict, float, str]]
) -> str:
    headers = ("Rank", "Song Title", "Artist", "Final Score", "Recommendation Reasons")
    rows = [
        (str(rank), song["title"], song["artist"], f"{score:.2f}", _wrap_reasons(explanation))
        for rank, (song, score, explanation) in enumerate(recommendations, start=1)
    ]
    return _render_table(headers, rows)


def _format_song_details(result: WorkflowResult) -> str:
    headers = ("Pos", "Rank", "ID", "Genre", "Mood", "Energy", "Base Score", "Final Score")
    rows = [
        (str(song.position), str(song.rank), str(song.id), song.genre, song.mood,
         f"{song.energy:.2f}", f"{song.base_score:.2f}", f"{song.score:.2f}")
        for song in result.songs
    ]
    return _render_table(headers, rows)


def _describe_value(value) -> str:
    """Readable text for a provenance value (hard constraints get a short form)."""
    if hasattr(value, "is_satisfied_by"):  # a HardConstraint
        if value.values:
            return f"{value.type} {list(value.values)}"
        return f"{value.type} [{value.minimum}, {value.maximum}]"
    return str(value)


def _provenance_detail(entry) -> str:
    parts = []
    if entry.evidence:
        parts.append(repr(entry.evidence))
    if entry.note:
        parts.append(entry.note)
    return " | ".join(parts)


def _print_plan(result: WorkflowResult) -> None:
    planning = result.planning
    print("Plan (non-default fields):")
    for entry in planning.provenance:
        if entry.source == "default":
            continue
        print(f"  {entry.field:26} {_describe_value(entry.value):24} [{entry.source}] {_provenance_detail(entry)}")
    for warning in planning.warnings:
        print(f"  WARNING: {warning}")


def _print_validation(result: WorkflowResult) -> None:
    report = result.validation
    if result.playlist.selection_mode == "duration":
        size_note = f"{len(result.songs)} songs selected by duration"
    else:
        size_note = f"{len(result.songs)} of {result.requirements.size} songs"
    print(f"\nPlaylist validation: {report.outcome.upper()} ({size_note})")
    for check in report.checks:
        flag = " [infeasible with this catalog]" if check.infeasible else ""
        print(f"  [{check.status:11}] {check.name}: {check.message}{flag}")
    for skipped in result.playlist.skipped:
        print(f"  skipped rank {skipped.rank} {skipped.title!r}: {skipped.rule} ({skipped.detail})")


def _print_agent_trace(result: WorkflowResult) -> None:
    trace = result.trace
    if trace is None:
        return
    print(f"\nAgent loop: {trace.final_status.upper()} after {len(trace.attempts)} of "
          f"{trace.max_attempts} attempt(s) - {trace.stop_reason}")
    for attempt in trace.attempts:
        line = f"  attempt {attempt.number} ({attempt.kind}): {attempt.outcome}"
        if attempt.changed_fields:
            line += f"; revised {', '.join(attempt.changed_fields)}"
        if attempt.observation is not None:
            line += f"; reasons: {', '.join(attempt.observation.codes)}"
        print(line)
        if attempt.rationale:
            print(f"    rationale: {attempt.rationale}")
    if trace.result_attempt is not None and trace.result_attempt != len(trace.attempts):
        print(f"  (showing the playlist from attempt {trace.result_attempt}, the last one built)")


def _print_workflow_result(result: WorkflowResult) -> None:
    strategy = result.strategy
    print(f"Request: {result.request}")
    print(f"Planner model: {result.planning.model}")
    print(f"Scoring mode: {strategy.name} "
          f"(genre weight {strategy.genre_weight}, mood weight {strategy.mood_weight})")
    print(f"Candidates: {result.candidate_count} of {result.catalog_size} catalog songs "
          "satisfy the hard constraints\n")

    _print_plan(result)
    _print_agent_trace(result)
    _print_validation(result)

    if not result.songs:
        print("\nNo songs could be selected.")
        return
    print("\nPlaylist (songs ranked by the original VibeMatch recommender; Pos = playlist order):\n")
    print(_format_song_details(result))
    print(_format_recommendation_table(result.as_recommendation_tuples()))


MOCK_SCENARIOS_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "scenarios.json"


def _mock_scenarios() -> Dict[str, Dict]:
    data = json.loads(MOCK_SCENARIOS_PATH.read_text(encoding="utf-8"))
    return {scenario["id"]: scenario for scenario in data["scenarios"]}


def _run_request(request: Optional[str], profile_path: Optional[str] = None,
                 mock_scenario: Optional[str] = None, catalog: str = "demo",
                 use_rag: bool = False) -> int:
    configure_workflow_logging()
    try:
        songs = load_songs(CATALOGS[catalog].csv_path)
        if mock_scenario is None:
            load_env_file()
            profile = UserProfileStore(profile_path or CATALOGS[catalog].profile_path, songs)
            retriever = None
            if use_rag:
                from src.retrieval import build_retriever

                retriever = build_retriever(songs)
            result = run_recommendation_workflow(
                request, saved_preferences=profile.preferences, songs=songs,
                retriever=retriever,
            )
        else:
            # Offline replay: a scripted planner stands in for the LLM; everything else
            # (ranking, playlist building, validation, revision loop) is the real code.
            # The saved profile is neither read nor changed, so output is reproducible.
            from evaluation.evaluate import _mock_responses
            from src.agent import MockLLMClient

            scenario = _mock_scenarios()[mock_scenario]
            print(f"[mock planner: replaying scenario {mock_scenario!r}; no API call is made]\n")
            result = run_recommendation_workflow(
                scenario["request"], saved_preferences=scenario.get("saved_preferences"),
                client=MockLLMClient(_mock_responses(scenario)), songs=songs,
                max_attempts=scenario.get("max_attempts", 3),
                request_id=f"mock-{mock_scenario}",
            )
    except (WorkflowError, ValueError) as error:
        print(f"Could not create recommendations: {error}", file=sys.stderr)
        trace = getattr(error, "trace", None)
        if trace is not None:
            print(f"Agent loop: {trace.final_status} after {len(trace.attempts)} attempt(s) - "
                  f"{trace.stop_reason}", file=sys.stderr)
        return 1
    _print_workflow_result(result)
    return 0 if result.validation.passed else 3


def main(argv: Optional[Sequence[str]] = None) -> Optional[int]:
    parser = argparse.ArgumentParser(description="Recommend songs by scoring mode.")
    parser.add_argument(
        "--mode",
        choices=("genre-first", "mood-first"),
        default=None,
        help="Ranking strategy to use (default: genre-first).",
    )
    parser.add_argument(
        "--request",
        help="Natural-language request, e.g. \"calm lofi music for studying\". "
        "The LLM planner chooses the scoring mode (requires GROQ_API_KEY).",
    )
    parser.add_argument(
        "--mock-scenario",
        metavar="SCENARIO_ID",
        help="Run the full TuneWise workflow offline with a scripted planner from "
        "evaluation/scenarios.json (no API key). Use --list-mock-scenarios to see IDs.",
    )
    parser.add_argument(
        "--list-mock-scenarios", action="store_true",
        help="List the offline scenario IDs and their requests, then exit.",
    )
    parser.add_argument(
        "--catalog", choices=tuple(CATALOGS), default="demo",
        help="Song catalog: 'demo' (synthetic, data/songs.csv) or 'real' (Deezer tracks, "
        "data/real_songs.csv, built by scripts/build_real_catalog.py).",
    )
    parser.add_argument(
        "--rag", action="store_true",
        help="With --request: retrieval-augmented planning and candidate selection.",
    )
    parser.add_argument(
        "--profile-path",
        help=f"JSON profile path (default: {DEFAULT_PROFILE_PATH}).",
    )
    profile_actions = parser.add_mutually_exclusive_group()
    profile_actions.add_argument(
        "--show-profile", action="store_true", help="show saved preferences and feedback count",
    )
    profile_actions.add_argument(
        "--reset-profile", action="store_true", help="clear saved preferences and feedback",
    )
    profile_actions.add_argument(
        "--feedback", nargs=2, metavar=("SONG_ID", "ACTION"),
        help="record catalog-song feedback: like, dislike, or skip",
    )
    args = parser.parse_args(argv)
    selected = CATALOGS[args.catalog]
    if not selected.exists:
        parser.error(f"catalog {args.catalog!r} not found at {selected.csv_path}; "
                     "run: python scripts/build_real_catalog.py")
    profile_path = args.profile_path or selected.profile_path
    if args.rag and args.request is None:
        parser.error("--rag requires --request")
    if args.mock_scenario is not None and args.catalog != "demo":
        parser.error("mock scenarios are scripted for the demo catalog")
    if args.list_mock_scenarios:
        for scenario_id, scenario in _mock_scenarios().items():
            print(f"{scenario_id:38} {scenario['expected_status']:24} {scenario['request']}")
        return 0
    if args.mock_scenario is not None:
        if args.request is not None or args.mode is not None:
            parser.error("--mock-scenario replays its own request; do not combine it with "
                         "--request or --mode")
        if args.mock_scenario not in _mock_scenarios():
            parser.error(f"unknown mock scenario {args.mock_scenario!r}; "
                         "see --list-mock-scenarios")
    if args.show_profile or args.reset_profile or args.feedback:
        if args.request is not None or args.mode is not None or args.mock_scenario is not None:
            parser.error("profile actions cannot be combined with --request, --mode, "
                         "or --mock-scenario")
        try:
            catalog = load_songs(selected.csv_path)
            profile = UserProfileStore(profile_path, catalog)
            if args.show_profile:
                print(json.dumps({
                    "preferences": profile.preferences,
                    "feedback_count": len(profile.feedback),
                }, indent=2, sort_keys=True))
            elif args.reset_profile:
                profile.reset()
                print("Saved profile reset.")
            else:
                song_id_text, action = args.feedback
                try:
                    song_id = int(song_id_text)
                except ValueError:
                    parser.error("SONG_ID must be an integer catalog song id")
                if action not in ("like", "dislike", "skip"):
                    parser.error("ACTION must be like, dislike, or skip")
                updated = submit_feedback(profile, song_id, action)
                print(json.dumps({"preferences": updated, "feedback_count": len(profile.feedback)},
                                 indent=2, sort_keys=True))
        except (WorkflowError, ValueError) as error:
            print(f"Could not update user profile: {error}", file=sys.stderr)
            return 1
        return 0
    if args.request is not None:
        if args.mode is not None:
            parser.error("--mode cannot be combined with --request; the plan selects the scoring mode")
        return _run_request(args.request, str(profile_path), catalog=args.catalog, use_rag=args.rag)
    if args.mock_scenario is not None:
        return _run_request(None, mock_scenario=args.mock_scenario)
    mode = args.mode or "genre-first"

    songs = load_songs(selected.csv_path)

    profiles = [
        (
            "High-Energy Pop",
            {
                "favorite_genre": "pop",
                "favorite_mood": "happy",
                "target_energy": 0.9,
                "target_popularity": 80,
                "preferred_release_decade": 2020,
                "preferred_detailed_mood": "euphoric",
                "target_instrumentalness": 0.05,
                "target_speechiness": 0.05,
            },
        ),
        (
            "Chill Lofi",
            {
                "favorite_genre": "lofi",
                "favorite_mood": "chill",
                "target_energy": 0.4,
                "target_popularity": 55,
                "preferred_release_decade": 2020,
                "preferred_detailed_mood": "cozy",
                "target_instrumentalness": 0.2,
                "target_speechiness": 0.04,
            },
        ),
        (
            "Deep Intense Rock",
            {
                "favorite_genre": "rock",
                "favorite_mood": "intense",
                "target_energy": 0.9,
                "target_popularity": 65,
                "preferred_release_decade": 2020,
                "preferred_detailed_mood": "defiant",
                "target_instrumentalness": 0.05,
                "target_speechiness": 0.06,
            },
        ),
    ]

    print(f"Scoring mode: {mode.title()}")
    for profile_name, user_prefs in profiles:
        recommendations = recommend_songs(user_prefs, songs, k=5, mode=mode)

        print(
            f"\nTop recommendations for {profile_name} "
            f"({user_prefs['favorite_genre']} / {user_prefs['favorite_mood']}, "
            f"target energy: {user_prefs['target_energy']:.2f}):\n"
        )
        print(_format_recommendation_table(recommendations))
        print()


if __name__ == "__main__":
    raise SystemExit(main())