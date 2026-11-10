"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if not __package__:  # run as "python src/main.py": make the "src" package importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import load_env_file
from src.orchestrator import WorkflowError, WorkflowResult, run_recommendation_workflow
from src.recommender import load_songs, recommend_songs


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


def _print_workflow_result(result: WorkflowResult) -> None:
    strategy = result.strategy
    print(f"Request: {result.request}")
    print(f"Planner model: {result.planning.model}")
    print(f"Scoring mode: {strategy.name} "
          f"(genre weight {strategy.genre_weight}, mood weight {strategy.mood_weight})")
    print(f"Candidates: {result.candidate_count} of {result.catalog_size} catalog songs "
          "satisfy the hard constraints\n")

    _print_plan(result)
    _print_validation(result)

    if not result.songs:
        print("\nNo songs could be selected.")
        return
    print("\nPlaylist (songs ranked by the original VibeMatch recommender; Pos = playlist order):\n")
    print(_format_song_details(result))
    print(_format_recommendation_table(result.as_recommendation_tuples()))


def _run_request(request: str) -> int:
    load_env_file()
    try:
        result = run_recommendation_workflow(request)
    except (WorkflowError, ValueError) as error:
        print(f"Could not create recommendations: {error}", file=sys.stderr)
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
    args = parser.parse_args(argv)
    if args.request is not None:
        if args.mode is not None:
            parser.error("--mode cannot be combined with --request; the plan selects the scoring mode")
        return _run_request(args.request)
    mode = args.mode or "genre-first"

    songs = load_songs("data/songs.csv") 

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