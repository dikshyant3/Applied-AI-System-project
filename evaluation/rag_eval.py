"""Retrieval evaluation for the RAG layer on the real catalog.

    python -m evaluation.rag_eval                       # retrieval only, offline
    python -m evaluation.rag_eval --output evaluation/rag_report.json
    python -m evaluation.rag_eval --live                # also full workflow, with vs without RAG
                                                        # (calls the LLM: needs GROQ_API_KEY)

Metric: label precision@K - the share of the top-K songs whose genre or mood
is in the query's accepted set (evaluation/rag_queries.json). Queries are
descriptive and avoid the label words, so they test meaning, not keywords.

Compared retrievers on the same catalog:
* semantic - FastEmbed sentence-transformers/all-MiniLM-L6-v2, the retriever TuneWise uses
* --compare MODEL ... - other FastEmbed models on the same queries
* lexical  - hashed word overlap (HashingEmbedder)
* random   - expected precision of K songs drawn at random (the label base rate)

Labels are inherited from Deezer playlists, so they are weak and noisy; treat
the numbers as a relative comparison, not absolute relevance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommender import load_songs  # noqa: E402
from src.retrieval import FastEmbedEmbedder, HashingEmbedder, SongRetriever  # noqa: E402

QUERIES = ROOT / "evaluation" / "rag_queries.json"
CATALOG = ROOT / "data" / "real_songs.csv"


def relevant(song: Mapping[str, Any], accept: Mapping[str, Sequence[str]]) -> bool:
    return song["genre"] in accept.get("genres", ()) or song["mood"] in accept.get("moods", ())


def precision(songs: Sequence[Mapping[str, Any]], accept) -> float:
    return sum(relevant(s, accept) for s in songs) / len(songs) if songs else 0.0


def evaluate_retrieval(catalog, queries, retrievers: Mapping[str, SongRetriever], k: int) -> Dict[str, Any]:
    per_query = []
    for q in queries:
        row = {"id": q["id"], "random": round(precision(catalog, q["accept"]), 4)}
        for name, retriever in retrievers.items():
            row[name] = round(precision([h.song for h in retriever.search(q["query"], k=k)], q["accept"]), 4)
        per_query.append(row)
    names = ["random", *retrievers]
    summary = {name: round(mean(r[name] for r in per_query), 4) for name in names}
    wins = {name: sum(r[name] > r["random"] for r in per_query) for name in retrievers}
    return {"k": k, "mean_precision": summary, "beats_random": wins, "queries": per_query}


def _run_with_backoff(run, retries: int = 4, wait: float = 30.0):
    """Retry a workflow run that failed only because the provider rate-limited it."""
    from src.agent import LLMRateLimitError
    from src.orchestrator import WorkflowError

    for attempt in range(retries + 1):
        try:
            return run()
        except WorkflowError as error:
            if not isinstance(error.cause, LLMRateLimitError) or attempt == retries:
                raise
            print(f"    rate limited; waiting {wait * (attempt + 1):.0f}s", file=sys.stderr)
            time.sleep(wait * (attempt + 1))


def evaluate_live(catalog, queries, retriever: SongRetriever, pause: float = 8.0) -> Dict[str, Any]:
    """Full workflow with and without RAG; precision of the final playlist.

    Runs that still fail (after rate-limit retries) are reported but excluded
    from the means, and queries are only compared when both runs succeeded.
    """
    from src.agent import load_env_file
    from src.orchestrator import WorkflowError, run_recommendation_workflow

    load_env_file()
    names = ("tunewise", "tunewise_rag")
    rows = []
    for q in queries:
        row: Dict[str, Any] = {"id": q["id"]}
        for name, rag in zip(names, (None, retriever)):
            started = time.perf_counter()
            try:
                result = _run_with_backoff(lambda: run_recommendation_workflow(
                    q["query"] + " (5 songs)", songs=catalog, retriever=rag,
                    request_id=f"rag-eval-{q['id']}"))
                row[name] = round(precision([dict(s.record) for s in result.songs], q["accept"]), 4)
                row[f"{name}_status"] = result.final_status
            except (WorkflowError, ValueError) as error:
                row[name], row[f"{name}_status"] = None, f"error: {error}"[:160]
            row[f"{name}_seconds"] = round(time.perf_counter() - started, 2)
            time.sleep(pause)
        rows.append(row)
        print(f"  {q['id']:18} plain={row['tunewise']} rag={row['tunewise_rag']}", file=sys.stderr)
    paired = [r for r in rows if all(r[n] is not None for n in names)]
    return {
        "completed": {n: sum(r[n] is not None for r in rows) for n in names},
        "paired_queries": len(paired),
        "mean_precision_paired": {n: round(mean(r[n] for r in paired), 4) if paired else None for n in names},
        "rag_better": sum(r["tunewise_rag"] > r["tunewise"] for r in paired),
        "tie": sum(r["tunewise_rag"] == r["tunewise"] for r in paired),
        "rag_worse": sum(r["tunewise_rag"] < r["tunewise"] for r in paired),
        "queries": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate TuneWise retrieval on the real catalog.")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--model", default=None, help="embedding model (default: TuneWise's)")
    parser.add_argument("--compare", nargs="*", default=[], help="extra embedding models to compare")
    parser.add_argument("--live", action="store_true", help="also run the full workflow with the LLM")
    parser.add_argument("--pause", type=float, default=8.0, help="seconds between live LLM runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    catalog = load_songs(str(args.catalog))
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    semantic = SongRetriever(catalog, FastEmbedEmbedder(args.model), ROOT / "data" / "cache")
    retrievers = {"semantic": semantic}
    for extra in args.compare:
        retrievers[extra] = SongRetriever(catalog, FastEmbedEmbedder(extra), ROOT / "data" / "cache")
    retrievers["lexical"] = SongRetriever(catalog, HashingEmbedder())
    report: Dict[str, Any] = {
        "catalog": str(args.catalog.relative_to(ROOT)) if args.catalog.is_relative_to(ROOT) else str(args.catalog),
        "catalog_size": len(catalog),
        "embedding_model": semantic.embedder.name,
        "retrieval": evaluate_retrieval(catalog, queries, retrievers, args.k),
    }
    if args.live:
        report["workflow"] = evaluate_live(catalog, queries, semantic, args.pause)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
