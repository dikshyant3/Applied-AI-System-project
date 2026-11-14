# TuneWise AI

TuneWise extends the original VibeMatch music recommender with a structured
LLM planning layer, deterministic playlist construction and validation,
bounded plan revision, local user feedback, structured workflow events, and an
offline evaluation harness. It runs on a **real song catalog built from the
public Deezer API** (album covers, playable 30-second previews, audio features
measured from those previews) and can use **retrieval-augmented generation
(RAG)** to understand descriptive requests such as "a rainy Sunday with a book".

## What is implemented

- **VibeMatch foundation:** the original content-based `recommend_songs` and
  `score_song` implementation remains the ranking baseline. Phase 6 does not
  change its scoring algorithm.
- **TuneWise planning and execution:** an LLM proposes a constrained plan;
  Python validates it, filters against hard constraints, calls the existing
  recommender, builds and validates a playlist, and can retry a correctable
  failure up to three total attempts.
- **Personalization:** like, dislike, and skip events for catalog song IDs are
  stored in a local JSON profile. Feedback uses deterministic catalog-derived
  updates. A preference explicitly stated in the current request takes
  precedence over saved preferences.
- **Observability:** the workflow emits JSON events with an opaque request ID,
  plan summary, strategy, tool/candidate counts, validation, retries, status,
  and elapsed time. It does not log the request text, evidence quotes, song
  titles, or API key. Plan preference values and hard constraints are included;
  handle logs accordingly.
- **Real catalog (Deezer):** `scripts/build_real_catalog.py` builds
  `data/real_songs.csv` from public Deezer playlists (no API key). Each track
  keeps its album cover, Deezer link and the titles of the playlists it was
  found in (free-text tags). Tempo and energy are **measured** from the
  30-second preview with librosa; danceability, acousticness, valence,
  instrumentalness and speechiness are **estimates** (audio proxies and
  label priors); genre and mood are inherited from the seed playlist search.
  The synthetic `data/songs.csv` is kept for tests and reproducible evaluation.
- **RAG:** each song becomes a text document (labels, album, playlist tags)
  embedded locally with the Hugging Face model `sentence-transformers/all-MiniLM-L6-v2`
  (ONNX, run by FastEmbed; override with `TUNEWISE_EMBEDDING_MODEL`). The closest songs'
  labels are given to the planner as `catalog_matches` hints, and only the
  top-k eligible songs by similarity are ranked by VibeMatch (falling back to
  all eligible songs if the pool is too small). Explanations show each song's
  semantic match score.
- **Web page:** album-cover cards; **▶ Play preview** fetches a fresh Deezer
  preview URL (they expire after ~15 minutes) and plays the 30-second clip. A
  "Vibe search" mode runs retrieval alone, without the LLM or an API key.
- **For You (no prompt) and Liked songs:** every song card has 👍 / 👎. A like
  saves the song to **Liked songs** (click again to remove it) and, like a
  dislike, updates the saved profile. The **For You** page then recommends
  unrated songs with no prompt: VibeMatch scores the saved preferences, and the
  candidate pool is the songs whose embeddings are closest to liked songs and
  furthest from disliked ones, each explained as "Because you liked …". New
  users get popular starter picks from different genres. Logic lives in
  `src/personalized.py`; no LLM call is made.
- **Evaluation:** ten scripted mock-LLM scenarios cover normal requests,
  saved/request preference conflicts, unsupported metadata, playlist rules,
  impossible requests, malformed responses, revision success, retry
  exhaustion, and a simulated API failure.

## Not implemented or unsupported

- Neither catalog provides language or other arbitrary attributes;
  unsupported requests are reported, not guessed.
- The real catalog is built offline (~1,200 tracks); TuneWise does not search
  Deezer live per request. Its genre/mood labels come from the playlist search a
  track was found in, and only tempo and energy are measured from audio.
- Feedback is local to one JSON profile. There is no account system, cloud
  synchronization, deletion service, or automated profile migration.
- The evaluation harness supports live calls only when explicitly run with
  `--mode live`; no live-model benchmark results are claimed here.
- The human-relevance export and scoring tool exists, but no human judgments
  or measured recommendation-relevance results are included yet.
- There is no arbitrary code execution, autonomous song lookup, or use of
  metadata outside the catalog.

## Install

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Build the real catalog once (about 20-40 minutes for ~1,000 tracks; every step
is cached in the git-ignored `data/cache/`, so an interrupted run resumes):

```bash
python scripts/build_real_catalog.py                # ~1,000 tracks
python scripts/build_real_catalog.py --per-seed 10  # quick, ~300 tracks
```

The first RAG use downloads the ~90 MB MiniLM model from Hugging Face once. If FastEmbed is
unavailable, TuneWise logs a warning and falls back to keyword retrieval.

The deterministic tests and mock evaluation do not need an LLM API key. Live
planning uses the configured Groq-compatible client. Set `GROQ_API_KEY` in the
environment or in the git-ignored `.env` file. Optional settings are
`TUNEWISE_LLM_MODEL` and `TUNEWISE_LLM_TIMEOUT`.

## Run

Run the original VibeMatch examples without an LLM:

```bash
python src/main.py --mode genre-first
python src/main.py --mode mood-first
```

Run the full TuneWise agent workflow **offline, with no API key**. A scripted
planner reply stands in for the LLM; filtering, ranking, playlist building,
validation and the revise-and-retry loop are the real code:

```bash
python src/main.py --list-mock-scenarios
python src/main.py --mock-scenario successful-plan-revision
```

### Web interface

```bash
streamlit run app.py
```

This opens the app in your browser (usually http://localhost:8501). In the
sidebar, choose:

- **Live LLM request** to type your own request (requires `GROQ_API_KEY`).
  Pick the **Real songs (Deezer)** or **Synthetic demo** catalog and turn
  **Retrieval (RAG)** on or off.
- **Vibe search** to find real songs by description using retrieval only (no
  key, no planning or validation).
- **For You** for prompt-free recommendations from your likes, dislikes and
  saved preferences, plus your **Liked songs** list.
- **Offline demo** to replay any evaluation scenario on the synthetic catalog
  without an API key.

Real songs appear as album-cover cards; press **▶ Play preview** to hear the
30-second Deezer clip. The result shows the playlist with per-song reasons, each
agent attempt and revision, the plan with its evidence, and every validation
check. It runs the same workflow as the CLI, and workflow JSON events still go
to the terminal.

### Live CLI request

Request a TuneWise playlist from your own text (requires the live model credentials):

```bash
python src/main.py --request "calm lofi music for studying"
python src/main.py --catalog real --rag --request "songs for a rainy afternoon with a book"
```

The default profile is `~/.tunewise/profile.json` (`~/.tunewise/profile-real.json`
for `--catalog real`, because feedback refers to catalog song IDs). Use
`--profile-path PATH` to select another local file. Feedback actions only
accept IDs in the selected catalog:

```bash
python src/main.py --feedback 2 like
python src/main.py --feedback 2 dislike
python src/main.py --feedback 2 skip
python src/main.py --show-profile
python src/main.py --reset-profile
```

Workflow JSON events are written to standard error for request-mode runs.
Redirect or retain that stream according to your local privacy and retention
requirements.

## Testing summary

376 automated tests passed (1 skipped). The offline evaluation matched
the expected outcome in 14 of 14 scenarios, including invalid model output, an
API rate limit, impossible requests and running out of retries, all handled
without crashing. In the 8 scenarios with a baseline, TuneWise satisfied every
checked playlist constraint (1.00) vs 0.75 for the original VibeMatch
recommender, which cannot enforce energy ordering. TuneWise also recovered in 1
of 3 multi-attempt scenarios by revising its own plan. Its weak spot: it can
drift outside the requested genre when a request has no hard genre constraint.
In the rock workout scenario it added an electronic track. Human relevance
ratings are not collected yet (see below).

## RAG retrieval evaluation (real catalog)

`python -m evaluation.rag_eval` embeds 15 descriptive requests that avoid the
catalog's label words ("songs for crying after a breakup", "driving through neon
city lights at 2am") and checks how many of the top 10 songs have an accepted
genre or mood:

| Retriever | Precision@10 | Beats random |
|---|---:|---:|
| Semantic: all-MiniLM-L6-v2 (TuneWise default) | **0.71** | 15/15 |
| Semantic: bge-small-en-v1.5 (`--compare`) | 0.74 | 14/15 |
| Lexical (word overlap) | 0.37 | 7/15 |
| Random (label base rate) | 0.20 | – |

MiniLM's weakest queries were "groovy bass lines and disco vibes from the
seventies" (0.1) and "angry fast songs about not fitting in" (0.2). Labels are inherited from Deezer playlists, so this is a relative
comparison, not human-judged relevance. `--live` additionally runs the full
workflow with and without RAG (LLM calls; needs `GROQ_API_KEY`).

**Full workflow, live LLM (`--live`, one run, 15 queries, 5 songs each; measured
with the earlier bge-small embedder):**
TuneWise without RAG reached label precision 1.00 and TuneWise + RAG 0.99
(14 ties, 1 worse: the "seventies disco" request, where plain TuneWise reported the
decade as unsupported metadata and RAG-narrowed ranking let one non-funk song in).
So on this metric RAG gives **no measurable gain** once the LLM planner is involved:
the planner already maps these descriptive requests to the right genre/mood labels,
and VibeMatch then ranks by exactly those labels, so the metric saturates. RAG's
demonstrated value here is retrieval without an LLM (Vibe search, 0.71 vs 0.37
lexical) and tag-grounded explanations; showing a playlist-quality gain would need
human ratings rather than these label checks.


## Test and evaluate

```bash
pytest -q
python -m evaluation.evaluate --mode mock
python -m evaluation.evaluate --mode mock --output evaluation/report.json
```

The mock mode replays `evaluation/scenarios.json` and is deterministic apart
from measured wall-clock latency and report timestamp. Live mode makes real
provider calls, requires credentials, can incur provider costs, and is
non-deterministic:

```bash
python -m evaluation.evaluate --mode live
```

The evaluator compares the original recommender and TuneWise on the same eight
successful, supported scenarios with baseline inputs. Baseline latency is the
ranking call only; TuneWise latency includes orchestration and a scripted mock
planner. These are component timings, not a live-service performance
comparison.

## Human relevance review (pending)

`evaluation/quality.py` turns six supported scenarios (listed in
`evaluation/quality_cases.json`) into a blinded review for people to rate.
Each scenario was picked because VibeMatch and TuneWise return different
songs or a different order. Scenarios where both return identical output are
left out because they can only tie.

- **Song ratings:** every song either system picked is listed **once** per
  scenario, shuffled, with no system name or rank. Raters set `relevance`
  from 0 to 4 using title, artist, genre, mood, detailed mood and energy.
- **Playlist ratings:** three requests are about the playlist as a whole
  ("build in energy", "get calmer", "different artists"). For these, raters
  also see both playlists in order as **A** and **B** and set `fit` from 0 to 4.
- **Answer key:** maps candidates and A/B labels back to systems and ranks,
  plus a fingerprint of the blank review. Scoring refuses any review whose
  songs or structure were edited, or that came from a different export.

Files go to `evaluation/human_review/`, which is gitignored together with any
`*answer_key*.json` or `*answer-key*.json` file, so the key cannot be committed
by accident:

```bash
python -m evaluation.quality export --seed 0
# -> evaluation/human_review/quality_review.json      (give a copy to each rater)
# -> evaluation/human_review/quality_answer_key.json  (keep private)

# After raters fill every null relevance/fit value, score one or more copies:
python -m evaluation.quality score \
  --review evaluation/human_review/rater1.json \
  --review evaluation/human_review/rater2.json \
  --answer-key evaluation/human_review/quality_answer_key.json \
  --output evaluation/quality_report.json
```

The report averages raters per item and gives, for each system: mean song
relevance, NDCG@K against a pooled ideal (the best ordering of every judged
song for that request, so a system loses credit for relevant songs it missed),
per-scenario results, a paired win/tie/loss comparison, mean playlist fit,
and rater agreement when there are two or more raters. Constraint
satisfaction is still measured separately by the deterministic harness.
Recognizable song metadata could still influence judgments, and no human
relevance results are claimed until real ratings are collected.

## Measured mock evaluation

One local mock-evaluation run on the checked-in catalog completed all
**14/14 expected status checks**. Measured values:

| Metric | Result |
|---|---:|
| Plan validity rate | 0.8571 |
| Constraint satisfaction rate | 0.6429 |
| Mean artist diversity | 0.8242 |
| Retry success rate | 0.3333 |
| Error handling accuracy | 1.0000 |
| Mean TuneWise execution latency | 206.81 ms |

For the eight comparable baseline scenarios, VibeMatch had artist diversity
**1.0000**, checked-constraint satisfaction **0.7500**, and mean ranking-call
latency **2.31 ms**. TuneWise had artist diversity **1.0000**, playlist
constraint satisfaction **1.0000**, and mean orchestration-plus-mock latency
**218.75 ms** in that same run. Latency varies by run. The metric definitions
and scenario-level results are emitted by the evaluator; unsupported,
impossible, and failed scenarios count as unsatisfied rather than being
silently omitted.

These are deterministic fixture outcomes and local timings—not evidence of
live-LLM quality or user satisfaction.

## Project documentation

- [Model card](./model_card.md)
- [AI interactions and privacy](./ai_interactions.md)
- [Code and pipeline walkthrough](./PROJECT_WALKTHROUGH.md)
- [Architecture diagram (current: real catalog, RAG, previews)](./diagrams/tunewise-architecture.mmd)
- [Architecture diagram (earlier agent workflow)](./diagrams/architecture.mmd)
