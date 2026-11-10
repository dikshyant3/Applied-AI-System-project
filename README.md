# TuneWise AI

TuneWise extends the original VibeMatch music recommender with a structured
LLM planning layer, deterministic playlist construction and validation,
bounded plan revision, local user feedback, structured workflow events, and an
offline evaluation harness.

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
- **Evaluation:** ten scripted mock-LLM scenarios cover normal requests,
  saved/request preference conflicts, unsupported metadata, playlist rules,
  impossible requests, malformed responses, revision success, retry
  exhaustion, and a simulated API failure.

## Not implemented or unsupported

- The catalog currently does not provide language or other arbitrary
  attributes; unsupported requests are reported, not guessed.
- Feedback is local to one JSON profile. There is no account system, cloud
  synchronization, deletion service, or automated profile migration.
- The evaluation harness supports live calls only when explicitly run with
  `--mode live`; no live-model benchmark results are claimed here.
- Recommendation relevance has not been evaluated with human judgments.
- There is no arbitrary code execution, autonomous song lookup, or use of
  metadata outside the catalog.

## Install

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

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

Request a TuneWise playlist (requires the live model credentials):

```bash
python src/main.py --request "calm lofi music for studying"
```

The default profile is `~/.tunewise/profile.json`. Use `--profile-path PATH`
to select another local file. Feedback actions only accept IDs in
`data/songs.csv`:

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

The evaluator compares the original recommender and TuneWise on the same four
successful, supported scenarios with baseline inputs. Baseline latency is the
ranking call only; TuneWise latency includes orchestration and a scripted mock
planner. These are component timings, not a live-service performance
comparison.

## Measured mock evaluation

One local mock-evaluation run on the checked-in catalog completed all
**10/10 expected status checks**. Measured values:

| Metric | Result |
|---|---:|
| Plan validity rate | 0.8000 |
| Constraint satisfaction rate | 0.5000 |
| Mean artist diversity | 0.7538 |
| Retry success rate | 0.3333 |
| Error handling accuracy | 1.0000 |
| Mean TuneWise execution latency | 90.33 ms |

For the four comparable baseline scenarios, VibeMatch had artist diversity
**1.0000**, checked-constraint satisfaction **0.7500**, and mean ranking-call
latency **0.92 ms**. TuneWise had artist diversity **1.0000**, playlist
constraint satisfaction **1.0000**, and mean orchestration-plus-mock latency
**101.97 ms** in that same run. Latency varies by run. The metric definitions
and scenario-level results are emitted by the evaluator; unsupported,
impossible, and failed scenarios count as unsatisfied rather than being
silently omitted.

These are deterministic fixture outcomes and local timings—not evidence of
live-LLM quality or user satisfaction.

## Project documentation

- [Model card](./model_card.md)
- [AI interactions and privacy](./ai_interactions.md)
- [Code and pipeline walkthrough](./PROJECT_WALKTHROUGH.md)
- [Architecture diagram](./diagrams/architecture.mmd)
