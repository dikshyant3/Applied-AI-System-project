# TuneWise AI Model Card

**Status:** Implemented features are identified below; this card does not
claim a production deployment or independent model evaluation.

## System overview

TuneWise is a music recommendation application built on the original VibeMatch
content-based scorer. A configured Groq-compatible LLM interprets a natural
language request into a structured plan. Python validates the plan, runs the
existing scorer and playlist tools, validates the output, and may request a
bounded revision after a retryable failure.

The model does not select tracks, rank tracks, calculate scores, create
metadata, write profile files, or execute generated code. The scoring
algorithm in `src/recommender.py` is unchanged.

## Intended use

- Recommend tracks that already exist in `data/songs.csv`.
- Translate supported natural-language preferences and playlist rules into
  validated fields.
- Use local, optional feedback to improve stored soft preferences.
- Demonstrate bounded tool-using workflow behavior with reproducible mock
  tests and evaluation scenarios.

This project is not a general music search engine, a streaming service, or a
high-stakes decision system.

## Model and data

- **Live provider:** Groq-compatible HTTP chat completions.
- **Default model:** `qwen/qwen3.8-27b` (override with
  `TUNEWISE_LLM_MODEL`).
- **Generation settings:** temperature 0; structured JSON-schema response.
- **Mock evaluation:** scripted `MockLLMClient` responses; no provider call.
- **Catalog:** local CSV with genre, mood, artist, energy, and selected
  supported attributes. Duration exists in this checked-in catalog, but
  applications must not assume other catalogs contain it.
- **Profile:** optional JSON at `~/.tunewise/profile.json`, overridable by
  `--profile-path`.

## Safeguards

- Plans are parsed and validated against supported schema values and catalog
  vocabulary.
- Explicit user choices and hard constraints are immutable during revision.
- The orchestrator caps total attempts at three by default and detects repeated
  plans.
- Impossible and unsupported conditions are reported; they are not relaxed.
- Playlist songs and metadata come from catalog rows only.
- Saved profiles are modified only by application feedback/profile code.
- Logs omit raw requests, evidence excerpts, song titles, and credentials.

## Evaluation

`evaluation/scenarios.json` includes ten deterministic cases. The latest
measured mock run had 10/10 expected status matches, 0.8000 plan validity,
0.5000 constraint satisfaction, 0.7538 mean artist diversity, 0.3333 retry
success, 1.0000 expected-status accuracy, and 90.33 ms mean end-to-end
orchestration latency. The rate metrics use all ten scenarios, including
expected unsupported, impossible, invalid, and failed outcomes; the evaluator
does not treat those outcomes as successes.

The VibeMatch comparison uses four scenarios with explicit matching baseline
preferences. On that run, baseline artist diversity was 1.0000, checked
constraint satisfaction was 0.7500, and recommender-call latency was 0.92 ms.
TuneWise artist diversity was 1.0000, constraint satisfaction was 1.0000, and
orchestration-plus-mock latency was 101.97 ms. This is a scripted, local
comparison: the latency measurements are not comparable to live API latency,
and no human relevance judgments were collected.

Live evaluation is opt-in (`python -m evaluation.evaluate --mode live`),
uses real provider calls, may cost money, and is not included in the mock
results above.

## Limitations and risks

- The dataset is small and may not represent a user's music catalog or taste.
- The model can misunderstand a request; deterministic validation limits the
  effects but does not guarantee recommendation relevance.
- Unsupported catalog properties such as language are not inferred.
- Feedback uses deterministic like/dislike votes and liked-song feature means;
  it is a simple preference heuristic, not a learned ranking model.
- Local profile and stderr-log retention are the operator's responsibility.
- Provider behavior, availability, and model outputs can change over time.

## Planned or unsupported

No user accounts, cloud profile storage, automated external metadata lookup,
human-rated quality benchmark, or production service-level guarantees are
implemented.
