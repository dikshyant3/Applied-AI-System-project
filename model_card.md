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

- Recommend tracks that already exist in the selected catalog (`data/songs.csv`
  or `data/real_songs.csv`) and let the listener hear a 30-second preview.
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
- **Real catalog:** `data/real_songs.csv`, 1,190 tracks by 941 artists collected
  from public Deezer playlists for 30 seed labels (20 genres). Cover art, links and
  30-second previews come from Deezer; audio is streamed by the listener and never
  stored by TuneWise.
  - *Measured* from each preview with librosa: tempo and energy (loudness, onset
    strength and brightness, as a percentile within the catalog). Tempo is halved
    for slow genres when the beat tracker reports double time.
  - *Estimated*: danceability and acousticness (audio proxies), valence (mood label
    prior + brightness), instrumentalness and speechiness (genre priors).
  - *Labels*: genre, mood and detailed mood are inherited from the seed search, not
    annotated per song, so they are weak and sometimes wrong. Release decade comes
    from Deezer release dates, which are often reissue dates (58% of tracks fall in
    the 2020s).
- **RAG embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
  (ONNX via FastEmbed), run locally; request text
  is embedded on the user's machine. Fallback: lexical hashing retriever.
- **Profile:** optional JSON at `~/.tunewise/profile.json` (`profile-real.json`
  for the real catalog), overridable by `--profile-path`.

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

`evaluation/scenarios.json` includes fourteen deterministic cases. The latest
measured mock run had 14/14 expected status matches, 0.8571 plan validity,
0.6429 constraint satisfaction, 0.8242 mean artist diversity, 0.3333 retry
success, 1.0000 expected-status accuracy, and 206.81 ms mean end-to-end
orchestration latency. The rate metrics use all fourteen scenarios, including
expected unsupported, impossible, invalid, and failed outcomes; the evaluator
does not treat those outcomes as successes.

The VibeMatch comparison uses eight scenarios with explicit matching baseline
preferences. On that run, baseline artist diversity was 1.0000, checked
constraint satisfaction was 0.7500, and recommender-call latency was 2.31 ms.
TuneWise artist diversity was 1.0000, constraint satisfaction was 1.0000, and
orchestration-plus-mock latency was 218.75 ms. This is a scripted, local
comparison: the latency measurements are not comparable to live API latency,
and no human relevance judgments were collected yet (the blinded review
workflow is described in the README).

Live evaluation is opt-in (`python -m evaluation.evaluate --mode live`),
uses real provider calls, may cost money, and is not included in the mock
results above.

## Retrieval (RAG) evaluation

`python -m evaluation.rag_eval` scores label precision@10 on 15 descriptive
queries (`evaluation/rag_queries.json`) that avoid the label words. On the real
catalog: MiniLM semantic retrieval **0.71** (bge-small-en-v1.5: 0.74), lexical
retrieval 0.37, random base rate 0.20; MiniLM beat random on 15/15 queries. Its weakest queries were
"groovy bass lines and disco vibes from the seventies" (0.1) and "angry fast songs
about not fitting in" (0.2). Because labels are inherited from
playlists, this is a relative comparison, not a human relevance judgment.

**Full workflow, live LLM (`--live`, one run, 15 queries, 5 songs each):**
TuneWise without RAG reached label precision 1.00 and TuneWise + RAG 0.99
(14 ties, 1 worse: the "seventies disco" request, where plain TuneWise reported the
decade as unsupported metadata and RAG-narrowed ranking let one non-funk song in).
So on this metric RAG gives **no measurable gain** once the LLM planner is involved:
the planner already maps these descriptive requests to the right genre/mood labels,
and VibeMatch then ranks by exactly those labels, so the metric saturates. RAG's
demonstrated value here is retrieval without an LLM (Vibe search, 0.71 vs 0.37
lexical) and tag-grounded explanations; showing a playlist-quality gain would need
human ratings rather than these label checks.


## Limitations and risks

- The synthetic dataset is small; the real catalog (~1,200 tracks) is a sample of
  popular Deezer playlists and over-represents recent and English-language music.
- Real-catalog genre/mood labels are playlist-level, and most audio features other
  than tempo and energy are estimates.
- For You can form a filter bubble: after a few likes in one genre, the
  similarity pool is dominated by that genre (VibeMatch's genre-diversity penalty
  only reorders within the pool). Synthetic demo songs have no cover or preview.
- Deezer previews depend on Deezer's availability, regional rights and terms.
- The model can misunderstand a request; deterministic validation limits the
  effects but does not guarantee recommendation relevance.
- Unsupported catalog properties such as language are not inferred.
- Feedback uses deterministic like/dislike votes and liked-song feature means;
  it is a simple preference heuristic, not a learned ranking model.
- Local profile and stderr-log retention are the operator's responsibility.
- Provider behavior, availability, and model outputs can change over time.

## Planned or unsupported

No user accounts, cloud profile storage, live per-request catalog lookup (the
catalog is built offline), human-rated quality benchmark, or production
service-level guarantees are implemented.
