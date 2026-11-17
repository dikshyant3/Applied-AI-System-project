# TuneWise AI

**TuneWise is a music recommender you can talk to.** You describe what you want
in plain words — *"5 songs that feel like driving through neon city lights at
2am"* — and TuneWise plans a playlist with an LLM, builds it from a catalog of
**1,190 real songs**, checks its own work, fixes the plan if a check fails, and
shows the result as album-cover cards you can **play as 30-second previews**.
It also learns from 👍 / 👎 and recommends songs **without any prompt** on a
personal *For You* page.

Why it matters: most "AI recommenders" let the language model pick songs
freely, so it can invent tracks or ignore rules like "no songs by X". In
TuneWise the LLM only *plans*; ordinary, tested Python code chooses the songs
and verifies every rule, so every song is real and every rule is checked.

---

## 1. The original project: VibeMatch (Modules 1–3)

TuneWise is built on my earlier project, **VibeMatch**. VibeMatch was a
content-based music recommender: you gave it a fixed taste profile (favorite
genre, favorite mood, target energy, and a few optional values), and it scored
every song in a small CSV catalog, applied a diversity penalty so the same
artist or genre didn't repeat, and printed the top songs with a reason for
every point of the score. It could not understand natural language, enforce
playlist rules, check its own output, or use real songs.

VibeMatch's scoring code (`src/recommender.py`) is **still the ranking engine
inside TuneWise and was not changed** — TuneWise wraps it with AI planning,
retrieval, validation, real data and a web interface. You can still run the
original behavior with `python src/main.py --mode genre-first`.

---

## 2. What TuneWise does

| Feature | What it means in practice | Where |
|---|---|---|
| **Agentic workflow** (main AI feature) | The LLM turns your request into a structured plan → Python filters, ranks and builds the playlist → a validator checks every rule → if a fixable check fails, the LLM gets a structured explanation and revises the plan (max 3 attempts). | `src/agent.py`, `src/orchestrator.py`, `src/observations.py` |
| **RAG** (retrieval-augmented generation) | Before planning, TuneWise searches the catalog *by meaning* and gives the planner the closest songs' labels and tags as hints. After filtering, only the songs most similar to your request are ranked. | `src/retrieval.py` |
| **Real catalog from an open API** | 1,190 real tracks collected from the free Deezer API, with album covers and preview links. | `scripts/build_real_catalog.py`, `src/providers/deezer.py` |
| **Audio features measured from sound** | Each 30-second preview was downloaded once and analysed with `librosa` to measure tempo and energy. | `src/audio_features.py` |
| **Web page with previews** | Album-cover cards; ▶ plays the 30-second clip from Deezer. | `app.py` |
| **Personalization** | 👍 saves a song to *Liked songs*; 👍 / 👎 update a saved taste profile; *For You* recommends songs with no prompt. | `src/feedback.py`, `src/personalized.py`, `src/user_profile.py` |
| **Guardrails and logging** | Plans are schema- and evidence-checked; your explicit rules can never be dropped on retry; every step emits a JSON log event. | `src/schemas.py`, `src/playlist_validator.py`, `src/workflow_logging.py` |
| **Reliability testing** | 385 automated tests, a 14-scenario evaluation harness, a retrieval evaluation, and a blinded human-review tool. | `tests/`, `evaluation/` |

### The agent workflow in plain words

1. **Retrieve (RAG).** Your request is turned into a vector (a list of numbers
   that captures its meaning) on your own computer, and compared with every
   song's description. The 8 closest songs' genre/mood labels and playlist tags
   are passed to the planner as *hints*.
2. **Plan (LLM).** The LLM (Groq, `qwen/qwen3.8-27b`) reads your request and the
   hints and returns a JSON plan: favorite genre and mood, target energy, rules
   like "5 songs" or "nothing by Neon Echo", and anything it *can't* support
   (e.g. language). Every value is labelled **explicit** (it must quote your
   exact words) or **inferred** (its guess, with a reason). A value labelled
   explicit without a real quote is downgraded to inferred.
3. **Check the plan.** Python validates the plan against a schema and the
   catalog's real labels. Invalid plans are rejected, never "fixed silently".
4. **Act (tools).** Python removes songs that break hard rules → keeps the most
   similar songs (RAG) → **VibeMatch** scores them → the playlist builder picks
   and orders songs (unique artists, total duration, rising/falling energy).
5. **Observe.** A validator checks the playlist: right size, real unmodified
   catalog songs, no duplicates, duration in range, energy order, etc.
6. **Revise or stop.** If a check fails and it's fixable, the LLM gets a
   structured observation (e.g. `duration_out_of_range`) and may only change
   *soft* preferences — your explicit choices and rules are locked. If the
   request is impossible or unsupported, or 3 attempts are used, TuneWise
   stops and says so instead of pretending.

---

## 3. Architecture overview

Two Mermaid diagrams live in [`diagrams/`](./diagrams):

- **[initial-architecture.mmd](./diagrams/initial-architecture.mmd)** — the
  design I proposed at the start: an LLM planner, the VibeMatch recommender, a
  playlist builder, a validator with a retry loop, saved preferences and logs,
  all on a small synthetic CSV catalog.
- **[final-architecture.mmd](./diagrams/final-architecture.mmd)** — what was
  actually built. Everything **added after the initial design is orange with a
  dashed border**, and the places where **tests or humans check the AI are red**.

How data flows through the final system:

```
input:   request text ──┐
                        ▼
process: RAG retrieve → LLM plan → validate plan → filter → RAG narrow
         → VibeMatch rank → build playlist → validate playlist
         → (fixable failure? revise plan, max 3 attempts)
                        ▼
output:  playlist cards with covers, ▶ previews, reasons, 👍 / 👎
```

Main components:

- **Catalog build (offline, once):** Deezer API → labelled tracks → audio
  measured from previews → `data/real_songs.csv` → embedding index.
- **Agent:** retriever, LLM planner, validators, VibeMatch, playlist builder,
  revision loop.
- **Personalization:** saved profile → taste centroid (average of liked songs'
  vectors, pushed away from disliked ones) → *For You* page.
- **Where results are checked:** `pytest` checks every component; the mock
  evaluation checks 14 end-to-end scenarios; the RAG evaluation measures
  retrieval; humans check results through 👍 / 👎 and the blinded review tool;
  JSON logs record every decision.

---

## 4. Setup

Requires **Python 3.10+** (tested on 3.13, macOS).

```bash
git clone <this repo> && cd Applied-AI-System-project
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

**API key (only for live LLM requests).** Create a file named `.env` in the
project root (it is git-ignored):

```
GROQ_API_KEY=your_key_here
```

Optional: `TUNEWISE_LLM_MODEL`, `TUNEWISE_LLM_TIMEOUT`, and
`TUNEWISE_EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`).

**No key needed for:** the tests, the offline demo, the evaluation harness,
Vibe search, For You, and song previews. The Deezer API is public and needs no
key. The first time retrieval runs, the ~90 MB MiniLM model is downloaded from
Hugging Face once.

**Real catalog:** `data/real_songs.csv` is already in the repo. To rebuild it
(about 20–40 minutes; progress is cached in `data/cache/`, so an interrupted
run resumes):

```bash
python scripts/build_real_catalog.py                # ~1,200 tracks
python scripts/build_real_catalog.py --per-seed 10  # quick, ~300 tracks
```

---

## 5. How to run

### Web app (recommended)

```bash
streamlit run app.py        # opens http://localhost:8501
```

Pick a mode in the sidebar:

| Mode | Needs key? | What you get |
|---|---|---|
| **Live LLM request** | Groq | Type a request; full agent workflow on real or demo songs; RAG on/off toggle; tabs for playlist, agent attempts, plan with evidence, validation checks, retrieval. |
| **For You (no prompt)** | No | Recommendations from your 👍 / 👎 history, plus your *Liked songs* list. |
| **Vibe search** | No | Find real songs by description using retrieval only. |
| **Offline demo** | No | Replays the 14 evaluation scenarios with a scripted planner (the rest of the pipeline is real). |

Real songs show their album cover; press **▶ Play preview** to hear 30 seconds
from Deezer.

### Command line

```bash
# Original VibeMatch, no AI
python src/main.py --mode genre-first

# Full agent workflow offline (scripted planner, no key)
python src/main.py --list-mock-scenarios
python src/main.py --mock-scenario successful-plan-revision

# Live request on real songs with RAG (needs GROQ_API_KEY)
python src/main.py --catalog real --rag --request "songs for a rainy afternoon with a book"

# Feedback and profile (each catalog has its own profile file)
python src/main.py --catalog real --feedback 42 like
python src/main.py --catalog real --show-profile
python src/main.py --catalog real --reset-profile
```

Workflow JSON events are printed to standard error (`2> trace.jsonl` saves them).

### Tests and evaluations

```bash
pytest -q                                   # 385 tests, no key needed
python -m evaluation.evaluate --mode mock   # 14 scenarios, pass/fail + metrics
python -m evaluation.rag_eval               # retrieval quality on real songs
python -m evaluation.rag_eval --live        # full workflow with vs without RAG (uses Groq)
```

---

## 6. Sample interactions

All outputs below are real runs (titles shortened only where noted).

### Example 1 — Live request on real songs, **before vs after RAG**

Request: **"5 songs that feel like driving through neon city lights at 2am"**
(`python src/main.py --catalog real [--rag] --request "..."`, empty profile)

| | Without RAG | With RAG |
|---|---|---|
| Planner's genre / mood | synthwave / **moody** (inferred) | synthwave / **nostalgic** (inferred, "found in catalog matches") |
| Scoring mode chosen | mood-first | genre-first |
| Playlist | Blind Boy – Earl Stan (hip-hop) · Buns of Steel – Ace Buchannon (synthwave) · hot tub DREAM Machine – Tobi Lou (hip-hop) · 2 more hip-hop | See You Next Year – Arimasen · REI – KoruSe · Shadows – Hotel Pools · Divinity – Stellar · Indigo – Foudroie (**all synthwave / nostalgic / nocturnal**) |
| Validation | passed, 1 attempt | passed, 1 attempt |

Without RAG the planner guessed a reasonable vibe, but in this catalog "moody"
is mostly hip-hop, so 4 of 5 songs were hip-hop. With RAG the planner could see
which labels the catalog actually uses for night-drive music and chose
"nostalgic" synthwave, giving a coherent playlist. Each song's explanation now
includes e.g. *"semantic match to request 0.52 (retrieval rank 14)"*. The full
saved run is in [`logs/agent_traces/real-catalog-rag.out.txt`](./logs/agent_traces/real-catalog-rag.out.txt).

### Example 2 — The agent fixes its own plan (offline, reproducible)

Request: **"Give me 5 calm lofi songs for studying, about 20 minutes long, nothing by Neon Echo."**
(`python src/main.py --mock-scenario successful-plan-revision`, output abridged)

```
Plan: favorite_genre lofi [explicit 'lofi'] · exclude_artist Neon Echo [explicit]
      · playlist_size 5 [explicit] · target_duration 20 min [explicit]
Agent loop: SUCCEEDED after 2 of 3 attempt(s)
  attempt 1 (initial):  constraints_unsatisfied; reasons: duration_out_of_range
  attempt 2 (revision): passed; revised preferred_detailed_mood
    rationale: Cozy lofi tracks better meet the requested total duration.
Playlist validation: PASSED (5 of 5 songs)
```

The first playlist was too short. The validator reported
`duration_out_of_range`; the planner changed only a soft preference (focused →
cozy); the locked rules (lofi, 5 songs, no Neon Echo, 20 minutes) stayed. Also
note the guardrail: the LLM labelled the mood "chill" as explicit, but "chill"
is not in the request, so TuneWise downgraded it to *inferred*.

### Example 3 — For You with no prompt

Starting from an empty profile, *For You* shows popular starter songs from
different genres (KATSEYE – *Animal* (pop), ROSÉ – *APT.* (k-pop), sombr –
*back to friends* (indie pop), TLC – *No Scrubs* (r&b)).

After 👍 *Me and Mrs. Jones* (Michael Bublé) and *Here Comes the Sun* (Nina
Simone) and 👎 one metal track, the saved profile becomes *jazz · relaxed ·
mellow · energy 0.25* and *For You* shows:

| Song | Artist | Reason shown |
|---|---|---|
| Take Five | Dave Brubeck | Because you liked "Here Comes the Sun" (similarity 0.54) |
| Fly Me To The Moon | Frank Sinatra | Because you liked "Here Comes the Sun" (similarity 0.55) |
| Cantaloupe Island | Herbie Hancock | Because you liked "Here Comes the Sun" (similarity 0.51) |
| In The Mood | Glenn Miller | Because you liked "Here Comes the Sun" (similarity 0.64) |

### Example 4 — Vibe search (retrieval only, no key)

Query **"songs for crying after a breakup"** → *Save Your Tears* – The Weeknd ·
*Good Reason* – Gracie Abrams · *Cry Me a River* – Justin Timberlake · *How I
Weep* – Norah Jones · *honeybee* – Olivia Rodrigo. None of these songs contain
the words "crying" or "breakup" in their labels; they were found by meaning.

---

## 7. Design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| **The LLM plans; Python picks songs** | Songs can't be invented, rules are always enforced, and everything is testable. | Less "creative" than letting the LLM choose freely. |
| **Explicit vs inferred, with evidence quotes** | Stops the LLM from claiming you asked for something you didn't; explicit values become locked rules. | Some correct guesses get downgraded to "inferred". |
| **Bounded retry loop (max 3) with locked fields** | Lets the agent fix mistakes without looping forever or quietly dropping your rules. | Some fixable requests still fail within 3 attempts (1 of 3 multi-attempt scenarios recovered). |
| **Keep VibeMatch unchanged** | A fair baseline to compare against, and the original tests still pass. | Its simple label-based scoring limits how "smart" ranking can be. |
| **Deezer for real data** | Free, no API key, gives covers and 30 s previews. Spotify removed previews and audio features for new apps in 2024. | Deezer has no mood or energy data, so I had to create it. |
| **Measure audio myself (Option B) + labels from playlists (Option A)** | Deezer's BPM was usually `0`; measuring the preview gives real tempo/energy. Audio can't reliably detect happy vs sad, so mood comes from the playlist a song was found in. | Only tempo and energy are truly measured; the rest are estimates, and labels are playlist-level, not per song. |
| **Build the catalog offline, not per request** | Measuring audio takes ~1 s per song; offline building keeps requests fast and results reproducible. | The catalog is a fixed snapshot of ~1,200 songs. |
| **Fetch preview links on click** | Deezer preview links expire after ~15 minutes, so storing them would break. | One small API call per play. |
| **Local embeddings (MiniLM via FastEmbed)** | No key, no cost, request text never leaves the computer; ONNX avoids installing PyTorch. | Slightly weaker than large hosted embedding models (bge-small scored 0.74 vs 0.71). |
| **RAG gives hints, not evidence** | Retrieved songs help the LLM guess vibes, but can never become "explicit" user requirements or rules. | The hints can still steer a guess in the wrong direction. |
| **Two catalogs** | The synthetic one keeps tests and the evaluation deterministic; the real one is for actual use. | Two profiles (feedback refers to song IDs, which differ per catalog). |

---

## 8. Testing summary

**385 of 385 automated tests passed (1 skipped — an optional live API test).**
They cover the recommender, plan schemas, the LLM interpreter, the agent loop,
the playlist builder and validator, feedback, the Deezer client (with fake
network responses), audio feature math, retrieval, For You, the CLI and the
web app.

### Evaluation harness (offline, reproducible)

`python -m evaluation.evaluate --mode mock` runs 14 predefined scenarios and
compares each outcome to the expected one: **14 / 14 matched**, including
invalid LLM output, an API rate limit, impossible requests and running out of
retries — all handled without crashing.

| Metric (14 scenarios) | Result |
|---|---:|
| Expected outcome matched | **14 / 14** |
| Plan validity rate | 0.86 |
| Constraint satisfaction (all scenarios, failures count as 0) | 0.64 |
| Retry success rate | 0.33 (1 of 3) |
| Error-handling accuracy | 1.00 |

On the 8 scenarios that the original VibeMatch can also run, **TuneWise met
1.00 of playlist rules vs 0.75 for VibeMatch** (VibeMatch can't enforce energy
ordering).

### Retrieval (RAG) evaluation on real songs

`python -m evaluation.rag_eval`: 15 descriptive requests that avoid label
words; a hit is a top-10 song with an accepted genre or mood.

| Retriever | Precision@10 | Better than random |
|---|---:|---:|
| **all-MiniLM-L6-v2 (TuneWise default)** | **0.71** | **15 / 15** |
| bge-small-en-v1.5 (`--compare`) | 0.74 | 14 / 15 |
| Keyword overlap | 0.37 | 7 / 15 |
| Random | 0.20 | – |

### Example evaluation table (parseable)

| Test input | Evaluation criteria | Result |
|---|---|---|
| "5 calm lofi songs … about 20 minutes … nothing by Neon Echo" | Recover from too-short playlist, keep locked rules | Pass — fixed on attempt 2 |
| Malformed LLM JSON | Reject safely, report `invalid_plan` | Pass |
| Simulated API rate limit | Report `api_failure`, no crash | Pass |
| Request needing song language | Report as unsupported, don't guess | Pass |
| Impossible request (more songs than exist) | Report `impossible_request` | Pass |
| "neon city lights at 2am" (live, real songs) | Coherent genre in playlist | Without RAG: 1/5 synthwave · With RAG: 5/5 |
| "groovy bass lines and disco vibes from the seventies" (retrieval) | Top-10 are funk | Weak — 0.1 precision |

### What worked, what didn't, what I learned

- **Worked:** guardrails (14/14 expected outcomes), rule enforcement (1.00 vs
  0.75), meaning-based retrieval (0.71 vs 0.37 keywords), and the agent
  repairing a failed plan.
- **Didn't:** across 15 live requests, RAG did **not** raise the label-based
  score of full playlists (1.00 without vs 0.99 with) — the planner already
  maps most requests to the right labels, so this metric can't show the
  difference that Example 1 shows. Retrieval was weak on era-specific requests
  ("seventies disco"). Only 1 of 3 multi-attempt scenarios recovered. For You
  tends to stay inside one genre after a few likes.
- **Learned:** a metric can be "saturated" and hide real differences; checking
  individual outputs (Example 1) mattered as much as averages. And free APIs
  have real limits — Groq rate-limited my evaluation until I added backoff.

**Human review (pending):** `evaluation/quality.py` exports a blinded review
(songs shuffled, system names hidden) and scores ratings with mean relevance,
NDCG and rater agreement. The tool is built and tested, but no human ratings
have been collected yet, so no human-relevance results are claimed.

---

## 9. Limitations (short version)

- The real catalog is ~1,200 songs from popular Deezer playlists; recent and
  English-language music is over-represented.
- Genre/mood labels come from the playlist a song was found in, not per-song
  annotation; only tempo and energy are measured from audio. Release years are
  often reissue dates.
- No song language, lyrics or live Deezer search; unsupported requests are
  reported, not guessed.
- Profiles are local JSON files; there are no user accounts.

Full details, bias discussion, misuse and the AI-collaboration reflection are
in the **[model card](./model_card.md)**.

---

## 10. Reflection

This project taught me that the most useful thing an AI component can do is
often the narrowest one. Letting the LLM *plan* while ordinary code *decides
and checks* made the system more trustworthy than a larger "do everything"
prompt would have been, and it made every failure visible and testable. I also
learned that real data is messy: the "easy" step of using a public API turned
into measuring audio myself, guessing labels from playlists and handling
expiring links and rate limits. Finally, evaluation changed my conclusions —
RAG clearly helped retrieval and one concrete example, but my aggregate
playlist metric showed no gain, and I had to report that honestly instead of
assuming the feature worked.

---

## 11. Project map

```
app.py                       Streamlit web page (all four modes)
src/agent.py                 LLM planner, prompts, evidence checks, revision
src/orchestrator.py          Plan → Act → Observe → Revise loop, RAG hooks
src/retrieval.py             RAG: song documents, embeddings, search
src/personalized.py          For You recommendations
src/recommender.py           Original VibeMatch scoring (unchanged)
src/playlist_builder.py      Playlist selection and ordering
src/playlist_validator.py    Deterministic playlist checks
src/observations.py          Turns failures into structured observations
src/schemas.py               Plan schema and validation
src/feedback.py              Likes / dislikes → preferences
src/user_profile.py          Local JSON profile storage
src/providers/deezer.py      Deezer API client (no key)
src/audio_features.py        librosa audio measurement
scripts/build_real_catalog.py  Builds data/real_songs.csv
evaluation/                  evaluate.py (14 scenarios), rag_eval.py, quality.py (human review)
logs/agent_traces/           Saved agent runs (summaries + JSON traces)
diagrams/                    initial-architecture.mmd, final-architecture.mmd
tests/                       385 automated tests
```

More documentation: [Model card](./model_card.md) ·
[AI interactions, traces and privacy](./ai_interactions.md) ·
[Code walkthrough](./PROJECT_WALKTHROUGH.md)
