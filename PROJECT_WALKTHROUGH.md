# TuneWise Project Walkthrough

This guide explains how to follow a request through the code, what each layer
does, and the shapes of the data passed between them. Start with the short
[README](./README.md) for setup and commands; use this guide when you want to
trace the implementation end to end.

## 1. The project in one picture

The original **VibeMatch** recommender is still the ranking engine. **TuneWise**
adds a Python-controlled planning and playlist workflow around it:

```text
request
  -> load catalog and local profile
  -> LLM returns structured plan data
  -> Python interprets and validates the plan
  -> filter catalog by hard constraints
  -> original VibeMatch scores/ranks eligible songs
  -> playlist builder selects and orders real songs
  -> deterministic validator checks the playlist
  -> return result, or send a structured observation for a bounded revision
  -> record JSON workflow events
```

The LLM does not choose song IDs, score songs, create metadata, execute code,
or edit the profile. Saved preference values are kept in Python and are not
sent in the planning prompt. They are merged into the plan locally.

## 2. First run and CLI routing

Read [README.md](./README.md) first for installation, environment variables,
and commands. The CLI entry point is [main.py](./src/main.py).

`main()` parses command-line options and chooses one of two main paths:

1. `--mode genre-first` or `--mode mood-first` (the default path) runs the
   original VibeMatch examples directly. It loads the CSV and calls
   `recommend_songs`; it does not need the LLM.
2. `--request "..."` loads the local catalog/profile and calls
   `run_recommendation_workflow`. This path needs the live model credentials
   unless a test or caller injects a mock client.

Profile commands (`--feedback`, `--show-profile`, `--reset-profile`) are
handled locally by `main()` and do not call the LLM. Request-mode workflow
events go to standard error; the human-readable recommendation output goes to
standard output.

## 3. The data, from CSV row to plan

### 3.1 Catalog input

The catalog is [songs.csv](./data/songs.csv). Each row is loaded by
[`load_songs`](./src/recommender.py#L76) into a Python dictionary. A typical
row resembles:

```python
{
    "id": 2,
    "title": "Midnight Coding",
    "artist": "LoRoom",
    "genre": "lofi",
    "mood": "chill",
    "energy": 0.42,
    "tempo_bpm": 78,
    "popularity": 58,
    "release_decade": 2020,
    "detailed_mood": "cozy",
    "instrumentalness": 0.18,
    "speechiness": 0.04,
    "duration_seconds": 165
}
```

Values are typed while loading: IDs, tempo, popularity, and decade are
integers; score attributes and duration are numeric. Every recommendation
must ultimately refer to one of these catalog rows.

### 3.2 The LLM response is only a proposal

[`agent.py`](./src/agent.py) asks the provider for a constrained JSON object.
Its shape is roughly:

```json
{
  "preferences": {
    "favorite_genre": {
      "value": "lofi",
      "source": "explicit",
      "evidence": "lofi",
      "reason": null
    },
    "favorite_mood": {
      "value": "chill",
      "source": "explicit",
      "evidence": "calm",
      "reason": null
    },
    "target_energy": {
      "level": "low",
      "number": null,
      "source": "inferred",
      "evidence": null,
      "reason": "calm music is usually low energy"
    }
  },
  "strategy": null,
  "playlist": {
    "playlist_size": {"value": 5, "evidence": "5 songs"},
    "unique_artists": null,
    "energy_progression": null
  },
  "hard_constraints": [],
  "duration_minutes": null,
  "unsupported_requests": []
}
```

Optional preference fields may be present too. This is raw model output, not
trusted application state. The model response has no song list or song IDs.

The source labels mean:

- `explicit`: supported by the current request. The Python interpreter checks
  the evidence text; a claim without a real quote is treated as inferred.
- `inferred`: a soft preference the model derived from the request.
- `saved`: a previous preference added by Python, not by the model.
- `default`: no value was selected, so the existing scorer's defaults apply.

### 3.3 Python interpretation, merge, and typed plan

`create_recommendation_plan()` in [agent.py](./src/agent.py#L649) performs the
initial plan call. `_Interpreter.run()` parses each response section and
collects problems. `_Interpreter._merge()` uses this precedence:

```text
current explicit request > saved profile > inferred response > default
```

Saved preferences never override current explicit choices. They also do not
override a matching hard constraint. Again, saved values are merged inside
Python; they are not included in the LLM prompt.

The interpreted JSON then passes through [`validate_plan`](./src/schemas.py#L249)
and becomes a typed `RecommendationPlan`, made up of:

- `Preference`: one preference value and its source.
- `HardConstraint`: a catalog filter, such as allowed genre or excluded
  artist.
- `PlaylistSpec`: size, whether artists must be unique, and requested energy
  progression.

Soft preferences affect ranking; hard constraints remove ineligible songs.
Playlist rules are applied after ranking. Keeping these meanings separate
prevents a soft preference from being silently treated as a hard rule.

[`to_recommender_kwargs`](./src/schemas.py#L296) converts only the supported
scoring preferences into the original recommender's `user_prefs`, `k`, and
`mode` arguments. It does not pass playlist rules or hard constraints to the
scorer.

The `PlanningResult` around the plan also carries provenance, warnings,
unsupported requests, model name, requested duration, and (for revisions)
rationale and changed fields.

## 4. How ranking becomes a playlist

### 4.1 Keep only songs that satisfy hard constraints

`_eligible_songs()` in [orchestrator.py](./src/orchestrator.py#L537) checks
each catalog row against every `HardConstraint`. It does not change song
metadata.

### 4.2 Rank with original VibeMatch

[`recommend_songs`](./src/recommender.py#L192) uses `_score_song()` to score
each eligible song against the plan's soft preferences, then greedily selects
the best-ranked songs while applying the existing artist/genre diversity
penalties. A ranked result has this shape:

```python
(song_dictionary, numeric_score, "reason 1; reason 2; ...")
```

`score_song()` exposes the same original feature scorer for one song. The
algorithm is deliberately unchanged by TuneWise.

### 4.3 Build and order the playlist

[`build_playlist`](./src/playlist_builder.py#L174) consumes the ranked tuples
and a `PlaylistRequirements` object. It selects only ranked catalog songs:

- avoids duplicate IDs;
- skips repeated artists when uniqueness is required;
- honors a requested size or duration-driven selection;
- optionally sorts selected tracks by energy (stable rank breaks ties);
- reports skipped songs and shortfalls instead of inventing or relaxing data.

`PlaylistEntry` retains the catalog song, its original rank, score,
explanation, and final playlist position.

### 4.4 Validate; do not assume success

[`validate_playlist`](./src/playlist_validator.py#L128) checks song metadata,
catalog membership, duplicate IDs, size, unique artists, hard constraints,
energy progression, duration (when requested and available), and unsupported
requests. It returns a `ValidationReport` with a status and evidence for each
check.

Possible check results include:

- `passed`: the check holds;
- `violated`: a requirement was not met;
- `unsupported`: the data needed to check/apply it is unavailable;
- `infeasible`: the catalog and immutable requirements make success
  impossible.

An infeasible/unsupported result is not a successful playlist, even if a
partial list is returned for inspection.

## 5. The Plan → Act → Observe → Revise loop

[`run_recommendation_workflow`](./src/orchestrator.py#L161) validates inputs,
loads the catalog/client, then creates `_AgentLoop`. The loop is bounded by
`max_attempts` (three by default).

For each attempt, `_AgentLoop._attempt()` does this:

1. **Plan:** call `create_recommendation_plan()` initially, or
   `revise_recommendation_plan()` after a correctable observation.
2. **Re-check:** validate the plan again in the orchestrator and resolve its
   scoring strategy.
3. **Repeat detection:** compare a fingerprint of all parameters that can
   affect tool output. An already-run plan stops without repeating the tools.
4. **Act:** `_execute_plan()` calls hard filtering, the original recommender,
   playlist building, and validation.
5. **Observe:** `observe_validation()` in [observations.py](./src/observations.py#L146)
   turns validator results into machine-readable failure codes.
6. **Revise or stop:** only retryable invalid plans/unsatisfied constraints
   may be retried. Unsupported metadata, impossible requests, API/tool
   failures, and success stop immediately. Revisions may change only allowed
   soft preferences/strategy; Python preserves and checks hard constraints
   and explicit fields.

Failure categories include `invalid_plan`, `constraints_unsatisfied`,
`unsupported_metadata`, `impossible_request`, `api_failure`, and
`tool_failure`. The trace records each attempt, plan snapshot, tool calls,
validation, observation, final status, and which real playlist is returned.
No playlist is fabricated for attempts that did not build one.

### Concrete revision example

The `successful-plan-revision` scenario in
[scenarios.json](./evaluation/scenarios.json) requests five calm lofi tracks
about 20 minutes long and excludes Neon Echo. The first valid plan selects
tracks whose total is below the allowed duration window. The validator reports
a retryable duration failure; the observation is sent back to the revision
function. The mock revises an inferred detailed mood to `cozy`, while the
explicit request, size, duration, and excluded artist remain fixed. The second
execution is checked again; it succeeds only if its actual selected songs
meet the duration and other checks.

## 6. Feedback and persistent preferences

[`UserProfileStore`](./src/user_profile.py#L21) reads/writes local JSON
atomically. The default path is `~/.tunewise/profile.json`; the CLI can
override it with `--profile-path`.

The saved shape is:

```json
{
  "version": 1,
  "preferences": {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.42
  },
  "feedback": [
    {"song_id": 2, "action": "like"},
    {"song_id": 1, "action": "skip"}
  ]
}
```

[`submit_feedback`](./src/feedback.py#L26) first validates that the song ID is
an integer present in the catalog and that the action is `like`, `dislike`, or
`skip`. It recomputes preferences from feedback deterministically:

- categorical preferences use like-minus-dislike votes; highest positive vote
  wins, and catalog order breaks ties;
- numeric preferences are the mean feature values of liked songs;
- dislikes subtract from categorical votes; if no positive vote remains, that
  preference is cleared;
- skips are recorded but do not influence preference values.

The next `--request` loads these preferences. A preference explicitly stated
by the new request still wins. The LLM never receives a profile file path or
writes profile data.

Useful commands:

```bash
python src/main.py --feedback 2 like
python src/main.py --show-profile
python src/main.py --reset-profile
```

## 7. Structured events and evaluation

### Workflow logs

[`log_workflow_event`](./src/workflow_logging.py#L22) emits one JSON object per
event. Events use a generated or supplied `request_id`; examples include
`workflow_started`, `initial_plan_validated`, `tool_executed`,
`validation_result`, `retry_scheduled`, and `workflow_finished`. Plan log
summaries omit evidence quotes and raw request text. Preference values and
hard-constraint values are still present, so protect log files.

### Evaluation harness

- [scenarios.json](./evaluation/scenarios.json) defines the deterministic
  scenarios and their expected final statuses.
- [`evaluate_scenarios`](./evaluation/evaluate.py#L191) runs them with
  `MockLLMClient` in mock mode and measures actual elapsed time.
- `_baseline_metrics()` runs the original `recommend_songs` using explicit
  comparable inputs. The comparison reports artist diversity, selected
  constraint satisfaction, and component latency.

Metrics are computed over the scenario set:

- **Plan validity:** fraction with at least one valid plan.
- **Constraint satisfaction:** fraction with a returned playlist whose full
  validator report passed.
- **Artist diversity:** average distinct-artist ratio; no-song outcomes count
  as zero.
- **Retry success:** fraction of scenarios that used more than one attempt and
  ultimately succeeded.
- **Error handling:** fraction whose observed final status matched the
  scenario's expected status; mock scenarios only.
- **Execution latency:** measured elapsed workflow time. Mock latency does
  not include a network request.

Run the reproducible evaluation with:

```bash
python -m evaluation.evaluate --mode mock
```

`--mode live` is separate and opt-in. It sends requests to the configured
provider, may incur costs, and is not the source of the checked-in mock
measurements.

## 8. File map and recommended reading order

| Read | File | Main job |
|---:|---|---|
| 1 | [main.py](./src/main.py) | CLI argument routing, profile commands, output rendering. |
| 2 | [recommender.py](./src/recommender.py) | CSV loading and original unchanged scoring/ranking. |
| 3 | [schemas.py](./src/schemas.py) | Typed plans, schema validation, conversion to scorer arguments. |
| 4 | [agent.py](./src/agent.py) | LLM client, response interpretation, saved/explicit merge, plan revision. |
| 5 | [orchestrator.py](./src/orchestrator.py) | Workflow entry point, bounded agent loop, tool orchestration and traces. |
| 6 | [playlist_builder.py](./src/playlist_builder.py) | Select and order catalog-backed songs from ranked tuples. |
| 7 | [playlist_validator.py](./src/playlist_validator.py) | Deterministic constraint checks and infeasibility evidence. |
| 8 | [observations.py](./src/observations.py) | Structured failure categories, attempts, and execution trace. |
| 9 | [user_profile.py](./src/user_profile.py) and [feedback.py](./src/feedback.py) | JSON persistence and deterministic feedback updates. |
| 10 | [workflow_logging.py](./src/workflow_logging.py) | Request-scoped JSON event output. |
| 11 | [evaluate.py](./evaluation/evaluate.py) and [scenarios.json](./evaluation/scenarios.json) | Mock/live evaluation and expected outcomes. |

Tests mirror the code layers: [test_recommender.py](./tests/test_recommender.py),
[test_schemas.py](./tests/test_schemas.py),
[test_agent.py](./tests/test_agent.py),
[test_orchestrator.py](./tests/test_orchestrator.py),
[test_agent_loop.py](./tests/test_agent_loop.py),
[test_playlist.py](./tests/test_playlist.py),
[test_feedback.py](./tests/test_feedback.py),
[test_workflow_logging.py](./tests/test_workflow_logging.py), and
[test_evaluation.py](./tests/test_evaluation.py).

To follow a real request in a debugger, set breakpoints in this order:

1. [`main()`](./src/main.py#L187)
2. [`_run_request()`](./src/main.py#L167)
3. [`run_recommendation_workflow()`](./src/orchestrator.py#L161)
4. [`create_recommendation_plan()`](./src/agent.py#L649)
5. [`_AgentLoop._attempt()`](./src/orchestrator.py#L274)
6. [`_execute_plan()`](./src/orchestrator.py#L456)
7. [`recommend_songs()`](./src/recommender.py#L192)
8. [`build_playlist()`](./src/playlist_builder.py#L174)
9. [`validate_playlist()`](./src/playlist_validator.py#L128)
10. [`observe_validation()`](./src/observations.py#L146)

The actual source and tests are the authority if this guide and an
implementation detail ever differ.
