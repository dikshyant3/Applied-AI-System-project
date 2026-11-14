# AI Interactions and Data Flow

## Implemented live request

1. The CLI loads the local catalog and optional saved preferences.
2. The planning client sends the natural-language request, supported catalog
   vocabulary, and response schema to the configured Groq-compatible model.
   Saved preference values stay in Python and are not included in the LLM
   prompt.
3. Python validates the returned plan and applies deterministic precedence:
   current explicit user preferences override saved values, which override
   inferred preferences.
4. Python filters/ranks real catalog rows, builds a playlist, and validates
   supported requirements.
5. On a correctable failure, Python sends the LLM a structured observation and
   an allowlist of revisable soft fields. Hard constraints and explicit fields
   are copied and checked by Python.
6. The bounded loop returns a real validated playlist or an explicit failure
   status; it never invents success.

The model's output is JSON data only. It cannot run Python or modify the
profile file. `src/feedback.py` updates the local profile from a user-selected
catalog song and action.

## Local profile and feedback

The default file is `~/.tunewise/profile.json`; `--profile-path` selects a
different file. Feedback supports `like`, `dislike`, and `skip` for catalog
IDs. Likes contribute positively and dislikes negatively to categorical
preference votes; positive score wins, with catalog order resolving ties.
Numeric preferences are means over liked songs. A skip is retained in history
but does not alter preference values. The user can inspect or clear this data
with `--show-profile` or `--reset-profile`.

The profile is local JSON, not encrypted. Protect its filesystem permissions
and avoid sharing it. Current request preferences remain authoritative over
historical profile values.

## Logs

Each workflow event is JSON and includes the request ID, validated plan
summary (values and sources, without evidence excerpts), selected strategy,
tool events, candidate counts, validation check statuses, retries, final
status, and duration. Raw request text, evidence quotes, song titles, and API
credentials are omitted. Preference values and hard constraints can still be
sensitive; logs are emitted to stderr and should be retained only as needed.
Applications using the Python API can call
`src.workflow_logging.configure_workflow_logging()` to configure JSON event
output.

## Saved agent reasoning traces

These traces come from the real CLI and workflow code. Only the planner's
replies are scripted, so they run offline and give the same output every time
(except timing). Each run has a readable summary (`.out.txt`) and the full JSON
event log (`.trace.jsonl`):

| Scenario | What the agent does | Files |
|---|---|---|
| `successful-plan-revision` | Plan → build → validation fails (playlist too short) → planner revises the detailed mood → rebuild → passes | [summary](./logs/agent_traces/successful-plan-revision.out.txt), [JSON trace](./logs/agent_traces/successful-plan-revision.trace.jsonl) |
| `retry-budget-exhaustion` | Plan → fails → revises → fails again → stops at the 2-attempt limit and reports the failure instead of returning a bad playlist as valid | [summary](./logs/agent_traces/retry-budget-exhaustion.out.txt), [JSON trace](./logs/agent_traces/retry-budget-exhaustion.trace.jsonl) |

Regenerate them with:

```bash
python src/main.py --mock-scenario successful-plan-revision \
  > logs/agent_traces/successful-plan-revision.out.txt \
  2> logs/agent_traces/successful-plan-revision.trace.jsonl
```

Excerpt from `successful-plan-revision` (request: *"Give me 5 calm lofi songs
for studying, about 20 minutes long, nothing by Neon Echo."*):

```text
Agent loop: SUCCEEDED after 2 of 3 attempt(s) - playlist passed validation on attempt 2
  attempt 1 (initial): constraints_unsatisfied; reasons: duration_out_of_range
  attempt 2 (revision): passed; revised preferred_detailed_mood
    rationale: Cozy lofi tracks better meet the requested total duration.
```

The matching JSON events, trimmed to the decision points:

```json
{"attempt": 1, "event": "initial_plan_validated", "plan": {"preferences": {"favorite_genre": {"source": "explicit", "value": "lofi"}, "preferred_detailed_mood": {"source": "inferred", "value": "focused"}}, "hard_constraints": [{"type": "exclude_artist", "values": ["Neon Echo"]}], "target_duration_minutes": 20}, "strategy": "genre-first"}
{"attempt": 1, "event": "tool_executed", "tool": "filter_hard_constraints", "eligible_songs": 246, "catalog_size": 250}
{"attempt": 1, "event": "tool_executed", "tool": "recommend_songs", "ranked_candidate_count": 246}
{"attempt": 1, "event": "tool_executed", "tool": "build_playlist", "selected_count": 5}
{"attempt": 1, "event": "validation_result", "outcome": "failed", "checks": [{"name": "duration", "status": "violated"}]}
{"attempt": 1, "event": "retry_scheduled", "failure_codes": ["duration_out_of_range"], "next_attempt": 2}
{"attempt": 2, "event": "revised_plan_validated", "plan": {"preferences": {"preferred_detailed_mood": {"source": "inferred", "value": "cozy"}}}}
{"attempt": 2, "event": "validation_result", "outcome": "passed", "checks": [{"name": "duration", "status": "passed"}]}
{"event": "workflow_finished", "final_status": "succeeded", "attempts_used": 2, "result_attempt": 2}
```

## Mock versus live operation

- `python -m evaluation.evaluate --mode mock` uses only fixed JSON responses.
  It is reproducible and is the mode used for the checked-in evaluation
  results.
- `python -m evaluation.evaluate --mode live` sends scenario requests to the
  configured live model. It requires `GROQ_API_KEY`, can incur provider costs,
  and is not deterministic. It does not use expected mock statuses to claim
  accuracy.

Live requests may contain personal information in their request text; saved
preference values stay in Python and are not sent to the model. Do not send
sensitive information; review provider terms and configure local profile/log
retention appropriately.

## Unsupported interactions

The model cannot add missing metadata such as language or duration. If a
request needs unsupported fields, the workflow reports the limitation. No
external search, arbitrary tool execution, or automatic profile modification
by the model is implemented.

## Real catalog, previews and RAG (privacy notes)

- **Catalog build** sends only fixed seed phrases (e.g. "late night jazz") and track
  IDs to the public Deezer API. Preview audio is downloaded once to measure tempo and
  energy and then discarded; only numbers are stored in `data/real_songs.csv`.
- **Previews in the web page**: the server asks Deezer for a fresh preview URL by track
  ID (URLs expire after ~15 minutes); the listener's browser then streams the clip from
  Deezer. Request text is never sent to Deezer.
- **RAG** embeds the request locally with FastEmbed. The planner receives at most eight
  retrieved songs' genre/mood labels and playlist tags as `catalog_matches`; the system
  prompt tells it to treat them as hints, never as user evidence or instructions. Logs
  record only counts and the top similarity score, not titles or request text.
