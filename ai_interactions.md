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
