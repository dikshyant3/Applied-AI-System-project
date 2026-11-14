"""TuneWise web interface.

    streamlit run app.py

Modes:
* Live request   - the full agent workflow (plan -> filter/retrieve/rank -> build ->
                   validate -> revise) with the LLM (needs GROQ_API_KEY), on the real
                   Deezer catalog (data/real_songs.csv) or the synthetic demo catalog.
* For You        - no prompt: songs recommended from the saved profile (likes,
                   dislikes, saved preferences) plus similarity to liked songs.
                   👍 saves a song to "Liked songs"; 👎 steers away from it.
* Vibe search    - retrieval only (no LLM, no key): the request is embedded and the
                   closest real songs are shown.
* Offline demo   - replays a scripted planner from evaluation/scenarios.json on the
                   synthetic demo catalog (no API key).

Songs from the real catalog show their album cover; pressing play fetches a fresh
30-second Deezer preview (preview links expire, so they are never stored).
"""

import os

import pandas as pd
import streamlit as st

from evaluation.evaluate import _mock_responses
from src.agent import MockLLMClient, load_env_file
from src.catalogs import CATALOGS
from src.main import _describe_value, _mock_scenarios, _provenance_detail
from src.orchestrator import WorkflowError, run_recommendation_workflow
from src.recommender import load_songs
from src.user_profile import UserProfileStore
from src.workflow_logging import configure_workflow_logging

DEMO_CATALOG = CATALOGS["demo"].csv_path
REAL_CATALOG = CATALOGS["real"].csv_path
HAS_REAL_CATALOG = CATALOGS["real"].exists
PROFILE_PATHS = {c.csv_path: c.profile_path for c in CATALOGS.values()}
STATUS_STYLE = {"succeeded": st.success, "passed": st.success}
CHECK_ICON = {"passed": "✅", "violated": "❌", "skipped": "⏭️", "unsupported": "⚠️"}
MODES = {
    "offline": "Offline demo (no API key)",
    "foryou": "For You (no prompt)",
    "live": "Live LLM request",
    "search": "Vibe search (RAG only, no key)",
}

st.set_page_config(page_title="TuneWise", page_icon="🎧", layout="centered")
st.markdown(
    "<style>div.block-container{padding-top:2rem;max-width:960px}"
    ".tw-title{font-weight:600;line-height:1.25;margin-top:.35rem}"
    ".tw-meta{color:#888;font-size:.85rem;line-height:1.3}</style>",
    unsafe_allow_html=True,
)


@st.cache_data
def catalog(path):
    return load_songs(path)


@st.cache_resource(show_spinner="Loading the song embeddings…")
def retriever(path):
    from src.retrieval import build_retriever

    return build_retriever(catalog(path))


@st.cache_resource
def deezer():
    from src.providers.deezer import DeezerClient

    return DeezerClient()


@st.cache_data
def scenarios():
    return _mock_scenarios()


@st.cache_resource
def _logging_configured() -> bool:
    configure_workflow_logging()  # JSON workflow events go to the terminal (stderr)
    return True


def run(request, mode, scenario_id=None, catalog_path=DEMO_CATALOG, use_rag=False):
    if mode == "offline":
        scenario = scenarios()[scenario_id]
        return run_recommendation_workflow(
            scenario["request"], saved_preferences=scenario.get("saved_preferences"),
            client=MockLLMClient(_mock_responses(scenario)), songs=catalog(DEMO_CATALOG),
            max_attempts=scenario.get("max_attempts", 3), request_id=f"web-mock-{scenario_id}",
        )
    songs = catalog(catalog_path)
    profile = UserProfileStore(PROFILE_PATHS[catalog_path], songs)
    return run_recommendation_workflow(
        request, saved_preferences=profile.preferences, songs=songs,
        retriever=retriever(catalog_path) if use_rag else None,
    )


# ---- song cards -------------------------------------------------------------


def cover(record):
    """Album cover drawn over a 🎵 placeholder: if the image fails to load, the placeholder shows."""
    url = str(record.get("cover_url") or "").replace("'", "%27").replace('"', "%22")
    image = f"background-image:url('{url}');" if url.startswith("https://") else ""
    label = "" if url else "<div style='font-size:.7rem;margin-top:.3rem'>demo song · no cover</div>"
    st.markdown(
        f"<div style=\"aspect-ratio:1;border-radius:8px;background-color:#8882;{image}"
        "background-size:cover;background-position:center;display:flex;flex-direction:column;"
        "align-items:center;justify-content:center;font-size:2.5rem;color:#888\">"
        f"{'' if url else '🎵'}{label}</div>", unsafe_allow_html=True)


def song_card(song, key, caption=None, rate_path=None):
    """Cover, labels, a play button for the 30-second preview and (with rate_path) 👍 / 👎."""
    record = song if isinstance(song, dict) else dict(song.record)
    deezer_id = str(record.get("deezer_id") or "").strip()
    cover(record)
    st.markdown(f"<div class='tw-title'>{_escape(record['title'])}</div>"
                f"<div class='tw-meta'>{_escape(record['artist'])}<br>{_escape(record['genre'])} · "
                f"{_escape(record['mood'])} · energy {float(record['energy']):.2f}</div>",
                unsafe_allow_html=True)
    if caption:
        st.caption(caption)
    if rate_path is not None:
        rating_buttons(record, key, rate_path)
    if not deezer_id:
        st.caption("No preview (synthetic demo song)")
        return
    if st.session_state.get("playing") == key:
        url = _preview(deezer_id)
        if url:
            st.audio(url, format="audio/mpeg", autoplay=True)
        else:
            st.caption("Preview unavailable right now.")
        if record.get("deezer_link"):
            st.markdown(f"[Open on Deezer ↗]({record['deezer_link']})")
    elif st.button("▶ Play preview", key=f"play-{key}", width="stretch"):
        st.session_state["playing"] = key
        st.rerun()


def profile_store(path):
    return UserProfileStore(PROFILE_PATHS[path], catalog(path))


def rating_buttons(record, key, path):
    """👍 saves the song to Liked songs and teaches the profile; 👎 steers away from it."""
    from src.feedback import clear_feedback, latest_reactions, submit_feedback

    store = profile_store(path)
    current = latest_reactions(store).get(record["id"])
    like_col, dislike_col = st.columns(2)
    liked, disliked = current == "like", current == "dislike"
    if like_col.button("💚 Liked" if liked else "👍 Like", key=f"like-{key}", width="stretch",
                       type="primary" if liked else "secondary"):
        clear_feedback(store, record["id"]) if liked else submit_feedback(store, record["id"], "like")
        st.toast("Removed from Liked songs" if liked else f"Saved “{record['title']}” to Liked songs")
        st.rerun()
    if dislike_col.button("👎 Disliked" if disliked else "👎", key=f"dislike-{key}", width="stretch",
                          type="primary" if disliked else "secondary"):
        clear_feedback(store, record["id"]) if disliked else submit_feedback(store, record["id"], "dislike")
        st.toast("Dislike removed" if disliked else "Got it — fewer songs like this")
        st.rerun()


def _preview(deezer_id):
    from src.providers.deezer import DeezerError

    try:
        return deezer().preview_url(int(deezer_id))
    except (DeezerError, ValueError):
        return None


def _escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def song_grid(items, prefix, columns=3, rate_path=None):
    """items: (song, caption) pairs."""
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, (song, caption) in zip(cols, items[start:start + columns]):
            song_id = song["id"] if isinstance(song, dict) else song.id
            with col:  # keyed by song id so a card keeps its state when the grid reflows
                song_card(song, f"{prefix}-{song_id}", caption, rate_path)


# ---- workflow result ----------------------------------------------------------


def show_attempts(trace):
    if trace is None:
        return
    st.markdown(f"**{trace.final_status.replace('_', ' ').title()}** after "
                f"{len(trace.attempts)} of {trace.max_attempts} attempt(s): {trace.stop_reason}")
    for attempt in trace.attempts:
        icon = "✅" if attempt.outcome == "passed" else "🔁"
        lines = [f"{icon} **Attempt {attempt.number}** ({attempt.kind}): `{attempt.outcome}`"]
        if attempt.changed_fields:
            lines.append(f"revised: {', '.join(attempt.changed_fields)}")
        if attempt.observation is not None:
            lines.append(f"reasons: {', '.join(attempt.observation.codes)}")
        st.markdown(" · ".join(lines))
        if attempt.rationale:
            st.caption(f"Planner rationale: {attempt.rationale}")


def show_result(result, rate_path=None):
    status = result.final_status
    STATUS_STYLE.get(status, st.warning)(
        f"Result: **{status.replace('_', ' ')}** — playlist validation {result.validation.outcome}"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Songs", len(result.songs))
    c2.metric("Attempts", result.attempts_used)
    c3.metric("Eligible songs", f"{result.candidate_count}/{result.catalog_size}")

    names = ["Playlist", "Agent loop", "Plan", "Validation"] + (["Retrieval"] if result.retrieval else [])
    tabs = st.tabs(names)
    with tabs[0]:
        if not result.songs:
            st.info("No songs could be selected.")
        else:
            st.caption(f"Scoring mode: {result.strategy.name}")
            song_grid([(song, f"#{song.position} · score {song.score:.2f}") for song in result.songs],
                      prefix="playlist", rate_path=rate_path)
            st.dataframe(pd.DataFrame([{
                "#": song.position, "Title": song.title, "Artist": song.artist,
                "Genre": song.genre, "Mood": song.mood, "Energy": round(song.energy, 2),
                "Score": round(song.score, 2),
            } for song in result.songs]), hide_index=True, width="stretch")
            for song in result.songs:
                with st.expander(f"Why “{song.title}”?"):
                    st.markdown("\n".join(f"- {reason}" for reason in song.explanations))
    with tabs[1]:
        show_attempts(result.trace)
    with tabs[2]:
        st.dataframe(pd.DataFrame([{
            "Field": entry.field, "Value": _describe_value(entry.value),
            "Source": entry.source, "Evidence / note": _provenance_detail(entry),
        } for entry in result.planning.provenance if entry.source != "default"]),
            hide_index=True, width="stretch")
        for warning in result.planning.warnings:
            st.warning(warning)
    with tabs[3]:
        for check in result.validation.checks:
            note = " _(infeasible with this catalog)_" if check.infeasible else ""
            st.markdown(f"{CHECK_ICON.get(check.status, '•')} **{check.name}** — {check.message}{note}")
    if result.retrieval:
        with tabs[4]:
            info = result.retrieval
            st.markdown(f"Embedder `{info['embedder']}` · ranked pool **{info['pool']}** songs"
                        + (" · _fell back to all eligible songs_" if info["fallback_to_all_eligible"] else ""))
            st.caption("Catalog matches given to the planner as hints (not as user evidence):")
            st.dataframe(pd.DataFrame(info["catalog_matches"]), hide_index=True, width="stretch")


# ---- For You ------------------------------------------------------------------


def show_for_you(path, use_similarity):
    from src.feedback import latest_reactions
    from src.personalized import recommend_for_profile

    songs = catalog(path)
    store = profile_store(path)
    reactions = latest_reactions(store)
    by_id = {song["id"]: song for song in songs}
    liked = [by_id[i] for i, action in reactions.items() if action == "like"]
    for_you, library = st.tabs(["For you", f"Liked songs ({len(liked)})"])
    with for_you:
        picks = recommend_for_profile(
            songs, store.preferences, reactions, k=12,
            retriever=retriever(path) if use_similarity and liked else None)
        if not picks:
            st.info("You have rated every song in this catalog.")
        elif picks[0].kind == "starter":
            st.info("Like or dislike a few songs to personalize this page. "
                    "Until then, here are popular songs from different genres.")
        else:
            prefs = store.preferences
            st.caption(f"Based on {len(liked)} liked song(s) and your saved taste: "
                       f"{prefs['favorite_genre']} · {prefs['favorite_mood']} · "
                       f"energy {float(prefs['target_energy']):.2f}")
        song_grid([(dict(p.song), p.reasons[0]) for p in picks], prefix="foryou", rate_path=path)
    with library:
        if not liked:
            st.caption("Songs you 👍 are saved here.")
        song_grid([(song, None) for song in reversed(liked)], prefix="liked", rate_path=path)


# ---- page -------------------------------------------------------------------

_logging_configured()
load_env_file()
has_key = bool(os.environ.get("GROQ_API_KEY"))

st.title("🎧 TuneWise")
st.caption("Describe the playlist you want. An LLM plans it, Python builds it from the "
           "catalog and checks it, and the planner revises the plan if a check fails.")

with st.sidebar:
    st.header("Planner")
    mode = st.radio("Mode", list(MODES), format_func=MODES.get)
    catalog_path, use_rag = DEMO_CATALOG, False
    if mode == "live":
        if not has_key:
            st.error("GROQ_API_KEY is not set. Add it to `.env` or use another mode.")
        options = [REAL_CATALOG, DEMO_CATALOG] if HAS_REAL_CATALOG else [DEMO_CATALOG]
        catalog_path = st.radio(
            "Catalog", options,
            format_func=lambda p: "Real songs (Deezer)" if p == REAL_CATALOG else "Synthetic demo")
        use_rag = st.toggle("Retrieval (RAG)", value=catalog_path == REAL_CATALOG,
                            help="Embed the request, give the planner the closest catalog songs as "
                                 "hints, and rank only the most similar eligible songs.")
    elif mode == "foryou":
        options = [REAL_CATALOG, DEMO_CATALOG] if HAS_REAL_CATALOG else [DEMO_CATALOG]
        catalog_path = st.radio(
            "Catalog", options,
            format_func=lambda p: "Real songs (Deezer)" if p == REAL_CATALOG else "Synthetic demo")
        use_rag = st.toggle("Similar to liked songs (embeddings)", value=True,
                            help="Rank songs closest to what you liked, away from what you disliked.")
    elif mode == "search":
        catalog_path = REAL_CATALOG if HAS_REAL_CATALOG else DEMO_CATALOG
    st.divider()
    st.caption(f"Catalog: {len(catalog(catalog_path))} songs from `{catalog_path}`")
    if mode in ("live", "foryou", "search"):
        st.caption(f"Saved profile: `{PROFILE_PATHS[catalog_path]}`")
    if catalog_path == REAL_CATALOG:
        st.caption("Previews and artwork: Deezer. Tempo and energy are measured from the previews; "
                   "other audio features are estimates.")

scenario_id = request = None
if mode == "offline":
    options = scenarios()
    scenario_id = st.selectbox(
        "Scenario", list(options),
        format_func=lambda sid: f"{options[sid]['request']}  ·  {sid}",
    )
    st.caption(f"Expected outcome: `{options[scenario_id]['expected_status']}` — the planner's "
               "reply is scripted; ranking, building, validation and retries are real.")
elif mode != "foryou":
    request = st.text_area(
        "Your request", height=90,
        placeholder="e.g. Five calm lofi songs for studying, about 20 minutes"
        if mode == "live" else "e.g. songs for a rainy Sunday afternoon with a book")

inputs = (mode, scenario_id, request, catalog_path, use_rag)
if st.session_state.get("inputs") != inputs:
    for stale in ("result", "error", "hits", "playing"):  # inputs changed: no stale result
        st.session_state.pop(stale, None)

if mode == "foryou":
    show_for_you(catalog_path, use_rag)
elif mode == "search":
    if st.button("Search", type="primary", disabled=not (request or "").strip()):
        st.session_state["hits"] = retriever(catalog_path).search(request, k=12)
        st.session_state["inputs"] = inputs
    if "hits" in st.session_state:
        st.caption("Closest songs by meaning (retrieval only — no planning, constraints or validation).")
        song_grid([(dict(hit.song), f"similarity {hit.score:.2f}") for hit in st.session_state["hits"]],
                  prefix="search", rate_path=catalog_path)
else:
    disabled = mode == "live" and (not has_key or not (request or "").strip())
    if st.button("Recommend", type="primary", disabled=disabled):
        with st.spinner("Planning, building and validating…"):
            try:
                st.session_state["result"] = run(request, mode, scenario_id, catalog_path, use_rag)
                st.session_state.pop("error", None)
            except (WorkflowError, ValueError) as error:
                st.session_state.pop("result", None)
                st.session_state["error"] = error
            st.session_state["inputs"] = inputs
            st.session_state.pop("playing", None)

    if "error" in st.session_state:
        error = st.session_state["error"]
        st.error(f"Could not create recommendations: {error}")
        show_attempts(getattr(error, "trace", None))
    elif "result" in st.session_state:
        show_result(st.session_state["result"], rate_path=catalog_path if mode == "live" else None)
