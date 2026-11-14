import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _app():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    return at


def test_offline_scenario_shows_playlist_and_agent_loop(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = _app()
    at.selectbox[0].set_value("successful-plan-revision").run()
    at.button[0].click().run()

    assert not at.exception
    assert "succeeded" in at.success[0].value
    assert [m.value for m in at.metric] == ["5", "2", "246/250"]
    assert any("Attempt 2" in m.value and "passed" in m.value for m in at.markdown)


def test_workflow_error_is_shown_not_raised():
    at = _app()
    at.selectbox[0].set_value("invalid-llm-output").run()
    at.button[0].click().run()

    assert not at.exception
    assert "Could not create recommendations" in at.error[0].value


def test_result_is_cleared_when_scenario_changes():
    at = _app()
    at.button[0].click().run()
    assert at.metric
    at.selectbox[0].set_value("ambient-wind-down").run()
    assert not at.metric


def test_live_mode_needs_a_request(monkeypatch):
    at = _app()
    at.sidebar.radio[0].set_value("live").run()
    assert at.button[0].disabled


def test_for_you_cold_start_then_like_saves_song(monkeypatch, tmp_path):
    import src.catalogs as catalogs
    from src.catalogs import Catalog

    monkeypatch.setitem(catalogs.CATALOGS, "demo", Catalog("demo", "data/songs.csv", tmp_path / "p.json"))
    monkeypatch.setitem(catalogs.CATALOGS, "real", Catalog("real", "data/missing.csv", tmp_path / "r.json"))
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    at.sidebar.radio[0].set_value("foryou").run()
    assert not at.exception
    assert any("popular songs" in i.value for i in at.info)
    like = next(b for b in at.button if b.label == "👍 Like")
    like.click().run()
    assert not at.exception
    assert any("Liked songs (1)" in t.label for t in at.tabs)
