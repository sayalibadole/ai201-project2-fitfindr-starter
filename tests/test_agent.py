# tests/test_agent.py
"""
Tests for the agent layer: the query parser (_parse_query) and the planning
loop (run_agent).

run_agent calls the LLM tools, so those tests patch suggest_outfit /
create_fit_card on the `agent` module (agent imports them by name, so the
bound references live there, not on `tools`).
"""

import pytest

import agent
from agent import _parse_query, run_agent


# ── _parse_query ─────────────────────────────────────────────────────────────────

def test_parse_basic_query():
    parsed = _parse_query("vintage graphic tee under $30, size M")
    assert parsed == {
        "description": "vintage graphic tee",
        "size": "M",
        "max_price": 30.0,
    }


def test_parse_leadin_phrase_stripped():
    parsed = _parse_query("looking for a vintage graphic tee under $30")
    assert parsed["description"] == "vintage graphic tee"
    assert parsed["size"] is None
    assert parsed["max_price"] == 30.0


def test_parse_conversational_query_no_filler_leak():
    # Regression: trailing wardrobe context and chit-chat must NOT leak into
    # the search keywords. Only the first-sentence item request survives.
    query = (
        "I'm looking for a vintage graphic tee under $30. I mostly wear baggy "
        "jeans and chunky sneakers. What's out there and how would I style it?"
    )
    parsed = _parse_query(query)
    assert parsed["description"] == "vintage graphic tee"
    assert parsed["max_price"] == 30.0
    # None of the second/third-sentence words should appear.
    for leaked in ("baggy", "jeans", "sneakers", "style", "what", "how", "wear"):
        assert leaked not in parsed["description"].split()


def test_parse_price_synonyms():
    assert _parse_query("oversized flannel less than 25")["max_price"] == 25.0
    assert _parse_query("denim jacket below 40")["max_price"] == 40.0


def test_parse_size_from_explicit_and_standalone():
    assert _parse_query("track jacket in size M")["size"] == "M"
    assert _parse_query("track jacket XL")["size"] == "XL"


def test_parse_no_constraints():
    parsed = _parse_query("black denim jacket")
    assert parsed["description"] == "black denim jacket"
    assert parsed["size"] is None
    assert parsed["max_price"] is None


# ── run_agent (planning loop) ────────────────────────────────────────────────────

@pytest.fixture
def stub_llm_tools(monkeypatch):
    """Replace the two LLM tools with deterministic stubs and record their args."""
    calls = {}

    def fake_suggest(new_item, wardrobe):
        calls["suggest_item"] = new_item
        return "STUB OUTFIT"

    def fake_fitcard(outfit, new_item):
        calls["fitcard_outfit"] = outfit
        calls["fitcard_item"] = new_item
        return "STUB FIT CARD"

    monkeypatch.setattr(agent, "suggest_outfit", fake_suggest)
    monkeypatch.setattr(agent, "create_fit_card", fake_fitcard)
    return calls


def test_run_agent_happy_path_state_flow(stub_llm_tools):
    wardrobe = {"items": []}
    session = run_agent("vintage graphic tee under $30", wardrobe)

    assert session["error"] is None
    assert isinstance(session["selected_item"], dict)
    # selected_item is the top search result and the SAME object handed to both tools.
    assert session["selected_item"] is session["search_results"][0]
    assert stub_llm_tools["suggest_item"] is session["selected_item"]
    assert stub_llm_tools["fitcard_item"] is session["selected_item"]
    # fit card is built from the outfit suggestion, not regenerated.
    assert session["outfit_suggestion"] == "STUB OUTFIT"
    assert stub_llm_tools["fitcard_outfit"] == "STUB OUTFIT"
    assert session["fit_card"] == "STUB FIT CARD"


def test_run_agent_no_results_early_exit(stub_llm_tools):
    # Impossible constraints → error set, downstream tools never called.
    session = run_agent("designer ballgown size XXS under $5", {"items": []})
    assert session["error"] is not None
    assert session["selected_item"] is None
    assert session["outfit_suggestion"] is None
    assert session["fit_card"] is None
    assert "suggest_item" not in stub_llm_tools  # suggest_outfit was NOT called


def test_run_agent_relaxes_size_filter(stub_llm_tools):
    # A size with no match should trigger the relaxed-filter retry and a note.
    session = run_agent("graphic tee size XXXL under $30", {"items": []})
    assert session["error"] is None
    assert session["selected_item"] is not None
    assert any("relaxed" in note.lower() for note in session["notes"])
