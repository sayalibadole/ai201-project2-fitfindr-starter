# tests/test_tools.py
"""
Tests for the three FitFindr tools.

search_listings is deterministic and hits the local dataset directly.
suggest_outfit and create_fit_card call the LLM, so their tests mock the
Groq client — this keeps the suite fast, offline, and free of API-key
requirements while still exercising every branch (empty wardrobe, empty
outfit guard, and the template fallback).
"""

import pytest

import tools
from tools import create_fit_card, search_listings, suggest_outfit


# ── Fake Groq client ────────────────────────────────────────────────────────────

class _FakeChoice:
    def __init__(self, content):
        self.message = type("Msg", (), {"content": content})()


class _FakeClient:
    """Stand-in for a Groq client whose chat.completions.create echoes content."""

    def __init__(self, content="A great little outfit idea."):
        self._content = content
        self.chat = type(
            "Chat", (), {"completions": self}
        )()

    def create(self, **kwargs):
        # Record the last call so tests can assert on the prompt if needed.
        self.last_kwargs = kwargs
        return type("Resp", (), {"choices": [_FakeChoice(self._content)]})()


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch _get_groq_client so LLM tools return a canned response."""
    client = _FakeClient()
    monkeypatch.setattr(tools, "_get_groq_client", lambda: client)
    return client


@pytest.fixture
def sample_item():
    return {
        "id": "lst_999",
        "title": "Test Denim Jacket",
        "description": "A cropped light-wash denim jacket.",
        "category": "outerwear",
        "style_tags": ["vintage", "denim"],
        "size": "M",
        "condition": "good",
        "price": 38.00,
        "colors": ["blue"],
        "brand": "Levi's",
        "platform": "depop",
    }


# ── Tool 1: search_listings ─────────────────────────────────────────────────────

def test_search_returns_results():
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    assert isinstance(results, list)
    assert len(results) > 0


def test_search_empty_results():
    results = search_listings("designer ballgown", size="XXS", max_price=5)
    assert results == []   # empty list, no exception


def test_search_price_filter():
    results = search_listings("jacket", size=None, max_price=10)
    assert all(item["price"] <= 10 for item in results)


def test_search_size_filter_case_insensitive():
    # "m" should match sizes like "S/M" or "M" regardless of case.
    results = search_listings("tee", size="m", max_price=None)
    assert all("m" in item["size"].lower() for item in results)


def test_search_sorted_by_relevance():
    # More overlapping keywords → higher score → earlier in the list.
    results = search_listings("vintage denim jacket", size=None, max_price=None)
    assert len(results) > 1
    # Scores are non-increasing: recompute the simple overlap score and check order.
    def score(item):
        hay = " ".join(
            [item["title"], item["description"], item["category"],
             " ".join(item["style_tags"]), " ".join(item["colors"]),
             item.get("brand") or ""]
        ).lower()
        return sum(w in hay for w in "vintage denim jacket".split())
    scores = [score(item) for item in results]
    assert scores == sorted(scores, reverse=True)


# ── Tool 2: suggest_outfit ───────────────────────────────────────────────────────

def test_suggest_outfit_with_wardrobe(mock_llm, sample_item):
    wardrobe = {"items": [{"name": "Black jeans", "category": "bottoms"}]}
    result = suggest_outfit(sample_item, wardrobe)
    assert isinstance(result, str)
    assert result.strip() != ""


def test_suggest_outfit_empty_wardrobe(mock_llm, sample_item):
    # Failure mode: no wardrobe items → still returns a non-empty string.
    result = suggest_outfit(sample_item, {"items": []})
    assert isinstance(result, str)
    assert result.strip() != ""


def test_suggest_outfit_missing_items_key(mock_llm, sample_item):
    # Failure mode: malformed wardrobe dict must not crash.
    result = suggest_outfit(sample_item, {})
    assert isinstance(result, str)
    assert result.strip() != ""


def test_suggest_outfit_llm_failure_returns_string(monkeypatch, sample_item):
    # Failure mode: LLM/client error must be caught and returned as a string.
    def boom():
        raise RuntimeError("api down")
    monkeypatch.setattr(tools, "_get_groq_client", boom)
    result = suggest_outfit(sample_item, {"items": []})
    assert isinstance(result, str)
    assert result.strip() != ""


# ── Tool 3: create_fit_card ──────────────────────────────────────────────────────

def test_create_fit_card_returns_string(mock_llm, sample_item):
    result = create_fit_card("Jacket + black jeans + white sneakers.", sample_item)
    assert isinstance(result, str)
    assert result.strip() != ""


def test_create_fit_card_empty_outfit(sample_item):
    # Failure mode: empty outfit → descriptive error string, no exception, no LLM call.
    result = create_fit_card("", sample_item)
    assert isinstance(result, str)
    assert result.strip() != ""


def test_create_fit_card_whitespace_outfit(sample_item):
    # Failure mode: whitespace-only outfit is treated as empty.
    result = create_fit_card("   \n  ", sample_item)
    assert isinstance(result, str)
    assert result.strip() != ""


def test_create_fit_card_fallback_on_llm_failure(monkeypatch, sample_item):
    # Failure mode: LLM error → template caption built from the item details.
    def boom():
        raise RuntimeError("api down")
    monkeypatch.setattr(tools, "_get_groq_client", boom)
    result = create_fit_card("Jacket + black jeans.", sample_item)
    assert isinstance(result, str)
    assert sample_item["title"] in result        # template uses the item title
    assert "depop" in result                       # ...and the platform
