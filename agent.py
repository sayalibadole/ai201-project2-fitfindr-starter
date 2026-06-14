"""
agent.py

The FitFindr planning loop. Orchestrates the three tools in response to a
natural language user query, passing state between them via a session dict.

Complete tools.py and test each tool in isolation before implementing this file.

Usage (once implemented):
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import re

from tools import search_listings, suggest_outfit, create_fit_card


# ── query parsing ───────────────────────────────────────────────────────────────

# Known clothing sizes, longest-first so "XXL" matches before "XL"/"L".
_SIZES = ["XXXL", "XXL", "XXS", "XS", "XL", "S", "M", "L"]

# Filler words that aren't useful as search keywords. Covers query connectives
# plus conversational filler ("i'm", "what's out there", "how would i style it").
_STOPWORDS = {
    "looking", "for", "a", "an", "the", "i", "im", "want", "wanna", "need",
    "under", "below", "less", "than", "max", "size", "please", "find", "me",
    "some", "that", "is", "in", "of", "to", "my", "with", "and", "or",
    "dollars", "it", "its", "what", "whats", "out", "there", "how", "would",
    "show", "get", "really", "just", "something", "mostly", "wear", "wearing",
    "but", "so", "like", "hey", "hi", "thanks", "around", "about",
}

# Phrases that introduce the actual item request — we keep only what follows
# the last one in the first sentence.
_LEADIN_RE = re.compile(
    r"\b(?:looking for|searching for|search for|wanna|want|need|find|"
    r"show me|get me|after)\b",
    flags=re.IGNORECASE,
)


def _parse_query(query: str) -> dict:
    """
    Extract a search description, size, and max_price from a free-text query.

    Uses regex/string parsing (no LLM call) — it's fast, deterministic, and
    cheap to unit-test. Returns a dict with keys: description, size, max_price.

    Description extraction is scoped to the FIRST sentence and to whatever
    follows a lead-in phrase ("looking for ...") so that trailing wardrobe
    context or chit-chat ("I mostly wear baggy jeans. What's out there?")
    doesn't leak into the search keywords. Size and price are still scanned
    across the whole query, since they can appear in any sentence.
    """
    text = query.strip()

    # 1. max_price: "under $30", "below 30", "$30", "less than 25.50".
    max_price = None
    price_match = re.search(
        r"(?:under|below|less than|max|<|\$)\s*\$?\s*(\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if price_match:
        max_price = float(price_match.group(1))
        text = text[: price_match.start()] + text[price_match.end():]

    # 2. size: prefer an explicit "size M"; else a standalone size token.
    size = None
    size_match = re.search(r"\bsize\s+([A-Za-z0-9/]+)", text, flags=re.IGNORECASE)
    if size_match:
        size = size_match.group(1).upper()
        text = text[: size_match.start()] + text[size_match.end():]
    else:
        for candidate in _SIZES:
            # Standalone, case-sensitive uppercase token (avoids matching the
            # "s" in "shoes"); word boundaries keep it from matching substrings.
            if re.search(rf"\b{candidate}\b", text):
                size = candidate
                text = re.sub(rf"\b{candidate}\b", "", text, count=1)
                break

    # 3. description: scope to the first sentence, then to text after the last
    #    lead-in phrase, then drop stopwords / 1-char tokens.
    first_sentence = re.split(r"[.!?]", text, maxsplit=1)[0]
    leadins = list(_LEADIN_RE.finditer(first_sentence))
    item_text = first_sentence[leadins[-1].end():] if leadins else first_sentence

    words = re.findall(r"[A-Za-z0-9']+", item_text.lower())
    keywords = [w for w in words if len(w) > 1 and w not in _STOPWORDS]
    description = " ".join(keywords)

    return {"description": description, "size": size, "max_price": max_price}


# ── session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialize and return a fresh session dict for one user interaction.

    The session dict is the single source of truth for everything that happens
    during a run — it stores the original query, parsed parameters, tool results,
    and any error that caused early termination.

    You may add fields to this dict as needed for your implementation.
    """
    return {
        "query": query,              # original user query
        "parsed": {},                # extracted description / size / max_price
        "search_results": [],        # list of matching listing dicts
        "selected_item": None,       # top result, passed into suggest_outfit
        "wardrobe": wardrobe,        # user's wardrobe dict
        "outfit_suggestion": None,   # string returned by suggest_outfit
        "fit_card": None,            # string returned by create_fit_card
        "notes": [],                 # user-facing notes (e.g. relaxed filters)
        "error": None,               # set if the interaction ended early
    }


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Main agent entry point. Runs the FitFindr planning loop for a single
    user interaction and returns the completed session dict.

    Args:
        query:    Natural language user request
                  (e.g., "vintage graphic tee under $30, size M")
        wardrobe: User's wardrobe dict — use get_example_wardrobe() or
                  get_empty_wardrobe() from utils/data_loader.py

    Returns:
        The session dict after the interaction completes. Check session["error"]
        first — if it is not None, the interaction ended early and the other
        output fields (outfit_suggestion, fit_card) will be None.

    TODO — implement this function using the planning loop you designed in planning.md:

        Step 1: Initialize the session with _new_session().

        Step 2: Parse the user's query to extract a description, size, and
                max_price. You can use regex, string splitting, or ask the LLM
                to parse it — document your choice in planning.md.
                Store the result in session["parsed"].

        Step 3: Call search_listings() with the parsed parameters.
                Store results in session["search_results"].
                If no results: set session["error"] to a helpful message and
                return the session early. Do NOT proceed to suggest_outfit
                with empty input.

        Step 4: Select the item to use (e.g., the top result).
                Store it in session["selected_item"].

        Step 5: Call suggest_outfit() with the selected item and wardrobe.
                Store the result in session["outfit_suggestion"].

        Step 6: Call create_fit_card() with the outfit suggestion and selected item.
                Store the result in session["fit_card"].

        Step 7: Return the session.

    Before writing code, complete the Planning Loop and State Management sections
    of planning.md — your implementation should match what you described there.
    """
    # Step 1: initialize the session.
    session = _new_session(query, wardrobe)

    # Step 2: parse the query into description / size / max_price.
    parsed = _parse_query(query)
    
    session["parsed"] = parsed

    # Step 3: search the listings.
    results = search_listings(
        description=parsed["description"],
        size=parsed["size"],
        max_price=parsed["max_price"],
    )

    # Retry once with relaxed constraints (drop the size filter) before giving
    # up, per the Planning Loop / Error Handling sections of planning.md.
    if not results and parsed["size"] is not None:
        relaxed = search_listings(
            description=parsed["description"],
            size=None,
            max_price=parsed["max_price"],
        )
        if relaxed:
            session["notes"].append(
                f"No exact match for size {parsed['size']} — relaxed the size "
                "filter to show the closest items."
            )
            results = relaxed

    session["search_results"] = results

    if not results:
        constraints = []
        if parsed["size"]:
            constraints.append(f"size {parsed['size']}")
        if parsed["max_price"] is not None:
            constraints.append(f"under ${parsed['max_price']:.0f}")
        suffix = f" ({', '.join(constraints)})" if constraints else ""
        session["error"] = (
            f"No listings matched '{parsed['description'] or query}'{suffix}. "
            "Try broadening your search — relax the size or budget, or use "
            "different keywords."
        )
        return session

    # Step 4: select the top (most relevant) result.
    session["selected_item"] = results[0]

    # Step 5: suggest an outfit using the selected item and the wardrobe.
    session["outfit_suggestion"] = suggest_outfit(session["selected_item"], wardrobe)

    # Step 6: generate a shareable fit card.
    session["fit_card"] = create_fit_card(
        session["outfit_suggestion"], session["selected_item"]
    )

    # Step 7: return the completed session.
    return session


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    def _show(session: dict) -> None:
        for note in session["notes"]:
            print(f"Note: {note}")
        if session["error"]:
            print(f"Error: {session['error']}")
            return
        item = session["selected_item"]
        print(f"Found: {item['title']} — ${item['price']:.0f} on {item['platform']}")
        print(f"\nOutfit: {session['outfit_suggestion']}")
        print(f"\nFit card: {session['fit_card']}")

    # Scenario 1 — happy path: valid search → outfit → fit card.
    print("=== Happy path: graphic tee (example wardrobe) ===\n")
    _show(run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    ))

    # Scenario 2 — no results: early exit, no downstream tool calls.
    print("\n\n=== No-results path ===\n")
    _show(run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    ))

    # Scenario 3 — empty wardrobe: still produces an outfit + fit card.
    print("\n\n=== Empty wardrobe: fallback styling ===\n")
    _show(run_agent(
        query="oversized flannel under $30",
        wardrobe=get_empty_wardrobe(),
    ))
