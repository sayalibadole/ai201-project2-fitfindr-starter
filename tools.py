"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


# ── Formatting helpers ──────────────────────────────────────────────────────────

def _describe_item(item: dict) -> str:
    """Format a listing dict into a readable one-paragraph summary for the LLM."""
    parts = [item.get("title", "Untitled item")]
    if item.get("brand"):
        parts.append(f"by {item['brand']}")
    if item.get("category"):
        parts.append(f"({item['category']})")
    detail = " ".join(parts)

    extras = []
    if item.get("colors"):
        extras.append(f"colors: {', '.join(item['colors'])}")
    if item.get("style_tags"):
        extras.append(f"style: {', '.join(item['style_tags'])}")
    if item.get("description"):
        extras.append(item["description"])
    if extras:
        detail += " — " + "; ".join(extras)
    return detail


def _describe_wardrobe_item(item: dict) -> str:
    """Format a wardrobe item dict into a short, named one-liner for the LLM."""
    name = item.get("name", "Unnamed piece")
    bits = []
    if item.get("colors"):
        bits.append(", ".join(item["colors"]))
    if item.get("style_tags"):
        bits.append(", ".join(item["style_tags"]))
    if item.get("notes"):
        bits.append(item["notes"])
    return f"{name} ({'; '.join(bits)})" if bits else name


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
    """
    listings = load_listings()

    # Keyword tokens from the description (lowercased, dedup-friendly).
    keywords = [word for word in description.lower().split() if word]

    scored: list[tuple[int, dict]] = []
    for listing in listings:
        # 2. Filter by price ceiling and size, if provided.
        if max_price is not None and listing["price"] > max_price:
            continue
        if size is not None and size.lower() not in listing["size"].lower():
            continue

        # 3. Score by keyword overlap against the listing's searchable text.
        haystack_parts = [
            listing["title"],
            listing["description"],
            listing["category"],
            " ".join(listing.get("style_tags", [])),
            " ".join(listing.get("colors", [])),
            listing.get("brand") or "",
        ]
        haystack = " ".join(haystack_parts).lower()

        score = sum(1 for word in keywords if word in haystack)

        # 4. Drop listings with no relevant matches.
        if score > 0:
            scored.append((score, listing))

    # 5. Sort by score, highest first, and return just the listing dicts.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [listing for _, listing in scored]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    item_summary = _describe_item(new_item)
    items = (wardrobe or {}).get("items") or []

    if not items:
        # No wardrobe to work with — give general styling advice for the item.
        prompt = (
            f"A user is considering buying this secondhand item:\n{item_summary}\n\n"
            "They haven't told us what's in their wardrobe yet. Suggest how to style "
            "this piece: what kinds of items pair well with it, what vibe/aesthetic it "
            "suits, and 1–2 complete outfit ideas built around it. Keep it concise and "
            "practical."
        )
    else:
        wardrobe_lines = "\n".join(f"- {_describe_wardrobe_item(it)}" for it in items)
        prompt = (
            f"A user is considering buying this secondhand item:\n{item_summary}\n\n"
            f"Here is what they already own:\n{wardrobe_lines}\n\n"
            "Suggest 1–2 complete outfits that combine the new item with specific, "
            "named pieces from their wardrobe. For each outfit, describe the overall "
            "aesthetic and briefly explain why the pieces work together. Keep it "
            "concise and practical."
        )

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sharp, friendly personal stylist who specializes in "
                        "secondhand and vintage fashion."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001 — tool must return a string, never raise
        return f"Sorry, I couldn't generate outfit suggestions right now ({e})."


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    # 1. Guard against an empty or whitespace-only outfit string.
    if not outfit or not outfit.strip():
        return (
            "Couldn't create a fit card — no outfit suggestion was provided. "
            "Generate an outfit with suggest_outfit() first."
        )

    title = new_item.get("title", "this thrifted find")
    price = new_item.get("price")
    platform = new_item.get("platform", "secondhand")
    price_str = f"${price:.0f}" if isinstance(price, (int, float)) else "a steal"

    # 2. Build the prompt.
    prompt = (
        f"Write a short, shareable Instagram/TikTok caption for an OOTD post about "
        f"a thrifted find.\n\n"
        f"Item: {title}\n"
        f"Price: {price_str}\n"
        f"Platform: {platform}\n"
        f"Outfit it's styled in:\n{outfit.strip()}\n\n"
        "Guidelines:\n"
        "- 2–4 sentences, casual and authentic like a real OOTD post (NOT a product "
        "description).\n"
        f"- Mention the item name, the price ({price_str}), and the platform "
        f"({platform}) naturally, once each.\n"
        "- Capture the outfit's vibe in specific terms.\n"
        "- A couple of relevant emojis are welcome.\n"
        "Return only the caption text."
    )

    # 3. Call the LLM (higher temperature so captions vary run to run).
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write fun, authentic social-media captions for thrifted "
                        "outfit posts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
        )
        return response.choices[0].message.content.strip()
    except Exception:  # noqa: BLE001 — fall back to a template caption, never raise
        tags = ", ".join(new_item.get("style_tags", [])) or "thrifted"
        return (
            f"Thrifted this {title} for {price_str} on {platform} 🛍️ "
            f"Styling it up with major {tags} energy. Sustainable never looked so good ✨"
        )
