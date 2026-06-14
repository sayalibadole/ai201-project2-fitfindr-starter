# FitFindr 🛍️

FitFindr is an AI-powered thrifting assistant. Given a natural-language request, it
searches a mock secondhand-listings dataset for matching items, suggests outfits that
pair the find with the user's existing wardrobe, and writes a short, shareable "fit
card" caption. A small planning loop orchestrates three tools and passes state between
them through a single session dictionary.

```
User query ──▶ Planning loop ──▶ search_listings ──▶ suggest_outfit ──▶ create_fit_card ──▶ result
                     │                  │                                                      
                     └── session dict (shared state) ◀──────────────────────────────────────┘
```

---

## Setup

```bash
pip install -r requirements.txt
```

Set your Groq API key in a `.env` file (free key at [console.groq.com](https://console.groq.com)):

```
GROQ_API_KEY=your_key_here
```

`suggest_outfit` and `create_fit_card` call the Groq-hosted `llama-3.3-70b-versatile`
model, so the key is required to run those tools. `search_listings` and the query
parser are fully offline.

## Running

```bash
python agent.py     # CLI: runs three end-to-end scenarios (happy / no-results / empty wardrobe)
python app.py       # Gradio web UI (open the localhost URL it prints)
pytest              # full test suite (22 tests)
```

---

## Tool Inventory

### 1. `search_listings`

| | |
|---|---|
| **Purpose** | Search the mock listings dataset for items matching a description, optional size, and optional price ceiling. Filters then ranks results by relevance so the agent can pick the best match. Fully deterministic; no LLM call. |
| **Inputs** | `description: str` — keywords describing the item (e.g. `"vintage graphic tee"`)<br>`size: str \| None` — size to filter by; case-insensitive substring match (`"M"` matches `"S/M"`). `None` skips size filtering.<br>`max_price: float \| None` — inclusive price ceiling. `None` skips price filtering. |
| **Output** | `list[dict]` — matching listing dicts sorted by relevance (best first). Each dict has `id, title, description, category, style_tags, size, condition, price, colors, brand, platform`. Returns `[]` when nothing matches — never raises. |

### 2. `suggest_outfit`

| | |
|---|---|
| **Purpose** | Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits. Calls the LLM. |
| **Inputs** | `new_item: dict` — a listing dict (the item being considered)<br>`wardrobe: dict` — wardrobe dict with an `"items"` key (a list of wardrobe-item dicts); may be empty |
| **Output** | `str` — a non-empty outfit suggestion. With a populated wardrobe it names specific pieces and explains why they work; with an empty wardrobe it gives general styling advice for the item alone. Never raises (LLM errors are caught and returned as a string). |

### 3. `create_fit_card`

| | |
|---|---|
| **Purpose** | Turn an outfit suggestion into a short, shareable Instagram/TikTok-style caption. Calls the LLM at higher temperature so captions vary run to run. |
| **Inputs** | `outfit: str` — the outfit string from `suggest_outfit`<br>`new_item: dict` — the listing dict for the thrifted item |
| **Output** | `str` — a 2–4 sentence caption mentioning the item name, price, and platform naturally. If `outfit` is empty/whitespace it returns a descriptive error string; if the LLM call fails it returns a template caption built from the item's title, price, platform, and style tags. Never raises. |

> Tools live in [`tools.py`](tools.py). Each is independently runnable and unit-tested in
> [`tests/test_tools.py`](tests/test_tools.py) before being wired into the agent.

---

## Planning Loop

The planning loop lives in `run_agent(query, wardrobe)` in [`agent.py`](agent.py). It is
**conditional, not a blind pipeline** — the no-results branch exits before any LLM tool runs.

1. **Initialize** a fresh session with `_new_session()`.
2. **Parse** the query into `{description, size, max_price}` with `_parse_query()` (regex,
   no LLM call) and store it in `session["parsed"]`.
3. **Search** via `search_listings()`.
   - If the result is empty **and** a size filter was applied, **retry once with the size
     filter dropped** and record a note explaining the relaxation.
   - If still empty, set `session["error"]` to a helpful message and **return early** —
     `suggest_outfit` and `create_fit_card` are never called.
4. **Select** the top (most relevant) result → `session["selected_item"]`.
5. **Suggest an outfit** with the selected item + wardrobe → `session["outfit_suggestion"]`.
6. **Create the fit card** from that outfit string + the selected item → `session["fit_card"]`.
7. **Return** the completed session.

The loop is "done" when a fit card has been produced, or when the search step fails (after
the relaxed-filter retry) and `error` is set.

**Query parsing choice.** Parameters are extracted with regex/string parsing rather than an
LLM call — it's fast, deterministic, and cheap to unit-test. The description is scoped to the
first sentence and to whatever follows a lead-in phrase ("looking for …"), so trailing
wardrobe context or chit-chat ("I mostly wear baggy jeans. What's out there?") does not leak
into the search keywords. Size and price are scanned across the whole query.

---

## State Management

A single **session dictionary** is the source of truth for one interaction and the shared
memory between tools. Each step reads from and writes to it, so later tools see earlier
outputs without the user repeating anything.

| Field | Set by | Holds |
|---|---|---|
| `query` | init | the original user request |
| `parsed` | step 2 | `{description, size, max_price}` |
| `search_results` | step 3 | list of matching listing dicts |
| `selected_item` | step 4 | the single chosen listing (top result) |
| `wardrobe` | init | the user's wardrobe dict |
| `outfit_suggestion` | step 5 | string returned by `suggest_outfit` |
| `fit_card` | step 6 | string returned by `create_fit_card` |
| `notes` | step 3 | user-facing notes (e.g. "relaxed the size filter") |
| `error` | step 3 | message set when the run ends early; `None` on success |

Data flows strictly forward: `parsed → search_results → selected_item → outfit_suggestion
→ fit_card`. The agent never recomputes a stored value — `selected_item` is the exact same
object handed to both `suggest_outfit` and `create_fit_card`, and `fit_card` is built only
from the stored `outfit_suggestion`. This was verified with an instrumented identity check
(wrapping each tool to compare `id()` of the objects passed in): `selected_item is
search_results[0]`, the same object reaches both downstream tools, and
`session["outfit_suggestion"]` is exactly the value `suggest_outfit` returned. If `error` is
set, the function returns immediately so no downstream tool runs on missing data.

---

## Error Handling

| Tool | Failure mode | Agent response |
|---|---|---|
| `search_listings` | No results match | Retry once with the size filter dropped; if still empty, set `session["error"]` to a broaden-your-search message and **return early without calling any downstream tool**. |
| `suggest_outfit` | Wardrobe is empty (or missing `items` key) | Still returns a string — general styling advice for the item alone instead of failing. LLM/network errors are caught and returned as a descriptive string. |
| `create_fit_card` | `outfit` empty/whitespace, or LLM call fails | Empty outfit → descriptive error string (no LLM call). LLM failure → template caption built from the item's title, price, platform, and style tags. Never returns empty or raises. |

**Concrete example from testing — `search_listings` no-results branch.** Running
`python agent.py` on the second scenario, `"designer ballgown size XXS under $5"`:

```
session["error"]             = "No listings matched 'designer ballgown' (size XXS, under $5).
                                Try broadening your search — relax the size or budget, or use
                                different keywords."
session["fit_card"]          = None
session["outfit_suggestion"] = None
session["selected_item"]     = None
suggest_outfit call count    = 0   ← never called
create_fit_card call count   = 0   ← never called
```

The downstream call counts were captured by wrapping `suggest_outfit`/`create_fit_card`
with counters before the run; both stayed at 0, confirming the early exit. (The query also
exercises the relaxed-filter retry first: the `XXS` size is dropped and the search re-run,
but nothing under $5 matches, so `error` is set.)

**Concrete example — relaxed-filter retry succeeds.** `"graphic tee size XXXL under $30"`
finds no XXXL match, so the loop drops the size filter, re-searches, and surfaces a result
with `session["notes"] = ["No exact match for size XXXL — relaxed the size filter to show
the closest items."]` and `error = None`.

---

## Spec Reflection

- **What matched the plan.** The three-tool design, the forward-flowing session dict, and the
  conditional planning loop with an early-exit error branch were implemented as written in
  `planning.md`. The "Complete Interaction" example (vintage graphic tee → outfit → fit card)
  runs end-to-end and produces a coherent caption.
- **What I added beyond the first cut.** `planning.md` specified a *retry with relaxed
  constraints* before giving up on an empty search; my first implementation of the loop
  errored out immediately, so I added the size-relaxation retry and a `notes` field to carry
  the "what changed" message — bringing the code in line with the spec.
- **A spec limitation I found.** The regex query parser is heuristic. It assumes the item
  request appears in the first sentence, and it can misread a bare number without "under"/"$"
  as a price. These are acceptable for the project's query style but are known boundaries
  rather than guarantees.
- **Verification discipline.** Each tool was tested in isolation before wiring; the planning
  loop was checked against three end-to-end scenarios (happy / no-results / empty wardrobe);
  and state-passing was verified with object-identity assertions rather than eyeballing output.

---

## AI Usage

I used **Claude Code** as the implementation assistant throughout. Specific instances:

**1. `search_listings` (Tool 1).**
*Input I gave it:* the Tool 1 spec from `planning.md` (inputs, the listing return schema, and
the "returns empty list, does not raise" failure mode), the function docstring's numbered TODO,
and the `data/listings.json` field structure via `load_listings()`.
*What it produced:* a filter-then-score implementation — price/size filtering followed by
keyword-overlap scoring, dropping zero-score listings and sorting by score.
*What I changed/kept:* I had it score against *multiple* fields (title, description,
style_tags, colors, brand) rather than the description alone, and make the size filter a
case-insensitive substring match so `"M"` matches `"S/M"`. I verified it against three queries
(valid, empty result, price filter) before trusting it.

**2. The planning loop (`run_agent`).**
*Input I gave it:* the full `planning.md` — the Planning Loop section with its conditional
logic, the State Management section, and the Mermaid architecture diagram showing the error
branch off `search_listings`.
*What it produced:* a first `run_agent` that parsed the query, searched, selected the top
result, and chained the two LLM tools with an early exit on empty results.
*What I overrode:* the first version gave up immediately when search returned nothing. The
diagram and Error Handling table both call for a *relaxed-constraint retry* first, so I had it
add the drop-the-size-filter retry and a `notes` field, then re-verified all three scenarios.

**3. Query parser hardening.**
*Input I gave it:* a state-flow verification run that surfaced a noisy parse — the
conversational query `"I'm looking for a vintage graphic tee under $30. I mostly wear baggy
jeans and chunky sneakers. What's out there…"` was producing a description full of filler
words.
*What it produced:* a tightened `_parse_query` that scopes the description to the first
sentence and to text after a lead-in phrase, plus an expanded stopword list.
*What I changed/kept:* I accepted the first-sentence heuristic but flagged its limitation in
the spec reflection, and added regression tests (`tests/test_agent.py`) asserting the noisy
query parses to exactly `"vintage graphic tee"`.

---

## Project Layout

```
├── agent.py                  # planning loop + query parser (run_agent)
├── tools.py                  # the three tools
├── app.py                    # Gradio web UI
├── data/
│   ├── listings.json         # 40 mock secondhand listings
│   └── wardrobe_schema.json  # wardrobe format + example (10 items) + empty template
├── utils/data_loader.py      # data-loading helpers
├── tests/
│   ├── test_tools.py         # per-tool tests incl. every failure mode
│   └── test_agent.py         # parser + planning-loop tests
├── planning.md               # design doc (specs, loop, state, architecture)
└── pytest.ini                # pythonpath = . , testpaths = tests
```

## The Data

`data/listings.json` — 40 mock listings across categories (tops, bottoms, outerwear, shoes,
accessories) and styles (vintage, y2k, grunge, cottagecore, streetwear, …). Load with
`load_listings()`.

`data/wardrobe_schema.json` — `schema` (field definitions), `example_wardrobe` (10 items for
testing), and `empty_wardrobe` (new-user template). Load with `get_example_wardrobe()` or
`get_empty_wardrobe()`.
