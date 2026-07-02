# SHL Conversational Assessment Recommender

A stateless conversational agent that turns a vague hiring need into a
grounded shortlist of SHL Individual Test Solutions, built for the SHL AI
Intern take-home assignment.

## Architecture

```
POST /chat
   |
   v
[1] Guardrail pre-check (code, no LLM) --> off-topic / injection / legal? --> refuse
   |
   v
[2] Analyzer (LLM call #1) -- re-derives intent + full constraint set from
    the WHOLE conversation history every turn
   |
   v
[3] Router
   - CLARIFY   -> templated question, recommendations = []
   - RECOMMEND / REFINE -> retrieve + respond (below)
   - COMPARE   -> resolve named assessments, grounded comparison reply
   - OFF_TOPIC -> refusal, recommendations = []
   |
   v
[4] Retriever: FAISS similarity search over the scraped catalog only
   |
   v
[5] Responder (LLM call #2): selects which retrieved candidates to keep
    (by id, from a closed list -- never free-typed) + writes the reply
   |
   v
[6] Validator: rebuilds `recommendations` from our own catalog records,
    cross-checks every URL against the live catalog, enforces the 1-10 bound
   |
   v
Schema-validated JSON response
```

Full design rationale is in `APPROACH.md`.

## Project layout

```
app/
  main.py                 FastAPI app (/health, /chat)
  config.py               tunables (LLM provider, thresholds, turn budget)
  schemas.py               Pydantic request/response models
  pipeline/
    analyzer.py            LLM call #1: intent + constraint extraction
    responder.py            LLM call #2: candidate selection + reply text
    retriever.py             FAISS search over the catalog
    guardrails.py            code-level off-topic/injection/legal pre-check
    validator.py             final schema + catalog cross-check
    router.py                orchestrates the above per turn
    llm_client.py            Gemini / Groq / Mock LLM backends
    json_utils.py            defensive JSON extraction from LLM output
  prompts/                 system prompts (analyzer, responder, compare)
  data/
    catalog.json            normalized SHL catalog (377 Individual Test Solutions)
    catalog_index.faiss     FAISS index (built by scripts/build_index.py)
    catalog_ids.json        id ordering matching the FAISS index
scripts/
  raw_catalog.json          the original SHL-provided catalog dump
  build_catalog.py          raw_catalog.json -> app/data/catalog.json
  build_index.py            app/data/catalog.json -> FAISS index
  run_eval.py               replays the 10 given traces, computes Recall@10
  dev_tfidf_sanity_check.py DEV-ONLY offline retrieval sanity check (no internet needed)
traces/                    the 10 provided conversation traces
tests/                     pytest suite (guardrails, schema compliance, trace replay)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and add your **Gemini API key**
   (https://aistudio.google.com/apikey) — this is required regardless of
   which provider you pick for chat, because embeddings always go through
   Gemini's `text-embedding-004` API (see "Why Gemini embeddings" below).
   If you want chat completions on Groq instead of Gemini, also add a
   `GROQ_API_KEY` and set `LLM_PROVIDER=groq` — embeddings still use Gemini.

2. Build the catalog and embedding index (one-time; needs internet + your
   Gemini key, since embeddings are now generated via API call, not a local
   model download):

   ```bash
   python scripts/build_catalog.py
   python scripts/build_index.py
   ```

   `app/data/catalog.json` is already included in this submission (built
   from the provided `raw_catalog.json`), but re-run `build_catalog.py`
   if you replace the raw dump. `catalog_index.faiss` / `catalog_ids.json`
   are **not** committed (they're regenerated from `catalog.json`) --
   you must run `build_index.py` at least once before starting the server.

3. Run the server:

   ```bash
   uvicorn app.main:app --reload
   ```

4. Check it's alive:

   ```bash
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders"}]}'
   ```

## Running tests

```bash
pytest tests/ -v
```

Tests use a `FakeEmbedder` (deterministic, no internet/model download
required) and `MockLLMClient` (keyword-based stand-in, no API key required)
so the full suite runs anywhere with zero external dependencies. They verify
**pipeline correctness** — schema compliance, guardrail behavior, every
branch producing a valid response, no crashes across all 10 real trace
conversations — not retrieval *quality*, which needs the real embedding
model. See `tests/conftest.py` for why.

## Measuring Recall@10 against the real thing

Once the server is running with the real embedding model and a real LLM key:

```bash
python scripts/run_eval.py --url http://localhost:8000
# or against a deployed instance:
python scripts/run_eval.py --url https://your-app.onrender.com
```

This replays each of the 10 provided traces' user turns against the live
`/chat` endpoint and reports per-trace and mean Recall@10 against the
ground-truth shortlists parsed out of the trace files.

## Deployment (Render)

`render.yaml` is included — connect the repo in the Render dashboard, set
`GEMINI_API_KEY` (or switch `LLM_PROVIDER`/keys for Groq) as an environment
variable, and deploy. The build command runs `build_catalog.py` and
`build_index.py` automatically so the index is always in sync with
`raw_catalog.json`. `Procfile` is included for Railway or any other
Procfile-based platform.

## Why Gemini embeddings instead of a local model

The original design used `sentence-transformers/all-MiniLM-L6-v2` running
locally (free, no per-call API cost). That worked in local dev, but
deploying to Render's free tier failed with **"Ran out of memory (used over
512MB)"** — PyTorch's own footprint alone is close to that ceiling before
any of our own code runs. Rather than upgrading to a paid instance, the
embedding backend was swapped to `GeminiEmbedder`
(`app/pipeline/retriever.py`), which calls Google's hosted
`text-embedding-004` API instead — near-zero local memory, reuses the same
`GEMINI_API_KEY` already needed for chat completions, and required **zero
changes** to the retrieval logic itself, since it was built against a
pluggable `Embedder` interface from the start rather than a specific
backend. `SentenceTransformerEmbedder` is still in the codebase as an
option for anyone self-hosting with more available memory.

## Known data-quality issues found and fixed (see `scripts/build_catalog.py`)

1. One catalog record's `name` field lost a word to a scraping artifact
   (raw control character) — repaired narrowly for that one record by
   rebuilding from its URL slug, after an earlier broader heuristic was
   found (via full-catalog testing) to corrupt 22 other valid names and was
   scrapped.
2. Multi-category "report" products collapse to Test Type `"D"` rather than
   joining every tagged category letter, matching the one confirmed example
   in the provided traces — flagged as a heuristic inferred from limited
   evidence, not a documented rule.

## What I'd improve with more time

- Real Recall@10 numbers from `run_eval.py` against the deployed service
  (requires a live LLM key + internet access neither of which were
  available in the sandbox this was developed in — see `APPROACH.md`).
- A second-pass reranker (cross-encoder) on top of the bi-encoder FAISS
  retrieval for higher precision at low K.
- Structured extraction of `skills_measured` tags per catalog item (currently
  relying on the raw `description` text for skill-level semantic matching)
  if the catalog page exposes that facet separately from free text.
