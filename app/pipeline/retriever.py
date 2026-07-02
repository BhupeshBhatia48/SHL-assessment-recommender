"""
Retrieval layer: given a free-text query built from conversation constraints,
return the top-K catalog records by semantic similarity.

Backend is intentionally abstracted behind `Embedder` so the retrieval logic
(FAISS search, top-K cutoff, record hydration) never has to change depending
on which embedding model actually produces the vectors:

- SentenceTransformerEmbedder: the real production backend
  (sentence-transformers/all-MiniLM-L6-v2, matches scripts/build_index.py).
  Requires one-time internet access to download model weights from
  HuggingFace -- available on any normal dev machine or on Render/Railway at
  deploy time.
- This module does NOT hardcode which backend to use at import time; the
  caller (app startup) constructs the embedder and passes it in, so tests
  or restricted environments can swap in a different backend without
  touching this file.
"""
import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
import requests

from app import config

DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_PATH = DATA_DIR / "catalog_index.faiss"
IDS_PATH = DATA_DIR / "catalog_ids.json"


class Embedder(Protocol):
    def encode(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalized embeddings.
        task_type distinguishes documents being indexed from live queries
        for backends that support asymmetric retrieval embeddings (see
        GeminiEmbedder); backends that don't support this distinction (e.g.
        SentenceTransformerEmbedder) simply ignore the parameter."""
        ...


class SentenceTransformerEmbedder:
    """Local embedding backend using PyTorch. Accurate and free, but loading
    sentence-transformers pulls in PyTorch, which alone commonly exceeds the
    512MB RAM ceiling on free-tier hosts (Render/Railway free plans) just on
    import -- before your own code runs. Kept here as an option for anyone
    self-hosting with more available memory; GeminiEmbedder (below) is the
    default in scripts/build_index.py and app/main.py specifically because
    it avoids this problem. See GeminiEmbedder's docstring for the full
    story -- this is exactly the kind of swap the Embedder interface was
    designed to make painless."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        # task_type is part of the shared Embedder interface but this model
        # doesn't support asymmetric query/document embeddings -- ignored.
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")


class GeminiEmbedder:
    """Embedding backend using Google's hosted gemini-embedding-001 model via
    REST API instead of a locally-loaded PyTorch model.

    WHY THIS EXISTS: the original design used SentenceTransformerEmbedder
    (local, free, no per-call cost). That worked fine in local dev, but
    deploying to Render's free tier failed with "Ran out of memory (used
    over 512MB)" -- PyTorch's own footprint alone is enough to blow that
    budget on a free instance. Rather than paying for a bigger instance,
    this swaps to a hosted embeddings API: near-zero local memory, uses the
    same GEMINI_API_KEY already required for the chat LLM calls, and
    requires zero changes anywhere else in the pipeline because retrieval
    logic was built against the Embedder protocol, not a specific backend.

    MODEL NAME NOTE: this originally targeted "text-embedding-004", which
    returned a 404 on batchEmbedContents when actually deployed -- that
    model is legacy/deprecated. The current generally-available embedding
    model is "gemini-embedding-001", which is what's used here. Caught via
    a live deploy failure, not local testing (no internet to the real
    endpoint was available during initial development -- see APPROACH.md).

    DIMENSIONALITY NOTE: gemini-embedding-001 defaults to 3072 dimensions
    unless explicitly truncated. We fix it to 768 (one of the officially
    supported MRL sizes, alongside 1536/3072) explicitly on every request --
    both to keep the FAISS index compact and because the dimension must be
    IDENTICAL between whatever built the index and whatever embeds a live
    query, so leaving it to an implicit default would be fragile.

    TASK TYPE: Gemini's embedding model supports asymmetric retrieval task
    types -- documents indexed with RETRIEVAL_DOCUMENT and queries embedded
    with RETRIEVAL_QUERY are optimized differently for this exact
    query-vs-document matching scenario, rather than treating both sides
    symmetrically. encode() takes a task_type param so build_index.py (which
    embeds catalog *documents*) and retriever.py's live search (which embeds
    a *query*) each pass the correct one.

    Uses the batchEmbedContents endpoint (chunked to respect a conservative
    per-request batch size) for building the offline index, and encode()
    with a single text for live queries at request time.
    """

    MODEL_NAME = "models/gemini-embedding-001"
    OUTPUT_DIM = 768
    BATCH_SIZE = 50  # conservative chunk size for batchEmbedContents

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for GeminiEmbedder (used for embeddings "
                "even if LLM_PROVIDER=groq for chat completions)."
            )

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/{self.MODEL_NAME}"
            f":batchEmbedContents?key={self.api_key}"
        )
        requests_payload = [
            {
                "model": self.MODEL_NAME,
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
                "outputDimensionality": self.OUTPUT_DIM,
            }
            for t in texts
        ]
        resp = requests.post(
            url, json={"requests": requests_payload}, timeout=config.LLM_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        data = resp.json()
        return [e["values"] for e in data["embeddings"]]

    def encode(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            chunk = texts[i : i + self.BATCH_SIZE]
            all_vectors.extend(self._embed_batch(chunk, task_type))

        arr = np.array(all_vectors, dtype="float32")
        # Normalize for cosine similarity via inner product, matching
        # SentenceTransformerEmbedder's convention so IndexFlatIP behaves
        # identically regardless of which embedder built the index.
        # gemini-embedding-001 embeddings are not pre-normalized when a
        # non-default output_dimensionality is requested, so this step is
        # required, not just a convention match.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class CatalogRetriever:
    def __init__(
        self,
        embedder: Embedder,
        index_path: Path = INDEX_PATH,
        ids_path: Path = IDS_PATH,
        catalog_path: Path = CATALOG_PATH,
    ):
        self.embedder = embedder
        self.catalog: list[dict] = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self.catalog_by_id = {c["id"]: c for c in self.catalog}
        self.catalog_by_name_lower = {c["name"].lower(): c for c in self.catalog}
        self.index = faiss.read_index(str(index_path))
        self.ids: list[str] = json.loads(Path(ids_path).read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return up to top_k catalog records ranked by similarity to query."""
        if not query.strip():
            return []
        # RETRIEVAL_QUERY (vs. RETRIEVAL_DOCUMENT used when the catalog was
        # indexed) matters for backends supporting asymmetric retrieval
        # embeddings -- see GeminiEmbedder's docstring.
        query_vec = self.embedder.encode([query], task_type="RETRIEVAL_QUERY")
        scores, indices = self.index.search(query_vec, min(top_k, len(self.ids)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = dict(self.catalog_by_id[self.ids[idx]])
            record["_score"] = float(score)
            results.append(record)
        return results

    def get_by_name(self, name: str) -> dict | None:
        """Exact (case-insensitive) name lookup, used by the Compare branch
        to resolve named assessments directly rather than via similarity
        search -- comparisons need the specific named item, not "close
        matches"."""
        return self.catalog_by_name_lower.get(name.strip().lower())

    def is_valid_record(self, name: str, url: str) -> bool:
        """Used by the final validator: every (name, url) pair the API
        returns must correspond to a real catalog record, checked by URL
        (the more reliable key) with a name fallback."""
        for c in self.catalog:
            if c["url"] == url:
                return True
        return name.strip().lower() in self.catalog_by_name_lower


def build_query_from_constraints(constraints: dict) -> str:
    """Turns the analyzer's extracted constraint dict into a single text
    query for embedding similarity search. Order/repetition is chosen to
    weight the most decision-relevant fields (role, skills, test types)
    without needing per-field vector weighting."""
    parts = []
    if constraints.get("role"):
        parts.append(str(constraints["role"]))
    if constraints.get("seniority"):
        parts.append(str(constraints["seniority"]))
    skills = constraints.get("skills") or []
    if skills:
        parts.append(", ".join(skills))
    test_types = constraints.get("test_types_wanted") or []
    if test_types:
        parts.append(", ".join(test_types))
    languages = constraints.get("languages") or []
    if languages:
        parts.append("Languages: " + ", ".join(languages))
    if constraints.get("other_notes"):
        parts.append(str(constraints["other_notes"]))
    return " | ".join(parts)
