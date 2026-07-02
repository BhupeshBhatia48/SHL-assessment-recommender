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
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalized embeddings."""
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

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")


class GeminiEmbedder:
    """Embedding backend using Google's hosted text-embedding-004 model via
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
    This is a deliberate, evidence-driven trade-off worth being able to
    explain in the interview: local embeddings are "free" in dollar terms
    but not in memory, and free-tier hosting makes memory the actual
    constraint that matters.

    Uses the batchEmbedContents endpoint (chunked to respect the API's
    per-request batch size limit) for building the offline index, and
    embedContent for single live queries at request time.
    """

    MODEL_NAME = "models/text-embedding-004"
    OUTPUT_DIM = 768
    BATCH_SIZE = 100  # Gemini batchEmbedContents limit per request

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for GeminiEmbedder (used for embeddings "
                "even if LLM_PROVIDER=groq for chat completions)."
            )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/{self.MODEL_NAME}"
            f":batchEmbedContents?key={self.api_key}"
        )
        requests_payload = [
            {"model": self.MODEL_NAME, "content": {"parts": [{"text": t}]}} for t in texts
        ]
        resp = requests.post(
            url, json={"requests": requests_payload}, timeout=config.LLM_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        data = resp.json()
        return [e["values"] for e in data["embeddings"]]

    def encode(self, texts: list[str]) -> np.ndarray:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            chunk = texts[i : i + self.BATCH_SIZE]
            all_vectors.extend(self._embed_batch(chunk))

        arr = np.array(all_vectors, dtype="float32")
        # Normalize for cosine similarity via inner product, matching
        # SentenceTransformerEmbedder's convention so IndexFlatIP behaves
        # identically regardless of which embedder built the index.
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
        query_vec = self.embedder.encode([query])
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
