"""
Shared pytest fixtures.

IMPORTANT: these tests inject a FakeEmbedder instead of the real
SentenceTransformerEmbedder. This is a deliberate, narrowly-scoped
substitution -- the real embedder needs to download model weights from
HuggingFace, which requires internet access this CI/dev environment may not
have. FakeEmbedder produces deterministic, non-semantic vectors of the
correct dimensionality (384, matching all-MiniLM-L6-v2) purely so the
FAISS index -- which WAS built with the real model in
app/data/catalog_index.faiss -- can still be loaded and searched without
error.

Because FakeEmbedder's vectors aren't semantically meaningful, these tests
deliberately do NOT assert on retrieval *quality* (which specific catalog
items come back for a query) -- that is exactly what
scripts/run_eval.py against the real deployed service (with the real
embedder) is for, and what should be run before submission. These tests
instead verify pipeline *correctness*: schema compliance, guardrail
behavior, and that every branch (clarify/recommend/refine/compare/refuse)
produces a valid, well-formed response.
"""
import json
from pathlib import Path

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.pipeline.llm_client import MockLLMClient
from app.pipeline.retriever import CatalogRetriever

CATALOG_PATH = Path(__file__).parent.parent / "app" / "data" / "catalog.json"


class FakeEmbedder:
    """Deterministic, hash-based fake embeddings -- no internet, no model
    download. NOT semantically meaningful; see module docstring above."""

    DIM = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.standard_normal(self.DIM).astype("float32")
            v /= np.linalg.norm(v)
            vectors.append(v)
        return np.vstack(vectors)


@pytest.fixture(scope="session")
def fake_index_paths(tmp_path_factory):
    """Builds a throwaway FAISS index over the real catalog using
    FakeEmbedder, once per test session, so tests don't depend on the real
    sentence-transformers index having been built (see module docstring)."""
    tmp_dir = tmp_path_factory.mktemp("fake_index")
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    embedder = FakeEmbedder()
    texts = [c["name"] + " " + c["description"] for c in catalog]
    vectors = embedder.encode(texts)

    index = faiss.IndexFlatIP(embedder.DIM)
    index.add(vectors)

    index_path = tmp_dir / "fake_catalog_index.faiss"
    ids_path = tmp_dir / "fake_catalog_ids.json"
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps([c["id"] for c in catalog]), encoding="utf-8")

    return index_path, ids_path


@pytest.fixture
def client(fake_index_paths):
    index_path, ids_path = fake_index_paths
    main_module._retriever = CatalogRetriever(
        embedder=FakeEmbedder(),
        index_path=index_path,
        ids_path=ids_path,
        catalog_path=CATALOG_PATH,
    )
    main_module._llm = MockLLMClient()
    return TestClient(main_module.app)
