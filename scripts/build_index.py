"""
Builds a FAISS similarity index over app/data/catalog.json.

EMBEDDING BACKEND: uses GeminiEmbedder (hosted API, text-embedding-004) by
default, NOT the local sentence-transformers model. This was a deliberate
pivot: the original design used a local PyTorch-based model, which works
fine for local dev but exceeded the 512MB memory ceiling on Render's free
tier during deployment (PyTorch's own footprint alone is close to that
limit). GeminiEmbedder uses the same GEMINI_API_KEY already required for
chat completions and has near-zero local memory footprint. See
app/pipeline/retriever.py's GeminiEmbedder docstring for the full story.

If you have a self-hosted environment with more available memory and would
rather avoid API calls / rate limits for the one-time index build, you can
swap in SentenceTransformerEmbedder instead (also defined in retriever.py)
-- the rest of this script and the whole retrieval pipeline are unaffected
either way, since everything is built against the Embedder interface, not a
specific backend.

Design decisions that DON'T change with the embedder swap:
- Embedding text template per record intentionally folds in more than just
  the raw `description`, because the raw description alone often doesn't
  contain the words a recruiter would actually type. E.g. a description for
  a Java test might never say "mid-level" or "senior" even though the
  catalog's own job_levels field tags it that way. Concretely we embed:
      <name> | <spelled-out categories> | <job levels> | <description>
- Similarity metric: cosine similarity via L2-normalized vectors + FAISS
  IndexFlatIP (inner product on normalized vectors == cosine similarity).
  IndexFlatIP is an exact (non-approximate) search -- with only 377 items
  there's no need for an approximate index like IVF/HNSW.
- Output: catalog_index.faiss (the vector index) + catalog_ids.json (an
  ordered list of catalog record `id`s matching the index's row order).
"""
import json
import sys
from pathlib import Path

# Ensure the project root (parent of this scripts/ folder) is importable as
# `app.*` regardless of how this script is invoked. Running `python
# scripts/build_index.py` only puts scripts/ itself on sys.path by default,
# not the project root -- this line is what makes `from app.pipeline...`
# work whether you run this from the project root, from inside scripts/, or
# as part of a platform build command (e.g. Render's buildCommand).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss

from app.pipeline.retriever import GeminiEmbedder

DATA_DIR = Path(__file__).parent.parent / "app" / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_PATH = DATA_DIR / "catalog_index.faiss"
IDS_PATH = DATA_DIR / "catalog_ids.json"


def build_embedding_text(record: dict) -> str:
    parts = [record["name"]]
    if record.get("categories"):
        parts.append(", ".join(record["categories"]))
    if record.get("job_levels"):
        parts.append("Job levels: " + ", ".join(record["job_levels"]))
    if record.get("description"):
        parts.append(record["description"])
    return " | ".join(parts)


def main():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(catalog)} catalog records")

    texts = [build_embedding_text(r) for r in catalog]
    ids = [r["id"] for r in catalog]

    print("Embedding catalog via Gemini text-embedding-004 API...")
    embedder = GeminiEmbedder()
    embeddings = embedder.encode(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    IDS_PATH.write_text(json.dumps(ids, indent=2), encoding="utf-8")

    print(f"Wrote FAISS index ({index.ntotal} vectors, dim={dim}) -> {INDEX_PATH}")
    print(f"Wrote id mapping -> {IDS_PATH}")


if __name__ == "__main__":
    main()
