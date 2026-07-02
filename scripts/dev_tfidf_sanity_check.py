"""
DEV-ONLY sanity check -- NOT part of the production pipeline.

This sandbox has no internet access to huggingface.co, so the real
sentence-transformers model (used by scripts/build_index.py and
app/pipeline/retriever.py) can't be downloaded here. That's a limitation of
this specific dev environment only -- any normal machine or the actual
Render/Railway deployment has full internet and should just run
`python scripts/build_index.py` for real.

To still validate the *retrieval logic itself* (query construction, ranking,
whether sensible assessments come back for a realistic query) before wiring
up the LLM, this script builds a quick TF-IDF + cosine-similarity index
instead, using the exact same embedding-text template as build_index.py.
TF-IDF is weaker than real sentence embeddings (no semantic
generalization -- e.g. it won't connect "leadership" to "management" the way
a real embedding model would), so results here are a lower bound on what the
real pipeline will do, not a replacement for it.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "app" / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"


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
    texts = [build_embedding_text(r) for r in catalog]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    queries = [
        "senior Java developer with stakeholder management skills",
        "safety critical plant operator dependability",
        "entry level contact centre customer service English",
        "graduate management trainee cognitive personality situational judgement",
        "senior Rust engineer high performance networking",
        "bilingual healthcare admin Spanish HIPAA",
        "quick screen admin assistant Excel Word",
        "senior leadership executive selection benchmark",
    ]

    for q in queries:
        q_vec = vectorizer.transform([q])
        sims = cosine_similarity(q_vec, matrix)[0]
        top_idx = sims.argsort()[::-1][:5]
        print(f"\nQUERY: {q}")
        for i in top_idx:
            r = catalog[i]
            print(f"  [{sims[i]:.3f}] {r['name']} ({r['test_type']})")


if __name__ == "__main__":
    main()
