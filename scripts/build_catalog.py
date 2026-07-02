"""
Cleans the raw SHL-provided catalog dump (scripts/raw_catalog.json) into a
normalized app/data/catalog.json that the rest of the pipeline consumes.

Key findings from inspecting the raw file (kept here as documentation since
they drive every decision below):

- Raw file is a JSON list of 377 records, one per Individual Test Solution.
  It contains a few raw control characters inside string fields (line breaks
  copy-pasted from the site), so it must be parsed with `strict=False`.
- Fields present on every record: entity_id, name, link, scraped_at,
  job_levels (list), job_levels_raw (str), languages (list), languages_raw,
  duration (str, human readable e.g. "13 minutes", sometimes ""),
  duration_raw, status ("ok" for all 377), remote ("yes" for all 377),
  adaptive ("yes"/"no"), description, keys (list of category names).
- `keys` values are exactly the 8 SHL solution categories, and each maps 1:1
  to the single-letter "Test Type" code shown in SHL's own catalog UI and
  used throughout the provided conversation traces:
      Ability & Aptitude          -> A
      Biodata & Situational Judgment -> B
      Competencies                -> C
      Development & 360           -> D
      Assessment Exercises        -> E
      Knowledge & Skills          -> K
      Personality & Behavior      -> P
      Simulations                 -> S
  Multi-category items get a comma-joined code, e.g. "K,S" or "C, K" -- we
  normalize to a comma-with-no-space form ("K,S") for consistency, since the
  traces show both spacing styles used inconsistently by humans.
- `duration` is unreliable for filtering ("" for ~16% of records, and values
  like "Untimed"/"Variable" appear in the traces even though the raw catalog
  doesn't always carry them) -- we parse a numeric `duration_minutes` when
  possible and keep the original string for display, but never *require* it.
- KNOWN DATA QUIRK #1 -- "Development & 360" collapsing: for the one product
  observed in the traces that carries many `keys` categories at once
  ("Global Skills Development Report"), the trace's displayed Test Type is
  simply "D", not the full joined set of every category letter. This matches
  the general SHL convention that post-assessment *report* products (as
  opposed to standalone administered tests) are classified as Test Type "D"
  regardless of how many content domains the report can summarize. We
  replicate that: if "Development & 360" is present alongside 2+ other
  categories, we collapse test_type to just "D". This is a heuristic inferred
  from one confirmed example, not a documented rule -- flagged here so it can
  be revisited if a counter-example shows up in eval.
- KNOWN DATA QUIRK #2 -- one scraped `name` field ("Microsoft \n    365
  (New)") lost the word "Excel" to a scraping artifact (an embedded raw
  control character, likely from an icon/SVG between "Microsoft" and "365"
  in the source HTML). Repaired by rebuilding the name from the URL slug,
  but ONLY for the record that actually contains a raw control character --
  an earlier, broader "slug word must appear in name" heuristic was tried
  and found to corrupt 22 other perfectly valid names (".NET Framework 4.5",
  "C# Programming", "Sales Transformation 2.0", etc.) because punctuation
  and version numbers don't tokenize the same way in names vs. slugs. Kept
  narrow on purpose after catching that regression against the full catalog.
"""
import json
import re
from pathlib import Path

RAW_PATH = Path(__file__).parent / "raw_catalog.json"
OUT_PATH = Path(__file__).parent.parent / "app" / "data" / "catalog.json"

CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def slug_from_link(link: str) -> str:
    """Stable id derived from the catalog URL slug, e.g. 'opq32r'-style."""
    match = re.search(r"/view/([^/]+)/?$", link)
    return match.group(1) if match else link


def parse_duration_minutes(duration_str: str):
    if not duration_str:
        return None
    match = re.search(r"(\d+)", duration_str)
    return int(match.group(1)) if match else None


STOPWORDS = {"new", "the", "and", "of", "a"}


def repair_name(name: str, link: str) -> str:
    """Fix scraping artifacts (control chars swallowing a word).

    IMPORTANT: this used to be a general "does the name contain every
    significant word from the URL slug" heuristic. That was tested against
    the full 377-record catalog and found to silently corrupt 22 perfectly
    good names (".NET Framework 4.5" -> "Net Framework 4 5", "C# Programming"
    -> "C Programming", "Sales Transformation 2.0" -> a mangled slug dump,
    etc.) because punctuation like '.', '#', '/', version numbers, and
    abbreviations don't tokenize the same way as URL slugs. A heuristic that
    "fixes" 1 real bug by introducing 22 new ones is a net loss, so it was
    scrapped in favor of only ever touching the one record we've manually
    confirmed is actually broken (verified by checking for raw control
    characters in the untouched raw name -- see KNOWN DATA QUIRK #2).
    """
    if re.search(r"[\x00-\x1f]", name):
        slug = slug_from_link(link)
        slug_words = [w for w in slug.split("-") if w and w.lower() not in STOPWORDS]
        rebuilt = " ".join(w.capitalize() for w in slug_words)
        if "new" in slug.split("-"):
            rebuilt += " (New)"
        return rebuilt
    return re.sub(r"\s+", " ", name).strip()


def normalize_test_type(keys: list) -> str:
    codes = [CATEGORY_TO_CODE[k] for k in keys if k in CATEGORY_TO_CODE]
    # KNOWN DATA QUIRK #1: report-style products tagging many categories at
    # once collapse to "D" when Development & 360 is one of them.
    if "D" in codes and len(codes) > 2:
        return "D"
    return ",".join(codes)


def normalize_record(raw: dict) -> dict:
    keys = raw.get("keys", [])
    test_type = normalize_test_type(keys)

    duration_str = (raw.get("duration") or "").strip()
    link = raw.get("link", "").strip()

    return {
        "id": slug_from_link(link),
        "entity_id": raw.get("entity_id"),
        "name": repair_name(raw.get("name", ""), link),
        "url": raw.get("link", "").strip(),
        "test_type": test_type,
        "categories": keys,
        "description": (raw.get("description") or "").strip(),
        "job_levels": raw.get("job_levels", []),
        "languages": raw.get("languages", []),
        "duration_display": duration_str,
        "duration_minutes": parse_duration_minutes(duration_str),
        "remote_testing": raw.get("remote") == "yes",
        "adaptive_irt": raw.get("adaptive") == "yes",
    }


def main():
    raw_text = RAW_PATH.read_text(encoding="utf-8")
    raw_data = json.loads(raw_text, strict=False)

    seen_ids = set()
    catalog = []
    for raw in raw_data:
        record = normalize_record(raw)
        if not record["name"] or not record["url"]:
            continue  # guard against malformed rows
        if record["id"] in seen_ids:
            continue  # guard against duplicate scrapes
        seen_ids.add(record["id"])
        catalog.append(record)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Normalized {len(catalog)} / {len(raw_data)} raw records -> {OUT_PATH}")
    no_desc = sum(1 for c in catalog if not c["description"])
    no_type = sum(1 for c in catalog if not c["test_type"])
    print(f"  records missing description: {no_desc}")
    print(f"  records missing test_type:   {no_type}")


if __name__ == "__main__":
    main()
