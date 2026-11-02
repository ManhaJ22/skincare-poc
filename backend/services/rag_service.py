import re
from pathlib import Path

import chromadb
from openai import OpenAI

from models.schemas import SkinProfile
from services import tone_rules

EMBEDDING_MODEL = "text-embedding-ada-002"
COLLECTION_NAME = "dermalis_kb"
KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
KB_FOLDERS = {"products": "product", "ingredients": "ingredient"}

_openai: OpenAI | None = None
_collection = None

# Routine order (also the display order for recommendations).
ROUTINE_STEPS_PRIORITY = [
    "cleanser", "toner", "serum", "treatment", "exfoliant",
    "spot treatment", "eye cream", "moisturizer", "spf", "mask",
]

# Keyword -> canonical step, checked in order so combined labels resolve sensibly:
# "Moisturizer + SPF (combined)" -> spf, "Treatment / Serum" -> treatment,
# "Spot treatment" -> spot treatment, "Eye cream" -> eye cream.
STEP_KEYWORDS = [
    ("spf", "spf"),
    ("sunscreen", "spf"),
    ("spot treatment", "spot treatment"),
    ("eye", "eye cream"),
    ("mask", "mask"),
    ("exfoliant", "exfoliant"),
    ("toner", "toner"),
    ("treatment", "treatment"),
    ("cleanser", "cleanser"),
    ("moisturizer", "moisturizer"),
    ("serum", "serum"),
    ("essence", "serum"),
]


def _embed(texts: list[str]) -> list[list[float]]:
    global _openai
    if _openai is None:
        _openai = OpenAI()
    response = _openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def init_collection() -> int:
    global _collection
    client = chromadb.Client()
    _collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, docs, metadatas = [], [], []
    for folder, doc_type in KB_FOLDERS.items():
        for path in sorted((KB_DIR / folder).glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            metadata = {"type": doc_type, "filename": path.name}
            if doc_type == "product":
                tone_min, tone_max = _parse_monk_range(text, path.name)
                metadata.update(
                    routine_step=_parse_routine_step(text) or "other",
                    monk_tones_min=tone_min,
                    monk_tones_max=tone_max,
                    **_parse_tone_fields(text, path.name),
                    **_parse_listing(text, path.name),
                )
            ids.append(f"{doc_type}:{path.stem}")
            docs.append(text)
            metadatas.append(metadata)

    _collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=_embed(docs))
    return _collection.count()


def build_query(skin_profile: SkinProfile) -> str:
    parts = []
    if skin_profile.monk_tone:
        parts.append(f"Monk skin tone {skin_profile.monk_tone}")
    concerns = list(dict.fromkeys(skin_profile.confirmed_concerns + skin_profile.observed_concerns))
    if concerns:
        parts.append("concerns: " + ", ".join(concerns))
    if skin_profile.skin_type:
        parts.append(f"{skin_profile.skin_type} skin type")
    if skin_profile.goals:
        parts.append("goals: " + ", ".join(skin_profile.goals))
    if skin_profile.budget_monthly is not None:
        parts.append(f"budget ${skin_profile.budget_monthly} per month")
    if skin_profile.fragrance_free:
        parts.append("fragrance free")
    if skin_profile.allergies:
        parts.append("avoid: " + ", ".join(skin_profile.allergies))
    return "; ".join(parts) or "gentle daily skincare routine"


def _parse_routine_step(doc_text: str) -> str | None:
    """Return the canonical routine step of a product doc, or None for docs without one (ingredients)."""
    for line in doc_text.splitlines():
        if line.lower().startswith("routine step:"):
            raw = line.partition(":")[2].lower()
            for keyword, step in STEP_KEYWORDS:
                if keyword in raw:
                    return step
            return "other"
    return None


def _parse_monk_range(doc_text: str, filename: str) -> tuple[int, int]:
    """
    Read a product's suitable Monk tone range. Handles both KB formats:
    "Monk tones suitable: 4, 5, 6, 7" and "MONK SKIN TONES: best match for Monk 1-5".
    Assumes the tones form a contiguous range, which holds for every file in the KB.
    """
    for line in doc_text.splitlines():
        label, _, value = line.partition(":")
        if label.strip().lower() in ("monk tones suitable", "monk skin tones"):
            tone_range = _parse_range(value)
            if tone_range:
                return tone_range
    raise ValueError(f"{filename}: no Monk tone range found; add a 'Monk tones suitable:' line")


def _parse_range(value: str) -> tuple[int, int] | None:
    """'1-5' -> (1, 5); '4, 5, 6' -> (4, 6). The first explicit range wins."""
    range_match = re.search(r"(\d+)\s*-\s*(\d+)", value)
    numbers = [int(n) for n in (range_match.groups() if range_match else re.findall(r"\d+", value))]
    return (min(numbers), max(numbers)) if numbers else None


def _field(doc_text: str, label: str) -> str | None:
    """Value of the first 'Label: value' line, matching the label case-insensitively."""
    for line in doc_text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == label:
            return value.strip()
    return None


def _parse_listing(doc_text: str, filename: str) -> dict:
    """ASIN and display price; handles both 'ASIN:'/'Price:' and 'AMAZON ASIN:'/'PRICE RANGE:' formats."""
    asin = _field(doc_text, "asin") or _field(doc_text, "amazon asin")
    if not asin:
        raise ValueError(f"{filename}: add an 'ASIN:' line")
    price = _field(doc_text, "price") or _field(doc_text, "price range") or ""
    # Drop trailing notes like "$6-$12 (30 mL; about 2-3 months ...)".
    return {"asin": asin, "price": price.split(" (")[0].strip()}


def _parse_tone_fields(doc_text: str, filename: str) -> dict:
    """Structured fields tone_rules ranks on. Missing required fields fail loudly at startup."""
    best = _parse_range(_field(doc_text, "monk tones best") or "")
    if best is None:
        raise ValueError(f"{filename}: add a 'Monk tones best: N-M' line")
    irritation = (_field(doc_text, "irritation potential") or "").lower()
    if irritation not in ("low", "medium", "high"):
        raise ValueError(f"{filename}: add an 'Irritation potential: low|medium|high' line")
    return {
        "monk_best_min": best[0],
        "monk_best_max": best[1],
        "irritation": irritation,
        # Only sunscreens declare a cast; Chroma metadata can't hold None.
        "white_cast": (_field(doc_text, "white cast") or "n/a").lower(),
        "targets": (_field(doc_text, "targets") or "").lower(),
        "fragrance_free": (_field(doc_text, "fragrance free") or "").lower().startswith("yes"),
    }


def _step_rank(doc: str) -> int:
    step = _parse_routine_step(doc)
    if step in ROUTINE_STEPS_PRIORITY:
        return ROUTINE_STEPS_PRIORITY.index(step)
    return len(ROUTINE_STEPS_PRIORITY) + (step is None)


def deduplicate_by_step(raw_docs: list[str], max_results: int = 5) -> list[str]:
    """
    Walk docs in similarity order, keeping the best product per routine step plus every
    doc without a step (ingredients). Fill leftover slots with the skipped duplicates,
    then order products by routine step with ingredients last.
    """
    seen_steps: set[str] = set()
    selected: list[str] = []
    overflow: list[str] = []

    for doc in raw_docs:
        step = _parse_routine_step(doc)
        if step is not None and step in seen_steps:
            overflow.append(doc)
            continue
        if step is not None:
            seen_steps.add(step)
        selected.append(doc)

    output = (selected + overflow)[:max_results]
    return sorted(output, key=_step_rank)


def _product_filter(monk_tone: int | None) -> dict:
    """Chroma where-clause for products, hard-filtered to the user's Monk tone (when known)."""
    clauses = [{"type": {"$eq": "product"}}]
    if monk_tone:
        clauses += [
            {"monk_tones_min": {"$lte": monk_tone}},
            {"monk_tones_max": {"$gte": monk_tone}},
        ]
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _query(embedding: list[float], where: dict, n_results: int) -> list[dict]:
    result = _collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": id_, "doc": doc, "meta": meta, "similarity": 1 - distance}
        for id_, doc, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def rank_products(embedding: list[float], monk_tone: int | None) -> list[dict]:
    """Every tone-eligible product, minus tone-rule exclusions, sorted by tone-adjusted score."""
    hits = _query(embedding, _product_filter(monk_tone), n_results=_collection.count())
    hits = [hit for hit in hits if not tone_rules.is_excluded(hit["meta"], monk_tone)]
    for hit in hits:
        hit["score"] = tone_rules.score(hit["meta"], hit["similarity"], monk_tone)
    return sorted(hits, key=lambda hit: hit["score"], reverse=True)


def _eligible_products(skin_profile: SkinProfile) -> list[dict]:
    """Tone-ranked products for this profile, minus fragranced ones when the user wants fragrance-free."""
    if _collection is None:
        raise RuntimeError("Knowledge base not initialized; call init_collection() first.")
    embedding = _embed([build_query(skin_profile)])[0]
    ranked = rank_products(embedding, skin_profile.monk_tone or None)
    if skin_profile.fragrance_free:
        ranked = [hit for hit in ranked if hit["meta"]["fragrance_free"]]
    return ranked


def group_by_step(ranked: list[dict], per_step: int) -> dict[str, list[dict]]:
    """Top `per_step` hits for each routine step, keeping rank order, steps in routine order."""
    groups = {step: [hit for hit in ranked if hit["meta"]["routine_step"] == step][:per_step]
              for step in ROUTINE_STEPS_PRIORITY}
    return {step: hits for step, hits in groups.items() if hits}


def retrieve_routine_candidates(skin_profile: SkinProfile, per_step: int = 2) -> dict[str, list[dict]]:
    """Best tone-fit products per routine step: a primary pick plus alternatives to swap in."""
    return group_by_step(_eligible_products(skin_profile), per_step)


def retrieve_relevant_products(skin_profile: SkinProfile, top_k: int = 5, doc_type: str | None = None) -> list[str]:
    """
    Semantic search over the KB. Products whose Monk tone range excludes the user's tone are
    never returned. monk_tone 0 means analysis couldn't determine a tone, so no tone filter applies.
    """
    if doc_type == "product":
        ranked = _eligible_products(skin_profile)
        # Best tone-fit product for each routine step so the routine covers its bases...
        picked = []
        for step in ROUTINE_STEPS_PRIORITY:
            best = next((hit for hit in ranked if hit["meta"]["routine_step"] == step), None)
            if best:
                picked.append(best)
        # ...then fill remaining slots with the next-best fits, and hand them over best-first.
        picked += [hit for hit in ranked if hit not in picked]
        picked = sorted(picked[:top_k], key=lambda hit: hit["score"], reverse=True)
        return [hit["doc"] for hit in picked]

    if _collection is None:
        raise RuntimeError("Knowledge base not initialized; call init_collection() first.")
    monk_tone = skin_profile.monk_tone or None
    embedding = _embed([build_query(skin_profile)])[0]
    if doc_type == "ingredient":
        where = {"type": {"$eq": "ingredient"}}
    else:
        # Mixed context: every ingredient is eligible; products still have to cover the tone.
        where = {"$or": [{"type": {"$eq": "ingredient"}}, _product_filter(monk_tone)]}
    hits = _query(embedding, where, n_results=15)
    docs = [hit["doc"] for hit in hits if not tone_rules.is_excluded(hit["meta"], monk_tone)]
    return deduplicate_by_step(docs, max_results=top_k)