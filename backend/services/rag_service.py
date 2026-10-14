import re
from pathlib import Path

import chromadb
from openai import OpenAI

from models.schemas import SkinProfile

EMBEDDING_MODEL = "text-embedding-ada-002"
COLLECTION_NAME = "skinsage_kb"
KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
KB_FOLDERS = {"products": "product", "ingredients": "ingredient"}

_openai: OpenAI | None = None
_collection = None

ROUTINE_STEPS_PRIORITY = ["cleanser", "moisturizer", "serum", "spf", "treatment"]

# Keyword -> canonical step, checked in order so combined labels resolve sensibly:
# "Moisturizer + SPF (combined)" -> spf, "Treatment / Serum" -> treatment.
STEP_KEYWORDS = [
    ("spf", "spf"),
    ("sunscreen", "spf"),
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
            range_match = re.search(r"(\d+)\s*-\s*(\d+)", value)
            tones = [int(n) for n in (range_match.groups() if range_match else re.findall(r"\d+", value))]
            if tones:
                return min(tones), max(tones)
    raise ValueError(f"{filename}: no Monk tone range found; add a 'Monk tones suitable:' line")


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


def _product_filter(monk_tone: int | None, step: str | None = None) -> dict:
    """Chroma where-clause for products, hard-filtered to the user's Monk tone (when known) and step."""
    clauses = [{"type": {"$eq": "product"}}]
    if step:
        clauses.append({"routine_step": {"$eq": step}})
    if monk_tone:
        clauses += [
            {"monk_tones_min": {"$lte": monk_tone}},
            {"monk_tones_max": {"$gte": monk_tone}},
        ]
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _query(embedding: list[float], where: dict, n_results: int) -> tuple[list[str], list[str]]:
    result = _collection.query(query_embeddings=[embedding], n_results=n_results, where=where, include=["documents"])
    return result["ids"][0], result["documents"][0]


def retrieve_relevant_products(skin_profile: SkinProfile, top_k: int = 5, doc_type: str | None = None) -> list[str]:
    """
    Semantic search over the KB. Products whose Monk tone range excludes the user's tone are
    never returned. monk_tone 0 means analysis couldn't determine a tone, so no tone filter applies.
    """
    if _collection is None:
        raise RuntimeError("Knowledge base not initialized; call init_collection() first.")

    monk_tone = skin_profile.monk_tone or None
    embedding = _embed([build_query(skin_profile)])[0]

    if doc_type == "product":
        # Best tone-suitable product for each routine step, in routine order...
        ids, docs = [], []
        for step in ROUTINE_STEPS_PRIORITY:
            step_ids, step_docs = _query(embedding, _product_filter(monk_tone, step), n_results=1)
            ids += step_ids
            docs += step_docs
        # ...then fill any remaining slots with the next-best tone-suitable products.
        more_ids, more_docs = _query(embedding, _product_filter(monk_tone), n_results=15)
        docs += [doc for id_, doc in zip(more_ids, more_docs) if id_ not in ids]
        return docs[:top_k]

    if doc_type == "ingredient":
        where = {"type": {"$eq": "ingredient"}}
    else:
        # Mixed context: every ingredient is eligible; products still have to cover the tone.
        where = {"$or": [{"type": {"$eq": "ingredient"}}, _product_filter(monk_tone)]}
    _, docs = _query(embedding, where, n_results=15)
    return deduplicate_by_step(docs, max_results=top_k)