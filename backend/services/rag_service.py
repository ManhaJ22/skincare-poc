from pathlib import Path

import chromadb
from openai import OpenAI

from models.schemas import SkinProfile

EMBEDDING_MODEL = "text-embedding-ada-002"
COLLECTION_NAME = "skinsage_kb"
KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
# Folder name -> metadata "type" value.
KB_FOLDERS = {"products": "product", "ingredients": "ingredient"}

_openai: OpenAI | None = None
_collection = None


def _embed(texts: list[str]) -> list[list[float]]:
    global _openai
    if _openai is None:
        _openai = OpenAI()
    response = _openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def init_collection() -> int:
    """Load every knowledge base file into an in-memory Chroma collection. Returns the doc count."""
    global _collection
    client = chromadb.Client()
    _collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, docs, metadatas = [], [], []
    for folder, doc_type in KB_FOLDERS.items():
        for path in sorted((KB_DIR / folder).glob("*.txt")):
            # Files are short, so each file is a single chunk.
            ids.append(f"{doc_type}:{path.stem}")
            docs.append(path.read_text(encoding="utf-8"))
            metadatas.append({"type": doc_type, "filename": path.name})

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


def retrieve_relevant_products(skin_profile: SkinProfile, top_k: int = 3, doc_type: str | None = None) -> list[str]:
    if _collection is None:
        raise RuntimeError("Knowledge base not initialized; call init_collection() first.")
    result = _collection.query(
        query_embeddings=_embed([build_query(skin_profile)]),
        n_results=top_k,
        where={"type": doc_type} if doc_type else None,
    )
    return result["documents"][0]
