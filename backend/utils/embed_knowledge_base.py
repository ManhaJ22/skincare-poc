"""
Embed the ingredient knowledge base into a persistent Chroma collection for RAG.

Walks backend/knowledge_base/ingredients/ and backend/knowledge_base/ingredients_progress_kb/,
chunks every .txt file, embeds the chunks with OpenAI text-embedding-ada-002 in batches of 100,
and stores them in the "skinsage_ingredients" collection (cosine distance) under backend/chroma_db/.

Chunking: documents are first split into sections at ruled headings (a single line between two
"=====" / "-----" rules, e.g. "SECTION 2: EFFECTIVE CONCENTRATIONS"); rule lines are dropped.
Within a section, paragraphs (split on blank lines) are packed up to MAX_TOKENS; an oversized
paragraph is split on single newlines. Chunks never span sections, and each one starts with a
"[<document> — <section>]" line so it is self-describing when retrieved.

Re-runs are incremental: every chunk carries its file's sha256 and the chunker version, so
unchanged files are skipped, edited files are re-embedded, and chunks from deleted files are removed.

    python embed_knowledge_base.py        (from backend/utils, or any folder)
"""

import hashlib
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

import chromadb
import tiktoken

BACKEND_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BACKEND_DIR / "knowledge_base"
SOURCE_DIRS = [KB_DIR / "ingredients", KB_DIR / "ingredients_progress_kb"]
CHROMA_DIR = BACKEND_DIR / "chroma_db"
COLLECTION_NAME = "skinsage_ingredients"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_TOKENS = 500
BATCH_SIZE = 100
CHUNKER_VERSION = 2  # bump when chunking changes, so every file is re-embedded
HEADER_SECTION = "Document header"

_encoding = tiktoken.get_encoding("cl100k_base")  # ada-002's tokenizer
_RULE_RE = re.compile(r"^\s*(?:={5,}|-{5,})\s*$")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def doc_name(filename: str) -> str:
    """'niacinamide_knowledge_base (1).txt' -> 'niacinamide knowledge base'."""
    stem = re.sub(r"\s*\(\d+\)$", "", Path(filename).stem)
    return stem.replace("_", " ").strip()


def _is_rule(line: str) -> bool:
    return bool(_RULE_RE.match(line))


def split_sections(text: str) -> list[tuple[str, str]]:
    """
    [(heading, body), ...]. A heading is a single non-blank line between two rule lines. Text
    before the first heading is HEADER_SECTION. Rule lines are dropped. Empty sections are skipped.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    sections, heading, body = [], HEADER_SECTION, []
    i = 0
    while i < len(lines):
        if (_is_rule(lines[i]) and i + 2 < len(lines) and lines[i + 1].strip()
                and not _is_rule(lines[i + 1]) and _is_rule(lines[i + 2])):
            sections.append((heading, "\n".join(body)))
            heading, body = lines[i + 1].strip(), []
            i += 3
            continue
        if not _is_rule(lines[i]):
            body.append(lines[i])
        i += 1
    sections.append((heading, "\n".join(body)))
    return [(h, b.strip()) for h, b in sections if b.strip()]


def _pack(pieces: list[str], sep: str, max_tokens: int) -> list[str]:
    """Greedily join consecutive pieces with sep while the result stays within max_tokens."""
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, max_tokens: int) -> list[str]:
    tokens = _encoding.encode(text)
    return [_encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def _chunk_body(text: str, max_tokens: int) -> list[str]:
    """Paragraphs packed up to max_tokens; oversized paragraphs split on newlines, then by tokens."""
    pieces = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if count_tokens(paragraph) <= max_tokens:
            pieces.append(paragraph)
            continue
        lines = []
        for line in (l for l in paragraph.split("\n") if l.strip()):
            lines.extend(_hard_split(line, max_tokens) if count_tokens(line) > max_tokens else [line])
        pieces.extend(_pack(lines, "\n", max_tokens))
    return _pack(pieces, "\n\n", max_tokens)


def chunk_text(text: str, max_tokens: int = MAX_TOKENS, name: str | None = None) -> list[str]:
    """
    Section-aware chunks of at most max_tokens. With a name, each chunk is prefixed with
    "[name — section]" (or "[name]" for an unsectioned document); the prefix counts toward the cap.
    """
    sections = split_sections(text)
    has_headings = any(h != HEADER_SECTION for h, _ in sections)
    chunks = []
    for heading, body in sections:
        prefix = ""
        if name:
            prefix = f"[{name} — {heading}]\n" if has_headings else f"[{name}]\n"
        budget = max_tokens - count_tokens(prefix)
        chunks.extend(prefix + chunk for chunk in _chunk_body(body, budget))
    return chunks


def get_collection(path: Path = CHROMA_DIR):
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def embed_batches(texts: list[str], client=None) -> list[list[float]]:
    """Embed texts with ada-002, BATCH_SIZE at a time, preserving order."""
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[i:i + BATCH_SIZE])
        vectors.extend(item.embedding for item in response.data)
    return vectors


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest(source_dirs: list[Path] = SOURCE_DIRS, collection=None,
           embed: Callable[[list[str]], list[list[float]]] = embed_batches) -> dict:
    """
    Bring the collection in line with source_dirs. Chunk metadata "source" is "<folder>/<filename>".
    Returns {"added", "skipped", "removed": [source, ...], "chunks": n_embedded}.
    """
    collection = collection if collection is not None else get_collection()
    stats = {"added": [], "skipped": [], "removed": [], "chunks": 0}
    ids, docs, metadatas, current = [], [], [], set()

    for folder in source_dirs:
        for path in sorted(Path(folder).glob("*.txt")):
            source = f"{folder.name}/{path.name}"
            current.add(source)
            digest = file_hash(path)
            existing = collection.get(where={"source": source}, include=["metadatas"])
            if existing["ids"] and all(m.get("file_hash") == digest and m.get("chunker_version") == CHUNKER_VERSION
                                       for m in existing["metadatas"]):
                stats["skipped"].append(source)
                print(f"  skip   {source} (unchanged, {len(existing['ids'])} chunks)")
                continue
            if existing["ids"]:
                collection.delete(where={"source": source})
            name = doc_name(path.name)
            chunks = chunk_text(path.read_text(encoding="utf-8"), name=name)
            id_stem = f"{folder.name}/{name.replace(' ', '_')}"
            for i, chunk in enumerate(chunks):
                ids.append(f"{id_stem}:{i}")
                docs.append(chunk)
                metadatas.append({"source": source, "folder": folder.name, "file_hash": digest,
                                  "chunker_version": CHUNKER_VERSION, "chunk_index": i,
                                  "n_tokens": count_tokens(chunk)})
            stats["added"].append(source)
            print(f"  embed  {source} ({len(chunks)} chunks)")

    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embed(docs))
    stats["chunks"] = len(docs)

    stored = {m["source"] for m in collection.get(include=["metadatas"])["metadatas"]}
    for source in sorted(stored - current):
        collection.delete(where={"source": source})
        stats["removed"].append(source)
        print(f"  remove {source} (no longer in the knowledge base)")
    return stats


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (add it to .env)")
    print(f"Ingesting {', '.join(d.name for d in SOURCE_DIRS)} -> {CHROMA_DIR} [{COLLECTION_NAME}]")
    stats = ingest()
    total = get_collection().count()
    print(f"Done: {len(stats['added'])} embedded ({stats['chunks']} chunks), {len(stats['skipped'])} skipped, "
          f"{len(stats['removed'])} removed; collection has {total} chunks")


if __name__ == "__main__":
    main()
