import asyncio
from types import SimpleNamespace

import chromadb
import pytest

from services import rag_service
from utils import embed_knowledge_base as ekb

# Captured at import, before conftest's autouse stub replaces it for each test.
REAL_RETRIEVE_INGREDIENT_CONTEXT = rag_service.retrieve_ingredient_context


def _norm(text):
    return " ".join(text.split())


# --- Chunking ---------------------------------------------------------------------------------
def test_chunks_stay_within_limit_and_keep_all_text():
    text = (ekb.KB_DIR / "ingredients" / "niacinamide.txt").read_text(encoding="utf-8")
    chunks = ekb.chunk_text(text, max_tokens=120)
    assert len(chunks) > 1
    assert all(ekb.count_tokens(c) <= 120 for c in chunks)
    assert _norm(" ".join(chunks)) == _norm(text)


def test_small_paragraphs_are_packed_together():
    chunks = ekb.chunk_text("one\n\ntwo\n\nthree", max_tokens=50)
    assert chunks == ["one\n\ntwo\n\nthree"]


def test_unsectioned_doc_gets_name_only_prefix():
    assert ekb.chunk_text("one\n\ntwo", name="retinol") == ["[retinol]\none\n\ntwo"]


def test_oversized_paragraph_splits_on_single_newlines():
    paragraph = "\n".join(f"line {i} " + "word " * 20 for i in range(10))
    chunks = ekb.chunk_text(paragraph, max_tokens=60)
    assert len(chunks) > 1
    assert all(ekb.count_tokens(c) <= 60 for c in chunks)
    assert all(c.startswith("line ") for c in chunks)  # breaks fall on line boundaries


def test_single_huge_line_is_hard_split():
    chunks = ekb.chunk_text("word " * 500, max_tokens=100)
    assert all(ekb.count_tokens(c) <= 100 for c in chunks)
    assert _norm("".join(chunks)) == _norm("word " * 500)


SECTIONED = """SKINSAGE KNOWLEDGE BASE
TOPIC: Test
================================================================================
SECTION 1: MECHANISM
================================================================================

How it works.

Second paragraph.

--------------------------------------------------------------------------------
SECTION 2: TIMELINES
--------------------------------------------------------------------------------
SUBHEAD
-------
Weeks 4-8.
"""


def test_split_sections_finds_headings_and_drops_rules():
    sections = ekb.split_sections(SECTIONED)
    assert [h for h, _ in sections] == [ekb.HEADER_SECTION, "SECTION 1: MECHANISM", "SECTION 2: TIMELINES"]
    assert sections[0][1] == "SKINSAGE KNOWLEDGE BASE\nTOPIC: Test"
    assert sections[2][1] == "SUBHEAD\nWeeks 4-8."  # a lone underline rule is dropped, not a heading
    assert not any("=====" in b or "-----" in b for _, b in sections)


def test_chunks_never_span_sections_and_carry_prefix():
    assert ekb.chunk_text(SECTIONED, name="test doc") == [
        "[test doc — Document header]\nSKINSAGE KNOWLEDGE BASE\nTOPIC: Test",
        "[test doc — SECTION 1: MECHANISM]\nHow it works.\n\nSecond paragraph.",
        "[test doc — SECTION 2: TIMELINES]\nSUBHEAD\nWeeks 4-8.",
    ]


def test_doc_name_strips_download_suffix():
    assert ekb.doc_name("niacinamide_knowledge_base (1).txt") == "niacinamide knowledge base"
    assert ekb.doc_name("retinol.txt") == "retinol"


def test_real_progress_doc_chunks():
    path = ekb.KB_DIR / "ingredients_progress_kb" / "niacinamide_knowledge_base (1).txt"
    if not path.exists():
        pytest.skip("progress KB not present")
    text = path.read_text(encoding="utf-8")
    name = ekb.doc_name(path.name)
    sections = ekb.split_sections(text)
    headings = {h for h, _ in sections}
    assert len(headings) > 5
    chunks = ekb.chunk_text(text, name=name)
    for chunk in chunks:
        assert ekb.count_tokens(chunk) <= ekb.MAX_TOKENS
        first, _, body = chunk.partition("\n")
        assert first.startswith(f"[{name} — ") and first[len(name) + 4:-1] in headings
        assert "=====" not in body
    # Nothing lost apart from rule lines, the heading lines (which move into the prefixes) and the
    # ruled "END OF DOCUMENT" banner (a heading with no body, so no section).
    dropped = headings | {"END OF DOCUMENT"}
    kept = [l for l in text.splitlines() if l.strip() and not ekb._is_rule(l) and l.strip() not in dropped]
    assert _norm(" ".join(kept)) == _norm(" ".join(c.partition("\n")[2] for c in chunks))


# --- Ingestion --------------------------------------------------------------------------------
class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.5] for t in texts]


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    name = f"test_{id(object())}"
    yield client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    client.delete_collection(name)


def test_rerun_skips_unchanged_reembeds_changed_and_removes_deleted(tmp_path, collection):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.txt").write_text("Alpha\n\nmore alpha", encoding="utf-8")
    (kb / "b.txt").write_text("Beta", encoding="utf-8")
    embed = FakeEmbedder()

    stats = ekb.ingest([kb], collection, embed)
    assert stats["added"] == ["kb/a.txt", "kb/b.txt"] and len(embed.calls) == 1
    metas = collection.get()["metadatas"]
    assert {m["source"] for m in metas} == {"kb/a.txt", "kb/b.txt"}
    assert all(m["folder"] == "kb" and m["chunker_version"] == ekb.CHUNKER_VERSION for m in metas)

    stats = ekb.ingest([kb], collection, embed)
    assert stats["skipped"] == ["kb/a.txt", "kb/b.txt"] and stats["added"] == []
    assert len(embed.calls) == 1  # nothing re-embedded

    (kb / "a.txt").write_text("Alpha, edited", encoding="utf-8")
    stats = ekb.ingest([kb], collection, embed)
    assert stats["added"] == ["kb/a.txt"] and stats["skipped"] == ["kb/b.txt"]
    assert embed.calls[-1] == ["[a]\nAlpha, edited"]
    assert collection.get(where={"source": "kb/a.txt"})["documents"] == ["[a]\nAlpha, edited"]

    (kb / "b.txt").unlink()
    stats = ekb.ingest([kb], collection, embed)
    assert stats["removed"] == ["kb/b.txt"]
    assert collection.get(where={"source": "kb/b.txt"})["ids"] == []


def test_two_folders_same_filename_do_not_collide(tmp_path, collection):
    for folder in ("one", "two"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "same.txt").write_text(f"from {folder}", encoding="utf-8")
    ekb.ingest([tmp_path / "one", tmp_path / "two"], collection, FakeEmbedder())
    assert sorted(m["source"] for m in collection.get()["metadatas"]) == ["one/same.txt", "two/same.txt"]


def test_old_chunker_version_is_reembedded(tmp_path, collection):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.txt").write_text("Alpha", encoding="utf-8")
    collection.add(ids=["old"], documents=["Alpha"], embeddings=[[1.0, 1.0, 1.0]],
                   metadatas=[{"source": "kb/a.txt", "file_hash": ekb.file_hash(kb / "a.txt"), "chunker_version": 1}])
    stats = ekb.ingest([kb], collection, FakeEmbedder())
    assert stats["added"] == ["kb/a.txt"]
    assert collection.get(where={"source": "kb/a.txt"})["ids"] == ["kb/a:0"]


def test_embed_batches_in_groups_of_100():
    sizes = []

    class Client:
        class embeddings:
            @staticmethod
            def create(model, input):
                assert model == "text-embedding-ada-002"
                sizes.append(len(input))
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0]) for _ in input])

    vectors = ekb.embed_batches([f"t{i}" for i in range(250)], client=Client())
    assert sizes == [100, 100, 50] and len(vectors) == 250


# --- Retrieval --------------------------------------------------------------------------------
def test_retrieve_context_returns_top_documents(collection, monkeypatch):
    collection.add(ids=["x:0", "y:0"], documents=["niacinamide fades marks", "spf daily"],
                   embeddings=[[1.0, 0.0], [0.0, 1.0]], metadatas=[{"source": "x"}, {"source": "y"}])

    class FakeAsyncOpenAI:
        class embeddings:
            @staticmethod
            async def create(model, input):
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.9, 0.1])])

    monkeypatch.setattr(rag_service, "_ingredient_collection", collection)
    monkeypatch.setattr(rag_service, "_async_openai", FakeAsyncOpenAI())
    docs = asyncio.run(rag_service.retrieve_context("dark marks", n_results=5))
    assert docs == ["niacinamide fades marks", "spf daily"]


def test_sync_retrieval_returns_empty_without_collection(collection, monkeypatch):
    # The real function: conftest's autouse fixture stubs it out for every test.
    monkeypatch.setattr(rag_service, "_ingredient_collection", collection)
    monkeypatch.setattr(rag_service, "_embed", lambda texts: (_ for _ in ()).throw(AssertionError("no embed call")))
    assert REAL_RETRIEVE_INGREDIENT_CONTEXT("dark marks") == []


def test_sync_retrieval_returns_top_chunks(collection, monkeypatch):
    collection.add(ids=["x:0", "y:0"], documents=["niacinamide fades marks", "spf daily"],
                   embeddings=[[1.0, 0.0], [0.0, 1.0]], metadatas=[{"source": "x"}, {"source": "y"}])
    monkeypatch.setattr(rag_service, "_ingredient_collection", collection)
    monkeypatch.setattr(rag_service, "_embed", lambda texts: [[0.9, 0.1]])
    assert REAL_RETRIEVE_INGREDIENT_CONTEXT("dark marks", n_results=1) == ["niacinamide fades marks"]
