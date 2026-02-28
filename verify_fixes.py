"""
verify_fixes.py - Quick verification script for the V16 bug fixes.
Run with:  python verify_fixes.py
"""

import sys, traceback

PASS = "\u2705"
FAIL = "\u274c"
results = []

def check(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True))
    except Exception as e:
        print(f"{FAIL} {name}")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        results.append((name, False))


# ─── Test 1: models.py ────────────────────────────────────────────────────────
def test_models():
    from models import MultimodalChunk, SearchResult, GlobalElementRegistry
    chunk = MultimodalChunk(
        chunk_id="test_001", text="Test content", doc_id="doc1",
        page_num=0, chunk_type="text",
        metadata={"section": "Intro", "content_priority": 1.0}
    )
    assert chunk.metadata["section"] == "Intro"
    assert chunk.metadata.get("content_priority") == 1.0
    assert chunk.page_number == 0          # backward-compat alias
    # SearchResult is iterable
    r = SearchResult(chunk=chunk, similarity_score=0.9, rank=0)
    ch, sc = r
    assert ch is chunk

check("models.py — MultimodalChunk & SearchResult", test_models)


# ─── Test 2: vector_store.py ──────────────────────────────────────────────────
def test_vector_store():
    from vector_store import UnifiedVectorStore
    from models import MultimodalChunk
    store = UnifiedVectorStore()
    assert hasattr(store, "add_chunks"), "add_chunks() is missing!"
    assert not hasattr(store, "add_documents"), "add_documents() should NOT exist"
    chunk = MultimodalChunk(
        chunk_id="v_001", text="hello world", doc_id="doc1",
        page_num=0, chunk_type="text",
        metadata={"section": "Intro", "content_priority": 1.0}
    )
    store.add_chunks([chunk])
    results = store.search("hello", top_k=1)
    assert len(results) == 1

check("vector_store.py — UnifiedVectorStore.add_chunks()", test_vector_store)


# ─── Test 3: document_chunker.py ──────────────────────────────────────────────
def test_chunker_no_old_class():
    import document_chunker as dc
    assert not hasattr(dc, "Chunk"), \
        "Old local Chunk class must be removed from document_chunker.py!"
    from models import MultimodalChunk
    assert hasattr(dc, "StructureAwareChunker")
    assert hasattr(dc, "ContentTypePriorityBooster")
    assert hasattr(dc, "build_multimodal_chunks")

check("document_chunker.py — no old Chunk class", test_chunker_no_old_class)


def test_boost():
    from document_chunker import ContentTypePriorityBooster
    from models import MultimodalChunk
    chunks = [
        MultimodalChunk("c1", "eq text", "d1", 0, "equation",
                        {"content_priority": 2.0, "section": "A"}),
        MultimodalChunk("c2", "normal text", "d1", 0, "text",
                        {"content_priority": 1.0, "section": "A"}),
    ]
    boosted = ContentTypePriorityBooster.boost_chunks_by_query(chunks, "what is the equation?")
    eq_chunk = next(c for c in boosted if c.chunk_type == "equation")
    assert eq_chunk.metadata["content_priority"] > 2.0, "Boost must increase priority"

check("document_chunker.py — ContentTypePriorityBooster reads from metadata", test_boost)


# ─── Test 4: hybrid_retrieval.py ──────────────────────────────────────────────
def test_retrieval_import():
    from hybrid_retrieval import ContentTypePriorityRetriever, SmartRetriever
    from vector_store import UnifiedVectorStore
    from models import MultimodalChunk
    store = UnifiedVectorStore()
    chunk = MultimodalChunk(
        chunk_id="r_001", text="neural network loss equation", doc_id="d1",
        page_num=0, chunk_type="equation",
        metadata={"section": "Methods", "content_priority": 2.0}
    )
    store.add_chunks([chunk])
    retriever = SmartRetriever(vector_store=store, enable_bm25=False, enable_reranker=False)
    retriever.index([chunk])
    results = retriever.search("equation", top_k=1)
    # results is List[Tuple[str, float, Dict]]
    assert isinstance(results, list)
    if results:
        text, score, metadata = results[0]
        assert isinstance(text, str)
        assert isinstance(score, float)
        assert isinstance(metadata, dict)
        assert "chunk_id" in metadata

check("hybrid_retrieval.py — SmartRetriever returns (text, score, metadata)", test_retrieval_import)


# ─── Test 5: multimodal_agentic_rag.py imports ────────────────────────────────
def test_rag_import():
    from multimodal_agentic_rag import MultiModalAgenticRAG, MultiModalRAGConfig
    config = MultiModalRAGConfig()   # no API key needed for import test
    # Don't fully init (avoids Groq call) — just check class exists
    assert MultiModalAgenticRAG is not None

check("multimodal_agentic_rag.py — imports without error", test_rag_import)


# ─── Summary ──────────────────────────────────────────────────────────────────
print()
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print("=" * 55)
print(f"Results: {passed}/{total} checks passed")
if passed == total:
    print("\u2705 All checks passed — system ready!")
else:
    print("\u274c Some checks failed. Review output above.")
sys.exit(0 if passed == total else 1)
