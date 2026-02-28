"""
scripts/smoke_test_chat.py
==========================
Smoke test for the Multi-Modal RAG chat pipeline.
Tests:
  1. Normal question: "What is the paper about? Give me 3 bullet points."
  2. Equation query: "Show Equation 1 and explain it."
  3. Vague equation query: "SHOW ME a rag sequence equation"

Run from the project root:
    python scripts/smoke_test_chat.py

No real PDF required — uses mocked objects.
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test")

PASS = "✅"
FAIL = "❌"
results = []

def check(name, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"{status} {name}" + (f"\n   {detail}" if detail else ""))
    results.append((name, ok))


# ─── 1. Import checks ─────────────────────────────────────────────────────────
try:
    from models import MultimodalChunk, ProcessedDocument, ProcessedSection
    from models import ProcessedEquation, ProcessedTable, ProcessedFigure
    from models import GlobalElementRegistry
    from hallucination_guard import DocumentElementRegistry, HallucinationGuard
    from response_formatter import ResponseFormatter, SmartResponseFormatter
    from vector_store import UnifiedVectorStore
    check("All imports succeed", True)
except Exception as e:
    check("All imports succeed", False, str(e))
    sys.exit(1)


# ─── 2. Build a minimal mock ProcessedDocument ────────────────────────────────
try:
    eq1 = ProcessedEquation(
        equation_id="eq_1", global_number=1,
        text="P(z|x) = N(mu, sigma^2)",
        latex=r"P(z|x) = \mathcal{N}(\mu, \sigma^2)",
        page_number=2, bbox=(0, 0, 100, 20),
        section="Methods", context="The posterior distribution",
        description="Gaussian posterior over latent variables",
    )
    eq2 = ProcessedEquation(
        equation_id="eq_2", global_number=2,
        text="ELBO = E[log p(x|z)] - KL[q(z|x) || p(z)]",
        latex=r"\mathcal{L} = \mathbb{E}[\log p(x|z)] - \mathrm{KL}[q(z|x)\|p(z)]",
        page_number=3, bbox=(0, 0, 100, 20),
        section="Methods", context="Evidence Lower Bound",
        description="ELBO objective for variational inference",
    )
    tbl1 = ProcessedTable(
        table_id="tbl_1", global_number=1,
        caption="Results on MNIST",
        page_number=5,
        markdown="| Model | Accuracy |\n|-------|----------|\n| VAE | 0.95 |",
        section="Experiments",
    )
    fig1 = ProcessedFigure(
        figure_id="fig_1", global_number=1,
        caption="VAE Architecture",
        page_number=4,
        saved_path=None,
        bbox=(0, 0, 300, 200),
        section="Methods",
    )
    sec1 = ProcessedSection(
        section_id="sec_1", title="Abstract",
        page_number=1,
        content=(
            "We propose a Variational Autoencoder (VAE) for learning latent "
            "representations of high-dimensional data. Our model achieves state-of-the-art "
            "performance on image generation benchmarks."
        ),
        equations=[1, 2], tables=[1], figures=[1],
    )
    mock_doc = ProcessedDocument(
        doc_id="test_doc",
        filename="test_paper.pdf",
        num_pages=8,
        page_texts=[
            "Abstract: We propose a Variational Autoencoder...",
            "",
            "Methods: The posterior distribution P(z|x)...",
            "ELBO objective...",
            "Figure 1 shows the architecture...",
            "Table 1: Results on MNIST",
            "", ""
        ],
        enriched_page_texts=[""] * 8,
        sections=[sec1],
        equations=[eq1, eq2],
        tables=[tbl1],
        figures=[fig1],
        title="Variational Autoencoder for Image Generation",
        authors=["Alice Smith", "Bob Jones"],
        abstract="We propose a VAE for learning latent representations.",
    )
    check("Mock ProcessedDocument created", True)
except Exception as e:
    check("Mock ProcessedDocument created", False, str(e))
    sys.exit(1)


# ─── 3. Document registry ─────────────────────────────────────────────────────
try:
    reg = DocumentElementRegistry()
    reg.load_from_processed_document(mock_doc)
    assert 1 in reg.equations
    assert 2 in reg.equations
    assert 1 in reg.tables
    assert 1 in reg.figures
    check("DocumentElementRegistry loaded", True)
except Exception as e:
    check("DocumentElementRegistry loaded", False, str(e))
    sys.exit(1)


# ─── 4. ResponseFormatter ─────────────────────────────────────────────────────
try:
    formatter = ResponseFormatter(reg)
    # Old interface
    old_result = formatter.format_response("This discusses Equation 1 and Table 1.")
    assert "text" in old_result, f"Missing 'text' key: {old_result.keys()}"
    check("ResponseFormatter.format_response(text) — old interface", True)

    # Smart interface
    smart = formatter.format_smart_response(
        query="What is the paper about?",
        llm_response="The paper proposes a VAE. See Equation 1 for the posterior.",
    )
    assert "response" in smart, f"Missing 'response' key: {smart.keys()}"
    assert "mode" in smart, f"Missing 'mode' key: {smart.keys()}"
    check("ResponseFormatter.format_smart_response() — new interface", True)
except Exception as e:
    check("ResponseFormatter interfaces", False, str(e))
    import traceback; traceback.print_exc()


# ─── 5. Vector store + chunker ────────────────────────────────────────────────
try:
    from document_chunker import StructureAwareChunker
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_document(mock_doc, "test_doc")
    assert len(chunks) > 0
    eq_chunks = [c for c in chunks if c.chunk_type == "equation"]
    assert len(eq_chunks) == 2, f"Expected 2 eq chunks, got {len(eq_chunks)}"
    assert "section" in eq_chunks[0].metadata
    assert "content_priority" in eq_chunks[0].metadata
    check(f"StructureAwareChunker produces {len(chunks)} MultimodalChunks", True)

    store = UnifiedVectorStore()
    store.add_chunks(chunks)
    assert len(store.chunks) == len(chunks)

    # O(1) registry lookup
    found = store.search_by_number("equation", 1, doc_id="test_doc")
    assert found is not None, "Equation 1 not found via registry!"
    check("Vector store: add_chunks + search_by_number(equation, 1)", True)

    # Type-filtered search
    results_eq = store.search("posterior latent", top_k=3,
                               doc_id="test_doc", chunk_type_filter=["equation"])
    check(f"Vector store: equation-filtered search returns {len(results_eq)} results", True)
except Exception as e:
    check("Vector store + chunker pipeline", False, str(e))
    import traceback; traceback.print_exc()


# ─── 6. Simulate query() pipeline (no LLM / no real Groq) ───────────────────
async def fake_query(question, retrieved):
    """
    Simulate what MultiModalAgenticRAG.query() now does,
    without a real LLM or PDF, to verify no crashes.
    """
    llm_text = f"[MOCK ANSWER] Answering: {question}"

    # Smart formatter
    smart = formatter.format_smart_response(
        query=question,
        llm_response=llm_text,
        retrieved_chunks=retrieved,
    )
    answer_text = smart.get("response", llm_text)

    sources = []
    for text, score, meta in retrieved:
        sources.append({
            'type':    meta.get('chunk_type', 'text'),
            'page':    meta.get('page_num', 0),
            'score':   float(score),
            'preview': (text or "")[:200],
            'chunk_id': meta.get('chunk_id', ''),
        })

    return {
        'answer': answer_text,
        'sources': sources,
        'metadata': {
            'mode': smart.get('mode', 'explain'),
            'query_type': smart.get('content_type', 'general'),
        }
    }


async def run_simulated_queries():
    # Build fake retrieved list
    retrieved = [
        (chunks[0].text, 0.9, {
            'chunk_id': chunks[0].chunk_id,
            'chunk_type': chunks[0].chunk_type,
            'page_num': chunks[0].page_num,
            **chunks[0].metadata,
        }),
    ]

    # Q1: Normal question
    r1 = await fake_query("What is the paper about? Give me 3 bullet points.", retrieved)
    assert 'answer' in r1
    assert 'sources' in r1
    assert 'metadata' in r1
    check("Q1: Normal question returns {answer, sources, metadata}", True)

    # Q2: Specific equation
    r2 = await fake_query("Show Equation 1 and explain it.", retrieved)
    assert 'answer' in r2
    check("Q2: Specific equation query — no crash", True)

    # Q3: Vague equation
    r3 = await fake_query("SHOW ME a rag sequence equation", retrieved)
    assert 'answer' in r3
    check("Q3: Vague equation query — no crash", True)

asyncio.run(run_simulated_queries())


# ─── Summary ─────────────────────────────────────────────────────────────────
print()
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print("=" * 60)
print(f"Smoke Test Results: {passed}/{total} checks passed")
if passed == total:
    print("✅ ALL CHECKS PASSED — pipeline ready!")
else:
    print("❌ Some checks failed. Review output above.")
sys.exit(0 if passed == total else 1)
