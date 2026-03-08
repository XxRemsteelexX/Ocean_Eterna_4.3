"""test OE Double Helix — category classification, storage, and search."""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from classifier import classify, classify_query
from category_store import CategoryStore, Chunk

# ── Classifier Tests ──────────────────────────────────────────────────

def test_classify_coding():
    text = "def hello_world():\n    print('hello')\n    return True"
    assert classify(text) == "coding", f"expected coding, got {classify(text)}"

    text = "The Python function uses async await to handle HTTP requests via the API endpoint"
    assert classify(text) == "coding"

def test_classify_chat():
    text = "[user]: Can you help me with this?\n[assistant]: Sure, let me look into that for you.\n[user]: Thanks, the conversation has been really helpful.\nSession tokens: 15000, context window almost full."
    assert classify(text) == "chat_logs", f"expected chat_logs, got {classify(text)}"

def test_classify_work():
    text = "Meeting notes from Q4 sprint planning. Client deliverables due next milestone. Revenue target discussion."
    assert classify(text) == "work", f"expected work, got {classify(text)}"

def test_classify_personal():
    text = "Today I went for a workout and felt great. My goal is to exercise every morning. Family birthday dinner tonight."
    assert classify(text) == "personal", f"expected personal, got {classify(text)}"

def test_classify_documents():
    text = "Section 3 of the research paper discusses methodology. See Figure 1 and Table 2 in the appendix."
    assert classify(text) == "documents", f"expected documents, got {classify(text)}"

def test_classify_by_filename():
    text = "some generic content"
    assert classify(text, "script.py") == "coding"
    assert classify(text, "chat_log_march.txt") == "chat_logs"

def test_classify_query():
    cats = classify_query("python function async")
    assert cats[0] == "coding"

    cats = classify_query("meeting notes Q4 budget")
    assert cats[0] == "work"

# ── Category Store Tests ──────────────────────────────────────────────

def test_store_add_and_search():
    tmpdir = tempfile.mkdtemp()
    try:
        store = CategoryStore(tmpdir)

        # add a coding chunk
        chunk = Chunk(
            chunk_id="test_001",
            source_file="test.py",
            text="The quicksort algorithm sorts an array by partitioning it into subarrays",
            keywords=["quicksort", "algorithm", "sort", "array", "partition", "subarrays"],
            token_count=15,
            category="coding",
        )
        store.add_chunk("coding", chunk)

        # search should find it
        results = store.search_bm25("coding", "quicksort algorithm")
        assert len(results) > 0, "should find the chunk"
        assert results[0]["chunk_id"] == "test_001"

        # search wrong category should not find it
        results = store.search_bm25("personal", "quicksort algorithm")
        assert len(results) == 0, "should not find in wrong category"

    finally:
        shutil.rmtree(tmpdir)

def test_store_persistence():
    tmpdir = tempfile.mkdtemp()
    try:
        # create and save
        store = CategoryStore(tmpdir)
        chunk = Chunk(
            chunk_id="persist_001",
            source_file="notes.txt",
            text="Machine learning uses neural networks for classification tasks",
            keywords=["machine", "learning", "neural", "networks", "classification"],
            token_count=12,
            category="coding",
        )
        store.add_chunk("coding", chunk)
        store.save_all()

        # reload from disk
        store2 = CategoryStore(tmpdir)
        results = store2.search_bm25("coding", "neural networks machine learning")
        assert len(results) > 0, "should find after reload"
        assert results[0]["chunk_id"] == "persist_001"

    finally:
        shutil.rmtree(tmpdir)

def test_store_lazy_loading():
    tmpdir = tempfile.mkdtemp()
    try:
        store = CategoryStore(tmpdir)

        # add to two categories
        store.add_chunk("coding", Chunk(
            chunk_id="code_001", source_file="a.py",
            text="Python function for data processing",
            keywords=["python", "function", "data", "processing"],
            token_count=6, category="coding",
        ))
        store.save_category("coding")

        store.add_chunk("personal", Chunk(
            chunk_id="pers_001", source_file="journal.txt",
            text="Today I went hiking in the mountains",
            keywords=["hiking", "mountains", "today"],
            token_count=8, category="personal",
        ))
        store.save_category("personal")

        # active should be personal (last loaded)
        assert store.active_category == "personal"

        # search coding — should swap
        results = store.search_bm25("coding", "python function")
        assert store.active_category == "coding"
        assert len(results) > 0

        # personal should be unloaded
        assert not store.indexes["personal"].loaded

    finally:
        shutil.rmtree(tmpdir)

def test_store_cross_category_search():
    tmpdir = tempfile.mkdtemp()
    try:
        store = CategoryStore(tmpdir)

        store.add_chunk("coding", Chunk(
            chunk_id="c1", source_file="a.py",
            text="Python machine learning tensorflow keras neural network training",
            keywords=["python", "machine", "learning", "tensorflow", "keras", "neural", "network", "training"],
            token_count=10, category="coding",
        ))
        store.save_category("coding")

        store.add_chunk("documents", Chunk(
            chunk_id="d1", source_file="paper.pdf",
            text="Research paper on machine learning applications in healthcare neural network",
            keywords=["research", "paper", "machine", "learning", "healthcare", "neural", "network"],
            token_count=12, category="documents",
        ))
        store.save_category("documents")

        # cross-category search
        results = store.search_all_categories("machine learning neural network")
        assert len(results) >= 2, f"expected 2+ results, got {len(results)}"

    finally:
        shutil.rmtree(tmpdir)


# ── Run Tests ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_classify_coding,
        test_classify_chat,
        test_classify_work,
        test_classify_personal,
        test_classify_documents,
        test_classify_by_filename,
        test_classify_query,
        test_store_add_and_search,
        test_store_persistence,
        test_store_lazy_loading,
        test_store_cross_category_search,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} — {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
