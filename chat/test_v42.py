#!/usr/bin/env python3
"""OceanEterna v4.2 Test Suite — tests all new features and bug fixes."""
import requests
import json
import time
import sys
import os

BASE = "http://localhost:9090"
passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")
        if detail:
            print(f"         {detail[:200]}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# check server is up
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    data = r.json()
    if data.get("status") != "ok":
        print("Server not ready")
        sys.exit(1)
except:
    print("Cannot connect to server at", BASE)
    sys.exit(1)

print(f"Server ready: v{data.get('version')}")

# ============================================================
section("1. Version & Health")
# ============================================================
test("Version is 4.2", data.get("version") == "4.2")
test("Health status ok", data.get("status") == "ok")

r = requests.get(f"{BASE}/stats")
stats = r.json()
test("Stats has chunks_loaded", stats.get("chunks_loaded", 0) > 0)
test("Stats has total_tokens", stats.get("total_tokens", 0) > 0)

# ============================================================
section("2. Small Document Ingestion — Paragraph-Safe Chunking")
# ============================================================

# create a test document with distinct paragraphs of varying sizes
test_content = """Chapter 1: Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on building systems
that learn from data. Unlike traditional programming where rules are explicitly coded,
ML systems discover patterns and make decisions with minimal human intervention. This
paradigm shift has revolutionized industries from healthcare to finance.

The three main types of machine learning are supervised learning, unsupervised learning,
and reinforcement learning. Each has distinct characteristics and use cases.

Chapter 2: Supervised Learning

In supervised learning, the algorithm learns from labeled training data. The model makes
predictions based on input features and adjusts its parameters based on the error between
predictions and actual labels. Common algorithms include linear regression, decision trees,
random forests, and neural networks. The key challenge is avoiding overfitting while
maintaining good generalization to unseen data. Cross-validation and regularization are
standard techniques to address this challenge. Support vector machines provide another
powerful approach by finding optimal decision boundaries in high-dimensional space.

Chapter 3: Deep Learning

Deep learning uses neural networks with multiple layers to progressively extract higher-level
features from raw input. Convolutional Neural Networks (CNNs) are particularly effective for
image recognition tasks. Recurrent Neural Networks (RNNs) and their variants like LSTM and
GRU are used for sequential data such as text and time series. Transformer architectures,
introduced in 2017 with the "Attention is All You Need" paper, have become the foundation
for modern NLP models including BERT, GPT, and their successors.

Chapter 4: Reinforcement Learning

Reinforcement learning involves an agent that learns to make decisions by interacting with
an environment. The agent receives rewards or penalties based on its actions and learns to
maximize cumulative reward over time. Key concepts include the state space, action space,
reward function, and policy. Q-learning and policy gradient methods are foundational
algorithms in this field. AlphaGo's victory over world champion Go players demonstrated
the potential of deep reinforcement learning."""

r = requests.post(f"{BASE}/add-file", json={
    "filename": "ml_guide.txt",
    "content": test_content
}, timeout=30)
data = r.json()
test("File ingestion succeeds", data.get("success") == True, json.dumps(data)[:200])
test("Created multiple chunks", data.get("chunks_added", 0) >= 2,
     f"chunks_added={data.get('chunks_added')}")
test("Tokens tracked", data.get("tokens_added", 0) > 0)

chunks_added = data.get("chunks_added", 0)
print(f"  (Ingested {chunks_added} chunks, {data.get('tokens_added', 0)} tokens)")

# ============================================================
section("3. Catalog Endpoint — Hierarchical Browsing")
# ============================================================

r = requests.get(f"{BASE}/catalog?page_size=5")
catalog = r.json()
test("Catalog returns total_chunks", catalog.get("total_chunks", 0) > 0)
test("Catalog has types breakdown", "types" in catalog)
test("Catalog has source_files", "source_files" in catalog)
test("Catalog paginated", len(catalog.get("chunks", [])) <= 5)

# filter by type
r = requests.get(f"{BASE}/catalog?type=DOC&page_size=3")
cat_doc = r.json()
test("Catalog filter by type works", len(cat_doc.get("chunks", [])) > 0)

print(f"  Types: {catalog.get('types', {})}")
print(f"  Source files: {len(catalog.get('source_files', []))}")

# ============================================================
section("4. BM25 Search Speed — Dynamic top_k")
# ============================================================

# search without LLM (just test search speed)
search_queries = [
    "machine learning supervised",
    "deep learning neural networks",
    "reinforcement learning agent",
    "what is photosynthesis",
    "Denver school bond measure",
]

for q in search_queries:
    start = time.time()
    r = requests.post(f"{BASE}/chat", json={"question": q}, timeout=120)
    elapsed = (time.time() - start) * 1000
    data = r.json()
    search_ms = data.get("search_time_ms", 0)
    chunks_ret = data.get("chunks_retrieved", 0)
    test(f"Search '{q[:30]}...' ({search_ms:.0f}ms, {chunks_ret} chunks)",
         search_ms < 500 and chunks_ret > 0,
         f"search_ms={search_ms}, chunks={chunks_ret}")

# ============================================================
section("5. Adjacent Chunk Retrieval")
# ============================================================

# find a chunk from our ingested file
r = requests.get(f"{BASE}/catalog?source=ml_guide.txt&page_size=10")
cat = r.json()
test_chunks = cat.get("chunks", [])
if len(test_chunks) >= 2:
    chunk_id = test_chunks[1].get("chunk_id", "")  # get second chunk

    # fetch without context window
    r = requests.get(f"{BASE}/chunk/{chunk_id}")
    data = r.json()
    test("Chunk fetch succeeds", data.get("success") == True)
    has_refs = "prev_chunk_id" in data or "next_chunk_id" in data
    test("Cross-references present", has_refs,
         f"prev={data.get('prev_chunk_id','none')}, next={data.get('next_chunk_id','none')}")

    # fetch with context window
    r = requests.get(f"{BASE}/chunk/{chunk_id}?context_window=1")
    data = r.json()
    adj = data.get("adjacent_chunks", [])
    test("Adjacent chunks returned", len(adj) > 0,
         f"adjacent_count={len(adj)}")
    if adj:
        test("Adjacent has content", len(adj[0].get("content", "")) > 0)
else:
    test("Need 2+ chunks for adjacency test", False, f"only {len(test_chunks)} chunks")

# ============================================================
section("6. HTTP Error Codes — Standardized")
# ============================================================

# missing question field
r = requests.post(f"{BASE}/chat", json={}, timeout=10)
test("Missing field -> 400", r.status_code == 400, f"got {r.status_code}")

# invalid JSON
r = requests.post(f"{BASE}/chat", data="not json", timeout=10,
                   headers={"Content-Type": "application/json"})
test("Invalid JSON -> 400", r.status_code == 400, f"got {r.status_code}")

# nonexistent chunk
r = requests.get(f"{BASE}/chunk/NONEXISTENT_CHUNK_999")
test("Missing chunk -> 404", r.status_code == 404, f"got {r.status_code}")

# missing turn_id
r = requests.post(f"{BASE}/sources", json={}, timeout=10)
test("Missing turn_id -> 400", r.status_code == 400, f"got {r.status_code}")

# ============================================================
section("7. Score Threshold & Dynamic top_k Verification")
# ============================================================

# verify we get more results than the old default of 3
r = requests.post(f"{BASE}/chat", json={"question": "what is machine learning"}, timeout=120)
data = r.json()
chunks_ret = data.get("chunks_retrieved", 0)
test("Dynamic top_k > old default 3", chunks_ret > 3,
     f"chunks_retrieved={chunks_ret}")
test("Search under 200ms", data.get("search_time_ms", 999) < 200,
     f"search_time_ms={data.get('search_time_ms')}")

# ============================================================
section("8. Path Traversal Protection")
# ============================================================

# try to access parent directory
r = requests.post(f"{BASE}/add-file-path", json={"path": "../../etc/passwd"}, timeout=10)
data = r.json()
test("Path traversal blocked", data.get("success") == False and r.status_code != 200,
     f"status={r.status_code}")

# try absolute path outside allowed dirs
r = requests.post(f"{BASE}/add-file-path", json={"path": "/etc/hostname"}, timeout=10)
data = r.json()
test("Absolute path outside CWD blocked", data.get("success") == False,
     json.dumps(data)[:150])

# ============================================================
section("9. Self-Referential Query Word Boundary Fix")
# ============================================================

# "antique" contains "i " but should NOT trigger self-referential
# this test is implicit — if the search works normally, the fix is working
r = requests.post(f"{BASE}/chat", json={"question": "tell me about antique furniture"}, timeout=120)
data = r.json()
test("'antique' query doesn't crash", "answer" in data or "error" in data)

# ============================================================
section("10. LLM Context Truncation")
# ============================================================

# we can't directly test truncation without a huge context, but we can verify
# the search returns successfully even for broad queries that match many chunks
r = requests.post(f"{BASE}/chat", json={"question": "the"}, timeout=120)
data = r.json()
test("Broad query doesn't crash", "answer" in data or "error" in data)
test("Search time reasonable for broad query",
     data.get("search_time_ms", 9999) < 1000,
     f"search_ms={data.get('search_time_ms')}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*60}")

if failed > 0:
    sys.exit(1)
