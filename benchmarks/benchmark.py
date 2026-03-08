#!/usr/bin/env python3
"""comprehensive ocean eterna benchmark suite.
tests against isolated test server on port 9191 — does NOT touch production.
uses /chat endpoint (the only search endpoint) with fast-fail LLM config."""

import requests, time, json, os, sys, subprocess, signal, statistics, glob
import concurrent.futures
from pathlib import Path

BASE_URL = "http://localhost:9191"
RESULTS_DIR = os.path.expanduser("~/oe-benchmark-test/results")
TEST_DATA_DIR = os.path.expanduser("~/oe-benchmark-test/test_data")
SERVER_DIR = os.path.expanduser("~/oe-benchmark-test/chat")
SERVER_BIN = os.path.join(SERVER_DIR, "ocean_test_server")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── helper functions ───────────────────────────────────────────────

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def get_memory_usage(pid):
    """get RSS memory in MB for a process."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # KB -> MB
    except:
        pass
    return 0

def wait_for_server(timeout=60):
    """wait for server to respond on health endpoint."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def start_server():
    """start the test server and return process handle."""
    log("starting test server on port 9191...")
    corpus_dir = os.path.join(SERVER_DIR, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    for f in glob.glob(os.path.join(corpus_dir, "*")):
        if os.path.isfile(f):
            os.remove(f)

    proc = subprocess.Popen(
        [SERVER_BIN, "9191"],
        cwd=SERVER_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    time.sleep(2)
    if wait_for_server(30):
        log(f"server started (pid={proc.pid})")
        return proc
    else:
        log("ERROR: server failed to start!")
        proc.kill()
        return None

def start_server_keep_corpus():
    """start the test server WITHOUT clearing corpus (for restart-after-ingest)."""
    log("starting test server (keeping corpus)...")
    proc = subprocess.Popen(
        [SERVER_BIN, "9191"],
        cwd=SERVER_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    time.sleep(2)
    if wait_for_server(30):
        log(f"server started (pid={proc.pid})")
        return proc
    else:
        log("ERROR: server failed to start!")
        proc.kill()
        return None

def stop_server(proc):
    """gracefully stop the server."""
    if proc and proc.poll() is None:
        log(f"stopping server (pid={proc.pid})...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except:
            proc.kill()
        log("server stopped")

def do_chat(question, timeout=10):
    """perform a /chat query. mock LLM responds instantly.
    returns (response, elapsed_ms)."""
    start = time.time()
    try:
        r = requests.post(f"{BASE_URL}/chat", json={"question": question}, timeout=timeout)
        elapsed = (time.time() - start) * 1000
        return r, elapsed
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return None, elapsed

def ingest_content(filename, content, timeout=60):
    """ingest content via /add-file and return (response, elapsed_ms).
    uses longer timeout to handle index lock contention under load."""
    start = time.time()
    try:
        r = requests.post(f"{BASE_URL}/add-file",
                          json={"filename": filename, "content": content}, timeout=timeout)
        elapsed = (time.time() - start) * 1000
        return r, elapsed
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return None, elapsed

def get_stats():
    """get server stats."""
    try:
        r = requests.get(f"{BASE_URL}/stats", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except:
        return {}

def extract_search_metrics(response):
    """extract BM25 search metrics from /chat response.
    even when LLM fails, search_time_ms and sources are populated."""
    if not response:
        return None
    try:
        data = response.json()
        return {
            "search_time_ms": data.get("search_time_ms", -1),
            "chunks_retrieved": data.get("chunks_retrieved", 0),
            "sources": data.get("sources", []),
            "llm_time_ms": data.get("llm_time_ms", -1),
            "total_time_ms": data.get("total_time_ms", -1),
            "answer": data.get("answer", ""),
            "creature_tier": data.get("creature_tier", ""),
        }
    except:
        return None


# ─── benchmark tests ────────────────────────────────────────────────

class BenchmarkResults:
    def __init__(self):
        self.sections = {}

    def add(self, section, key, value):
        if section not in self.sections:
            self.sections[section] = {}
        self.sections[section][key] = value

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.sections, f, indent=2, default=str)

    def summary(self):
        lines = []
        for section, data in self.sections.items():
            lines.append(f"\n{'='*60}")
            lines.append(f"  {section}")
            lines.append(f"{'='*60}")
            for k, v in data.items():
                if isinstance(v, dict) and len(str(v)) > 200:
                    lines.append(f"  {k}: [complex data - see JSON]")
                elif isinstance(v, list) and len(str(v)) > 200:
                    lines.append(f"  {k}: [{len(v)} items - see JSON]")
                else:
                    lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def bench_server_startup(results):
    """measure cold start time with empty corpus."""
    log("=== BENCHMARK: server cold start ===")
    stats = get_stats()
    results.add("01_startup", "health_check", "PASS" if stats else "FAIL")
    results.add("01_startup", "initial_stats", stats)


def bench_ingestion(results, corpus_size_name, proc, max_docs=None, label_override=None):
    """measure document ingestion speed at different scales."""
    label = label_override or corpus_size_name
    log(f"=== BENCHMARK: ingestion ({label}) ===")

    data_dir = os.path.join(TEST_DATA_DIR, corpus_size_name)
    if not os.path.exists(data_dir):
        log(f"  skipping {corpus_size_name} — data not found")
        return

    files = sorted(glob.glob(os.path.join(data_dir, "doc_*.txt")))
    if max_docs:
        files = files[:max_docs]
    if not files:
        log(f"  no docs found in {data_dir}")
        return

    total_bytes = sum(os.path.getsize(f) for f in files)
    section = f"02_ingestion_{label}"

    results.add(section, "doc_count", len(files))
    results.add(section, "total_bytes", total_bytes)
    results.add(section, "total_mb", round(total_bytes / 1024 / 1024, 2))

    mem_before = get_memory_usage(proc.pid)
    results.add(section, "memory_before_mb", round(mem_before, 1))

    ingest_times = []
    errors = 0
    chunks_total = 0
    start_total = time.time()

    for i, fpath in enumerate(files):
        with open(fpath) as f:
            content = f.read()
        fname = os.path.basename(fpath)
        r, elapsed = ingest_content(fname, content)
        if r and r.status_code == 200:
            ingest_times.append(elapsed)
            try:
                chunks_total += r.json().get("chunks_added", 0)
            except:
                pass
        else:
            errors += 1
            if errors <= 3:
                status = r.status_code if r else "no response"
                log(f"  ingest error on {fname}: {status}")

        if (i + 1) % 500 == 0:
            log(f"  ingested {i+1}/{len(files)}...")

    total_time = time.time() - start_total
    mem_after = get_memory_usage(proc.pid)

    results.add(section, "total_ingest_time_sec", round(total_time, 2))
    results.add(section, "docs_per_second", round(len(files) / total_time, 1) if total_time > 0 else 0)
    results.add(section, "mb_per_second", round((total_bytes / 1024 / 1024) / total_time, 2) if total_time > 0 else 0)
    results.add(section, "total_chunks_created", chunks_total)
    results.add(section, "errors", errors)
    results.add(section, "memory_after_mb", round(mem_after, 1))
    results.add(section, "memory_delta_mb", round(mem_after - mem_before, 1))

    if ingest_times:
        results.add(section, "ingest_latency_p50_ms", round(statistics.median(ingest_times), 2))
        results.add(section, "ingest_latency_p95_ms", round(sorted(ingest_times)[int(len(ingest_times) * 0.95)], 2))
        results.add(section, "ingest_latency_p99_ms", round(sorted(ingest_times)[int(len(ingest_times) * 0.99)], 2))
        results.add(section, "ingest_latency_max_ms", round(max(ingest_times), 2))
        results.add(section, "ingest_latency_min_ms", round(min(ingest_times), 2))

    stats = get_stats()
    results.add(section, "server_stats_after", stats)
    log(f"  done: {len(files)} docs ({chunks_total} chunks) in {total_time:.1f}s ({len(files)/total_time:.0f} docs/s), mem: {mem_before:.0f} -> {mem_after:.0f} MB")


def bench_search_latency(results, label):
    """measure BM25 search latency via /chat endpoint.
    LLM fails fast (2s timeout) — we extract search_time_ms from response."""
    log(f"=== BENCHMARK: search latency ({label}) ===")
    section = f"03_search_latency_{label}"

    queries = [
        # single word queries
        "neural", "networks", "quantum", "cooking", "philosophy",
        "blockchain", "evolution", "printing", "fermentation", "microservices",
        # multi-word queries
        "machine learning gradient descent", "database indexing performance",
        "quantum mechanics wave particle", "natural language processing",
        "reinforcement learning reward", "supply chain management",
        "french revolution democracy", "genetic inheritance DNA",
        "container orchestration kubernetes", "zero trust security",
        # questions
        "how do neural networks learn", "what is quantum computing",
        "how does fermentation work", "what causes earthquakes",
        "how does evolution work", "what is blockchain technology",
        # edge cases
        "a", "the", "is",
        "null", "undefined", "NaN", "true", "false",
    ]

    search_times = []  # BM25 search times from server response
    total_times = []   # end-to-end request times
    per_query = {}
    queries_with_results = 0

    for q in queries:
        q_search_times = []
        q_total_times = []
        chunks_found = 0
        for _ in range(3):
            r, elapsed = do_chat(q)
            if r and r.status_code == 200:
                metrics = extract_search_metrics(r)
                if metrics:
                    st = metrics["search_time_ms"]
                    if st >= 0:
                        q_search_times.append(st)
                    q_total_times.append(elapsed)
                    chunks_found = max(chunks_found, metrics["chunks_retrieved"])

        if q_search_times:
            per_query[q[:50]] = {
                "bm25_avg_ms": round(statistics.mean(q_search_times), 3),
                "bm25_min_ms": round(min(q_search_times), 3),
                "e2e_avg_ms": round(statistics.mean(q_total_times), 1),
                "chunks_found": chunks_found,
            }
            search_times.extend(q_search_times)
            total_times.extend(q_total_times)
            if chunks_found > 0:
                queries_with_results += 1

    if search_times:
        sorted_st = sorted(search_times)
        results.add(section, "total_queries", len(search_times))
        results.add(section, "queries_returning_results", queries_with_results)
        results.add(section, "bm25_p50_ms", round(statistics.median(sorted_st), 3))
        results.add(section, "bm25_p95_ms", round(sorted_st[int(len(sorted_st) * 0.95)], 3))
        results.add(section, "bm25_p99_ms", round(sorted_st[int(len(sorted_st) * 0.99)], 3))
        results.add(section, "bm25_mean_ms", round(statistics.mean(sorted_st), 3))
        results.add(section, "bm25_min_ms", round(min(sorted_st), 3))
        results.add(section, "bm25_max_ms", round(max(sorted_st), 3))
        results.add(section, "bm25_stdev_ms", round(statistics.stdev(sorted_st), 3) if len(sorted_st) > 1 else 0)

        sorted_tt = sorted(total_times)
        results.add(section, "e2e_p50_ms", round(statistics.median(sorted_tt), 1))
        results.add(section, "e2e_mean_ms", round(statistics.mean(sorted_tt), 1))
        results.add(section, "e2e_max_ms", round(max(sorted_tt), 1))

    results.add(section, "per_query_detail", per_query)
    if search_times:
        log(f"  done: {len(search_times)} queries, BM25 p50={statistics.median(sorted(search_times)):.3f}ms, {queries_with_results}/{len(queries)} found results")
    else:
        log("  no search results obtained")


def bench_search_relevance(results, label):
    """test search accuracy — do retrieved chunks match expected topics?"""
    log(f"=== BENCHMARK: search relevance ({label}) ===")
    section = f"04_search_relevance_{label}"

    test_cases = [
        ("gradient descent optimization", "machine_learning"),
        ("neural network layers activation", "machine_learning"),
        ("kubernetes container deployment", "software_engineering"),
        ("database indexing query", "software_engineering"),
        ("French Revolution democracy monarchy", "history"),
        ("Roman Republic Empire Augustus", "history"),
        ("quantum mechanics particles waves", "science"),
        ("DNA double helix genetic", "science"),
        ("venture capital startup equity", "business"),
        ("supply chain manufacturing", "business"),
        ("Maillard reaction cooking flavor", "cooking"),
        ("fermentation bread cheese", "cooking"),
        ("existentialism individual existence", "philosophy"),
        ("utilitarianism consequences happiness", "philosophy"),
        ("blockchain distributed ledger", "technology"),
        ("edge computing IoT latency", "technology"),
    ]

    correct = 0
    total = len(test_cases)
    details = []

    for query, expected_topic in test_cases:
        r, elapsed = do_chat(query)
        if r and r.status_code == 200:
            metrics = extract_search_metrics(r)
            sources = metrics["sources"] if metrics else []

            found_topic = False
            source_ids = []
            for src in sources:
                chunk_id = src.get("chunk_id", "")
                source_ids.append(chunk_id)
                # chunk IDs from our test data contain the topic in the filename
                if expected_topic in chunk_id.lower() or expected_topic.replace("_", " ") in chunk_id.lower():
                    found_topic = True

            # also check the answer text for topic keywords
            if not found_topic and metrics:
                answer = metrics.get("answer", "").lower()
                topic_words = expected_topic.replace("_", " ").split()
                if sum(1 for w in topic_words if w in answer) >= len(topic_words) // 2 + 1:
                    found_topic = True

            if found_topic:
                correct += 1
            details.append({
                "query": query,
                "expected": expected_topic,
                "found": found_topic,
                "sources": source_ids[:3],
                "search_ms": round(metrics["search_time_ms"], 3) if metrics else -1,
            })
        else:
            details.append({
                "query": query,
                "expected": expected_topic,
                "found": False,
                "error": r.status_code if r else "timeout",
            })

    accuracy = correct / total * 100 if total > 0 else 0
    results.add(section, "accuracy_pct", round(accuracy, 1))
    results.add(section, "correct", correct)
    results.add(section, "total", total)
    results.add(section, "details", details)
    log(f"  relevance: {correct}/{total} ({accuracy:.0f}%)")


def bench_concurrent_load(results, label, num_workers=10, queries_per_worker=10):
    """test concurrent query throughput. measures BM25 performance under load."""
    log(f"=== BENCHMARK: concurrent load ({label}, {num_workers}w x {queries_per_worker}q) ===")
    section = f"05_concurrent_{label}"

    queries = [
        "neural networks", "quantum computing", "french revolution",
        "database indexing", "machine learning", "blockchain technology",
        "climate change", "supply chain", "fermentation process",
        "natural language processing",
    ]

    all_search_times = []
    all_e2e_times = []
    errors = 0
    start = time.time()

    def worker(worker_id):
        s_times = []
        e_times = []
        errs = 0
        for i in range(queries_per_worker):
            q = queries[i % len(queries)]
            r, elapsed = do_chat(q)
            if r and r.status_code == 200:
                metrics = extract_search_metrics(r)
                if metrics and metrics["search_time_ms"] >= 0:
                    s_times.append(metrics["search_time_ms"])
                e_times.append(elapsed)
            else:
                errs += 1
        return s_times, e_times, errs

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(num_workers)]
        for f in concurrent.futures.as_completed(futures):
            st, et, errs = f.result()
            all_search_times.extend(st)
            all_e2e_times.extend(et)
            errors += errs

    total_time = time.time() - start
    total_queries = num_workers * queries_per_worker

    results.add(section, "total_queries", total_queries)
    results.add(section, "successful", len(all_e2e_times))
    results.add(section, "errors", errors)
    results.add(section, "error_rate_pct", round(errors / total_queries * 100, 2) if total_queries > 0 else 0)
    results.add(section, "wall_clock_sec", round(total_time, 2))

    if all_search_times:
        sorted_st = sorted(all_search_times)
        results.add(section, "bm25_p50_ms", round(statistics.median(sorted_st), 3))
        results.add(section, "bm25_p95_ms", round(sorted_st[int(len(sorted_st) * 0.95)], 3))
        results.add(section, "bm25_p99_ms", round(sorted_st[int(len(sorted_st) * 0.99)], 3))
        results.add(section, "bm25_mean_ms", round(statistics.mean(sorted_st), 3))
        results.add(section, "bm25_max_ms", round(max(sorted_st), 3))

    if all_e2e_times:
        results.add(section, "throughput_qps", round(len(all_e2e_times) / total_time, 1))
        results.add(section, "e2e_p50_ms", round(statistics.median(sorted(all_e2e_times)), 1))
        results.add(section, "e2e_mean_ms", round(statistics.mean(all_e2e_times), 1))

    log(f"  done: {len(all_e2e_times)}/{total_queries} ok in {total_time:.1f}s, {len(all_e2e_times)/total_time:.1f} qps")


def bench_edge_cases(results):
    """test edge cases, security, and error handling."""
    log("=== BENCHMARK: edge cases & error handling ===")
    section = "06_edge_cases"
    tests = {}

    # empty question
    r, elapsed = do_chat("")
    tests["empty_question"] = {
        "status": r.status_code if r else "fail",
        "handled": r is not None and r.status_code in [200, 400],
        "latency_ms": round(elapsed, 1),
    }

    # very long query (10KB)
    r, elapsed = do_chat("a " * 5000)
    tests["10kb_query"] = {
        "status": r.status_code if r else "fail",
        "handled": r is not None,
        "latency_ms": round(elapsed, 1),
    }

    # unicode queries
    for name, q in [("chinese", "机器学习 深度学习"), ("arabic", "التعلم الآلي"), ("emoji", ""), ("mixed", "neural 网络 learning")]:
        r, elapsed = do_chat(q)
        tests[f"unicode_{name}"] = {"status": r.status_code if r else "fail", "latency_ms": round(elapsed, 1), "handled": r is not None}

    # special characters / injection attempts
    for name, q in [
        ("html_tags", "<h1>test</h1>"),
        ("sql_inject", "'; DROP TABLE--"),
        ("path_traversal", "../../etc/passwd"),
        ("newlines", "test\n\nquery\n"),
        ("xss", "<script>alert('xss')</script>"),
    ]:
        r, elapsed = do_chat(q)
        tests[f"special_{name}"] = {"status": r.status_code if r else "fail", "latency_ms": round(elapsed, 1), "handled": r is not None}

    # malformed JSON to /chat
    try:
        r = requests.post(f"{BASE_URL}/chat", data="not json", headers={"Content-Type": "application/json"}, timeout=10)
        tests["malformed_json"] = {"status": r.status_code, "handled": r.status_code in [400, 422, 500]}
    except Exception as e:
        tests["malformed_json"] = {"error": str(e), "handled": False}

    # missing required field
    try:
        r = requests.post(f"{BASE_URL}/chat", json={"wrong_field": "test"}, timeout=10)
        tests["missing_question_field"] = {"status": r.status_code, "correct_400": r.status_code == 400}
    except Exception as e:
        tests["missing_question_field"] = {"error": str(e)}

    # non-existent endpoint
    try:
        r = requests.get(f"{BASE_URL}/nonexistent", timeout=10)
        tests["404_handling"] = {"status": r.status_code, "correct": r.status_code == 404}
    except Exception as e:
        tests["404_handling"] = {"error": str(e)}

    # invalid chunk ID
    try:
        r = requests.get(f"{BASE_URL}/chunk/nonexistent_id_12345", timeout=10)
        tests["invalid_chunk_id"] = {"status": r.status_code, "correct": r.status_code in [404, 400]}
    except Exception as e:
        tests["invalid_chunk_id"] = {"error": str(e)}

    # path traversal on add-file-path
    try:
        r = requests.post(f"{BASE_URL}/add-file-path", json={"path": "/etc/passwd"}, timeout=10)
        tests["path_traversal_ingest"] = {"status": r.status_code, "blocked": r.status_code in [400, 403]}
    except Exception as e:
        tests["path_traversal_ingest"] = {"error": str(e)}

    # ingest with empty content
    r, _ = ingest_content("empty.txt", "")
    tests["empty_content_ingest"] = {"status": r.status_code if r else "fail", "handled": r is not None}

    # ingest with huge filename
    r, _ = ingest_content("a" * 1000 + ".txt", "test content")
    tests["huge_filename_ingest"] = {"status": r.status_code if r else "fail", "handled": r is not None}

    passed = sum(1 for v in tests.values() if isinstance(v, dict) and v.get("handled", v.get("correct", v.get("blocked", v.get("correct_400", False)))))
    results.add(section, "tests_run", len(tests))
    results.add(section, "tests_handled_gracefully", passed)
    results.add(section, "details", tests)
    log(f"  {passed}/{len(tests)} edge cases handled gracefully")


def bench_large_document_ingest(results, proc):
    """test ingestion of documents at increasing sizes."""
    log("=== BENCHMARK: large document ingestion ===")
    section = "07_large_doc_ingest"

    sizes = [
        ("1KB", 1024),
        ("10KB", 10 * 1024),
        ("100KB", 100 * 1024),
        ("500KB", 500 * 1024),
        ("1MB", 1024 * 1024),
        ("5MB", 5 * 1024 * 1024),
    ]

    for name, target_bytes in sizes:
        content = f"# Large Document Test - {name}\n\n"
        paragraph = "This is a test paragraph with multiple sentences to simulate real document content. It covers various topics including technology, science, and engineering. " * 5 + "\n\n"
        while len(content) < target_bytes:
            content += paragraph

        mem_before = get_memory_usage(proc.pid)
        r, elapsed = ingest_content(f"large_test_{name}.txt", content)
        mem_after = get_memory_usage(proc.pid)

        if r:
            results.add(section, f"{name}_status", r.status_code)
            results.add(section, f"{name}_latency_ms", round(elapsed, 1))
            results.add(section, f"{name}_actual_bytes", len(content))
            results.add(section, f"{name}_mem_delta_mb", round(mem_after - mem_before, 2))
            results.add(section, f"{name}_throughput_mb_s", round((len(content) / 1024 / 1024) / (elapsed / 1000), 2) if elapsed > 0 else 0)
            if r.status_code == 200:
                try:
                    data = r.json()
                    results.add(section, f"{name}_chunks_created", data.get("chunks_added", "n/a"))
                    results.add(section, f"{name}_tokens_added", data.get("tokens_added", "n/a"))
                except:
                    pass
            log(f"  {name}: {r.status_code} in {elapsed:.0f}ms (mem +{mem_after-mem_before:.1f}MB)")
        else:
            results.add(section, f"{name}_status", "TIMEOUT/ERROR")
            log(f"  {name}: FAILED")


def bench_stats_endpoint(results, proc):
    """capture final server stats and memory."""
    log("=== BENCHMARK: final server stats ===")
    section = "08_server_stats"

    stats = get_stats()
    results.add(section, "stats", stats)

    mem = get_memory_usage(proc.pid)
    results.add(section, "process_memory_mb", round(mem, 1))

    endpoints = ["/health", "/stats", "/catalog"]
    for path in endpoints:
        try:
            r = requests.get(f"{BASE_URL}{path}", timeout=10)
            results.add(section, f"endpoint_{path}", {"status": r.status_code, "responds": True})
        except Exception as e:
            results.add(section, f"endpoint_{path}", {"error": str(e), "responds": False})

    log(f"  memory: {mem:.0f}MB, chunks: {stats.get('total_chunks', 'n/a')}")


def bench_sustained_load(results, duration_sec=30, workers=5):
    """sustained load test — checks for latency degradation over time."""
    log(f"=== BENCHMARK: sustained load ({duration_sec}s, {workers} workers) ===")
    section = "09_sustained_load"

    queries = ["neural networks", "quantum computing", "database indexing",
               "machine learning", "blockchain", "climate change",
               "cooking fermentation", "philosophy ethics", "roman empire",
               "container orchestration"]

    all_search_times = []
    all_e2e_times = []
    errors = 0

    def worker(worker_id):
        nonlocal errors
        s_times = []
        e_times = []
        count = 0
        start = time.time()
        while time.time() - start < duration_sec:
            q = queries[count % len(queries)]
            r, elapsed = do_chat(q)
            if r and r.status_code == 200:
                metrics = extract_search_metrics(r)
                if metrics and metrics["search_time_ms"] >= 0:
                    s_times.append(metrics["search_time_ms"])
                e_times.append(elapsed)
            else:
                errors += 1
            count += 1
        return s_times, e_times

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, i) for i in range(workers)]
        for f in concurrent.futures.as_completed(futures):
            st, et = f.result()
            all_search_times.extend(st)
            all_e2e_times.extend(et)
    total_time = time.time() - start

    results.add(section, "duration_sec", round(total_time, 1))
    results.add(section, "total_queries", len(all_e2e_times))
    results.add(section, "errors", errors)

    if all_search_times:
        sorted_st = sorted(all_search_times)
        results.add(section, "avg_qps", round(len(all_e2e_times) / total_time, 1))
        results.add(section, "bm25_p50_ms", round(statistics.median(sorted_st), 3))
        results.add(section, "bm25_p95_ms", round(sorted_st[int(len(sorted_st) * 0.95)], 3))
        results.add(section, "bm25_p99_ms", round(sorted_st[int(len(sorted_st) * 0.99)], 3))
        results.add(section, "bm25_mean_ms", round(statistics.mean(sorted_st), 3))
        results.add(section, "bm25_max_ms", round(max(sorted_st), 3))

        # latency drift check
        quarter = len(all_search_times) // 4
        if quarter > 0:
            q1 = statistics.mean(all_search_times[:quarter])
            q4 = statistics.mean(all_search_times[-quarter:])
            results.add(section, "first_quarter_bm25_avg_ms", round(q1, 3))
            results.add(section, "last_quarter_bm25_avg_ms", round(q4, 3))
            drift = ((q4 - q1) / q1 * 100) if q1 > 0 else 0
            results.add(section, "latency_drift_pct", round(drift, 1))
            results.add(section, "stable", abs(drift) < 20)

    if all_e2e_times:
        results.add(section, "e2e_p50_ms", round(statistics.median(sorted(all_e2e_times)), 1))
        results.add(section, "e2e_mean_ms", round(statistics.mean(all_e2e_times), 1))

    log(f"  {len(all_e2e_times)} queries in {total_time:.0f}s = {len(all_e2e_times)/total_time:.1f} qps, errors: {errors}")


def bench_api_completeness(results):
    """test all API endpoints exist and respond."""
    log("=== BENCHMARK: API completeness ===")
    section = "10_api_completeness"

    endpoints = {
        "GET /health": ("GET", "/health", None),
        "GET /stats": ("GET", "/stats", None),
        "GET /catalog": ("GET", "/catalog", None),
        "GET /guide": ("GET", "/guide", None),
        "GET /originals": ("GET", "/originals", None),
        "POST /chat": ("POST", "/chat", {"question": "test"}),
        "POST /chat/stream": ("POST", "/chat/stream", {"question": "test"}),
        "POST /add-file": ("POST", "/add-file", {"filename": "api_test.txt", "content": "api test"}),
        "POST /add-file-path": ("POST", "/add-file-path", {"path": "/tmp/oe_test_nonexistent.txt"}),
        "POST /sources": ("POST", "/sources", {"turn_id": "CH1"}),
        "POST /tell-me-more": ("POST", "/tell-me-more", {"prev_turn_id": "CH1"}),
        "POST /clear-database": ("POST", "/clear-database", None),
        "POST /reload-catalog": ("POST", "/reload-catalog", None),
        "GET /chunk/{id}": ("GET", "/chunk/test_id", None),
    }

    available = 0
    for name, (method, path, body) in endpoints.items():
        try:
            if method == "GET":
                r = requests.get(f"{BASE_URL}{path}", timeout=15)
            else:
                r = requests.post(f"{BASE_URL}{path}", json=body, timeout=15)
            exists = r.status_code != 405
            if exists:
                available += 1
            results.add(section, name, {"status": r.status_code, "exists": exists})
        except Exception as e:
            results.add(section, name, {"error": str(e)[:100], "exists": False})

    results.add(section, "endpoints_available", f"{available}/{len(endpoints)}")
    log(f"  {available}/{len(endpoints)} endpoints available")


def bench_chunk_retrieval(results, label):
    """test chunk retrieval and adjacent chunk features."""
    log(f"=== BENCHMARK: chunk retrieval ({label}) ===")
    section = f"11_chunk_retrieval_{label}"

    # first, get some chunk IDs from a search
    r, _ = do_chat("neural networks machine learning")
    if not r or r.status_code != 200:
        results.add(section, "status", "SKIP - no search results")
        return

    metrics = extract_search_metrics(r)
    if not metrics or not metrics["sources"]:
        results.add(section, "status", "SKIP - no sources returned")
        return

    chunk_ids = [s["chunk_id"] for s in metrics["sources"]]
    results.add(section, "chunk_ids_from_search", len(chunk_ids))

    # test retrieving each chunk
    retrieval_times = []
    for cid in chunk_ids[:5]:
        start = time.time()
        try:
            r = requests.get(f"{BASE_URL}/chunk/{cid}", timeout=10)
            elapsed = (time.time() - start) * 1000
            if r.status_code == 200:
                retrieval_times.append(elapsed)
                data = r.json()
                results.add(section, f"chunk_{cid[:20]}", {
                    "status": 200,
                    "has_content": "content" in data,
                    "has_source_file": "source_file" in data,
                    "latency_ms": round(elapsed, 1),
                })
        except:
            pass

    # test adjacent chunk retrieval (context_window)
    if chunk_ids:
        cid = chunk_ids[0]
        for cw in [1, 2, 3]:
            try:
                start = time.time()
                r = requests.get(f"{BASE_URL}/chunk/{cid}?context_window={cw}", timeout=10)
                elapsed = (time.time() - start) * 1000
                if r.status_code == 200:
                    data = r.json()
                    adj = data.get("adjacent_chunks", [])
                    results.add(section, f"context_window_{cw}", {
                        "adjacent_count": len(adj),
                        "latency_ms": round(elapsed, 1),
                    })
            except:
                pass

    if retrieval_times:
        results.add(section, "avg_retrieval_ms", round(statistics.mean(retrieval_times), 2))

    log(f"  retrieved {len(retrieval_times)} chunks, avg {statistics.mean(retrieval_times):.1f}ms" if retrieval_times else "  no chunks retrieved")


def bench_ingestion_stress(results, proc):
    """stress test: ingest docs one at a time until server becomes unresponsive.
    documents the ingestion wall and server behavior under sustained writes."""
    log("=== BENCHMARK: ingestion stress test ===")
    section = "13_ingestion_stress"

    data_dir = os.path.join(TEST_DATA_DIR, "medium")
    files = sorted(glob.glob(os.path.join(data_dir, "doc_*.txt")))[:600]

    successful = 0
    failed = 0
    timeouts = 0
    first_failure_at = None
    server_unresponsive_at = None
    ingest_times = []

    for i, fpath in enumerate(files):
        with open(fpath) as f:
            content = f.read()
        fname = os.path.basename(fpath)

        r, elapsed = ingest_content(fname, content, timeout=30)
        if r and r.status_code == 200:
            successful += 1
            ingest_times.append(elapsed)
        elif r is None:
            timeouts += 1
            if first_failure_at is None:
                first_failure_at = i
                log(f"  first timeout at doc {i} ({fname})")
        else:
            failed += 1
            if first_failure_at is None:
                first_failure_at = i

        # check server health every 50 docs
        if (i + 1) % 50 == 0:
            try:
                hr = requests.get(f"{BASE_URL}/health", timeout=3)
                health_ok = hr.status_code == 200
            except:
                health_ok = False
            if not health_ok and server_unresponsive_at is None:
                server_unresponsive_at = i
                log(f"  server unresponsive at doc {i}")
            log(f"  doc {i+1}: {successful} ok, {timeouts} timeouts, {failed} errors, health={'OK' if health_ok else 'FAIL'}")

        # stop if server is totally dead (5 consecutive timeouts)
        if timeouts >= 5 and i - first_failure_at >= 10:
            log(f"  stopping stress test at doc {i} — server unresponsive")
            break

    results.add(section, "docs_attempted", len(files))
    results.add(section, "docs_ingested", successful)
    results.add(section, "docs_timed_out", timeouts)
    results.add(section, "docs_failed", failed)
    results.add(section, "first_failure_at_doc", first_failure_at)
    results.add(section, "server_unresponsive_at_doc", server_unresponsive_at)

    if ingest_times:
        results.add(section, "avg_ingest_ms", round(statistics.mean(ingest_times), 1))
        results.add(section, "p95_ingest_ms", round(sorted(ingest_times)[int(len(ingest_times) * 0.95)], 1))
        # check if latency degrades over time
        if len(ingest_times) >= 20:
            first_20 = statistics.mean(ingest_times[:20])
            last_20 = statistics.mean(ingest_times[-20:])
            results.add(section, "first_20_avg_ms", round(first_20, 1))
            results.add(section, "last_20_avg_ms", round(last_20, 1))
            results.add(section, "degradation_factor", round(last_20 / first_20, 2) if first_20 > 0 else "n/a")

    mem = get_memory_usage(proc.pid)
    results.add(section, "final_memory_mb", round(mem, 1))
    log(f"  stress test done: {successful}/{len(files)} ingested, wall at doc {first_failure_at}")


def bench_catalog(results, label):
    """test catalog endpoint with pagination."""
    log(f"=== BENCHMARK: catalog ({label}) ===")
    section = f"12_catalog_{label}"

    try:
        start = time.time()
        r = requests.get(f"{BASE_URL}/catalog", timeout=10)
        elapsed = (time.time() - start) * 1000
        if r.status_code == 200:
            data = r.json()
            results.add(section, "status", 200)
            results.add(section, "latency_ms", round(elapsed, 1))
            results.add(section, "response_keys", list(data.keys()) if isinstance(data, dict) else "array")
            if isinstance(data, dict):
                results.add(section, "total_items", data.get("total", data.get("count", len(data))))
        else:
            results.add(section, "status", r.status_code)
    except Exception as e:
        results.add(section, "error", str(e)[:100])

    log(f"  catalog check complete")


# ─── main runner ────────────────────────────────────────────────────

def run_full_benchmark():
    """run the complete benchmark suite."""
    log("=" * 60)
    log("  OCEAN ETERNA BENCHMARK SUITE")
    log("  test server on port 9191 (isolated)")
    log("  LLM disabled (fast-fail) — testing BM25 search engine")
    log("=" * 60)

    results = BenchmarkResults()
    results.add("00_metadata", "timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    results.add("00_metadata", "server_port", 9191)
    results.add("00_metadata", "test_data_dir", TEST_DATA_DIR)
    results.add("00_metadata", "note", "LLM disabled (timeout=2s, retries=0). search_time_ms = pure BM25 latency.")

    try:
        cpu_info = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip()
        mem_info = subprocess.check_output("free -h | grep Mem", shell=True).decode().strip()
        results.add("00_metadata", "cpu", cpu_info)
        results.add("00_metadata", "system_memory", mem_info)
    except:
        pass

    # start mock LLM server so /chat doesn't hang
    log("starting mock LLM server on port 11434...")
    mock_llm = subprocess.Popen(
        [sys.executable, os.path.expanduser("~/oe-benchmark-test/mock_llm.py")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    time.sleep(1)
    log(f"mock LLM started (pid={mock_llm.pid})")

    proc = start_server()
    if not proc:
        log("FATAL: could not start server")
        mock_llm.kill()
        return

    try:
        # ── phase 1: empty server tests ──
        log("\n--- PHASE 1: empty server ---")
        bench_server_startup(results)
        bench_api_completeness(results)

        # ── phase 2: small corpus (100 docs) ──
        # NOTE: after ingesting, restart server to trigger build_stemmed_index
        # (known bug: dynamically added docs don't update stem index)
        log("\n--- PHASE 2: small corpus (100 docs) ---")
        bench_ingestion(results, "small", proc)

        # restart to build stem index on the ingested corpus
        log("  restarting server to build stem index on ingested data...")
        stop_server(proc)
        time.sleep(1)
        # DON'T clear corpus — we want to keep the ingested data
        proc = start_server_keep_corpus()
        if not proc: return

        bench_search_latency(results, "small_100docs")
        bench_search_relevance(results, "small_100docs")
        bench_chunk_retrieval(results, "small_100docs")
        bench_catalog(results, "small_100docs")
        bench_edge_cases(results)

        # ── phase 3: medium corpus (300 docs) ──
        stop_server(proc)
        time.sleep(2)
        proc = start_server()  # clean start
        if not proc: return

        log("\n--- PHASE 3: medium corpus (300 docs) ---")
        bench_ingestion(results, "medium", proc, max_docs=300)

        # restart to build stem index
        log("  restarting server to build stem index...")
        stop_server(proc)
        time.sleep(1)
        proc = start_server_keep_corpus()
        if not proc: return

        bench_search_latency(results, "medium_300docs")
        bench_search_relevance(results, "medium_300docs")
        bench_concurrent_load(results, "medium_300docs", num_workers=5, queries_per_worker=10)
        bench_large_document_ingest(results, proc)
        bench_sustained_load(results, duration_sec=15, workers=3)

        # ── phase 4: ingestion stress test ──
        stop_server(proc)
        time.sleep(2)
        proc = start_server()  # clean start
        if not proc: return

        log("\n--- PHASE 4: ingestion stress test ---")
        bench_ingestion_stress(results, proc)

        # final stats
        bench_stats_endpoint(results, proc)

    finally:
        stop_server(proc)
        if mock_llm and mock_llm.poll() is None:
            mock_llm.kill()
            log("mock LLM stopped")

    # save results
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    results.save(json_path)
    log(f"\nresults saved to {json_path}")

    summary = results.summary()
    summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    log(f"summary saved to {summary_path}")

    print(summary)
    return results


if __name__ == "__main__":
    run_full_benchmark()
