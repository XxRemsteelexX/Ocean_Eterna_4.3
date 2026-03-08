# Ocean Eterna v4.3

A high-performance BM25 search engine for personal knowledge management. Single C++ binary, sub-millisecond search, 10 MB base RAM. No GPU, no cloud, no Python runtime required.

---

## Why Ocean Eterna

Most RAG systems use vector embeddings that need GPUs, hundreds of MB of RAM, and complex dependencies. Ocean Eterna uses BM25 keyword search with Porter stemming -- it runs on anything with a C++ compiler, searches in microseconds, and fits on a Raspberry Pi.

When used as an MCP tool, Ocean Eterna handles retrieval only. The calling agent (Claude, GPT, etc.) provides the LLM for answer synthesis. No local LLM needed.

---

## Performance

Benchmarked on Intel i9-14900KS, 94 GB RAM, Linux 6.12.

| Metric | Value |
|--------|-------|
| BM25 search latency (p50) | 0.019 ms |
| Concurrent throughput | 1,351 queries/sec |
| Sustained load | 53.7 qps (stable) |
| Ingestion speed | 249-327 docs/sec |
| Large doc throughput | 48 MB/sec |
| Base RAM | 10 MB |
| Per-document cost | ~46 KB |
| Binary size | 1.2 MB |

### Comparison

| Feature | Ocean Eterna | ChromaDB | Weaviate | Pinecone |
|---------|-------------|----------|----------|----------|
| Search latency | <0.1 ms | 5-50 ms | 10-100 ms | 50-200 ms |
| Memory (300 docs) | 24 MB | ~200 MB | ~500 MB | Cloud |
| Ingestion speed | 249 docs/s | 50 docs/s | 100 docs/s | API limit |
| Single binary | Yes | No (Python) | No (Go+Docker) | Cloud only |
| Self-hosted | Yes | Yes | Yes | No |
| Cost | Free (personal) | Free | Free/Paid | $70+/mo |

---

## Quick Start

### Build

```bash
# install dependencies (Ubuntu/Debian)
sudo apt install g++ liblz4-dev libcurl4-openssl-dev libzstd-dev

# compile
cd chat
g++ -O3 -std=c++17 -march=native -fopenmp \
    -o ocean_chat_server ocean_chat_server.cpp \
    -llz4 -lcurl -lpthread -lzstd
```

### Run

```bash
./ocean_chat_server
```

Server starts on `http://localhost:8888` by default.

### Add a document

```bash
curl -X POST http://localhost:8888/add-file \
  -H "Content-Type: application/json" \
  -d '{"filename": "notes.txt", "content": "Your text content here..."}'
```

### Search

```bash
curl -X POST http://localhost:8888/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "search query here"}'
```

---

## API Reference

### Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Corpus statistics (chunks, tokens, RAM, query counts) |
| `/guide` | GET | Chapter navigation guide |
| `/catalog` | GET | Browse indexed documents |
| `/originals` | GET | List preserved original files |

### Search and Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | BM25 search + LLM answer synthesis |
| `/chat/stream` | POST | Same as /chat but with SSE streaming |
| `/sources` | POST | Get source references for a previous turn |
| `/tell-me-more` | POST | Expand context on previous answer |
| `/chunk/:id` | GET | Retrieve a specific chunk by ID with adjacent context |

**POST /chat request:**
```json
{"question": "What is photosynthesis?"}
```

**Response:**
```json
{
  "answer": "Photosynthesis is...",
  "turn_id": "CH42",
  "search_time_ms": 0.03,
  "llm_time_ms": 1200,
  "sources": [
    {"chunk_id": "doc_001", "score": 12.5, "text": "...", "source_file": "biology.txt"}
  ]
}
```

### Ingestion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/add-file` | POST | Ingest text content (JSON body: `{filename, content}`) |
| `/add-file-path` | POST | Ingest file from server disk path |
| `/clear-database` | POST | Clear conversation history |
| `/reload-catalog` | POST | Hot-reload originals catalog |

Supports: PDF, DOCX, XLSX, CSV, PPTX, PNG/JPG (OCR), TXT, MD, HTML, Jupyter notebooks, and code files (.py, .js, .cpp, .java, .sql, .go, .rs, .rb, .r).

Documents are chunked at paragraph boundaries (no mid-paragraph splits), compressed with LZ4/Zstd, and indexed with BM25 keyword extraction + Porter stemming.

---

## Configuration

All settings in `chat/config.json`. Environment variables override config file values.

```json
{
  "server":     { "port": 8888, "host": "0.0.0.0" },
  "llm":        { "use_external": false, "local_url": "http://localhost:11434/v1/chat/completions",
                  "local_model": "llama3:8b", "timeout_sec": 30, "max_retries": 2 },
  "search":     { "top_k": 8, "bm25_k1": 1.5, "bm25_b": 0.75, "score_threshold": 0.2 },
  "reranker":   { "enabled": false, "url": "http://127.0.0.1:8889/rerank" },
  "corpus":     { "manifest": "corpus/manifest.jsonl", "storage": "corpus/storage.bin" },
  "auth":       { "enabled": false, "api_key": "" },
  "rate_limit": { "enabled": false, "requests_per_minute": 60 }
}
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OCEAN_API_KEY` | API key for external LLM service |
| `OCEAN_API_URL` | Override external LLM endpoint |
| `OCEAN_MODEL` | Override external LLM model name |
| `OCEAN_SERVER_API_KEY` | Server authentication key (for X-API-Key header) |

---

## MCP Integration

Ocean Eterna includes an MCP (Model Context Protocol) server for use with AI coding agents.

```bash
pip install -r mcp_requirements.txt
python3 mcp_server.py
```

**MCP tools:** `oe_search`, `oe_add_file`, `oe_add_file_path`, `oe_add_document`, `oe_get_chunk`, `oe_stats`, `oe_catalog`, `oe_tell_me_more`, `oe_get_original`, `oe_reconstruct`

When used via MCP, the calling agent's LLM handles answer synthesis. Ocean Eterna just returns raw BM25 search results and chunk text.

---

## Architecture

```
ocean_chat_server.cpp    Main server (~3100 lines C++17)
search_engine.hpp        BM25 TAAT search with Porter stemming
config.hpp               JSON config with env variable overrides
llm_client.hpp           LLM HTTP client with retry/backoff
porter_stemmer.hpp       Porter 1980 stemming algorithm
binary_manifest.hpp      Binary manifest I/O for fast startup
httplib.h                cpp-httplib HTTP server (bundled)
json.hpp                 nlohmann JSON parser (bundled)
```

### Data Flow

```
Query -> HTTP Server -> Keyword Extraction -> Porter Stemmer
      -> BM25 TAAT Search (inverted index) -> Decompress Chunks (LZ4/Zstd)
      -> Build Context -> LLM API Call (optional) -> JSON Response
```

### Dependencies

| Library | Purpose |
|---------|---------|
| liblz4 | LZ4 chunk compression |
| libcurl | HTTP client for LLM API |
| libzstd | Zstd compression |
| OpenMP | Parallel processing |
| cpp-httplib | HTTP server (bundled, no install needed) |
| nlohmann/json | JSON parsing (bundled, no install needed) |

---

## Benchmarks

The `benchmarks/` directory contains the automated benchmark suite used to generate the performance numbers above.

```bash
# run benchmarks against a test server (requires mock LLM)
python3 benchmarks/mock_llm.py &       # mock LLM on port 11434
python3 benchmarks/generate_corpus.py  # generate test data
python3 benchmarks/benchmark.py        # run full suite
```

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v4.3 | Mar 2026 | Fix stem index on dynamic ingestion, fix ingestion lock contention |
| v4.2 | Feb 2026 | Creature tier system, smart chunking, 16 bug fixes |
| v4.1 | Feb 2026 | Thread safety, SSE streaming, TF-aware BM25 |
| v4.0 | Feb 2026 | 10-100x faster search (TAAT inverted index), modular architecture |
| v3.0 | Feb 2026 | Binary manifest, 3x faster startup |
| v1.0 | Jan 2026 | Initial release, BM25 search, LZ4 compression |

---

## License

Licensed under the [Business Source License 1.1](LICENSE) (BSL 1.1).

- **Personal use**: Free -- education, research, hobby projects, individual use
- **Commercial use**: Requires a paid license
- **Change Date**: March 8, 2030 -- automatically converts to Apache 2.0

Same license used by MariaDB, HashiCorp, CockroachDB, and Sentry.

For commercial licensing: [chainlinks.ai](https://chainlinks.ai)
