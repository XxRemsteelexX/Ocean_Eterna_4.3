# OceanEterna Changelog

All notable changes to OceanEterna are documented in this file.

---

## [v3.0] - February 1, 2026

### Added
- **Binary Manifest Format** - New binary format for manifest files
  - 57% smaller file size (3.3 GB → 1.4 GB)
  - 3.2x faster loading (41s → 13s)
  - Keyword dictionary eliminates string repetition
  - Automatic fallback to JSONL if binary not available

- **Manifest Converter Utility** (`convert_manifest`)
  - Converts JSONL manifest to binary format
  - One-time conversion, reusable binary file
  - Usage: `./convert_manifest manifest.jsonl`

- **New Files**
  - `binary_manifest.hpp` - Header-only binary manifest library
  - `convert_manifest` - Standalone converter executable
  - `accuracy_test.py` - Automated accuracy testing script
  - `VERSION_COMPARISON.md` - Comparison of all versions

### Changed
- Server version updated to v3.0
- Startup message now shows "Binary Manifest + Fast BM25"
- Default manifest loading checks for binary version first

### Removed
- BM25S pre-computed index (from v2) - caused slower search
- BlockWeakAnd optimization (from v2) - not effective

### Performance
| Metric | v1 | v3 | Change |
|--------|-----|-----|--------|
| Startup | 41s | 13s | -68% |
| Search | 500ms | 700ms | +40% |
| Accuracy | 100% | 100% | Same |
| Manifest | 3.3GB | 1.4GB | -57% |

---

## [v2.0] - February 1, 2026 (Experimental)

### Added
- BM25S pre-computed score matrix
- BlockWeakAnd early termination
- Binary manifest format

### Issues
- Search became slower (500ms → 700ms)
- BM25S overhead exceeded benefits
- Kept binary manifest, removed BM25S in v3

---

## [v1.0] - January 30, 2026

### Initial Release
- BM25 keyword search with OpenMP parallelization
- LZ4 compression for chunk storage
- HTTP API server on port 8888
- Chat interface with LLM integration
- File indexing (parallel, 32 cores)
- Conversation history tracking

### Fixes Applied (Jan 29-30)
- HTTP request parsing (Content-Length handling)
- UTF-8 sanitization for JSON responses
- Keyword extraction (frequency-based, no limit)
- Query extraction (matches keyword format)
- Chunk boundary detection (3+ newlines, paragraph breaks)

### Performance Achieved
- **Indexing:** 111 MB/sec, 19M tokens/sec
- **Search:** ~500ms for 5M chunks
- **Accuracy:** 100% (10/10 test questions)
- **Startup:** 41 seconds

---

## Binary Manifest Format Specification

```
HEADER (24 bytes):
  magic[4]        = "OEM1"
  version[4]      = 1
  chunk_count[8]  = number of chunks
  keyword_count[8]= number of unique keywords

KEYWORD DICTIONARY:
  For each keyword:
    len[2] + string[len]

CHUNK ENTRIES:
  For each chunk:
    id_len[2] + id[id_len]
    summary_len[2] + summary[summary_len]
    offset[8]
    length[8]
    token_start[4]
    token_end[4]
    timestamp[8]
    kw_count[2]
    kw_indices[kw_count * 4]
```

---

## Test Results

### Accuracy Test (v3.0 - February 1, 2026)
```
[PASS] Who is Tony Balay (758ms)
[PASS] BBQ class Missoula cost (791ms)
[PASS] Carbon Copy Cloner Mac (676ms)
[PASS] Denver school bond (681ms)
[PASS] whale migration (751ms)
[PASS] black hat SEO (678ms)
[PASS] photosynthesis (587ms)
[PASS] machine learning (668ms)
[PASS] September 22 BBQ class (704ms)
[PASS] Beginners BBQ (727ms)
------------------------------------
Result: 10/10 passed (100%)
```

---

## Migration Guide

### From v1 to v3

1. Copy v3 files to your installation:
   ```bash
   cp binary_manifest.hpp /path/to/chat/
   cp ocean_chat_server.cpp /path/to/chat/
   ```

2. Rebuild the server:
   ```bash
   g++ -O3 -std=c++17 -march=native -fopenmp \
       -o ocean_chat_server ocean_chat_server.cpp \
       -llz4 -lcurl -lpthread
   ```

3. Convert manifest to binary (one-time):
   ```bash
   g++ -O3 -std=c++17 -DBINARY_MANIFEST_CONVERTER -x c++ \
       -o convert_manifest binary_manifest.hpp
   ./convert_manifest guten_9m_build/manifest_guten9m.jsonl
   ```

4. Start the server:
   ```bash
   ./ocean_chat_server
   ```

The server will automatically use the binary manifest if it exists and is newer than the JSONL.
