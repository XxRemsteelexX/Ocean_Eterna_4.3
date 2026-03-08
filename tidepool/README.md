# Tidepool -- Partitioned Mode for Low-RAM Systems

Experimental partitioned mode for running Ocean Eterna on RAM-constrained devices (Raspberry Pi, cheap VPS, embedded systems).

Splits the index into 5 categories and only loads one at a time, reducing RAM by ~50-80% at scale.

## How It Works

Instead of loading every chunk into RAM, Tidepool partitions data into 5 broad categories:

| Category | Contents |
|----------|----------|
| `personal` | journals, notes, goals, health, hobbies |
| `work` | meetings, projects, clients, business docs |
| `documents` | PDFs, research papers, articles, reference |
| `coding` | source code, configs, technical docs |
| `chat_logs` | conversations, transcripts, session logs |

Only one category is loaded at a time. A keyword classifier routes queries to the right category automatically.

## Tradeoffs

| | Main OE | Tidepool |
|---|---------|----------|
| RAM (5K chunks) | 50 MB | 25 MB |
| Targeted search | 13 ms | 6.5 ms (2x faster) |
| Cross-category search | N/A | 190 ms (searches all 5 sequentially) |
| Category swap | N/A | ~30 ms |

Targeted search is faster because the index is 5x smaller. Cross-category search is slower because it has to load each partition from disk.

## Status

**Proof of concept.** Not production-tested. Known limitations:

- Category classifier uses simple keyword matching -- may misclassify ambiguous content
- Cross-category search loads/unloads all 5 categories sequentially
- No incremental disk sync (saves full category on swap)
- Python prototype only (main OE is C++)

## Usage

```bash
cd tidepool
python3 server.py
# runs on port 9292 by default

# add a document (auto-classified into category)
curl -X POST http://localhost:9292/add-file \
  -H "Content-Type: application/json" \
  -d '{"filename": "notes.txt", "content": "your text here"}'

# search (auto-routed to best category)
curl -X POST http://localhost:9292/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "search query"}'

# search all categories
curl -X POST http://localhost:9292/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "search query", "search_all": true}'

# check which category a query would route to
curl -X POST http://localhost:9292/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "python function async"}'
```

## Tests

```bash
python3 tests/test_double_helix.py
# 11/11 tests passing
```
