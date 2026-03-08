#!/usr/bin/env python3
"""OE Double Helix — category-partitioned BM25 search server.

HTTP API compatible with Ocean Eterna's core endpoints.
"""

import json
import os
import re
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from category_store import CategoryStore, Chunk
from classifier import classify, classify_query, CATEGORIES

PORT = int(os.environ.get("OE_DH_PORT", "9292"))
DATA_DIR = os.environ.get("OE_DH_DATA", os.path.join(os.path.dirname(__file__), "categories"))


def tokenize(text: str) -> list[str]:
    """extract keywords from text."""
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    # remove common stopwords
    stops = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "ought",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "because", "but", "and", "or", "if", "while",
        "about", "up", "its", "it", "this", "that", "these", "those",
        "he", "she", "they", "them", "we", "you", "my", "your", "his",
        "her", "our", "their", "what", "which", "who", "whom",
    }
    return list(set(w for w in words if w not in stops))


def chunk_text(text: str, max_chunk_size: int = 1500) -> list[str]:
    """split text into paragraph-safe chunks."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > max_chunk_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:max_chunk_size]]


store = CategoryStore(DATA_DIR)


class DH_Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {format % args}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            self._send_json({
                "status": "ok",
                "version": "double_helix_0.1",
                "active_category": store.active_category,
            })

        elif path == "/stats":
            stats = store.get_stats()
            import psutil
            proc = psutil.Process() if "psutil" in sys.modules else None
            stats["memory_mb"] = round(proc.memory_info().rss / 1024 / 1024, 1) if proc else "unknown"
            self._send_json(stats)

        elif path == "/catalog":
            stats = store.get_stats()
            self._send_json(stats)

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/add-file":
            try:
                body = json.loads(self._read_body())
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON"}, 400)
                return

            filename = body.get("filename", "unknown.txt")
            content = body.get("content", "")
            if not content:
                self._send_json({"error": "content required"}, 400)
                return

            # classify into category
            category = classify(content, filename)
            text_chunks = chunk_text(content)
            chunk_ids = []

            for i, chunk_text_content in enumerate(text_chunks):
                cid = f"{category}_{uuid.uuid4().hex[:12]}"
                keywords = tokenize(chunk_text_content)
                token_count = int(len(chunk_text_content.split()) * 1.3)

                chunk = Chunk(
                    chunk_id=cid,
                    source_file=filename,
                    text=chunk_text_content,
                    keywords=keywords,
                    token_count=token_count,
                    category=category,
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                store.add_chunk(category, chunk)
                chunk_ids.append(cid)

            store.save_category(category)

            self._send_json({
                "status": "ok",
                "category": category,
                "chunks_created": len(chunk_ids),
                "chunk_ids": chunk_ids,
                "tokens_added": sum(
                    store.indexes[category].chunks[cid].token_count
                    for cid in chunk_ids
                    if cid in store.indexes[category].chunks
                ),
            })

        elif path == "/chat" or path == "/search":
            try:
                body = json.loads(self._read_body())
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON"}, 400)
                return

            query = body.get("question", body.get("query", ""))
            if not query:
                self._send_json({"error": "question required"}, 400)
                return

            top_k = body.get("top_k", 8)
            search_all = body.get("search_all", False)

            t0 = time.time()

            if search_all:
                results = store.search_all_categories(query, top_k=top_k)
                searched_categories = CATEGORIES
            else:
                target_categories = classify_query(query)
                results = store.search_bm25(target_categories[0], query, top_k=top_k)
                searched_categories = [target_categories[0]]

                # if few results, try next category
                if len(results) < 3 and len(target_categories) > 1:
                    more = store.search_bm25(target_categories[1], query, top_k=top_k)
                    results.extend(more)
                    results.sort(key=lambda x: x["score"], reverse=True)
                    results = results[:top_k]
                    searched_categories.append(target_categories[1])

            search_ms = (time.time() - t0) * 1000

            self._send_json({
                "answer": f"Found {len(results)} results across {', '.join(searched_categories)}",
                "sources": results,
                "search_ms": round(search_ms, 3),
                "categories_searched": searched_categories,
            })

        elif path == "/classify":
            try:
                body = json.loads(self._read_body())
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON"}, 400)
                return
            text = body.get("text", "")
            filename = body.get("filename", None)
            category = classify(text, filename)
            self._send_json({"category": category})

        else:
            self._send_json({"error": "not found"}, 404)


def main():
    server = HTTPServer(("0.0.0.0", PORT), DH_Handler)
    print(f"OE Double Helix server running on http://localhost:{PORT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Categories: {', '.join(CATEGORIES)}")
    stats = store.get_stats()
    for cat, info in stats["categories"].items():
        print(f"  {cat}: {info['chunks']} chunks, {info['tokens']} tokens")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
        store.save_all()
        server.server_close()


if __name__ == "__main__":
    main()
