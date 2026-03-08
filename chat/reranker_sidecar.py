#!/usr/bin/env python3
"""
OceanEterna v4.2 Reranker Sidecar
Purpose-built reranker using bge-reranker-v2-m3 (278M params, CPU-friendly).
Runs as a local HTTP service on port 8889.

Two-stage pipeline:
1. BM25 retrieves top-50 candidates (fast, ~100ms) — done by C++ server
2. This sidecar scores each candidate → returns reranked top-k

Usage:
    python3 reranker_sidecar.py [--port 8889] [--model BAAI/bge-reranker-v2-m3]

API:
    POST /rerank
    {
        "query": "search query",
        "documents": [
            {"chunk_id": "id1", "content": "text1", "score": 1.5},
            {"chunk_id": "id2", "content": "text2", "score": 1.2}
        ],
        "top_k": 8
    }
    Response:
    {
        "results": [
            {"chunk_id": "id1", "content": "text1", "bm25_score": 1.5, "rerank_score": 0.95},
            ...
        ],
        "rerank_time_ms": 45.2
    }
"""

import argparse
import json
import time
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

# global model reference
reranker = None
tokenizer = None
model_name = "BAAI/bge-reranker-v2-m3"

def load_model(name: str):
    """load the reranker model. falls back to cross-encoder if bge not available."""
    global reranker, tokenizer, model_name
    model_name = name

    try:
        from sentence_transformers import CrossEncoder
        print(f"loading reranker model: {name}")
        reranker = CrossEncoder(name, max_length=512)
        print(f"model loaded successfully")
        return True
    except ImportError:
        print("sentence-transformers not installed. install with:")
        print("  pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"failed to load model {name}: {e}")
        # try fallback model
        try:
            fallback = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            print(f"trying fallback model: {fallback}")
            reranker = CrossEncoder(fallback, max_length=512)
            model_name = fallback
            print(f"fallback model loaded")
            return True
        except Exception as e2:
            print(f"fallback also failed: {e2}")
            return False


def rerank(query: str, documents: list, top_k: int = 8) -> list:
    """rerank documents using the loaded model."""
    if reranker is None:
        # no model loaded — return documents sorted by original score
        return sorted(documents, key=lambda d: d.get("score", 0), reverse=True)[:top_k]

    # prepare query-document pairs for the model
    pairs = [(query, doc.get("content", "")[:512]) for doc in documents]

    start = time.time()
    scores = reranker.predict(pairs)
    elapsed_ms = (time.time() - start) * 1000

    # combine with original docs
    for i, doc in enumerate(documents):
        doc["rerank_score"] = float(scores[i])
        doc["bm25_score"] = doc.get("score", 0)

    # sort by rerank score (higher = more relevant)
    ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)

    return ranked[:top_k], elapsed_ms


class RerankerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/rerank":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body)
                query = data.get("query", "")
                documents = data.get("documents", [])
                top_k = data.get("top_k", 8)

                if not query or not documents:
                    self.send_error(400, "missing query or documents")
                    return

                results, rerank_ms = rerank(query, documents, top_k)

                response = {
                    "results": results,
                    "rerank_time_ms": round(rerank_ms, 1),
                    "model": model_name,
                    "input_count": len(documents),
                    "output_count": len(results)
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except json.JSONDecodeError:
                self.send_error(400, "invalid json")
            except Exception as e:
                self.send_error(500, str(e))

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "status": "ok",
                "model": model_name,
                "model_loaded": reranker is not None
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404, "not found")

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "status": "ok",
                "model": model_name,
                "model_loaded": reranker is not None
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404, "not found")

    def log_message(self, format, *args):
        # quieter logging
        sys.stderr.write(f"[reranker] {args[0]} {args[1]} {args[2]}\n")


def main():
    parser = argparse.ArgumentParser(description="OceanEterna Reranker Sidecar")
    parser.add_argument("--port", type=int, default=8889, help="port to listen on")
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-v2-m3",
                        help="reranker model name")
    args = parser.parse_args()

    if not load_model(args.model):
        print("WARNING: running without model — will pass through BM25 scores")

    server = HTTPServer(("127.0.0.1", args.port), RerankerHandler)
    print(f"reranker sidecar running on http://127.0.0.1:{args.port}")
    print(f"model: {model_name}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down reranker")
        server.shutdown()


if __name__ == "__main__":
    main()
