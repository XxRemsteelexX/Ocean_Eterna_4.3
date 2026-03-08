#!/usr/bin/env python3
"""ultra-minimal mock LLM server that returns instant responses.
used for benchmarking BM25 search without LLM latency."""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class MockLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # handle any POST path (including /v1/chat/completions)
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length else b''

        response = {
            "choices": [{
                "message": {
                    "content": "Mock LLM response for benchmarking."
                },
                "finish_reason": "stop"
            }],
            "model": "mock-benchmark",
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}
        }

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        resp_bytes = json.dumps(response).encode()
        self.send_header('Content-Length', len(resp_bytes))
        self.end_headers()
        self.write(resp_bytes)

    def log_message(self, format, *args):
        pass  # suppress logs

    def write(self, data):
        self.wfile.write(data)

if __name__ == "__main__":
    server = HTTPServer(('127.0.0.1', 11434), MockLLMHandler)
    print("mock LLM server running on port 11434")
    server.serve_forever()
