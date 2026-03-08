#!/usr/bin/env python3
import subprocess
import time
import json
import requests
import sys

print("=" * 50)
print("  OceanEterna v3 Performance Test")
print("=" * 50)
print()

# Start server
print("Starting server...")
server = subprocess.Popen(
    ["./ocean_chat_server_v3"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Wait for server to be ready
time.sleep(20)

# Check if server is ready
try:
    r = requests.get("http://localhost:8888/stats", timeout=5)
    stats = r.json()
    print(f"Server ready! Chunks loaded: {stats.get('chunks_loaded', 'N/A')}")
except:
    print("Server may still be loading, waiting more...")
    time.sleep(20)

print()
print("Running 10 test queries...")
print("-" * 50)

queries = [
    "Who is Tony Balay",
    "What is photosynthesis",
    "How does machine learning work",
    "What is black hat SEO",
    "Tell me about whale migration",
    "Denver school bond measure",
    "Carbon Copy Cloner Mac backup",
    "BBQ cooking class Missoula",
    "Neural network deep learning",
    "Climate change effects"
]

search_times = []
for i, q in enumerate(queries, 1):
    try:
        r = requests.post(
            "http://localhost:8888/chat",
            json={"question": q},
            timeout=30
        )
        data = r.json()
        search_ms = data.get("search_time_ms", 0)
        search_times.append(search_ms)
        print(f"Query {i:2d}: {search_ms:7.1f}ms - {q}")
    except Exception as e:
        print(f"Query {i:2d}: FAILED - {e}")
    time.sleep(1)

print("-" * 50)
if search_times:
    avg = sum(search_times) / len(search_times)
    min_t = min(search_times)
    max_t = max(search_times)
    print(f"Average: {avg:.1f}ms")
    print(f"Min:     {min_t:.1f}ms")
    print(f"Max:     {max_t:.1f}ms")
print("=" * 50)

# Stop server
server.terminate()
print("Server stopped.")
