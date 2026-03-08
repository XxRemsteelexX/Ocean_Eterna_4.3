#!/usr/bin/env python3
"""
OceanEterna v4 Accuracy Test
Tests the same 10 questions from the original benchmark
"""
import subprocess
import time
import json
import requests
import sys
import os

# Kill any existing server
os.system("pkill -9 -f ocean_chat_server 2>/dev/null")
time.sleep(2)

# Update binary manifest timestamp
os.system("touch guten_9m_build/manifest_guten9m.bin")

print("=" * 60)
print("  OceanEterna v4 Accuracy Test")
print("=" * 60)
print()

# Start server
print("Starting server...")
server = subprocess.Popen(
    ["./ocean_chat_server_v4"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)

# Wait for server
print("Waiting for server to load...")
time.sleep(25)

# Test questions with expected keywords in answer
tests = [
    ("Who is Tony Balay", ["tony", "balay"]),
    ("How much does the BBQ class in Missoula cost", ["$", "cost", "class"]),
    ("What is Carbon Copy Cloner for Mac", ["backup", "mac", "clone"]),
    ("What is the Denver school bond", ["denver", "school", "bond"]),
    ("Tell me about whale migration", ["whale", "migrat"]),
    ("What is black hat SEO", ["seo", "black hat", "search"]),
    ("What is photosynthesis", ["light", "plant", "energy"]),
    ("What is machine learning", ["learn", "algorithm", "data"]),
    ("When is the September 22 BBQ class", ["september", "22", "bbq"]),
    ("What is the Beginners BBQ class about", ["bbq", "beginner", "class"]),
]

print()
print("Running 10 accuracy tests...")
print("-" * 60)

passed = 0
failed = 0

for i, (question, keywords) in enumerate(tests, 1):
    try:
        r = requests.post(
            "http://localhost:8888/chat",
            json={"question": question},
            timeout=30
        )
        data = r.json()
        answer = data.get("answer", "").lower()
        search_ms = data.get("search_time_ms", 0)

        # Check if any keyword is in answer
        found = any(kw.lower() in answer for kw in keywords)

        if found:
            print(f"[PASS] Q{i}: {question} ({search_ms:.0f}ms)")
            passed += 1
        else:
            print(f"[FAIL] Q{i}: {question}")
            print(f"       Answer: {answer[:100]}...")
            failed += 1

    except Exception as e:
        print(f"[ERROR] Q{i}: {question} - {e}")
        failed += 1

    time.sleep(1)

print("-" * 60)
print(f"Result: {passed}/10 passed ({passed*10}%)")
print("=" * 60)

# Stop server
server.terminate()
