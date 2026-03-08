#!/bin/bash
# OceanEterna Demo - Quick Start
# Created: January 24, 2026

cd "$(dirname "$0")/chat"

echo "Starting OceanEterna Chat Server..."
./ocean_chat_server &

sleep 2

echo ""
echo "Server running at http://localhost:8888"
echo "Open in browser: file://$(pwd)/ocean_demo.html"
echo ""
echo "Press Enter to stop the server..."
read
pkill -f ocean_chat_server
echo "Server stopped."
