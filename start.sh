#!/usr/bin/env bash
set -e

# Build & start containers
docker-compose up --build -d

# wait for server to be ready
sleep 3

# open the UI
if command -v xdg-open &>/dev/null; then
  xdg-open "index.html"
elif command -v open &>/dev/null; then
  open "index.html"
else
  echo "Navigate to index.html in your browser"
fi
