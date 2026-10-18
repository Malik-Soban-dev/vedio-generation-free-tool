#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# WanGP Studio Dashboard - local launcher (Linux / macOS / WSL)
#
# Opens the dashboard in a REAL new browser window at http://localhost:7860
# No E2B sandbox, no traffic-access-token, no 403.
# ---------------------------------------------------------------------------
set -e

PORT="${PORT:-7860}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 was not found on your PATH."
  echo "Install Python 3.8+ from https://www.python.org/downloads/ and try again."
  exit 1
fi

URL="http://localhost:${PORT}/"

echo "=============================================================="
echo "  WanGP v18.17 Studio - Interactive Dashboard Preview"
echo "=============================================================="
echo "  Serving : $DIR"
echo "  Opening : $URL"
echo "  Press Ctrl+C in this window to stop the server."
echo "=============================================================="

# Start the browser in the background once the server is up.
(
  for _ in $(seq 1 40); do
    if curl -s -o /dev/null "http://127.0.0.1:${PORT}/" 2>/dev/null; then
      case "$(uname -s)" in
        Darwin*) open "$URL" ;;
        Linux*)  xdg-open "$URL" >/dev/null 2>&1 || true ;;
        *)       true ;;
      esac
      break
    fi
    sleep 0.25
  done
) &

exec python3 preview_server.py
