#!/usr/bin/env bash
# Expose the locally running studio (localhost:8000) on a public HTTPS URL via a
# Cloudflare quick tunnel - no account, no inbound ports. The URL is random and
# only lives while this process runs; re-run to get a new one.
# Requires AUTH_TOKEN to be set on the server (see .env) - never expose it unauthenticated.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8000}
BIN=${CLOUDFLARED:-$HOME/.local/bin/cloudflared}
if [ ! -x "$BIN" ]; then
  mkdir -p "$(dirname "$BIN")"
  curl -sSL -o "$BIN" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$BIN"
fi
if ! curl -s --max-time 3 "http://localhost:$PORT/api/health" | grep -q '"auth":true'; then
  echo "refusing: server on :$PORT is not running with AUTH_TOKEN set (start it with: set -a; . ./.env; set +a; python serve.py)"; exit 1
fi
: > tunnel.log
"$BIN" tunnel --url "http://localhost:$PORT" --no-autoupdate >> tunnel.log 2>&1 &
for _ in $(seq 1 30); do
  url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' tunnel.log | head -1)
  [ -n "$url" ] && break; sleep 1
done
echo "public URL: ${url:-<none - see tunnel.log>}"
echo "token     : $(grep '^AUTH_TOKEN=' .env | cut -d= -f2)"
