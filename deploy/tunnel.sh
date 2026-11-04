#!/usr/bin/env bash
# Expose the locally running studio (localhost:8000) on a public HTTPS URL via a
# Cloudflare quick tunnel - no account, no inbound ports. The URL is random and
# only lives while this process runs; re-run to get a new one.
# Requires AUTH_TOKEN to be set on the server (from the environment or .env) - never
# expose it unauthenticated. Prints ready-to-open links that carry the token.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8000}
if [ -z "${AUTH_TOKEN:-}" ] && [ -f .env ]; then
  AUTH_TOKEN=$(grep '^AUTH_TOKEN=' .env | cut -d= -f2- || true)
fi
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
"$BIN" tunnel --url "http://localhost:$PORT" --no-autoupdate --protocol http2 >> tunnel.log 2>&1 &
for _ in $(seq 1 30); do
  url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' tunnel.log | head -1 || true)
  [ -n "$url" ] && break; sleep 1
done
if [ -z "$url" ]; then
  echo "no tunnel URL after 30 s - see tunnel.log"; exit 1
fi
echo "public URL : $url"
echo "token      : ${AUTH_TOKEN:-<unknown: not in the environment or .env>}"
# links that log the browser in on first open (the pages drop the token from the address bar)
echo "studio     : $url/?token=$AUTH_TOKEN"
echo "camera     : $url/camera/?token=$AUTH_TOKEN"
echo "from Pages : https://mwihoti.github.io/3dvid/?studio=$url&token=$AUTH_TOKEN   (needs CORS_ORIGINS=https://mwihoti.github.io on the server)"
