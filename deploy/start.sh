#!/usr/bin/env bash
# Idempotent: (re)start the studio server + Cloudflare quick tunnel in tmux and print the public URL.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
if ! curl -s --max-time 2 http://127.0.0.1:8000/api/health >/dev/null; then
  tmux kill-session -t studio 2>/dev/null
  tmux new-session -d -s studio "set -a; . ./.env; set +a; exec ./nunif/venv/bin/python serve.py >> serve.log 2>&1"
  for _ in $(seq 1 30); do curl -s --max-time 2 http://127.0.0.1:8000/api/health >/dev/null && break; sleep 1; done
fi
if ! pgrep -x cloudflared >/dev/null; then
  tmux kill-session -t tunnel 2>/dev/null
  rm -f tunnel.log public_url.txt
  tmux new-session -d -s tunnel "set -a; . ./.env; set +a; bash deploy/tunnel.sh; sleep infinity"
  for _ in $(seq 1 60); do grep -q "Registered tunnel connection" tunnel.log 2>/dev/null && break; sleep 1; done
fi
url=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" tunnel.log | head -1)
echo "$url" > public_url.txt
echo "studio : $(curl -s --max-time 2 http://127.0.0.1:8000/api/health >/dev/null && echo up || echo DOWN)"
echo "tunnel : $(grep -q "Registered tunnel connection" tunnel.log && echo registered || echo NOT registered)"
echo "URL    : $url"
echo "token  : $AUTH_TOKEN"
