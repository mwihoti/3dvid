#!/usr/bin/env bash
# One-shot setup on an EC2 GPU instance (Deep Learning Base AMI, Ubuntu).
# Run from the cloned repo:  bash deploy/ec2.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null; then
  echo "docker not found - use the 'Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu)' or install docker + nvidia-container-toolkit first"; exit 1
fi
GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  GPU=1; echo "GPU: $(nvidia-smi -L | head -1)"
else
  echo "no GPU detected - building the CPU image"
fi

if [ ! -f .env ] || ! grep -q '^AUTH_TOKEN=' .env; then
  tok=$(openssl rand -hex 24)
  echo "AUTH_TOKEN=$tok" >> .env
  echo "generated AUTH_TOKEN (saved in .env): $tok"
fi
mkdir -p data

if [ "$GPU" = 1 ]; then
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
else
  docker compose up -d --build
fi

ip=$(curl -s --max-time 3 http://169.254.169.254/latest/meta-data/public-ipv4 || echo "<instance-ip>")
echo
echo "  3dvid studio -> http://$ip:8080     token: $(grep '^AUTH_TOKEN=' .env | cut -d= -f2)"
echo "  logs: docker compose logs -f studio"
