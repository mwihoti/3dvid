#!/bin/bash
# Usage: ./convert3d.sh input.mp4 [start_time] [end_time]
# Converts a normal video to half side-by-side 3D (VR-ready).
set -e
IN="$1"
[ -f "$IN" ] || { echo "File not found: $IN"; exit 1; }
OUT="${IN%.*}_3d.mp4"
EXTRA=""
[ -n "$2" ] && EXTRA="$EXTRA --start-time $2"
[ -n "$3" ] && EXTRA="$EXTRA --end-time $3"
cd /root/3dvid/nunif
exec ./venv/bin/python -m iw3.cli --gpu -1 --depth-model Any_V2_S \
  --divergence 2.0 --convergence 0.5 --half-sbs \
  --max-output-height 720 \
  -i "$IN" -o "$OUT" --yes $EXTRA
