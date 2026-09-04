# 3dvid

Turn a talking-head clip into a stylised portfolio intro, or into stereo 3D for a headset. CPU-only pipeline built on [nunif / iw3](https://github.com/nagadomi/nunif) for depth.

- `stylize.py` — voxel / neon-contour effects, procedural backgrounds (plasma, grid, dark) or the filmed background, layouts (wide / portrait / square / native), face-clear mode with automatic neck detection, lossless audio passthrough, H.264 + WebM output.
- `serve.py` + `web/` — local web UI: upload, pick options, watch progress, play the result.
- `convert3d.sh` — half-SBS stereo 3D via iw3.

## Setup

```bash
git clone --recurse-submodules <this-repo> 3dvid && cd 3dvid
cd nunif && python3 -m venv venv && ./venv/bin/pip install -r requirements-torch.txt -r requirements.txt && cd ..
./nunif/venv/bin/pip install -r requirements-stylize.txt
./nunif/venv/bin/python -m iw3.download_models   # run from nunif/ if needed
```

`nunif/` is a pinned submodule; its venv and any `.env` stay untracked.

## Use

```bash
# single frame for tuning (fast once depth/mask are cached)
./nunif/venv/bin/python stylize.py clip.mp4 --frame 120 --effect voxel --bg plasma --layout wide

# 3 s preview, face kept clear down to the collar, sharpened for soft footage
./nunif/venv/bin/python stylize.py clip.mp4 --preview 3 --effect voxel --face-clear --sharpen 0.6 --bg plasma --layout square

# full render, web-sized
./nunif/venv/bin/python stylize.py clip.mp4 --effect contour --bg grid --layout wide --crf 32 --webm-crf 50

# web UI (bind is 127.0.0.1; tunnel in with: ssh -L 8000:localhost:8000 <host>)
./nunif/venv/bin/python serve.py
```

Depth and person masks are cached in `.stylize_cache/` per clip, so re-renders skip both model passes. Rotated phone clips are handled (display dimensions are used, not stored ones).

## Deploy (Docker / IBM Cloud Code Engine)

The `Dockerfile` builds a CPU image with all model weights baked in (~3 GB). Mutable state
(uploads, renders, depth/mask cache, job history) lives under `DATA_DIR` — mount a volume or
an Object Storage bucket there or it is lost on restart.

| Env var | Default | Purpose |
|---|---|---|
| `HOST` / `PORT` | `0.0.0.0` / `8080` in the image | bind address |
| `AUTH_TOKEN` | unset | when set, `/api/*` and `/media/*` require `Authorization: Bearer <token>` (or `?token=`). **Set it on anything internet-facing.** |
| `DATA_DIR` | `/app/data` | uploads/, outputs/, .stylize_cache/, jobs.json |
| `STYLIZE_PYTHON` | `python3` | interpreter used for render subprocesses |

Code Engine, from the console: **Applications → Create → Build container image from source code**,
repo URL + a read-only GitHub token as the code-repo secret, strategy *Dockerfile*. Pick the
largest CPU/memory size offered, **min instances 1 while rendering** (scale-to-zero kills a job
mid-way; set 0 when idle), max 1. Env: `PORT=8080`, `AUTH_TOKEN=<long random>`. Mount your
bucket at `/app/data`. First build takes 10–20 min (torch + weights).

Render speed scales with cores: the ~8 min / 30 s figure is for 32 cores. A 4-vCPU container
is roughly 8x slower. GPU in Code Engine is offered as *Fleet* (batch), not for applications.

## Notes

- CPU only: ~8 min per 30 s stylise render on 32 cores, plus a one-off depth+mask pass per clip.
- Half-SBS / half-TB output looks like a duplicated image on a flat screen — that is correct; it resolves in a headset. Use anaglyph to check depth on a normal screen.
