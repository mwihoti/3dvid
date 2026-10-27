# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is and who it is for

3dvid turns a **single talking-head clip** (a person speaking to camera, typically phone footage) into either:

- a **stylised portfolio intro**: voxel / neon-contour effect over a procedural background, framed for web or social (`stylize.py`), or
- a **stereo 3D video** for a VR headset (half side-by-side via iw3, `convert3d.sh`).

End user: one person, usually the repo owner, rendering short clips of themselves for a portfolio, LinkedIn, a launch video, or headset viewing. It is a **single-user tool, not a multi-tenant service**: one render at a time, in-memory job state, a single shared bearer token. Do not turn it into accounts/queues/billing unless asked.

CPU-only by design (no GPU on the dev host). Cost model to keep in mind: ~8 min per 30 s of stylised footage on 32 cores, scaling linearly with cores; a one-off depth + mask pass per clip is cached so re-renders skip both models.

## Setup and commands

`nunif/` is a **git submodule** (nagadomi/nunif, pinned commit in `.gitmodules` and duplicated as `NUNIF_COMMIT` in the Dockerfile; keep them in sync). Its venv is the project interpreter.

```bash
git submodule update --init                     # if nunif/ is empty
cd nunif && python3 -m venv venv && ./venv/bin/pip install -r requirements-torch.txt -r requirements.txt && cd ..
./nunif/venv/bin/pip install -r requirements-stylize.txt
```

Render (always through the nunif venv python):

```bash
./nunif/venv/bin/python stylize.py clip.mp4 --frame 120 --effect voxel --bg plasma   # one PNG, fastest iteration
./nunif/venv/bin/python stylize.py clip.mp4 --preview 3 --effect contour --face-clear # first 3 s
./nunif/venv/bin/python stylize.py clip.mp4 --effect voxel --bg grid --layout wide    # full render
./convert3d.sh clip.mp4 [start] [end]                                                 # half-SBS 3D
./nunif/venv/bin/python serve.py                                                      # web UI on 127.0.0.1:8000
```

There is no test suite and no linter config. Verify changes by rendering `--frame N` (seconds) or `--preview 3` (tens of seconds) rather than full clips. `stylize.py` prints a per-stage timing table at the end; use it when touching performance.

Container: `docker compose up -d --build` (CPU) or add `-f docker-compose.gpu.yml` (CUDA). The image bakes all model weights, so a Dockerfile change that touches the pip/weights layers costs a 10–20 min rebuild.

## Architecture

Three layers, deliberately decoupled by process boundaries:

1. **`nunif/` (vendored, do not modify)** provides iw3: the Depth-Anything-V2-Small depth model and the stereo synthesis CLI. `stylize.py` imports it (`sys.path.insert(0, NUNIF)`); `convert3d.sh` and `serve.py` shell out to `python -m iw3.cli`. Model weights download on first use into `nunif/iw3/pretrained_models/hub/`.

2. **`stylize.py`** is a self-contained per-frame pipeline: `probe` → `read_frames` (raw RGB over an ffmpeg pipe) → `get_cached` (depth via iw3 with EMA smoothing + person mask via rembg `u2net_human_seg`, both cached as half-res uint8 `.npz` in `.stylize_cache/` keyed by file hash + frame count + EMA) → effect (`voxel_effect` / `contour_effect`) → `compose` (procedural `bg_*`, `layout_dims`, optional `face_protect` which auto-detects the neck boundary) → `encode` (frames dir → H.264 + WebM via ffmpeg, audio passthrough). The cache means model changes need `--refresh-cache`; effect/background/layout changes do not.

3. **`serve.py` + `web/index.html`** is a FastAPI wrapper that builds the exact CLI command in `_build_cmd` (the two pipelines' flags live there and must mirror `convert3d.sh`), runs it as a subprocess in its own session, and scrapes tqdm/iw3 progress lines from stderr into weighted stage percentages (`STAGE_WEIGHTS`). One job at a time (409 otherwise). Jobs persist to `jobs.json` only so a restarted server can mark orphans as errors. The single HTML page keeps the bearer token in `localStorage`; `/api/*` and `/media/*` require it when `AUTH_TOKEN` is set, `/` and `/api/health` never do.

Everything mutable lives under `DATA_DIR` (`uploads/`, `outputs/`, `.stylize_cache/`, `jobs.json`); defaults to the repo dir locally and `/app/data` in the image. Mount a volume there or renders vanish on restart.

## Gotchas

- `probe()` returns **display** dimensions: phone clips carry a rotation tag and ffmpeg auto-rotates the pixel stream, so reshaping with stored width/height scrambles frames. Keep using display dims.
- iw3 time args want `hh:mm:ss`, `mm:ss`, or integer seconds; a float like `2.0` is rejected (see the preview handling in `_build_cmd`).
- `convert3d.sh` hardcodes `cd /root/3dvid/nunif`; it is written for the qm-personal host and the container, not for arbitrary checkouts.
- `.gitignore` excludes all media, caches, `venv/`, `.env`; `.dockerignore` additionally excludes `nunif/` because the Dockerfile clones it itself.

## Deployment targets

- **Fly.io** (primary, stable URL): app `3dvid`, `fly.toml` in repo, `~/.fly/bin/flyctl deploy --remote-only --ha=false`. Auto-stop is off so renders survive idle; stop the machine by hand.
- **qm-personal** (SSH host, free, 32+ cores): an on-demand microVM that stops when idle and loses all processes. `deploy/start.sh` restarts serve.py + a Cloudflare quick tunnel in tmux and writes the (random, per-restart) URL to `public_url.txt`; a `.bashrc` hook runs it on login. QUIC is blocked there, hence `--protocol http2` in `deploy/tunnel.sh`.
- **AWS EC2 GPU** (`deploy/ec2.sh`, `aws-launch.sh`): blocked until the account's G-instance quota is raised; the CPU path of the same script works on any Ubuntu box with Docker.
- Model download and CUDA-vs-CPU selection are automatic: `serve.py` passes `--gpu 0` to iw3 when torch sees CUDA, else `-1`; the Dockerfile picks the wheel index via `TORCH_INDEX`.
