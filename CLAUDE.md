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
./nunif/venv/bin/python stylize.py photo.jpg --effect voxel --bg plasma              # a still renders as one frame
./nunif/venv/bin/python stylize.py clip.mp4 --preview 3 --effect contour --face-clear # first 3 s
./nunif/venv/bin/python stylize.py clip.mp4 --effect voxel --bg grid --layout wide    # full render
./convert3d.sh clip.mp4 [start] [end]                                                 # half-SBS 3D
./nunif/venv/bin/python serve.py                                                      # web UI on 127.0.0.1:8000
```

There is no test suite and no linter config. For `web/camera/`, `node --check web/camera/*.js` catches syntax; shaders only compile at runtime, so load the page (headless Chromium with `--use-fake-device-for-media-stream --use-file-for-fake-video-capture=<clip>.y4m` works) and watch the console for `shader compile failed`. Verify changes by rendering `--frame N` (seconds) or `--preview 3` (tens of seconds) rather than full clips. `stylize.py` prints a per-stage timing table at the end; use it when touching performance.

Container: `docker compose up -d --build` (CPU) or add `-f docker-compose.gpu.yml` (CUDA). The image bakes all model weights, so a Dockerfile change that touches the pip/weights layers costs a 10–20 min rebuild.

## Architecture

Three layers, deliberately decoupled by process boundaries:

1. **`nunif/` (vendored, do not modify)** provides iw3: the Depth-Anything-V2-Small depth model and the stereo synthesis CLI. `stylize.py` imports it (`sys.path.insert(0, NUNIF)`); `convert3d.sh` and `serve.py` shell out to `python -m iw3.cli`. Model weights download on first use into `nunif/iw3/pretrained_models/hub/`.

2. **`stylize.py`** is a self-contained per-frame pipeline: `probe` → `read_frames` (raw RGB over an ffmpeg pipe) → `get_cached` (depth via iw3 with EMA smoothing + person mask via rembg `u2net_human_seg`, both cached as half-res uint8 `.npz` in `.stylize_cache/` keyed by file hash + frame count + EMA) → `render_one` (effect `voxel_effect` / `contour_effect`, then `compose`: procedural `bg_*`, `layout_dims`, optional `face_protect` which auto-detects the neck boundary) → `encode` (frames dir → H.264 + WebM via ffmpeg, audio passthrough). The cache means model changes need `--refresh-cache`; effect/background/layout changes do not.

   There are **two entry paths into the same `render_one`**. A video goes through `get_cached` (two full model passes over the clip, then the render loop). A **still** — `--frame N`, or any image input — goes through `read_one_frame` + `get_cached_one`, which run exactly **one** depth and one mask inference and return in seconds. Stills reuse the full-clip `dm_<key>.npz` when it already covers the frame, and otherwise write a *separate* `one_<key>_f<N>.npz`; they must never write `dm_`, since a 1-frame array there would silently discard an existing full-clip pass and cost the next render another 8 minutes.

3. **`serve.py` + `web/index.html`** is a FastAPI wrapper that builds the exact CLI command in `_build_cmd` (the two pipelines' flags live there and must mirror `convert3d.sh`), runs it as a subprocess in its own session, and scrapes tqdm/iw3 progress lines from stderr into weighted stage percentages (`STAGE_WEIGHTS`). One job at a time (409 otherwise). Jobs persist to `jobs.json` only so a restarted server can mark orphans as errors. The single HTML page keeps the bearer token in `localStorage`; `/api/*` and `/media/*` require it when `AUTH_TOKEN` is set, `/` and `/api/health` never do.

4. **`web/camera/`** is a static, client-only live camera (served by `serve.py` at `/camera/`, outside the token gate, and published to GitHub Pages by `.github/workflows/camera-pages.yml`). `segment.js` runs MediaPipe's selfie segmenter (pinned `@mediapipe/tasks-vision` version, CDN) on a 256 px copy of each frame and ports `head_bbox` + the silhouette half of `neck_boundary`; `palette.js` ports `build_palette`; `shaders.js` ports the effects, `bg_*`, `compose`, `face_protect`, `chromatic` and `grain` to GLSL; `render.js` runs the passes (edges → fx → composite → canvas); `recorder.js` records the canvas with MediaRecorder and saves via the share sheet / download. **Changing a look in `stylize.py` means changing `shaders.js` too** if the live preview should keep matching. Deliberate differences: depth is a silhouette-based stand-in (`nearness()`), voxel tiling is a quadtree, bokeh positions differ. A studio-served page can upload the raw take (`/api/upload` with `source=camera`, which re-encodes to constant 30 fps because MediaRecorder output is VFR with a ~1000 fps nominal rate) and start the exact render.

   Orientation rule in the shaders: texture row 0 is the top of the image and offscreen passes treat `gl_FragCoord.y` as a y-down row; only the final pass to the canvas flips. `uMirror` flips camera-space lookups only; segmenter stats are in unmirrored camera space and are mirrored in `render.js`.

Everything mutable lives under `DATA_DIR` (`uploads/`, `outputs/`, `.stylize_cache/`, `jobs.json`); defaults to the repo dir locally and `/app/data` in the image. Mount a volume there or renders vanish on restart.

## Gotchas

- `probe()` returns **display** dimensions: phone clips carry a rotation tag and ffmpeg auto-rotates the pixel stream, so reshaping with stored width/height scrambles frames. Keep using display dims.
- iw3 time args want `hh:mm:ss`, `mm:ss`, or integer seconds; a float like `2.0` is rejected (see the preview handling in `_build_cmd`).
- Never pass `-ss 0` to ffmpeg for frame 0: on a JPEG (the `image2` demuxer) it emits **zero bytes and no error**, so every photo render fails silently. `read_one_frame` omits `-ss` entirely when `idx == 0`. PNG (`png_pipe`) and mp4 are unaffected.
- `probe()` detects a still from the container `format_name` (`image2` / `*_pipe`), not from fps or duration: ffprobe invents a nominal `25/1` rate for a JPEG and omits `format.duration` for a PNG (which used to make `probe()` raise `KeyError`).
- Camera recording: never put plain `video/mp4` before WebM in `recorder.js`'s list. A browser without an H.264 encoder honours it with VP9 inside MP4, a `.mp4` that photo apps reject.
- `/api/health` must stay importable without torch: the camera page uses it to detect the studio.
- `convert3d.sh` hardcodes `cd /root/3dvid/nunif`; it is written for the qm-personal host and the container, not for arbitrary checkouts.
- `.gitignore` excludes all media, caches, `venv/`, `.env`; `.dockerignore` additionally excludes `nunif/` because the Dockerfile clones it itself.

## Deployment targets

- **Fly.io** (stable URL, but the expensive one): app `3dvid`, `fly.toml` in repo, `~/.fly/bin/flyctl deploy --remote-only --ha=false` (`--remote-only` builds on Fly, so it needs no local disk). **A `performance-8x` left running bills ~$264/month and once cost $225**, so `fly.toml` now defaults to `shared-cpu-4x` with `auto_stop_machines = "suspend"`. Stopping a machine does not stop volume billing. Read the cost warning at the top of `fly.toml` before changing either; a render only survives idle while a browser tab polls `/api/jobs`.
- **qm-personal** (SSH host, free, 32+ cores): an on-demand microVM that stops when idle and loses all processes. `deploy/start.sh` restarts serve.py + a Cloudflare quick tunnel in tmux and writes the (random, per-restart) URL to `public_url.txt`; a `.bashrc` hook runs it on login. QUIC is blocked there, hence `--protocol http2` in `deploy/tunnel.sh`.
- **Kaggle** (free GPU, ~30 h/week): `kaggle/3dvid.ipynb` clones the repo **from GitHub** (`REPO_REF`, default `main`), so unpushed changes never reach it. It reuses Kaggle's preinstalled CUDA torch/opencv and installs only nunif's requirements + `rembg[gpu]`. Only 4 CPUs, so the GPU speeds up depth/mask but effects stay slow; full renders belong in a background commit run. CLI only: no `serve.py` there.
- **AWS EC2 GPU** (`deploy/ec2.sh`, `aws-launch.sh`): blocked until the account's G-instance quota is raised; the CPU path of the same script works on any Ubuntu box with Docker.
- Model download and CUDA-vs-CPU selection are automatic: `serve.py` passes `--gpu 0` to iw3 when torch sees CUDA, else `-1`; the Dockerfile picks the wheel index via `TORCH_INDEX`.
