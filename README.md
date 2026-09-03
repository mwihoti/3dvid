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

## Notes

- CPU only: ~8 min per 30 s stylise render on 32 cores, plus a one-off depth+mask pass per clip.
- Half-SBS / half-TB output looks like a duplicated image on a flat screen — that is correct; it resolves in a headset. Use anaglyph to check depth on a normal screen.
