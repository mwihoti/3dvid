#!/usr/bin/env python3
"""
stylize.py - turn a talking-head clip into a stylised portfolio intro.

Pipeline (per frame):
  depth (iw3 / Depth-Anything-V2, EMA-smoothed)  ->  person mask (rembg)
  ->  effect layer (voxel | contour | both)  ->  procedural background
  ->  layout framing  ->  H.264 + WebM encode

Depth and mask are cached to .npz so re-renders skip both model passes.
Does not modify nunif/ - it imports iw3 the same way convert3d.sh shells out to it.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from collections import OrderedDict
from contextlib import contextmanager

import cv2
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
NUNIF = os.path.join(HERE, "nunif")
sys.path.insert(0, NUNIF)

CACHE_DIR = os.environ.get("STYLIZE_CACHE_DIR", os.path.join(HERE, ".stylize_cache"))
CACHE_SCALE = 0.5          # depth/mask cached at half res (uint8) to keep .npz sane
DEPTH_MODEL = "Any_V2_S"   # same model convert3d.sh uses

# ---------------------------------------------------------------- timing ----
TIMINGS = OrderedDict()

@contextmanager
def stage(name):
    t = time.time()
    yield
    TIMINGS[name] = TIMINGS.get(name, 0.0) + (time.time() - t)

def print_timings(total_frames):
    print("\n--- stage timing " + "-" * 44)
    tot = sum(TIMINGS.values())
    for k, v in TIMINGS.items():
        per = f"{v / total_frames * 1000:7.1f} ms/f" if total_frames else ""
        print(f"  {k:<26s} {v:7.2f}s  {v / tot * 100:5.1f}%  {per}")
    print(f"  {'TOTAL':<26s} {tot:7.2f}s")

# ---------------------------------------------------------------- video I/O ----
def probe(path):
    """Returns DISPLAY dimensions. Phone clips are often stored landscape with a
    90/270 rotation tag; ffmpeg auto-rotates the pixels it emits, so the raw
    pipe is (h, w) not (w, h) - reshaping with stored dims scrambles every frame."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames:stream_tags=rotate"
         ":stream_side_data=rotation:format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    dur = float(d["format"]["duration"])
    n = int(s.get("nb_frames") or round(dur * fps))
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if not rot and s.get("tags", {}).get("rotate"):
        rot = int(float(s["tags"]["rotate"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    return dict(w=w, h=h, fps=fps, n=n, dur=dur, rotation=rot)

def read_frames(path, limit=None):
    """Yield RGB uint8 frames via ffmpeg rawvideo pipe."""
    info = probe(path)
    w, h = info["w"], info["h"]
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if limit:
        cmd += ["-frames:v", str(limit)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    nbytes = w * h * 3
    try:
        while True:
            buf = p.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close(); p.wait()

# ---------------------------------------------------------------- cache ----
def cache_key(video):
    st = os.stat(video)
    raw = f"v2|{os.path.abspath(video)}|{st.st_size}|{int(st.st_mtime)}|{CACHE_SCALE}|{DEPTH_MODEL}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def compute_depth(video, n_frames, ema_alpha, batch=8):
    """iw3 Depth-Anything-V2 -> per-frame [0,1] (1 = nearest), EMA-smoothed."""
    import torch
    from iw3.depth_model_factory import create_depth_model

    model = create_depth_model(DEPTH_MODEL)
    model.load(gpu=-1)                       # CPU: torch here is +cpu, no CUDA
    info = probe(video)
    ch, cw = int(info["h"] * CACHE_SCALE), int(info["w"] * CACHE_SCALE)
    out = np.zeros((n_frames, ch, cw), np.uint8)

    buf, idx, ema = [], 0, None
    pbar = tqdm(total=n_frames, desc="depth", unit="f")

    def flush(frames):
        nonlocal idx, ema
        if not frames:
            return
        x = torch.stack([torch.from_numpy(f.copy()).permute(2, 0, 1).float() / 255. for f in frames])
        with torch.inference_mode():
            d = model.infer(x)
        d = d.squeeze(1).float().cpu().numpy()
        for dd in d:
            lo, hi = float(dd.min()), float(dd.max())
            dn = (dd - lo) / max(hi - lo, 1e-6)          # 1.0 = nearest
            ema = dn if ema is None else (ema_alpha * dn + (1 - ema_alpha) * ema)
            out[idx] = np.clip(cv2.resize(ema, (cw, ch), interpolation=cv2.INTER_AREA) * 255, 0, 255).astype(np.uint8)
            idx += 1; pbar.update(1)

    for fr in read_frames(video, n_frames):
        buf.append(fr)
        if len(buf) == batch:
            flush(buf); buf = []
    flush(buf); pbar.close()
    return out

def compute_masks(video, n_frames, model_name="u2net_human_seg"):
    from rembg import new_session, remove
    from PIL import Image
    sess = new_session(model_name)
    info = probe(video)
    ch, cw = int(info["h"] * CACHE_SCALE), int(info["w"] * CACHE_SCALE)
    out = np.zeros((n_frames, ch, cw), np.uint8)
    for i, fr in enumerate(tqdm(read_frames(video, n_frames), total=n_frames, desc="mask", unit="f")):
        m = np.asarray(remove(Image.fromarray(fr), session=sess, only_mask=True))
        out[i] = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_AREA)
    return out

def get_cached(video, n_frames, ema_alpha, refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    f = os.path.join(CACHE_DIR, f"dm_{cache_key(video)}.npz")
    if os.path.exists(f) and not refresh:
        z = np.load(f)
        if z["depth"].shape[0] >= n_frames:
            print(f"cache hit  {os.path.relpath(f, HERE)}  ({z['depth'].shape[0]} frames)")
            return z["depth"][:n_frames], z["mask"][:n_frames]
        print(f"cache has {z['depth'].shape[0]} frames, need {n_frames} - recomputing")
    with stage("depth (model)"):
        depth = compute_depth(video, n_frames, ema_alpha)
    with stage("mask (model)"):
        mask = compute_masks(video, n_frames)
    np.savez_compressed(f, depth=depth, mask=mask)
    print(f"cache write {os.path.relpath(f, HERE)}  ({os.path.getsize(f)/1e6:.1f} MB)")
    return depth, mask

# ---------------------------------------------------------------- palette ----
def build_palette(frame, mask=None, k=24, fg_share=0.7):
    """k-means on the first frame. Sampling is biased toward the person, or a
    bright background eats most of the 24 clusters and skin goes flat."""
    from sklearn.cluster import MiniBatchKMeans
    rs = np.random.RandomState(0)
    flat = frame.reshape(-1, 3).astype(np.float32)
    N = 40000
    if mask is not None:
        fg = np.where(mask.reshape(-1) > 0.5)[0]
        bg = np.where(mask.reshape(-1) <= 0.5)[0]
        if len(fg) and len(bg):
            nf = min(len(fg), int(N * fg_share))
            nb = min(len(bg), N - nf)
            px = np.concatenate([flat[rs.choice(fg, nf, replace=False)],
                                 flat[rs.choice(bg, nb, replace=False)]])
        else:
            px = flat[rs.choice(len(flat), min(N, len(flat)), replace=False)]
    else:
        px = flat[rs.choice(len(flat), min(N, len(flat)), replace=False)]
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=4, max_iter=60).fit(px)
    return np.clip(km.cluster_centers_, 0, 255).astype(np.float32)

def apply_palette(colors, palette):
    d = ((colors[:, None, :] - palette[None, :, :]) ** 2).sum(2)
    return palette[d.argmin(1)]

# ---------------------------------------------------------------- effect A: voxel ----
def voxel_effect(rgb, depth, mask, palette, min_cell=8, max_cell=24,
                 gran=4, face_scale=1.5, face_contrast=1.18):
    """Depth-driven adaptive cube tiling. Near -> min_cell, far -> max_cell."""
    H, W = depth.shape
    fg = mask > 0.35

    # face region = person AND within the top 30% of the person's vertical extent
    ys = np.where(fg.any(1))[0]
    face = np.zeros_like(fg)
    if len(ys):
        face[ys[0]:ys[0] + max(1, int(0.30 * (ys[-1] - ys[0] + 1)))] = True
        face &= fg

    # stretch depth across the person's own range, else the masked subject is
    # uniformly "near" and every cell lands on min_cell
    if fg.any():
        lo, hi = np.percentile(depth[fg], [5, 95])
        dn = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        dn = depth
    size = max_cell + (min_cell - max_cell) * dn           # dn 1 (near) -> min_cell
    size = np.where(face, size / face_scale, size)
    size = np.clip(size, gran, max_cell)

    gh, gw = H // gran, W // gran
    steps = np.clip(np.round(cv2.resize(size, (gw, gh), interpolation=cv2.INTER_AREA) / gran),
                    1, max_cell // gran).astype(np.int32)

    # greedy raster tiling on the coarse grid
    idmap = np.full((gh, gw), -1, np.int32)
    ys_, xs_, ss_ = [], [], []
    cid = 0
    for gy in range(gh):
        row = idmap[gy]
        for gx in range(gw):
            if row[gx] >= 0:
                continue
            s = min(int(steps[gy, gx]), gh - gy, gw - gx)
            idmap[gy:gy + s, gx:gx + s] = cid
            ys_.append(gy); xs_.append(gx); ss_.append(s); cid += 1
    cy = np.array(ys_, np.int32) * gran
    cx = np.array(xs_, np.int32) * gran
    cs = np.array(ss_, np.int32) * gran

    # per-cell mean colour via integral image (O(1) per cell, fully vectorised)
    integ = cv2.integral(rgb.astype(np.float32))            # (H+1, W+1, 3)
    y2 = np.minimum(cy + cs, H); x2 = np.minimum(cx + cs, W)
    area = ((y2 - cy) * (x2 - cx)).astype(np.float32)[:, None]
    col = (integ[y2, x2] - integ[cy, x2] - integ[y2, cx] + integ[cy, cx]) / area

    # face cells: slight contrast boost so features survive quantisation
    face_cell = face[np.clip(cy + cs // 2, 0, H - 1), np.clip(cx + cs // 2, 0, W - 1)]
    col[face_cell] = np.clip((col[face_cell] - 128.0) * face_contrast + 128.0, 0, 255)
    col = apply_palette(col, palette)

    # expand coarse maps to full res
    up = lambda a: np.repeat(np.repeat(a, gran, 0), gran, 1)[:H, :W]
    ids = up(idmap)
    oy, ox, sz = up(np.array(ys_)[idmap] * gran), up(np.array(xs_)[idmap] * gran), up(np.array(ss_)[idmap] * gran)
    out = col[ids]

    Y, X = np.mgrid[0:H, 0:W]
    ly, lx = Y - oy, X - ox
    t = np.maximum(1, sz // 4)      # top face depth
    b = np.maximum(1, sz // 5)      # side face depth

    shade = np.ones((H, W), np.float32)
    shade[ly < t] = 1.20                      # top  +20%
    shade[(lx < b) & (ly >= t)] = 0.85        # left -15%
    shade[(lx >= sz - b) & (ly >= t)] = 0.70  # right -30%
    shade[(ly == 0) | (lx == 0)] = 0.35       # 1px dark seam
    return np.clip(out * shade[..., None], 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- effect B: neon contour ----
NAVY = np.array([0x0a, 0x12, 0x30], np.float32)
RAMP = [np.array([0xff, 0x6a, 0x00], np.float32),   # orange
        np.array([0xff, 0x2d, 0x95], np.float32),   # pink
        np.array([0x8a, 0x2b, 0xff], np.float32)]   # violet
CYAN = np.array([0x39, 0xc6, 0xff], np.float32)

def _ramp(u):
    u = np.clip(u, 0, 1) * 2.0
    i = np.clip(u.astype(np.int32), 0, 1)
    f = (u - i)[..., None]
    a = np.stack(RAMP)[i]; b = np.stack(RAMP)[i + 1]
    return a * (1 - f) + b * f

def contour_effect(rgb, depth, mask, frame_idx, bands=12, glow=9):
    H, W = depth.shape
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    lum = cv2.GaussianBlur(lum, (5, 5), 0)
    field = 0.70 * lum + 0.30 * depth
    phase = frame_idx * 0.05                                  # threshold drift
    q = np.floor(field * bands + phase)

    edge = np.zeros((H, W), np.float32)
    edge[:, 1:] += (q[:, 1:] != q[:, :-1])
    edge[1:, :] += (q[1:, :] != q[:-1, :])
    edge = np.clip(edge, 0, 1)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)                  # anti-alias

    band_u = (q % bands) / max(bands - 1, 1)
    line_col = _ramp(band_u)

    out = np.repeat(NAVY[None, None, :], H, 0).repeat(W, 1).copy()
    out += line_col * edge[..., None]
    k = glow | 1
    bloom = cv2.GaussianBlur(edge, (k, k), 0)[..., None] * line_col
    out += bloom * 0.55

    # cyan rim on the brighter side of the face
    gx = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
    rim = np.clip(-np.sign(cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=5)) * gx, 0, None)
    rim = cv2.GaussianBlur(rim, (0, 0), 3.0)
    if rim.max() > 1e-6:
        rim /= rim.max()
    out += CYAN * (rim ** 1.5)[..., None] * 1.4
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- backgrounds ----
_NOISE = {}

def _fbm(h, w, t, octaves=4, seed=0, ds=2):
    """Rendered at 1/ds res and upsampled - fBm is low-frequency, so this is
    visually identical and ~4x cheaper."""
    if ds > 1:
        small = _fbm(h // ds, w // ds, t, octaves, seed, ds=1)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return _fbm_impl(h, w, t, octaves, seed)

def _fbm_impl(h, w, t, octaves=4, seed=0):
    """Cheap animated value-noise fBm: small random grids upsampled + scrolled."""
    key = (h, w, seed, octaves)
    if key not in _NOISE:
        rs = np.random.RandomState(seed)
        # oversized canvases (2x) so the scroll window never wraps -> no seam
        _NOISE[key] = [cv2.resize(rs.rand(4 << o, 4 << o).astype(np.float32),
                                  (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                       for o in range(octaves)]
    acc = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    for o, n in enumerate(_NOISE[key]):
        sx = int(abs(((t * 9 * (o + 1)) % (2 * w)) - w))
        sy = int(abs(((t * 5 * (o + 1)) % (2 * h)) - h))
        acc += amp * n[sy:sy + h, sx:sx + w]
        norm += amp; amp *= 0.5
    return acc / norm

def bg_plasma(W, H, t, head, ds=2):
    """Rendered at 1/ds and upsampled - every term here is smooth, and the
    full-res transcendentals were ~2/3 of total frame cost."""
    if ds > 1:
        hx, hy, hw = head
        small = bg_plasma(W // ds, H // ds, t, (hx / ds, hy / ds, hw / ds), ds=1)
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    hx, hy, hw = head
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((H, W, 3), np.float32)
    out[:] = np.array([4, 5, 14], np.float32)

    r = max(hw * 1.6, 80.0)
    ang = t * 0.25                                            # slow rotation
    dx, dy = (X - hx) / r, (Y - hy) / r
    rx = dx * np.cos(ang) - dy * np.sin(ang)
    ry = dx * np.sin(ang) + dy * np.cos(ang)
    d = np.sqrt(rx * rx + ry * ry)

    n = _fbm(H, W, t, seed=1)
    orb = np.exp(-(d ** 2) * 1.7) * (0.55 + 0.45 * n)
    out += np.array([0x1a, 0x4d, 0xff], np.float32) * orb[..., None] * 1.5   # electric blue

    th = np.arctan2(ry, rx)
    warp = n * 7.0 + np.sin(th * 3.0 + t * 0.7) * 1.3      # noise + angular wobble
    fil = np.abs(np.sin((d * 4.2 - t * 1.1) * np.pi + warp))
    fil = np.clip(1.0 - fil, 0, 1) ** 7 * np.exp(-(d ** 2) * 1.1)
    out += np.array([0x1a, 0x4d, 0xff], np.float32) * fil[..., None] * 2.2
    out += np.array([160, 240, 255], np.float32) * (fil ** 3)[..., None] * 1.8   # cyan cores

    rim = np.exp(-(((d - 1.05) * 3.2) ** 2)) * np.clip(ry, 0, 1)
    out += np.array([0xff, 0x2a, 0xc0], np.float32) * rim[..., None] * 0.42      # magenta bottom
    return np.clip(out, 0, 255).astype(np.uint8)

def bg_grid(W, H, t, head):
    out = np.zeros((H, W, 3), np.float32)
    out[:] = np.array([0x04, 0x26, 0x2e], np.float32)
    CY = np.array([0x19, 0xd4, 0xc8], np.float32)
    horizon = int(H * 0.46)
    vx = W * 0.5
    lay = np.zeros((H, W), np.float32)

    # floor + ceiling receding lines (scroll toward camera)
    for sgn, base in ((1, horizon), (-1, horizon)):
        for i in range(1, 22):
            u = (i + (t * 0.9) % 1.0) / 22.0
            y = base + sgn * (H * 0.60) * (u ** 2.4)
            if 0 <= y < H:
                cv2.line(lay, (0, int(y)), (W, int(y)), float(max(0.05, 1.0 - u) ** 1.6), 1, cv2.LINE_AA)
    # one-point perspective verticals converging on the vanishing point
    for i in range(-16, 17):
        x_far = vx + i * (W * 0.030)
        x_near = vx + i * (W * 0.42)
        cv2.line(lay, (int(x_far), horizon), (int(x_near), H), 0.42, 1, cv2.LINE_AA)
        cv2.line(lay, (int(x_far), horizon), (int(x_near), 0), 0.30, 1, cv2.LINE_AA)

    fade = np.clip(np.abs(np.mgrid[0:H, 0:W][0] - horizon) / (H * 0.55), 0, 1)[..., None]
    out += CY * (lay[..., None] * (0.25 + 0.75 * fade)) * 1.35

    # drifting bokeh
    rs = np.random.RandomState(7)
    cols = [np.array(c, np.float32) for c in ([0x19, 0xd4, 0xc8], [0xff, 0xd8, 0x4d], [0xff, 0x6f, 0xb5])]
    bok = np.zeros((H, W, 3), np.float32)
    for i in range(30):
        z = (rs.rand() + t * 0.05) % 1.0                       # 0 far -> 1 near
        x = int((rs.rand() + t * 0.012 * (0.4 + z)) % 1.0 * W)
        y = int((rs.rand() + t * 0.020 * (0.4 + z)) % 1.0 * H)
        rad = int(3 + 26 * z)
        cv2.circle(bok, (x, y), rad, (cols[i % 3] * (0.10 + 0.30 * z)).tolist(), -1, cv2.LINE_AA)
    k = 31
    out += cv2.GaussianBlur(bok, (k, k), 0) * 2.6
    return np.clip(out, 0, 255).astype(np.uint8)

def bg_dark(W, H, t, head):
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((X - W / 2) / (W / 2)) ** 2 + ((Y - H / 2) / (H / 2)) ** 2)
    v = np.clip(1.0 - 0.72 * d ** 1.7, 0, 1)
    return np.clip(np.array([0x05, 0x07, 0x0f], np.float32) * (0.35 + 1.9 * v)[..., None], 0, 255).astype(np.uint8)

BACKGROUNDS = {"plasma": bg_plasma, "grid": bg_grid, "dark": bg_dark}

# ---------------------------------------------------------------- layout / composite ----
LAYOUTS = {"portrait": (1080, 1920, 0.50), "wide": (1920, 1080, 0.72), "square": (1080, 1080, 0.50)}

def layout_dims(layout, sw, sh):
    """native = keep the source frame size exactly (h264 needs even dims)."""
    if layout == "native":
        return (sw - sw % 2, sh - sh % 2, 0.50)
    return LAYOUTS[layout]

def head_bbox(mask):
    fg = mask > 0.35
    ys = np.where(fg.any(1))[0]
    if not len(ys):
        return mask.shape[1] * 0.5, mask.shape[0] * 0.35, mask.shape[1] * 0.4
    top = ys[0]
    band = fg[top:top + max(1, int(0.30 * (ys[-1] - top + 1)))]
    xs = np.where(band.any(0))[0]
    if not len(xs):
        return mask.shape[1] * 0.5, top + 20.0, mask.shape[1] * 0.4
    return (xs[0] + xs[-1]) / 2.0, top + band.shape[0] * 0.45, float(xs[-1] - xs[0] + 1)

def neck_boundary(fg, rgb=None):
    """Row just below the neck (the collar line).
    Step 1: jaw = narrowest silhouette row between head and shoulders.
    Step 2: from there, walk down while the central strip is still skin -
            that is the neck - and stop where clothing starts. The silhouette
            alone cannot do step 2 in a close-up: shoulders widen right under
            the chin, so width finds the jaw, not the collar."""
    ys = np.where(fg.any(1))[0]
    if len(ys) < 20:
        return None
    top, bot = ys[0], ys[-1]
    Hp = bot - top + 1
    width = fg.sum(1).astype(np.float32)
    width = np.convolve(width, np.ones(9) / 9, mode="same")
    lo, hi = top + int(0.25 * Hp), top + int(0.75 * Hp)
    if hi - lo < 10:
        return None
    jaw = lo + int(np.argmin(width[lo:hi]))
    if width[jaw] > 0.85 * width[top:jaw].max():
        return None
    if rgb is None:
        return jaw + int(0.12 * Hp)

    # skin model sampled from THIS face (central strip, cheeks -> jaw) rather than
    # fixed thresholds: chroma barely moves in the shadow under the chin, but
    # fixed Cr/Cb bounds still lose it and the walk stops at the beard line
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    xs = np.where(fg[jaw])[0]
    if not len(xs):
        return jaw + int(0.12 * Hp)
    cx = int(xs.mean()); half = max(8, int(0.20 * (xs[-1] - xs[0] + 1)))
    x0, x1 = max(0, cx - half), min(fg.shape[1], cx + half)
    fy0 = top + int(0.35 * Hp)
    samp = ycc[fy0:jaw, x0:x1][fg[fy0:jaw, x0:x1]]
    if len(samp) < 50:
        return jaw + int(0.12 * Hp)
    cr0, cb0 = np.median(samp[:, 1]), np.median(samp[:, 2])
    tol_cr = max(10.0, 3.0 * np.median(np.abs(samp[:, 1] - cr0)))
    tol_cb = max(10.0, 3.0 * np.median(np.abs(samp[:, 2] - cb0)))
    skin = (np.abs(ycc[..., 1] - cr0) < tol_cr) & (np.abs(ycc[..., 2] - cb0) < tol_cb) & fg
    # walk down while the skin band stays about as wide as the neck. A V-neck or
    # open collar narrows the band; a crew neck / hoodie cuts it off - either way
    # the collar line is where skin width collapses relative to the neck.
    skin_w = skin.sum(1).astype(np.float32)
    ref = np.median(skin_w[jaw:min(bot, jaw + max(6, int(0.04 * Hp)))])
    if ref < 4:
        return jaw + int(0.12 * Hp)
    end = jaw
    miss = 0
    for y in range(jaw, min(bot, jaw + int(0.25 * Hp))):
        if skin_w[y] >= 0.55 * ref:
            end = y; miss = 0
        else:
            miss += 1
            if miss > int(0.03 * Hp) + 3:        # beard shadow / collar edge rows
                break
    return end + int(0.03 * Hp)


def face_protect(mask, frac="auto", soft=61, rgb=None, pad=0.0):
    """Soft mask over the head (+ neck). Used to keep the face as original pixels
    while the body / background get the effect. frac='auto' finds the neck from
    the silhouette; a number is a fraction of the person's height from the top.

    pad extends the cut downwards by that fraction of the person's height. The
    skin walk in neck_boundary stops where skin chroma stops, which on a bearded
    or shadowed jaw is the beard line rather than the collar - the slack keeps
    the chin and beard inside the mask."""
    fg = mask > 0.35
    ys = np.where(fg.any(1))[0]
    fm = np.zeros(mask.shape, np.float32)
    if len(ys):
        top = ys[0]
        person_h = ys[-1] - top + 1
        cut = None
        if frac == "auto":
            cut = neck_boundary(fg, rgb)
        if cut is None:
            f = 0.62 if frac == "auto" else float(frac)
            cut = top + max(1, int(f * person_h))
        cut = min(int(cut + pad * person_h), mask.shape[0])
        fm[top:cut] = 1.0
        fm *= mask
        k = soft | 1
        fm = cv2.GaussianBlur(fm, (k, k), 0)
    return fm


def cover_scale(img, CW, CH):
    """Scale to cover the canvas and centre-crop."""
    h, w = img.shape[:2]
    sc = max(CW / w, CH / h)
    r = cv2.resize(img, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))),
                   interpolation=cv2.INTER_AREA)
    y0 = (r.shape[0] - CH) // 2; x0 = (r.shape[1] - CW) // 2
    return r[y0:y0 + CH, x0:x0 + CW]


def unsharp(img, amount, radius=1.6):
    """Unsharp mask. Restores edge contrast on soft sources; cannot invent detail."""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def chromatic(img, px=1.6):
    b, g, r = img[..., 2], img[..., 1], img[..., 0]
    M = lambda s: np.float32([[1, 0, s], [0, 1, 0]])
    H, W = img.shape[:2]
    r = cv2.warpAffine(r, M(px), (W, H), borderMode=cv2.BORDER_REPLICATE)
    b = cv2.warpAffine(b, M(-px), (W, H), borderMode=cv2.BORDER_REPLICATE)
    return np.stack([r, g, b], -1)

def grain(img, amt, rs):
    return np.clip(img.astype(np.float32) + rs.normal(0, amt, img.shape[:2])[..., None], 0, 255).astype(np.uint8)

def compose(layer, alpha, bgname, layout, t, frame_idx, cab, grn, rs, src_bg=None):
    sh, sw = layer.shape[:2]
    CW, CH, ax = layout_dims(layout, sw, sh)
    sc = CH / sh
    nw, nh = max(1, int(round(sw * sc))), CH
    lay = cv2.resize(layer, (nw, nh), interpolation=cv2.INTER_AREA)
    al = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)[..., None]

    hx, hy, hw = head_bbox(alpha)
    ox = int(round(CW * ax - nw / 2))
    canvas_head = (ox + hx * sc, hy * sc, max(hw * sc, 40.0))
    if src_bg is not None:
        # filmed background: blurred cover-fill, then the frame placed with the
        # SAME transform as the subject so the stylised wall lines up behind them
        bg = cv2.GaussianBlur(cover_scale(src_bg, CW, CH), (0, 0), 18).astype(np.float32) * 0.55
        sb = cv2.resize(src_bg, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
        bx0, bx1 = max(0, ox), min(CW, ox + nw)
        if bx1 > bx0:
            bg[:, bx0:bx1] = sb[:, bx0 - ox:bx1 - ox]
    else:
        bg = BACKGROUNDS[bgname](CW, CH, t, canvas_head).astype(np.float32)

    x0, x1 = max(0, ox), min(CW, ox + nw)
    if x1 > x0:
        s0, s1 = x0 - ox, x1 - ox
        roi = bg[:, x0:x1]
        bg[:, x0:x1] = roi * (1 - al[:, s0:s1]) + lay[:, s0:s1].astype(np.float32) * al[:, s0:s1]

    out = np.clip(bg, 0, 255).astype(np.uint8)
    if cab:
        out = chromatic(out)
    if grn:
        out = grain(out, grn, rs)
    return out

# ---------------------------------------------------------------- encode ----
def probe_audio(path):
    """(has_audio, codec_name) for the first audio stream."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
                       capture_output=True, text=True).stdout.strip()
    return (bool(r), r)


def encode(frames_dir, fps, out_base, src, keep_audio, av1=False, crf=24, webm_crf=36):
    results = []
    common = ["-framerate", f"{fps:.6f}", "-i", os.path.join(frames_dir, "%06d.png")]

    has_audio, acodec = probe_audio(src)
    keep_audio = keep_audio and has_audio
    if keep_audio:
        # copy when the source codec is already legal in MP4 - re-encoding a
        # 155 kbps AAC down to 96 kbps is audible loss for no reason
        acopy = ["-c:a", "copy"] if acodec in ("aac", "mp3", "ac3") else ["-c:a", "aac", "-b:a", "192k"]
        audio = ["-i", src, "-map", "0:v", "-map", "1:a", *acopy, "-shortest"]
    else:
        audio = ["-an"]

    mp4 = out_base + ".mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *common, *audio,
                    "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4], check=True)
    results.append(mp4)

    webm = out_base + ".webm"
    if av1:
        vargs = ["-c:v", "libsvtav1", "-crf", str(webm_crf + 2), "-preset", "6"]
    else:
        vargs = ["-c:v", "libvpx-vp9", "-crf", str(webm_crf), "-b:v", "0", "-row-mt", "1"]
    # WebM cannot carry AAC, so Opus is forced here; 128k is transparent for speech
    wa = (["-i", src, "-map", "0:v", "-map", "1:a", "-c:a", "libopus", "-b:a", "128k", "-shortest"]
          if keep_audio else ["-an"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", *common, *wa, *vargs,
                    "-pix_fmt", "yuv420p", webm], check=True)
    results.append(webm)
    return results

# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description="Stylise a talking-head clip into a portfolio intro.")
    ap.add_argument("input")
    ap.add_argument("--effect", choices=["voxel", "contour", "both", "clean"], default="voxel",
                    help="clean = person kept as original pixels, only the background changes")
    ap.add_argument("--bg", choices=["plasma", "grid", "dark", "source", "filmed"], default="plasma",
                    help="source = the filmed background with the chosen effect applied to it; "
                         "filmed = the filmed background left as shot (pair with --face-mode only "
                         "for a real room behind a stylised head)")
    ap.add_argument("--face-clear", action="store_true",
                    help="alias for --face-mode clear")
    ap.add_argument("--face-mode", choices=["off", "clear", "only"], default=None,
                    help="off = effect over the whole person; clear = face kept as original "
                         "pixels, effect on body + background; only = effect on the face "
                         "alone, body kept as original pixels")
    ap.add_argument("--face-pad", type=float, default=0.05,
                    help="grow the face mask downwards by this fraction of the person's "
                         "height (0-0.4). The neck walk stops at a beard or a shadowed "
                         "jaw, so a little slack keeps the chin inside the mask")
    ap.add_argument("--sharpen", type=float, default=0.0,
                    help="unsharp-mask the original-pixel regions (clean / face-clear), 0-1.5")
    ap.add_argument("--face-frac", default="auto",
                    help="'auto' = clear down to just below the neck (detected from the "
                         "silhouette); or a number = fraction of the person's height, e.g. 0.55 for chin")
    ap.add_argument("--layout", choices=["portrait", "wide", "square", "native"], default="wide")
    ap.add_argument("--bands", type=int, default=12)
    ap.add_argument("--min-cell", type=int, default=8)
    ap.add_argument("--max-cell", type=int, default=24)
    ap.add_argument("--preview", type=float, default=0, help="render only first N seconds")
    ap.add_argument("--frame", type=int, default=None, help="dump a single PNG and exit")
    ap.add_argument("--glow", type=int, default=9, help="contour bloom radius (odd)")
    ap.add_argument("--ema", type=float, default=0.4, help="depth EMA alpha")
    ap.add_argument("--feather", type=int, default=6, help="mask feather px")
    ap.add_argument("--chromatic", action="store_true")
    ap.add_argument("--grain", type=float, default=0.0, help="film grain sigma, e.g. 4")
    ap.add_argument("--keep-audio", action="store_true",
                    help="(default) carry the source audio through untouched")
    ap.add_argument("--no-audio", action="store_true", help="strip the audio track")
    ap.add_argument("--av1", action="store_true", help="WebM via SVT-AV1 instead of VP9")
    ap.add_argument("--crf", type=int, default=24, help="H.264 CRF (higher = smaller)")
    ap.add_argument("--webm-crf", type=int, default=36, help="VP9/AV1 CRF")
    ap.add_argument("--target-mb", type=float, default=5.0, help="size budget per 15s, for the report")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    face_mode = a.face_mode or ("clear" if a.face_clear else "off")
    info = probe(a.input)
    n = info["n"]
    if a.frame is not None:
        n = min(n, a.frame + 1)
    elif a.preview:
        n = min(n, int(round(a.preview * info["fps"])))
    print(f"input {a.input}  {info['w']}x{info['h']}  {info['fps']:.3f} fps  using {n} frames")

    depth_c, mask_c = get_cached(a.input, n, a.ema, a.refresh_cache)

    out_base = a.out or os.path.join(HERE, f"intro_{a.effect}_{a.bg}_{a.layout}")
    frames_dir = out_base + "_frames"
    if a.frame is None:
        shutil.rmtree(frames_dir, ignore_errors=True); os.makedirs(frames_dir)

    palette, rs = None, np.random.RandomState(3)
    k = a.feather | 1
    idx = 0
    written = 0

    for fr in tqdm(read_frames(a.input, n), total=n, desc="render", unit="f"):
        if a.frame is not None and idx != a.frame:
            idx += 1; continue
        H, W = fr.shape[:2]
        with stage("upsample cache"):
            d = cv2.resize(depth_c[idx].astype(np.float32) / 255., (W, H), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(mask_c[idx].astype(np.float32) / 255., (W, H), interpolation=cv2.INTER_LINEAR)
            m = cv2.GaussianBlur(m, (k, k), 0)                 # ~6px feather
        if palette is None:
            with stage("palette (kmeans)"):
                palette = build_palette(fr, m)

        fr_sharp = unsharp(fr, a.sharpen) if a.sharpen > 0 else fr
        if a.effect == "clean":
            layer = fr_sharp.copy()
        if a.effect in ("voxel", "both"):
            with stage("effect voxel"):
                layer = voxel_effect(fr, d, m, palette, a.min_cell, a.max_cell)
        if a.effect in ("contour", "both"):
            with stage("effect contour"):
                c = contour_effect(fr, d, m, idx, a.bands, a.glow)
            layer = c if a.effect == "contour" else np.clip(
                layer.astype(np.float32) * 0.45 + c.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)

        # effects render the whole frame; keep that copy for a stylised filmed background
        full_fx = layer
        if face_mode != "off" and a.effect != "clean":
            with stage("face " + face_mode):
                ff = a.face_frac if a.face_frac == "auto" else float(a.face_frac)
                fm = face_protect(m, ff, rgb=fr, pad=a.face_pad)[..., None]
                if face_mode == "only":
                    # keep the effect on the head, revert everything else to source
                    fm = 1.0 - fm
                layer = np.clip(layer.astype(np.float32) * (1 - fm) + fr_sharp.astype(np.float32) * fm,
                                0, 255).astype(np.uint8)

        with stage("background+composite"):
            # "source" reuses the stylised frame behind the subject; "filmed" uses
            # the untouched frame, so only the person carries the effect
            src_bg = full_fx if a.bg == "source" else (fr if a.bg == "filmed" else None)
            out = compose(layer, m, a.bg, a.layout, idx / info["fps"], idx,
                          a.chromatic, a.grain, rs, src_bg=src_bg)

        if a.frame is not None:
            p = out_base + f"_f{a.frame}.png"
            cv2.imwrite(p, out[..., ::-1])
            print("wrote", p); print_timings(1); return
        with stage("png write"):
            cv2.imwrite(os.path.join(frames_dir, f"{written:06d}.png"), out[..., ::-1])
        written += 1; idx += 1

    with stage("encode"):
        files = encode(frames_dir, info["fps"], out_base, a.input, not a.no_audio,
                       a.av1, a.crf, a.webm_crf)
    shutil.rmtree(frames_dir, ignore_errors=True)

    print_timings(max(written, 1))
    print("\n--- output " + "-" * 50)
    dur = written / info["fps"]
    for f in files:
        mb = os.path.getsize(f) / 1e6
        per15 = mb / max(dur, 1e-6) * 15
        flag = "ok" if per15 <= a.target_mb else f"OVER {a.target_mb:.0f}MB - raise --crf/--webm-crf"
        print(f"  {os.path.relpath(f, HERE):<44s} {mb:6.2f} MB   ({per15:5.2f} MB/15s)  {flag}")

if __name__ == "__main__":
    main()
