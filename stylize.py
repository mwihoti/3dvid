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
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time
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
         ":stream_side_data=rotation:format=duration,format_name", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    fmt = d.get("format", {})
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    # a still has no real duration or rate: ffprobe reports a nominal 25/1 for a
    # JPEG and omits format.duration entirely for a PNG, so trust the container
    # name instead - image2 / *_pipe is a single picture, not a clip.
    fname = fmt.get("format_name", "")
    is_image = fname in ("image2", "image2pipe") or fname.endswith("_pipe")
    dur = float(fmt.get("duration") or 0.0)
    n = int(s.get("nb_frames") or (round(dur * fps) if fps else 1))
    if is_image:
        fps, n, dur = 1.0, 1, 0.0
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if not rot and s.get("tags", {}).get("rotate"):
        rot = int(float(s["tags"]["rotate"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    return dict(w=w, h=h, fps=fps, n=n, dur=dur, rotation=rot, is_image=is_image)

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

def read_one_frame(path, idx):
    """One RGB frame by index, via a seek. Decoding every frame up to idx just to
    throw them away is what made a still cost minutes instead of seconds.
    -ss before -i seeks to the preceding keyframe and decodes forward, so it is
    both fast and frame-accurate; on a variable-rate clip it can land a frame off,
    which is irrelevant for a preview still."""
    info = probe(path)
    w, h = info["w"], info["h"]
    nbytes = w * h * 3
    # No -ss for frame 0: there is nothing to seek to, and "-ss 0" on a JPEG (the
    # image2 demuxer) emits zero bytes and no error at all, which would make every
    # photo render fail silently.
    seek = ["-ss", f"{idx / info['fps']:.6f}"] if idx > 0 else []
    cmd = ["ffmpeg", "-v", "error", *seek,
           "-i", path, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=nbytes)
    try:
        buf = p.stdout.read(nbytes)
    finally:
        p.stdout.close(); p.wait()
    if len(buf) < nbytes:
        raise RuntimeError(f"could not read frame {idx} of {path}")
    return np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()

# ---------------------------------------------------------------- cache ----
def cache_key(video):
    st = os.stat(video)
    raw = f"v2|{os.path.abspath(video)}|{st.st_size}|{int(st.st_mtime)}|{CACHE_SCALE}|{DEPTH_MODEL}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def _depth_model():
    import torch
    from iw3.depth_model_factory import create_depth_model

    model = create_depth_model(DEPTH_MODEL)
    gpu = 0 if torch.cuda.is_available() else -1   # CUDA build + GPU host -> GPU, else CPU
    model.load(gpu=gpu)
    print(f"depth model on {'cuda:0' if gpu == 0 else 'cpu'}", flush=True)
    return model

def _depth_batch(model, frames):
    """frames -> raw per-frame depth maps, before normalisation/EMA."""
    import torch
    x = torch.stack([torch.from_numpy(f.copy()).permute(2, 0, 1).float() / 255. for f in frames])
    # iw3's infer() runs on whatever device the batch is on, so a CPU batch against
    # a CUDA model fails ("Input type (torch.FloatTensor) and weight type
    # (torch.cuda.FloatTensor) should be the same"). Only CPU hosts hid this.
    x = x.to(model.device)
    with torch.inference_mode():
        d = model.infer(x)
    return d.squeeze(1).float().cpu().numpy()

def _norm01(dd):
    """Per-frame min/max stretch; 1.0 = nearest."""
    lo, hi = float(dd.min()), float(dd.max())
    return (dd - lo) / max(hi - lo, 1e-6)

def _cache_dims(video):
    info = probe(video)
    return int(info["h"] * CACHE_SCALE), int(info["w"] * CACHE_SCALE)

def _to_cache(a, ch, cw):
    return np.clip(cv2.resize(a, (cw, ch), interpolation=cv2.INTER_AREA) * 255, 0, 255).astype(np.uint8)

def _upsample(d_u8, m_u8, W, H, feather):
    """Cached half-res uint8 depth/mask -> full-res float [0,1], mask feathered."""
    d = cv2.resize(d_u8.astype(np.float32) / 255., (W, H), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(m_u8.astype(np.float32) / 255., (W, H), interpolation=cv2.INTER_LINEAR)
    return d, cv2.GaussianBlur(m, (feather, feather), 0)

def compute_depth(video, n_frames, ema_alpha, batch=8):
    """iw3 Depth-Anything-V2 -> per-frame [0,1] (1 = nearest), EMA-smoothed."""
    model = _depth_model()
    info = probe(video)
    ch, cw = int(info["h"] * CACHE_SCALE), int(info["w"] * CACHE_SCALE)
    out = np.zeros((n_frames, ch, cw), np.uint8)

    buf, idx, ema = [], 0, None
    pbar = tqdm(total=n_frames, desc="depth", unit="f")

    def flush(frames):
        nonlocal idx, ema
        if not frames:
            return
        for dd in _depth_batch(model, frames):
            dn = _norm01(dd)
            ema = dn if ema is None else (ema_alpha * dn + (1 - ema_alpha) * ema)
            out[idx] = _to_cache(ema, ch, cw)
            idx += 1; pbar.update(1)

    for fr in read_frames(video, n_frames):
        buf.append(fr)
        if len(buf) == batch:
            flush(buf); buf = []
    flush(buf); pbar.close()
    return out

def _mask_session(model_name="u2net_human_seg"):
    from rembg import new_session
    return new_session(model_name)

def _mask_one(sess, frame):
    from rembg import remove
    from PIL import Image
    return np.asarray(remove(Image.fromarray(frame), session=sess, only_mask=True))

def compute_masks(video, n_frames, model_name="u2net_human_seg"):
    sess = _mask_session(model_name)
    info = probe(video)
    ch, cw = int(info["h"] * CACHE_SCALE), int(info["w"] * CACHE_SCALE)
    out = np.zeros((n_frames, ch, cw), np.uint8)
    for i, fr in enumerate(tqdm(read_frames(video, n_frames), total=n_frames, desc="mask", unit="f")):
        out[i] = cv2.resize(_mask_one(sess, fr), (cw, ch), interpolation=cv2.INTER_AREA)
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

def get_cached_one(video, frame_idx, frame, refresh=False):
    """depth + mask for ONE frame, for --frame stills.

    `frame` is the already-decoded RGB frame, so the still path decodes once.
    Prefers the full-video cache when it already covers frame_idx - free, and
    identical to what a video render sees. Otherwise infers exactly one frame
    into its own cache file. That separate file is deliberate: writing a 1-frame
    array into dm_<key>.npz would silently discard an existing full-clip depth
    pass, so a quick preview would cost the next full render another 8 minutes.

    A still has no frame history, so the depth EMA is a no-op here and the result
    is marginally less temporally smoothed than the same frame inside a video.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = cache_key(video)
    full = os.path.join(CACHE_DIR, f"dm_{key}.npz")
    if os.path.exists(full) and not refresh:
        z = np.load(full)
        if z["depth"].shape[0] > frame_idx:
            print(f"cache hit  {os.path.relpath(full, HERE)}  (frame {frame_idx})")
            return z["depth"][frame_idx], z["mask"][frame_idx]
    one = os.path.join(CACHE_DIR, f"one_{key}_f{frame_idx}.npz")
    if os.path.exists(one) and not refresh:
        z = np.load(one)
        print(f"cache hit  {os.path.relpath(one, HERE)}")
        return z["depth"], z["mask"]

    ch, cw = _cache_dims(video)
    with stage("depth (model)"):
        depth = _to_cache(_norm01(_depth_batch(_depth_model(), [frame])[0]), ch, cw)
    with stage("mask (model)"):
        mask = cv2.resize(_mask_one(_mask_session(), frame), (cw, ch), interpolation=cv2.INTER_AREA)
    np.savez_compressed(one, depth=depth, mask=mask)
    print(f"cache write {os.path.relpath(one, HERE)}")
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

# ---- studio .. circuit: numpy ports of the GLSL in web/camera/shaders.js (same
# formulas, same hash12 / vnoise in float32), so the live preview and the render
# match: change both together. Smooth ones are evaluated at 1/2 or 1/4 resolution and
# upsampled; anything that does not move is cached per canvas size.
F32 = np.float32
_BGC = {}

def _memo(key, make):
    if key not in _BGC:
        _BGC[key] = make()
    return _BGC[key]

def _px(W, H, ds=1):
    """The shader's P: pixel-centre coordinates (y down) of a W x H canvas, sampled every
    ds px at block centres (where cv2.resize puts them when upsampling)."""
    def make():
        xs = (np.arange(W // ds, dtype=F32) + F32(0.5)) * F32(ds)
        ys = (np.arange(H // ds, dtype=F32) + F32(0.5)) * F32(ds)
        return np.meshgrid(xs, ys)
    return _memo(("px", W, H, ds), make)

def _fract(x):
    return x - np.floor(x)

def _hash12(x, y):
    """shaders.js hash12(vec2(x, y))."""
    a = _fract(np.asarray(x, F32) * F32(0.1031))
    b = _fract(np.asarray(y, F32) * F32(0.1031))
    c = a
    d = a * (b + F32(33.33)) + b * (c + F32(33.33)) + c * (a + F32(33.33))
    return _fract((a + d + b + d) * (c + d))

def _vnoise(x, y):
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash12(ix, iy), _hash12(ix + 1, iy)
    c, d = _hash12(ix, iy + 1), _hash12(ix + 1, iy + 1)
    top = a + (b - a) * ux
    return top + (c + (d - c) * ux - top) * uy

def _ss(e0, e1, x):
    """GLSL smoothstep."""
    s = np.clip((x - e0) / (e1 - e0), 0, 1)
    return s * s * (3 - 2 * s)

def _rgb(*c):
    return np.array(c, F32) / 255

def _fbm_gl(X, Y, W, H, k, t):
    """shaders.js fbm(): not _fbm, which samples a cached random grid instead."""
    acc, amp, norm = 0, 1.0, 0.0
    for o in range(4):
        sx, sy = 9 * t * (o + 1) * 2 * k, 5 * t * (o + 1) * 2 * k
        acc = acc + amp * _vnoise((X + sx) / (2 * W) * (4 << o) + o * 17, (Y + sy) / (2 * H) * (4 << o) + o * 17)
        norm += amp; amp *= 0.5
    return acc / norm

def _out(col, W, H):
    """float RGB 0..1 at any resolution -> uint8 W x H, like the other backgrounds.
    (convertScaleAbs: one rounded, saturated pass; the colours are never negative.)"""
    if col.shape[:2] != (H, W):
        col = cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.convertScaleAbs(col, alpha=255)

def _k(W, H):
    return min(W, H) / 1080.0          # the shader's uK

def _stars(W, H, cell, density, t):
    """shaders.js stars() at full resolution. Where each star sits never changes, so
    only its twinkle (one value per lit cell) is computed per frame."""
    k = _k(W, H)
    def make():
        X, Y = _px(W, H)
        s = F32(cell * k)
        gx, gy = np.floor(X / s), np.floor(Y / s)
        G = np.meshgrid(np.arange(int(W / s) + 2, dtype=F32), np.arange(int(H / s) + 2, dtype=F32))
        h = _hash12(*G)
        cx = _hash12(G[0] + F32(3.1), G[1] + F32(3.1)) * F32(0.6) + F32(0.2)
        cy = _hash12(G[0] + F32(7.7), G[1] + F32(7.7)) * F32(0.6) + F32(0.2)
        ix, iy = gx.astype(np.int32), gy.astype(np.int32)
        d2 = ((X / s - gx - cx[iy, ix]) ** 2 + (Y / s - gy - cy[iy, ix]) ** 2) * F32(cell * cell)
        glow = np.exp(-d2 * F32(0.9))
        sel = np.nonzero((h[iy, ix] <= density) & (glow > 1e-3))
        return (sel, glow[sel], iy[sel], ix[sel], h,
                F32(0.8) + F32(2.4) * _hash12(G[0] + F32(1.9), G[1] + F32(1.9)),
                F32(0.35) + F32(0.65) * _hash12(G[0] + F32(5.3), G[1] + F32(5.3)))
    sel, glow, iy, ix, h, speed, bright = _memo(("stars", W, H, cell, density), make)
    amp = (0.65 + 0.35 * np.sin(t * speed + h * 400.0)) * bright
    out = np.zeros((H, W), F32)
    out[sel] = glow * amp[iy, ix]
    return out

def bg_studio(W, H, t, head):
    k = _k(W, H)
    X, Y = _px(W, H, 2)
    hx, hy, hw = head
    col = _rgb(0x2b, 0x32, 0x3e) + (_rgb(0x0d, 0x10, 0x16) - _rgb(0x2b, 0x32, 0x3e)) * _ss(0, 1, Y / H)[..., None]
    cx, cy = hx + math.sin(t * 0.11) * 0.05 * W, hy + 0.06 * H
    r = max(hw * 3.0, 380 * k)
    dx, dy = (X - cx) / (r * 1.3), (Y - cy) / r
    col += _rgb(0x5c, 0x6c, 0x84) * (np.exp(-(dx * dx + dy * dy) * 1.3) * 0.75)[..., None]
    qx, qy = (X - W / 2) / (W / 2), (Y - H / 2) / (H / 2)
    col *= (1 - 0.14 * (qx * qx + qy * qy))[..., None]
    col = cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR)
    dither = _memo(("dither", W, H), lambda: ((_hash12(*_px(W, H)) - 0.5) / 255)[..., None])
    return _out(col + dither, W, H)

_BOKEH_TINTS = [_rgb(255, 176, 80), _rgb(255, 120, 60), _rgb(255, 96, 128), _rgb(255, 226, 170)]

def bg_bokeh(W, H, t, head):
    k = _k(W, H)
    Y = (np.arange(H, dtype=F32) + 0.5)[:, None, None]
    col = np.broadcast_to(_rgb(0x14, 0x0c, 0x16) + (_rgb(0x24, 0x14, 0x0e) - _rgb(0x14, 0x0c, 0x16)) * (Y / H),
                          (H, W, 3)).copy()
    for i in range(36):
        z = float(_hash12(i, 2.3))
        rad = (28 + 80 * z) * k
        span = W + 2 * rad
        cx = (float(_hash12(i, 8.1)) * span + t * (6 + 14 * z) * k) % span - rad
        cy = (0.12 + 0.8 * float(_hash12(i, 4.6))) * H + math.sin(t * 0.3 + i) * 12 * k
        reach = rad + 3 * k
        x0, x1 = max(0, int(cx - reach)), min(W, int(cx + reach) + 2)
        y0, y1 = max(0, int(cy - reach)), min(H, int(cy + reach) + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1, dtype=F32) + 0.5
        ys = np.arange(y0, y1, dtype=F32)[:, None] + 0.5
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        disc = 1 - _ss(rad - 3 * k, rad + 2 * k, d)
        rim = _ss(rad * 0.55, rad, d) * disc
        tw = 0.75 + 0.25 * math.sin(t * (0.5 + float(_hash12(i, 6.2))) + i * 1.7)
        col[y0:y1, x0:x1] += _BOKEH_TINTS[i % 4] * ((disc * 0.16 + rim * 0.10) * (0.5 + 0.6 * z) * tw)[..., None]
    return _out(col, W, H)

def bg_aurora(W, H, t, head):
    X, Y = _px(W, H, 2)
    y, x = Y / H, X[:1] / H                         # the curtains' shape depends on x only
    col = _rgb(0x02, 0x04, 0x10) + (_rgb(0x06, 0x1a, 0x2a) - _rgb(0x02, 0x04, 0x10)) * _ss(0.1, 1, y)[..., None]
    for i in range(3):
        edge = (0.42 + 0.12 * i + 0.07 * np.sin(x * (1.3 + 0.4 * i) + t * (0.12 + 0.05 * i) + i * 2.0)
                + 0.10 * (_vnoise(x * 2.2 + i * 9.0, np.full_like(x, t * 0.15)) - 0.5))
        above = edge - y
        glow = np.exp(-np.maximum(above, 0) / (0.16 + 0.05 * i)) * np.exp(-np.minimum(above, 0) ** 2 / 0.0006)
        rays = 0.45 + 0.55 * _vnoise(x * 38.0 + i * 31.0, np.full_like(x, t * 0.35))
        tint = np.array([0.20, 1.0, 0.55], F32) + np.array([0.42, -0.70, 0.45], F32) * np.clip(above * 3, 0, 1)[..., None]
        col += tint * (glow * rays * (0.42 - 0.1 * i))[..., None]
    col = cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR)
    fade = (1 - 0.7 * (np.arange(H, dtype=F32) + 0.5) / H)[:, None]
    col += np.array([0.85, 0.9, 1.0], F32) * (_stars(W, H, 9.0, 0.05, t) * fade)[..., None]
    return _out(col, W, H)

def bg_nebula(W, H, t, head):
    k = _k(W, H)
    X, Y = _px(W, H, 4)
    a = _fbm_gl(X, Y, W, H, k, t * 0.3)
    b = _fbm_gl(X + (a - 0.5) * 900 * k + 5000 * k, Y + (0.5 - a) * 900 * k + 5000 * k, W, H, k, t * 0.2)
    col = np.broadcast_to(_rgb(0x03, 0x03, 0x0c), a.shape + (3,)).copy()
    col += np.array([0.30, 0.10, 0.62], F32) * (_ss(0.38, 0.72, b) * 0.9)[..., None]
    col += np.array([1.00, 0.24, 0.62], F32) * (_ss(0.52, 0.80, b) * _ss(0.45, 0.70, a) * 0.8)[..., None]
    col += np.array([0.16, 0.70, 1.00], F32) * (_ss(0.55, 0.85, a) * (1 - _ss(0.40, 0.70, b)) * 0.45)[..., None]
    veil = 1 - 0.7 * _ss(0.5, 0.8, b)
    col = cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR)
    veil = cv2.resize(veil, (W, H), interpolation=cv2.INTER_LINEAR)
    col += (_stars(W, H, 8.0, 0.06, t) * veil)[..., None]
    return _out(col, W, H)

def bg_warp(W, H, t, head):
    k = _k(W, H)
    def make():
        X, Y = _px(W, H)
        dx, dy = X - W / 2, Y - H / 2
        rpx = np.sqrt(dx * dx + dy * dy)
        r = rpx / F32(0.5 * np.hypot(W, H))
        ang = np.arctan2(dy, dx) / F32(2 * np.pi) + F32(0.5)
        base = _rgb(0x0c, 0x08, 0x24) + (_rgb(0x02, 0x02, 0x08) - _rgb(0x0c, 0x08, 0x24)) * _ss(0, 0.9, r)[..., None]
        base += np.array([0.25, 0.18, 0.75], F32) * (np.exp(-r * r * 9) * 0.5)[..., None]
        layers = []
        for L in range(3):
            N = 70.0 + 55.0 * L
            s = ang * F32(N) + F32(L * 0.37)
            lane = np.mod(np.floor(s), N).astype(np.int32)
            dpx = np.abs(_fract(s) - 0.5) * F32(2 * np.pi / N) * rpx
            ids = np.arange(int(N), dtype=F32)
            layers.append((N, lane, dpx, _hash12(ids, L + 1.0), _hash12(ids, L + 11.0)))
        return r, base, layers
    r, base, layers = _memo(("warp", W, H), make)
    col = base.copy()
    for L, (N, lane, dpx, h, h2) in enumerate(layers):
        z = _fract(h2 + t * (0.18 + 0.22 * h))
        z = np.where(h < 0.35, -1.0, z)            # not every lane has a star
        head_, ln = 0.04 + 1.15 * z * z, 0.02 + 0.45 * z * z * z
        zp, hp, lp = z[lane], head_[lane], ln[lane]
        sel = np.nonzero((zp >= 0) & (r > hp - lp) & (r < hp + 0.004))
        zs, rs_ = zp[sel], r[sel]
        along = _ss(hp[sel] - lp[sel], hp[sel], rs_) * (1 - _ss(hp[sel], hp[sel] + 0.004, rs_))
        half = (0.25 + 1.1 * zs) * k
        across = 1 - _ss(half, half + k, dpx[sel])
        tint = np.array([0.45, 0.55, 1.0], F32) + np.array([0.47, 0.40, 0.0], F32) * zs[:, None]
        fade = 1 - _ss(0.85, 1.0, zs)                # gone before the lane wraps to z = 0
        col[sel] += tint * (along * across * (0.25 + 1.1 * zs) * fade)[:, None]
    return _out(col, W, H)

def _sunset_sky(W, H, k, t, horizon, hz):
    """shaders.js sunsetSky() for the rows above the horizon. The gradient is per row,
    the glow is smooth (quarter resolution), and the sun's banded disc and the
    mountains are drawn only where they can be."""
    yr = (np.arange(hz, dtype=F32) + 0.5) / horizon
    c0, c1, c2 = _rgb(0x14, 0x06, 0x2e), _rgb(0x6a, 0x12, 0x5a), _rgb(0xff, 0x6a, 0x3c)
    grad = c0 + (c1 - c0) * _ss(0, 0.75, yr)[:, None]
    grad = grad + (c2 - grad) * _ss(0.65, 1.0, yr)[:, None]
    col = np.broadcast_to(grad[:, None, :], (hz, W, 3)).copy()
    Sm = min(W, H)                                  # sized by the short side: portrait too
    R = 0.27 * Sm
    scx, scy = W * 0.5, horizon - R * 0.42
    Xg, Yg = _px(W, H, 4)
    glow = np.exp(-(np.maximum(np.sqrt((Xg - scx) ** 2 + (Yg - scy) ** 2) - R, 0) / (0.22 * Sm)) ** 2) * 0.35
    col += np.array([1.0, 0.35, 0.45], F32) * cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)[:hz, :, None]
    X, Y = _px(W, H)
    y0, y1 = max(0, int(scy - R - 3)), min(hz, int(scy + R + 3))
    x0, x1 = max(0, int(scx - R - 3)), min(W, int(scx + R + 3))
    Xb, Yb = X[y0:y1, x0:x1], Y[y0:y1, x0:x1]
    d = np.sqrt((Xb - scx) ** 2 + (Yb - scy) ** 2)
    v = (Yb - scy + R) / (2 * R)
    sun = np.array([1.0, 0.9, 0.36], F32) + np.array([0.0, -0.65, 0.19], F32) * np.clip(v, 0, 1)[..., None]
    cut = (v > 0.2) & (np.clip((v - 0.2) * 0.8, 0, 0.6) >= _fract(v * 9.0 - t * 0.35))
    m = (1 - _ss(R - 1.5 * k, R + 1.5 * k, d)) * ~cut
    sub = col[y0:y1, x0:x1]
    sub += (sun - sub) * m[..., None]
    # two ridges: functions of x only; below the highest one everything is computed
    xs = X[0]
    top = max(0, int(horizon - Sm * 0.2))
    Yc = Y[top:hz, :1]
    for i in range(2):
        x = xs / H * (2.0 + 1.5 * i) + t * (0.02 + 0.03 * i) + i * 7.0
        fi = np.full_like(x, i)
        ridge = horizon - Sm * ((0.13 - 0.05 * i) * _vnoise(x, fi) + 0.05 * _vnoise(x * 4.0, fi + 3.0) + 0.01)
        rock = (_rgb(0x2a, 0x0a, 0x3c) if i == 0 else _rgb(0x12, 0x04, 0x22)) \
            + np.array([1.0, 0.3, 0.6], F32) * (np.exp(-np.maximum(Yc - ridge, 0) / (3 * k)) * 0.35)[..., None]
        band = col[top:hz]
        band += (rock - band) * _ss(ridge - k, ridge + k, Yc)[..., None]
    return col

def bg_sunset(W, H, t, head):
    """Above the horizon the sky; below, the same sky mirrored, rippled sideways and
    darkened (a remap of the sky rows, where the shader evaluates it again)."""
    k = _k(W, H)
    horizon = float(np.floor(H * 0.64))
    hz = int(horizon)
    sky = _sunset_sky(W, H, k, t, horizon, hz)
    X, Y = _px(W, H)
    Xw, Yw = X[hz:], Y[hz:]
    dy = Yw - horizon
    wob = np.sin(dy / k * 0.35 - t * 2.2 + np.sin(Xw / k * 0.013) * 2.0) * (2.0 + dy * 0.05)
    water = cv2.remap(sky, Xw + wob - 0.5, horizon - dy - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    shade = (1 - 0.5 * dy / (H - horizon)) * (0.88 + 0.12 * np.sin(dy / k * 0.9 - t * 3.0))
    water *= np.array([0.55, 0.5, 0.75], F32) * shade[..., None]
    return _out(np.concatenate([sky, water]), W, H)

def _rain_layer(W, H, k, t, cw, speed, seed):
    """shaders.js rainLayer(), evaluated once per glyph dot (7 x 9 per cell): returns the
    dots' (brightness, leading glyph) and, per pixel, the index of its dot."""
    cx, cy = cw * k, cw * 1.3 * k
    ncx, ncy = int(W / cx) + 1, int(H / cy) + 1
    gx = np.arange(ncx, dtype=F32)
    hc = _hash12(gx, np.full_like(gx, seed))
    ln = 8 + 22 * _hash12(gx, np.full_like(gx, seed + 1.0))
    cycle = np.floor(H / F32(cy)) + ln
    head_row = np.floor(np.mod(hc * cycle + t * speed * (0.6 + 0.8 * hc), cycle))
    G = np.meshgrid(gx, np.arange(ncy, dtype=F32))
    behind = head_row[None, :] - G[1]                       # per cell
    lit = (behind >= 0) & (behind <= ln[None, :])
    hg = _hash12(G[0] + F32(seed * 7.0), G[1] + F32(seed * 7.0))
    glyph = np.floor(t * (0.5 + 3.0 * hg) + hg * 10.0)
    level = np.where(lit, np.clip(1 - behind / (ln[None, :] + 1), 0, 1) ** 1.5, 0).astype(F32)
    lead = (lit & (behind <= 0.5)).astype(F32)
    # every dot as (cell row, dot row, cell column, dot column)
    shape = (ncy, 9, ncx, 7)
    # glyphBar(): 12 bars per glyph, then each dot is lit if a bar through it is
    bars = [_hash12(b + glyph * 13, G[0] * 7 + G[1] * 131) >= 0.5 for b in range(12)]
    bars[9] = bars[9] & ~(bars[2] & bars[3] & bars[8])     # never the whole centre cross
    on = np.zeros(shape, bool)
    for sy in range(7):
        for sx in range(5):
            hit = []
            if sy % 3 == 0:
                hit += [sy // 3 * 2] * (sx <= 2) + [sy // 3 * 2 + 1] * (sx >= 2)
            if sx % 2 == 0:
                hit += [6 + sx] * (sy <= 3) + [7 + sx] * (sy >= 3)
            if hit:
                on[:, sy + 1, :, sx + 1] = np.logical_or.reduce([bars[b] for b in hit])
    bright = (on * level[:, None, :, None]).reshape(-1)
    leadd = (on * lead[:, None, :, None]).reshape(-1)
    def make():
        X, Y = _px(W, H)
        ix = np.minimum(np.floor(X[0] / F32(cx) * 7).astype(np.int32), ncx * 7 - 1)
        iy = np.minimum(np.floor(Y[:, 0] / F32(cy) * 9).astype(np.int32), ncy * 9 - 1)
        return iy[:, None] * (ncx * 7) + ix[None, :]
    return bright, leadd, _memo(("rain", W, H, cw), make)

def bg_rain(W, H, t, head):
    """Coloured per dot, then each pixel picks up its dot's colour (np.take)."""
    k = _k(W, H)
    fb, _, fidx = _rain_layer(W, H, k, t, 11.0, 7.0, 3.0)
    nb, nl, nidx = _rain_layer(W, H, k, t, 20.0, 9.0, 1.0)
    far = np.array([0.05, 0.45, 0.2], F32) * (fb * 0.45)[:, None]
    near = (np.array([0.12, 0.95, 0.45], F32) + np.array([0.73, 0.05, 0.45], F32) * nl[:, None]) * nb[:, None]
    far[:, 1] += 0.03; far[:, 2] += 0.012                   # the dark green ground, once
    return _out(np.take(far, fidx, axis=0) + np.take(near, nidx, axis=0), W, H)

def bg_ocean(W, H, t, head):
    k = _k(W, H)
    X, Y = _px(W, H, 2)
    y = Y / H
    col = _rgb(0x0b, 0x5a, 0x8c) + (_rgb(0x02, 0x12, 0x2a) - _rgb(0x0b, 0x5a, 0x8c)) * _ss(0, 1, y)[..., None]
    # the network's wobble and the light shafts are smooth: 1/8 resolution, upsampled
    X8, Y8 = _px(W, H, 8)
    p8x, p8y = X8 / F32(150 * k), Y8 / F32(150 * k)
    wob = np.dstack([_vnoise(p8x * 0.5 + t * 0.2, p8y * 0.5 + t * 0.2),
                     _vnoise(p8x * 0.5 + 7.0 - t * 0.2, p8y * 0.5 + 7.0 - t * 0.2),
                     _vnoise(X8 / H * 6.0 + Y8 / H * 1.2, np.full_like(X8, t * 0.15)),
                     _vnoise(p8x * 0.35 + 40.0 + t * 0.1, p8y * 0.35 + 40.0 + t * 0.1)])
    wob = cv2.resize(wob, X.shape[::-1], interpolation=cv2.INTER_LINEAR)
    px = X / F32(150 * k) + (wob[..., 0] - 0.5) * 0.6
    py = Y / F32(150 * k) + (wob[..., 1] - 0.5) * 0.6
    # voronoiEdge: the wandering points, one per cell, computed per cell and gathered
    gx, gy = np.floor(px), np.floor(py)
    fx, fy = px - gx, py - gy
    o = 3
    G = np.meshgrid(np.arange(-o, int(W / (150 * k)) + o + 1, dtype=F32),
                    np.arange(-o, int(H / (150 * k)) + o + 1, dtype=F32))
    hx, hy = _hash12(*G), _hash12(G[0] + F32(19.7), G[1] + F32(19.7))
    offx = 0.5 + 0.4 * np.sin(t * (0.6 + 0.6 * hx) + hx * 6.2831)
    offy = 0.5 + 0.4 * np.sin(t * (0.6 + 0.6 * hy) + hy * 6.2831)
    ncol = G[0].shape[1]
    base = (gy.astype(np.int32) + o) * ncol + gx.astype(np.int32) + o
    offx, offy = offx.reshape(-1), offy.reshape(-1)
    d1 = np.full(px.shape, 8.0, F32); d2 = d1.copy()
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            n = base + (j * ncol + i)
            ex = i + np.take(offx, n) - fx
            ey = j + np.take(offy, n) - fy
            d = np.sqrt(ex * ex + ey * ey)
            d2 = np.where(d < d1, d1, np.minimum(d2, d))
            d1 = np.minimum(d1, d)
    caus = (1 - _ss(0, 0.16, d2 - d1)) ** 3 * (0.35 + 1.1 * wob[..., 3])
    col += np.array([0.55, 0.95, 1.0], F32) * (caus * (0.6 - 0.35 * y))[..., None]
    shaft = wob[..., 2]
    col += np.array([0.35, 0.75, 0.9], F32) * (shaft ** 3 * (1 - y) * 0.35)[..., None]
    col = cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR)
    for i in range(14):                                          # bubbles: thin rings, full res
        rad = (3 + 7 * float(_hash12(i, 2.0))) * k
        span = H + 4 * rad
        cx = float(_hash12(i, 4.0)) * W + math.sin(t * 1.3 + i) * 6 * k
        cy = H + 2 * rad - (float(_hash12(i, 5.0)) * span + t * (40 + 60 * float(_hash12(i, 3.0))) * k) % span
        x0, x1 = max(0, int(cx - rad - 3 * k)), min(W, int(cx + rad + 3 * k) + 2)
        y0, y1 = max(0, int(cy - rad - 3 * k)), min(H, int(cy + rad + 3 * k) + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1, dtype=F32) + 0.5
        ys = np.arange(y0, y1, dtype=F32)[:, None] + 0.5
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        ring = _ss(rad - 1.5 * k, rad, d) - _ss(rad, rad + k, d)
        col[y0:y1, x0:x1] += np.array([0.7, 0.95, 1.0], F32) * (ring * 0.5)[..., None]
    return _out(col, W, H)

def bg_lava(W, H, t, head):
    X, Y = _px(W, H, 2)
    y = Y / H
    field = np.zeros_like(X)
    for i in range(8):
        hx, hy = float(_hash12(i, 1.0)), float(_hash12(i, 2.0))
        cx = (0.08 + 0.84 * hx) * W + math.sin(t * 0.21 + i * 2.1) * 0.06 * W
        cy = (0.5 + 0.55 * math.sin(t * (0.07 + 0.06 * hy) + i * 1.7)) * H
        r = (0.07 + 0.07 * hy) * min(W, H)
        dx, dy = (X - cx) / r, (Y - cy) / r
        field += 1 / (dx * dx + dy * dy + 1e-3)
    col = _rgb(0x2a, 0x06, 0x3a) + (_rgb(0x0a, 0x02, 0x18) - _rgb(0x2a, 0x06, 0x3a)) * y[..., None]
    wax = np.array([1.0, 0.75, 0.2], F32) + np.array([0.0, -0.55, 0.25], F32) * y[..., None]
    col += wax * (_ss(0.35, 1.0, field) * 0.10)[..., None]
    blob = wax * (0.72 + 0.28 * np.clip((field - 1) / 2, 0, 1))[..., None]
    col += (blob - col) * _ss(0.97, 1.03, field)[..., None]
    return _out(col, W, H)

def bg_circuit(W, H, t, head):
    """The board never changes, so it is drawn once per canvas size; per frame only
    the pulses are evaluated, and only on the pixels near a trace."""
    k = _k(W, H)
    def make():
        X, Y = _px(W, H)
        cell = F32(44 * k)
        gx, gy = np.floor(X / cell), np.floor(Y / cell)
        fx, fy = X / cell - gx - F32(0.5), Y / cell - gy - F32(0.5)
        G = np.meshgrid(np.arange(-1, int(W / cell) + 2, dtype=F32), np.arange(-1, int(H / cell) + 2, dtype=F32))
        hh = _hash12(*G) >= 0.55                                      # edge to the right
        hv = _hash12(G[0] + F32(71.3), G[1] + F32(71.3)) >= 0.60      # edge downwards
        ix, iy = gx.astype(np.int32) + 1, gy.astype(np.int32) + 1
        e = [hh[iy, ix], hh[iy, ix - 1], hv[iy, ix], hv[iy - 1, ix]]
        a = [np.clip(fx, 0, 0.5), np.clip(-fx, 0, 0.5), np.clip(fy, 0, 0.5), np.clip(-fy, 0, 0.5)]
        dk = [np.hypot(fx - a[0], fy), np.hypot(fx + a[1], fy), np.hypot(fx, fy - a[2]), np.hypot(fx, fy + a[3])]
        dk = [np.where(ek, d * cell, F32(1e9)) for ek, d in zip(e, dk)]
        dist = np.minimum.reduce(dk)
        dh, dv = np.minimum(dk[0], dk[1]), np.minimum(dk[2], dk[3])
        w = 2.2 * k
        col = np.broadcast_to(_rgb(0x04, 0x16, 0x12), (H, W, 3)).copy()
        col += np.array([0.20, 0.55, 0.42], F32) * (0.55 * (1 - _ss(w - 0.75 * k, w + 0.75 * k, dist)))[..., None]
        deg = sum(ek.astype(np.int32) for ek in e)
        ring = np.abs(np.hypot(fx, fy) * cell - 5.5 * k)
        pad = np.where(deg == 1, 1 - _ss(1.6 * k, 2.6 * k, ring), 0)[..., None]
        col += (np.array([0.85, 0.65, 0.25], F32) * 0.8 - col) * pad
        # the pixels a pulse can light, with what a pulse there depends on
        near = []
        for dd, along, lane in ((dh, X, gy), (dv, Y, gx + 500)):
            sel = np.nonzero(dd < 4 * k)
            hl = _hash12(lane[sel], np.full(len(sel[0]), 9.0, F32))
            coord = along[sel] / F32(k) * np.where(hl > 0.5, 1, -1).astype(F32)
            near.append((sel, coord, hl, 1 - _ss(2 * k, 4 * k, dd[sel])))
        return col, near
    base, near = _memo(("circuit", W, H), make)
    pulse = np.zeros((H, W), F32)
    for sel, coord, hl, cover in near:
        ph = _fract(coord / 520.0 + t * (0.25 + 0.3 * hl) + hl * 7.0)
        pulse[sel] = np.maximum(pulse[sel], np.exp(-ph * 14.0) * cover)
    return _out(base + np.array([0.5, 1.0, 0.85], F32) * pulse[..., None], W, H)

BACKGROUNDS = {"plasma": bg_plasma, "grid": bg_grid, "dark": bg_dark,
               "studio": bg_studio, "bokeh": bg_bokeh, "aurora": bg_aurora, "nebula": bg_nebula,
               "warp": bg_warp, "sunset": bg_sunset, "rain": bg_rain, "ocean": bg_ocean,
               "lava": bg_lava, "circuit": bg_circuit}

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

def compose(layer, alpha, bgname, layout, t, frame_idx, cab, grn, rs, src_bg=None, subject_x=None):
    sh, sw = layer.shape[:2]
    CW, CH, ax = layout_dims(layout, sw, sh)
    if subject_x is not None:
        ax = subject_x                     # override the layout's own subject position
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
def render_one(fr, d, m, palette, a, face_mode, t, idx, rs):
    """Per-frame pipeline: effect -> face handling -> background composite.
    Shared by the video loop and the --frame still path."""
    fr_sharp = unsharp(fr, a.sharpen) if a.sharpen > 0 else fr
    layer = None
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
        return compose(layer, m, a.bg, a.layout, t, idx,
                       a.chromatic, a.grain, rs, src_bg=src_bg, subject_x=a.subject_x)

def main():
    ap = argparse.ArgumentParser(description="Stylise a talking-head clip into a portfolio intro.")
    ap.add_argument("input", help="video clip, or a still image (jpg/png/webp) "
                                  "for a one-frame render")
    ap.add_argument("--effect", choices=["voxel", "contour", "both", "clean"], default="voxel",
                    help="clean = person kept as original pixels, only the background changes")
    ap.add_argument("--bg", choices=[*BACKGROUNDS, "source", "filmed"], default="plasma",
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
    ap.add_argument("--subject-x", type=float, default=None,
                    help="horizontal centre of the subject as a fraction of the canvas width; "
                         "0.5 = centred. Default: 0.72 for wide (room for a title on the left), "
                         "0.5 for the other layouts")
    ap.add_argument("--bands", type=int, default=12)
    ap.add_argument("--min-cell", type=int, default=8)
    ap.add_argument("--max-cell", type=int, default=24)
    ap.add_argument("--preview", type=float, default=0, help="render only first N seconds")
    ap.add_argument("--frame", type=int, default=None,
                    help="frame INDEX to dump as a single PNG, then exit. Runs one depth "
                         "and one mask inference, so it returns in seconds")
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
    if a.subject_x is not None and not 0.0 <= a.subject_x <= 1.0:
        ap.error("--subject-x must be between 0 and 1 (0.5 = centred)")

    face_mode = a.face_mode or ("clear" if a.face_clear else "off")
    info = probe(a.input)
    if info["is_image"]:
        # a photo is just a one-frame render: same effects, same backgrounds, no encode
        if a.preview:
            print("still image input - ignoring --preview")
        if a.frame is None:
            a.frame = 0
    n = info["n"]
    if a.frame is not None:
        n = 1
    elif a.preview:
        n = min(n, int(round(a.preview * info["fps"])))
    scope = f"frame {a.frame} only" if a.frame is not None else f"using {n} frames"
    print(f"input {a.input}  {info['w']}x{info['h']}  {info['fps']:.3f} fps  {scope}")

    out_base = a.out or os.path.join(HERE, f"intro_{a.effect}_{a.bg}_{a.layout}")
    rs = np.random.RandomState(3)
    k = a.feather | 1

    # ---- still: one depth + one mask inference, not a.frame + 1 of each ----
    if a.frame is not None:
        fi = max(0, min(a.frame, info["n"] - 1))
        if fi != a.frame:
            print(f"clip has {info['n']} frames; using frame {fi}")
        with stage("read frame"):
            fr = read_one_frame(a.input, fi)
        d_u8, m_u8 = get_cached_one(a.input, fi, fr, a.refresh_cache)
        H, W = fr.shape[:2]
        with stage("upsample cache"):
            d, m = _upsample(d_u8, m_u8, W, H, k)
        with stage("palette (kmeans)"):
            palette = build_palette(fr, m)
        out = render_one(fr, d, m, palette, a, face_mode, fi / info["fps"], fi, rs)
        out_png = out_base + f"_f{fi}.png"
        cv2.imwrite(out_png, out[..., ::-1])
        print("wrote", out_png)
        print_timings(1)
        return

    depth_c, mask_c = get_cached(a.input, n, a.ema, a.refresh_cache)

    frames_dir = out_base + "_frames"
    shutil.rmtree(frames_dir, ignore_errors=True); os.makedirs(frames_dir)

    palette = None
    written = 0

    for idx, fr in enumerate(tqdm(read_frames(a.input, n), total=n, desc="render", unit="f")):
        H, W = fr.shape[:2]
        with stage("upsample cache"):
            d, m = _upsample(depth_c[idx], mask_c[idx], W, H, k)   # ~6px feather
        if palette is None:
            with stage("palette (kmeans)"):
                palette = build_palette(fr, m)

        out = render_one(fr, d, m, palette, a, face_mode, idx / info["fps"], idx, rs)
        with stage("png write"):
            cv2.imwrite(os.path.join(frames_dir, f"{written:06d}.png"), out[..., ::-1])
        written += 1

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
