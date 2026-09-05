#!/usr/bin/env python3
"""
serve.py - tiny web UI for the render pipelines.

  ./nunif/venv/bin/python serve.py            # http://localhost:8000

Upload a clip, pick a pipeline, render, watch progress, play the result.
Two pipelines:
  stylize : stylize.py  (voxel / contour effect x plasma|grid|dark bg x layout)
  sbs3d   : convert3d.sh (half-SBS stereo 3D for a headset)

Jobs run as subprocesses; progress is scraped from the tqdm/iw3 stderr stream.
State is in-memory - this is a single-user local tool, not a service.
"""
import asyncio, json, os, re, shutil, signal, subprocess, sys, time, uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
_VENV_PY = os.path.join(HERE, "nunif", "venv", "bin", "python")
PY = os.environ.get("STYLIZE_PYTHON") or (_VENV_PY if os.path.exists(_VENV_PY) else sys.executable)
DATA_DIR = os.environ.get("DATA_DIR", HERE)          # mount a volume here in containers
UPLOADS = os.path.join(DATA_DIR, "uploads")
OUTPUTS = os.path.join(DATA_DIR, "outputs")
JOBS_FILE = os.path.join(DATA_DIR, "jobs.json")
WEB = os.path.join(HERE, "web")
os.environ.setdefault("STYLIZE_CACHE_DIR", os.path.join(DATA_DIR, ".stylize_cache"))  # inherited by stylize.py
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "").strip()  # empty = no auth (local / tunnel use)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


HAS_CUDA = _has_cuda()
GPU_FLAG = "0" if HAS_CUDA else "-1"           # passed to iw3 for the stereo pipeline
for d in (UPLOADS, OUTPUTS):
    os.makedirs(d, exist_ok=True)

MAX_UPLOAD = 512 * 1024 * 1024
ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

# tqdm lines look like "render:  47%|####  | 42/90 [...]"; iw3 uses the same bar
RE_STAGE = re.compile(r"(depth|mask|render)\s*:\s*(\d+)%")
RE_IW3 = re.compile(r":\s*(\d+)%\|")
RE_OUT = re.compile(r"^\s+(\S+\.(?:mp4|webm))\s", re.M)

STAGE_WEIGHTS = {"depth": (0.0, 0.30), "mask": (0.30, 0.60), "render": (0.60, 0.97)}


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"          # queued | running | done | error | cancelled
    stage: str = ""
    percent: float = 0.0
    message: str = ""
    outputs: list = field(default_factory=list)
    log: list = field(default_factory=list)
    started: float = field(default_factory=time.time)
    ended: Optional[float] = None
    proc: Optional[object] = None

    def public(self):
        # build explicitly: asdict() deep-copies every field and blows up on the
        # live subprocess handle before any exclusion filter could run
        return {"id": self.id, "kind": self.kind, "label": self.label,
                "status": self.status, "stage": self.stage,
                "percent": self.percent, "message": self.message,
                "outputs": list(self.outputs), "log": self.log[-14:],
                "elapsed": round((self.ended or time.time()) - self.started, 1)}


JOBS: Dict[str, Job] = {}
app = FastAPI(title="3dvid studio")


def save_jobs():
    """Persist job history so it survives restarts (in-memory alone is lost on redeploy)."""
    try:
        tmp = JOBS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump([j.public() for j in JOBS.values()], f)
        os.replace(tmp, JOBS_FILE)
    except Exception:
        pass


def load_jobs():
    if not os.path.exists(JOBS_FILE):
        return
    try:
        for d in json.load(open(JOBS_FILE)):
            j = Job(id=d["id"], kind=d["kind"], label=d["label"], status=d["status"],
                    stage=d.get("stage", ""), percent=d.get("percent", 0.0),
                    message=d.get("message", ""), outputs=d.get("outputs", []), log=d.get("log", []))
            j.started = time.time() - d.get("elapsed", 0); j.ended = time.time()
            if j.status in ("running", "queued"):          # process died with the old server
                j.status, j.message = "error", "server restarted during render"
            JOBS[j.id] = j
    except Exception:
        pass


load_jobs()


@app.middleware("http")
async def require_token(request: Request, call_next):
    """Bearer-token gate for everything except the page itself and /api/health.
    Media URLs are loaded by <video>/<a> which cannot set headers, so ?token= is accepted too."""
    if AUTH_TOKEN and (request.url.path.startswith("/api/") or request.url.path.startswith("/media/")) \
       and request.url.path != "/api/health":
        auth = request.headers.get("authorization", "")
        supplied = auth[7:].strip() if auth.lower().startswith("bearer ") else request.query_params.get("token", "")
        if supplied != AUTH_TOKEN:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


def _safe_name(name: str) -> str:
    base = os.path.basename(name or "clip.mp4").replace("\x00", "")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:80]
    return base or "clip.mp4"


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(WEB, "index.html")) as f:
        return f.read()


@app.get("/api/videos")
def list_videos():
    out = []
    for f in sorted(os.listdir(UPLOADS)):
        p = os.path.join(UPLOADS, f)
        if os.path.isfile(p):
            out.append({"name": f, "mb": round(os.path.getsize(p) / 1e6, 2)})
    return out


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported type {ext!r}; use {sorted(ALLOWED_EXT)}")
    name = _safe_name(file.filename)
    dest = os.path.join(UPLOADS, name)
    n = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            n += len(chunk)
            if n > MAX_UPLOAD:
                out.close(); os.remove(dest)
                raise HTTPException(413, "file too large (512 MB cap)")
            out.write(chunk)
    # probe so the UI can show duration/size and warn about long CPU renders
    try:
        # one combined -show_entries; a second flag silently replaces the first
        # reuse stylize.probe so rotated phone clips report display dims (720x1280, not 1280x720)
        import importlib.util
        spec = importlib.util.spec_from_file_location("stylize", os.path.join(HERE, "stylize.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        info = mod.probe(dest)
        w, h, dur = info["w"], info["h"], info["dur"]
    except Exception:
        w = h = 0; dur = 0.0
    return {"name": name, "mb": round(n / 1e6, 2), "w": w, "h": h, "duration": round(dur, 2)}


def _build_cmd(kind, video, p):
    src = os.path.join(UPLOADS, video)
    if kind == "stylize":
        _fm = p.get("face_mode", "off")
        stem = f"{os.path.splitext(video)[0]}_{p['effect']}_{p['bg']}_{p['layout']}" \
               + ("" if _fm == "off" else f"_face{_fm}")
        base = os.path.join(OUTPUTS, stem)
        cmd = [PY, os.path.join(HERE, "stylize.py"), src,
               "--effect", p["effect"], "--bg", p["bg"], "--layout", p["layout"],
               "--bands", str(p["bands"]), "--min-cell", str(p["min_cell"]),
               "--max-cell", str(p["max_cell"]), "--crf", str(p["crf"]),
               "--webm-crf", str(p["webm_crf"]), "-o", base]
        if p.get("preview"):
            cmd += ["--preview", str(p["preview"])]
        if not p.get("keep_audio"):
            cmd.append("--no-audio")
        if p.get("face_mode", "off") != "off":
            cmd += ["--face-mode", p["face_mode"],
                    "--face-frac", "auto" if p.get("face_to", "neck") == "neck" else "0.55",
                    "--face-pad", str(p.get("face_pad", 0.05))]
        if p.get("sharpen"):
            cmd += ["--sharpen", str(p["sharpen"])]
        if p.get("chromatic"):
            cmd.append("--chromatic")
        if p.get("grain"):
            cmd += ["--grain", str(p["grain"])]
        return cmd, [base + ".mp4", base + ".webm"]

    # half-SBS stereo via iw3, mirroring convert3d.sh's flags
    out = os.path.join(OUTPUTS, f"{os.path.splitext(video)[0]}_3d.mp4")
    cmd = [PY, "-m", "iw3.cli", "--gpu", GPU_FLAG, "--depth-model", "Any_V2_S",
           "--divergence", str(p.get("divergence", 2.0)), "--convergence", "0.5",
           f"--{p.get('sbs_format', 'half-sbs')}",
           "--max-output-height", str(p.get("max_height", 720)),
           "-i", src, "-o", out, "--yes"]
    if p.get("preview"):
        # iw3 wants hh:mm:ss / mm:ss / integer seconds - a bare "2.0" is rejected
        cmd += ["--start-time", "0", "--end-time", str(int(round(float(p["preview"]))))]
    return cmd, [out]


async def _run(job: Job, cmd, expected, cwd):
    job.status = "running"
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    job.proc = proc
    buf = b""
    try:
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                m = RE_STAGE.search(line)
                if m:
                    stage, pct = m.group(1), float(m.group(2))
                    lo, hi = STAGE_WEIGHTS[stage]
                    job.stage, job.percent = stage, round(lo + (hi - lo) * pct / 100, 3) * 100
                elif job.kind == "sbs3d" and (m2 := RE_IW3.search(line)):
                    job.stage, job.percent = "render", float(m2.group(1)) * 0.97
                if "%|" not in line:
                    job.log.append(line)
                    if "cache hit" in line:
                        job.message = "reusing cached depth + mask"
        rc = await proc.wait()
    except asyncio.CancelledError:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        job.status, job.ended = "cancelled", time.time()
        save_jobs()
        raise

    job.ended = time.time()
    if rc != 0:
        job.status, job.message = "error", f"exit {rc}: " + (job.log[-1] if job.log else "")
        save_jobs()
        return
    found = [os.path.basename(p) for p in expected if os.path.exists(p)]
    if not found:
        job.status, job.message = "error", "process finished but produced no output file"
        save_jobs()
        return
    job.status, job.percent, job.stage = "done", 100.0, ""
    job.outputs = [{"name": n, "mb": round(os.path.getsize(os.path.join(OUTPUTS, n)) / 1e6, 2)}
                   for n in found]
    job.message = "  ".join(f"{o['name']} {o['mb']} MB" for o in job.outputs)
    save_jobs()


@app.post("/api/render")
async def render(
    video: str = Form(...),
    kind: str = Form("stylize"),
    effect: str = Form("voxel"),
    bg: str = Form("plasma"),
    layout: str = Form("wide"),
    bands: int = Form(12),
    min_cell: int = Form(8),
    max_cell: int = Form(24),
    crf: int = Form(32),
    webm_crf: int = Form(44),
    preview: float = Form(0),
    keep_audio: bool = Form(True),
    face_mode: str = Form("off"),
    face_pad: float = Form(0.05),
    sharpen: float = Form(0),
    face_to: str = Form("neck"),
    chromatic: bool = Form(False),
    grain: float = Form(0),
    divergence: float = Form(2.0),
    sbs_format: str = Form("half-sbs"),
    max_height: int = Form(720),
):
    if kind not in ("stylize", "sbs3d"):
        raise HTTPException(400, "kind must be stylize or sbs3d")
    if os.path.basename(video) != video or not os.path.exists(os.path.join(UPLOADS, video)):
        raise HTTPException(404, "unknown video")
    if face_mode not in ("off", "clear", "only"):
        raise HTTPException(400, "bad face_mode")
    face_pad = min(max(face_pad, 0.0), 0.4)
    if effect not in ("voxel", "contour", "both", "clean") \
       or bg not in ("plasma", "grid", "dark", "source", "filmed") \
       or layout not in ("portrait", "wide", "square", "native"):
        raise HTTPException(400, "bad effect/bg/layout")
    if sbs_format not in ("half-sbs", "tb", "half-tb", "anaglyph", "vr180"):
        raise HTTPException(400, "bad sbs format")

    if any(j.status in ("queued", "running") for j in JOBS.values()):
        raise HTTPException(409, "a render is already running - wait or cancel it")

    p = dict(effect=effect, bg=bg, layout=layout, bands=bands, min_cell=min_cell,
             max_cell=max_cell, crf=crf, webm_crf=webm_crf, preview=preview,
             keep_audio=keep_audio, chromatic=chromatic, grain=grain,
             face_mode=face_mode, face_pad=face_pad, sharpen=sharpen, face_to=face_to,
             divergence=divergence, sbs_format=sbs_format, max_height=max_height)
    cmd, expected = _build_cmd(kind, video, p)
    label = (f"{effect} / {bg} / {layout}" if kind == "stylize" else f"3D {sbs_format}")
    job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
    JOBS[job.id] = job
    save_jobs()
    cwd = HERE if kind == "stylize" else os.path.join(HERE, "nunif")
    job._task = asyncio.create_task(_run(job, cmd, expected, cwd))   # noqa
    return {"job": job.id}


@app.get("/api/jobs")
def jobs():
    return [j.public() for j in sorted(JOBS.values(), key=lambda j: -j.started)][:20]


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "no such job")
    t = getattr(j, "_task", None)
    if t and not t.done():
        t.cancel()
    return {"ok": True}


@app.get("/media/{name}")
def media(name: str):
    if os.path.basename(name) != name:
        raise HTTPException(400, "bad name")
    p = os.path.join(OUTPUTS, name)
    if not os.path.exists(p):
        raise HTTPException(404, "not found")
    mime = "video/webm" if name.endswith(".webm") else "video/mp4"
    return FileResponse(p, media_type=mime, filename=name)


@app.get("/api/health")
def health():
    import torch
    return {"cuda": torch.cuda.is_available(), "torch": torch.__version__,
            "auth": bool(AUTH_TOKEN), "cpus": os.cpu_count(),
            "note": ("GPU: ~1 min per 30s stylise render" if HAS_CUDA
                     else "CPU-only: ~8 min per 30s stylise render on 32 cores; scales with cores")}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n  3dvid studio -> http://{host}:{port}   auth={'on' if AUTH_TOKEN else 'OFF'}  data={DATA_DIR}")
    if host == "127.0.0.1":
        print(f"  remote? on your laptop:  ssh -L {port}:localhost:{port} qm-personal")
    elif not AUTH_TOKEN:
        print("  WARNING: bound to a non-loopback address with no AUTH_TOKEN set")
    print()
    uvicorn.run(app, host=host, port=port, log_level="warning")
