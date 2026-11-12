#!/usr/bin/env python3
"""Build the 3dvid camera browser extension (standard library only).

    python3 extension/build.py

1. copies the camera page's shared modules (web/camera/*.js) into extension/lib/, so
   the extension renders with exactly the page's effect code;
2. downloads the pinned MediaPipe files and person model into extension/vendor/ and
   checks them against the sha256 sums below (the Chrome Web Store forbids remote code,
   so they ship inside the extension); already-verified files are not fetched again;
3. renders missing toolbar icons from extension/icons/icon.svg (needs rsvg-convert);
4. zips the loadable extension to dist/3dvid-camera-extension.zip.

extension/ is then loadable unpacked as it stands. lib/, vendor/ and dist/ are build
output (gitignored). To move to a newer MediaPipe, bump VERSION in web/camera/segment.js
and MEDIAPIPE here together, delete extension/vendor/, run once with --print-sums and
paste the printed sums into VENDOR.
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

EXT = Path(__file__).resolve().parent
ROOT = EXT.parent
CAMERA = ROOT / 'web' / 'camera'
LIB = EXT / 'lib'
VENDOR = EXT / 'vendor'
ZIP = ROOT / 'dist' / '3dvid-camera-extension.zip'

# what render.js / palette.js / segment.js need, and nothing of the page's UI
SHARED = ['gl.js', 'render.js', 'shaders.js', 'palette.js', 'segment.js', 'effects.js']

MEDIAPIPE = '1.2.0'
CDN = f'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{MEDIAPIPE}'
# the page loads ".../float16/latest/..."; "1" is the same file (identical md5 on GCS), pinned
MODEL = ('https://storage.googleapis.com/mediapipe-models/image_segmenter/'
         'selfie_segmenter/float16/1/selfie_segmenter.tflite')
APACHE = 'https://www.apache.org/licenses/LICENSE-2.0.txt'
# path under vendor/, source URL, sha256. Only the SIMD wasm build: every browser the
# extension supports (Chrome 116+) has wasm SIMD.
VENDOR_FILES = [
    ('vision_bundle.mjs', f'{CDN}/vision_bundle.mjs',
     '870da5cfc43bfcf2a8b0f14e8152dae8c7e701722e1191b0e592b94b81b4127a'),
    ('wasm/vision_wasm_internal.js', f'{CDN}/wasm/vision_wasm_internal.js',
     '85c1a342be70427987f900969a5f26e17557351111364e6585e97fb03e93e108'),
    ('wasm/vision_wasm_internal.wasm', f'{CDN}/wasm/vision_wasm_internal.wasm',
     'bcc13df22d3d69ec8cbe8e280989356d844ae00f4523eb82072d2484f44134dc'),
    ('selfie_segmenter.tflite', MODEL,
     '191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b'),
    ('LICENSE-2.0.txt', APACHE,
     'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'),
]

NOTICE = f"""Third-party files in this directory
===================================

vision_bundle.mjs, wasm/vision_wasm_internal.js, wasm/vision_wasm_internal.wasm
  MediaPipe Tasks Vision {MEDIAPIPE} (npm package @mediapipe/tasks-vision)
  Copyright The MediaPipe Authors. Licensed under the Apache License, Version 2.0.
  Source: https://github.com/google-ai-edge/mediapipe
  Downloaded unmodified from {CDN}/

selfie_segmenter.tflite
  MediaPipe Selfie Segmentation model (float16, version 1)
  Copyright Google LLC. Licensed under the Apache License, Version 2.0.
  Model card: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf
  Downloaded unmodified from {MODEL}

LICENSE-2.0.txt: the Apache License, Version 2.0 ({APACHE})
"""

ICONS = [(16, True), (32, True), (48, False), (128, False)]   # (px, drop the "fine" detail)
# files of extension/ that are sources for the build, not part of the extension
NOT_SHIPPED = {'build.py', 'README.md', 'icons/icon.svg'}
FIXED_TIME = (2024, 1, 1, 0, 0, 0)   # zip entry dates, so an unchanged build zips identically


def die(msg):
    sys.exit(f'build.py: {msg}')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_shared():
    m = re.search(r"const VERSION = '([^']+)'", (CAMERA / 'segment.js').read_text())
    if not m or m.group(1) != MEDIAPIPE:
        die(f"web/camera/segment.js loads MediaPipe {m and m.group(1)}, this script vendors {MEDIAPIPE}; "
            'update MEDIAPIPE and the sums together (see the docstring)')
    if LIB.exists():
        shutil.rmtree(LIB)
    LIB.mkdir()
    for name in SHARED:
        src = (CAMERA / name).read_text()
        # a new relative import in the page's modules must be copied too, or the extension breaks
        for dep in re.findall(r"""from\s+['"]\./([^'"]+)['"]""", src):
            if dep not in SHARED:
                die(f'web/camera/{name} imports ./{dep}, which is not in SHARED')
        if name == 'segment.js':
            # the page loads MediaPipe from the CDN by default; the shipped copy must not even
            # name it (the Web Store's remote-code review flags any import() of a remote URL)
            cdn = "const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}`;"
            if cdn not in src:
                die(f'web/camera/segment.js no longer has the line\n  {cdn}\nupdate copy_shared()')
            src = src.replace(cdn, "const CDN = new URL('../vendor', import.meta.url).href;")
        (LIB / name).write_text(src)
    remote = [p.name for p in LIB.iterdir() if 'cdn.jsdelivr.net' in p.read_text()]
    if remote:
        die(f'lib/{", lib/".join(remote)} still names cdn.jsdelivr.net (remote code in the package)')
    print(f'lib/: {len(SHARED)} modules from web/camera/')


def fetch_vendor(print_sums=False):
    for rel, url, want in VENDOR_FILES:
        dest = VENDOR / rel
        if dest.exists() and sha256(dest) == want:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f'downloading {url}')
        req = urllib.request.Request(url, headers={'User-Agent': '3dvid-extension-build'})
        tmp = dest.with_suffix(dest.suffix + '.part')
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f)
        got = sha256(tmp)
        if print_sums:
            print(f'  {rel}: {got}')
        elif got != want:
            tmp.unlink()
            die(f'{url}\n  sha256 {got}\n  expected {want}\n  (the file changed upstream; refusing to ship it)')
        tmp.replace(dest)
    (VENDOR / 'NOTICE').write_text(NOTICE)
    print(f'vendor/: MediaPipe {MEDIAPIPE} + selfie segmenter, sha256 verified')


def icons():
    svg = EXT / 'icons' / 'icon.svg'
    for px, small in ICONS:
        png = EXT / 'icons' / f'icon-{px}.png'
        if png.exists() and png.stat().st_mtime >= svg.stat().st_mtime:
            continue
        if not shutil.which('rsvg-convert'):
            if png.exists():
                continue          # older than the SVG, but usable
            die(f'{png.name} is missing and rsvg-convert is not installed (librsvg2-bin)')
        cmd = ['rsvg-convert', '-w', str(px), str(svg), '-o', str(png)]
        with tempfile.NamedTemporaryFile('w', suffix='.css') as css:
            if small:
                css.write('.fine{display:none}')
                css.flush()
                cmd[3:3] = ['-s', css.name]
            subprocess.run(cmd, check=True)
        print(f'icons/{png.name} rendered')


def check_manifest():
    manifest = json.loads((EXT / 'manifest.json').read_text())
    refs = set()
    for cs in manifest.get('content_scripts', []):
        refs.update(cs.get('js', []))
    refs.add(manifest['background']['service_worker'])
    refs.add(manifest['action']['default_popup'])
    refs.update(manifest['icons'].values())
    missing = [r for r in sorted(refs) if not (EXT / r).exists()]
    if missing:
        die(f'manifest.json names missing files: {", ".join(missing)}')
    return manifest['version']


def pack(version):
    files = sorted(p for p in EXT.rglob('*') if p.is_file()
                   and p.relative_to(EXT).as_posix() not in NOT_SHIPPED
                   and not p.name.endswith('.part') and '__pycache__' not in p.parts
                   and not p.name.startswith('.'))
    ZIP.parent.mkdir(exist_ok=True)
    tmp = ZIP.with_suffix('.zip.part')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(EXT).as_posix(), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
    tmp.replace(ZIP)
    print(f'{ZIP.relative_to(ROOT)}: version {version}, {len(files)} files, {ZIP.stat().st_size / 1e6:.1f} MB')


def main():
    print_sums = '--print-sums' in sys.argv[1:]
    copy_shared()
    fetch_vendor(print_sums)
    icons()
    pack(check_manifest())
    print('load unpacked: chrome://extensions -> Developer mode -> Load unpacked -> extension/')


if __name__ == '__main__':
    main()
