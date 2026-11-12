// The styled camera, in the extension's isolated world (imported by content.js the first
// time a page asks for a camera). Per real camera:
//
//   real track -> MediaStreamTrackProcessor -> the camera page's Renderer (layout
//   "native") -> new VideoFrame(canvas) -> MediaStreamTrackGenerator -> one clone per
//   page consumer
//
// Driven by camera frames, not requestAnimationFrame, so it keeps running in a
// background tab, which is where a meeting usually is. Browsers without the two track
// APIs get a <video> + canvas.captureStream() version, which a hidden tab may slow.
// The person mask comes from SegClient: MediaPipe in a hidden extension iframe, the one
// place a meeting page's CSP cannot stop it from loading its wasm.
import { Renderer } from './lib/render.js';
import { buildPalette } from './lib/palette.js';
import { DEFAULTS, loadSettings, sanitize } from './settings.js';

let S = { ...DEFAULTS };
const settingsReady = loadSettings().then(s => { S = s; }, () => {});
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'local') return;
  const next = { ...S };
  for (const [k, { newValue }] of Object.entries(changes)) if (k in DEFAULTS) next[k] = newValue;
  const q = S.quality;
  S = sanitize(next);
  if (S.quality !== q) cameras.forEach(c => c.fitQuality());
});

const renderSettings = () => ({
  effect: S.effect, bg: S.bg, face: S.face, mirror: S.mirror, chroma: S.chroma, grain: S.grain,
  layout: 'native', quality: String(S.quality), subjectX: '',
  bands: 12, minCell: 8, maxCell: 24,        // the camera page's defaults
});
// the <video> fallback cannot hand over the camera's own frames; this is the closest look
const PLAIN = { ...renderSettings(), effect: 'clean', bg: 'filmed', face: 'off', mirror: false, chroma: false, grain: 0 };

const sizeOf = s => [s.displayWidth ?? s.videoWidth ?? s.width, s.displayHeight ?? s.videoHeight ?? s.height];
const even = v => v - (v % 2);

// ---- person mask: segment.js in seg.html, an extension page in a hidden iframe. It has
// the extension's origin and CSP ('wasm-unsafe-eval', scripts from the extension), so
// MediaPipe loads there whatever the meeting page's CSP says. Frames go over a private
// MessageChannel as 256 px ImageBitmaps (transferred); masks come back the same way.
// seg.html is web-accessible only at a per-session dynamic URL (use_dynamic_url), so other
// sites cannot embed it to run segmenters in the extension's process. The document loaded
// there still has the extension's own origin, which is what postMessage must target.
const SEG_URL = chrome.runtime.getURL('seg.html');
const EXT_ORIGIN = `chrome-extension://${chrome.runtime.id}`;
const SEG_LONG_SIDE = 256;

class SegClient {
  constructor() {
    this.busy = 0;                     // when the frame in flight was sent; one at a time
    this.listeners = new Map();        // camera id -> mask callback
    const f = this.el = document.createElement('iframe');
    f.src = SEG_URL;
    f.setAttribute('data-3dvid', '');  // also where the debug state is shown (data-3dvid-*)
    f.setAttribute('aria-hidden', 'true');
    f.tabIndex = -1;
    Object.assign(f.style, {
      position: 'fixed', left: '-10px', top: '-10px', width: '1px', height: '1px',
      border: '0', opacity: '0', pointerEvents: 'none',
    });
    const ch = new MessageChannel();
    this.port = ch.port1;
    this.port.onmessage = e => this.onMessage(e.data);
    f.addEventListener('load', () => {
      // targetOrigin: only the extension's own page may receive the port. A frame-src
      // block leaves an error page there instead, which never says hello.
      f.contentWindow.postMessage({ type: '3dvid-seg' }, EXT_ORIGIN, [ch.port2]);
      setTimeout(() => { if (this.state === 'loading') this.set('blocked'); }, 4000);
    }, { once: true });
    this.set('loading');
    document.documentElement.append(f);
  }

  set(state, detail) {
    this.state = state;
    this.el.setAttribute('data-3dvid-seg', state);
    if (detail) this.el.setAttribute('data-3dvid-seg-error', detail);
  }

  push(id, src, tsMs) {
    if (this.state !== 'active') return;
    if (this.busy && performance.now() - this.busy < 2000) return;   // a lost reply must not stall it
    const [w0, h0] = sizeOf(src), k = SEG_LONG_SIDE / Math.max(w0, h0);
    this.busy = performance.now();
    // the bitmap holds its own reference to the frame, so the caller may close it now
    createImageBitmap(src, {
      resizeWidth: Math.max(1, Math.round(w0 * k)), resizeHeight: Math.max(1, Math.round(h0 * k)),
      resizeQuality: 'low',
    }).then(bmp => this.port.postMessage({ type: 'frame', id, ts: tsMs, bmp }, [bmp]),
            () => { this.busy = 0; });
  }

  onMessage(m) {
    if (m?.type === 'hello') this.set('starting');
    else if (m?.type === 'ready') this.set('active');
    else if (m?.type === 'error') this.set('error', m.message);
    else if (m?.type === 'mask') { this.busy = 0; if (m.mask) this.listeners.get(m.id)?.(m); }
  }
}
let segClient = null;
// no person anywhere: what render() sends until the first mask, so a meeting sees only the
// chosen background, never the unmasked room, while MediaPipe loads
const EMPTY = { mask: new Uint8Array(1), w: 1, h: 1, stats: { present: false } };

// ---- cameras: one real camera per document and device, shared by every consumer
const cameras = new Set();
let nextId = 1;

class Camera {
  constructor(real, keys) {
    this.id = nextId++;
    this.real = real;
    this.keys = keys;
    this.native = real.getSettings();
    this.consumers = new Set();
    this.els = new Set();              // the requests' shared elements: the page side reads the live size there
    this.outW = 0; this.outH = 0;
    this.handedOut = false;
    this.alive = true;
    this.seg = { mask: null, w: 0, h: 0, stats: { present: false } };
    this.palette = null;
    this.paletteHasPerson = false;
    this.frames = 0;
    this.warned = false;
    segClient ??= new SegClient();
    segClient.listeners.set(this.id, m => {
      const was = this.seg.stats?.present;
      Object.assign(this.seg, { mask: m.mask, w: m.w, h: m.h, stats: m.stats });
      if (was !== m.stats?.present) this.debug();
    });
    if (typeof MediaStreamTrackProcessor === 'function' && typeof MediaStreamTrackGenerator === 'function') this.startTrack();
    else this.startElement();
    this.fitQuality();
    cameras.add(this);
    this.debug();
  }

  startTrack() {
    this.mode = 'track';
    this.canvas = new OffscreenCanvas(2, 2);
    this.renderer = new Renderer(this.canvas);
    this.out = new MediaStreamTrackGenerator({ kind: 'video' });
    const reader = new MediaStreamTrackProcessor({ track: this.real }).readable.getReader();
    const writer = this.out.writable.getWriter();
    (async () => {
      for (;;) {
        const { value: frame, done } = await reader.read().catch(() => ({ done: true }));
        if (done) break;
        if (!this.check()) { frame.close(); break; }
        if (++this.frames % 15 === 1) this.debug();
        let out = frame;
        if (S.enabled && this.render(frame, frame.timestamp / 1e6)) {
          out = new VideoFrame(this.canvas, {
            timestamp: frame.timestamp, ...(frame.duration ? { duration: frame.duration } : {}),
          });
          frame.close();
        }
        this.publish(out.displayWidth, out.displayHeight);
        await writer.write(out).catch(() => {});
        out.close();
      }
      // ending the generator ends the page's tracks too, as an unplugged camera would
      writer.close().catch(() => {});
      this.stop();
      // frames already queued would be garbage collected unclosed, which stalls the camera
      // pipeline; the stopped track ends the stream, so this finishes
      for (;;) {
        const { value, done } = await reader.read().catch(() => ({ done: true }));
        if (done) break;
        value.close();
      }
    })();
  }

  startElement() {
    this.mode = 'element';
    const v = this.video = document.createElement('video');
    v.muted = true; v.playsInline = true;
    v.srcObject = new MediaStream([this.real]);
    v.play().catch(() => {});
    this.canvas = document.createElement('canvas');
    this.renderer = new Renderer(this.canvas);
    this.out = this.canvas.captureStream().getVideoTracks()[0];
    const t0 = performance.now();
    const step = () => {
      if (!this.check()) return;
      if (++this.frames % 15 === 1) this.debug();
      const t = (performance.now() - t0) / 1000;
      if (S.enabled) this.render(v, t);
      else this.renderer.frame({ video: v, settings: PLAIN, seg: null, palette: this.palette ?? new Float32Array(72), t });
      this.publish(this.canvas.width, this.canvas.height);
      v.requestVideoFrameCallback(step);
    };
    v.requestVideoFrameCallback(step);
  }

  // one styled frame onto this.canvas; false = send the plain camera frame instead
  render(src, t) {
    try {
      segClient.push(this.id, src, t * 1000);
      // null (the whole frame is the person) only once segmentation has failed for good
      const seg = this.seg.mask ? this.seg
        : ['error', 'blocked'].includes(segClient.state) ? null : EMPTY;
      const person = !!seg?.stats?.present;
      // like the camera page: built from the first frame, rebuilt once a person is found
      if (!this.palette || (!this.paletteHasPerson && person)) {
        this.palette = buildPalette(src, person ? seg : null);
        this.paletteHasPerson = person;
      }
      return this.renderer.frame({ video: src, settings: renderSettings(), seg, palette: this.palette, t });
    } catch (e) {
      if (!this.warned) { this.warned = true; console.warn('[3dvid] render failed, sending the plain camera:', e); }
      return false;
    }
  }

  // state for tests and curious developers, on the hidden iframe's element
  debug() {
    const el = segClient.el;
    el.setAttribute('data-3dvid-cameras', String(cameras.size));
    el.setAttribute('data-3dvid-mode', this.mode);
    el.setAttribute('data-3dvid-frames', String(this.frames));
    el.setAttribute('data-3dvid-person', this.seg.stats?.present ? '1' : '0');
  }

  // scale the real camera down to the quality setting (its short side), or back up
  async fitQuality() {
    const { width: w, height: h, frameRate } = this.native;
    if (!w || !h || !this.alive) return;
    const short = Math.min(w, h), k = Math.min(1, S.quality / short);
    const cur = this.real.getSettings();
    const W = even(Math.round(w * k)), H = even(Math.round(h * k));
    if (Math.abs(cur.width - W) < 4 && Math.abs(cur.height - H) < 4) return;
    await this.real.applyConstraints({
      width: { ideal: W }, height: { ideal: H }, ...(frameRate ? { frameRate: { ideal: frameRate } } : {}),
    }).catch(() => {});
  }

  // the size being sent changed (quality setting, or the camera itself): tell the page side
  publish(w, h, els = this.els) {
    if (w === this.outW && h === this.outH && els === this.els) return;
    this.outW = w; this.outH = h;
    const v = JSON.stringify({ width: w, height: h, aspectRatio: w / h });
    for (const el of els) el.setAttribute('data-3dvid-live', v);
  }

  consumer(el) {
    this.els.add(el);
    if (this.outW) this.publish(this.outW, this.outH, [el]);
    const t = this.out.clone();
    this.consumers.add(t);
    this.handedOut = true;
    return t;
  }

  // the page stopped (or dropped and ended) its tracks: once none is live, release the camera
  check() {
    for (const t of this.consumers) if (t.readyState === 'ended') this.consumers.delete(t);
    if (this.handedOut && !this.consumers.size) this.stop();
    return this.alive;
  }

  info(constraints) {
    const s = this.real.getSettings();
    return {
      label: this.real.label,
      settings: { ...s, width: even(s.width), height: even(s.height) },
      capabilities: this.real.getCapabilities?.() ?? {},
      constraints,
    };
  }

  stop() {
    if (!this.alive) return;
    this.alive = false;
    cameras.delete(this);
    this.els.clear();
    segClient?.listeners.delete(this.id);
    this.real.stop();
    if (this.mode === 'element') { this.out.stop(); this.video.srcObject = null; }
    this.renderer?.gl.getExtension('WEBGL_lose_context')?.loseContext();
    this.debug();
  }
}

function deviceKey(video) {
  let d = video?.deviceId;
  if (d && typeof d === 'object' && !Array.isArray(d)) d = d.exact ?? d.ideal;
  if (Array.isArray(d)) d = d[0];
  return typeof d === 'string' && d ? d : 'default';
}
const live = key => [...cameras].find(c => c.alive && c.keys.has(key));

// req: { video: constraints object, audio: constraints | false } from the page side.
// Sets el.srcObject to [styled video, real audio...]; returns what the page side needs
// to dress the styled track as the real camera. Camera and microphone are asked for in
// one getUserMedia call, so the user sees a single permission prompt, as without us.
export async function open(req, el) {
  await settingsReady;
  const video = req.video && typeof req.video === 'object' ? req.video : {};
  const key = deviceKey(video);
  let cam = live(key), stream = null, fresh = null;
  try {
    if (!cam) {
      // this world's getUserMedia: the page's wrapper lives in the other world
      stream = await navigator.mediaDevices.getUserMedia({ video, audio: req.audio || false });
      const real = stream.getVideoTracks()[0];
      const id = real.getSettings().deviceId;
      cam = live(key) || live(id);       // a parallel request opened it meanwhile
      if (cam) real.stop();
      else fresh = cam = new Camera(real, new Set([key, id]));
    } else if (req.audio) {
      stream = await navigator.mediaDevices.getUserMedia({ audio: req.audio });
    }
    el.srcObject = new MediaStream([cam.consumer(el), ...(stream?.getAudioTracks() ?? [])]);
    return { ok: true, sid: cam.id, info: cam.info(video) };
  } catch (e) {
    stream?.getTracks().forEach(t => t.stop());
    fresh?.stop();
    throw e;
  }
}

// a consumer the page made with track.clone()
export function adopt(sid, stream) {
  const cam = [...cameras].find(c => c.id === sid && c.alive);
  const t = stream?.getVideoTracks()[0];
  if (cam && t) cam.consumers.add(t);
}

export function check() { cameras.forEach(c => c.check()); }
