// Person mask, live: MediaPipe's selfie segmenter (the browser stand-in for
// stylize.py's rembg u2net_human_seg), plus the silhouette measurements that
// stylize.py's head_bbox / neck_boundary derive from the same mask.
const VERSION = '1.2.0';
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}`;
const MODEL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/' +
              'selfie_segmenter/float16/latest/selfie_segmenter.tflite';
const LONG_SIDE = 256;   // the model runs at 256x256 anyway; feeding it more only costs readback
const FG = 89;           // 0.35 * 255, the threshold stylize.py uses for "person"

export class Segmenter {
  static async create() {
    const { FilesetResolver, ImageSegmenter } = await import(`${CDN}/vision_bundle.mjs`);
    const files = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
    const opts = delegate => ({
      baseOptions: { modelAssetPath: MODEL, delegate },
      runningMode: 'VIDEO', outputConfidenceMasks: true, outputCategoryMask: false,
    });
    let seg;
    try { seg = await ImageSegmenter.createFromOptions(files, opts('GPU')); }
    catch { seg = await ImageSegmenter.createFromOptions(files, opts('CPU')); }
    return new Segmenter(seg);
  }

  constructor(seg) {
    this.seg = seg;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.w = 0; this.h = 0;
    this.mask = null;        // Uint8Array w*h, row 0 = top, camera (unmirrored) space
    this.prob = null;        // temporally smoothed confidence
    this.lastTs = 0;
    this.stats = { present: false };
  }

  update(video, tsMs) {
    const vw = video.videoWidth, vh = video.videoHeight;
    const k = LONG_SIDE / Math.max(vw, vh);
    const w = Math.max(1, Math.round(vw * k)), h = Math.max(1, Math.round(vh * k));
    if (w !== this.w || h !== this.h) {
      this.w = w; this.h = h;
      this.canvas.width = w; this.canvas.height = h;
      this.mask = new Uint8Array(w * h);
      this.prob = null;
    }
    this.ctx.drawImage(video, 0, 0, w, h);
    const ts = Math.max(tsMs, this.lastTs + 1);          // VIDEO mode needs strictly increasing time
    this.lastTs = ts;
    // the callback runs synchronously and its mask is only valid inside it
    this.seg.segmentForVideo(this.canvas, ts, res => {
      const masks = res.confidenceMasks;
      if (!masks?.length) return;
      const f = masks[0].getAsFloat32Array();
      const first = !this.prob;
      if (first) this.prob = new Float32Array(w * h);
      const p = this.prob, m = this.mask;
      for (let i = 0; i < f.length; i++) {
        // light EMA: the raw mask shimmers along hair and shoulders between frames
        p[i] = first ? f[i] : p[i] * 0.35 + f[i] * 0.65;
        m[i] = p[i] * 255;
      }
    });
    this.stats = smoothStats(this.stats, personStats(this.mask, w, h));
    return this;
  }
}

// head_bbox + the silhouette half of neck_boundary, normalised to 0..1 (y down)
export function personStats(m, w, h) {
  const width = new Float32Array(h);
  let top = -1, bot = -1, area = 0;
  for (let y = 0; y < h; y++) {
    let n = 0;
    for (let x = 0, i = y * w; x < w; x++, i++) if (m[i] > FG) n++;
    width[y] = n; area += n;
    if (n) { if (top < 0) top = y; bot = y; }
  }
  if (area < 0.01 * w * h) return { present: false };
  const Hp = bot - top + 1;

  // head: the top 30% of the person's vertical extent
  const band = Math.max(1, Math.floor(0.30 * Hp));
  let x0 = w, x1 = -1;
  for (let y = top; y < Math.min(h, top + band); y++)
    for (let x = 0, i = y * w; x < w; x++, i++) if (m[i] > FG) { if (x < x0) x0 = x; if (x > x1) x1 = x; }
  if (x1 < 0) { x0 = 0; x1 = w - 1; }

  // jaw = narrowest (smoothed) row between 25% and 75% of the person; the collar
  // sits ~12% of the person's height below it. If nothing narrows (a hood, a
  // wide hairstyle) fall back to 62% down, as face_protect does.
  const sm = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let s = 0, c = 0;
    for (let j = Math.max(0, y - 4); j <= Math.min(h - 1, y + 4); j++) { s += width[j]; c++; }
    sm[y] = s / c;
  }
  const lo = top + Math.floor(0.25 * Hp), hi = top + Math.floor(0.75 * Hp);
  let jaw = lo;
  for (let y = lo; y < hi; y++) if (sm[y] < sm[jaw]) jaw = y;
  let above = 0;
  for (let y = top; y <= jaw; y++) above = Math.max(above, sm[y]);
  const cut = (hi - lo < 10 || sm[jaw] > 0.85 * above) ? top + 0.62 * Hp : jaw + 0.12 * Hp;

  return {
    present: true,
    top: top / h, bottom: (bot + 1) / h, faceBottom: (top + band) / h,
    headX: (x0 + x1 + 1) / 2 / w, headY: (top + band * 0.45) / h, headW: (x1 - x0 + 1) / w,
    cut: Math.min(1, cut / h),
  };
}

// measurements jitter frame to frame; ease them so backgrounds and the face cut don't twitch
function smoothStats(prev, cur) {
  if (!cur.present || !prev.present) return cur;
  const out = { present: true };
  for (const k of Object.keys(cur)) if (k !== 'present') out[k] = prev[k] * 0.75 + cur[k] * 0.25;
  return out;
}
