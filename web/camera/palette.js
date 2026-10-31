// build_palette for the browser: k-means to 24 colours on a downscaled frame,
// sampled 70% from the person. Without that bias a bright background takes most
// of the clusters and skin goes flat.
const K = 24, SAMPLES = 4000, FG_SHARE = 0.7, ITERS = 12;

const work = document.createElement('canvas');
const ctx = work.getContext('2d', { willReadFrequently: true });

// small seeded PRNG, so "re-sample colours" on the same frame gives the same palette
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// video: the camera <video>; seg: a Segmenter with a person in frame, or null
// returns Float32Array(K * 3), colours in 0..1
export function buildPalette(video, seg) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const W = 160, H = Math.max(1, Math.round(W * vh / vw));
  work.width = W; work.height = H;
  ctx.drawImage(video, 0, 0, W, H);
  const px = ctx.getImageData(0, 0, W, H).data;

  const fg = [], bg = [];
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const person = seg?.mask &&
        seg.mask[Math.floor(y * seg.h / H) * seg.w + Math.floor(x * seg.w / W)] > 127;
      (person ? fg : bg).push(i);
    }
  const rnd = mulberry32(0);
  const pick = (list, n) => Array.from({ length: n }, () => list[Math.floor(rnd() * list.length)]);
  let idx;
  if (fg.length && bg.length) {
    const nf = Math.min(fg.length, Math.round(SAMPLES * FG_SHARE));
    idx = pick(fg, nf).concat(pick(bg, SAMPLES - nf));
  } else {
    idx = pick(fg.length ? fg : bg, SAMPLES);
  }
  const n = idx.length, pts = new Float32Array(n * 3);
  idx.forEach((p, j) => { pts[j * 3] = px[p * 4]; pts[j * 3 + 1] = px[p * 4 + 1]; pts[j * 3 + 2] = px[p * 4 + 2]; });

  // k-means++ seeding, then Lloyd iterations
  const c = new Float32Array(K * 3), d2 = new Float32Array(n).fill(Infinity);
  const dist = (j, k) => {
    const a = pts[j * 3] - c[k * 3], b = pts[j * 3 + 1] - c[k * 3 + 1], e = pts[j * 3 + 2] - c[k * 3 + 2];
    return a * a + b * b + e * e;
  };
  let first = Math.floor(rnd() * n);
  c.set(pts.subarray(first * 3, first * 3 + 3), 0);
  for (let k = 1; k < K; k++) {
    let total = 0;
    for (let j = 0; j < n; j++) { d2[j] = Math.min(d2[j], dist(j, k - 1)); total += d2[j]; }
    let r = rnd() * total, j = 0;
    while (j < n - 1 && (r -= d2[j]) > 0) j++;
    c.set(pts.subarray(j * 3, j * 3 + 3), k * 3);
  }
  const assign = new Int32Array(n), sum = new Float64Array(K * 3), cnt = new Int32Array(K);
  for (let it = 0; it < ITERS; it++) {
    sum.fill(0); cnt.fill(0);
    for (let j = 0; j < n; j++) {
      let best = 0, bd = Infinity;
      for (let k = 0; k < K; k++) { const d = dist(j, k); if (d < bd) { bd = d; best = k; } }
      assign[j] = best; cnt[best]++;
      sum[best * 3] += pts[j * 3]; sum[best * 3 + 1] += pts[j * 3 + 1]; sum[best * 3 + 2] += pts[j * 3 + 2];
    }
    for (let k = 0; k < K; k++) {
      if (!cnt[k]) continue;                    // an empty cluster keeps its old centre
      c[k * 3] = sum[k * 3] / cnt[k]; c[k * 3 + 1] = sum[k * 3 + 1] / cnt[k]; c[k * 3 + 2] = sum[k * 3 + 2] / cnt[k];
    }
  }
  return c.map(v => v / 255);
}
