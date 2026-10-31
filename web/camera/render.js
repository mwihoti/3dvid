// Per-frame GPU pipeline: camera -> composite (layout + background) -> canvas.
import { createGL, Pass, Target, texture, uploadImage, uploadR8 } from './gl.js';
import * as S from './shaders.js';

// same canvases as stylize.py's LAYOUTS: (width, height, subject centre as a fraction of width)
const LAYOUTS = { wide: [1920, 1080, 0.72], portrait: [1080, 1920, 0.50], square: [1080, 1080, 0.50] };
const BG = { plasma: 0, grid: 1, dark: 2, filmed: 3, source: 4 };
const EFFECT = { voxel: 0, contour: 1, both: 2, clean: 3 };
const FX_LONG_SIDE = 1280;   // effects run at camera resolution, capped for 1080p+ cameras

export function layoutDims(layout, quality, vw, vh) {
  if (layout === 'native') return [vw - (vw % 2), vh - (vh % 2), 0.5];
  const [W, H, ax] = LAYOUTS[layout];
  const k = Math.min(1, +quality / Math.min(W, H));     // 720p -> 1280x720, 720x1280, 720x720
  const even = v => Math.round(v * k / 2) * 2;
  return [even(W), even(H), ax];
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = this.gl = createGL(canvas);
    this.fxPass = new Pass(gl, S.FX);
    this.composite = new Pass(gl, S.COMPOSITE);
    this.output = new Pass(gl, S.OUTPUT);
    this.video = texture(gl, true);
    this.mask = texture(gl, true);
    this.fx = new Target(gl, true);      // mipmapped: the filmed-FX background blurs it
    this.comp = new Target(gl);
  }

  // seg: a Segmenter (or null while it loads); t: seconds, drives the animated backgrounds
  frame({ video, settings, seg, t }) {
    const gl = this.gl;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return false;
    const [CW, CH, ax] = layoutDims(settings.layout, settings.quality, vw, vh);
    if (this.canvas.width !== CW || this.canvas.height !== CH) {
      this.canvas.width = CW; this.canvas.height = CH;
    }
    this.comp.resize(CW, CH);
    uploadImage(gl, this.video, video, true);
    const hasMask = !!seg?.mask;
    if (hasMask) uploadR8(gl, this.mask, seg.w, seg.h, seg.mask, true);

    const fk = Math.min(1, FX_LONG_SIDE / Math.max(vw, vh));
    const FW = Math.round(vw * fk), FH = Math.round(vh * fk);
    this.fx.resize(FW, FH);
    this.fxPass.use()
      .tex('uVideo', this.video).tex('uMask', this.mask)
      .i('uMirror', settings.mirror ? 1 : 0).f('uSrc', FW, FH).i('uEffect', EFFECT[settings.effect])
      .draw(this.fx);

    const sc = CH / vh, nw = vw * sc, ox = CW * ax - nw / 2;
    const k = Math.min(CW, CH) / 1080;
    // head in canvas px; without a person, head_bbox's fallback (centre, upper third)
    const st = seg?.stats?.present ? seg.stats : { headX: 0.5, headY: 0.35, headW: 0.4 };
    const hx = ox + (settings.mirror ? 1 - st.headX : st.headX) * nw;
    const head = [hx, st.headY * CH, Math.max(st.headW * nw, 40 * k)];

    this.composite.use()
      .tex('uVideo', this.video).tex('uFx', this.fx.tex).tex('uMask', this.mask)
      .i('uHasMask', hasMask ? 1 : 0).i('uMirror', settings.mirror ? 1 : 0)
      .f('uCanvas', CW, CH).f('uSrc', vw, vh).f('uSc', sc).f('uOx', ox)
      .i('uBg', BG[settings.bg]).f('uHead', ...head).f('uK', k).f('uTime', t)
      .draw(this.comp);

    this.output.use().tex('uComp', this.comp.tex).f('uCanvas', CW, CH).draw(null);
    return true;
  }
}
