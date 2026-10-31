// Per-frame GPU pipeline: camera -> composite (layout + background) -> canvas.
import { createGL, Pass, Target, texture, uploadImage, uploadR8 } from './gl.js';
import * as S from './shaders.js';

// same canvases as stylize.py's LAYOUTS: (width, height, subject centre as a fraction of width)
const LAYOUTS = { wide: [1920, 1080, 0.72], portrait: [1080, 1920, 0.50], square: [1080, 1080, 0.50] };
const BG = { plasma: 0, grid: 1, dark: 2, filmed: 3, source: 4 };
const EFFECT = { voxel: 0, contour: 1, both: 2, clean: 3 };
const FACE = { off: 0, clear: 1, only: 2 };
const FACE_PAD = 0.05;       // the studio's default chin/beard slack, in person heights
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
    this.edgePass = new Pass(gl, S.EDGES);
    this.fxPass = new Pass(gl, S.FX);
    this.composite = new Pass(gl, S.COMPOSITE);
    this.output = new Pass(gl, S.OUTPUT);
    this.video = texture(gl, true);
    this.mask = texture(gl, true);
    this.edges = new Target(gl, true).resize(1, 1);   // mipmapped: bloom/rim are its blurred levels
    this.fx = new Target(gl, true);      // mipmapped: the filmed-FX background blurs it
    this.comp = new Target(gl);
  }

  // seg: a Segmenter (or null while it loads); t: seconds, drives the animated backgrounds
  // palette: Float32Array(72) from buildPalette
  frame({ video, settings, seg, t, palette }) {
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
    const st = seg?.stats?.present ? seg.stats : null;
    const minC = Math.max(2, settings.minCell), maxC = Math.max(minC, settings.maxCell);
    // quadtree cells are min_cell * 2^k; the top level is the first one >= max_cell
    const top = minC * 2 ** Math.ceil(Math.log2(maxC / minC));
    if (settings.effect === 'contour' || settings.effect === 'both') {
      this.edges.resize(FW, FH);
      this.edgePass.use()
        .tex('uVideo', this.video).tex('uMask', this.mask).i('uHasMask', hasMask ? 1 : 0)
        .i('uMirror', settings.mirror ? 1 : 0).f('uSrc', FW, FH)
        // stylize.py drifts the band thresholds by 0.05 per frame; 30 fps keeps the speed
        .f('uBands', settings.bands).f('uPhase', t * 30 * 0.05)
        .draw(this.edges);
    }
    this.fxPass.use()
      .tex('uVideo', this.video).tex('uMask', this.mask).tex('uEdge', this.edges.tex)
      .i('uHasMask', hasMask ? 1 : 0)
      .i('uMirror', settings.mirror ? 1 : 0).f('uSrc', FW, FH).i('uEffect', EFFECT[settings.effect])
      .f('uPerson', st ? st.top : 0, st ? st.faceBottom : 0, st ? 1 : 0)
      .v3('uPal', palette).f('uMinCell', minC).f('uMaxCell', maxC).f('uTopCell', top).f('uCamPerFx', vw / FW)
      .draw(this.fx);

    const sc = CH / vh, nw = vw * sc, ox = CW * ax - nw / 2;
    const k = Math.min(CW, CH) / 1080;
    // head in canvas px; without a person, head_bbox's fallback (centre, upper third)
    const hs = st || { headX: 0.5, headY: 0.35, headW: 0.4 };
    const hx = ox + (settings.mirror ? 1 - hs.headX : hs.headX) * nw;
    const head = [hx, hs.headY * CH, Math.max(hs.headW * nw, 40 * k)];

    this.composite.use()
      .tex('uVideo', this.video).tex('uFx', this.fx.tex).tex('uMask', this.mask)
      .i('uHasMask', hasMask ? 1 : 0).i('uMirror', settings.mirror ? 1 : 0)
      .f('uCanvas', CW, CH).f('uSrc', vw, vh).f('uSc', sc).f('uOx', ox)
      .i('uBg', BG[settings.bg]).f('uHead', ...head).f('uK', k).f('uTime', t)
      // no person, or the clean effect: nothing to protect
      .i('uFace', st && settings.effect !== 'clean' ? FACE[settings.face] : 0)
      .f('uFaceRows', st ? st.top : 0, st ? Math.min(1, st.cut + FACE_PAD * (st.bottom - st.top)) : 0)
      .draw(this.comp);

    this.output.use().tex('uComp', this.comp.tex).f('uCanvas', CW, CH)
      .i('uChroma', settings.chroma ? 1 : 0).f('uGrain', settings.grain).f('uK', k)
      .f('uSeed', (t * 997) % 1000)
      .draw(null);
    return true;
  }
}
