// Per-frame GPU pipeline: camera -> composite (layout + background) -> canvas.
import { createGL, Pass, Target, texture, uploadImage } from './gl.js';
import * as S from './shaders.js';

// same canvases as stylize.py's LAYOUTS: (width, height, subject centre as a fraction of width)
const LAYOUTS = { wide: [1920, 1080, 0.72], portrait: [1080, 1920, 0.50], square: [1080, 1080, 0.50] };

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
    this.composite = new Pass(gl, S.COMPOSITE);
    this.output = new Pass(gl, S.OUTPUT);
    this.video = texture(gl, true);
    this.comp = new Target(gl);
  }

  frame({ video, settings }) {
    const gl = this.gl;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return false;
    const [CW, CH, ax] = layoutDims(settings.layout, settings.quality, vw, vh);
    if (this.canvas.width !== CW || this.canvas.height !== CH) {
      this.canvas.width = CW; this.canvas.height = CH;
    }
    this.comp.resize(CW, CH);
    uploadImage(gl, this.video, video, true);

    const sc = CH / vh, nw = vw * sc;
    this.composite.use()
      .tex('uVideo', this.video)
      .i('uMirror', settings.mirror ? 1 : 0)
      .f('uCanvas', CW, CH).f('uSrc', vw, vh).f('uSc', sc).f('uOx', CW * ax - nw / 2)
      .draw(this.comp);

    this.output.use().tex('uComp', this.comp.tex).f('uCanvas', CW, CH).draw(null);
    return true;
  }
}
