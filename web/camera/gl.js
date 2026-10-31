// Minimal WebGL2 plumbing: fullscreen-triangle passes, textures, render targets.
//
// Orientation convention used everywhere: texture row 0 is the TOP of the image
// (UNPACK_FLIP_Y stays false), and every offscreen pass treats gl_FragCoord.y as a
// y-down row index. Only the final pass to the canvas flips, because the canvas is
// the one surface whose row 0 is displayed at the bottom.

const VS = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export function createGL(canvas) {
  const gl = canvas.getContext('webgl2', {
    alpha: false, antialias: false, depth: false, stencil: false,
    premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance',
  });
  if (!gl) throw new Error('This browser has no WebGL2, which the live styles need');
  gl.bindVertexArray(gl.createVertexArray());   // attribute-less draw still needs a VAO bound
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  return gl;
}

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const numbered = src.split('\n').map((l, i) => `${String(i + 1).padStart(4)}  ${l}`).join('\n');
    throw new Error(`shader compile failed:\n${gl.getShaderInfoLog(s)}\n${numbered}`);
  }
  return s;
}

export class Pass {
  constructor(gl, fs) {
    this.gl = gl;
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('link failed: ' + gl.getProgramInfoLog(p));
    this.p = p;
    this.loc = new Map();
    this.unit = 0;
  }
  use() { this.gl.useProgram(this.p); this.unit = 0; return this; }
  _l(name) {
    if (!this.loc.has(name)) this.loc.set(name, this.gl.getUniformLocation(this.p, name));
    return this.loc.get(name);
  }
  // unused uniforms are optimised out and have a null location; setting them is a no-op
  f(name, ...v) {
    const l = this._l(name), gl = this.gl;
    if (l !== null) [null, gl.uniform1f, gl.uniform2f, gl.uniform3f, gl.uniform4f][v.length].call(gl, l, ...v);
    return this;
  }
  i(name, v) { const l = this._l(name); if (l !== null) this.gl.uniform1i(l, v); return this; }
  v3(name, arr) { const l = this._l(name); if (l !== null) this.gl.uniform3fv(l, arr); return this; }
  tex(name, texture) {
    const gl = this.gl, unit = this.unit++;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const l = this._l(name);
    if (l !== null) gl.uniform1i(l, unit);
    return this;
  }
  // target: a Target, or null for the canvas
  draw(target) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fb : null);
    gl.viewport(0, 0, target ? target.w : gl.canvas.width, target ? target.h : gl.canvas.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (target?.mip) {
      gl.bindTexture(gl.TEXTURE_2D, target.tex);
      gl.generateMipmap(gl.TEXTURE_2D);
    }
  }
}

export function texture(gl, mip = false) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mip ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return t;
}

// video / canvas / image -> RGBA8; mipmaps double as cheap box blurs and cell averages
export function uploadImage(gl, tex, src, mip = false) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
  if (mip) gl.generateMipmap(gl.TEXTURE_2D);
}

export function uploadR8(gl, tex, w, h, data, mip = false) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, data);
  if (mip) gl.generateMipmap(gl.TEXTURE_2D);
}

export class Target {
  constructor(gl, mip = false) {
    this.gl = gl; this.mip = mip; this.w = 0; this.h = 0;
    this.tex = texture(gl, mip);
    this.fb = gl.createFramebuffer();
  }
  resize(w, h) {
    if (w === this.w && h === this.h) return this;
    const gl = this.gl;
    this.w = w; this.h = h;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
    // a mipmapped texture is "incomplete" until its levels exist; build them now so a
    // target that is bound before it is first drawn samples as black, not an error
    if (this.mip) gl.generateMipmap(gl.TEXTURE_2D);
    return this;
  }
}
