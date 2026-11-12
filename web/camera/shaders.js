// GLSL for the live styles. Ports of stylize.py's effects / backgrounds / compose,
// adapted to run per pixel at camera frame rate. Coordinates are y-down pixels
// (see gl.js); uMirror flips camera-space lookups only.

const COMMON = `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
uniform bool uMirror;
const float PI = 3.14159265;
vec2 camUV(vec2 uv) { return uMirror ? vec2(1.0 - uv.x, uv.y) : uv; }
`;

// shared by the passes that need the person mask and the stand-in depth
const DEPTH = `
uniform sampler2D uVideo;
uniform sampler2D uMask;
uniform bool uHasMask;
uniform vec2 uSrc;      // fx target px
float maskAt(vec2 uv) { return uHasMask ? texture(uMask, camUV(uv)).r : 1.0; }

// Stand-in for Depth-Anything: there is no depth model running live, so
// "nearness" grows with distance inside the silhouette (a heavily blurred mask),
// plus a little luminance for relief. The background is far (largest cells),
// exactly where the percentile stretch puts it in voxel_effect.
float nearness(vec2 uv) {
  if (!uHasMask) return 0.5;
  vec2 cuv = camUV(uv);
  if (texture(uMask, cuv).r < 0.35) return 0.0;
  // wide blur (mip 4 of a 256 px mask) so the gradient spans the whole torso,
  // not just a rim: cubes then grow from the face out to the shoulders
  float body = textureLod(uMask, cuv, 4.0).r;
  float lum = dot(textureLod(uVideo, cuv, 2.0).rgb, vec3(0.299, 0.587, 0.114));
  return smoothstep(0.30, 1.0, body) * 0.8 + 0.2 * lum;
}

`;

// value noise for the animated backgrounds and the hand-drawn / glitch effects
const NOISE = `
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x),
             mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
`;

// glyphs in the ASCII atlas (render.js draws them, sparse to dense, side by side)
export const GLYPHS = ' .:-=+*#%@';
export const GLYPH_W = 40, GLYPH_H = 64;

// camera-sized: the stylised frame, in display (mirrored) space. Effects render the
// whole frame like stylize.py's do; the composite decides where it shows.
export const FX = COMMON + DEPTH + NOISE + `
uniform sampler2D uEdge;  // EDGES output: rgb = coloured band edges, a = raw rim
uniform int uEffect;    // index in effects.js: 0 voxel, 1 contour, 2 both, 3 clean, 4.. live only
uniform vec3 uPerson;   // top, face bottom (0..1, y down), present
uniform vec3 uPal[24];  // build_palette's 24 colours
uniform float uMinCell, uMaxCell, uTopCell, uCamPerFx;
uniform float uTime;      // seconds, for the animated effects
uniform sampler2D uGlyphs;  // ASCII atlas: GLYPHS cells of GLYPH_W x GLYPH_H, white on black
const float GLYPH_N = ${GLYPHS.length}.0, GLYPH_H = ${GLYPH_H}.0;

// face = person AND the top 30% of the person's vertical extent
bool isFace(vec2 uv) {
  return uPerson.z > 0.5 && uv.y >= uPerson.x && uv.y <= uPerson.y && maskAt(uv) > 0.35;
}

float wantCell(vec2 pix) {
  vec2 uv = pix / uSrc;
  float s = mix(uMaxCell, uMinCell, nearness(uv));   // near -> min_cell, far -> max_cell
  return isFace(uv) ? s / 1.5 : s;
}

vec3 quantise(vec3 c) {
  vec3 best = c; float bd = 1e9;
  for (int i = 0; i < 24; i++) {
    vec3 d = c - uPal[i]; float e = dot(d, d);
    if (e < bd) { bd = e; best = uPal[i]; }
  }
  return best;
}

// voxel_effect's greedy raster tiling is sequential; a quadtree gives the same
// "small cubes near, big cubes far" tiling without overlaps, per pixel: a block
// splits while the depth at its centre asks for cells well under its size.
vec3 voxel(vec2 pix) {
  float size = uTopCell;
  vec2 org = floor(pix / size) * size;
  for (int i = 0; i < 6; i++) {
    if (wantCell(org + size * 0.5) >= size * 0.7 || size * 0.5 < uMinCell - 0.01) break;
    size *= 0.5;
    org = floor(pix / size) * size;
  }
  vec2 c = (org + size * 0.5) / uSrc;
  // the mip level whose texels are cell-sized holds the cell's mean colour
  vec3 col = textureLod(uVideo, camUV(c), log2(size * uCamPerFx)).rgb;
  if (isFace(c)) col = clamp((col - 0.5) * 1.18 + 0.5, 0.0, 1.0);   // features survive quantising
  col = quantise(col);

  vec2 l = pix - org;                                 // y down inside the cube
  float t = max(1.0, floor(size / 4.0)), b = max(1.0, floor(size / 5.0));
  float shade = 1.0;
  if (l.y < t) shade = 1.20;                          // top face
  else if (l.x < b) shade = 0.85;                     // left face
  else if (l.x >= size - b) shade = 0.70;             // right face
  if (l.y < 1.0 || l.x < 1.0) shade = 0.35;           // 1 px dark seam
  return clamp(col * shade, 0.0, 1.0);
}

const vec3 NAVY = vec3(0x0a, 0x12, 0x30) / 255.0;
const vec3 CYAN = vec3(0x39, 0xc6, 0xff) / 255.0;

// contour_effect's second half: navy + anti-aliased edges + bloom + cyan rim
vec3 contour(vec2 pix) {
  vec2 t = pix / uSrc;
  vec3 col = NAVY + textureLod(uEdge, t, 0.4).rgb;        // 0.4: a touch of the 3x3 AA blur
  // GaussianBlur(edge, glow 9) * 0.55; mip 2 (a 4x4 average) is the same width of blur
  col += textureLod(uEdge, t, 2.0).rgb * 0.55;
  float rim = clamp(textureLod(uEdge, t, 2.5).a * 4.0, 0.0, 1.0);
  col += CYAN * pow(rim, 1.5) * 1.4;
  return clamp(col, 0.0, 1.0);
}

// ---- live-only effects (no stylize.py counterpart)

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
// lod 0 of the camera is finer than an fx pixel when the camera is over 1280 px
float pixLod() { return log2(max(uCamPerFx, 1.0)); }
vec3 camAt(vec2 pix, float lod) { return textureLod(uVideo, camUV(pix / uSrc), lod).rgb; }
vec3 boostSat(vec3 c, float k) { return clamp(mix(vec3(luma(c)), c, k), 0.0, 1.0); }

// Sobel on luminance, taps r fx px apart, at a pre-blurred mip so sensor noise
// and skin texture do not turn into strokes
float sobel(vec2 pix, float r, float lod) {
  float tl = luma(camAt(pix + vec2(-r, -r), lod)), tc = luma(camAt(pix + vec2(0, -r), lod));
  float tr = luma(camAt(pix + vec2(r, -r), lod)), ml = luma(camAt(pix + vec2(-r, 0), lod));
  float mr = luma(camAt(pix + vec2(r, 0), lod)), bl = luma(camAt(pix + vec2(-r, r), lod));
  float bc = luma(camAt(pix + vec2(0, r), lod)), br = luma(camAt(pix + vec2(r, r), lod));
  float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
  float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
  return length(vec2(gx, gy));
}
// the person's outline: the mask's gradient, r fx px across (the mask is ~5x coarser
// than the frame, so this is already a few px wide)
float silhouette(vec2 uv, float r) {
  if (!uHasMask) return 0.0;
  vec2 d = r / uSrc;
  vec2 g = vec2(maskAt(uv + vec2(d.x, 0)) - maskAt(uv - vec2(d.x, 0)),
                maskAt(uv + vec2(0, d.y)) - maskAt(uv - vec2(0, d.y)));
  return length(g);
}

// ASCII: one glyph per cell, picked from a sparse-to-dense ramp by the cell's
// mean luminance and tinted with the cell's own colour
vec3 ascii(vec2 pix) {
  float ch = max(7.0, floor(uMinCell * 1.25 + 0.5));
  vec2 cell = vec2(floor(ch * 0.62 + 0.5), ch);     // monospace aspect
  vec2 org = floor(pix / cell) * cell;
  vec2 c = (org + cell * 0.5) / uSrc;
  float lod = log2(sqrt(cell.x * cell.y) * uCamPerFx);
  vec3 col = textureLod(uVideo, camUV(c), lod).rgb;
  // local contrast against a ~4-cell neighbourhood, so eyes and mouth pick different
  // glyphs from the skin around them, not just a darker shade of the same one
  float l = luma(col), around = luma(textureLod(uVideo, camUV(c), lod + 2.0).rgb);
  float t = pow(clamp((l + (l - around) * 1.6 - 0.03) / 0.8, 0.0, 1.0), 0.8);
  float idx = floor(t * (GLYPH_N - 1.0) + 0.5);
  if (maskAt(c) > 0.5) idx = max(idx, 1.0);            // keep the silhouette readable in the shadows
  vec2 local = (pix - org) / cell;
  float g = textureLod(uGlyphs, vec2((idx + local.x) / GLYPH_N, local.y), log2(GLYPH_H / ch) - 0.5).r;   // a touch sharper than the cell's mip
  // the cell's hue at full brightness, then a little more saturated
  vec3 tint = boostSat(col / max(max(col.r, max(col.g, col.b)), 0.18), 1.35);
  vec3 ink = tint * (0.3 + 0.9 * t);
  // a faint wash of the cell colour keeps the picture legible from a distance
  return mix(vec3(0.012, 0.016, 0.024) + col * 0.16, ink, g);
}

// pop-art halftone: a 45-degree dot screen, dot area following tone, each dot one
// palette colour, on dark ink
vec3 halftone(vec2 pix) {
  float d = max(7.0, uMinCell * 1.15);
  const mat2 R = mat2(0.70710678, 0.70710678, -0.70710678, 0.70710678);
  vec2 q = R * pix;
  vec2 cq = (floor(q / d) + 0.5) * d;
  vec2 cp = transpose(R) * cq;                           // dot centre, frame px
  vec3 col = textureLod(uVideo, camUV(cp / uSrc), log2(d * uCamPerFx)).rgb;
  float t = pow(clamp((luma(col) - 0.02) / 0.85, 0.0, 1.0), 0.75);
  vec3 ink = boostSat(quantise(col), 1.7) * 1.12;
  float r = d * 0.56 * sqrt(t);                          // just touching neighbours at full tone
  float dist = length(q - cq);
  float aa = length(fwidth(q)) * 0.6;
  float dotA = 1.0 - smoothstep(r - aa, r + aa, dist);
  vec3 paper = vec3(0x0b, 0x10, 0x2a) / 255.0;
  return clamp(mix(paper, ink, dotA), 0.0, 1.0);
}

// RGB <-> HSV (Sam Hocevar's branchless form)
vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + 1e-10)), d / (q.x + 1e-10), q.x);
}
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

// comic: flat cel tones, bold ink lines, a dot screen in the shadow band
vec3 comic(vec2 pix) {
  vec2 uv = pix / uSrc;
  float L = pixLod();
  vec3 hsv = rgb2hsv(camAt(pix, L + 1.0));
  float v = clamp(pow(hsv.z, 0.8) * 1.1, 0.0, 0.999);
  float lv = floor(v * 4.0);                              // 4 cel levels
  float vq = lv < 0.5 ? 0.14 : lv < 1.5 ? 0.48 : lv < 2.5 ? 0.78 : 1.0;
  vec3 col = hsv2rgb(vec3(hsv.x, clamp(hsv.y * 1.4 + 0.05, 0.0, 1.0), vq));
  if (lv > 0.5 && lv < 1.5) {
    // Ben-Day dots in the shadow band, on their own 45-degree grid
    vec2 q = mat2(0.70710678, 0.70710678, -0.70710678, 0.70710678) * pix / 5.0;
    float dd = length(fract(q) - 0.5);
    col *= 1.0 - 0.55 * (1.0 - smoothstep(0.24, 0.32, dd));
  }
  float e = sobel(pix, 1.5, L + 1.0);
  float ink = smoothstep(0.30, 0.48, e);
  ink = max(ink, smoothstep(0.35, 0.7, silhouette(uv, 3.0)));
  return mix(col, vec3(0.02, 0.02, 0.04), ink);
}

// thermal camera: false-colour heat from nearness (the person is warm, the face
// hottest) and luminance, slightly soft like a microbolometer
vec3 heatRamp(float h) {
  h = clamp(h, 0.0, 1.0) * 5.0;
  vec3 c = mix(vec3(0.0, 0.0, 0.02), vec3(0.30, 0.04, 0.50), clamp(h, 0.0, 1.0));       // black -> purple
  c = mix(c, vec3(0.86, 0.06, 0.24), clamp(h - 1.0, 0.0, 1.0));                       // -> red
  c = mix(c, vec3(1.0, 0.48, 0.02), clamp(h - 2.0, 0.0, 1.0));                         // -> orange
  c = mix(c, vec3(1.0, 0.88, 0.16), clamp(h - 3.0, 0.0, 1.0));                         // -> yellow
  return mix(c, vec3(1.0, 1.0, 0.94), clamp(h - 4.0, 0.0, 1.0));                       // -> white
}
vec3 thermal(vec2 pix) {
  vec2 uv = pix / uSrc;
  vec3 col = camAt(pix, pixLod() + 2.5);
  float l = luma(col);
  float body = uHasMask ? textureLod(uMask, camUV(uv), 3.0).r : 0.6;
  // skin runs hotter than clothes and hair: a red-over-blue cue, from a wide sample
  // so a specular highlight (r ~ b there) still reads as skin, not a cold spot
  vec3 sc = camAt(pix, pixLod() + 4.5);
  float skin = smoothstep(0.04, 0.2, sc.r - sc.b) * step(sc.b, sc.g + 0.04) * smoothstep(0.15, 0.4, luma(sc));
  float face = uPerson.z * smoothstep(uPerson.y + 0.06, uPerson.y - 0.02, uv.y) * body;
  float heat = 0.06 + 0.2 * l + body * (0.1 + 0.28 * nearness(uv)) + 0.2 * skin + 0.08 * face;
  return heatRamp(heat);
}

// hologram: see-through cyan projection, scanlines, a sweep, flicker, a rim glow and
// now and then a glitch that shears a few horizontal slices. Returns alpha too: the
// dark fill lets the background show through in the composite.
vec4 hologram(vec2 pix) {
  vec2 uv = pix / uSrc;
  float blk = floor(uTime * 7.0);                        // glitch decisions change ~7x a second
  float glitch = step(0.9, hash12(vec2(blk, 3.7)));
  float slice = floor(uv.y * 30.0 + hash12(vec2(blk, 9.1)) * 30.0);
  float shear = glitch * step(0.6, hash12(vec2(slice, blk))) * (hash12(vec2(slice + 5.0, blk)) - 0.5) * 0.07;
  uv.x += shear;
  vec2 p = uv * uSrc;
  float L = pixLod();
  float l = luma(camAt(p, L + 0.5));
  float e = smoothstep(0.12, 0.45, sobel(p, 1.0, L + 0.8));
  vec3 deep = vec3(0.0, 0.07, 0.20), cyan = vec3(0.22, 0.80, 1.0), white = vec3(0.80, 1.0, 1.0);
  float tone = pow(clamp((l - 0.04) / 0.8, 0.0, 1.0), 0.8);
  vec3 col = mix(deep, cyan, tone) * (0.25 + 0.75 * tone);
  col = mix(col, white, smoothstep(0.7, 1.0, tone) * 0.5);
  col += cyan * e * 0.9;
  col *= 0.72 + 0.28 * cos(pix.y * PI * 0.5);           // 4 px scanlines (on the unsheared row)
  float sweep = fract(uTime * 0.16) * 1.5 - 0.25;
  col += cyan * exp(-pow((uv.y - sweep) * 12.0, 2.0)) * 0.45;
  float rim = 0.0;
  if (uHasMask) {
    vec2 cuv = camUV(uv);
    rim = clamp((texture(uMask, cuv).r - textureLod(uMask, cuv, 3.0).r) * 3.5, 0.0, 1.0);
  }
  col += cyan * rim * 1.3 + white * rim * rim * 0.4;
  col *= 0.9 + 0.1 * hash12(vec2(floor(uTime * 24.0), 1.7)) - glitch * 0.08;
  if (glitch > 0.5 && shear != 0.0) col.rb += vec2(0.25, -0.1) * abs(shear) * 14.0;   // colour split on the sheared slices
  float alpha = clamp(0.38 + 0.9 * max(luma(col), rim), 0.0, 1.0);
  return vec4(clamp(col, 0.0, 1.0), alpha);
}

// 8-bit: big square pixels, palette colours mixed by a 4x4 Bayer ordered dither
const float BAYER[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0,
                                  3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
vec3 pixel(vec2 pix) {
  float size = max(uMinCell, 6.0);
  vec2 cell = floor(pix / size);
  vec2 c = (cell + 0.5) * size / uSrc;
  vec3 col = textureLod(uVideo, camUV(c), log2(size * uCamPerFx)).rgb;
  col = boostSat(clamp((col - 0.5) * 1.15 + 0.5, 0.0, 1.0), 1.1);
  // the two nearest palette colours, and how far col sits from the first toward the second
  vec3 a = uPal[0], b = uPal[1];
  float da = 1e9, db = 1e9;
  for (int i = 0; i < 24; i++) {
    vec3 d = col - uPal[i]; float e = dot(d, d);
    if (e < da) { db = da; b = a; da = e; a = uPal[i]; }
    else if (e < db) { db = e; b = uPal[i]; }
  }
  vec3 ab = b - a;
  float f = clamp(dot(col - a, ab) / max(dot(ab, ab), 1e-5), 0.0, 1.0);
  ivec2 m = ivec2(mod(cell, 4.0));
  float th = (BAYER[m.y * 4 + m.x] + 0.5) / 16.0;
  return f > th ? b : a;
}

// pencil sketch: graphite on paper - outlines from edges, then hatching that gains a
// second and a third direction as the tone darkens. The strokes are redrawn a few
// times a second with a small jitter, like consecutive hand-drawn frames.
float hatch(vec2 p, vec2 dir, float spacing, float w) {
  float d = abs(fract(dot(p, dir) / spacing) - 0.5) * spacing;
  // fades in with w, or a zero-width line would still leave a half-covered pixel
  return (1.0 - smoothstep(w - 0.6, w + 0.6, d)) * smoothstep(0.0, 0.3, w);
}
vec3 sketch(vec2 pix) {
  float blk = floor(uTime * 5.0);
  vec2 jit = vec2(vnoise(pix * 0.045 + blk * 17.0), vnoise(pix * 0.045 + 50.0 + blk * 11.0)) - 0.5;
  vec2 p = pix + jit * 3.0;
  vec2 uv = p / uSrc;
  float L = pixLod();
  float fine = luma(camAt(p, L + 0.5)), wide = luma(camAt(p, L + 2.5));
  float tone = pow(clamp((wide - 0.04) / 0.8, 0.0, 1.0), 0.7);
  float dark = 1.0 - tone;
  // outlines: difference of Gaussians, inked only where it is darker than its
  // surroundings - thin pencil lines along features rather than Sobel's double edges
  float ink = smoothstep(0.025, 0.09, wide - fine) * 0.9;
  ink = max(ink, smoothstep(0.3, 0.7, silhouette(uv, 2.0)) * 0.8);
  // hatching: wobble the line coordinate a little so strokes are not ruler-straight
  vec2 hp = p + (vnoise(p * 0.02) - 0.5) * 4.0;
  ink = max(ink, hatch(hp, vec2(0.70710678, 0.70710678), 6.0, clamp((dark - 0.4) * 1.6, 0.0, 0.8)) * 0.5);
  ink = max(ink, hatch(hp, vec2(0.70710678, -0.70710678), 6.0, clamp((dark - 0.6) * 2.0, 0.0, 0.8)) * 0.6);
  ink = max(ink, hatch(hp, vec2(0.0, 1.0), 4.0, clamp((dark - 0.8) * 3.0, 0.0, 0.9)) * 0.7);
  float grain = (hash12(floor(pix)) - 0.5) * 0.05 + (vnoise(pix * vec2(0.012, 0.05)) - 0.5) * 0.05;
  vec3 paper = vec3(0.955, 0.94, 0.90) + grain;
  vec3 graphite = vec3(0.17, 0.17, 0.20);
  return clamp(mix(paper, graphite, ink), 0.0, 1.0);
}

void main() {
  vec2 pix = gl_FragCoord.xy;
  vec3 col;
  float alpha = 1.0;
  if (uEffect == 0) col = voxel(pix);
  else if (uEffect == 1) col = contour(pix);
  else if (uEffect == 2) col = clamp(voxel(pix) * 0.45 + contour(pix) * 0.75, 0.0, 1.0);
  else if (uEffect == 4) col = ascii(pix);
  else if (uEffect == 5) col = halftone(pix);
  else if (uEffect == 6) col = comic(pix);
  else if (uEffect == 7) col = thermal(pix);
  else if (uEffect == 8) { vec4 h = hologram(pix); col = h.rgb; alpha = h.a; }
  else if (uEffect == 9) col = pixel(pix);
  else if (uEffect == 10) col = sketch(pix);
  else col = texture(uVideo, camUV(pix / uSrc)).rgb;
  // alpha < 1 = see-through (the composite shows the background there)
  outColor = vec4(col, alpha);
}`;

// contour_effect's first half, at fx resolution: quantise a luminance + depth
// field into bands, mark where the band index changes, colour by band.
export const EDGES = COMMON + DEPTH + `
uniform float uBands, uPhase;
const vec3 R0 = vec3(0xff, 0x6a, 0x00) / 255.0;   // orange
const vec3 R1 = vec3(0xff, 0x2d, 0x95) / 255.0;   // pink
const vec3 R2 = vec3(0x8a, 0x2b, 0xff) / 255.0;   // violet

vec3 ramp(float u) {
  u = clamp(u, 0.0, 1.0) * 2.0;
  return u < 1.0 ? mix(R0, R1, u) : mix(R1, R2, u - 1.0);
}
// mip 1 ~ the 5x5 GaussianBlur stylize.py applies to luminance first
float lumAt(vec2 uv) { return dot(textureLod(uVideo, camUV(uv), 1.0).rgb, vec3(0.299, 0.587, 0.114)); }
float band(vec2 pix) {
  vec2 uv = pix / uSrc;
  return floor((0.70 * lumAt(uv) + 0.30 * nearness(uv)) * uBands + uPhase);
}

void main() {
  vec2 p = gl_FragCoord.xy;
  float q = band(p);
  float edge = (q != band(p + vec2(1, 0)) || q != band(p + vec2(0, 1))) ? 1.0 : 0.0;
  vec3 col = ramp(mod(q, uBands) / max(uBands - 1.0, 1.0)) * edge;

  // cyan rim on the brighter side of the face: the mask's x-gradient where it
  // opposes the luminance gradient (Sobel x on both in stylize.py)
  vec2 uv = p / uSrc, dx = vec2(2.0 / uSrc.x, 0.0);
  float gm = maskAt(uv + dx) - maskAt(uv - dx);
  float gl = lumAt(uv + dx) - lumAt(uv - dx);
  float rim = uHasMask ? clamp(-sign(gl) * gm * 3.0, 0.0, 1.0) : 0.0;
  outColor = vec4(col, rim);
}`;

// canvas-sized: places the camera frame like compose() - scaled to the canvas
// height, centred at uOx - and fills everything else with the background.
export const COMPOSITE = COMMON + `
uniform sampler2D uVideo;
uniform sampler2D uFx;  // stylised frame, display space, mipmapped
uniform sampler2D uMask;
uniform bool uHasMask;  // false until the segmenter has loaded: show the whole frame
uniform vec2 uCanvas;   // canvas px
uniform vec2 uSrc;      // camera px
uniform float uSc;      // canvas px per camera px
uniform float uOx;      // canvas x of the camera frame's left edge
uniform int uFace;      // 0 effect everywhere, 1 keep face clear, 2 face only
uniform vec2 uFaceRows; // person top, collar cut (0..1, y down)

uniform int uBg;        // 0 plasma, 1 grid, 2 dark, 3 filmed, 4 filmed FX
uniform vec3 uHead;     // head centre x, y and width, canvas px (for the plasma orb)
uniform float uK;       // canvas scale vs the 1080-line layouts stylize.py renders
uniform float uTime;

${NOISE}// _fbm: (4 << o) cells across a 2x canvas per octave, each octave scrolling faster
float fbm(vec2 P, float t) {
  float acc = 0.0, amp = 1.0, norm = 0.0;
  for (int o = 0; o < 4; o++) {
    float fo = float(o);
    vec2 scroll = vec2(9.0, 5.0) * t * (fo + 1.0) * 2.0 * uK;
    acc += amp * vnoise((P + scroll) / (2.0 * uCanvas) * float(4 << o) + fo * 17.0);
    norm += amp; amp *= 0.5;
  }
  return acc / norm;
}

vec3 bgPlasma(vec2 P, float t) {
  vec3 blue = vec3(0x1a, 0x4d, 0xff) / 255.0;
  vec3 col = vec3(4.0, 5.0, 14.0) / 255.0;
  float r = max(uHead.z * 1.6, 160.0 * uK);
  float ang = t * 0.25;                                     // slow rotation
  vec2 dv = (P - uHead.xy) / r;
  float rx = dv.x * cos(ang) - dv.y * sin(ang);
  float ry = dv.x * sin(ang) + dv.y * cos(ang);
  float d = length(vec2(rx, ry));
  float n = fbm(P, t);
  col += blue * exp(-d * d * 1.7) * (0.55 + 0.45 * n) * 1.5; // electric-blue orb
  float th = atan(ry, rx);
  float warp = n * 7.0 + sin(th * 3.0 + t * 0.7) * 1.3;     // noise + angular wobble
  float fil = abs(sin((d * 4.2 - t * 1.1) * PI + warp));
  fil = pow(clamp(1.0 - fil, 0.0, 1.0), 7.0) * exp(-d * d * 1.1);
  col += blue * fil * 2.2;
  col += vec3(160, 240, 255) / 255.0 * pow(fil, 3.0) * 1.8;  // cyan filament cores
  float rim = exp(-pow((d - 1.05) * 3.2, 2.0)) * clamp(ry, 0.0, 1.0);
  col += vec3(0xff, 0x2a, 0xc0) / 255.0 * rim * 0.42;       // magenta underside
  return col;
}

// 1 px anti-aliased line coverage at distance dist (px)
float line(float dist) { return clamp(1.0 - abs(dist) / max(uK, 0.75), 0.0, 1.0); }

vec3 bgGrid(vec2 P, float t) {
  vec2 C = uCanvas;
  vec3 cy = vec3(0x19, 0xd4, 0xc8) / 255.0;
  float horizon = floor(C.y * 0.46), vx = C.x * 0.5;
  float lay = 0.0;
  // floor + ceiling lines receding toward the horizon, scrolling at the viewer
  for (int i = 1; i < 22; i++) {
    float u = (float(i) + fract(t * 0.9)) / 22.0;
    float off = C.y * 0.60 * pow(u, 2.4);
    float w = pow(max(0.05, 1.0 - u), 1.6);
    lay = max(lay, w * line(P.y - (horizon + off)));
    lay = max(lay, w * line(P.y - (horizon - off)));
  }
  // one-point perspective verticals: line i passes x = vx + i*C.x*(0.03 + 0.39*s)
  float below = step(horizon, P.y);
  float s = below > 0.5 ? (P.y - horizon) / (C.y - horizon) : (horizon - P.y) / horizon;
  float spacing = C.x * (0.03 + 0.39 * s);
  float fi = (P.x - vx) / spacing;
  float ri = round(fi);
  if (abs(ri) <= 16.0) {
    // horizontal distance -> perpendicular, or the slanted lines render thin
    float slope = ri * C.x * 0.39 / (below > 0.5 ? C.y - horizon : horizon);
    float dist = abs(fi - ri) * spacing / sqrt(1.0 + slope * slope);
    lay = max(lay, (below > 0.5 ? 0.42 : 0.30) * line(dist));
  }
  float fade = clamp(abs(P.y - horizon) / (C.y * 0.55), 0.0, 1.0);
  vec3 col = vec3(0x04, 0x26, 0x2e) / 255.0 + cy * lay * (0.25 + 0.75 * fade) * 1.35;

  // drifting bokeh, nearer = bigger, brighter and faster
  vec3 bc[3] = vec3[3](vec3(0x19, 0xd4, 0xc8) / 255.0, vec3(0xff, 0xd8, 0x4d) / 255.0,
                       vec3(0xff, 0x6f, 0xb5) / 255.0);
  for (int i = 0; i < 30; i++) {
    float fi2 = float(i);
    float z = fract(hash12(vec2(fi2, 1.3)) + t * 0.05);
    vec2 c = vec2(fract(hash12(vec2(fi2, 7.1)) + t * 0.012 * (0.4 + z)) * C.x,
                  fract(hash12(vec2(fi2, 3.7)) + t * 0.020 * (0.4 + z)) * C.y);
    float rad = (3.0 + 26.0 * z) * uK, soft = 8.0 * uK;
    float cover = 1.0 - smoothstep(rad - soft, rad + soft, length(P - c));
    col += bc[i % 3] * (0.10 + 0.30 * z) * cover * 2.6 * (rad / (rad + soft));
  }
  return col;
}

vec3 bgDark(vec2 P) {
  vec2 q = (P - uCanvas * 0.5) / (uCanvas * 0.5);
  float v = clamp(1.0 - 0.72 * pow(length(q), 1.7), 0.0, 1.0);
  return vec3(5.0, 7.0, 15.0) / 255.0 * (0.35 + 1.9 * v);
}

// compose(src_bg=...): a blurred, dimmed cover-fill of the frame, with the sharp
// frame placed under the subject with the SAME transform, so the wall lines up.
// fx = true ("Filmed FX") uses the stylised frame instead of the real one.
// a single high mip level magnifies into visible blocks; nine taps spaced one
// texel of that level apart turn it into a smooth wide blur (GaussianBlur sigma 18)
vec3 wideBlur(sampler2D tex, vec2 uv, bool cam) {
  const float LOD = 4.0;
  vec2 step = exp2(LOD) / vec2(textureSize(tex, 0));
  vec3 acc = vec3(0.0);
  for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
      vec2 q = uv + vec2(i, j) * step;
      acc += textureLod(tex, cam ? camUV(q) : q, LOD).rgb;
    }
  return acc / 9.0;
}

vec3 bgFilmed(vec2 P, vec2 uv, bool inside, bool fx) {
  if (inside) return fx ? texture(uFx, uv).rgb : texture(uVideo, camUV(uv)).rgb;
  float cs = max(uCanvas.x / uSrc.x, uCanvas.y / uSrc.y);
  vec2 cuv = (P - (uCanvas - uSrc * cs) * 0.5) / cs / uSrc;
  return (fx ? wideBlur(uFx, cuv, false) : wideBlur(uVideo, cuv, true)) * 0.55;
}

vec3 background(vec2 P, vec2 uv, bool inside) {
  if (uBg == 0) return bgPlasma(P, uTime);
  if (uBg == 1) return bgGrid(P, uTime);
  if (uBg == 3 || uBg == 4) return bgFilmed(P, uv, inside, uBg == 4);
  return bgDark(P);
}

void main() {
  vec2 P = gl_FragCoord.xy;                       // y-down canvas px
  vec2 s = vec2((P.x - uOx) / uSc, P.y / uSc);    // camera px
  bool inside = s.x >= 0.0 && s.x < uSrc.x;
  vec2 uv = s / uSrc;
  vec4 styled = texture(uFx, uv);
  vec3 layer = styled.rgb;
  // < 1 where an effect is see-through (hologram), 1 for the rest. A filmed background
  // has no clean plate: behind the subject is the subject, so it stays opaque there.
  float see = uBg == 3 || uBg == 4 ? 1.0 : styled.a;
  if (uFace != 0 && uHasMask) {
    // face_protect: head + neck down to the collar, times the person mask, softened
    // (its 61 px Gaussian is ~2.5% of the frame height)
    const float SOFT = 0.025;
    float fm = (1.0 - smoothstep(uFaceRows.y - SOFT, uFaceRows.y + SOFT, uv.y))
             * smoothstep(uFaceRows.x - SOFT, uFaceRows.x, uv.y)
             * textureLod(uMask, camUV(uv), 1.0).r;
    if (uFace == 2) fm = 1.0 - fm;           // effect on the head, the rest stays real
    layer = mix(layer, texture(uVideo, camUV(uv)).rgb, fm);
    see = mix(see, 1.0, fm);
  }
  // the mask is 256 px on its long side: bilinear upsampling + a narrow smoothstep
  // gives a clean edge without the halo a raw soft mask leaves at this scale
  float m = uHasMask ? smoothstep(0.25, 0.75, texture(uMask, camUV(uv)).r) : 1.0;
  float a = inside ? m * see : 0.0;
  // a see-through effect shows the background only faintly behind the person: plasma
  // centres its brightest rings on the head, and at full strength they scribble over
  // the face (the other effects have see == 1 and are unchanged)
  vec3 behind = background(P, uv, inside);
  if (see < 0.999) behind *= 1.0 - 0.65 * (inside ? m : 0.0);
  outColor = vec4(mix(behind, layer, a), 1.0);
}`;

// composite target -> canvas: the one vertical flip, then compose()'s finishing:
// chromatic aberration and film grain
export const OUTPUT = COMMON + `
uniform sampler2D uComp;
uniform vec2 uCanvas;
uniform bool uChroma;
uniform float uGrain;   // sigma in 0..255 units, like --grain
uniform float uK;       // canvas scale vs 1080 lines
uniform float uSeed;    // per frame, so the grain moves

float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec2 P = vec2(gl_FragCoord.x, uCanvas.y - gl_FragCoord.y);
  vec2 uv = P / uCanvas;
  vec3 c = texture(uComp, uv).rgb;
  if (uChroma) {
    // chromatic(): red sampled 1.6 px to the left, blue 1.6 px to the right
    vec2 off = vec2(1.6 * uK / uCanvas.x, 0.0);
    c.r = texture(uComp, uv - off).r;
    c.b = texture(uComp, uv + off).b;
  }
  if (uGrain > 0.0) {
    // grain(): one Gaussian sample per pixel, the same on all channels (Box-Muller)
    float u1 = max(hash(P + uSeed), 1e-6), u2 = hash(P.yx + uSeed * 1.7);
    c += sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2) * uGrain / 255.0;
  }
  outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;
