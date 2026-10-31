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

// camera-sized: the stylised frame, in display (mirrored) space. Effects render the
// whole frame like stylize.py's do; the composite decides where it shows.
export const FX = COMMON + `
uniform sampler2D uVideo;
uniform sampler2D uMask;
uniform bool uHasMask;
uniform vec2 uSrc;      // fx target px
uniform int uEffect;    // 0 voxel, 1 contour, 2 both, 3 clean
uniform vec3 uPerson;   // top, face bottom (0..1, y down), present
uniform vec3 uPal[24];  // build_palette's 24 colours
uniform float uMinCell, uMaxCell, uTopCell, uCamPerFx;

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

void main() {
  vec2 pix = gl_FragCoord.xy;
  vec3 col = uEffect == 0 ? voxel(pix) : texture(uVideo, camUV(pix / uSrc)).rgb;
  outColor = vec4(col, 1.0);
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

uniform int uBg;        // 0 plasma, 1 grid, 2 dark, 3 filmed, 4 filmed FX
uniform vec3 uHead;     // head centre x, y and width, canvas px (for the plasma orb)
uniform float uK;       // canvas scale vs the 1080-line layouts stylize.py renders
uniform float uTime;

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
// _fbm: (4 << o) cells across a 2x canvas per octave, each octave scrolling faster
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
  vec3 layer = texture(uFx, uv).rgb;
  // the mask is 256 px on its long side: bilinear upsampling + a narrow smoothstep
  // gives a clean edge without the halo a raw soft mask leaves at this scale
  float m = uHasMask ? smoothstep(0.25, 0.75, texture(uMask, camUV(uv)).r) : 1.0;
  float a = inside ? m : 0.0;
  outColor = vec4(mix(background(P, uv, inside), layer, a), 1.0);
}`;

// composite target -> canvas: the one vertical flip, plus film-style finishing later
export const OUTPUT = COMMON + `
uniform sampler2D uComp;
uniform vec2 uCanvas;
void main() {
  vec2 P = vec2(gl_FragCoord.x, uCanvas.y - gl_FragCoord.y);
  outColor = vec4(texture(uComp, P / uCanvas).rgb, 1.0);
}`;
