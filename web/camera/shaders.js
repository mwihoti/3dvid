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

// canvas-sized: places the camera frame like compose() - scaled to the canvas
// height, centred at uOx - and fills everything else with the background.
export const COMPOSITE = COMMON + `
uniform sampler2D uVideo;
uniform sampler2D uMask;
uniform bool uHasMask;  // false until the segmenter has loaded: show the whole frame
uniform vec2 uCanvas;   // canvas px
uniform vec2 uSrc;      // camera px
uniform float uSc;      // canvas px per camera px
uniform float uOx;      // canvas x of the camera frame's left edge

uniform int uBg;        // 0 plasma, 1 grid, 2 dark (3/4 filmed come with the filmed backgrounds)
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

vec3 background(vec2 P) {
  if (uBg == 0) return bgPlasma(P, uTime);
  if (uBg == 1) return bgGrid(P, uTime);
  return bgDark(P);
}

void main() {
  vec2 P = gl_FragCoord.xy;                       // y-down canvas px
  vec2 s = vec2((P.x - uOx) / uSc, P.y / uSc);    // camera px
  bool inside = s.x >= 0.0 && s.x < uSrc.x;
  vec2 uv = s / uSrc;
  vec3 layer = texture(uVideo, camUV(uv)).rgb;
  // the mask is 256 px on its long side: bilinear upsampling + a narrow smoothstep
  // gives a clean edge without the halo a raw soft mask leaves at this scale
  float m = uHasMask ? smoothstep(0.25, 0.75, texture(uMask, camUV(uv)).r) : 1.0;
  float a = inside ? m : 0.0;
  outColor = vec4(mix(background(P), layer, a), 1.0);
}`;

// composite target -> canvas: the one vertical flip, plus film-style finishing later
export const OUTPUT = COMMON + `
uniform sampler2D uComp;
uniform vec2 uCanvas;
void main() {
  vec2 P = vec2(gl_FragCoord.x, uCanvas.y - gl_FragCoord.y);
  outColor = vec4(texture(uComp, P / uCanvas).rgb, 1.0);
}`;
