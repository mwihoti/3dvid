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
uniform vec2 uCanvas;   // canvas px
uniform vec2 uSrc;      // camera px
uniform float uSc;      // canvas px per camera px
uniform float uOx;      // canvas x of the camera frame's left edge

vec3 background(vec2 P) {
  return vec3(5.0, 7.0, 15.0) / 255.0;
}

void main() {
  vec2 P = gl_FragCoord.xy;                       // y-down canvas px
  vec2 s = vec2((P.x - uOx) / uSc, P.y / uSc);    // camera px
  bool inside = s.x >= 0.0 && s.x < uSrc.x;
  vec2 uv = s / uSrc;
  vec3 layer = texture(uVideo, camUV(uv)).rgb;
  float a = inside ? 1.0 : 0.0;
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
