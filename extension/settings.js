// Settings shared by the popup, the service worker and the camera engine. They live in
// chrome.storage.local under these keys; a missing or unknown value reads as the default.
import { EFFECTS, BACKGROUNDS, FACE_MODES } from './lib/effects.js';

export const DEFAULTS = {
  enabled: true,
  effect: 'voxel', bg: 'plasma', face: 'off',
  // meeting apps mirror the self-view themselves; mirroring here would flip what others see
  mirror: false,
  chroma: false,
  grain: 0,           // the camera page's film grain amount; the popup toggles 0 / 6
  quality: 720,       // the camera is scaled to this many rows (short side) before styling
};
export const QUALITIES = [480, 720, 1080];

const ok = (list, v) => list.some(e => e.id === v);

export function sanitize(s) {
  const out = { ...DEFAULTS, ...s };
  if (!ok(EFFECTS, out.effect)) out.effect = DEFAULTS.effect;
  if (!ok(BACKGROUNDS, out.bg)) out.bg = DEFAULTS.bg;
  if (!ok(FACE_MODES, out.face)) out.face = DEFAULTS.face;
  if (!QUALITIES.includes(+out.quality)) out.quality = DEFAULTS.quality;
  out.quality = +out.quality;
  out.grain = Math.max(0, Math.min(20, +out.grain || 0));
  out.enabled = out.enabled !== false;
  out.mirror = !!out.mirror; out.chroma = !!out.chroma;
  return out;
}

export async function loadSettings() {
  return sanitize(await chrome.storage.local.get(Object.keys(DEFAULTS)));
}
