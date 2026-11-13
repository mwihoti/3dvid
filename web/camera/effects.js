// The live camera's choices, in one place: the page, meeting-mode keys and any other
// UI (a browser-extension popup) build their controls from these lists. No DOM and
// no imports, so it loads anywhere a module does.
//
// EFFECTS order matters: an effect's index is the FX shader's uEffect value.
// studio: true = stylize.py renders this effect too, so a raw take can be re-rendered
// exactly in the studio; the others exist only live.
export const EFFECTS = [
  { id: 'voxel', label: 'Voxel', studio: true },
  { id: 'contour', label: 'Neon', studio: true },
  { id: 'both', label: 'Both', studio: true },
  { id: 'clean', label: 'Clean', studio: true },
  { id: 'ascii', label: 'ASCII', studio: false },
  { id: 'halftone', label: 'Halftone', studio: false },
  { id: 'comic', label: 'Comic', studio: false },
  { id: 'thermal', label: 'Thermal', studio: false },
  { id: 'hologram', label: 'Hologram', studio: false },
  { id: 'pixel', label: '8-bit', studio: false },
  { id: 'sketch', label: 'Sketch', studio: false },
];

// BACKGROUNDS order matters too: an index is the COMPOSITE shader's uBg value (the
// shader gets one BG_<ID> constant per entry). Every background is in stylize.py as
// well (its BACKGROUNDS dict, plus filmed / source), so all of them work in the studio.
export const BACKGROUNDS = [
  { id: 'plasma', label: 'Plasma' },
  { id: 'grid', label: 'Grid' },
  { id: 'dark', label: 'Dark' },
  { id: 'studio', label: 'Studio' },
  { id: 'bokeh', label: 'Bokeh' },
  { id: 'aurora', label: 'Aurora' },
  { id: 'nebula', label: 'Nebula' },
  { id: 'warp', label: 'Warp' },
  { id: 'sunset', label: 'Sunset' },
  { id: 'rain', label: 'Code rain' },
  { id: 'ocean', label: 'Ocean' },
  { id: 'lava', label: 'Lava lamp' },
  { id: 'circuit', label: 'Circuit' },
  { id: 'filmed', label: 'Filmed' },
  { id: 'source', label: 'Filmed FX' },
];

export const FACE_MODES = [
  { id: 'off', label: 'Effect everywhere' },
  { id: 'clear', label: 'Keep face clear' },
  { id: 'only', label: 'Face only' },
];

export const effectIndex = id => EFFECTS.findIndex(e => e.id === id);
export const isStudioEffect = id => !!EFFECTS.find(e => e.id === id)?.studio;
export const bgIndex = id => BACKGROUNDS.findIndex(b => b.id === id);
