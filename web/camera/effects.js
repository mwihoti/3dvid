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

export const BACKGROUNDS = [
  { id: 'plasma', label: 'Plasma' },
  { id: 'grid', label: 'Grid' },
  { id: 'dark', label: 'Dark' },
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
