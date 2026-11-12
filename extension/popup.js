// Toolbar popup: every control writes chrome.storage.local, which the pages' engines
// watch, so a running call restyles immediately. Controls are built from the same
// lists as the camera page (lib/effects.js).
import { EFFECTS, BACKGROUNDS, FACE_MODES } from './lib/effects.js';
import { QUALITIES, loadSettings } from './settings.js';

const $ = s => document.querySelector(s);
const save = patch => chrome.storage.local.set(patch);

function segmented(id, items, current, toValue = v => v) {
  const box = $('#' + id);
  for (const { id: v, label, title } of items) {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.v = v;
    b.textContent = label;
    b.title = title ?? label;
    b.setAttribute('aria-pressed', String(String(v) === String(current)));
    box.append(b);
  }
  box.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    box.querySelectorAll('button').forEach(c => c.setAttribute('aria-pressed', String(c === b)));
    save({ [id]: toValue(b.dataset.v) });
  });
}

const s = await loadSettings();
segmented('effect', EFFECTS, s.effect);
segmented('bg', BACKGROUNDS, s.bg);
// the page's labels are too long for three columns at this width
const SHORT = { off: 'Everywhere', clear: 'Face clear', only: 'Face only' };
segmented('face', FACE_MODES.map(f => ({ ...f, title: f.label, label: SHORT[f.id] ?? f.label })), s.face);
segmented('quality', QUALITIES.map(q => ({ id: q, label: `${q}p` })), s.quality, Number);

const enabled = $('#enabled');
enabled.checked = s.enabled;
document.body.classList.toggle('off', !s.enabled);
enabled.addEventListener('change', () => {
  document.body.classList.toggle('off', !enabled.checked);
  save({ enabled: enabled.checked });
});
for (const id of ['mirror', 'chroma']) {
  $('#' + id).checked = s[id];
  $('#' + id).addEventListener('change', e => save({ [id]: e.target.checked }));
}
$('#grain').checked = s.grain > 0;
$('#grain').addEventListener('change', e => save({ grain: e.target.checked ? 6 : 0 }));

// the shortcut may have been changed (or lost to a clash) in chrome://extensions/shortcuts
chrome.commands?.getAll?.().then(cmds => {
  const key = cmds.find(c => c.name === 'cycle-effect')?.shortcut;
  const b = $('#note b');
  if (!key) b.parentNode.replaceChild(document.createTextNode('A shortcut (chrome://extensions/shortcuts)'), b);
  else b.textContent = key;
});
