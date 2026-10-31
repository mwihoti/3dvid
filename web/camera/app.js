// 3dvid live camera: open the camera, style it in the browser, record, save on the device.
import { Renderer } from './render.js';
import { Segmenter } from './segment.js';

const $ = s => document.querySelector(s);

// ---- settings (mirrors the studio's stylize.py options where they exist)
export const settings = {
  effect: 'voxel', bg: 'plasma', face: 'off', layout: 'wide', quality: '720',
  bands: 12, minCell: 8, maxCell: 24, grain: 0, mirror: true, chroma: false, keepRaw: false,
};

document.querySelectorAll('.seg').forEach(seg => {
  seg.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...seg.children].forEach(c => c.setAttribute('aria-pressed', c === b));
    settings[seg.id] = b.dataset.v;
    onSettings(seg.id);
  });
});
document.querySelectorAll('.range input').forEach(inp => {
  inp.addEventListener('input', () => {
    inp.nextElementSibling.value = inp.value;
    settings[inp.id] = +inp.value;
    onSettings(inp.id);
  });
});
['mirror', 'chroma', 'keepRaw'].forEach(id => $('#' + id).addEventListener('change', e => {
  settings[id] = e.target.checked; onSettings(id);
}));
$('#setBtn').onclick = () => $('#panel').classList.toggle('open');
document.addEventListener('pointerdown', e => {          // tap outside closes the mobile sheet
  if (!e.target.closest('#panel') && !e.target.closest('#setBtn')) $('#panel').classList.remove('open');
});

const listeners = [];
export const onSettingsChange = fn => listeners.push(fn);
function onSettings(key) { listeners.forEach(fn => fn(key)); }

export function setStatus(text) { $('#status').textContent = text; }

// ---- camera
const cam = $('#cam');
let stream = null, facing = 'user';

function showOverlay(html, button = true) {
  $('#overlayMsg').innerHTML = html;
  $('#startBtn').classList.toggle('hide', !button);
  $('#overlay').classList.remove('hide');
}

export async function startCamera() {
  if (!window.isSecureContext) {
    showOverlay('<b>Camera needs HTTPS</b><br>Open this page over https:// (or on localhost). ' +
                'Browsers block cameras on plain http.', false);
    throw new Error('insecure context');
  }
  stream?.getTracks().forEach(t => t.stop());
  const video = { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video, audio: { echoCancellation: true, noiseSuppression: true } });
  } catch (err) {
    if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
      showOverlay('<b>Camera access was blocked</b><br>Allow it in the address bar, then try again.');
      throw err;
    }
    // no microphone (or it is busy): carry on silently rather than refuse to film
    stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  }
  cam.srcObject = stream;
  await cam.play();
  $('#overlay').classList.add('hide');
  // only a user-facing camera is mirrored by default; the rear one would read backwards
  const settingsFacing = stream.getVideoTracks()[0].getSettings().facingMode;
  if (settingsFacing) {
    settings.mirror = settingsFacing === 'user';
    $('#mirror').checked = settings.mirror;
  }
  setStatus(`${cam.videoWidth}×${cam.videoHeight}`);
  return stream;
}

export const camera = () => ({ video: cam, stream });

$('#flipBtn').onclick = async () => {
  facing = facing === 'user' ? 'environment' : 'user';
  try { await startCamera(); if (renderer) startLoop(); } catch (e) { setStatus(e.message); }
};
$('#startBtn').onclick = () => startCamera().then(() => {
  if (!renderer) return;
  cam.classList.add('offscreen'); $('#out').classList.remove('hide'); startLoop();
}).catch(e => setStatus(e.message));

// ---- studio (only when this page is served by serve.py, not as a static site)
export const studio = { available: false };
async function detectStudio() {
  try {
    const r = await fetch(new URL('../api/health', location.href));
    if (!r.ok) return;
    const h = await r.json();
    if (!('auth' in h)) return;
    studio.available = true;
    $('#studioLink').classList.remove('hide');
    $('#studioOpts').classList.remove('hide');
  } catch { /* static hosting: no studio, the page still works on its own */ }
}

// ---- render loop: one styled frame per camera frame
let renderer = null, seg = null, loopId = 0;
const fps = { n: 0, since: performance.now(), value: 0 };
const segCost = { every: 1, n: 0, ms: 0 };

function segment() {
  if (!seg || !cam.videoWidth) return;
  // a slow device segments every other frame instead of dropping the whole frame rate
  if (segCost.n++ % segCost.every) return;
  const t0 = performance.now();
  seg.update(cam, t0);
  segCost.ms = segCost.ms * 0.9 + (performance.now() - t0) * 0.1;
  segCost.every = segCost.ms > 22 ? 2 : 1;
  $('#hint').classList.toggle('hide', seg.stats.present);
}

function tick(id) {
  if (id !== loopId) return;                       // a newer loop replaced this one
  segment();
  if (renderer.frame({ video: cam, settings, seg })) {
    fps.n++;
    const now = performance.now();
    if (now - fps.since >= 1000) {
      fps.value = Math.round(fps.n * 1000 / (now - fps.since));
      fps.n = 0; fps.since = now;
      setStatus(`${fps.value} fps · ${$('#out').width}×${$('#out').height}`);
    }
  }
  schedule(id);
}
function schedule(id) {
  if (cam.requestVideoFrameCallback) cam.requestVideoFrameCallback(() => tick(id));
  else requestAnimationFrame(() => tick(id));
}
function startLoop() { schedule(++loopId); }

async function main() {
  detectStudio();
  try {
    renderer = new Renderer($('#out'));
  } catch (e) {
    showOverlay(`<b>Live styles are not supported here</b><br>${e.message}`, false);
    return;
  }
  try {
    await startCamera();
  } catch (e) {
    if ($('#overlay').classList.contains('hide')) showOverlay(`<b>Could not open the camera</b><br>${e.message}`);
    return;
  }
  cam.classList.add('offscreen');
  $('#out').classList.remove('hide');
  startLoop();
  setStatus('loading person model…');
  try {
    seg = await Segmenter.create();
  } catch (e) {
    // without a mask the effects still run on the whole frame
    console.error('segmenter failed to load', e);
    setStatus('person model unavailable - full-frame styles');
  }
}
main();
