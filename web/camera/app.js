// 3dvid live camera: open the camera, style it in the browser, record, save on the device.
import { Renderer } from './render.js';
import { Segmenter } from './segment.js';
import { buildPalette } from './palette.js';
import { Recorder, saveToDevice, extFor, stamp } from './recorder.js';

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
  try { await startCamera(); palette = null; if (renderer) startLoop(); } catch (e) { setStatus(e.message); }
};
$('#startBtn').onclick = () => startCamera().then(() => {
  if (!renderer) return;
  cam.classList.add('offscreen'); $('#out').classList.remove('hide'); startLoop();
}).catch(e => setStatus(e.message));

// ---- studio (only when this page is served by serve.py, not as a static site)
export const studio = { available: false };

// same bearer-token handling as the studio page (shared localStorage key)
let TOKEN = localStorage.getItem('token') || '';
async function api(path, opts = {}) {
  const url = new URL(path, location.href);
  opts.headers = { ...(opts.headers || {}), ...(TOKEN ? { Authorization: 'Bearer ' + TOKEN } : {}) };
  let r = await fetch(url, opts);
  if (r.status === 401) {
    const t = prompt('The studio needs its access token:');
    if (t === null) throw new Error('no token');
    TOKEN = t.trim(); localStorage.setItem('token', TOKEN);
    opts.headers.Authorization = 'Bearer ' + TOKEN;
    r = await fetch(url, opts);
  }
  return r;
}
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
// palette: built from the first frame at once, rebuilt once a person is found so it
// is weighted toward them (build_palette does this on the first frame of a clip)
let palette = null, paletteHasPerson = false;
function refreshPalette() {
  const person = !!seg?.stats?.present;
  palette = buildPalette(cam, person ? seg : null);
  paletteHasPerson = person;
}
$('#paletteBtn').onclick = () => { if (cam.videoWidth) refreshPalette(); };
const t0 = performance.now();
const fps = { n: 0, since: performance.now(), value: 0 };
const segCost = { every: 1, n: 0, ms: 0 };

function segment() {
  if (!seg || !cam.videoWidth) return;
  // a slow device segments every other frame instead of dropping the whole frame rate
  if (segCost.n++ % segCost.every) return;
  const start = performance.now();
  seg.update(cam, start);
  segCost.ms = segCost.ms * 0.9 + (performance.now() - start) * 0.1;
  segCost.every = segCost.ms > 22 ? 2 : 1;
  $('#hint').classList.toggle('hide', seg.stats.present);
}

function tick(id) {
  if (id !== loopId) return;                       // a newer loop replaced this one
  segment();
  if (cam.videoWidth && (!palette || (!paletteHasPerson && seg?.stats?.present))) refreshPalette();
  if (palette && renderer.frame({ video: cam, settings, seg, palette, t: (performance.now() - t0) / 1000 })) {
    if (snapPending) takeSnapshot();
    if ($('#recBtn').disabled && !recording) { $('#recBtn').disabled = false; $('#snapBtn').disabled = false; }
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

// ---- photo: read the canvas straight after a draw, while the drawing buffer is
// still intact (it is not preserved between frames)
let snapPending = false;
$('#snapBtn').onclick = () => { snapPending = true; };
function takeSnapshot() {
  snapPending = false;
  $('#out').toBlob(blob => blob && showResult({ blob, name: `3dvid-${settings.effect}-${stamp()}.png`, still: true }),
                   'image/png');
}

// ---- recording
const MAX_SECONDS = 600;
let recording = null;      // { styled, raw, started, timer, lock }

async function startRecording() {
  const out = $('#out');
  if (!out.captureStream) return alert('This browser cannot record a canvas.');
  const mic = stream.getAudioTracks();
  const styled = new Recorder(new MediaStream([...out.captureStream(30).getVideoTracks(), ...mic]),
                              settings.quality === '1080' ? 12e6 : 6e6);
  // the untouched camera, for an exact re-render in the studio with the real depth model
  const raw = settings.keepRaw && studio.available ? new Recorder(stream, 8e6) : null;
  styled.start(); raw?.start();
  recording = { styled, raw, started: performance.now() };
  recording.lock = await navigator.wakeLock?.request('screen').catch(() => null);   // keep the screen on
  recording.timer = setInterval(() => {
    const sec = (performance.now() - recording.started) / 1000;
    $('#recTime').textContent = `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, '0')}`;
    if (sec >= MAX_SECONDS) stopRecording();
  }, 250);
  $('#recTime').textContent = '0:00';
  $('#recBadge').classList.remove('hide');
  $('#recBtn').classList.add('on');
  // a new canvas size or camera mid-take would corrupt the file
  document.querySelectorAll('#layout button, #quality button').forEach(b => { b.disabled = true; });
  $('#flipBtn').disabled = true;
}

async function stopRecording() {
  const r = recording;
  if (!r) return;
  recording = null;
  clearInterval(r.timer);
  r.lock?.release().catch(() => {});
  $('#recBadge').classList.add('hide');
  $('#recBtn').classList.remove('on');
  document.querySelectorAll('#layout button, #quality button').forEach(b => { b.disabled = false; });
  $('#flipBtn').disabled = false;
  const seconds = (performance.now() - r.started) / 1000;
  const [blob, rawBlob] = await Promise.all([r.styled.stop(), r.raw?.stop()]);
  const ext = extFor(blob.type);
  showResult({ blob, rawBlob, seconds, name: `3dvid-${settings.effect}-${settings.bg}-${stamp()}.${ext}` });
}

$('#recBtn').onclick = () => (recording ? stopRecording() : startRecording().catch(e => {
  recording = null; alert('Could not start recording: ' + e.message);
}));

// ---- result sheet
let result = null;
function showResult(res) {
  result = { ...res, url: URL.createObjectURL(res.blob) };
  const take = $('#take'), still = $('#still');
  take.classList.toggle('hide', !!res.still);
  still.classList.toggle('hide', !res.still);
  if (res.still) still.src = result.url; else { take.src = result.url; take.play().catch(() => {}); }
  const mb = (res.blob.size / 1e6).toFixed(1);
  $('#takeMeta').textContent = res.still ? `${$('#out').width}×${$('#out').height} PNG · ${mb} MB`
    : `${res.seconds.toFixed(1)} s · ${$('#out').width}×${$('#out').height} · ${extFor(res.blob.type).toUpperCase()} · ${mb} MB`;
  $('#dlLink').href = result.url;
  $('#dlLink').download = res.name;
  // on a desktop "Save" already downloads; the plain link is the phone's alternative to sharing
  $('#dlLink').classList.toggle('hide', !matchMedia('(pointer: coarse)').matches);
  $('#hqStatus').textContent = res.still || res.rawBlob || !studio.available ? ''
    : 'Tip: turn on "Also keep the raw take" to render the exact version in the studio.';
  $('#hqBtn').classList.toggle('hide', !res.rawBlob);
  $('#hqBtn').disabled = false;
  $('#sheet').classList.remove('hide');
}

// upload the untouched camera take and start the same look in the studio, where
// the real depth model and rembg replace the live approximations
$('#hqBtn').onclick = async () => {
  const btn = $('#hqBtn'), status = $('#hqStatus');
  btn.disabled = true;
  try {
    status.textContent = 'Uploading the raw take…';
    const fd = new FormData();
    fd.append('file', new File([result.rawBlob], `camera-${stamp()}.${extFor(result.rawBlob.type)}`));
    fd.append('source', 'camera');
    let r = await api('../api/upload', { method: 'POST', body: fd });
    const up = await r.json();
    if (!r.ok) throw new Error(up.detail || 'upload failed');

    const job = new FormData();
    Object.entries({
      video: up.name, kind: 'stylize', effect: settings.effect, bg: settings.bg, layout: settings.layout,
      face_mode: settings.face, bands: settings.bands, min_cell: settings.minCell, max_cell: settings.maxCell,
      preview: 0, keep_audio: true, chromatic: settings.chroma, grain: settings.grain,
    }).forEach(([k, v]) => job.append(k, v));
    r = await api('../api/render', { method: 'POST', body: job });
    if (r.status === 409) {
      status.innerHTML = `Uploaded as <b>${up.name}</b>. The studio is busy with another render; start this one from the <a href="../">studio</a>.`;
      return;
    }
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || 'render failed to start');
    status.innerHTML = `Rendering <b>${up.name}</b> in the studio - follow it on the <a href="../">studio page</a>.`;
  } catch (e) {
    status.textContent = 'Studio render failed: ' + e.message;
    btn.disabled = false;
  }
};
$('#saveBtn').onclick = async () => {
  const how = await saveToDevice(result.blob, result.name);
  $('#hqStatus').textContent = how === 'shared' ? 'Saved.' : how === 'downloaded' ? `Downloaded ${result.name}` : '';
};
$('#backBtn').onclick = () => {
  $('#take').pause();
  URL.revokeObjectURL(result.url);
  result = null;
  $('#sheet').classList.add('hide');
};

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
