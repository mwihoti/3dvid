// 3dvid live camera: open the camera, style it in the browser, record, save on the device.
import { Renderer } from './render.js';
import { Segmenter } from './segment.js';
import { buildPalette } from './palette.js';
import { Recorder, saveToDevice, extFor, stamp } from './recorder.js';
import { EFFECTS, BACKGROUNDS, effectIndex, isStudioEffect, bgIndex } from './effects.js';

const $ = s => document.querySelector(s);

// ---- settings (mirrors the studio's stylize.py options where they exist)
const DEFAULTS = {
  effect: 'voxel', bg: 'plasma', face: 'off', layout: 'wide', quality: '720', subjectX: '',
  bands: 12, minCell: 8, maxCell: 24, grain: 0, mirror: true, chroma: false, keepRaw: false,
  studio: '',      // a studio elsewhere (tunnel URL); '' = the server this page came from, if any
};
const SEGS = ['effect', 'bg', 'face', 'layout', 'quality', 'subjectX'];
const RANGES = ['bands', 'minCell', 'maxCell', 'grain'];
const CHECKS = ['mirror', 'chroma', 'keepRaw'];
const params = new URLSearchParams(location.search);
// Meeting mode (?meeting): just the styled picture, nothing else on screen, for a
// virtual camera (OBS window capture) to feed Zoom / Meet / Teams. Mirror is off
// because meeting apps mirror your self-view themselves and send others the
// unmirrored picture; "native" keeps the webcam's own size and aspect.
export const MEETING = params.has('meeting');
export const settings = { ...DEFAULTS, ...(MEETING ? { mirror: false, layout: 'native' } : {}) };

// remembered choices, then anything given in the URL (?effect=contour&bg=grid&face=clear)
try { Object.assign(settings, JSON.parse(localStorage.getItem('camSettings') || '{}')); } catch { /* ignore */ }
if (MEETING) { settings.mirror = false; settings.keepRaw = false; }
for (const [k, v] of params) {
  if (!(k in DEFAULTS) || k === 'keepRaw') continue;
  settings[k] = RANGES.includes(k) ? +v : CHECKS.includes(k) ? v !== '0' && v !== 'false' : v;
}
// an effect saved by (or linked from) another version of the page may not exist here
if (effectIndex(settings.effect) < 0) settings.effect = DEFAULTS.effect;
if (bgIndex(settings.bg) < 0) settings.bg = DEFAULTS.bg;
// A token belongs to one studio. The server this page came from shares the studio
// page's 'token' key; a studio elsewhere gets a key of its own, so a link that names
// another studio is never sent a token that was given for this one.
const tokenKey = () => (settings.studio && settings.studio !== location.origin ? 'token:' + settings.studio : 'token');
// What people really paste as a studio URL: no scheme, a trailing slash, a path, or one
// of the printed links with ?token= in it. Reduce it to the origin (every API call is
// resolved against that) and hand back the token it carried. null = not a URL.
function parseStudio(v) {
  v = String(v || '').trim();
  if (!v) return { studio: '', token: '' };
  if (!/^https?:\/\//i.test(v)) v = 'https://' + v;
  try {
    const u = new URL(v);
    // new URL accepts almost anything after https://; a studio is a hostname or an IP
    if (!/^[a-z0-9.-]+$|^\[[0-9a-f:.]+\]$/i.test(u.hostname)) return null;
    return { studio: u.origin, token: (u.searchParams.get('token') || '').trim() };
  } catch { return null; }
}
let studioError = '';
function setStudio(v) {
  const p = parseStudio(v);
  studioError = p ? '' : `"${String(v).trim()}" is not a URL. Paste the studio's https:// address.`;
  settings.studio = p ? p.studio : '';
  if (p && p.token) localStorage.setItem(tokenKey(), p.token);
}
setStudio(settings.studio);
// one link can carry the studio's token (the Kaggle notebook prints such a link);
// keep it in localStorage like a typed one and take it out of the address bar, so
// it is not left in history, bookmarks or anything the user copies
if (params.has('token')) {
  // a link that names no studio is the serving studio's own link: its token must not
  // be filed under (and later sent to) another studio remembered from an earlier visit
  localStorage.setItem(params.has('studio') ? tokenKey() : 'token', params.get('token').trim());
  params.delete('token');
  history.replaceState(null, '', location.pathname + (params.size ? '?' + params : '') + location.hash);
}
function saveSettings() {
  const { keepRaw, ...rest } = settings;          // keepRaw is a per-take choice, not a preference
  if (MEETING) { delete rest.mirror; delete rest.layout; }   // meeting-only overrides must not leak into the normal page
  localStorage.setItem('camSettings', JSON.stringify(rest));
}
// the effect buttons come from the shared list, so the page, the meeting keys and
// any other UI agree on what exists
$('#effect').replaceChildren(...EFFECTS.map(e => {
  const b = document.createElement('button');
  b.dataset.v = e.id; b.textContent = e.label; b.setAttribute('aria-pressed', 'false');
  if (!e.studio) b.title = 'Live only: the studio cannot render this effect';
  return b;
}));
$('#bg').replaceChildren(...BACKGROUNDS.map(b => {
  const btn = document.createElement('button');
  btn.dataset.v = b.id; btn.textContent = btn.title = b.label; btn.setAttribute('aria-pressed', 'false');
  return btn;
}));
const STUDIO_EFFECTS = EFFECTS.filter(e => e.studio).map(e => e.label);
const STUDIO_LIST = STUDIO_EFFECTS.slice(0, -1).join(', ') + ' and ' + STUDIO_EFFECTS.at(-1);
const effectLabel = id => EFFECTS[effectIndex(id)]?.label || id;
// the raw take only helps with an effect the studio has; say so next to the option
function syncLiveOnly() {
  const live = !isStudioEffect(settings.effect);
  $('#liveOnlyNote').textContent = live
    ? `The studio renders ${STUDIO_LIST}; ${effectLabel(settings.effect)} is live only.` : '';
  $('#liveOnlyNote').classList.toggle('hide', !live);
}
// push `settings` into the controls (after loading, and after a keyboard shortcut)
function syncUI() {
  for (const id of SEGS) {
    const seg = document.getElementById(id);
    [...seg.children].forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === String(settings[id]))));
  }
  for (const id of RANGES) {
    const inp = document.getElementById(id);
    inp.value = settings[id]; inp.nextElementSibling.value = settings[id];
  }
  for (const id of CHECKS) document.getElementById(id).checked = !!settings[id];
  $('#studioUrl').value = settings.studio;
  syncLiveOnly();
}
syncUI();
$('#studioUrl').addEventListener('change', e => {
  setStudio(e.target.value);
  if (!studioError) e.target.value = settings.studio;
  onSettings('studio');
  detectStudio();
});

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
function onSettings(key) { saveSettings(); listeners.forEach(fn => fn(key)); }
onSettingsChange(key => { if (key === 'effect') syncLiveOnly(); });

// ---- meeting mode: keyboard shortcuts, since the controls are hidden
// digits 1-9, 0 pick the first ten effects; n / b step through all of them, v / x
// through the backgrounds
const KEYS = {
  'q': ['bg', 'plasma'], 'w': ['bg', 'grid'], 'e': ['bg', 'dark'], 'r': ['bg', 'filmed'], 't': ['bg', 'source'],
};
const DIGITS = '1234567890';
const stepEffect = d => EFFECTS[(effectIndex(settings.effect) + d + EFFECTS.length) % EFFECTS.length].id;
const stepBg = d => BACKGROUNDS[(bgIndex(settings.bg) + d + BACKGROUNDS.length) % BACKGROUNDS.length].id;
const FACE_CYCLE = ['off', 'clear', 'only'];
if (MEETING) {
  document.body.classList.add('meeting');
  // key list for the effects, built from the same list (the rest is static in the page)
  const row = list => list.flatMap(([key, label]) => {
    const b = document.createElement('b'); b.textContent = key;
    return [b, label + ' '];
  });
  const digitKeys = EFFECTS.slice(0, DIGITS.length).map((e, i) => [DIGITS[i], e.label.toLowerCase()]);
  $('#effectKeys').replaceChildren(...row(digitKeys.slice(0, 5)), document.createElement('br'),
    ...row(digitKeys.slice(5)), document.createElement('br'), ...row([['n', 'next effect'], ['b', 'previous']]));
  document.addEventListener('keydown', ev => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    // typing in the panel (the studio URL) is text, not shortcuts
    if (ev.target instanceof Element && ev.target.closest('input, textarea, select, [contenteditable]')) return;
    const k = ev.key.toLowerCase();
    const digit = DIGITS.indexOf(k);
    if (k.length === 1 && digit >= 0 && digit < EFFECTS.length) settings.effect = EFFECTS[digit].id;
    else if (k === 'n') settings.effect = stepEffect(1);
    else if (k === 'b') settings.effect = stepEffect(-1);
    else if (KEYS[k]) [settings[KEYS[k][0]]] = [KEYS[k][1]];
    else if (k === 'v') settings.bg = stepBg(1);
    else if (k === 'x') settings.bg = stepBg(-1);
    else if (k === 'f') settings.face = FACE_CYCLE[(FACE_CYCLE.indexOf(settings.face) + 1) % 3];
    else if (k === 'm') settings.mirror = !settings.mirror;
    else if (k === 'c') settings.chroma = !settings.chroma;
    else if (k === 'g') settings.grain = settings.grain ? 0 : 6;
    else if (k === 'p') $('#paletteBtn').click();
    else if (k === 's') $('#panel').classList.toggle('open');
    else if (k === 'h' || k === '?') $('#keys').classList.toggle('hide');
    else return;
    syncUI(); onSettings(digit >= 0 || k === 'n' || k === 'b' ? 'effect' : k === 'v' || k === 'x' || KEYS[k] ? 'bg' : k);
  });
  // show the key list briefly on arrival
  $('#keys').classList.remove('hide');
  setTimeout(() => $('#keys').classList.add('hide'), 6000);
}

// ---- installable app: manifest.json + sw.js (offline, and the person model cached)
const STANDALONE = matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;
// next to this module, so the worker's scope is this directory: /3dvid/ on GitHub
// Pages, /camera/ under serve.py (where /api/ and /media/ stay outside it)
const swReady = 'serviceWorker' in navigator && window.isSecureContext
  ? navigator.serviceWorker.register(new URL('./sw.js', import.meta.url).href)
      .catch(e => { console.warn('service worker not registered', e); return null; })
  : Promise.resolve(null);
// MediaPipe files loaded before the worker took control (the first visit) went past
// it; hand their URLs over so the person model works offline after this visit too
async function cacheMediaPipe() {
  if (!(await swReady)) return;
  const reg = await navigator.serviceWorker.ready;
  const urls = performance.getEntriesByType('resource').map(e => e.name)
    .filter(u => u.includes('/@mediapipe/tasks-vision@') || u.includes('/mediapipe-models/'));
  if (urls.length) reg.active?.postMessage({ cacheMediaPipe: urls });
}
let installPrompt = null;
if (!MEETING && !STANDALONE) {
  // Chromium offers installing; keep its event for our own button rather than its banner
  addEventListener('beforeinstallprompt', e => {
    e.preventDefault();
    installPrompt = e;
    $('#installBtn').classList.remove('hide');
  });
  addEventListener('appinstalled', () => { installPrompt = null; $('#installBtn').classList.add('hide'); });
  $('#installBtn').onclick = async () => {
    if (!installPrompt) return;
    const p = installPrompt;
    installPrompt = null;               // a prompt can only be shown once
    $('#installBtn').classList.add('hide');
    await p.prompt();
  };
  // iOS has no install prompt at all; it only installs from the share sheet
  const ios = /iPhone|iPad|iPod/.test(navigator.userAgent)
    || (navigator.userAgent.includes('Macintosh') && navigator.maxTouchPoints > 1);   // iPadOS claims to be a Mac
  $('#iosHint').classList.toggle('hide', !ios);
}
// the browser-extension download: only where the zip is published next to this page
// (the GitHub Pages build adds it; serve.py has none, and offline nothing shows), and
// only on desktop Chromium: userAgentData is Chromium-only and .mobile is true on Android,
// and phones, Safari and Firefox cannot load the extension
async function detectExtension() {
  if (MEETING || !(navigator.userAgentData && !navigator.userAgentData.mobile)) return;
  try {
    const r = await fetch($('#extLink').href, { method: 'HEAD', cache: 'no-store' });
    $('#extBlock').classList.toggle('hide', r.status !== 200);
  } catch { /* offline: keep it hidden */ }
}

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

// ---- studio: the server this page came from (serve.py), or one at settings.studio
export const studio = { available: false };
// base URL every API call is resolved against, always ending in a slash
const studioBase = () => (settings.studio ? settings.studio + '/' : new URL('../', location.href).href);

// link to the studio page. A studio elsewhere is another origin with its own, empty
// localStorage, so the link carries that studio's token, like the printed "studio" link
// (the studio page stores it and takes it out of the address bar)
const studioHref = () => {
  const t = settings.studio ? localStorage.getItem(tokenKey()) : '';
  return studioBase() + (t ? '?token=' + encodeURIComponent(t) : '');
};
// same bearer-token handling as the studio page, but the token is looked up per studio
async function api(path, opts = {}) {
  const url = new URL(path, studioBase());
  const key = tokenKey();
  let token = localStorage.getItem(key) || '';
  opts.headers = { ...(opts.headers || {}), ...(token ? { Authorization: 'Bearer ' + token } : {}) };
  let r = await fetch(url, opts);
  if (r.status === 401) {
    const t = prompt(`${url.host} needs its access token:`);
    if (t === null) throw new Error('no token');
    token = t.trim(); localStorage.setItem(key, token);
    $('#studioLink').href = studioHref();
    opts.headers.Authorization = 'Bearer ' + token;
    r = await fetch(url, opts);
  }
  return r;
}
// status line with a name the server supplied and a link to the studio: built from
// nodes, never innerHTML, so neither can inject markup into this page
function studioStatus(el, before, name, mid, linkText) {
  const b = document.createElement('b'); b.textContent = name;
  const a = document.createElement('a'); a.href = studioHref(); a.target = '_blank'; a.rel = 'noopener'; a.textContent = linkText;
  el.replaceChildren(before, b, mid, a, '.');
}
const STUDIO_NOTE = $('#studioNote').innerHTML;
async function detectStudio() {
  studio.available = false;
  const link = $('#studioLink'), note = $('#studioNote');
  note.innerHTML = STUDIO_NOTE; note.style.color = '';
  link.classList.add('hide');
  $('#studioOpts').classList.add('hide');
  if (studioError) { note.textContent = studioError; note.style.color = 'var(--err)'; return; }
  const base = studioBase();
  try {
    const r = await fetch(new URL('api/health', base));
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const h = await r.json();
    if (h.app !== '3dvid' && !('auth' in h)) throw new Error('not a 3dvid studio');
    studio.available = true;
    link.href = studioHref();
    // name the studio when it is not the server this page came from
    link.textContent = settings.studio ? `Studio: ${new URL(base).host}${h.cuda ? ' · GPU' : ''} →` : 'Studio →';
    link.classList.remove('hide');
    if (!MEETING) $('#studioOpts').classList.remove('hide');
    if (settings.studio) note.textContent = `Connected to ${new URL(base).host}${h.cuda ? ' (GPU)' : ''}.`;
  } catch (e) {
    // a typed studio that does not answer deserves a message (under the field, since
    // the status pill is overwritten by the fps counter); static hosting with no
    // studio set does not - the page works on its own
    if (settings.studio) {
      note.textContent = `${new URL(base).host} is not answering (${e.message}). Is it running, and started with CORS_ORIGINS for this site?`;
      note.style.color = 'var(--err)';
    }
  }
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
  showResult({ blob, rawBlob, seconds, effect: settings.effect,
               name: `3dvid-${settings.effect}-${settings.bg}-${stamp()}.${ext}` });
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
  // the studio can only re-render the effects stylize.py has
  const exact = !res.still && isStudioEffect(res.effect);
  $('#hqStatus').textContent = res.still || !studio.available ? ''
    : !exact ? (res.rawBlob ? `${effectLabel(res.effect)} runs only live, so there is no exact studio version of this take ` +
                              `(the studio renders ${STUDIO_LIST}).` : '')
    : res.rawBlob ? '' : 'Tip: turn on "Also keep the raw take" to render the exact version in the studio.';
  $('#hqBtn').classList.toggle('hide', !(res.rawBlob && exact));
  $('#hqBtn').disabled = false;
  $('#sheet').classList.remove('hide');
}

// upload the untouched camera take and start the same look in the studio, where
// the real depth model and rembg replace the live approximations
$('#hqBtn').onclick = async () => {
  const btn = $('#hqBtn'), status = $('#hqStatus');
  if (!isStudioEffect(result.effect)) return;       // never ask the studio for a live-only effect
  btn.disabled = true;
  try {
    status.textContent = 'Uploading the raw take…';
    const fd = new FormData();
    fd.append('file', new File([result.rawBlob], `camera-${stamp()}.${extFor(result.rawBlob.type)}`));
    fd.append('source', 'camera');
    let r = await api('api/upload', { method: 'POST', body: fd });
    const up = await r.json();
    if (!r.ok) throw new Error(up.detail || 'upload failed');

    const job = new FormData();
    Object.entries({
      video: up.name, kind: 'stylize', effect: result.effect, bg: settings.bg, layout: settings.layout,
      face_mode: settings.face, subject_x: settings.subjectX, bands: settings.bands, min_cell: settings.minCell, max_cell: settings.maxCell,
      preview: 0, keep_audio: true, chromatic: settings.chroma, grain: settings.grain,
    }).forEach(([k, v]) => job.append(k, v));
    r = await api('api/render', { method: 'POST', body: job });
    if (r.status === 409) {
      studioStatus(status, 'Uploaded as ', up.name, '. The studio is busy with another render; start this one from the ', 'studio');
      return;
    }
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || 'render failed to start');
    studioStatus(status, 'Rendering ', up.name, ' in the studio - follow it on the ', 'studio page');
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
  detectExtension();
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
    cacheMediaPipe().catch(() => {});
  } catch (e) {
    // without a mask the effects still run on the whole frame
    console.error('segmenter failed to load', e);
    setStatus('person model unavailable - full-frame styles');
  }
}
main();
