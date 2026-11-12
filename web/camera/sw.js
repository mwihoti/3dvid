// 3dvid camera service worker: makes the page installable and usable offline.
//
// - The page's own files (this directory): network first, so an online visit always
//   gets one consistent version, never old JS under new HTML; the cache is only the
//   fallback for offline or a network that has not answered in a few seconds.
// - MediaPipe: the library is pinned to an exact version, so its files never change
//   (cache first); the model URL says "latest" (stale-while-revalidate). Both work
//   offline after one online visit.
// - Everything else is left alone (no respondWith): non-GET requests, other sites
//   (a studio on a tunnel), and same-origin paths outside this directory (serve.py's
//   /api/ and /media/ next to /camera/), plus the extension zip, which is a download.
//
// Shell updates need no version bump (network first fetches them); bump VERSION only
// to change what this file does, and keep SHELL_FILES in step with web/camera.
const VERSION = 'v1';
const SHELL = `3dvid-shell-${VERSION}`;
// not versioned: the library URLs are pinned and the model revalidates itself, and a
// bump must not delete the model the page loaded through the old worker (offline: no mask)
const LIB = '3dvid-mediapipe';
const SHELL_FILES = [
  './', 'app.js', 'render.js', 'shaders.js', 'effects.js', 'gl.js', 'segment.js', 'palette.js', 'recorder.js',
  'manifest.json', 'icons/favicon-32.png', 'icons/apple-touch-icon.png',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/maskable-512.png',
];
const NETWORK_TIMEOUT = 4000;
const MP_LIB = /^https:\/\/cdn\.jsdelivr\.net\/npm\/@mediapipe\/tasks-vision@\d+\.\d+\.\d+\//;
const MP_MODEL = /^https:\/\/storage\.googleapis\.com\/mediapipe-models\//;
const MATCH = { ignoreSearch: true, ignoreVary: true };

self.addEventListener('install', e => {
  // 'reload' skips the HTTP cache, so a new worker never precaches a stale copy
  e.waitUntil(caches.open(SHELL)
    .then(c => c.addAll(SHELL_FILES.map(f => new Request(f, { cache: 'reload' }))))
    .then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k.startsWith('3dvid-') && k !== SHELL && k !== LIB)
                                  .map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  if (MP_LIB.test(req.url)) return e.respondWith(cacheFirst(req));
  if (MP_MODEL.test(req.url)) return e.respondWith(staleWhileRevalidate(e));
  const url = new URL(req.url);
  if (url.origin !== location.origin || !req.url.startsWith(self.registration.scope)
      || url.pathname.endsWith('.zip')) return;
  e.respondWith(networkFirst(e));
});

// The page lists the MediaPipe files it loaded before this worker controlled it (the
// first visit), so those are cached too; they come from the HTTP cache, so it is cheap.
self.addEventListener('message', e => {
  const urls = (e.data?.cacheMediaPipe || []).filter(u => MP_LIB.test(u) || MP_MODEL.test(u));
  e.waitUntil(caches.open(LIB).then(c => Promise.all(urls.map(async u => {
    if (await c.match(u, MATCH)) return;
    const r = await fetch(u);
    if (r.ok) await c.put(u, r);
  }))).catch(() => {}));
});

// Cache key of a shell file: the URL without its query, since ?token=, ?studio= and
// ?meeting all load the same file. It also keeps tokens out of the cache.
const bare = href => { const u = new URL(href); u.search = ''; u.hash = ''; return u.href; };

async function networkFirst(e) {
  const req = e.request, key = bare(req.url);
  const cache = await caches.open(SHELL);
  // 'no-cache' revalidates with the server instead of trusting the HTTP cache
  // (GitHub Pages sends max-age=600), so a new version shows up on the next load
  const network = fetch(req, { cache: 'no-cache' });
  // a copy, because a Response keeps its own URL (with any ?token=) and the cache stores it
  e.waitUntil(network.then(r => (r.ok && r.type === 'basic' ? cache.put(key, new Response(r.clone().body, r)) : null))
    .catch(() => {}));
  // any page in this directory falls back to the shell, e.g. index.html offline
  const cached = async () => (await cache.match(key, MATCH))
    || (req.mode === 'navigate' ? cache.match(self.registration.scope, MATCH) : undefined);
  let timer;
  const slow = new Promise(resolve => { timer = setTimeout(resolve, NETWORK_TIMEOUT); });
  try {
    // a slow network yields to the cache; with nothing cached, keep waiting for it
    return askEveryTime(await Promise.race([network, slow.then(async () => (await cached()) || network)]));
  } catch (err) {
    const hit = await cached();
    if (hit) return askEveryTime(hit);
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// The page's in-memory cache sits in front of this worker: on a reload it reused a
// module it still thought fresh (max-age, or a heuristic from Last-Modified) next to
// newer HTML. "no-cache" makes every load ask here, and so the network.
function askEveryTime(r) {
  // cached copies are constructed Responses (type 'default'), network ones 'basic'
  if (r.type === 'opaque' || r.type === 'opaqueredirect' || !r.ok) return r;
  const headers = new Headers(r.headers);
  headers.set('Cache-Control', 'no-cache');
  headers.delete('Expires');
  // the body below is already decoded; these would describe the wire format
  headers.delete('Content-Encoding'); headers.delete('Content-Length');
  return new Response(r.body, { status: r.status, statusText: r.statusText, headers });
}

async function cacheFirst(req) {
  const cache = await caches.open(LIB);
  const hit = await cache.match(req, MATCH);
  // an opaque copy (from a no-cors <script>) cannot answer a CORS request
  if (hit && (hit.type !== 'opaque' || req.mode === 'no-cors')) return hit;
  const r = await fetch(req);
  if (r.ok || r.type === 'opaque') cache.put(req, r.clone()).catch(() => {});
  return r;
}

async function staleWhileRevalidate(e) {
  const cache = await caches.open(LIB);
  const hit = await cache.match(e.request, MATCH);
  const fresh = fetch(e.request);
  e.waitUntil(fresh.then(r => (r.ok ? cache.put(e.request, r.clone()) : null)).catch(() => {}));
  return hit || fresh;
}
