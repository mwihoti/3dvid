// 3dvid camera effects, page side: runs in the page's own JS world (world "MAIN",
// document_start, every frame) so it can wrap the getUserMedia the page calls.
//
// The styling happens in the extension's isolated world (content.js + engine.js), which
// the page's CSP cannot touch. The two worlds share DOM nodes but not JS objects, so a
// request travels on a detached <video>: this side writes the constraints on it and
// hands it over as a MouseEvent's relatedTarget; the other side acknowledges at once (a
// synchronous attribute), then answers by setting the element's srcObject to the styled
// MediaStream and firing an event on it. Strings and DOM only: no Trusted Types sink,
// no eval, nothing that the meeting apps' CSPs forbid. Anything that goes wrong falls
// back to the browser's own getUserMedia, so the page is never left without a camera.
(() => {
  const MD = globalThis.MediaDevices?.prototype;
  if (!MD?.getUserMedia || MD.getUserMedia.__3dvid) return;   // insecure context, or injected twice

  const gum = MD.getUserMedia;
  const T = MediaStreamTrack.prototype;
  const { stop: stopTrack, clone: cloneTrack } = T;
  const cloneStream = MediaStream.prototype.clone;
  const ours = new WeakMap();      // styled track -> { info, sid }

  function send(type, data, srcObject) {
    const el = document.createElement('video');
    el.setAttribute('data-3dvid', JSON.stringify(data));
    if (srcObject) el.srcObject = srcObject;
    document.dispatchEvent(new MouseEvent('3dvid:' + type, { relatedTarget: el }));
    return el;
  }

  // resolves with { res, stream, el }, or null when no isolated world answered (the
  // extension was reloaded, or this frame has no content script)
  function ask(data) {
    return new Promise(resolve => {
      let el;
      const done = () => resolve({ res: JSON.parse(el.getAttribute('data-3dvid-res')), stream: el.srcObject, el });
      try { el = send('gum', data); } catch { return resolve(null); }
      if (!el.hasAttribute('data-3dvid-ack')) return resolve(null);
      if (el.hasAttribute('data-3dvid-res')) return done();
      el.addEventListener('3dvid:res', done, { once: true });
    });
  }

  // The styled track is a generated track; give it the real camera's face, so device
  // pickers, "which camera is this" checks and resolution logic keep working. el is the
  // request's element, on which the engine keeps the size actually being sent.
  function dress(t, info, sid, el) {
    ours.set(t, { info, sid });
    let constraints = info.constraints;
    const own = (name, value) => Object.defineProperty(t, name, { value, configurable: true, writable: true });
    Object.defineProperty(t, 'label', { get: () => info.label, configurable: true });
    own('getSettings', function getSettings() {
      return { ...info.settings, ...JSON.parse(el.getAttribute('data-3dvid-live') || '{}') };
    });
    own('getCapabilities', function getCapabilities() { return structuredClone(info.capabilities); });
    own('getConstraints', function getConstraints() { return structuredClone(constraints); });
    // the styled size follows the real camera (shared by every consumer); accept the
    // request so callers proceed, and report it back as the track's constraints
    own('applyConstraints', async function applyConstraints(c) { constraints = c ?? {}; });
    own('stop', function stop() { stopTrack.call(this); send('check', { sid }); });
    own('clone', function clone() {
      const c = cloneTrack.call(this);
      dress(c, info, sid, el);
      send('adopt', { sid }, new MediaStream([c]));   // one more consumer keeps the camera open
      return c;
    });
    return t;
  }

  async function styled(constraints) {
    const v = constraints.video;
    const reply = await ask({ video: v === true ? {} : v, audio: constraints.audio ?? false });
    if (!reply) return null;
    const { res, stream, el } = reply;
    if (res.ok && stream) {
      const tracks = stream.getTracks();
      const video = tracks.find(t => t.kind === 'video');
      if (video) {
        dress(video, res.info, res.sid, el);
        return new MediaStream(tracks);
      }
      tracks.forEach(t => stopTrack.call(t));
    }
    // a refusal is final: asking the browser again would prompt the user a second time
    if (res.error === 'NotAllowedError') throw new DOMException(res.message || 'Permission denied', res.error);
    return null;
  }

  // a plain function (length 0, no prototype), like the native method
  const wrapped = {
    getUserMedia() {
      const constraints = arguments[0];
      // a page that styles its own camera (the 3dvid camera page, wherever it is served)
      // opts out with <meta name="3dvid-styles-own-camera">
      if (!constraints?.video || document.querySelector('meta[name="3dvid-styles-own-camera"]'))
        return gum.apply(this, arguments);
      return styled(constraints).catch(e => {
        if (e?.name === 'NotAllowedError') throw e;
        return null;
      }).then(s => s ?? gum.apply(this, [constraints]));
    },
    // MediaStream#clone copies tracks natively, past the per-track clone above
    clone() {
      if (!this.getTracks().some(t => ours.has(t))) return cloneStream.apply(this, arguments);
      return new MediaStream(this.getTracks().map(t => t.clone()));
    },
  };
  Object.defineProperty(wrapped.getUserMedia, '__3dvid', { value: true });

  const patch = (proto, name, fn) =>
    Object.defineProperty(proto, name, { ...Object.getOwnPropertyDescriptor(proto, name), value: fn });
  patch(MD, 'getUserMedia', wrapped.getUserMedia);
  patch(MediaStream.prototype, 'clone', wrapped.clone);

  // the callback-style legacy API calls the native code directly; route it through ours
  if (typeof Navigator.prototype.webkitGetUserMedia === 'function') {
    patch(Navigator.prototype, 'webkitGetUserMedia', {
      webkitGetUserMedia(constraints, ok, fail) { navigator.mediaDevices.getUserMedia(constraints).then(ok, fail); },
    }.webkitGetUserMedia);
  }
})();
