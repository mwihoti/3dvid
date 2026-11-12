// 3dvid camera effects, extension side of each page (isolated world, document_start,
// every frame). Answers main-world.js's requests; see there for the protocol. The render
// engine (engine.js and the camera page's modules in lib/) is imported only once a page
// actually asks for a camera, so pages that never do pay for one event listener.
(() => {
  let enabled = true;
  const ready = chrome.storage.local.get('enabled')
    .then(r => { enabled = r.enabled !== false; }, () => {});
  chrome.storage.onChanged.addListener((c, area) => {
    if (area === 'local' && c.enabled) enabled = c.enabled.newValue !== false;
  });

  let engine = null;
  const load = () => (engine ??= import(chrome.runtime.getURL('engine.js')));

  const reply = (el, res) => {
    el.setAttribute('data-3dvid-res', JSON.stringify(res));
    el.dispatchEvent(new Event('3dvid:res'));
  };
  const request = e => {
    const el = e.relatedTarget;
    if (!(el instanceof HTMLVideoElement) || !el.hasAttribute('data-3dvid')) return null;
    e.stopImmediatePropagation();
    return { el, data: JSON.parse(el.getAttribute('data-3dvid')) };
  };

  document.addEventListener('3dvid:gum', async e => {
    const r = request(e);
    if (!r) return;
    r.el.setAttribute('data-3dvid-ack', '');       // read synchronously by the page side
    try {
      await ready;
      if (!enabled) return reply(r.el, { pass: true });
      const { open } = await load();
      reply(r.el, await open(r.data, r.el));
    } catch (err) {
      // the page side falls back to the browser's own camera on anything but a refusal
      console.warn('[3dvid] styled camera unavailable:', err);
      reply(r.el, { error: err?.name || 'Error', message: String(err?.message ?? err) });
    }
  }, true);

  document.addEventListener('3dvid:adopt', e => {
    const r = request(e);
    if (r) load().then(m => m.adopt(r.data.sid, r.el.srcObject)).catch(() => {});
  }, true);

  document.addEventListener('3dvid:check', e => {
    if (request(e)) engine?.then(m => m.check()).catch(() => {});
  }, true);
})();
