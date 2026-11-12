// Person mask service for engine.js (see SegClient there). The page's content script
// sends a MessagePort; frames arrive on it as small ImageBitmaps and go back as the
// Uint8 mask plus personStats. MediaPipe and the model come from vendor/ (no remote
// code); the extension's CSP (connect-src 'self') also stops MediaPipe's usage logging
// from reaching the network.
import { Segmenter } from './lib/segment.js';

const base = new URL('vendor', location.href).href;
const model = new URL('vendor/selfie_segmenter.tflite', location.href).href;
let port = null;

addEventListener('message', async e => {
  // only the window that embeds us, and only once
  if (port || e.source !== parent || parent === window || e.data?.type !== '3dvid-seg' || !e.ports[0]) return;
  port = e.ports[0];
  port.postMessage({ type: 'hello' });
  let seg;
  try {
    seg = await Segmenter.create({ base, model, simdOnly: true });
  } catch (err) {
    port.postMessage({ type: 'error', message: String(err?.message ?? err) });
    return;
  }
  port.onmessage = ({ data }) => {
    if (data?.type !== 'frame' || !data.bmp) return;
    try {
      seg.update(data.bmp, data.ts);
      const mask = seg.mask.slice();
      port.postMessage({ type: 'mask', id: data.id, mask, w: seg.w, h: seg.h, stats: seg.stats }, [mask.buffer]);
    } catch (err) {
      port.postMessage({ type: 'mask', id: data.id, mask: null, error: String(err?.message ?? err) });
    } finally {
      data.bmp.close();
    }
  };
  port.postMessage({ type: 'ready' });
});
