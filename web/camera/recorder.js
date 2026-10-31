// Recording the styled canvas (plus the mic) and getting the file onto the device.

// MP4/H.264 first: phones' galleries and every social app take it as-is. Plain
// "video/mp4" comes LAST: a browser without an H.264 encoder (e.g. Chromium
// builds without proprietary codecs) honours it with VP9 inside MP4 - a .mp4 that
// photo apps reject - so an honest WebM is better there. Safari matches it with H.264.
const TYPES = [
  'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
  'video/mp4;codecs=avc1',
  'video/webm;codecs=vp9,opus',
  'video/webm;codecs=vp8,opus',
  'video/webm',
  'video/mp4',
];

export function pickType() {
  if (!window.MediaRecorder) return null;
  return TYPES.find(t => MediaRecorder.isTypeSupported(t)) ?? '';
}

export const extFor = type => (type || '').startsWith('video/mp4') ? 'mp4' : 'webm';

export class Recorder {
  constructor(stream, videoBitsPerSecond) {
    const type = pickType();
    if (type === null) throw new Error('This browser cannot record video (no MediaRecorder)');
    this.rec = new MediaRecorder(stream, {
      ...(type ? { mimeType: type } : {}), videoBitsPerSecond, audioBitsPerSecond: 128000,
    });
    this.chunks = [];
    this.rec.ondataavailable = e => { if (e.data.size) this.chunks.push(e.data); };
  }
  start() {
    this.chunks = [];
    this.rec.start(1000);            // 1 s chunks: a crash or reload loses at most a second
  }
  stop() {
    return new Promise(resolve => {
      this.rec.onstop = () => resolve(new Blob(this.chunks, { type: this.rec.mimeType || 'video/webm' }));
      this.rec.stop();
    });
  }
}

export function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
}

// Phones: the share sheet is the way into the camera roll ("Save Video" on iOS,
// Save/Files on Android). Desktops: a plain download. Must run inside the click
// handler - share() needs the user's tap, so nothing may be awaited before it.
export async function saveToDevice(blob, name) {
  const file = new File([blob], name, { type: blob.type });
  const phone = matchMedia('(pointer: coarse)').matches;
  if (phone && navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: name });
      return 'shared';
    } catch (e) {
      if (e.name === 'AbortError') return 'cancelled';
    }
  }
  download(blob, name);
  return 'downloaded';
}

export function stamp(d = new Date()) {
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
