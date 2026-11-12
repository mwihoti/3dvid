# 3dvid camera effects (browser extension)

Puts the live camera page's effects (voxel, neon, ASCII, halftone, comic, thermal,
hologram, 8-bit, sketch and the rest) on your camera **in any website**: Google Meet,
Microsoft Teams, Zoom's web client, Whereby and so on. The meeting receives the styled
video in place of the raw camera; the microphone is untouched. For Chrome, Edge, Brave
and other Chromium browsers, version 116 or newer.

It runs the same effect code as https://mwihoti.github.io/3dvid/ (the modules are copied
from `web/camera/` at build time, not forked).

## Use

Click the toolbar icon: turn it on or off, and pick the effect, background, face mode,
resolution, mirror, grain and chroma. Changes apply live in a running call. **Alt+Shift+E**
cycles through the effects (change it at `chrome://extensions/shortcuts`).

- Leave **Mirror** off: Meet, Teams and Zoom mirror your self-view themselves, and
  mirroring here would flip what everyone else sees.
- Switched on after the call had already opened the camera? Pick the camera again in
  the meeting's settings (or rejoin). Switching **off** during a call works at once: the
  plain camera passes straight through.
- **Camera resolution** caps the camera at 480p/720p/1080p. Use 480p on a slow laptop.

## Build and load

```bash
python3 extension/build.py      # python 3.8+, standard library only
```

This copies the shared modules into `extension/lib/`, downloads MediaPipe 1.2.0 and the
selfie-segmenter model into `extension/vendor/` (checked against pinned sha256 sums),
and writes `dist/3dvid-camera-extension.zip`. Running it again reuses the verified
downloads.

Load it: `chrome://extensions` -> turn on **Developer mode** -> **Load unpacked** ->
choose the `extension/` folder (or unzip the `.zip` and choose that). After changing a
file, press the reload arrow on the extension's card. The published camera site also
serves the zip at https://mwihoti.github.io/3dvid/3dvid-camera-extension.zip.

## Privacy

Everything happens on your device. The camera image, the person mask and your settings
never leave the browser: the extension has no server, sends no analytics and asks only
for the `storage` permission (for your settings). MediaPipe and the model ship inside the
extension, and the extension's own content security policy (`connect-src 'self'`) stops
the MediaPipe library's built-in usage logging from reaching the network.

**Why "read and change data on all websites"?** Chrome shows this warning because the
extension runs a small script on every https page (plus `localhost` / `127.0.0.1`, for
testing). It has to: a page's camera request must be intercepted before the page's own
code runs, and the extension cannot know in advance which sites you use for calls.
On a page that never asks for the camera, the scripts only wrap `getUserMedia` and
wait: the effect engine is not loaded and nothing on the page is read. The 3dvid camera page itself is
left alone, since it styles its own camera: GitHub Pages' copy is excluded in the
manifest, and any page (the studio's `/camera/`, say) can opt out with
`<meta name="3dvid-styles-own-camera">`.

## How it works

- `main-world.js` runs in the page's own JS world at `document_start` and wraps
  `navigator.mediaDevices.getUserMedia`. A request with video goes to the extension's
  isolated world. The page gets back the styled video track plus the real microphone
  tracks; the styled track reports the real camera's label, `deviceId`, `groupId`, size
  and frame rate. If anything fails, the page gets the browser's own camera, untouched.
  It never touches a Trusted Types sink, so Meet's and Teams' `require-trusted-types-for`
  policies are fine with it.
- `content.js` (isolated world) answers those requests, and on the first one imports
  `engine.js`. That module opens the real camera with the isolated world's own
  `getUserMedia`, then for every camera frame: `MediaStreamTrackProcessor` -> the page's
  `Renderer` (layout "native") -> `new VideoFrame(canvas)` -> `MediaStreamTrackGenerator`.
  Each consumer gets a clone of the generated track. Because it runs on camera frames
  rather than `requestAnimationFrame`, it keeps running when the meeting tab is in the
  background. One real camera is shared per page and device, and it is released when
  the page stops (or ends) its last track.
- The person mask comes from `seg.html`, a hidden 1 px extension iframe running
  `segment.js` with the vendored MediaPipe at the extension's origin, under the
  extension's CSP. The meeting page's CSP cannot stop it there; Zoom's and Teams'
  `frame-src` do not list `chrome-extension:`, but Chromium exempts extension frames.
  Frames go to the iframe as 256 px ImageBitmaps over a private MessageChannel; the mask
  and silhouette stats come back the same way. Until the first mask arrives the meeting
  gets only the chosen background (never the unmasked room); only if segmentation fails
  ("error" or "blocked") do the effects run on the whole frame. `seg.html` is
  web-accessible only at a per-session dynamic URL (`use_dynamic_url`), so other sites
  cannot embed it; `engine.js`, `settings.js` and `lib/*` cannot use one, because a
  dynamic URL breaks `engine.js`'s relative imports.
- `popup.html` / `popup.js` build their controls from `lib/effects.js`. `background.js`
  handles the shortcut and the OFF badge. Settings live in `chrome.storage.local`.

Debug state is shown on the hidden iframe:
`document.querySelector('iframe[data-3dvid]').dataset` holds `3dvidSeg`
(`active` / `blocked` / `error`), `3dvidMode` (`track` or the `element` fallback),
`3dvidPerson`, `3dvidCameras` and `3dvidFrames`.
