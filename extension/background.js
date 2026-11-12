// Service worker: fills in default settings, the Alt+Shift+E "next effect" command and
// an OFF badge while the effects are switched off. All state is in chrome.storage.local,
// which the pages' engines watch, so a change here restyles a running call at once.
import { EFFECTS } from './lib/effects.js';
import { DEFAULTS, loadSettings } from './settings.js';

chrome.runtime.onInstalled.addListener(async () => {
  const have = await chrome.storage.local.get(null);
  const missing = Object.fromEntries(Object.entries(DEFAULTS).filter(([k]) => !(k in have)));
  if (Object.keys(missing).length) await chrome.storage.local.set(missing);
  badge();
});
chrome.runtime.onStartup.addListener(badge);

chrome.commands.onCommand.addListener(async command => {
  if (command !== 'cycle-effect') return;
  const { effect } = await loadSettings();
  const i = EFFECTS.findIndex(e => e.id === effect);
  await chrome.storage.local.set({ effect: EFFECTS[(i + 1) % EFFECTS.length].id });
});

chrome.storage.onChanged.addListener((changes, area) => { if (area === 'local' && changes.enabled) badge(); });

async function badge() {
  const { enabled } = await loadSettings();
  await chrome.action.setBadgeBackgroundColor({ color: '#1d2748' });
  await chrome.action.setBadgeText({ text: enabled ? '' : 'OFF' });
}
