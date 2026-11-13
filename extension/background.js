// Service worker: fills in default settings, the Alt+Shift+E "next effect" and
// Alt+Shift+G "next background" commands, and an OFF badge while the effects are
// switched off. All state is in chrome.storage.local,
// which the pages' engines watch, so a change here restyles a running call at once.
import { EFFECTS, BACKGROUNDS } from './lib/effects.js';
import { DEFAULTS, loadSettings } from './settings.js';

chrome.runtime.onInstalled.addListener(async () => {
  const have = await chrome.storage.local.get(null);
  const missing = Object.fromEntries(Object.entries(DEFAULTS).filter(([k]) => !(k in have)));
  if (Object.keys(missing).length) await chrome.storage.local.set(missing);
  badge();
});
chrome.runtime.onStartup.addListener(badge);

const CYCLES = { 'cycle-effect': ['effect', EFFECTS], 'cycle-background': ['bg', BACKGROUNDS] };
chrome.commands.onCommand.addListener(async command => {
  if (!CYCLES[command]) return;
  const [key, list] = CYCLES[command];
  const current = (await loadSettings())[key];
  const i = list.findIndex(e => e.id === current);
  await chrome.storage.local.set({ [key]: list[(i + 1) % list.length].id });
});

chrome.storage.onChanged.addListener((changes, area) => { if (area === 'local' && changes.enabled) badge(); });

async function badge() {
  const { enabled } = await loadSettings();
  await chrome.action.setBadgeBackgroundColor({ color: '#1d2748' });
  await chrome.action.setBadgeText({ text: enabled ? '' : 'OFF' });
}
