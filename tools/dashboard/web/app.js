import {element, setText, age, state} from './format.js';
import {render} from './render.js';
let busy = false, lastGood = 0, nextAt = 0;
async function fetchReading() {
  const response = await fetch('/api/status', {cache: 'no-store', signal: AbortSignal.timeout(30000)});
  if (!response.ok) throw new Error(`Dashboard request failed (${response.status})`);
  const data = await response.json();
  if (data.connection_error) throw new Error(data.connection_error);
  if (!Array.isArray(data.rows) || !Number.isFinite(data.fetched_at)) throw new Error('The dashboard returned an incomplete reading.');
  return data;
}
function showError(error) {
  state('Connection unavailable', 'warn'); element('alert').hidden = false;
  setText('alert', error.message + (lastGood ? ' Showing the last successful reading.' : ''));
}
async function refresh() {
  if (busy) return;
  busy = true; element('refresh').disabled = true;
  try {
    const data = await fetchReading(); render(data); lastGood = data.fetched_at * 1000;
  } catch (error) { showError(error); }
  finally { busy = false; element('refresh').disabled = false; nextAt = Date.now() + 10000; }
}
function tick() {
  if (nextAt && Date.now() >= nextAt && !document.hidden && !busy) refresh();
  setText('next', busy ? 'Reading pod…' : document.hidden ? 'Refresh paused while hidden' : `Next refresh: ${Math.max(0, Math.ceil((nextAt - Date.now()) / 1000))}s`);
  setText('freshness', lastGood ? 'Reading age: ' + age((Date.now() - lastGood) / 1000) : 'Waiting for first reading');
  if (lastGood && Date.now() - lastGood > 40000) state('Readings are stale', 'warn');
}
element('refresh').addEventListener('click', refresh);
document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
setInterval(tick, 1000);
refresh();
