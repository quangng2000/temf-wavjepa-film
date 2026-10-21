export const element = id => document.getElementById(id);
export const setText = (id, text) => { element(id).textContent = text; };
export const number = value => Number.isFinite(value) ? value.toLocaleString(undefined, {maximumFractionDigits: 0}) : '—';
export const decimal = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : '—';
export const seconds = value => Number.isFinite(value) ? decimal(value, 2) + ' s' : '—';
export const date = value => Number.isFinite(value) ? new Date(value * 1000).toLocaleString() : '—';
export function age(value) {
  if (!Number.isFinite(value)) return 'unknown';
  const amount = Math.max(0, value);
  if (amount < 60) return `${Math.floor(amount)}s`;
  return amount < 3600 ? `${Math.floor(amount / 60)}m` : `${(amount / 3600).toFixed(1)}h`;
}
export function duration(value) {
  if (!Number.isFinite(value)) return '—';
  if (value < 3600) return `${Math.ceil(value / 60)} min`;
  return value < 172800 ? `${(value / 3600).toFixed(1)} hours` : `${(value / 86400).toFixed(1)} days`;
}
export function state(label, kind = '') {
  setText('status', label);
  element('status').className = `status ${kind}`;
}
