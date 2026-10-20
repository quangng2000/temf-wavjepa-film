import {element, setText, number, decimal, seconds, date, age, duration, state} from './format.js';
import {renderCharts} from './charts.js';
const STATES = {training: ['Training', 'good'], starting: ['Starting', 'warn'], waiting: ['No recent update', 'warn'],
  inactive: ['Training process absent', 'bad'], failed: ['Training error', 'bad'], complete: ['Target reached', 'good']};
const NOTICES = {waiting: 'The trainer has not logged a recent update. Validation, checkpoint writing or a data wait may explain the pause.',
  inactive: 'No training process was found. The pod remains allocated.', failed: 'The trainer reported a failure. The pod remains allocated.'};
function renderProgress(data) {
  const percent = data.target > 0 ? Math.min(100, data.step / data.target * 100) : 0;
  element('steps').replaceChildren(document.createTextNode(number(data.step) + ' '), Object.assign(document.createElement('span'), {textContent: '/ ' + number(data.target)}));
  setText('percent', percent.toFixed(2) + '%');
  element('fill').style.width = percent + '%';
  element('progress').setAttribute('aria-valuenow', data.step);
  element('progress').setAttribute('aria-valuemax', data.target);
  setText('activity', data.update_age == null ? 'Waiting for the first logged update' : `Latest update written ${age(data.update_age)} ago`);
  setText('schedule', number(data.target) + ' updates');
}
function renderPace(data) {
  const stable = Number.isFinite(data.seconds_per_update) && data.seconds_per_update > 0 && data.rate_sample_count >= 2 && data.observation_seconds >= 60;
  setText('speed', seconds(data.seconds_per_update));
  setText('speed-note', `Wall-clock pace · ${number(data.rate_sample_count)} updates over ${age(data.observation_seconds)}`);
  setText('eta', data.status === 'complete' ? 'Complete' : data.status === 'training' && stable
    ? duration(Math.max(0, data.target - data.step) * data.seconds_per_update) : 'Measuring progress');
}
function renderMetrics(data) {
  const latest = data.rows.at(-1) || {};
  setText('loss', decimal(latest.loss, 6)); setText('raw-loss', decimal(latest.raw_loss, 6));
  setText('gradient', decimal(latest.gradient_norm));
  element('gradient').className = 'value' + (latest.gradient_norm > 1 ? ' warn' : '');
  setText('gradient-note', latest.clipped ? 'Latest update clipped · threshold 1.0' : 'Clipping threshold: 1.0');
  setText('median', decimal(data.gradient_median));
  setText('clipping', Number.isFinite(data.clipping_fraction) ? `${(100 * data.clipping_fraction).toFixed(1)}%` : '—');
  setText('rms', `${decimal(latest.target_rms)} / ${decimal(latest.correction_rms)}`);
  const range = data.rows.length ? `updates ${number(data.rows[0].step)}–${number(data.rows.at(-1).step)}` : 'waiting for logged updates';
  setText('loss-caption', `Source: training metrics · ${range} · unsmoothed.`);
  setText('gradient-caption', `Source: pre-clipping metrics · ${range}. Median and clipping fraction use the latest ${number(Math.min(100, data.rows.length))} logged updates.`);
}
function renderDetails(data) {
  setText('machine', `${data.gpu?.name || 'GPU'} · ${data.region || 'Region unavailable'}`);
  setText('util', Number.isFinite(data.gpu?.utilization) ? `${decimal(data.gpu.utilization, 0)}%` : 'Unavailable');
  const gpu = data.gpu;
  setText('memory', Number.isFinite(gpu?.memory_used_mib) && Number.isFinite(gpu?.memory_total_mib)
    ? `${decimal(gpu.memory_used_mib / 1024, 1)} / ${decimal(gpu.memory_total_mib / 1024, 1)} GiB` : 'Unavailable');
  setText('checkpoint', data.checkpoint ? `${date(data.checkpoint.modified)}${Number.isFinite(data.checkpoint.step) ? ' · update ' + number(data.checkpoint.step) : ''}` : 'Not yet saved');
  setText('updated', date(data.fetched_at)); setText('pod', data.pod_id || '—'); setText('run', data.run_id || '—');
  setText('log', (data.log || []).join('\n'));
}
export function render(data) {
  element('content').hidden = false;
  state(...(STATES[data.status] || ['Unknown', 'warn']));
  element('alert').hidden = !NOTICES[data.status];
  setText('alert', NOTICES[data.status] || '');
  renderProgress(data); renderPace(data); renderMetrics(data); renderDetails(data); renderCharts(data);
}
