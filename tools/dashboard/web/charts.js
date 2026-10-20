import {element, number} from './format.js';
const WIDTH = 600, HEIGHT = 250, LEFT = 66, RIGHT = 16, TOP = 26, BOTTOM = 43;
const svgText = (x, y, text, anchor = 'start') => `<text x="${x}" y="${y}" text-anchor="${anchor}" fill="#a5b4a3" font-size="11">${text}</text>`;
const axisNumber = value => Math.abs(value) < .01 && value !== 0 ? value.toExponential(1) : value.toFixed(value < 1 ? 3 : 2);
function collectSeries(rows, lines) {
  return lines.map(line => ({...line, points: rows.filter(row => Number.isFinite(row.step) && Number.isFinite(row[line.key]))
    .map(row => ({x: row.step, y: row[line.key]}))}));
}
function makeScale(series, threshold) {
  const points = series.flatMap(line => line.points), values = points.map(point => point.y);
  if (Number.isFinite(threshold)) values.push(threshold);
  const first = Math.min(...points.map(point => point.x)), last = Math.max(...points.map(point => point.x));
  const minimum = Math.min(0, ...values), maximum = Math.max(...values), pad = Math.max((maximum - minimum) * .08, 1e-7);
  const low = minimum < 0 ? minimum - pad : 0, high = maximum + pad;
  return {first, last, low, high,
    x: value => LEFT + (value - first) / (last - first || 1) * (WIDTH - LEFT - RIGHT),
    y: value => TOP + (high - value) / (high - low) * (HEIGHT - TOP - BOTTOM)};
}
function drawAxes(scale, label) {
  let result = svgText(LEFT, 14, label);
  for (let index = 0; index < 4; index++) {
    const value = scale.low + (scale.high - scale.low) * index / 3, y = scale.y(value);
    result += `<line x1="${LEFT}" y1="${y}" x2="${WIDTH - RIGHT}" y2="${y}" stroke="#303d30"/>`;
    result += svgText(LEFT - 8, y + 4, axisNumber(value), 'end');
  }
  result += svgText(LEFT, HEIGHT - 23, number(scale.first));
  result += svgText(WIDTH - RIGHT, HEIGHT - 23, number(scale.last), 'end');
  return result + svgText((LEFT + WIDTH - RIGHT) / 2, HEIGHT - 4, 'Completed optimizer update', 'middle');
}
function drawLine(line, scale) {
  const points = line.points.map(point => `${scale.x(point.x).toFixed(2)},${scale.y(point.y).toFixed(2)}`).join(' ');
  const path = `<polyline fill="none" stroke="${line.color}" stroke-width="2" points="${points}"/>`;
  return line.points.length === 1 ? `<circle cx="${scale.x(line.points[0].x)}" cy="${scale.y(line.points[0].y)}" r="3" fill="${line.color}"/>` : path;
}
function drawThreshold(scale, threshold) {
  if (!Number.isFinite(threshold)) return '';
  const y = scale.y(threshold);
  return `<line x1="${LEFT}" y1="${y}" x2="${WIDTH - RIGHT}" y2="${y}" stroke="#eac482" stroke-dasharray="5 5"/>`;
}
export function chart(id, rows, options) {
  const series = collectSeries(rows, options.lines), target = element(id);
  if (!series.some(line => line.points.length)) {
    target.innerHTML = '<div class="empty">Waiting for measurements</div>';
    return;
  }
  const scale = makeScale(series, options.threshold);
  target.innerHTML = `<svg viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="${options.description}">`
    + drawAxes(scale, options.label) + drawThreshold(scale, options.threshold)
    + series.map(line => drawLine(line, scale)).join('') + '</svg>';
}
export function renderCharts(data) {
  chart('loss-chart', data.rows, {label: 'Loss (dimensionless)', description: 'Raw and weighted TEMF loss by completed optimizer update',
    lines: [{key: 'loss', color: '#b6e38e'}, {key: 'raw_loss', color: '#82c5de'}]});
  chart('gradient-chart', data.rows, {label: 'Gradient L2 norm', description: 'Pre-clipping gradient norm with clipping threshold one', threshold: 1,
    lines: [{key: 'gradient_norm', color: '#b6e38e'}]});
  chart('speed-chart', data.rate_history || [], {label: 'Observed seconds / update', description: 'Observed wall-clock seconds per completed update',
    lines: [{key: 'seconds', color: '#b6e38e'}]});
}
