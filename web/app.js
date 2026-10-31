'use strict';

// ---------------------------------------------------------------------------------------
// Constants and helpers
// ---------------------------------------------------------------------------------------
const STORAGE_KEY = 'oscillator-app-v1';
const METHOD_COLORS = {
  euler: '#1f77b4', improved_euler: '#2ca02c', heun: '#8c564b', euler_cromer: '#17becf',
  verlet: '#9467bd', rk4: '#d62728', rk45: '#e377c2',
};
const DEFAULT_METHODS = ['euler', 'improved_euler', 'euler_cromer', 'verlet', 'rk4', 'rk45'];
const DAMPING_COLORS = ['#2a6fdb', '#e8871e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf'];
const PLOT_CONFIG = { responsive: true, displaylogo: false };
const BASE_LAYOUT = {
  margin: { l: 70, r: 20, t: 40, b: 50 },
  legend: { orientation: 'h', yanchor: 'top', y: -0.22 },
  hovermode: 'x unified',
  font: { family: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', size: 12 },
};

let methods = [];          // from /api/defaults
let notebook = null;       // notebook defaults
let selectedMethods = [];
let lastSimulation = null;
let simulationCounter = 0;

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const child of [].concat(children)) {
    node.append(child instanceof Node ? child : document.createTextNode(child));
  }
  return node;
}

function debounce(fn, delay) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined) return '∞';
  if (Number.isNaN(value)) return '–';
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(digits - 1);
  return Number(value.toPrecision(digits)).toString();
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function setStatus(id, text, kind = '') {
  const node = $(id);
  node.textContent = text;
  node.className = `status ${kind}`;
}

function num(id) {
  return parseFloat($(id).value);
}

const methodLabel = (key) => methods.find((m) => m.key === key)?.label ?? key;

// ---------------------------------------------------------------------------------------
// Persistence: every input marked data-persist, plus the chosen methods
// ---------------------------------------------------------------------------------------
function save() {
  const values = {};
  document.querySelectorAll('[data-persist]').forEach((input) => {
    values[input.id] = input.type === 'checkbox' ? input.checked : input.value;
  });
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ values, methods: selectedMethods }));
}

function restore() {
  let stored;
  try {
    stored = JSON.parse(localStorage.getItem(STORAGE_KEY));
  } catch {
    stored = null;
  }
  if (!stored) return;
  for (const [id, value] of Object.entries(stored.values || {})) {
    const input = $(id);
    if (!input) continue;
    if (input.type === 'checkbox') input.checked = Boolean(value);
    else input.value = value;
  }
  if (Array.isArray(stored.methods)) {
    selectedMethods = stored.methods.filter((key) => methods.some((m) => m.key === key));
  }
}

function resetAll() {
  localStorage.removeItem(STORAGE_KEY);
  location.reload();
}

// ---------------------------------------------------------------------------------------
// Reading the inputs
// ---------------------------------------------------------------------------------------
function readSystem() {
  return { mass: num('mass'), spring_constant: num('spring'), damping: num('damping') };
}

function readForce() {
  const kind = $('force-kind').value;
  if (kind === 'pulse') {
    return { kind, amplitude: num('force-amplitude'), start: num('pulse-start'), duration: num('pulse-duration') };
  }
  if (kind === 'sinusoidal') {
    return { kind, amplitude: num('force-amplitude'), angular_frequency: num('force-omega'), phase: num('force-phase') };
  }
  return { kind: 'none' };
}

function readRun() {
  return {
    system: readSystem(), force: readForce(), x0: num('x0'), v0: num('v0'),
    total_time: num('total-time'), rtol: parseFloat($('rtol').value), methods: selectedMethods.slice(),
  };
}

// ---------------------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------------------
function showTab(name) {
  document.querySelectorAll('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === name));
  document.querySelectorAll('.panel').forEach((panel) => panel.classList.toggle('active', panel.id === `panel-${name}`));
  $('run-settings').classList.toggle('hidden', !['compare', 'convergence'].includes(name));
  $('step-field').classList.toggle('hidden', name !== 'compare');
  document.querySelectorAll(`#panel-${name} .js-plotly-plot`).forEach((plot) => Plotly.Plots.resize(plot));
}

// ---------------------------------------------------------------------------------------
// Sidebar: damping presets and system summary
// ---------------------------------------------------------------------------------------
function dampingPresets() {
  const { mass, spring_constant: k } = readSystem();
  const bCrit = mass > 0 && k > 0 ? 2 * Math.sqrt(k * mass) : NaN;
  const presets = (notebook?.dampings ?? [0.1, 0.5, 2.0]).map((b) => ({ label: `${b}`, value: b }));
  if (Number.isFinite(bCrit)) {
    presets.push({ label: '½ b_cr', value: bCrit / 2 }, { label: 'b_cr', value: bCrit }, { label: '2 b_cr', value: 2 * bCrit });
  }
  return presets;
}

function renderPresets() {
  const container = $('damping-presets');
  const current = num('damping');
  container.replaceChildren(...dampingPresets().map((preset) => {
    const button = el('button', { type: 'button', textContent: preset.label, title: `b = ${fmt(preset.value, 6)} kg/s` });
    if (Math.abs(preset.value - current) <= 1e-9 * Math.max(1, Math.abs(current))) button.classList.add('active');
    button.addEventListener('click', () => {
      $('damping').value = preset.value;
      onInputChanged();
    });
    return button;
  }));
}

const updateSummary = debounce(async () => {
  const box = $('system-summary');
  try {
    const { summary: s } = await api('/api/system', { system: readSystem() });
    const rows = [
      ['ω₀ (rad/s)', fmt(s.natural_frequency, 4)],
      ['Period 2π/ω₀ (s)', fmt(s.natural_period, 4)],
      ['γ = b/2m (1/s)', fmt(s.gamma, 4)],
      ['b_cr = 2√(km)', fmt(s.critical_damping, 4)],
      ['ζ = b/b_cr', fmt(s.damping_ratio, 4)],
      ['Q = mω₀/b', fmt(s.quality_factor, 4)],
      ['ω_d (rad/s)', s.damped_frequency === null ? '–' : fmt(s.damped_frequency, 4)],
      ['Decay time (s)', fmt(s.decay_time, 4)],
    ];
    box.replaceChildren(
      el('span', { className: `regime ${s.regime.replace(' ', '-')}`, textContent: s.regime }),
      el('dl', {}, rows.flatMap(([label, value]) => [el('dt', {}, label), el('dd', {}, value)])),
    );
    $('omega-natural').dataset.value = s.natural_frequency;
  } catch (error) {
    box.replaceChildren(el('span', { className: 'status error', textContent: error.message }));
  }
}, 150);

// ---------------------------------------------------------------------------------------
// Method chips and method table
// ---------------------------------------------------------------------------------------
function renderMethodChips() {
  const container = $('method-chips');
  container.replaceChildren(...methods.map((method) => {
    const input = el('input', { type: 'checkbox', checked: selectedMethods.includes(method.key) });
    input.addEventListener('change', () => {
      selectedMethods = methods.map((m) => m.key).filter((key) =>
        key === method.key ? input.checked : selectedMethods.includes(key));
      onInputChanged();
    });
    const dot = el('span', { className: 'dot' });
    dot.style.background = METHOD_COLORS[method.key];
    return el('label', { className: 'chip', title: method.description }, [input, dot, method.label]);
  }));
}

function renderMethodsTable() {
  const tbody = $('methods-table').querySelector('tbody');
  tbody.replaceChildren(...methods.map((m) => el('tr', {}, [
    el('td', {}, m.label), el('td', {}, m.order === null ? 'adaptive' : String(m.order)), el('td', {}, m.description),
  ])));
}

// ---------------------------------------------------------------------------------------
// Method comparison
// ---------------------------------------------------------------------------------------
function updateForceFields() {
  const kind = $('force-kind').value;
  document.querySelectorAll('.force-param').forEach((field) => field.classList.toggle('hidden', !field.classList.contains(kind)));
}

const runSimulation = debounce(async () => {
  const payload = { ...readRun(), step: num('step') };
  const id = ++simulationCounter;
  setStatus('compare-status', 'Running…', 'busy');
  try {
    const result = await api('/api/simulate', payload);
    if (id !== simulationCounter) return;  // a newer run has started
    lastSimulation = { payload, result };
    renderSimulation();
    setStatus('compare-status', '');
  } catch (error) {
    if (id === simulationCounter) setStatus('compare-status', error.message, 'error');
  }
}, 350);

function renderSimulation() {
  if (!lastSimulation) return;
  const { payload, result } = lastSimulation;
  $('compare-results').classList.remove('hidden');
  const metric = $('rank-metric').value;
  const byKey = Object.fromEntries(result.results.map((r) => [r.key, r]));

  // Error table, best first
  const tbody = $('error-table').querySelector('tbody');
  tbody.replaceChildren(...result.ranking[metric].map((key, rank) => {
    const r = byKey[key];
    const method = methods.find((m) => m.key === key);
    const swatch = el('span', { className: 'swatch' });
    swatch.style.background = METHOD_COLORS[key];
    const row = el('tr', { className: rank === 0 ? 'best' : '' }, [
      el('td', {}, String(rank + 1)),
      el('td', {}, [swatch, r.label]),
      el('td', {}, method?.order === null ? 'adaptive' : String(method?.order)),
      el('td', {}, fmt(r.errors.mae)),
      el('td', {}, fmt(r.errors.mse)),
      el('td', {}, fmt(r.errors.rmse)),
      el('td', {}, fmt(r.errors.max)),
      el('td', {}, r.final_energy_error === null ? '–' : `${fmt(100 * r.final_energy_error)} %`),
      el('td', {}, r.runtime_ms.toFixed(1)),
    ]);
    return row;
  }));
  const shown = result.t.length < result.n_steps + 1 ? ` Plots show ${result.t.length.toLocaleString()} of the points.` : '';
  $('compare-summary').textContent = `${result.n_steps.toLocaleString()} steps of h = ${fmt(result.step, 6)} s ` +
    `(${result.summary.regime}). Errors are measured on every step against the exact solution.${shown}`;

  const t = result.t;
  const exactX = result.exact.x;
  const xScale = Math.max(...exactX.map(Math.abs).filter(Number.isFinite), 1e-12);
  const blownUp = result.results.some((r) => r.x.some((x) => x === null || Math.abs(x) > 4 * xScale));
  const methodTraces = (field, extra = {}) => result.results.map((r) => ({
    x: t, y: r[field], name: r.label, mode: 'lines', line: { color: METHOD_COLORS[r.key], width: 1.5 }, ...extra,
  }));

  // Position
  const positionTraces = [
    ...methodTraces('x'),
    { x: t, y: exactX, name: 'Exact', mode: 'lines', line: { color: '#111', dash: 'dash', width: 2 } },
  ];
  const positionLayout = {
    ...BASE_LAYOUT, title: { text: 'Position x(t)' },
    xaxis: { title: { text: 'Time (s)' } },
    yaxis: { title: { text: 'Position (m)' }, ...(blownUp ? { range: [-2 * xScale, 2 * xScale] } : {}) },
  };
  if (payload.force.kind !== 'none') {
    positionTraces.push({ x: t, y: result.force, name: 'Force F(t)', yaxis: 'y2', mode: 'lines',
      line: { color: '#999', width: 1 }, fill: 'tozeroy', fillcolor: 'rgba(150,150,150,0.12)' });
    positionLayout.yaxis2 = { title: { text: 'Force (N)' }, overlaying: 'y', side: 'right', showgrid: false, zeroline: false };
    positionLayout.margin = { ...BASE_LAYOUT.margin, r: 60 };
  }
  if (blownUp) {
    positionLayout.annotations = [{ text: 'Some methods leave this range: zoom out or double-click to see them.',
      xref: 'paper', yref: 'paper', x: 0, y: 1.08, showarrow: false, font: { size: 11, color: '#a15c00' }, xanchor: 'left' }];
  }
  Plotly.react('plot-position', positionTraces, positionLayout, PLOT_CONFIG);

  // Error
  const logError = $('log-error').checked;
  const errorTraces = result.results.map((r) => ({
    x: t, y: logError ? r.error.map((e) => (e === null || e === 0 ? null : Math.abs(e))) : r.error,
    name: r.label, mode: 'lines', line: { color: METHOD_COLORS[r.key], width: 1.2 },
  }));
  Plotly.react('plot-error', errorTraces, {
    ...BASE_LAYOUT, title: { text: logError ? '|x_numerical − x_exact|' : 'x_numerical − x_exact' },
    xaxis: { title: { text: 'Time (s)' } },
    yaxis: { title: { text: 'Error (m)' }, type: logError ? 'log' : 'linear', exponentformat: 'power' },
  }, PLOT_CONFIG);

  // Energy
  const exactEnergy = result.exact.energy;
  const eMax = Math.max(...exactEnergy.filter(Number.isFinite), 1e-12);
  const energyBlown = result.results.some((r) => r.energy.some((e) => e === null || e > 3 * eMax));
  Plotly.react('plot-energy', [
    ...methodTraces('energy'),
    { x: t, y: exactEnergy, name: 'Exact', mode: 'lines', line: { color: '#111', dash: 'dash', width: 2 } },
  ], {
    ...BASE_LAYOUT, showlegend: false, title: { text: 'Mechanical energy ½mv² + ½kx² (colours as above)' },
    xaxis: { title: { text: 'Time (s)' } },
    yaxis: { title: { text: 'Energy (J)' }, ...(energyBlown ? { range: [0, 2 * eMax] } : {}) },
  }, PLOT_CONFIG);

  // Phase portrait
  const vScale = Math.max(...result.exact.v.map(Math.abs).filter(Number.isFinite), 1e-12);
  Plotly.react('plot-phase', [
    ...result.results.map((r) => ({ x: r.x, y: r.v, name: r.label, mode: 'lines',
      line: { color: METHOD_COLORS[r.key], width: 1 } })),
    { x: exactX, y: result.exact.v, name: 'Exact', mode: 'lines', line: { color: '#111', dash: 'dash', width: 1.5 } },
  ], {
    ...BASE_LAYOUT, showlegend: false, hovermode: 'closest', title: { text: 'Phase portrait (colours as above)' },
    xaxis: { title: { text: 'x (m)' }, ...(blownUp ? { range: [-2 * xScale, 2 * xScale] } : {}) },
    yaxis: { title: { text: 'v (m/s)' }, ...(blownUp ? { range: [-2 * vScale, 2 * vScale] } : {}) },
  }, PLOT_CONFIG);
}

function downloadCsv() {
  if (!lastSimulation) return;
  const { result } = lastSimulation;
  const header = ['t', 'x_exact', 'v_exact', ...result.results.flatMap((r) => [`x_${r.key}`, `v_${r.key}`])];
  const lines = [header.join(',')];
  result.t.forEach((time, i) => {
    const row = [time, result.exact.x[i], result.exact.v[i], ...result.results.flatMap((r) => [r.x[i], r.v[i]])];
    lines.push(row.map((value) => (value === null ? '' : value)).join(','));
  });
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const link = el('a', { href: URL.createObjectURL(blob), download: 'oscillator_trajectories.csv' });
  link.click();
  URL.revokeObjectURL(link.href);
}

// ---------------------------------------------------------------------------------------
// Convergence
// ---------------------------------------------------------------------------------------
async function runConvergence() {
  const payload = { ...readRun(), h_min: num('h-min'), h_max: num('h-max'), n_points: parseInt($('h-points').value, 10),
    metric: $('conv-metric').value };
  $('run-convergence').disabled = true;
  setStatus('convergence-status', 'Running…', 'busy');
  try {
    const result = await api('/api/convergence', payload);
    renderConvergence(payload, result);
    $('convergence-stale').classList.add('hidden');
    setStatus('convergence-status', '');
  } catch (error) {
    setStatus('convergence-status', error.message, 'error');
  } finally {
    $('run-convergence').disabled = false;
  }
}

function renderConvergence(payload, result) {
  $('convergence-results').classList.remove('hidden');
  const steps = result.steps;
  const traces = payload.methods.map((key) => ({
    x: steps, y: result.errors[key].map((e) => (e === null || e <= 0 ? null : e)), name: methodLabel(key),
    mode: 'lines+markers', line: { color: METHOD_COLORS[key] }, marker: { size: 6 },
  }));
  // Reference slopes h^p, anchored just below the first method that has that order
  for (const p of [1, 2, 4]) {
    const key = payload.methods.find((k) => methods.find((m) => m.key === k)?.order === p);
    if (!key) continue;
    const errors = result.errors[key];
    const i = errors.findIndex((e) => e !== null && e > 1e-12 && e < 1);
    if (i < 0) continue;
    const anchor = errors[i] / 3;
    traces.push({
      x: steps, y: steps.map((h) => anchor * (h / steps[i]) ** p), name: `∝ h${p === 1 ? '' : p === 2 ? '²' : '⁴'}`,
      mode: 'lines', line: { color: '#888', dash: 'dot', width: 1 }, hoverinfo: 'skip',
    });
  }
  const metricName = { max: 'max |error|', rmse: 'RMSE', mae: 'MAE' }[result.metric];
  Plotly.react('plot-convergence', traces, {
    ...BASE_LAYOUT, hovermode: 'closest', title: { text: `Relative error (${metricName} / max |x|) against step size` },
    xaxis: { title: { text: 'Step size h (s)' }, type: 'log', exponentformat: 'power' },
    yaxis: { title: { text: 'Relative error' }, type: 'log', exponentformat: 'power' },
  }, PLOT_CONFIG);

  const tbody = $('order-table').querySelector('tbody');
  tbody.replaceChildren(...payload.methods.map((key) => {
    const expected = methods.find((m) => m.key === key)?.order;
    const observed = result.orders[key];
    const cell = el('td', {}, observed === null ? '–' : observed.toFixed(2));
    if (expected !== null && observed !== null && Math.abs(observed - expected) > 0.3) cell.classList.add('warn');
    return el('tr', {}, [el('td', {}, methodLabel(key)), el('td', {}, expected === null ? 'adaptive' : String(expected)), cell]);
  }));
}

// ---------------------------------------------------------------------------------------
// Resonance
// ---------------------------------------------------------------------------------------
function readDampings() {
  return $('res-dampings').value.split(/[,\s;]+/).filter(Boolean).map(Number);
}

async function runResonance() {
  const system = readSystem();
  const payload = {
    mass: system.mass, spring_constant: system.spring_constant, dampings: readDampings(),
    amplitude: num('res-amplitude'), omega_min: num('res-min'), omega_max: num('res-max'),
    n_points: parseInt($('res-points').value, 10), numeric: $('res-numeric').checked,
    total_time: num('res-time'), step: num('res-step'), x0: num('x0'), v0: num('v0'),
  };
  if (payload.dampings.some(Number.isNaN)) {
    setStatus('resonance-status', 'Damping coefficients must be numbers separated by commas.', 'error');
    return;
  }
  $('run-resonance').disabled = true;
  setStatus('resonance-status', payload.numeric ? 'Running RK4 for every frequency…' : 'Computing…', 'busy');
  try {
    const result = await api('/api/resonance', payload);
    renderResonance(payload, result);
    $('resonance-stale').classList.add('hidden');
    setStatus('resonance-status', '');
  } catch (error) {
    setStatus('resonance-status', error.message, 'error');
  } finally {
    $('run-resonance').disabled = false;
  }
}

function renderResonance(payload, result) {
  $('resonance-results').classList.remove('hidden');
  const omegas = result.omegas;
  const w0 = result.natural_frequency;
  const traces = [];
  result.curves.forEach((curve, i) => {
    const color = DAMPING_COLORS[i % DAMPING_COLORS.length];
    traces.push({ x: omegas, y: curve.exact, name: `b = ${curve.damping} (exact)`, mode: 'lines', line: { color, width: 2 } });
    if (curve.numeric) {
      traces.push({ x: omegas, y: curve.numeric, name: `b = ${curve.damping} (RK4)`, mode: 'markers',
        marker: { color, size: 5, symbol: 'circle-open' } });
    }
  });
  const vline = (x, text) => ({
    shapes: [{ type: 'line', x0: x, x1: x, yref: 'paper', y0: 0, y1: 1, line: { color: '#111', dash: 'dash', width: 1 } }],
    annotations: [{ x, yref: 'paper', y: 1.02, text, showarrow: false, font: { size: 11 } }],
  });
  Plotly.react('plot-resonance', traces, {
    ...BASE_LAYOUT, hovermode: 'closest',
    title: { text: `Steady-state amplitude, F₀ = ${payload.amplitude} N` },
    xaxis: { title: { text: 'Driving angular frequency ω (rad/s)' } },
    yaxis: { title: { text: 'Amplitude (m)' } },
    ...vline(w0, `ω₀ = ${fmt(w0, 3)}`),
  }, PLOT_CONFIG);

  $('resonance-table').querySelector('tbody').replaceChildren(...result.curves.map((c) => el('tr', {}, [
    el('td', {}, String(c.damping)),
    el('td', {}, c.regime),
    el('td', {}, fmt(c.quality_factor, 4)),
    el('td', {}, c.half_power_q === null ? 'no half-power width' : fmt(c.half_power_q, 4)),
    el('td', {}, c.resonance_frequency === null ? 'no peak (ω = 0)' : fmt(c.resonance_frequency, 4)),
    el('td', {}, fmt(c.peak_amplitude, 4)),
    el('td', {}, fmt(c.decay_time, 4)),
    el('td', { className: payload.numeric && !c.settled ? 'warn' : '' },
      payload.numeric ? (c.settled ? 'yes' : 'no: run longer') : '–'),
  ])));

  Plotly.react('plot-phase-lag', result.curves.map((c, i) => ({
    x: omegas, y: c.phase_lag, name: `b = ${c.damping}`, mode: 'lines',
    line: { color: DAMPING_COLORS[i % DAMPING_COLORS.length] },
  })), {
    ...BASE_LAYOUT, hovermode: 'closest', title: { text: 'Phase lag of the response' },
    xaxis: { title: { text: 'ω (rad/s)' } }, yaxis: { title: { text: 'Phase lag δ (degrees)' }, range: [0, 180] },
    ...vline(w0, 'ω₀'),
  }, PLOT_CONFIG);

  const q = result.q_curve;
  Plotly.react('plot-q', [
    { x: q.damping, y: q.formula, name: 'Q = mω₀/b', mode: 'lines', line: { color: '#2a6fdb' } },
    { x: q.damping, y: q.half_power, name: 'From peak width', mode: 'lines', line: { color: '#e8871e', dash: 'dash' } },
    { x: result.curves.map((c) => c.damping), y: result.curves.map((c) => c.quality_factor), name: 'Chosen b',
      mode: 'markers', marker: { color: '#111', size: 8 } },
  ], {
    ...BASE_LAYOUT, hovermode: 'closest', title: { text: 'Quality factor against damping' },
    legend: { x: 0.98, xanchor: 'right', y: 0.98, bgcolor: 'rgba(255,255,255,0.8)' },
    xaxis: { title: { text: 'Damping coefficient b (kg/s)' } },
    yaxis: { title: { text: 'Q' }, type: 'log', exponentformat: 'power' },
    ...vline(q.critical_damping, 'b_cr'),
  }, PLOT_CONFIG);
}

// ---------------------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------------------
function onInputChanged() {
  save();
  renderPresets();
  updateForceFields();
  updateSummary();
  renderMethodChips();
  runSimulation();
  if (!$('convergence-results').classList.contains('hidden')) $('convergence-stale').classList.remove('hidden');
  if (!$('resonance-results').classList.contains('hidden')) $('resonance-stale').classList.remove('hidden');
}

async function init() {
  try {
    ({ methods, notebook } = await api('/api/defaults'));
  } catch (error) {
    document.querySelector('main').prepend(el('div', { className: 'notice' },
      `Could not reach the server (${error.message}). Start it with: python server.py`));
    return;
  }
  selectedMethods = DEFAULT_METHODS.filter((key) => methods.some((m) => m.key === key));
  restore();
  renderMethodChips();
  renderMethodsTable();
  updateForceFields();
  renderPresets();
  updateSummary();

  document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => showTab(tab.dataset.tab)));
  document.querySelectorAll('.sidebar input, #run-settings input, #run-settings select').forEach((input) => {
    input.addEventListener('input', onInputChanged);
  });
  document.querySelectorAll('#panel-convergence input, #panel-convergence select').forEach((input) => {
    input.addEventListener('input', () => {
      save();
      if (!$('convergence-results').classList.contains('hidden')) $('convergence-stale').classList.remove('hidden');
    });
  });
  document.querySelectorAll('#panel-resonance input').forEach((input) => {
    input.addEventListener('input', () => {
      save();
      if (!$('resonance-results').classList.contains('hidden')) $('resonance-stale').classList.remove('hidden');
    });
  });
  $('rank-metric').addEventListener('input', () => { save(); renderSimulation(); });
  $('log-error').addEventListener('input', () => { save(); renderSimulation(); });
  $('omega-natural').addEventListener('click', (event) => {
    event.preventDefault();
    const value = parseFloat(event.target.dataset.value);
    if (Number.isFinite(value)) {
      $('force-omega').value = Number(value.toPrecision(10));
      onInputChanged();
    }
  });
  $('run-convergence').addEventListener('click', runConvergence);
  $('run-resonance').addEventListener('click', runResonance);
  $('reset-all').addEventListener('click', resetAll);
  $('download-csv').addEventListener('click', downloadCsv);

  showTab('compare');
  runSimulation();
}

init();
