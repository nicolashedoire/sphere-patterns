'use strict';

const element = id => document.getElementById('pose-' + id);
const modeSelect = element('mode');
const sceneSelect = element('frame');
const canvas = element('sphere');
const context = canvas.getContext('2d');
const timeline = element('timeline');
const svg = element('boxes');
const photo = element('photo');
const namespace = 'http://www.w3.org/2000/svg';
const modes = {
  direct: 'Direct calculation',
  sphere: 'Sphere',
  feedForward: 'Feed-forward',
  cut: 'Sphere with one route cut',
};
const splits = {
  'control-dev': 'Synthetic controls · calibration',
  'control-test': 'Synthetic controls · evaluation',
  'control-validation': 'Synthetic controls · separate validation',
  development: 'T-LESS photos · calibration',
  heldout: 'T-LESS photos · held-out evaluation',
};
const signalColors = ['#67e8f9', '#fabe72', '#c495ff', '#94f5a9'];
const number = new Intl.NumberFormat('en-US', {maximumFractionDigits: 3});
const formatNumber = value => Number.isFinite(value) ? number.format(value) : '—';

let data;
let frame;
let tick = 0;
let replayStart = 0;
let replayEnd = 0;
let animation = 0;
let previousTime = 0;
let angleX = -0.65;
let angleY = 0.32;
let drag = null;

function isCut(hop, edge) {
  return edge && hop.from === edge[0] && hop.to === edge[1];
}

function getCircuit() {
  const mode = modeSelect.value;
  if (mode === 'direct') return null;
  const circuit = data?.circuit;
  if (mode === 'feedForward') return circuit?.feedRoutes ? {...circuit, routes: circuit.feedRoutes} : null;
  return circuit;
}

function trackedBest() {
  return (modeSelect.value === 'cut' ? frame.sphere : frame[modeSelect.value])?.best;
}

function signalLevel(group) {
  const tracked = trackedBest();
  if (group < 3) return tracked?.votes?.[group] ?? 0;
  const arrived = modeSelect.value === 'cut' ? frame.cut?.atSphereBest : tracked?.votes;
  return arrived?.reduce((total, vote) => total + vote, 0) ?? 0;
}

function labelRoute(route) {
  return route.label ?? (route.group === 3 ? 'Probe sum' : frame.groups?.[route.group] ?? 'Detail ' + (route.group + 1));
}

function draw() {
  const circuit = getCircuit();
  if (!circuit || canvas.hidden) return;
  const {width, height} = canvas.getBoundingClientRect();
  if (width <= 0 || height <= 0) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const radius = Math.min(width, height) * .43;
  const cx = width / 2, cy = height / 2;
  const sinX = Math.sin(angleX), cosX = Math.cos(angleX);
  const sinY = Math.sin(angleY), cosY = Math.cos(angleY);
  const points = circuit.nodes.map(([x, y, z]) => {
    const horizontal = x * cosX + z * sinX;
    const depth = z * cosX - x * sinX;
    const vertical = y * cosY - depth * sinY;
    const facing = depth * cosY + y * sinY;
    const scale = radius / (1 - facing * .12);
    return [cx + horizontal * scale, cy + vertical * scale, facing];
  });

  context.strokeStyle = '#75bbd553';
  context.lineWidth = 1;
  context.beginPath();
  context.arc(cx, cy, radius, 0, Math.PI * 2);
  context.stroke();

  function line(a, b) {
    context.moveTo(points[a][0], points[a][1]);
    context.lineTo(points[b][0], points[b][1]);
  }

  for (const [front, color] of [[false, '#85b5da20'], [true, '#85b5da3d']]) {
    context.beginPath();
    for (const [a, b] of circuit.edges) {
      if ((points[a][2] + points[b][2] >= 0) === front) line(a, b);
    }
    context.strokeStyle = color;
    context.stroke();
  }

  const cutEdge = modeSelect.value === 'cut' ? circuit.cut : null;
  for (const route of circuit.routes) {
    const color = signalColors[route.group];
    const level = Math.abs(signalLevel(route.group));
    let blocked = false;
    for (const hop of route.hops) {
      const cutHere = isCut(hop, cutEdge);
      if (cutHere) blocked = true;
      context.beginPath();
      line(hop.from, hop.to);
      context.lineWidth = cutHere ? 2.4 : 1.6;
      context.strokeStyle = cutHere ? '#ff9eab' : color;
      context.globalAlpha = blocked && !cutHere ? .16 : .72;
      if (cutHere) context.setLineDash([4, 4]);
      context.stroke();
      if (cutHere) context.setLineDash([]);
      context.globalAlpha = 1;
      if (blocked || level <= 0 || tick < hop.start || tick >= hop.end) continue;
      const from = points[hop.from], to = points[hop.to];
      const fraction = (tick - hop.start) / (hop.end - hop.start);
      const x = from[0] + (to[0] - from[0]) * fraction;
      const y = from[1] + (to[1] - from[1]) * fraction;
      context.fillStyle = color;
      context.shadowColor = color;
      context.shadowBlur = 14;
      context.beginPath();
      context.arc(x, y, 3 + 3 * Math.min(level, 1), 0, Math.PI * 2);
      context.fill();
      context.shadowBlur = 0;
    }
  }

  for (const [node, label, color] of [[circuit.mixer, 'Junction', '#f5f8ff'], [circuit.probe, 'Probe', signalColors[3]]]) {
    if (!points[node]) continue;
    const [x, y] = points[node];
    context.fillStyle = color;
    context.beginPath();
    context.arc(x, y, node === circuit.probe && tick >= replayEnd && signalLevel(3) > 0 ? 6 : 4, 0, Math.PI * 2);
    context.fill();
    context.font = '11px system-ui, sans-serif';
    context.textAlign = x > width * .7 ? 'right' : 'left';
    context.fillText(label, x + (context.textAlign === 'right' ? -8 : 8), y - 7);
  }
}

function renderReplay() {
  const displayedTick = Math.floor(tick);
  timeline.value = String(displayedTick);
  element('clock').value = getCircuit() ? 'Relative tick ' + displayedTick.toLocaleString('en-US') : '';
  element('back').disabled = timeline.disabled || displayedTick <= replayStart;
  element('next').disabled = timeline.disabled || displayedTick >= replayEnd;
  draw();
}

function pause() {
  if (animation) cancelAnimationFrame(animation);
  animation = 0;
  previousTime = 0;
  element('play').textContent = 'Play';
}

function play(now) {
  const rate = Math.max(1, replayEnd - replayStart) / 8 * Number(element('speed').value);
  if (previousTime) tick = Math.min(replayEnd, tick + Math.min(now - previousTime, 100) * rate / 1000);
  previousTime = now;
  renderReplay();
  if (tick >= replayEnd) pause();
  else animation = requestAnimationFrame(play);
}

function seek(value) {
  pause();
  tick = Math.max(replayStart, Math.min(replayEnd, value));
  renderReplay();
}

function addBox(row, style, label) {
  if (!Array.isArray(row.box) || row.box.length !== 4) return;
  const [x, y, width, height] = row.box;
  if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return;
  const group = document.createElementNS(namespace, 'g');
  group.setAttribute('class', style);
  const rectangle = document.createElementNS(namespace, 'rect');
  for (const [name, value] of Object.entries({x, y, width, height})) rectangle.setAttribute(name, value);
  const caption = document.createElementNS(namespace, 'text');
  caption.setAttribute('x', x + 3);
  const labelY = style === 'annotation' ? Math.min(frame.height - 4, y + height - 5) : Math.max(y - 4, 18);
  caption.setAttribute('y', labelY);
  caption.textContent = label;
  group.append(rectangle, caption);
  svg.append(group);
}

function renderBoxes() {
  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 ' + frame.width + ' ' + frame.height);
  if (element('show-truth').checked) {
    for (const row of frame.truth) addBox(row, 'annotation', row.part ? 'Target ' + row.part : 'Target');
    for (const row of frame.decoys ?? []) addBox(row, 'false-alarm', 'Decoy');
  }
  for (const row of frame[modeSelect.value]?.detections ?? []) {
    addBox(row, 'detection', (row.part ? 'Part ' + row.part + ' · ' : '') + formatNumber(row.score));
  }
}

function setPreview(visible) {
  element('scene').hidden = !visible;
  element('no-preview').hidden = visible;
}

function renderScene() {
  pause();
  const mode = modeSelect.value;
  const run = frame[mode];
  element('scene-meta').textContent = [frame.id, ...(frame.truth.length ? [] : ['Target absent'])].join(' · ');
  element('credit').hidden = frame.domain !== 'real';
  if (frame.preview) {
    const source = new URL('photo-data/' + encodeURI(frame.preview), document.baseURI).href;
    photo.alt = 'Input for scene ' + frame.id;
    if (photo.src !== source) {
      setPreview(true);
      photo.src = source;
    } else {
      setPreview(!photo.complete || photo.naturalWidth > 0);
    }
  } else {
    photo.removeAttribute('src');
    setPreview(false);
  }
  renderBoxes();
  const count = run?.detections?.length ?? 0;
  const details = run?.best
    ? ' Highest score ' + formatNumber(run.best.score) + '.'
    : '';
  element('verdict').textContent = !run ? 'No measurements for this mechanism in the selected scene.'
    : (count ? count + ' accepted proposal' + (count === 1 ? '.' : 's.') : 'No proposal accepted.') + details;
  const circuit = getCircuit();
  canvas.hidden = !circuit;
  const empty = element('no-graph');
  empty.hidden = Boolean(circuit);
  empty.textContent = mode === 'direct' ? 'Direct calculation has no signal path.'
    : 'No captured signal path for this mechanism.';
  element('phase').textContent = circuit ? 'Drag to rotate' : '';
  element('tracked').textContent = circuit && trackedBest()
    ? (mode === 'cut' ? 'Following the same pose as the intact sphere' : 'Following the highest-scoring pose')
    : circuit ? 'No pose received enough agreeing votes to replay.' : '';
  const legend = element('legend');
  legend.replaceChildren();
  if (circuit && trackedBest()) {
    for (const route of circuit.routes) {
      const entry = document.createElement('li');
      entry.style.color = signalColors[route.group];
      const launched = signalLevel(route.group);
      const arrived = mode === 'cut' && route.group < 3 ? run?.atSphereBest?.[route.group] : undefined;
      entry.textContent = labelRoute(route) + ' ' + formatNumber(launched) +
        (Number.isFinite(arrived) && arrived !== launched ? ' → ' + formatNumber(arrived) : '');
      legend.append(entry);
    }
  }
  const hops = circuit?.routes.flatMap(route => route.hops) ?? [];
  replayStart = hops.length ? Math.min(...hops.map(hop => hop.start)) : 0;
  replayEnd = hops.length ? Math.max(...hops.map(hop => hop.end), circuit.duration ?? 0) : 0;
  timeline.min = String(replayStart);
  timeline.max = String(replayEnd);
  timeline.disabled = !hops.length || !trackedBest();
  element('play').disabled = timeline.disabled;
  element('speed').disabled = timeline.disabled;
  tick = replayStart;
  renderReplay();
  renderSummary();
}

function appendCell(row, kind, value) {
  const cell = document.createElement(kind);
  if (kind === 'th') cell.scope = row.parentElement.tagName === 'THEAD' ? 'col' : 'row';
  cell.textContent = value;
  row.append(cell);
}

function renderSummary() {
  const panel = data.summary?.panels?.[frame.split];
  const mechanism = data.summary?.mechanism;
  element('results').hidden = !panel;
  if (!panel) return;
  const head = element('summary-head');
  const rows = element('summary-rows');
  head.replaceChildren();
  rows.replaceChildren();
  const heading = head.insertRow();
  for (const label of ['Mechanism', 'Targets found at cutoff', 'False alarms at cutoff', 'Targets in top five',
    'False top-five proposals']) {
    appendCell(heading, 'th', label);
  }
  for (const [mode, label] of Object.entries(modes)) {
    const measured = panel[mode];
    if (!measured) continue;
    const row = rows.insertRow();
    if (mode === modeSelect.value) row.setAttribute('aria-current', 'true');
    appendCell(row, 'th', label);
    appendCell(row, 'td', measured.hits + ' / ' + panel.targets);
    appendCell(row, 'td', formatNumber(measured.falseAlarms));
    appendCell(row, 'td', measured.topFiveHits + ' / ' + panel.targets);
    appendCell(row, 'td', formatNumber(measured.topFiveFalse));
  }
  element('summary-split').textContent = splits[frame.split] ?? frame.split;
  const missing = panel.targetsWithoutFeatures;
  element('summary-note').textContent = panel.frames + ' scenes. ' + (panel.threshold === null
    ? 'The calibration cutoff rejects every proposal.'
    : 'All mechanisms share the calibration cutoff of ' + formatNumber(panel.threshold) + '.') +
    (missing > 0 ? ' The extractor found no local feature in ' + missing + ' of the ' + panel.targets + ' target boxes.' : '');
  element('summary-agreement').textContent = mechanism ?
    'Across all ' + mechanism.frames + ' scenes, the full sphere matched the direct vote maps in ' + mechanism.sameDirect +
    ' and the feed-forward maps in ' + mechanism.sameFeedForward + '. Cutting one route changed the maps in ' +
    mechanism.changedByCut + ' scenes. Delaying it by one tick changed them in ' + mechanism.changedByDelay + '.' : '';
}

function loadExperiment(source) {
  if (!Array.isArray(source.frames) || !source.frames.length) throw new Error('No measured scenes were found.');
  data = source;
  for (const [index, item] of data.frames.entries()) {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = [item.id, item.domain === 'real' ? 'T-LESS photo' : 'Synthetic control',
      item.truth.length ? 'Target present' : 'Target absent'].join(' · ');
    sceneSelect.append(option);
  }
  for (const option of modeSelect.options) {
    option.disabled = !data.frames.some(item => item[option.value]);
  }
  const preferred = data.frames.findIndex(item => item.split === 'control-test' && item.preview && item.truth.length && item.decoys?.length &&
    item.sphere?.best);
  const firstPreview = data.frames.findIndex(item => item.preview);
  sceneSelect.value = String(Math.max(0, preferred >= 0 ? preferred : firstPreview));
  frame = data.frames[Number(sceneSelect.value)];
  element('status').hidden = true;
  element('experiment').hidden = false;
  renderScene();
}

sceneSelect.addEventListener('change', () => {
  frame = data.frames[Number(sceneSelect.value)];
  renderScene();
});
modeSelect.addEventListener('change', renderScene);
element('show-truth').addEventListener('change', renderBoxes);
photo.addEventListener('error', () => setPreview(false));
timeline.addEventListener('input', () => seek(Number(timeline.value)));
element('back').addEventListener('click', () => seek(Math.floor(tick) - 1));
element('next').addEventListener('click', () => seek(Math.floor(tick) + 1));
element('play').addEventListener('click', () => {
  if (animation) {
    pause();
    return;
  }
  if (tick >= replayEnd) seek(replayStart);
  element('play').textContent = 'Pause';
  animation = requestAnimationFrame(play);
});
canvas.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  canvas.setPointerCapture(event.pointerId);
  drag = {id: event.pointerId, x: event.clientX, y: event.clientY};
});
canvas.addEventListener('pointermove', event => {
  if (drag?.id !== event.pointerId) return;
  angleX += (event.clientX - drag.x) * .01;
  angleY = Math.min(1.45, Math.max(-1.45, angleY + (event.clientY - drag.y) * .01));
  drag = {id: event.pointerId, x: event.clientX, y: event.clientY};
  draw();
});
canvas.addEventListener('lostpointercapture', () => { drag = null; });
new ResizeObserver(draw).observe(canvas);

fetch('photo-data/pose-consensus.json')
  .then(response => {
    if (!response.ok) throw new Error('Measurements could not be loaded.');
    return response.json();
  })
  .then(loadExperiment)
  .catch(error => { element('status').textContent = error.message; });
