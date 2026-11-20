'use strict';

const get = name => document.getElementById('neural-' + name);
const format = new Intl.NumberFormat('en-US', {maximumFractionDigits: 4});
const value = number => Number.isFinite(number) ? format.format(number) : '—';
const svgns = 'http://www.w3.org/2000/svg';
const photo = get('photo');
const overlay = get('overlay');
const sphere = get('sphere');
const linkedScene = new URLSearchParams(location.search).get('scene');
const methods = {
  sift: 'SIFT baseline',
  coarse: 'Neural first pass',
  fixed: 'Fixed crops',
  targeted: 'Hypothesis-directed crops',
  wave: 'Hypothesis-directed crops · sphere',
  cut: 'Sphere with return cut',
};
const splitLabels = {
  'control-dev': 'Synthetic calibration',
  'control-test': 'Synthetic evaluation',
  'control-validation': 'Synthetic validation',
  development: 'T-LESS calibration',
  heldout: 'T-LESS evaluation replay',
  confirmation: 'T-LESS extra replay views',
};
const stages = [
  ['coarse', 'First pass', 'The full-frame pass supplies the starting pose candidates.'],
  ['request', 'Request crops', 'The selected method marks the regions it rereads at full resolution.'],
  ['result', 'Read evidence', 'The measured reread evidence updates the ranking.'],
];
let data;
let frame;
let stage = 'result';
let trace;
let tick = 0;
let scheduled = 0;
let lastTime = 0;
let rotationX = -.5;
let rotationY = .4;
let drag;

function item(tag, content, className) {
  const element = document.createElement(tag);
  element.textContent = content;
  if (className) element.className = className;
  return element;
}

function options(select, rows) {
  select.replaceChildren(...rows.map(([key, label]) => new Option(label, key)));
}

function sceneLabel(row) {
  return row.label ?? [row.dataset ?? splitLabels[row.split], row.id,
    row.targets?.length === 0 ? 'Target absent' : null].filter(Boolean).join(' · ');
}

function mark(box, className, label) {
  const [x, y, width, height] = box;
  const group = document.createElementNS(svgns, 'g');
  group.setAttribute('class', className);
  const rect = document.createElementNS(svgns, 'rect');
  for (const [key, number] of Object.entries({x, y, width, height})) rect.setAttribute(key, number);
  group.append(rect);
  if (label) {
    const text = document.createElementNS(svgns, 'text');
    text.setAttribute('x', x + 5);
    text.setAttribute('y', y + Math.min(20, height * .6));
    text.textContent = label;
    group.append(text);
  }
  overlay.append(group);
}

function cropPreview(crop, index, frame) {
  const figure = document.createElement('figure');
  const [x, y, width, height] = crop.box;
  const canvas = document.createElement('canvas');
  const scaleX = photo.naturalWidth / frame.width;
  const scaleY = photo.naturalHeight / frame.height;
  canvas.width = 128;
  canvas.height = Math.max(1, Math.round(128 * height / width));
  canvas.getContext('2d').drawImage(photo, x * scaleX, y * scaleY, width * scaleX,
    height * scaleY, 0, 0, canvas.width, canvas.height);
  canvas.setAttribute('aria-hidden', 'true');
  const matched = Number.isFinite(crop.match) ? ` · match ${value(crop.match)}` : '';
  figure.append(canvas, item('figcaption', `Crop ${index + 1} · ${value(width)} × ${value(height)} px${matched}`));
  return figure;
}

function updateImage(frame, candidates, crops) {
  overlay.setAttribute('viewBox', `0 0 ${frame.width} ${frame.height}`);
  photo.style.aspectRatio = `${frame.width} / ${frame.height}`;
  overlay.replaceChildren();
  if (get('show-truth').checked) {
    frame.targets.forEach(target => mark(target.box, 'annotation', 'Target'));
  }
  candidates.slice(0, 5).forEach((candidate, index) => mark(candidate.box, 'detection', String(index + 1)));
  crops.forEach((crop, index) => mark(crop.box, 'neural-crop', `C${index + 1}`));

  const stageElement = get('stage');
  const message = get('image-status');
  const cropList = get('crops');
  const src = frame.image ? new URL(frame.image, new URL('photo-data/', location.href)).href : null;
  stageElement.hidden = !src;
  message.hidden = Boolean(src);
  if (!src) message.textContent = 'Measurements are available for this scene without a preview image.';
  cropList.hidden = true;
  cropList.replaceChildren();
  if (!src) return;
  photo.alt = `Input image for ${sceneLabel(frame)}`;
  if (photo.src !== src) photo.src = src;
  if (photo.complete && photo.naturalWidth && crops.length) {
    cropList.replaceChildren(...crops.map((crop, index) => cropPreview(crop, index, frame)));
    cropList.hidden = false;
  }
}

function rank(candidates) {
  return new Map(candidates.map((candidate, index) => [candidate.id, {...candidate, place: index + 1}]));
}

function rankCell(candidate) {
  return candidate ? `#${candidate.place} · ${value(candidate.score)}` : '—';
}

function updateRanking(coarse, after) {
  const startRanks = rank(coarse);
  const endRanks = rank(after);
  const ids = [...new Set([...endRanks.keys(), ...startRanks.keys()])];
  const rows = ids.map(id => {
    const row = document.createElement('tr');
    const first = startRanks.get(id);
    const last = endRanks.get(id);
    const title = item('th', first?.label ?? last?.label ?? `Pose ${first?.place ?? last?.place}`);
    title.scope = 'row';
    row.dataset.first = (last?.place ?? first?.place) === 1;
    row.append(title, item('td', rankCell(first)), item('td', rankCell(last)));
    return row;
  });
  if (!rows.length) {
    const row = document.createElement('tr');
    const cell = item('td', 'No pose candidates recorded.');
    cell.colSpan = 3;
    row.append(cell);
    rows.push(row);
  }
  get('ranking').replaceChildren(...rows);
}

function showReadout(fields) {
  get('readout-section').hidden = !fields.length;
  get('readout').replaceChildren(...fields.map(([label, reading]) => {
    const group = document.createElement('div');
    group.append(item('dt', label), item('dd', Number.isFinite(reading) ? value(reading) : reading));
    return group;
  }));
}

function summaryTable(headers, records) {
  const holder = document.createElement('div');
  holder.className = 'neural-table';
  const table = document.createElement('table');
  const head = table.createTHead().insertRow();
  for (const header of headers) {
    const cell = item('th', header);
    cell.scope = 'col';
    head.append(cell);
  }
  const body = table.createTBody();
  for (const [label, ...fields] of records) {
    const row = body.insertRow();
    const cell = item('th', label);
    cell.scope = 'row';
    row.append(cell, ...fields.map(field => item('td', Number.isFinite(field) ? value(field) : field ?? '—')));
  }
  holder.append(table);
  return holder;
}

function selectStage(key) {
  stage = key;
  render();
}

function setupStages() {
  get('steps').replaceChildren(...stages.map(([key, label]) => {
    const button = item('button', label);
    button.type = 'button';
    button.dataset.stage = key;
    button.addEventListener('click', () => selectStage(key));
    return button;
  }));
}

function renderSummary() {
  const {metadata = {}, summaries = []} = data;
  const protocol = get('protocol');
  get('encoder').textContent = metadata.encoder ?? metadata.model ?? data.model ?? '';
  protocol.textContent = metadata.protocol ?? '';
  protocol.hidden = !protocol.textContent;
  const sections = summaries.map(summary => {
    const section = document.createElement('section');
    const hits = (count, total) => Number.isFinite(count) && Number.isFinite(total)
      ? `${value(count)} / ${value(total)}` : count;
    const comparisons = summary.rows.map(row => [methods[row.arm] ?? row.arm,
      row.frames, hits(row.topOneHits, row.targetCount), hits(row.topFiveHits, row.targetCount),
      hits(row.matches, row.targetCount), row.falseAlarms, row.meanPixels]);
    section.append(item('h3', summary.label), summaryTable([
      'Method', 'Scenes', 'Rank 1 hits', 'Top 5 hits', 'Accepted matches', 'False alarms', 'Mean reread pixels',
    ], comparisons));
    return section;
  });
  get('summaries').replaceChildren(...sections);
  get('results').hidden = !sections.length;
  const parity = get('parity');
  parity.textContent = typeof metadata.wordParity === 'boolean' && Number.isFinite(metadata.framesMatched)
    ? `Direct and wave scores matched in ${value(metadata.framesMatched)} scenes. ` +
      (Number.isFinite(metadata.cutReads) ? `The cut return path yielded ${value(metadata.cutReads)} rereads.` : '') : '';
  parity.hidden = !parity.textContent;
}

function stopPlayback() {
  cancelAnimationFrame(scheduled);
  scheduled = 0;
  get('play').textContent = 'Play';
}

function drawTrace() {
  if (!trace) return;
  const nodes = trace.nodes ?? data.topology?.nodes;
  const edges = trace.edges ?? data.topology?.edges;
  if (!nodes || !edges) return;
  const {width, height} = sphere.getBoundingClientRect();
  if (!width || !height) return;
  const ratio = Math.min(window.devicePixelRatio, 2);
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);
  if (sphere.width !== pixelWidth || sphere.height !== pixelHeight) {
    sphere.width = pixelWidth;
    sphere.height = pixelHeight;
  }
  const context = sphere.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.globalAlpha = 1;
  context.lineWidth = 1;
  const radius = Math.min(width, height) * .43;
  const cosX = Math.cos(rotationX);
  const sinX = Math.sin(rotationX);
  const cosY = Math.cos(rotationY);
  const sinY = Math.sin(rotationY);
  const points = nodes.map(([x, y, z]) => {
    const horizontal = x * cosY + z * sinY;
    const depth = z * cosY - x * sinY;
    const vertical = y * cosX - depth * sinX;
    const perspective = radius / (1 - (depth * cosX + y * sinX) * .12);
    return [width / 2 + horizontal * perspective, height / 2 + vertical * perspective];
  });
  context.beginPath();
  for (const [from, to] of edges) {
    context.moveTo(...points[from]);
    context.lineTo(...points[to]);
  }
  context.strokeStyle = '#75bbd553';
  context.stroke();

  const colors = {request: '#67e8f9', measurement: '#c495ff', readout: '#fabe72'};
  for (const {from, to, start, end, role} of trace.hops) {
    if (tick < start) continue;
    const [fromX, fromY] = points[from];
    const [toX, toY] = points[to];
    const fraction = end > start ? Math.min(1, (tick - start) / (end - start)) : 1;
    const x = fromX + (toX - fromX) * fraction;
    const y = fromY + (toY - fromY) * fraction;
    context.strokeStyle = colors[role] ?? '#eef2ff';
    context.globalAlpha = tick > end ? .34 : 1;
    context.lineWidth = 2.3;
    context.beginPath();
    context.moveTo(fromX, fromY);
    context.lineTo(x, y);
    context.stroke();
    if (tick <= end) {
      context.beginPath();
      context.arc(x, y, 3.5, 0, 2 * Math.PI);
      context.fillStyle = context.strokeStyle;
      context.fill();
    }
  }
  get('tick').value = Math.floor(tick);
  get('clock').textContent = `Tick ${value(Math.floor(tick))}`;
}

function animate(time) {
  const speed = Number(get('speed').value);
  tick = Math.min(trace.duration, tick + (time - lastTime) * trace.duration * speed / 4200);
  lastTime = time;
  drawTrace();
  if (tick < trace.duration) scheduled = requestAnimationFrame(animate);
  else stopPlayback();
}

function loadTrace(next) {
  if (trace !== next) {
    stopPlayback();
    trace = next;
    tick = 0;
  }
  get('trace').hidden = !trace?.hops?.length || !(trace.nodes ?? data.topology?.nodes);
  if (get('trace').hidden) return;
  get('tick').max = trace.duration;
  drawTrace();
}

function render() {
  if (!frame) return;
  const method = get('arm').value;
  const arm = frame.arms[method];
  const coarse = frame.coarse;
  const after = arm.candidates;
  get('scene-meta').textContent = sceneLabel(frame);
  get('budget').textContent = `${value(arm.pixels)} reread pixels`;
  get('stage-note').textContent = stages.find(([key]) => key === stage)[2];
  for (const button of get('steps').children) button.setAttribute('aria-pressed', button.dataset.stage === stage);
  updateImage(frame, stage === 'result' ? after : coarse, stage === 'coarse' ? [] : arm.crops);
  updateRanking(coarse, stage === 'result' ? after : []);
  const count = arm.detectedIds?.length;
  get('verdict').textContent = stage === 'result' && Number.isFinite(count) ? `${count} accepted ${count === 1 ? 'candidate' : 'candidates'}` : '';

  const fields = [];
  if (stage === 'result' && frame.waveCheck) {
    const {sameRequests, sameScores, cutReads} = frame.waveCheck;
    if (typeof sameRequests === 'boolean') fields.push(['Requests', sameRequests ? 'Identical' : 'Different']);
    if (typeof sameScores === 'boolean') fields.push(['Scores', sameScores ? 'Identical' : 'Different']);
    if (Number.isFinite(cutReads)) fields.push(['Rereads after cut', cutReads]);
    if (method === 'targeted' || method === 'wave') {
      const leading = frame.arms.wave?.candidates[0]?.id;
      const direct = frame.arms.targeted?.crops.find(crop => crop.candidateId === leading);
      const wave = frame.arms.wave?.crops.find(crop => crop.candidateId === leading);
      if (Number.isFinite(direct?.signal)) fields.push(['Direct evidence, rank 1', direct.signal]);
      if (Number.isFinite(wave?.signal)) fields.push(['Probe evidence, rank 1', wave.signal]);
    }
  }
  showReadout(fields);
  loadTrace(arm.trace ?? (method === 'wave' ? frame.trace : undefined));
  get('trace-caption').textContent = method === 'cut'
    ? 'The request stops at the cut link.' : 'Recorded request and response for the leading wave candidate.';
}

function selectScene() {
  const previous = get('arm').value;
  frame = data.frames.find(row => String(row.id) === get('scene').value);
  const rows = Object.keys(frame.arms).map(key => [key, methods[key] ?? key]);
  options(get('arm'), rows);
  if (frame.arms[previous]) get('arm').value = previous;
  else if (frame.arms.wave) get('arm').value = 'wave';
  else if (frame.arms.targeted) get('arm').value = 'targeted';
  render();
}

get('scene').addEventListener('change', selectScene);
get('arm').addEventListener('change', render);
get('show-truth').addEventListener('change', render);
photo.addEventListener('load', render);
photo.addEventListener('error', () => {
  get('stage').hidden = true;
  get('image-status').hidden = false;
  get('image-status').textContent = 'The preview image could not be loaded.';
});
get('tick').addEventListener('input', event => {
  stopPlayback();
  tick = Number(event.target.value);
  drawTrace();
});
get('play').addEventListener('click', () => {
  if (scheduled) {
    stopPlayback();
    return;
  }
  if (tick >= trace.duration) tick = 0;
  lastTime = performance.now();
  get('play').textContent = 'Pause';
  scheduled = requestAnimationFrame(animate);
});
sphere.addEventListener('pointerdown', event => {
  drag = [event.clientX, event.clientY];
  sphere.setPointerCapture(event.pointerId);
});
sphere.addEventListener('pointermove', event => {
  if (!drag) return;
  rotationY += (event.clientX - drag[0]) * .01;
  rotationX += (event.clientY - drag[1]) * .01;
  drag = [event.clientX, event.clientY];
  drawTrace();
});
sphere.addEventListener('lostpointercapture', () => { drag = undefined; });
sphere.addEventListener('keydown', event => {
  if (!event.key.startsWith('Arrow')) return;
  event.preventDefault();
  if (event.key === 'ArrowRight') rotationY += .12;
  if (event.key === 'ArrowLeft') rotationY -= .12;
  if (event.key === 'ArrowUp') rotationX -= .12;
  if (event.key === 'ArrowDown') rotationX += .12;
  drawTrace();
});
new ResizeObserver(drawTrace).observe(sphere);

fetch('photo-data/neural-feedback.json').then(response => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then(measurements => {
  data = measurements;
  if (!data.frames.length) throw new Error('No scenes in result');
  options(get('scene'), data.frames.map(row => [row.id, sceneLabel(row)]));
  const realPhotos = data.frames.filter(row => row.image && (row.domain === 'real' || /T-LESS/i.test(row.dataset)));
  const firstPhoto = realPhotos.find(row => row.split === 'heldout' && row.targets.length) ?? realPhotos[0];
  const initial = data.frames.find(row => String(row.id) === linkedScene) ?? firstPhoto;
  if (initial) get('scene').value = initial.id;
  setupStages();
  get('experiment').hidden = false;
  get('status').hidden = true;
  get('credit').hidden = !firstPhoto;
  renderSummary();
  selectScene();
}).catch(error => {
  console.error(error);
  get('experiment').hidden = true;
  get('status').hidden = false;
  get('status').textContent = 'Measurements could not be loaded.';
});
