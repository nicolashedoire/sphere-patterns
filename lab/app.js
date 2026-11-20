'use strict';

const byId = id => document.getElementById(id);
const sphere = byId('sphere');
const context = sphere.getContext('2d');
const settings = byId('settings');
const timeline = byId('timeline');
const play = byId('play');
const back = byId('back');
const next = byId('next');
const speed = byId('speed');
const status = byId('run-status');
const events = byId('events');

let geometry = null;
let clock = null;
let frames = [];
let position = 0;
let worker = null;
let playing = false;
let animation = 0;
let redraw = 0;
let currentTime = 0;
let lastTime = 0;
let yaw = -0.65;
let pitch = 0.32;
let zoom = 1;
let drag = null;

function project(point, centerX, centerY, size, rotation) {
  const [x, y, z] = point;
  const { cosY, sinY, cosP, sinP } = rotation;
  const horizontal = x * cosY + z * sinY;
  const depth = z * cosY - x * sinY;
  const vertical = y * cosP - depth * sinP;
  const facing = depth * cosP + y * sinP;
  const perspective = size / (1 - facing * 0.16);
  return [centerX + horizontal * perspective, centerY + vertical * perspective, facing];
}

function draw() {
  redraw = 0;
  const { width, height } = sphere.getBoundingClientRect();
  if (!width || !height) return;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const targetWidth = Math.round(width * pixelRatio);
  const targetHeight = Math.round(height * pixelRatio);
  if (sphere.width !== targetWidth || sphere.height !== targetHeight) {
    sphere.width = targetWidth;
    sphere.height = targetHeight;
  }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);

  const cx = width / 2, cy = height * 0.49;
  const radius = Math.min(width, height) * 0.37 * zoom;
  context.beginPath();
  context.arc(cx, cy, radius, 0, Math.PI * 2);
  context.strokeStyle = '#4da5c832';
  context.lineWidth = 1;
  context.stroke();
  if (!geometry) return;

  const rotation = { cosY: Math.cos(yaw), sinY: Math.sin(yaw), cosP: Math.cos(pitch), sinP: Math.sin(pitch) };
  const points = geometry.nodes.map(point => project(point, cx, cy, radius, rotation));
  context.lineWidth = 0.65;
  for (const [front, tone] of [[false, '#7aa8d222'], [true, '#7eb4dc56']]) {
    context.beginPath();
    for (const [a, b] of geometry.edges) {
      const from = points[a], to = points[b];
      if ((from[2] + to[2] >= 0) !== front) continue;
      context.moveTo(from[0], from[1]);
      context.lineTo(to[0], to[1]);
    }
    context.strokeStyle = tone;
    context.stroke();
  }
  context.fillStyle = '#a1c7dd70';
  for (const [x, y, depth] of points) {
    const size = depth >= 0 ? 1.5 : 0.9;
    context.fillRect(x, y, size, size);
  }

  const samples = frames[position]?.samples;
  if (samples) {
    context.globalCompositeOperation = 'lighter';
    for (let i = 0; i < samples.length; i += 3) {
      const [a, b] = geometry.arcs[samples[i]];
      const from = geometry.nodes[a], to = geometry.nodes[b], offset = samples[i + 1], amplitude = samples[i + 2];
      const point = from.map((value, axis) => value + (to[axis] - value) * offset);
      const [x, y, depth] = project(point, cx, cy, radius, rotation);
      const color = amplitude < 0 ? '#c78bff' : '#63e7ff';
      context.globalAlpha = Math.min(0.92, 0.34 + Math.abs(amplitude)) * (depth >= 0 ? 1 : 0.68);
      context.shadowColor = color;
      context.shadowBlur = 10;
      context.fillStyle = color;
      context.beginPath();
      context.arc(x, y, 1.4 + 2.1 * Math.abs(amplitude), 0, Math.PI * 2);
      context.fill();
    }
    context.globalAlpha = 1;
    context.shadowBlur = 0;
    context.globalCompositeOperation = 'source-over';
  }

  const marker = (node, color, label) => {
    const [x, y] = points[node];
    context.beginPath();
    context.arc(x, y, 4, 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();
    context.beginPath();
    context.arc(x, y, 7, 0, Math.PI * 2);
    context.strokeStyle = color;
    context.stroke();
    context.font = '11px system-ui, sans-serif';
    context.textAlign = x > width * 0.64 ? 'right' : 'left';
    context.fillText(label, x + (x > width * 0.64 ? -11 : 11), y - 7);
  };
  marker(geometry.junction, '#a6f5ff', 'Junction');
  marker(geometry.probe, '#fabe72', 'Probe');
}

function invalidate() {
  if (!redraw) redraw = requestAnimationFrame(draw);
}

function eventList() {
  if (!clock) return [];
  return [
    ['Inputs meet', clock.collected],
    ['Reference emitted', clock.emitted],
    ['Reference reaches probe', clock.referenceAtProbe],
    ['Reference returns', clock.returned],
    ['Result window opens', clock.output],
    ['Probe result window opens', clock.outputAtProbe]
  ];
}

function renderTransport() {
  const tick = frames[position]?.tick ?? 0;
  byId('clock').value = 'Tick ' + tick.toLocaleString('en-US');
  timeline.value = String(position);
  timeline.setAttribute('aria-valuetext', byId('clock').value);
  back.disabled = !frames.length || position <= 0;
  next.disabled = !frames.length || position >= frames.length - 1;
  play.textContent = playing ? 'Pause' : 'Play';
  const list = eventList();
  const nearest = list.findLastIndex(([, at]) => at <= tick);
  const label = nearest < 0 ? 'Ready to launch' : list[nearest][0];
  byId('phase').value = label;
  for (let i = 0; i < events.children.length; i++) {
    if (i === nearest) events.children[i].setAttribute('aria-current', 'true');
    else events.children[i].removeAttribute('aria-current');
  }
  invalidate();
}

function stopPlayback() {
  playing = false;
  if (animation) cancelAnimationFrame(animation);
  animation = 0;
  play.textContent = 'Play';
}

function showFrame(index) {
  position = Math.max(0, Math.min(frames.length - 1, index));
  currentTime = frames[position]?.tick ?? 0;
  renderTransport();
}

function seekTick(tick) {
  let low = 0, high = frames.length - 1;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (frames[mid].tick < tick) low = mid + 1;
    else high = mid;
  }
  stopPlayback();
  showFrame(low);
}

function animate(now) {
  if (!playing) return;
  if (lastTime) currentTime += Math.min(100, now - lastTime) * Number(speed.value) / 1000;
  lastTime = now;
  const previous = position;
  while (position + 1 < frames.length && frames[position + 1].tick <= currentTime) position++;
  if (position >= frames.length - 1) stopPlayback();
  if (position !== previous) renderTransport();
  if (playing) animation = requestAnimationFrame(animate);
}

function togglePlayback() {
  if (!frames.length) return;
  if (playing) {
    stopPlayback();
    return;
  }
  if (position >= frames.length - 1) showFrame(0);
  currentTime = frames[position].tick;
  lastTime = 0;
  playing = true;
  play.textContent = 'Pause';
  animation = requestAnimationFrame(animate);
}

function showMask(id, pixels, side, ink, selection) {
  const canvas = byId(id);
  const scale = 5;
  canvas.width = side * scale;
  canvas.height = side * scale;
  const graphics = canvas.getContext('2d');
  graphics.fillStyle = '#0a1323';
  graphics.fillRect(0, 0, canvas.width, canvas.height);
  if (!pixels) return;
  graphics.fillStyle = ink;
  pixels.forEach((value, index) => {
    if (value) graphics.fillRect((index % side) * scale, Math.floor(index / side) * scale, scale, scale);
  });
  if (!selection) return;
  const [x, y] = selection.center.map(value => (value + 0.5) * scale);
  graphics.beginPath();
  graphics.arc(x, y, 2.4, 0, Math.PI * 2);
  graphics.strokeStyle = '#ffcd85';
  graphics.lineWidth = 1.5;
  graphics.stroke();
}

function showCompleted(message) {
  frames = message.frames;
  timeline.max = String(frames.length - 1);
  timeline.disabled = false;
  play.disabled = false;

  for (const [label, tick] of eventList()) {
    const button = document.createElement('button');
    button.textContent = label;
    button.type = 'button';
    button.addEventListener('click', () => seekTick(tick));
    events.append(button);
  }

  const side = Math.sqrt(message.reference.length);
  showMask('reference', message.reference, side, '#83edfb');
  showMask('matched', message.matched, side, '#fabe72');
  showMask('scene', message.scene, side, '#9cc2d9', message.selection);
  const verdict = byId('verdict');
  verdict.classList.toggle('found', !!message.selection);
  verdict.textContent = message.options.disconnected ? 'No match · reference disconnected' :
    message.selection ? 'Match at the scene marker' : 'No accepted match';
  byId('loop').textContent = (clock.returned - clock.emitted).toLocaleString('en-US') + ' ticks';
  byId('overlap').textContent = message.matches + ' / ' + message.referencePixels;
  byId('detected-angle').textContent = message.selection ? message.selection.degrees + '°' : '—';
  byId('energy').textContent = message.energyError === 0 ? '0' : message.energyError.toExponential(2);
  status.textContent = 'Run complete';
  showFrame(0);
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) togglePlayback();
}

function startRun() {
  stopPlayback();
  worker?.terminate();
  worker = null;
  clock = null;
  frames = [];
  position = 0;
  events.replaceChildren();
  for (const control of [play, back, next, timeline]) control.disabled = true;
  byId('run').disabled = true;
  byId('run').textContent = 'Running…';
  byId('verdict').classList.remove('found');
  byId('verdict').textContent = 'Computing the probe signal';
  status.classList.remove('is-error');
  status.textContent = 'Starting run…';
  byId('phase').value = 'Computing';
  byId('clock').value = 'Tick 0';
  timeline.value = '0';
  for (const field of ['loop', 'overlap', 'detected-angle', 'energy']) byId(field).textContent = '—';
  for (const id of ['reference', 'matched']) showMask(id, null, 25);

  const form = new FormData(settings);
  const options = {
    shape: form.get('shape'),
    search: form.get('search'),
    angle: Number(form.get('angle')),
    x: Number(form.get('x')),
    y: Number(form.get('y')),
    target: form.has('target'),
    decoy: form.has('decoy'),
    disconnected: form.has('cut')
  };
  let scene;
  const finish = () => {
    worker?.terminate();
    worker = null;
    byId('run').disabled = false;
    byId('run').textContent = 'Run simulation';
  };
  const failure = message => {
    finish();
    status.classList.add('is-error');
    status.textContent = 'Error · ' + message;
    byId('phase').value = 'Run interrupted';
    byId('verdict').textContent = 'No probe readings';
  };

  try {
    worker = new Worker('worker.js');
  } catch (error) {
    failure(error.message);
    return;
  }
  worker.onerror = event => failure(event.message || 'The simulator could not start.');
  worker.onmessage = ({ data }) => {
    if (data.type === 'start') {
      geometry = data.geometry;
      clock = data.milestones;
      scene = data.scene;
      byId('network').textContent = geometry.nodes.length.toLocaleString('en-US') + ' nodes · ' +
        geometry.edges.length.toLocaleString('en-US') + ' links';
      showMask('view-a', data.viewA, data.size, '#a7c3d7');
      showMask('view-b', data.viewB, data.size, '#a7c3d7');
      showMask('scene', scene, data.size, '#9cc2d9');
      invalidate();
    } else if (data.type === 'progress') {
      status.textContent = 'Simulating · ' + Math.round(100 * data.tick / data.end) + '%';
      byId('clock').value = 'Tick ' + data.tick.toLocaleString('en-US');
    } else if (data.type === 'result') {
      finish();
      showCompleted({ ...data, scene });
    } else if (data.type === 'error') {
      failure(data.message);
    }
  };
  worker.postMessage(options);
}

timeline.addEventListener('input', () => {
  stopPlayback();
  showFrame(Number(timeline.value));
});
play.addEventListener('click', togglePlayback);
back.addEventListener('click', () => { stopPlayback(); showFrame(position - 1); });
next.addEventListener('click', () => { stopPlayback(); showFrame(position + 1); });
settings.addEventListener('submit', event => {
  event.preventDefault();
  startRun();
});

for (const id of ['angle', 'x', 'y']) {
  byId(id).addEventListener('input', event => {
    const value = Number(event.target.value);
    byId(id + '-value').value = id === 'angle' ? value + '°' : value + ' cells';
  });
}

function orbit(dx, dy) {
  yaw += dx;
  pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch + dy));
  invalidate();
}

sphere.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  drag = [event.clientX, event.clientY];
  sphere.setPointerCapture(event.pointerId);
});
sphere.addEventListener('pointermove', event => {
  if (!drag) return;
  orbit((event.clientX - drag[0]) / 150, (event.clientY - drag[1]) / 150);
  drag = [event.clientX, event.clientY];
});
sphere.addEventListener('lostpointercapture', () => { drag = null; });
sphere.addEventListener('wheel', event => {
  event.preventDefault();
  zoom = Math.max(0.55, Math.min(2.4, zoom * Math.exp(-0.0016 * event.deltaY)));
  invalidate();
}, { passive: false });
sphere.addEventListener('keydown', event => {
  const offsets = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.12], ArrowDown: [0, 0.12] };
  if (!offsets[event.key]) return;
  event.preventDefault();
  orbit(...offsets[event.key]);
});

new ResizeObserver(invalidate).observe(sphere);
startRun();
