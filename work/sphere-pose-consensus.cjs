'use strict';

const { createNetwork } = require('./sphere-kernel.js');

const GROUPS = 3;
const AMPLITUDE = 1024;
const ANGLES = 24;
const SCALES = Array.from({ length: 13 }, (_, index) => 2 ** ((index - 6) / 4));
const SLOT = 4;

function pathFrom(network, start, excluded, depth) {
  const seen = new Set(excluded);
  const queue = [{ node: start, path: [] }];
  seen.add(start);
  for (const { node, path } of queue) {
    if (path.length === depth) return path;
    for (const port of network.nodePorts[node]) {
      const { to } = network.arcs[port];
      if (seen.has(to)) continue;
      seen.add(to);
      queue.push({ node: to, path: [...path, port] });
    }
  }
  return queue.at(-1).path;
}

function returnLoop(network, origin, blocked) {
  for (const entry of network.nodePorts[origin]) {
    const first = network.arcs[entry].to;
    if (blocked.has(first)) continue;
    const used = new Set([...blocked, origin, first]);
    const queue = [{ node: first, path: [entry] }];
    for (const { node, path } of queue) {
      for (const arcIndex of network.nodePorts[node]) {
        const arc = network.arcs[arcIndex];
        if (arc.to === origin && path.length > 1) return [...path, arcIndex];
        if (used.has(arc.to)) continue;
        used.add(arc.to);
        queue.push({ node: arc.to, path: [...path, arcIndex] });
      }
    }
  }
  throw new Error('No distinct return loop');
}

function createCircuit() {
  const network = createNetwork({ seed: 260617, nodeCount: 1000 });
  const { nodes, nodePorts, arcs, edges } = network;
  const squaredRadius = point => point.reduce((sum, axis) => sum + axis ** 2, 0);
  const mixer = nodes.reduce((best, point, index) => squaredRadius(point) < squaredRadius(nodes[best]) ? index : best, 0);
  const ports = Array.from(nodePorts[mixer]);
  const blocked = new Set([mixer, ...ports.map(port => arcs[port].to)]);
  const paths = ports.map((port, group) => {
    const branch = [port, ...pathFrom(network, arcs[port].to, blocked, group === GROUPS ? 19 : 10)];
    for (const arc of branch) blocked.add(arcs[arc].to);
    return group === GROUPS ? branch : branch.reverse().map(arc => arcs[arc].reverse);
  });
  const origin = arcs[paths[0][0]].from;
  const loop = returnLoop(network, origin, blocked);
  const cut = loop.at(-1);
  const recurrent = [[...loop, ...paths[0]], ...paths.slice(1)];
  const delay = path => path.reduce((total, port) => total + arcs[port].delay, 0);
  const arrival = Math.max(...recurrent.slice(0, GROUPS).map(delay));
  const timeline = chosen => chosen.map((path, group) => {
    let tick = group === GROUPS ? arrival : arrival - delay(path);
    return { group, hops: path.map(port => {
      const { from, to, delay: ticks } = arcs[port];
      const start = tick;
      tick += ticks;
      return { from, to, start, end: tick };
    }) };
  });
  const probe = arcs[paths[GROUPS].at(-1)].to;
  return { arcs, cut, recurrent, paths, delay, arrival, view: {
    nodes, edges: edges.map(({ a, b }) => [a, b]), mixer, probe,
    cut: [arcs[cut].from, arcs[cut].to], routes: timeline(recurrent), feedRoutes: timeline(paths),
    duration: arrival + delay(paths[GROUPS]),
  } };
}

function squaredDistance(a, b) {
  let distance = 0;
  for (let index = 0; index < a.length; index++) distance += (a[index] - b[index]) ** 2;
  return distance;
}

function localMatches(point, scene) {
  const maximum = 0.57 ** 2;
  const candidates = [];
  for (const sample of scene) {
    const scale = sample.size / point.size;
    if (scale < SCALES[0] || scale > SCALES.at(-1)) continue;
    const error = squaredDistance(point.d, sample.d);
    if (error <= maximum) candidates.push({ sample, scale, error });
  }
  candidates.sort((left, right) => left.error - right.error);
  const limit = Math.min(maximum, (candidates[0]?.error ?? maximum) * 1.65 + 0.025);
  const distinct = [];
  for (const candidate of candidates) {
    if (candidate.error > limit || distinct.length >= 6) break;
    if (distinct.some(({ sample }) => Math.hypot(sample.x - candidate.sample.x, sample.y - candidate.sample.y) < 3)) continue;
    distinct.push(candidate);
  }
  return distinct;
}

function neighbors(coordinate, length, wrap = false) {
  const lower = Math.floor(coordinate);
  return [lower, lower + 1].filter(index => wrap || (index >= 0 && index < length)).map(index => ({
    index: wrap ? (index + length) % length : index,
    weight: 1 - 0.30 * Math.abs(index - coordinate),
  }));
}

function rotatedBox(reference, x, y, scale, radians) {
  const [left, top, width, height] = reference.bounds;
  const cosine = Math.cos(radians) * scale;
  const sine = Math.sin(radians) * scale;
  const corners = [[left, top], [left + width, top], [left, top + height], [left + width, top + height]];
  const xs = corners.map(([a, b]) => x + a * cosine - b * sine);
  const ys = corners.map(([a, b]) => y + a * sine + b * cosine);
  const minX = Math.min(...xs), minY = Math.min(...ys);
  return [minX, minY, Math.max(...xs) - minX, Math.max(...ys) - minY].map(Math.round);
}

function encodeSpace(frame, references) {
  const step = Math.max(5, Math.round(Math.min(frame.width, frame.height) / 50));
  const columns = Math.ceil(frame.width / step) + 1;
  const rows = Math.ceil(frame.height / step) + 1;
  const slot = (ref, scale, angle, x, y) => ((((ref * SCALES.length + scale) * ANGLES + angle) * columns + x) * rows + y);
  function decode(index) {
    const y = index % rows * step;
    index = Math.floor(index / rows);
    const x = index % columns * step;
    index = Math.floor(index / columns);
    const angle = index % ANGLES * 360 / ANGLES;
    index = Math.floor(index / ANGLES);
    const scale = SCALES[index % SCALES.length];
    const reference = references[Math.floor(index / SCALES.length)];
    const box = rotatedBox(reference, x, y, scale, angle * Math.PI / 180);
    return { part: reference.part, key: reference.key, box, x, y, angle, scale };
  }
  return { step, columns, rows, slot, decode };
}

// Each local junction sees only its own group of C descriptors. It can emit
// one weighted vote per pose slot. Repeated points from one group cannot count
// as agreement between two distinct parts of the reference.
function votesForGroup(references, scene, space, group) {
  const votes = new Map();
  for (const [index, reference] of references.entries()) {
    for (const point of reference.points) {
      if (point.group !== group) continue;
      for (const { sample, scale, error } of localMatches(point, scene)) {
        const angle = (sample.angle - point.angle + 360) % 360;
        const radians = angle * Math.PI / 180;
        const cx = sample.x - scale * (point.x * Math.cos(radians) - point.y * Math.sin(radians));
        const cy = sample.y - scale * (point.x * Math.sin(radians) + point.y * Math.cos(radians));
        const xs = neighbors(cx / space.step, space.columns);
        const ys = neighbors(cy / space.step, space.rows);
        const angles = neighbors(angle / 360 * ANGLES, ANGLES, true);
        const scales = neighbors(Math.log2(scale) * 4 + 6, SCALES.length);
        const similarity = 1 - Math.sqrt(error);
        for (const x of xs) for (const y of ys) for (const a of angles) for (const s of scales) {
          const slot = space.slot(index, s.index, a.index, x.index, y.index);
          const weight = Math.round(AMPLITUDE * similarity * x.weight * y.weight * a.weight * s.weight);
          if (weight > (votes.get(slot) ?? 0)) votes.set(slot, weight);
        }
      }
    }
  }
  return votes;
}

function mix(agreement, tick, group, amplitude) {
  let row = agreement.get(tick);
  if (!row) {
    row = { amplitude: 0, votes: [0, 0, 0] };
    agreement.set(tick, row);
  }
  row.amplitude += amplitude;
  row.votes[group] = amplitude;
}

function directVote(inputs) {
  const scores = new Map();
  inputs.forEach((votes, group) => {
    for (const [slot, amplitude] of votes) mix(scores, slot, group, amplitude);
  });
  return scores;
}

function transmit(circuit, path, portCut) {
  // On these dedicated delay lines, routing preserves each sample's amplitude.
  // Skipping empty ticks is exact: the sparse input is shifted by every link.
  let delay = 0;
  for (const port of path) {
    if (port === portCut) return null;
    delay += circuit.arcs[port].delay;
  }
  return delay;
}

function waveVote(inputs, circuit, mode, late = 0) {
  const paths = mode === 'feedForward' ? circuit.paths : circuit.recurrent;
  const coincidence = new Map();
  inputs.forEach((votes, group) => {
    const period = circuit.delay(paths[group]);
    const launch = circuit.arrival - period + (group === 0 ? late : 0);
    const transit = transmit(circuit, paths[group], mode === 'cut' ? circuit.cut : null);
    if (transit === null) return;
    for (const [slot, amplitude] of votes) mix(coincidence, slot * SLOT + launch + transit, group, amplitude);
  });
  const outputDelay = transmit(circuit, paths[GROUPS], null);
  const wave = new Map();
  for (const [tick, row] of coincidence) wave.set(tick + outputDelay, row);
  const readout = new Map();
  for (const [tick, row] of wave) {
    const elapsed = tick - circuit.view.duration;
    // The distant probe samples the predefined pose clock. Off-clock arrivals
    // are still in the model, but cannot be read at the intended pose slot.
    if (elapsed >= 0 && elapsed % SLOT === 0) readout.set(elapsed / SLOT, row);
  }
  return readout;
}

function iou(a, b) {
  const overlap = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0])) *
    Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  return overlap / Math.max(1, a[2] * a[3] + b[2] * b[3] - overlap);
}

function rank(scores, space) {
  const order = [...scores].filter(([, row]) => row.votes.filter(Boolean).length >= 2)
    .sort(([aSlot, a], [bSlot, b]) => b.amplitude - a.amplitude || aSlot - bSlot);
  const result = [];
  for (const [slot, row] of order) {
    const pose = space.decode(slot);
    if (result.some(other => iou(pose.box, other.box) > 0.40)) continue;
    result.push({ ...pose, slot, score: row.amplitude / (GROUPS * AMPLITUDE), votes: row.votes.map(value => value / AMPLITUDE) });
    if (result.length >= 5) break;
  }
  return result;
}

module.exports = { AMPLITUDE, GROUPS, createCircuit, encodeSpace, votesForGroup, directVote, waveVote, rank, iou };
