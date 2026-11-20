'use strict';

importScripts('engine.js');

const { createNetwork, FileSimulation, AlignmentCircuit, setup, START, C_LEVEL } = globalThis.SphereLab;
const SIZE = 25;
const STRIDE = 2 * SIZE - 1;
const MID = (SIZE - 1) / 2;
const TICKS = Array.from({ length: SIZE ** 2 }, (_, i) => Math.floor(i / SIZE) * STRIDE + i % SIZE);
const SPAN = TICKS.at(-1) + 1;
const AMPLITUDE = 0.5;
const RESULT_LEVEL = AMPLITUDE * Math.sin(Math.PI * AMPLITUDE * C_LEVEL);
const ACCEPTANCE = 0.86;
const FRAME_INTERVAL = 12;

function turn(x, y, degrees) {
  const angle = degrees * Math.PI / 180;
  return [x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle)];
}

function readout(degrees) {
  const indexes = new Int32Array(SPAN).fill(-1);
  for (let pixel = 0; pixel < TICKS.length; pixel++) {
    const [x, y] = turn(pixel % SIZE - MID, Math.floor(pixel / SIZE) - MID, -degrees);
    const sx = Math.round(MID + x), sy = Math.round(MID + y);
    if (sx >= 0 && sx < SIZE && sy >= 0 && sy < SIZE) indexes[TICKS[pixel]] = sy * STRIDE + sx;
  }
  return indexes;
}

function silhouette(shape, cx, cy, degrees) {
  return TICKS.map((_, pixel) => {
    const [x, y] = turn(pixel % SIZE - cx, Math.floor(pixel / SIZE) - cy, -degrees);
    if (shape === 'T') {
      return Number((Math.abs(x) < 5.5 && y > -4.5 && y < -1.5) ||
        (Math.abs(x) < 1.5 && y >= -1.5 && y < 4.5));
    }
    return Number((x > -3.5 && x < -0.5 && Math.abs(y) < 4.5) ||
      (x > -3.5 && x < 4.5 && y > 1.5 && y < 4.5));
  });
}

function encode(mask) {
  const wave = new Float64Array(SPAN);
  TICKS.forEach((tick, i) => { wave[tick] = mask[i] * AMPLITUDE; });
  return wave;
}

function decode(wave, level) {
  return TICKS.map(tick => {
    const bit = Number(wave[tick] > level / 2);
    if (Math.abs(wave[tick] - bit * level) > 1e-13) throw new Error('A probe sample is outside its expected signal levels.');
    return bit;
  });
}

class BrowserExperiment extends FileSimulation {
  constructor(angles, disconnected) {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    this.learning = new AlignmentCircuit(true, START, { size: SIZE });
    this.learnedAt = START + this.learning.emitAt;
    this.sceneAt = this.learnedAt + setup.loopDelays[1];
    this.comparison = new AlignmentCircuit(true, this.sceneAt, { size: SIZE });
    this.comparison.readouts = angles.map(readout);
    this.comparison.accumulated = new Float64Array(angles.length * this.comparison.tapDelays.length);
    this.disconnected = disconnected;
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);
    const [f0, b0, f1, b1] = this.lanePorts[this.nodeIndex++];
    const a = values[b0], b = values[b1], carrier = values[f0], returned = values[f1];
    const tick = this.tick + 1;
    // The cut changes the comparator input. The returning wave remains
    // in the simulated network, including its accounted energy.
    const phase = tick < this.sceneAt ? this.learning.receive(tick, a, b) :
      this.comparison.receive(tick, this.disconnected ? 0 : returned, b);
    values[f0] = (a + b) * Math.SQRT1_2;
    values[f1] = (a - b) * Math.SQRT1_2;
    values[b0] = carrier * Math.cos(phase) - returned * Math.sin(phase);
    values[b1] = carrier * Math.sin(phase) + returned * Math.cos(phase);
  }
}

function snapshot(arcs, tick) {
  const samples = [];
  for (let i = 0; i < arcs.length; i++) {
    const { buffer, delay, head } = arcs[i];
    for (let slot = 0; slot < delay; slot++) {
      const sample = buffer[slot];
      if (Math.abs(sample) <= 1e-12) continue;
      const remaining = (slot - head + delay) % delay;
      samples.push(i, (delay - remaining - 0.5) / delay, sample);
    }
  }
  return { tick, samples: Float32Array.from(samples) };
}

function geometry(sim) {
  const { nodes, edges, arcs, sourceNode } = sim.network;
  return {
    nodes,
    edges: edges.map(({ a, b }) => [a, b]),
    arcs: arcs.map(({ from, to }) => [from, to]),
    source: sourceNode,
    junction: setup.mixer,
    probe: setup.station.node
  };
}

function simulate(options) {
  const shape = options.shape === 'L' ? 'L' : 'T';
  const other = shape === 'T' ? 'L' : 'T';
  const angularStep = options.search === 'quarters' ? 90 : 15;
  const angles = Array.from({ length: 360 / angularStep }, (_, i) => i * angularStep);
  const sim = new BrowserExperiment(angles, options.disconnected);
  const { arcs, nodePorts } = sim.network;
  const carrier = arcs[nodePorts[setup.mixer][sim.lanePorts[setup.mixer][0]]].reverse;
  const probe = nodePorts[setup.station.node][sim.lanePorts[setup.station.node][2]];
  const resultAt = sim.sceneAt + sim.comparison.emitAt;
  const cDelay = setup.loopDelays[1] - setup.station.probes[1].delay;
  const target = silhouette(shape, MID + options.x, MID + options.y, options.angle);
  const decoy = silhouette(other, 19, 17, 37);
  const scene = target.map((pixel, index) => Number(Boolean(options.target && pixel || options.decoy && decoy[index])));
  const viewA = silhouette(shape, MID, MID, 0);
  const viewB = silhouette(shape, MID, MID, 90);
  const inputs = [
    [setup.sourceArcs[0], START, encode(viewA)],
    [setup.sourceArcs[1], START, encode(viewB)],
    [carrier, sim.learnedAt, new Float64Array(SPAN).fill(AMPLITUDE)],
    [setup.sourceArcs[1], sim.sceneAt, encode(scene)],
    [carrier, resultAt, new Float64Array(SPAN).fill(AMPLITUDE)]
  ].map(([arc, tick, values]) => ({ arc, from: tick - arcs[arc].delay, values }));
  const captures = [sim.learnedAt + cDelay, resultAt + cDelay]
    .map(from => ({ from, wave: new Float64Array(SPAN) }));
  const end = captures.at(-1).from + SPAN - 1;
  const milestones = {
    collected: START,
    emitted: sim.learnedAt,
    referenceAtProbe: captures[0].from,
    returned: sim.sceneAt,
    output: resultAt,
    outputAtProbe: captures[1].from,
    end
  };

  self.postMessage({ type: 'start', geometry: geometry(sim), viewA, viewB, scene, milestones, size: SIZE });
  const frames = [snapshot(arcs, 0)];
  const eventTicks = new Set(Object.values(milestones));

  for (let tick = 1; tick <= end; tick++) {
    sim.step();
    for (const { arc, from, values } of inputs) {
      const value = values[tick - from];
      if (value) sim.transmit(arc, value);
    }
    for (const { from, wave } of captures) {
      if (tick >= from && tick < from + SPAN) wave[tick - from] = sim.lastIncoming[probe];
    }
    if (tick % FRAME_INTERVAL === 0 || eventTicks.has(tick)) frames.push(snapshot(arcs, tick));
    if (tick % 240 === 0) self.postMessage({ type: 'progress', tick, end });
  }

  const [referenceWave, outputWave] = captures.map(capture => capture.wave);
  const { a, b, selectedTap, selectedQuarterTurns, aDelay, tapDelays } = sim.comparison;
  const incomingScene = encode(scene);
  for (let i = 0; i < SPAN; i++) {
    if (a[i] !== (options.disconnected ? 0 : referenceWave[i])) throw new Error('The returned reference does not match the captured reference.');
    if (b[i] !== incomingScene[i]) throw new Error('The captured scene does not match the injected scene.');
  }

  const reference = decode(referenceWave, C_LEVEL);
  const matched = decode(outputWave, RESULT_LEVEL);
  const referencePixels = reference.reduce((sum, bit) => sum + bit, 0);
  const matches = matched.reduce((sum, bit) => sum + bit, 0);
  const score = referencePixels > 0 ? matches / referencePixels : 0;
  let selection = null;
  if (score >= ACCEPTANCE && selectedTap !== null) {
    const delay = tapDelays[selectedTap] - aDelay;
    const dy = Math.floor((delay + SIZE - 1) / STRIDE), dx = delay - dy * STRIDE;
    const correction = angles[selectedQuarterTurns];
    const [x, y] = turn(-dx, -dy, -correction);
    selection = {
      center: [MID + x, MID + y],
      degrees: (360 - correction) % 360,
      distance: Math.hypot(x - options.x, y - options.y)
    };
  }

  const energyError = sim.getStats().relativeError;
  if (energyError >= 1e-12) throw new Error('The energy balance did not close.');
  self.postMessage({
    type: 'result',
    frames,
    reference,
    matched,
    referencePixels,
    matches,
    score,
    selection,
    angularStep,
    energyError,
    options
  }, frames.map(frame => frame.samples.buffer));
}

self.onmessage = ({ data }) => {
  try {
    simulate(data);
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message });
  }
};
