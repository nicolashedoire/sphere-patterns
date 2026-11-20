'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation } = require('./sphere-roundtrip-run.cjs');
const { AlignmentCircuit, START, C_LEVEL, setup } = require('./sphere-motif-retard-interne-run.cjs');

const SIZE = 25;
const STRIDE = 2 * SIZE - 1;
const MID = (SIZE - 1) / 2;
const TICKS = Array.from({ length: SIZE ** 2 }, (_, i) => Math.floor(i / SIZE) * STRIDE + i % SIZE);
const SPAN = TICKS.at(-1) + 1;
const AMPLITUDE = .5;
const RESULT_LEVEL = AMPLITUDE * Math.sin(Math.PI * AMPLITUDE * C_LEVEL);
const ANGLES = Array.from({ length: 24 }, (_, i) => i * 15);
const ACCEPT_AT = .86;

function spin(x, y, degrees) {
  const angle = degrees * Math.PI / 180;
  const c = Math.cos(angle), s = Math.sin(angle);
  return [x * c - y * s, x * s + y * c];
}

function readout(degrees) {
  const indexes = new Int32Array(SPAN).fill(-1);
  for (let i = 0; i < TICKS.length; i++) {
    const [x, y] = spin(i % SIZE - MID, Math.floor(i / SIZE) - MID, -degrees);
    const sx = Math.round(MID + x), sy = Math.round(MID + y);
    if (sx >= 0 && sx < SIZE && sy >= 0 && sy < SIZE) indexes[TICKS[i]] = sy * STRIDE + sx;
  }
  return indexes;
}

function silhouette(form, cx, cy, degrees) {
  return TICKS.map((_, i) => {
    const [x, y] = spin(i % SIZE - cx, Math.floor(i / SIZE) - cy, -degrees);
    if (form === 'T') {
      return Number((Math.abs(x) < 5.5 && y > -4.5 && y < -1.5) ||
        (Math.abs(x) < 1.5 && y >= -1.5 && y < 4.5));
    }
    return Number((x > -3.5 && x < -.5 && Math.abs(y) < 4.5) ||
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
    assert.ok(Math.abs(wave[tick] - bit * level) < 1e-13);
    return bit;
  });
}

function count(mask) {
  return mask.reduce((sum, bit) => sum + bit, 0);
}

class ObliqueSimulation extends FileSimulation {
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
    // The control masks the return only at the input to the comparator.
    // The physical wave still circulates and is included in the energy balance.
    const angle = tick < this.sceneAt ? this.learning.receive(tick, a, b) :
      this.comparison.receive(tick, this.disconnected ? 0 : returned, b);
    values[f0] = (a + b) * Math.SQRT1_2;
    values[f1] = (a - b) * Math.SQRT1_2;
    values[b0] = carrier * Math.cos(angle) - returned * Math.sin(angle);
    values[b1] = carrier * Math.sin(angle) + returned * Math.cos(angle);
  }
}

function run(scene, angles, disconnected = false) {
  const sim = new ObliqueSimulation(angles, disconnected);
  const { arcs, nodePorts } = sim.network;
  const carrier = arcs[nodePorts[setup.mixer][sim.lanePorts[setup.mixer][0]]].reverse;
  const probe = nodePorts[setup.station.node][sim.lanePorts[setup.station.node][2]];
  const resultAt = sim.sceneAt + sim.comparison.emitAt;
  const cDelay = setup.loopDelays[1] - setup.station.probes[1].delay;
  const inputs = [
    [setup.sourceArcs[0], START, encode(silhouette('T', MID, MID, 0))],
    [setup.sourceArcs[1], START, encode(silhouette('T', MID, MID, 90))],
    [carrier, sim.learnedAt, new Float64Array(SPAN).fill(AMPLITUDE)],
    [setup.sourceArcs[1], sim.sceneAt, encode(scene)],
    [carrier, resultAt, new Float64Array(SPAN).fill(AMPLITUDE)]
  ].map(([arc, tick, values]) => ({ arc, from: tick - arcs[arc].delay, values }));

  const captures = [sim.learnedAt + cDelay, resultAt + cDelay]
    .map(from => ({ from, wave: new Float64Array(SPAN) }));
  const end = captures.at(-1).from + SPAN - 1;
  for (let tick = 1; tick <= end; tick++) {
    sim.step();
    for (const { arc, from, values } of inputs) {
      const value = values[tick - from];
      if (value) sim.transmit(arc, value);
    }
    for (const { from, wave } of captures) {
      if (tick >= from && tick < from + SPAN) wave[tick - from] = sim.lastIncoming[probe];
    }
  }

  const [referenceWave, outputWave] = captures.map(item => item.wave);
  const { a, b, selectedTap, selectedQuarterTurns, aDelay, tapDelays } = sim.comparison;
  assert.deepEqual(a, disconnected ? new Float64Array(SPAN) : referenceWave);
  assert.deepEqual(b, encode(scene));
  const reference = decode(referenceWave, C_LEVEL);
  const matched = decode(outputWave, RESULT_LEVEL);
  const referencePixels = count(reference);
  const matches = count(matched);
  assert.ok(referencePixels > 0);
  const kept = matches / referencePixels >= ACCEPT_AT && selectedTap !== null;
  const selection = kept ? (() => {
    const delay = tapDelays[selectedTap] - aDelay;
    const dy = Math.floor((delay + SIZE - 1) / STRIDE), dx = delay - dy * STRIDE;
    const degrees = angles[selectedQuarterTurns];
    const [x, y] = spin(-dx, -dy, -degrees);
    return { inverseDegrees: degrees, dx, dy, center: [MID + x, MID + y] };
  })() : null;
  const energyError = sim.getStats().relativeError;
  assert.ok(energyError < 1e-12);
  return { matches, referencePixels, selection, energyError,
    referenceAtProbe: captures[0].from, referenceReturnedAt: sim.sceneAt, outputAtProbe: captures[1].from };
}

const poses = [
  { id: 'quarter-turn', target: [8, 9, 90] },
  { id: 'oblique-30', target: [8, 9, 30] },
  { id: 'oblique-58', target: [9, 10, 58] },
  { id: 'oblique-107', target: [8, 10, 107] },
  { id: 'decoy-only', target: null }
];
const decoy = silhouette('L', 19, 17, 37);
const results = poses.map(({ id, target }) => {
  const foreground = target ? silhouette('T', ...target) : decoy.map(() => 0);
  const scene = foreground.map((bit, i) => bit | decoy[i]);
  const quarters = run(scene, [0, 90, 180, 270]);
  const oblique = run(scene, ANGLES);
  const disabled = run(scene, ANGLES, true);
  assert.equal(disabled.selection, null);
  assert.equal(disabled.matches, 0);
  const judged = result => target ? !!result.selection &&
    Math.hypot(result.selection.center[0] - target[0], result.selection.center[1] - target[1]) <= 2 :
    result.selection === null;
  return { id, target, scene, quarters: { ...quarters, correct: judged(quarters) },
    oblique: { ...oblique, correct: judged(oblique) }, disabled };
});
const report = { size: SIZE, angles: ANGLES, threshold: ACCEPT_AT, decoy: 'L', results };
const output = process.argv[2] ?? path.join(__dirname, '../outputs/sphere-motif-angle-oblique-mesures.json');
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
for (const { id, quarters, oblique } of results) console.log(id, quarters.matches, quarters.correct, oblique.matches, oblique.correct);
