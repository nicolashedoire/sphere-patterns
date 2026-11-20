'use strict';

const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation } = require('./sphere-roundtrip-run.cjs');
const { AlignmentCircuit, START, C_LEVEL, setup } = require('./sphere-pattern-internal-delay-run.cjs');

const outputs = path.resolve(__dirname, '../outputs');
const input = JSON.parse(fs.readFileSync(path.join(__dirname, 'sphere-mechanical-part-masks.json')));
const source = fs.readFileSync(path.join(outputs, input.sourceFile));
assert.equal(createHash('sha256').update(source).digest('hex'), input.sourceSHA256);
const { size, threshold, cropMargin, masks: [a, b, scene] } = input;
const stride = 2 * size - 1;
const span = (size - 1) * stride + size;
const amplitude = 0.5;
const resultLevel = amplitude * Math.sin(Math.PI * amplitude * C_LEVEL);
const sampleIndexes = Array.from({ length: size ** 2 }, (_, i) => Math.floor(i / size) * stride + i % size);
const rotationCircuit = start => new AlignmentCircuit(true, start, { size });
const emitOffset = rotationCircuit(START).emitAt;
const learnedAt = START + emitOffset;
const loop = setup.loopDelays[1];
const cDelay = loop - setup.station.probes[1].delay;
const sceneAt = learnedAt + loop;
const resultAt = sceneAt + emitOffset;

class PhotoSimulation extends FileSimulation {
  constructor() {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    this.learn = rotationCircuit(START);
    this.compare = rotationCircuit(sceneAt);
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);
    const [f0, b0, f1, b1] = this.lanePorts[this.nodeIndex++];
    const a = values[b0], b = values[b1], carrier = values[f0], returned = values[f1];
    const tick = this.tick + 1;
    if (tick === learnedAt + span) this.learn = null;
    const angle = this.learn ? this.learn.receive(tick, a, b) : this.compare.receive(tick, returned, b);
    const sine = Math.sin(angle), cosine = Math.cos(angle);
    values[f0] = (a + b) * Math.SQRT1_2;
    values[f1] = (a - b) * Math.SQRT1_2;
    values[b0] = carrier * cosine - returned * sine;
    values[b1] = carrier * sine + returned * cosine;
  }
}

function encode(grid) {
  assert.equal(grid.length, size ** 2);
  const samples = new Float64Array(span);
  for (const [pixel, time] of sampleIndexes.entries()) samples[time] = amplitude * grid[pixel];
  return samples;
}

function decode(wave, level) {
  return sampleIndexes.map(time => {
    const bit = Number(wave[time] > level / 2);
    assert.ok(Math.abs(wave[time] - bit * level) < 1e-13);
    return bit;
  });
}

const sim = new PhotoSimulation();
sim.reset = () => { throw new Error('The sphere must retain the reference.'); };
const { arcs, nodePorts } = sim.network;
const [mixerForward0] = sim.lanePorts[setup.mixer];
const [, , photoProbe] = sim.lanePorts[setup.station.node];
const cPort = nodePorts[setup.station.node][photoProbe];
const carrierArc = arcs[nodePorts[setup.mixer][mixerForward0]].reverse;
const carrier = new Float64Array(span).fill(amplitude);
const inputs = [
  [setup.sourceArcs[0], START, encode(a)], [setup.sourceArcs[1], START, encode(b)],
  [carrierArc, learnedAt, carrier], [setup.sourceArcs[1], sceneAt, encode(scene)], [carrierArc, resultAt, carrier]
].map(([arc, tick, samples]) => ({ arc, launch: tick - arcs[arc].delay, samples }));
const captures = [
  [cPort, learnedAt + cDelay], [cPort, resultAt + cDelay],
  ...setup.station.probes.map(probe => [probe.receivingPort, sceneAt + probe.delay])
].map(([port, first]) => ({ port, first, wave: new Float64Array(span) }));
const stop = Math.max(...captures.map(observation => observation.first)) + span - 1;
for (let tick = 1; tick <= stop; tick++) {
  sim.step();
  for (const { arc, launch, samples } of inputs) {
    const value = samples[tick - launch];
    if (value) sim.transmit(arc, value);
  }
  for (const { port, first, wave } of captures) {
    const offset = tick - first;
    if (offset >= 0 && offset < span) wave[offset] = sim.lastIncoming[port];
  }
}

const [learned, output, sum, difference] = captures.map(observation => observation.wave);
assert.deepEqual(sim.compare.a, learned);
assert.deepEqual(sim.compare.b, encode(scene));
assert.deepEqual(decode(sum.map((value, i) => (value - difference[i]) * Math.SQRT1_2), amplitude), scene);
const reference = decode(learned, C_LEVEL);
const found = decode(output, resultLevel);
const { selectedTap, selectedQuarterTurns, accumulated } = sim.compare;
const selection = selectedTap === null ? null : (() => {
  const delay = sim.compare.tapDelays[selectedTap] - sim.compare.aDelay;
  const dy = Math.floor((delay + size - 1) / stride);
  return { dx: delay - dy * stride, dy, clockwiseTurns: selectedQuarterTurns };
})();
const matchedInScene = Array(size ** 2).fill(0);
if (selection) {
  for (const [i, pixel] of found.entries()) {
    if (!pixel) continue;
    let x = i % size - selection.dx, y = Math.floor(i / size) - selection.dy;
    for (let turn = selection.clockwiseTurns; turn < 4; turn++) [x, y] = [size - 1 - y, x];
    assert.ok(x >= 0 && x < size && y >= 0 && y < size && scene[y * size + x] === 1);
    matchedInScene[y * size + x] = 1;
  }
}

const scores = Array.from(accumulated);
const ranked = [...scores].sort((a, b) => b - a);
const pixels = grid => grid.reduce((total, bit) => total + bit, 0);
const score = Math.round(ranked[0] / resultLevel ** 2);
const runnerUp = Math.round(ranked[1] / resultLevel ** 2);
if (selection) assert.equal(pixels(matchedInScene), score);
const energyError = sim.getStats().relativeError;
assert.ok(energyError < 1e-12);
const result = { size, threshold, cropMargin, sourceFile: input.sourceFile, sourceSHA256: input.sourceSHA256,
  sourceGeneration: { tool: 'ImageGen', prompt: fs.readFileSync(path.join(__dirname, 'sphere-mechanical-part-prompt.txt'), 'utf8').trim() },
  patterns: { a, b, scene, reference, found, matchedInScene }, selection, score, runnerUp, referencePixels: pixels(reference), scores,
  experiment: { learnedAt, sceneAt, resultAt, span, loop, cDelay, energyError, sourceFrames: 3 },
  readings: { referenceAtProbe: Array.from(learned), returningReference: Array.from(sim.compare.a), resultAtProbe: Array.from(output) } };
fs.writeFileSync(path.join(outputs, 'sphere-mechanical-part-measurements.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ selection, score, runnerUp, referencePixels: pixels(reference), energyError,
  scenePixels: pixels(scene), knownMasksAreOnlyBrightness: true }));
