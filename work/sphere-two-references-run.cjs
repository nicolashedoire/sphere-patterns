'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation } = require('./sphere-roundtrip-run.cjs');
const { AlignmentCircuit, encode, decode, labelTap, SIZE, SPAN, TAP_COUNT, START, C_LEVEL, setup } =
  require('./sphere-pattern-internal-delay-run.cjs');

const AMP = 0.5;
const MATCH_LEVEL = AMP * Math.sin(Math.PI * C_LEVEL * AMP);
const emitOffset = new AlignmentCircuit(true).emitAt;
const backwardLoop = setup.loopDelays[1];
const probeDelay = backwardLoop - setup.station.probes[1].delay;
const trainingStarts = [START, 600];
const learningEmits = trainingStarts.map(tick => tick + emitOffset);
const referenceReturns = learningEmits.map(tick => tick + backwardLoop);
const matchingEmits = referenceReturns.map(tick => tick + emitOffset);
const sceneStart = referenceReturns[0];
const outputs = path.resolve(__dirname, '../outputs');

function pattern(points) {
  const pixels = Array(SIZE ** 2).fill(0);
  for (const [x, y] of points) pixels[y * SIZE + x] = 1;
  return pixels;
}

const priorL = JSON.parse(fs.readFileSync(path.join(outputs, 'sphere-internal-rotation-measurements.json'))).patterns;
const tShape = pattern([[0, 0], [1, 0], [2, 0], [1, 1], [1, 2]]);
const examples = [[priorL.a, priorL.b], [
  pattern([[0, 0], [1, 0], [2, 0], [1, 1], [1, 2], [5, 1]]),
  pattern([[6, 3], [6, 4], [6, 5], [5, 4], [4, 4], [1, 6]])
]];
const lInScene = [[5, 6], [5, 5], [5, 4], [4, 4], [3, 4]];
const tInScene = [[6, 1], [6, 2], [6, 3], [5, 2], [4, 2]];
const decoyPoints = [[0, 0], [0, 1], [0, 2], [1, 2]];
const scene = pattern([...lInScene, ...tInScene, ...decoyPoints]);
const decoy = pattern(decoyPoints);

class TwoReferenceSimulation extends FileSimulation {
  constructor() {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    this.training = trainingStarts.map(tick => new AlignmentCircuit(true, tick));
    this.matching = referenceReturns.map(tick => new AlignmentCircuit(true, tick));
    this.scene = new Float64Array(SPAN);
  }

  routeAngle(tick, a, b, returningReference) {
    for (const [i, circuit] of this.training.entries()) {
      if (!circuit || tick < trainingStarts[i]) continue;
      if (tick < learningEmits[i] + SPAN) return circuit.receive(tick, a, b);
      this.training[i] = null;
    }
    const offset = tick - sceneStart;
    if (offset >= 0 && offset < SPAN) this.scene[offset] = b;
    for (const [i, circuit] of this.matching.entries()) {
      if (tick >= referenceReturns[i] && tick < matchingEmits[i] + SPAN) {
        const sceneSample = this.scene[tick - referenceReturns[i]] ?? 0;
        return circuit.receive(tick, returningReference, sceneSample);
      }
    }
    return 0;
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);
    const [forward0, backward0, forward1, backward1] = this.lanePorts[this.nodeIndex++];
    const a = values[backward0], b = values[backward1];
    const carrier = values[forward0], returningReference = values[forward1];
    const angle = this.routeAngle(this.tick + 1, a, b, returningReference);
    values[forward0] = (a + b) * Math.SQRT1_2;
    values[forward1] = (a - b) * Math.SQRT1_2;
    values[backward0] = carrier * Math.cos(angle) - returningReference * Math.sin(angle);
    values[backward1] = carrier * Math.sin(angle) + returningReference * Math.cos(angle);
  }
}

function run(image) {
  const sim = new TwoReferenceSimulation();
  sim.reset = () => { throw new Error('The sphere must retain its references.'); };
  const { arcs, nodePorts } = sim.network;
  const [mixerForward0] = sim.lanePorts[setup.mixer];
  const [, , stationForward1] = sim.lanePorts[setup.station.node];
  const portC = nodePorts[setup.station.node][stationForward1];
  const carrierArc = arcs[nodePorts[setup.mixer][mixerForward0]].reverse;
  const carrier = new Float64Array(SPAN).fill(AMP);
  const pulses = examples.flatMap((pair, i) => pair.map((pixels, lane) => ({
    arc: setup.sourceArcs[lane], time: trainingStarts[i], data: encode(pixels)
  })));
  pulses.push({ arc: setup.sourceArcs[1], time: sceneStart, data: encode(image) });
  for (const time of [...learningEmits, ...matchingEmits]) pulses.push({ arc: carrierArc, time, data: carrier });
  const inputs = pulses.map(pulse => ({ ...pulse, launch: pulse.time - arcs[pulse.arc].delay }));
  assert.ok(inputs.every(input => input.launch >= 1));

  const read = times => times.map(time => ({ tick: time + probeDelay, port: portC, trace: new Float64Array(SPAN) }));
  const learned = read(learningEmits);
  const stillCirculating = read(referenceReturns);
  const matched = read(matchingEmits);
  const readScene = setup.station.probes.map(probe => ({ tick: sceneStart + probe.delay,
    port: probe.receivingPort, trace: new Float64Array(SPAN) }));
  const capture = [...learned, ...stillCirculating, ...matched, ...readScene];
  const stop = Math.max(...capture.map(item => item.tick)) + SPAN - 1;
  assert.ok(stop < learningEmits[0] + 2 * backwardLoop);

  for (let tick = 1; tick <= stop; tick++) {
    sim.step();
    for (const { arc, launch, data } of inputs) {
      const sample = data[tick - launch];
      if (sample) sim.transmit(arc, sample);
    }
    for (const item of capture) {
      const offset = tick - item.tick;
      if (offset >= 0 && offset < SPAN) item.trace[offset] = sim.lastIncoming[item.port];
    }
  }

  assert.ok(sim.training.every(circuit => circuit === null));
  assert.deepEqual(sim.scene, encode(image));
  const [sum, difference] = readScene.map(item => item.trace);
  assert.deepEqual(decode(sum.map((v, i) => (v - difference[i]) * Math.SQRT1_2), AMP), image);
  assert.deepEqual(decode(sum.map((v, i) => (v + difference[i]) * Math.SQRT1_2), AMP), pattern([]));
  const references = learned.map(item => decode(item.trace, C_LEVEL));
  assert.deepEqual(references, [priorL.c, tShape]);
  const matches = matched.map((item, i) => {
    const circuit = sim.matching[i];
    assert.deepEqual(circuit.a, learned[i].trace);
    assert.deepEqual(circuit.b, sim.scene);
    assert.deepEqual(stillCirculating[i].trace, learned[i].trace);
    const decoded = decode(item.trace, MATCH_LEVEL);
    const selection = circuit.selectedTap === null ? null : { ...labelTap(circuit.selectedTap),
      rotationDegrees: circuit.selectedQuarterTurns * 90, score: decoded.reduce((sum, bit) => sum + bit, 0) };
    const scores = Array.from(circuit.accumulated);
    const ranked = [...scores].sort((a, b) => b - a);
    return { selection, decoded, peak: Math.round(ranked[0] / MATCH_LEVEL ** 2), runnerUp: Math.round(ranked[1] / MATCH_LEVEL ** 2),
      scores, readings: { learned: Array.from(learned[i].trace), incomingReference: Array.from(circuit.a),
        stillCirculating: Array.from(stillCirculating[i].trace), emittedMatch: Array.from(item.trace) } };
  });
  const energyError = sim.getStats().relativeError;
  assert.ok(energyError < 1e-12);
  return { references, matches, energyError, sceneRead: { sum: Array.from(sum), difference: Array.from(difference) } };
}

function placeMatch(decoded, { dx, dy, rotationDegrees }) {
  return pattern(decoded.flatMap((bit, i) => {
    if (!bit) return [];
    let x = i % SIZE - dx, y = Math.floor(i / SIZE) - dy;
    for (let angle = rotationDegrees; angle < 360; angle += 90) [x, y] = [SIZE - 1 - y, x];
    assert.ok(x >= 0 && x < SIZE && y >= 0 && y < SIZE);
    return [[x, y]];
  }));
}

const main = run(scene);
const decoyOnly = run(decoy);
assert.ok(main.matches.every(match => match.selection?.score === 5 && match.runnerUp === 4));
const matchedInScene = main.matches.map(match => placeMatch(match.decoded, match.selection));
assert.deepEqual(matchedInScene, [pattern(lInScene), pattern(tInScene)]);
assert.ok(decoyOnly.matches.every(match => match.peak === 4));
const summary = { size: SIZE, patterns: { trainingPairs: examples, references: main.references, scene, matchedInScene },
  main, controls: { decoyOnly }, experiment: { seed: setup.seed, mixer: setup.mixer, station: setup.station.node,
    trainingStarts, learningEmits, referenceReturns, matchingEmits, probeDelay, backwardLoop,
    sceneStart, sceneInjectionsPerTrial: 1, candidatesPerReference: 4 * TAP_COUNT, referenceLevel: C_LEVEL, matchLevel: MATCH_LEVEL } };
fs.writeFileSync(path.join(outputs, 'sphere-two-references-measurements.json'), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ main: main.matches.map(match => match.selection), decoy: decoyOnly.matches.map(match => match.peak),
  energyError: main.energyError, experiment: summary.experiment }));
