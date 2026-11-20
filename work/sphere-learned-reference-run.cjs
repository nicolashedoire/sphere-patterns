'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation } = require('./sphere-roundtrip-run.cjs');
const { AlignmentCircuit, encode, decode, labelTap, SIZE, TAP_COUNT, SPAN, START, C_LEVEL, setup } =
  require('./sphere-pattern-internal-delay-run.cjs');

const AMP = 0.5;
const C2_LEVEL = AMP * Math.sin(Math.PI * C_LEVEL * AMP);
const emissionOffset = new AlignmentCircuit(true).emitAt;
const firstEmission = START + emissionOffset;
const referenceReturn = firstEmission + setup.loopDelays[1];
const secondEmission = referenceReturn + emissionOffset;
const cDelay = setup.loopDelays[1] - setup.station.probes[1].delay;
const firstCAtProbe = firstEmission + cDelay;
const secondCAtProbe = secondEmission + cDelay;
const outputs = path.resolve(__dirname, '../outputs');

class RetainedReferenceSimulation extends FileSimulation {
  constructor(readReturningC) {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    this.learn = new AlignmentCircuit(true, START);
    this.recognize = new AlignmentCircuit(true, referenceReturn);
    this.readReturningC = readReturningC;
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);

    const [forward0, backward0, forward1, backward1] = this.lanePorts[this.nodeIndex++];
    const a = values[backward0], b = values[backward1];
    const carrier = values[forward0], returningC = values[forward1];
    const tick = this.tick + 1;
    if (tick === firstEmission + SPAN) this.learn = null;
    // Only the signed sample physically returning on the backward link is
    // wired to the reference input of the second comparison circuit.
    const angle = this.learn ? this.learn.receive(tick, a, b) :
      this.recognize.receive(tick, this.readReturningC ? returningC : 0, b);
    values[forward0] = (a + b) * Math.SQRT1_2;
    values[forward1] = (a - b) * Math.SQRT1_2;
    values[backward0] = carrier * Math.cos(angle) - returningC * Math.sin(angle);
    values[backward1] = carrier * Math.sin(angle) + returningC * Math.cos(angle);
  }
}

function pattern(points) {
  const image = Array(SIZE * SIZE).fill(0);
  for (const [x, y] of points) image[y * SIZE + x] = 1;
  return image;
}

const targetPoints = [[5, 6], [5, 5], [5, 4], [4, 4], [3, 4]];
const decoyPoints = [[0, 0], [0, 1], [0, 2], [1, 2]];
const scene = pattern([...targetPoints, ...decoyPoints]);
const decoy = pattern(decoyPoints);
const previous = JSON.parse(fs.readFileSync(path.join(outputs, 'sphere-internal-rotation-measurements.json')));
assert.equal(previous.size, SIZE);
const { a, b } = previous.patterns;

function run(image, readReturningC) {
  const sim = new RetainedReferenceSimulation(readReturningC);
  const { arcs, nodePorts } = sim.network;
  const [mixerForward0] = sim.lanePorts[setup.mixer];
  const [, , stationForward1] = sim.lanePorts[setup.station.node];
  const portC = nodePorts[setup.station.node][stationForward1];
  const carrierArc = arcs[nodePorts[setup.mixer][mixerForward0]].reverse;
  const carrier = new Float64Array(SPAN).fill(AMP);
  const inputs = [
    { arc: setup.sourceArcs[0], t: START, samples: encode(a) },
    { arc: setup.sourceArcs[1], t: START, samples: encode(b) },
    { arc: carrierArc, t: firstEmission, samples: carrier },
    { arc: setup.sourceArcs[1], t: referenceReturn, samples: encode(image) },
    { arc: carrierArc, t: secondEmission, samples: carrier }
  ].map(input => ({ ...input, launch: input.t - arcs[input.arc].delay }));
  assert.ok(inputs.every(input => input.launch >= 1));

  const reference = new Float64Array(SPAN), result = new Float64Array(SPAN);
  const sceneReadAt = setup.station.probes.map(probe => referenceReturn + probe.delay);
  const sceneTraces = sceneReadAt.map(() => new Float64Array(SPAN));
  const lastTick = Math.max(secondCAtProbe, ...sceneReadAt) + SPAN - 1;
  assert.ok(lastTick < firstEmission + 2 * setup.loopDelays[1]);

  for (let tick = 1; tick <= lastTick; tick++) {
    sim.step();
    for (const { arc, launch, samples } of inputs) {
      const sample = samples[tick - launch];
      if (sample) sim.transmit(arc, sample);
    }
    const firstSample = tick - firstCAtProbe;
    const secondSample = tick - secondCAtProbe;
    if (firstSample >= 0 && firstSample < SPAN) reference[firstSample] = sim.lastIncoming[portC];
    if (secondSample >= 0 && secondSample < SPAN) result[secondSample] = sim.lastIncoming[portC];
    for (const [lane, firstTick] of sceneReadAt.entries()) {
      const index = tick - firstTick;
      if (index >= 0 && index < SPAN) sceneTraces[lane][index] = sim.lastIncoming[setup.station.probes[lane].receivingPort];
    }
  }

  const { selectedTap, selectedQuarterTurns, accumulated, selectionTick, a: storedC } = sim.recognize;
  assert.equal(sim.learn, null);
  const referencePixels = decode(reference, C_LEVEL);
  assert.deepEqual(referencePixels, previous.patterns.c);
  if (readReturningC) assert.deepEqual(storedC, reference);
  else assert.ok(storedC.every(value => value === 0));
  const [sum, difference] = sceneTraces;
  assert.deepEqual(decode(sum.map((value, i) => (value - difference[i]) * Math.SQRT1_2), AMP), image);
  const found = decode(result, C2_LEVEL);
  const selection = selectedTap === null ? null : { ...labelTap(selectedTap), rotationDegrees: selectedQuarterTurns * 90,
    score: found.reduce((total, bit) => total + bit, 0) };
  const energyError = sim.getStats().relativeError;
  assert.ok(energyError < 1e-12);
  return { referencePixels, found, selection, selectionTick, scores: Array.from(accumulated), energyError,
    readings: { firstCAtProbe: Array.from(reference), referenceAtComparator: Array.from(storedC),
      secondCAtProbe: Array.from(result), sum: Array.from(sum), difference: Array.from(difference) } };
}

function locationInScene(found, { dx, dy, rotationDegrees }) {
  return pattern(found.flatMap((bit, i) => {
    if (!bit) return [];
    let x = i % SIZE - dx, y = Math.floor(i / SIZE) - dy;
    for (let degrees = rotationDegrees; degrees < 360; degrees += 90) [x, y] = [SIZE - 1 - y, x];
    assert.ok(x >= 0 && x < SIZE && y >= 0 && y < SIZE);
    return [[x, y]];
  }));
}

const main = run(scene, true);
const decoyOnly = run(decoy, true);
const disconnected = run(scene, false);
assert.ok(main.selection);
assert.deepEqual(main.found, main.referencePixels);
const matchedInScene = locationInScene(main.found, main.selection);
assert.deepEqual(matchedInScene, pattern(targetPoints));
assert.equal(decoyOnly.selection?.score, 4);
assert.ok(!disconnected.selection && disconnected.found.every(bit => !bit));
const runnerUp = [...main.scores].sort((a, b) => b - a)[1];
const runnerUpScore = Math.round(runnerUp / C2_LEVEL ** 2);
assert.equal(runnerUpScore, 4);
const summary = { size: SIZE, patterns: { reference: main.referencePixels, scene, found: main.found, matchedInScene },
  selection: main.selection, runnerUpScore, main, controls: { decoyOnly, disconnected },
  experiment: { seed: setup.seed, mixer: setup.mixer, station: setup.station.node, firstEmission,
    firstCAtProbe, referenceReturn, referenceLoopDuration: setup.loopDelays[1], secondEmission, secondCAtProbe,
    referenceFromReturningWaveOnly: true, initialComparisonMemoryReleased: true, firstCLevel: C_LEVEL,
    c2Level: C2_LEVEL, candidates: 4 * TAP_COUNT } };
fs.writeFileSync(path.join(outputs, 'sphere-learned-reference-measurements.json'), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ selection: main.selection, runnerUpScore, decoy: decoyOnly.selection,
  disconnected: disconnected.selection, experiment: summary.experiment }));
