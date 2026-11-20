'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation, createSetup } = require('./sphere-roundtrip-run.cjs');

const setup = createSetup();
const amplitude = 0.5;
const tolerance = 1e-13;
const coordinates = 16;
const observationLength = 2 * coordinates + 4;
const relationshipBins = 2 * coordinates - 1;
const referenceLength = 2 * relationshipBins;
const sceneLength = 2 * observationLength;
const trainingTimes = [setup.firstMixerTick, 200];
const referenceTime = 400;
const loop = setup.loopDelays[1];
const sceneTime = referenceTime + loop;
const decisionTime = sceneTime + sceneLength + 1;
const backwardsToProbe = loop - setup.station.probes[1].delay;
const referenceLevel = amplitude * Math.sin(Math.PI * amplitude ** 2);
const decisionLevel = amplitude * Math.sin(Math.PI * amplitude * referenceLevel);
const directions = [[1, 0], [0, 1], [-1, 0], [0, -1]];

const examples = [
  { bolts: [{ x: 2, y: 2, orientation: 0 }], nuts: [{ x: 5, y: 2 }] },
  { bolts: [{ x: 6, y: 2, orientation: 1 }], nuts: [{ x: 6, y: 5 }] }
];
const scene = {
  bolts: [{ x: 5, y: 4, orientation: 2 }, { x: 10, y: 3, orientation: 2 }],
  nuts: [{ x: 10, y: 6 }, { x: 2, y: 4 }]
};
const decoy = { bolts: [scene.bolts[1]], nuts: [scene.nuts[0]] };

function encodePositions(parts) {
  const frame = new Float64Array(sceneLength);
  for (const [i, { x, y, orientation }] of parts.entries()) {
    const start = i * observationLength;
    frame[start + x] = amplitude;
    frame[start + coordinates + y] = amplitude;
    if (orientation !== undefined) frame[start + 2 * coordinates + orientation] = amplitude;
  }
  return frame;
}

function pulseIndex(samples, start, width, level) {
  let found = null;
  for (let index = 0; index < width; index++) {
    const sample = samples[start + index];
    if (Math.abs(sample) < tolerance) continue;
    assert.ok(Math.abs(sample - level) < tolerance && found === null);
    found = index;
  }
  return found;
}

function readPositions(samples, withOrientation) {
  const positions = [];
  for (let start = 0; start < samples.length; start += observationLength) {
    const x = pulseIndex(samples, start, coordinates, amplitude);
    const y = pulseIndex(samples, start + coordinates, coordinates, amplitude);
    if (x === null && y === null) continue;
    assert.ok(x !== null && y !== null);
    const position = { x, y };
    if (withOrientation) {
      position.orientation = pulseIndex(samples, start + 2 * coordinates, 4, amplitude);
      assert.ok(position.orientation !== null);
    }
    positions.push(position);
  }
  return positions;
}

function relationship(bolt, nut) {
  const x = nut.x - bolt.x, y = nut.y - bolt.y;
  const [dx, dy] = directions[bolt.orientation];
  return { along: dx * x + dy * y, lateral: dx * y - dy * x };
}

function relationIndexes({ along, lateral }) {
  const offset = coordinates - 1;
  return [along + offset, relationshipBins + lateral + offset];
}

function receiveScene(waves) {
  return { bolts: readPositions(waves[0], true), nuts: readPositions(waves[1], false) };
}

class AssemblySimulation extends FileSimulation {
  constructor(readReference) {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    const buffer = () => [new Float64Array(sceneLength), new Float64Array(sceneLength)];
    this.examples = trainingTimes.map(buffer);
    this.scene = buffer();
    this.returned = new Float64Array(referenceLength);
    this.readReference = readReference;
  }

  angleAt(tick, a, b, returningWave) {
    if (this.examples) {
      for (const [i, time] of trainingTimes.entries()) {
        const offset = tick - time;
        if (offset >= 0 && offset < sceneLength) {
          this.examples[i][0][offset] = a;
          this.examples[i][1][offset] = b;
        }
      }
      if (tick === referenceTime) {
        this.trainingRelations = this.examples.map(receiveScene).map(({ bolts, nuts }) => relationship(bolts[0], nuts[0]));
        this.examples = null;
      }
    }
    const referenceOffset = tick - referenceTime;
    if (referenceOffset >= 0 && referenceOffset < referenceLength) {
      const [first, second] = this.trainingRelations;
      return relationIndexes(first).includes(referenceOffset) && relationIndexes(second).includes(referenceOffset)
        ? Math.PI * amplitude ** 2 : 0;
    }
    if (tick === referenceTime + referenceLength) this.trainingRelations = null;

    const sceneOffset = tick - sceneTime;
    if (sceneOffset >= 0 && sceneOffset < sceneLength) {
      this.scene[0][sceneOffset] = a;
      this.scene[1][sceneOffset] = b;
      if (sceneOffset < referenceLength) this.returned[sceneOffset] = returningWave;
    }
    if (tick === decisionTime) {
      const { bolts, nuts } = receiveScene(this.scene);
      this.pairs = bolts.flatMap((bolt, boltIndex) => nuts.map((nut, nutIndex) => ({
        boltIndex, nutIndex, coordinates: relationship(bolt, nut)
      })));
      this.feedback = this.pairs.flatMap(pair => relationIndexes(pair.coordinates)
        .map(index => this.readReference ? Math.PI * amplitude * this.returned[index] : 0));
    }
    return this.feedback?.[tick - decisionTime] ?? 0;
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);
    const [f0, b0, f1, b1] = this.lanePorts[this.nodeIndex++];
    const a = values[b0], b = values[b1], carrier = values[f0], returned = values[f1];
    const angle = this.angleAt(this.tick + 1, a, b, returned);
    const sine = Math.sin(angle), cosine = Math.cos(angle);
    values[f0] = (a + b) * Math.SQRT1_2;
    values[f1] = (a - b) * Math.SQRT1_2;
    values[b0] = carrier * cosine - returned * sine;
    values[b1] = carrier * sine + returned * cosine;
  }
}

function run(inputScene, readReference = true) {
  const sim = new AssemblySimulation(readReference);
  sim.reset = () => { throw new Error('The reference must remain in the sphere.'); };
  const { arcs, nodePorts } = sim.network;
  const [mixerForward0] = sim.lanePorts[setup.mixer];
  const [, , stationForward1] = sim.lanePorts[setup.station.node];
  const cPort = nodePorts[setup.station.node][stationForward1];
  const carrierArc = arcs[nodePorts[setup.mixer][mixerForward0]].reverse;
  const maxPairSymbols = 8;
  const inputs = [];
  for (const [i, { bolts, nuts }] of examples.entries()) {
    inputs.push([setup.sourceArcs[0], trainingTimes[i], encodePositions(bolts)],
      [setup.sourceArcs[1], trainingTimes[i], encodePositions(nuts)]);
  }
  inputs.push([carrierArc, referenceTime, new Float64Array(referenceLength).fill(amplitude)],
    [setup.sourceArcs[0], sceneTime, encodePositions(inputScene.bolts)],
    [setup.sourceArcs[1], sceneTime, encodePositions(inputScene.nuts)],
    [carrierArc, decisionTime, new Float64Array(maxPairSymbols).fill(amplitude)]);
  const transmit = inputs.map(([arc, tick, samples]) => ({ arc, launch: tick - arcs[arc].delay, samples }));
  const measurements = [[cPort, referenceTime + backwardsToProbe, referenceLength],
    [cPort, sceneTime + backwardsToProbe, referenceLength], [cPort, decisionTime + backwardsToProbe, maxPairSymbols],
    ...setup.station.probes.map(probe => [probe.receivingPort, sceneTime + probe.delay, sceneLength])]
    .map(([port, first, length]) => ({ port, first, trace: new Float64Array(length) }));
  const stop = Math.max(...measurements.map(reading => reading.first + reading.trace.length - 1));
  assert.ok(stop < referenceTime + 2 * loop);
  for (let tick = 1; tick <= stop; tick++) {
    sim.step();
    for (const { arc, launch, samples } of transmit) {
      const value = samples[tick - launch];
      if (value) sim.transmit(arc, value);
    }
    for (const { port, first, trace } of measurements) {
      const offset = tick - first;
      if (offset >= 0 && offset < trace.length) trace[offset] = sim.lastIncoming[port];
    }
  }

  const [reference, stillCirculating, decision, sum, difference] = measurements.map(reading => reading.trace);
  assert.equal(sim.trainingRelations, null);
  assert.deepEqual(reference, sim.returned);
  assert.deepEqual(stillCirculating, reference);
  const along = pulseIndex(reference, 0, relationshipBins, referenceLevel);
  const lateral = pulseIndex(reference, relationshipBins, relationshipBins, referenceLevel);
  assert.ok(along !== null && lateral !== null);
  const learned = { along: along - coordinates + 1, lateral: lateral - coordinates + 1 };
  const observedScene = receiveScene([
    sum.map((sample, i) => (sample + difference[i]) * Math.SQRT1_2),
    sum.map((sample, i) => (sample - difference[i]) * Math.SQRT1_2)
  ]);
  assert.deepEqual(observedScene, inputScene);
  const candidates = sim.pairs.map((pair, i) => {
    const cues = Array.from(decision.slice(i * 2, i * 2 + 2), sample => {
      const bit = Number(sample > decisionLevel / 2);
      assert.ok(Math.abs(sample - bit * decisionLevel) < tolerance);
      return bit;
    });
    return { ...pair, cues, score: cues[0] + cues[1] };
  });
  assert.equal(candidates.length, inputScene.bolts.length * inputScene.nuts.length);
  const energyError = sim.getStats().relativeError;
  assert.ok(energyError < 1e-12);
  return { learned, candidates, matches: candidates.filter(pair => pair.score === 2), energyError,
    readings: { reference: Array.from(reference), returned: Array.from(sim.returned), decision: Array.from(decision),
      sum: Array.from(sum), difference: Array.from(difference) } };
}

const main = run(scene);
const decoyOnly = run(decoy);
const disconnected = run(scene, false);
assert.deepEqual(main.learned, { along: 3, lateral: 0 });
assert.deepEqual(main.matches.map(pair => [pair.boltIndex, pair.nutIndex]), [[0, 1]]);
assert.deepEqual(decoyOnly.matches, []);
assert.deepEqual(disconnected.matches, []);
const summary = { examples, scene, learned: main.learned, matches: main.matches, main, controls: { decoyOnly, disconnected },
  experiment: { seed: setup.seed, mixer: setup.mixer, station: setup.station.node, trainingTimes, referenceTime,
    sceneTime, loop, decisionTime, referenceLevel, decisionLevel, geometrySuppliedAsPulses: true, preassignedScenePairs: false } };
const report = path.resolve(__dirname, '../outputs/sphere-assembly-measurements.json');
fs.writeFileSync(report, JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ learned: main.learned, candidates: main.candidates, decoyOnly: decoyOnly.candidates, energyError: main.energyError }));
