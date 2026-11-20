'use strict';

const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation, createSetup } = require('./sphere-roundtrip-run.cjs');
const { observe } = require('./sphere-assembly-photo-vision.cjs');

const outputs = path.resolve(__dirname, '../outputs');
const [maskFile = 'sphere-assembly-photo-masks.json', reportFile = 'sphere-assembly-photo-measurements.json', scenePrompt,
  mode] = process.argv.slice(2);
assert.ok(mode === undefined || mode === 'multiple');
const multiple = mode === 'multiple';
const prepared = JSON.parse(fs.readFileSync(path.join(__dirname, maskFile)));
const photo = fs.readFileSync(path.join(outputs, prepared.sourceFile));
assert.equal(createHash('sha256').update(photo).digest('hex'), prepared.sourceSHA256);
if (prepared.sceneFile) {
  assert.equal(createHash('sha256').update(fs.readFileSync(path.join(outputs, prepared.sceneFile))).digest('hex'), prepared.sceneSHA256);
}
const { size, cropMargin, threshold, masks } = prepared;
const setup = createSetup();
const amplitude = 0.5;
const photoLength = size ** 2;
const decisionSymbols = multiple ? 36 : 8;
const stepsPerLength = 4;
const limit = 32;
const bins = 2 * limit + 1;
const templateLength = 2 * bins;
const trainingSlot = 100 * Math.ceil((photoLength + 20) / 100);
const trainingStarts = [setup.firstMixerTick, trainingSlot];
const templateStart = 2 * trainingSlot;
const loop = setup.loopDelays[1];
const sceneStart = templateStart + loop;
const matchStart = sceneStart + photoLength;
const backwardsToProbe = loop - setup.station.probes[1].delay;
const templateLevel = amplitude * Math.sin(Math.PI * amplitude ** 2);
const matchLevel = amplitude * Math.sin(Math.PI * amplitude * templateLevel);

function decodeMask(wave) {
  return Array.from(wave, sample => {
    const active = Number(sample >= amplitude / 2);
    assert.ok(Math.abs(sample - active * amplitude) < 1e-13);
    return active;
  });
}

function relation(bolt, nut) {
  const x = nut.x - bolt.x, y = nut.y - bolt.y;
  const [dx, dy] = bolt.axis;
  return { along: (dx * x + dy * y) / bolt.length, lateral: (dx * y - dy * x) / bolt.length };
}

function quantize({ along, lateral }) {
  return [Math.round(stepsPerLength * along), Math.round(stepsPerLength * lateral)];
}

function relationSlot(dimension, value) {
  if (Math.abs(value) > limit) return -1;
  return dimension * bins + value + limit;
}

function allowedSlots(quantized) {
  // A bin of tolerance on either side accounts for reading a sampled image.
  return new Set(quantized.flatMap((value, i) => [-1, 0, 1].map(offset => relationSlot(i, value + offset)))
    .filter(index => index >= 0));
}

class PhotoAssemblySimulation extends FileSimulation {
  constructor(readReference) {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    this.exampleImages = trainingStarts.map(() => new Float64Array(photoLength));
    this.sceneImage = new Float64Array(photoLength);
    this.returned = new Float64Array(templateLength);
    this.readReference = readReference;
  }

  angleAt(tick, imageSample, returningWave) {
    if (this.exampleImages) {
      for (const [i, start] of trainingStarts.entries()) {
        const offset = tick - start;
        if (offset >= 0 && offset < photoLength) this.exampleImages[i][offset] = imageSample;
      }
      if (tick === templateStart) {
        this.training = this.exampleImages.map(wave => observe(decodeMask(wave), size));
        this.trainingRelations = this.training.map(({ bolts, nuts }) => {
          assert.equal(bolts.length, 1);
          assert.equal(nuts.length, 1);
          return relation(bolts[0], nuts[0]);
        });
        this.supports = this.trainingRelations.map(relative => allowedSlots(quantize(relative)));
        this.exampleImages = null;
      }
    }
    const templateOffset = tick - templateStart;
    if (templateOffset >= 0 && templateOffset < templateLength) {
      return this.supports.every(support => support.has(templateOffset)) ? Math.PI * amplitude ** 2 : 0;
    }
    if (templateOffset === templateLength) this.supports = null;

    const sceneOffset = tick - sceneStart;
    if (sceneOffset >= 0 && sceneOffset < photoLength) {
      this.sceneImage[sceneOffset] = imageSample;
      if (sceneOffset < templateLength) this.returned[sceneOffset] = returningWave;
    }
    if (tick === matchStart) {
      this.scene = observe(decodeMask(this.sceneImage), size);
      const { bolts, nuts } = this.scene;
      this.pairs = bolts.flatMap((bolt, boltIndex) => nuts.map((nut, nutIndex) => {
        const offset = relation(bolt, nut);
        return { boltIndex, nutIndex, offset, quantized: quantize(offset) };
      }));
      this.responses = this.pairs.flatMap(({ quantized }) => quantized.map((value, i) =>
        this.readReference ? Math.PI * amplitude * (this.returned[relationSlot(i, value)] ?? 0) : 0));
    }
    return this.responses?.[tick - matchStart] ?? 0;
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);
    const [f0, b0, f1, b1] = this.lanePorts[this.nodeIndex++];
    const a = values[b0], b = values[b1], carrier = values[f0], returned = values[f1];
    const theta = this.angleAt(this.tick + 1, b, returned);
    const c = Math.cos(theta), s = Math.sin(theta);
    values[f0] = (a + b) * Math.SQRT1_2;
    values[f1] = (a - b) * Math.SQRT1_2;
    values[b0] = carrier * c - returned * s;
    values[b1] = carrier * s + returned * c;
  }
}

function run(readReference) {
  const sim = new PhotoAssemblySimulation(readReference);
  sim.reset = () => { throw Error('The reference must remain in the network.'); };
  const { arcs, nodePorts } = sim.network;
  const [forwardAtMixer] = sim.lanePorts[setup.mixer];
  const [, , forwardAtStation] = sim.lanePorts[setup.station.node];
  const carrier = arcs[nodePorts[setup.mixer][forwardAtMixer]].reverse;
  const cPort = nodePorts[setup.station.node][forwardAtStation];
  const caseStarts = [...trainingStarts, sceneStart];
  const transmitted = masks.map((mask, i) => [setup.sourceArcs[1], caseStarts[i], Float64Array.from(mask, bit => bit * amplitude)]);
  transmitted.push([carrier, templateStart, new Float64Array(templateLength).fill(amplitude)],
    [carrier, matchStart, new Float64Array(decisionSymbols).fill(amplitude)]);
  const launches = transmitted.map(([arc, arrival, samples]) => ({ arc, launch: arrival - arcs[arc].delay, samples }));
  assert.ok(launches.every(source => source.launch >= 1));
  const records = [[cPort, templateStart + backwardsToProbe, templateLength],
    [cPort, sceneStart + backwardsToProbe, templateLength], [cPort, matchStart + backwardsToProbe, decisionSymbols],
    ...setup.station.probes.map(probe => [probe.receivingPort, sceneStart + probe.delay, photoLength])]
    .map(([port, first, length]) => ({ port, first, samples: new Float64Array(length) }));
  const end = Math.max(...records.map(({ first, samples }) => first + samples.length - 1));
  assert.ok(end < templateStart + 2 * loop);
  for (let tick = 1; tick <= end; tick++) {
    sim.step();
    for (const { arc, launch, samples } of launches) {
      const sample = samples[tick - launch];
      if (sample) sim.transmit(arc, sample);
    }
    for (const { port, first, samples } of records) {
      const offset = tick - first;
      if (offset >= 0 && offset < samples.length) samples[offset] = sim.lastIncoming[port];
    }
  }
  const [learned, circulating, decisions, sum, difference] = records.map(record => record.samples);
  assert.equal(sim.exampleImages, null);
  assert.equal(sim.supports, null);
  assert.deepEqual(learned, sim.returned);
  assert.deepEqual(learned, circulating);
  assert.deepEqual(decodeMask(sim.sceneImage), masks[2]);
  assert.deepEqual(decodeMask(sum.map((v, i) => (v - difference[i]) * Math.SQRT1_2)), masks[2]);
  assert.ok(2 * sim.pairs.length <= decisionSymbols);
  const matches = sim.pairs.map((pair, i) => {
    const cues = Array.from(decisions.slice(2 * i, 2 * i + 2), value => {
      const active = Number(value > matchLevel / 2);
      assert.ok(Math.abs(value - active * matchLevel) < 1e-13);
      return active;
    });
    return { ...pair, cues, score: cues.reduce((total, cue) => total + cue, 0) };
  });
  const decodedReference = [[], []];
  for (const [index, sample] of learned.entries()) {
    if (sample < templateLevel / 2) continue;
    assert.ok(Math.abs(sample - templateLevel) < 1e-13);
    decodedReference[Math.floor(index / bins)].push((index % bins - limit) / stepsPerLength);
  }
  const energyError = sim.getStats().relativeError;
  assert.ok(energyError < 1e-12);
  return { training: sim.training, trainingRelations: sim.trainingRelations, scene: sim.scene, candidates: matches,
    matches: matches.filter(match => match.score === 2), decodedReference, energyError,
    readings: { learned: Array.from(learned), returned: Array.from(sim.returned), decision: Array.from(decisions) } };
}

const main = run(true), disconnected = run(false);
assert.equal(main.scene.bolts.length, multiple ? 3 : (prepared.sceneFile ? 1 : 2));
assert.equal(main.scene.nuts.length, multiple ? 6 : 2);
assert.deepEqual(disconnected.matches, []);
const data = { size, cropMargin, threshold, sourceFile: prepared.sourceFile, sourceSHA256: prepared.sourceSHA256,
  sourceGeneration: { tool: 'ImageGen', prompt: fs.readFileSync(path.join(__dirname, 'sphere-assembly-photo-prompt.txt'), 'utf8').trim() },
  ...main, controls: { disconnected }, experiment: { trainingStarts, templateStart, sceneStart, matchStart, loop,
    backwardsToProbe, templateLevel, matchLevel, stepsPerLength, bins, limit, masksOnlyAtInput: true }, masks };
if (prepared.sceneFile) {
  data.sceneFile = prepared.sceneFile;
  data.sceneSHA256 = prepared.sceneSHA256;
  if (scenePrompt) data.sceneGeneration = { tool: 'ImageGen', prompt: fs.readFileSync(path.join(__dirname, scenePrompt), 'utf8').trim() };
  data.automaticCenterDistances = main.candidates.map(({ boltIndex, nutIndex }) => {
    const bolt = main.scene.bolts[boltIndex], nut = main.scene.nuts[nutIndex];
    return Math.hypot(nut.x - bolt.x, nut.y - bolt.y);
  });
}
if (multiple) {
  assert.equal(main.matches.length, main.scene.bolts.length);
  assert.equal(new Set(main.matches.map(pair => pair.boltIndex)).size, main.scene.bolts.length);
  const matchedNuts = new Set(main.matches.map(pair => pair.nutIndex));
  assert.equal(matchedNuts.size, main.matches.length);
  data.distanceControls = main.matches.map(({ boltIndex, nutIndex: matchedNutIndex }) => {
    const bolt = main.scene.bolts[boltIndex];
    const distanceTo = nut => Math.hypot(nut.x - bolt.x, nut.y - bolt.y);
    const targetDistance = distanceTo(main.scene.nuts[matchedNutIndex]);
    const candidates = main.scene.nuts.map((nut, index) => ({ index, distance: distanceTo(nut) }))
      .filter(({ index }) => !matchedNuts.has(index))
      .sort((a, b) => Math.abs(a.distance - targetDistance) - Math.abs(b.distance - targetDistance));
    const { index: decoyNutIndex, distance: decoyDistance } = candidates[0];
    assert.ok(Math.abs(targetDistance - decoyDistance) < 1);
    assert.equal(Math.round(stepsPerLength * targetDistance / bolt.length), Math.round(stepsPerLength * decoyDistance / bolt.length));
    assert.equal(main.candidates.find(pair => pair.boltIndex === boltIndex && pair.nutIndex === decoyNutIndex).score, 0);
    return { boltIndex, matchedNutIndex, decoyNutIndex, targetDistance, decoyDistance };
  });
  assert.equal(new Set(data.distanceControls.map(control => control.decoyNutIndex)).size, main.scene.bolts.length);
}
fs.writeFileSync(path.join(outputs, reportFile), JSON.stringify(data, null, 2) + '\n');
console.log(JSON.stringify({ training: main.trainingRelations, learned: main.decodedReference, scene: main.scene,
  candidates: main.candidates, matches: main.matches, energyError: main.energyError }, null, 2));
