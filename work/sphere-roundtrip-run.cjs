'use strict';

const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { crc32, deflateSync } = require('node:zlib');
const { createNetwork } = require('./sphere-kernel.js');
const { Simulation: SeparatedLanes } = require('./sphere-separated-lanes-kernel.cjs');
const { encode } = require('./sphere-roundtrip-codec.cjs');

const SEED = 137531;
const AMPLITUDE = 0.5;
const MAX_FRAME_TICKS = (128 + 4) * 8;

class FileSimulation extends SeparatedLanes {
  mixLanes(values, first, second) {
    const a = values[first], b = values[second];
    values[first] = (a + b) * Math.SQRT1_2;
    values[second] = (a - b) * Math.SQRT1_2;
  }

  transmit(index, sample) {
    // Same additive source and work accounting as the base kernel, repeated at
    // two designated tails. No source state is recorded by the receiver.
    const arc = this.network.arcs[index];
    const tail = (arc.head + arc.delay - 1) % arc.delay;
    const before = arc.buffer[tail];
    const work = sample * sample + 2 * before * sample;
    arc.buffer[tail] = before + sample;
    this.lastOutgoing[index] = arc.buffer[tail];
    this.work += work;
    this.workMagnitude += Math.abs(work);
  }
}

function forwardTour(network, mixer, lane) {
  const { nodes, arcs, edges, nodePorts } = network;
  const route = new Map();
  let node = mixer, delay = 0;
  for (let hops = 1; hops <= nodes.length; hops++) {
    const index = nodePorts[node].find(port => {
      const { edge } = arcs[port];
      const onLane = lane === 0 ? edge < nodes.length : edge >= nodes.length;
      return onLane && edges[edge].ab === port;
    });
    const arc = arcs[index];
    delay += arc.delay;
    node = arc.to;
    assert.ok(!route.has(node));
    route.set(node, { from: arc.from, arc: index, receivingPort: arc.reverse, delay, hops });
  }
  assert.equal(node, mixer);
  return { route, lap: route.get(mixer) };
}

function createSetup() {
  const network = createNetwork({ seed: SEED });
  const mixer = network.arcs[network.sourcePort].to;
  const tours = [0, 1].map(lane => forwardTour(network, mixer, lane));
  const sourceArcs = tours.map(({ lap }) => lap.arc);
  assert.equal(sourceArcs[0], network.sourcePort);
  const noReturnBefore = Math.min(...tours.map(({ lap }) => lap.delay));
  const sourceDelays = sourceArcs.map(index => network.arcs[index].delay);
  const firstMixerTick = Math.max(...sourceDelays) + 1;
  const launch = sourceDelays.map(delay => firstMixerTick - delay);
  const farBand = JSON.parse(fs.readFileSync(path.join(__dirname, 'sphere-sondes-lointaines.json')));
  const origin = network.nodes[mixer];
  const candidates = farBand.nodes.map(node => {
    const probes = tours.map(({ route }) => route.get(node));
    const distances = probes.map(probe => probe.delay);
    const positionDistance = Math.hypot(...network.nodes[node].map((axis, i) => axis - origin[i]));
    const firstReturn = firstMixerTick + noReturnBefore + Math.min(...distances);
    const captureEnd = firstMixerTick + Math.max(...distances) + MAX_FRAME_TICKS - 1;
    return { node, position: network.nodes[node], probes, positionDistance, firstReturn, captureEnd };
  }).filter(candidate => candidate.captureEnd < candidate.firstReturn);
  candidates.sort((a, b) => b.positionDistance - a.positionDistance || a.node - b.node);
  assert.ok(candidates.length > 0);
  const station = candidates[0];
  for (const probe of station.probes) assert.ok(farBand.ports.includes(probe.receivingPort));
  return { seed: network.seed, nodes: network.nodes.length, links: network.edges.length,
    mixer, mixerPosition: origin, sourceArcs, sourceDelays, launch, firstMixerTick, station,
    loopDelays: tours.map(tour => tour.lap.delay) };
}

function beginRecording(setup, ticks) {
  const sim = new FileSimulation(createNetwork({ seed: SEED }), { writing: false });
  assert.ok(sim.angles.every(value => value === 0));
  const traces = setup.station.probes.map(() => new Float64Array(ticks));
  return { sim, traces };
}

function continueRecording(setup, { sim, traces }, streams, mixerTick, until) {
  assert.ok(until >= sim.tick && until <= traces[0].length);
  const launch = setup.sourceDelays.map(delay => mixerTick - delay);
  assert.ok(launch.every(tick => tick > sim.tick));
  for (let tick = sim.tick + 1; tick <= until; tick++) {
    sim.step();
    for (const [index, stream] of streams.entries()) {
      const sample = stream[tick - launch[index]];
      if (sample !== undefined && sample !== 0) sim.transmit(setup.sourceArcs[index], sample);
    }
    for (const [index, probe] of setup.station.probes.entries()) {
      traces[index][tick - 1] = sim.lastIncoming[probe.receivingPort];
    }
  }
  const stats = sim.getStats();
  assert.ok(stats.relativeError < 1e-12);
  return stats;
}

function record(setup, streams, ticks) {
  assert.ok(ticks < setup.station.firstReturn, 'The capture must end before possible returns.');
  const recording = beginRecording(setup, ticks);
  const stats = continueRecording(setup, recording, streams, setup.firstMixerTick, ticks);
  const { traces } = recording;
  return { traces, stats };
}

function calibrate(setup) {
  const ticks = setup.station.captureEnd - MAX_FRAME_TICKS + 1;
  const matrix = [[], []];
  const firstTicks = [];
  const energyErrors = [];
  for (let source = 0; source < 2; source++) {
    const impulse = Array.from({ length: 2 }, (_, i) => i === source ? [AMPLITUDE] : []);
    const calibration = record(setup, impulse, ticks);
    energyErrors.push(calibration.stats.relativeError);
    for (const [index, trace] of calibration.traces.entries()) {
      const first = trace.findIndex(value => value !== 0) + 1;
      assert.equal(first, setup.firstMixerTick + setup.station.probes[index].delay);
      if (source === 0) firstTicks[index] = first;
      else assert.equal(firstTicks[index], first);
      assert.equal(trace.filter(value => value !== 0).length, 1);
      matrix[index][source] = trace[first - 1] / AMPLITUDE;
    }
  }
  const determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
  assert.ok(matrix.every(row => row.every(value => Math.abs(value) > 0.7)));
  assert.ok(Math.abs(Math.abs(determinant) - 1) < 1e-14);
  return { firstTicks, matrix, determinant, energyErrors };
}

function pngChunk(type, data) {
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const length = Buffer.alloc(4), checksum = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  checksum.writeUInt32BE(crc32(body));
  return Buffer.concat([length, body, checksum]);
}

function makeImage(pixelAt = (row, col) => row === col || row + col === 7 ? 255 : 0) {
  const dimension = 8;
  const header = Buffer.alloc(13);
  header.writeUInt32BE(dimension, 0);
  header.writeUInt32BE(dimension, 4);
  header[8] = 8;
  const pixels = Buffer.alloc(dimension * (1 + dimension));
  for (let row = 0; row < dimension; row++) {
    for (let col = 0; col < dimension; col++) {
      pixels[row * (1 + dimension) + col + 1] = pixelAt(row, col);
    }
  }
  return Buffer.concat([Buffer.from('89504e470d0a1a0a', 'hex'), pngChunk('IHDR', header),
    pngChunk('IDAT', deflateSync(pixels)), pngChunk('IEND', Buffer.alloc(0))]);
}

function runFirstRoundTrip() {
  const setup = createSetup();
  const calibration = calibrate(setup);
  const scratch = fs.mkdtempSync(path.join(__dirname, 'sphere-roundtrip-'));
  const receiverDir = path.join(scratch, 'receiver');
  const senderDir = path.join(scratch, 'sender');
  const outputs = path.resolve(__dirname, '../outputs');
  fs.mkdirSync(receiverDir);
  fs.mkdirSync(senderDir);
  const originals = [Buffer.from('La sphère tourne.\n', 'utf8'), makeImage()];
  const sourceNames = ['original.txt', 'original.png'];
  for (const [i, original] of originals.entries()) fs.writeFileSync(path.join(senderDir, sourceNames[i]), original);
  const streams = sourceNames.map(name => encode(fs.readFileSync(path.join(senderDir, name)), AMPLITUDE));
  const length = Math.max(...streams.map(stream => stream.length));
  assert.ok(length <= MAX_FRAME_TICKS);
  const totalTicks = Math.max(...calibration.firstTicks) + length - 1;
  const transmission = record(setup, streams, totalTicks);
  const capture = { amplitude: AMPLITUDE, firstTicks: calibration.firstTicks, matrix: calibration.matrix,
    traces: transmission.traces.map(trace => Array.from(trace)) };
  const captureName = path.join(receiverDir, 'sondes.json');
  fs.writeFileSync(captureName, JSON.stringify(capture) + '\n');

  // Separate process. The receiver is given only its two signed measurements,
  // the calibrated gain matrix and the two measured first-arrival ticks.
  const receiverProgram = path.join(receiverDir, 'decoder.cjs');
  fs.copyFileSync(path.join(__dirname, 'sphere-roundtrip-codec.cjs'), receiverProgram);
  const receiverLog = JSON.parse(execFileSync(process.execPath, [receiverProgram, captureName, receiverDir],
    { cwd: receiverDir, encoding: 'utf8' }));

  let bothTransmitting = 0, cancellingAtSum = 0, cancellingAtDifference = 0, maxCaptureResidual = 0;
  for (let index = 0; index < length; index++) {
    const a = streams[0][index] ?? 0, b = streams[1][index] ?? 0;
    const measured = transmission.traces.map((trace, lane) => trace[calibration.firstTicks[lane] - 1 + index]);
    for (const [lane, row] of calibration.matrix.entries()) {
      maxCaptureResidual = Math.max(maxCaptureResidual, Math.abs(measured[lane] - (row[0] * a + row[1] * b)));
    }
    if (index >= 32 && a !== 0 && b !== 0) {
      bothTransmitting++;
      if (measured[0] === 0) cancellingAtSum++;
      if (measured[1] === 0) cancellingAtDifference++;
    }
  }
  assert.ok(maxCaptureResidual < 1e-13);
  assert.ok(cancellingAtSum > 0 && cancellingAtDifference > 0);

  const digest = data => createHash('sha256').update(data).digest('hex');
  const receivedNames = ['texte-reconstitue.txt', 'image-reconstituee.png'];
  const files = receivedNames.map((name, index) => {
    const original = fs.readFileSync(path.join(senderDir, sourceNames[index]));
    const received = fs.readFileSync(path.join(receiverDir, name));
    const byteForByte = received.equals(original);
    assert.ok(byteForByte, `Le fichier ${name} a changé.`);
    return { original: sourceNames[index], received: name, bytes: received.length,
      originalSHA256: digest(original), receivedSHA256: digest(received), byteForByte };
  });
  const detail = { setup, calibration, capture, receiverLog, files, totalTicks, bothTransmitting,
    cancellingAtSum, cancellingAtDifference, maxCaptureResidual,
    initialWaveEnergy: 0, finalEnergy: transmission.stats.energy, measuredRelativeEnergyError: transmission.stats.relativeError };
  for (const name of receivedNames) fs.copyFileSync(path.join(receiverDir, name), path.join(outputs, `sphere-${name}`));
  fs.writeFileSync(path.join(outputs, 'sphere-aller-retour-mesures.json'), JSON.stringify(detail, null, 2) + '\n');
  console.log(JSON.stringify({ setup, calibration, files, totalTicks, bothTransmitting, cancellingAtSum,
    cancellingAtDifference, maxCaptureResidual, receiverLog, measuredRelativeEnergyError: transmission.stats.relativeError }, null, 2));
}

if (require.main === module) runFirstRoundTrip();

module.exports = { FileSimulation, AMPLITUDE, MAX_FRAME_TICKS, createSetup, beginRecording, continueRecording, makeImage };
