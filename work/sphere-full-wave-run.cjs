'use strict';

const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const { Worker, isMainThread, parentPort, workerData } = require('node:worker_threads');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation, createSetup } = require('./sphere-roundtrip-run.cjs');
const { AlignmentCircuit } = require('./sphere-pattern-internal-delay-run.cjs');

const AMPLITUDE = 0.5;
const DELAY_FACTOR = 3;
const LIMIT = 5;
const PACKET_SIZE = 1 + 7 * LIMIT;
const base = createSetup();
const loops = base.loopDelays.map(delay => delay * DELAY_FACTOR);
const start = Math.max(...base.sourceDelays) * DELAY_FACTOR + 1;
const probeDelay = loops[1] - base.station.probes[1].delay * DELAY_FACTOR;

function slowerNetwork() {
  const network = createNetwork({ seed: base.seed });
  for (const edge of network.edges) edge.delay *= DELAY_FACTOR;
  for (const arc of network.arcs) {
    arc.delay *= DELAY_FACTOR;
    arc.buffer = new Float64Array(arc.delay);
  }
  return network;
}

function encode(pixels) {
  return Float64Array.from(pixels, value => AMPLITUDE * value / 255);
}

function decodeReference(value) {
  // C carries the coincident product of the two luminance amplitudes.
  const angle = Math.asin(Math.max(0, Math.min(1, value / AMPLITUDE)));
  return Math.sqrt(angle / Math.PI) / AMPLITUDE;
}

function pack(rows, width, height, count) {
  assert.ok(rows.length <= LIMIT);
  const packet = new Float64Array(PACKET_SIZE);
  packet[0] = rows.length / LIMIT;
  for (const [index, row] of rows.entries()) {
    const [x, y, w, h] = row.box;
    packet.set([x / width, y / height, w / width, h / height,
      Math.round(row.score * 1e6) / 1e6, row.reference / count, row.angle / 360], 1 + index * 7);
  }
  assert.ok(packet.every(value => value >= 0 && value <= 1));
  return packet;
}

function unpack(wave, width, height, count) {
  const payload = wave.map(value => value / AMPLITUDE);
  return Array.from({length: Math.round(payload[0] * LIMIT)}, (_, index) => {
    const [x, y, w, h, score, reference, angle] = payload.slice(1 + index * 7, 8 + index * 7);
    return {box: [Math.round(x * width), Math.round(y * height), Math.round(w * width), Math.round(h * height)],
      score: Math.round(score * 1e6) / 1e6, reference: Math.round(reference * count), angle: Math.round(angle * 360)};
  });
}

class FullFrameJunction {
  constructor(width, height, side, count, frameAt, referenceLength, cut, match) {
    this.width = width;
    this.height = height;
    this.side = side;
    this.count = count;
    this.start = frameAt;
    this.end = frameAt + width * height;
    this.cut = cut;
    this.match = match;
    this.reference = new Float64Array(referenceLength);
    this.scene = new Float64Array(width * height);
    this.packet = new Float64Array(PACKET_SIZE);
  }

  receive(tick, reference, pixel) {
    const elapsed = tick - this.start;
    if (elapsed < this.reference.length) this.reference[elapsed] = this.cut ? 0 : reference;
    if (tick < this.end) this.scene[elapsed] = pixel;
    if (tick === this.end - 1 && this.reference.some(Boolean)) {
      const cells = this.side ** 2;
      const templates = Array.from({length: this.count}, (_, i) =>
        this.reference.subarray(i * cells, (i + 1) * cells).map(decodeReference));
      this.packet = pack(this.match(templates, this.scene.map(value => value / AMPLITUDE)),
        this.width, this.height, this.count);
    }
    const value = this.packet[tick - this.end];
    return value ? Math.asin(value) : 0;
  }
}

class FullFrameSimulation extends FileSimulation {
  constructor(input, cut, match) {
    super(slowerNetwork(), { writing: false });
    const size = input.referenceSize ** 2 * input.references.length;
    // The isolated training examples are centered before encoding. A single
    // coincidence tap per sample forms C for each distinct pair in the bank.
    this.alignment = new AlignmentCircuit(false, start, {span: size, aDelay: 0, tapDelays: Int32Array.of(0)});
    this.emission = start + this.alignment.emitAt;
    this.sceneArrival = this.emission + loops[1];
    this.junction = new FullFrameJunction(...input.frameSize, input.referenceSize, input.references.length,
      this.sceneArrival, size, cut, match);
    this.workCorrection = 0;
  }

  transmit(index, sample) {
    const previousWork = this.work;
    const arc = this.network.arcs[index];
    const tail = (arc.head + arc.delay - 1) % arc.delay;
    const contribution = sample * sample + 2 * arc.buffer[tail] * sample;
    super.transmit(index, sample);
    // Long, nearly constant image rasters magnify ordinary addition error.
    const adjusted = contribution - this.workCorrection;
    this.work = previousWork + adjusted;
    this.workCorrection = (this.work - previousWork) - adjusted;
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);
    const [forward0, backward0, forward1, backward1] = this.lanePorts[this.nodeIndex++];
    const a = values[backward0], b = values[backward1];
    const carrier = values[forward0], returned = values[forward1];
    const tick = this.tick + 1;
    const angle = tick < this.sceneArrival ? this.alignment.receive(tick, a, b) : this.junction.receive(tick, returned, b);
    values[forward0] = (a + b) * Math.SQRT1_2;
    values[forward1] = (a - b) * Math.SQRT1_2;
    values[backward0] = carrier * Math.cos(angle) - returned * Math.sin(angle);
    values[backward1] = carrier * Math.sin(angle) + returned * Math.cos(angle);
  }
}

function runFrame(input, frame, cut, match) {
  const sim = new FullFrameSimulation(input, cut, match);
  const { arcs, nodePorts } = sim.network;
  const [width, height] = input.frameSize;
  assert.equal(frame.pixels.length, width * height);
  const a = encode(input.references.flatMap(pair => pair.a));
  const b = encode(input.references.flatMap(pair => pair.b));
  const photo = encode(frame.pixels);
  const carrier = arcs[nodePorts[base.mixer][sim.lanePorts[base.mixer][0]]].reverse;
  const probe = nodePorts[base.station.node][sim.lanePorts[base.station.node][2]];
  const outcomeAt = sim.junction.end;
  assert.equal(a.length, b.length);
  assert.ok(a.length <= photo.length && a.length + photo.length + PACKET_SIZE < Math.min(...loops),
    'An input echo would overlap the observation window');
  const injection = [
    [base.sourceArcs[0], start, a], [base.sourceArcs[1], start, b],
    [carrier, sim.emission, new Float64Array(a.length).fill(AMPLITUDE)],
    [base.sourceArcs[1], sim.sceneArrival, photo],
    [carrier, outcomeAt, new Float64Array(PACKET_SIZE).fill(AMPLITUDE)],
  ].map(([arc, at, wave]) => ({arc, start: at - arcs[arc].delay, wave}));
  const c = new Float64Array(a.length);
  const result = new Float64Array(PACKET_SIZE);
  const captureC = sim.emission + probeDelay;
  const captureResult = outcomeAt + probeDelay;
  const stop = captureResult + PACKET_SIZE - 1;
  for (let tick = 1; tick <= stop; tick++) {
    sim.step();
    for (const burst of injection) {
      const sample = burst.wave[tick - burst.start];
      if (sample) sim.transmit(burst.arc, sample);
    }
    if (tick >= captureC && tick < captureC + c.length) c[tick - captureC] = sim.lastIncoming[probe];
    if (tick >= captureResult) result[tick - captureResult] = sim.lastIncoming[probe];
  }
  assert.deepEqual(sim.junction.scene, photo, 'The junction must receive the complete encoded scene');
  assert.deepEqual(sim.junction.reference, cut ? new Float64Array(c.length) : c);
  assert.ok(c.length && c.some(Boolean));
  assert.ok(c.every((value, index) => Math.abs(value - AMPLITUDE * Math.sin(Math.PI * a[index] * b[index])) < 1e-13));
  assert.ok(result.every((value, i) => Math.abs(value / AMPLITUDE - sim.junction.packet[i]) < 1e-13));
  if (cut) assert.equal(result.some(Boolean), false);
  const stats = sim.getStats();
  assert.ok(stats.relativeError < 1e-12, `Relative energy error ${stats.relativeError}`);
  const output = unpack(result, width, height, input.references.length);
  const [sourceWidth, sourceHeight] = frame.originalSize;
  const detections = output.map(row => ({...row, box: row.box.map((v, i) =>
    Math.round(v * (i % 2 ? sourceHeight / height : sourceWidth / width)))}));
  return {id: frame.id, detections, probe: {cSamples: c.filter(Boolean).length,
    cSHA256: createHash('sha256').update(Buffer.from(c.buffer)).digest('hex'),
    sceneSamples: photo.length, resultSamples: result.filter(Boolean).length,
    cAt: captureC, cReturnsAt: sim.sceneArrival, resultAt: captureResult, energyError: stats.relativeError}};
}

function createMatcher(input) {
  return require('./sphere-full-wave-match.cjs').createMatcher(...input.frameSize, input.referenceSize, input.references.length);
}

async function main() {
  const [inputPath, outputPath, mode, ...ids] = process.argv.slice(2);
  const {frames, ...input} = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const cut = mode === 'cut';
  const selected = ids.length ? frames.filter(frame => ids.includes(frame.id)) : frames;
  assert.ok(selected.length && (!ids.length || ids.length === selected.length));
  const workers = Math.max(1, Math.min(Number.parseInt(process.env.SPHERE_WAVE_WORKERS ?? '4', 10) || 4, selected.length));
  const batches = Array.from({length: workers}, (_, i) => selected.filter((_, n) => n % workers === i));
  const rows = await Promise.all(batches.map(batch => new Promise((resolve, reject) => {
    const worker = new Worker(__filename, {workerData: {input, batch, cut}});
    worker.on('message', resolve);
    worker.on('error', reject);
    worker.on('exit', code => { if (code !== 0) reject(new Error(`Worker exited with status ${code}`)); });
  })));
  const found = new Map(rows.flat().map(row => [row.id, row]));
  const ordered = selected.map(frame => found.get(frame.id));
  const report = {frameSize: input.frameSize, referencePairs: input.references.map(pair => pair.frames),
    delayFactor: DELAY_FACTOR, loopTicks: loops, cut, frames: ordered};
  fs.writeFileSync(outputPath, JSON.stringify(report, null, 2) + '\n');
}

if (isMainThread && require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
if (!isMainThread) {
  const {input, batch, cut} = workerData;
  const match = cut ? null : createMatcher(input);
  parentPort.postMessage(batch.map(frame => {
    const result = runFrame(input, frame, cut, match);
    console.log(frame.id, result.detections[0]?.score ?? 'no match');
    return result;
  }));
}

module.exports = { runFrame };
