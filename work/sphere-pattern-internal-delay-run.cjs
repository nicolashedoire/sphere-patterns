'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation, createSetup } = require('./sphere-roundtrip-run.cjs');

const SIZE = 7;
const STRIDE = 2 * SIZE - 1;
const PIXEL_TICKS = Array.from({ length: SIZE ** 2 }, (_, i) => Math.floor(i / SIZE) * STRIDE + i % SIZE);
const SPAN = PIXEL_TICKS.at(-1) + 1;
const A_DELAY = (SIZE - 1) * (STRIDE + 1);
const TAP_COUNT = 2 * A_DELAY + 1;
const EMIT_C = SPAN + A_DELAY;
const AMPLITUDE = 0.5;
const C_LEVEL = AMPLITUDE * Math.sin(Math.PI * AMPLITUDE ** 2);
const EPSILON = 1e-13;
const setup = createSetup();
const START = setup.firstMixerTick;

function coincidenceAngle(a, b) {
  return Math.PI * a * b;
}

function gridReadout(turns, size, span) {
  const stride = 2 * size - 1;
  const addresses = new Int32Array(span).fill(-1);
  for (let i = 0; i < size ** 2; i++) {
    let x = i % size, y = Math.floor(i / size);
    const originalTick = y * stride + x;
    for (let turn = 0; turn < turns; turn++) [x, y] = [size - 1 - y, x];
    addresses[y * stride + x] = originalTick;
  }
  return addresses;
}

class AlignmentCircuit {
  constructor(allowRotation, start = START, options = {}) {
    // This is the local control circuit's delay memory. Reading a tap does
    // not copy a full-energy wave into the sphere. C is carried by a separately
    // injected and energy-accounted wave on the backward circuit.
    const size = options.size ?? (options.span === undefined ? SIZE : null);
    this.span = options.span ?? ((size - 1) * (2 * size - 1) + size);
    this.aDelay = options.aDelay ?? (size - 1) * (2 * size);
    this.tapDelays = options.tapDelays ?? Int32Array.from({ length: 2 * this.aDelay + 1 }, (_, i) => i);
    this.a = new Float64Array(this.span);
    this.b = new Float64Array(this.span);
    this.start = start;
    this.readouts = size === null ? [null] : (allowRotation ? [0, 1, 2, 3] : [0])
      .map(turns => gridReadout(turns, size, this.span));
    // Rotation can change which row arrives first. The rotated readouts wait
    // until the entire input frame has reached this junction.
    this.wait = allowRotation ? this.span : 0;
    this.emitAt = this.wait + this.span + this.aDelay;
    this.accumulated = new Float64Array(this.readouts.length * this.tapDelays.length);
    this.selectedTap = null;
    this.selectedQuarterTurns = null;
    this.selectionTick = null;
  }

  readB(orientation, index) {
    const readout = this.readouts[orientation];
    return this.b[readout ? readout[index] : index] ?? 0;
  }

  receive(tick, a, b) {
    const elapsed = tick - this.start;
    if (elapsed >= 0 && elapsed < this.span) {
      this.a[elapsed] = a;
      this.b[elapsed] = b;
    }

    const comparisonTick = elapsed - this.wait;
    const aIndex = comparisonTick - this.aDelay;
    if (aIndex >= 0 && aIndex < this.span && this.a[aIndex] !== 0) {
      for (let orientation = 0; orientation < this.readouts.length; orientation++) {
        const base = orientation * this.tapDelays.length;
        for (let tap = 0; tap < this.tapDelays.length; tap++) {
          const bDelayed = this.readB(orientation, comparisonTick - this.tapDelays[tap]);
          const tapAmplitude = AMPLITUDE * Math.sin(coincidenceAngle(this.a[aIndex], bDelayed));
          this.accumulated[base + tap] += tapAmplitude * tapAmplitude;
        }
      }
    }
    if (elapsed === this.emitAt - 1) this.latch(tick);

    const outputIndex = elapsed - this.emitAt;
    if (this.selectedTap === null || outputIndex < 0 || outputIndex >= this.span) return 0;
    const bAligned = this.readB(this.selectedQuarterTurns, outputIndex + this.aDelay - this.tapDelays[this.selectedTap]);
    return coincidenceAngle(this.a[outputIndex], bAligned);
  }

  latch(tick) {
    const max = Math.max(...this.accumulated);
    const winners = [...this.accumulated.keys()].filter(tap => Math.abs(this.accumulated[tap] - max) < EPSILON);
    if (max > EPSILON && winners.length === 1) {
      this.selectedTap = winners[0] % this.tapDelays.length;
      this.selectedQuarterTurns = Math.floor(winners[0] / this.tapDelays.length);
    }
    this.selectionTick = tick;
  }
}

class AutoAlignmentSimulation extends FileSimulation {
  constructor(allowRotation) {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    this.alignment = new AlignmentCircuit(allowRotation);
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);

    const [forward0, backward0, forward1, backward1] = this.lanePorts[this.nodeIndex++];
    const a = values[backward0], b = values[backward1];
    const carrier = values[forward0], side = values[forward1];
    const angle = this.alignment.receive(this.tick + 1, a, b);
    values[forward0] = (a + b) * Math.SQRT1_2;
    values[forward1] = (a - b) * Math.SQRT1_2;
    values[backward0] = carrier * Math.cos(angle) - side * Math.sin(angle);
    values[backward1] = carrier * Math.sin(angle) + side * Math.cos(angle);
  }
}

function encode(pixels) {
  assert.equal(pixels.length, PIXEL_TICKS.length);
  const samples = new Float64Array(SPAN);
  PIXEL_TICKS.forEach((tick, i) => { samples[tick] = AMPLITUDE * pixels[i]; });
  return samples;
}

function decode(samples, level) {
  return PIXEL_TICKS.map(tick => {
    const bit = Number(samples[tick] > level / 2);
    assert.ok(Math.abs(samples[tick] - bit * level) < EPSILON);
    return bit;
  });
}

function run(patternA, patternB, allowRotation = false) {
  const sim = new AutoAlignmentSimulation(allowRotation);
  const emitC = sim.alignment.emitAt;
  const { arcs, nodePorts } = sim.network;
  const [mixerForward0] = sim.lanePorts[setup.mixer];
  const [, , stationForward1] = sim.lanePorts[setup.station.node];
  const carrierArc = arcs[nodePorts[setup.mixer][mixerForward0]].reverse;
  const inputArcs = [...setup.sourceArcs, carrierArc];
  const launches = inputArcs.map((arc, i) => START - arcs[arc].delay + (i === 2 ? emitC : 0));
  // The sender supplies each image once and supplies a fixed, data-independent
  // carrier window. Neither the ports nor the schedule depend on the winner.
  const outgoing = [encode(patternA), encode(patternB), new Float64Array(SPAN).fill(AMPLITUDE)];
  const receivers = [...setup.station.probes.map(probe => probe.receivingPort), nodePorts[setup.station.node][stationForward1]];
  const cDelay = setup.loopDelays[1] - setup.station.probes[1].delay;
  const receiveAt = [...setup.station.probes.map(probe => START + probe.delay), START + emitC + cDelay];
  const stop = Math.max(...receiveAt) + SPAN - 1;
  assert.ok(launches.every(tick => tick >= 1));
  assert.ok(stop < START + Math.min(...setup.loopDelays));

  const measured = receivers.map(() => new Float64Array(SPAN));
  for (let tick = 1; tick <= stop; tick++) {
    sim.step();
    for (const [i, arc] of inputArcs.entries()) {
      const sample = outgoing[i][tick - launches[i]];
      if (sample) sim.transmit(arc, sample);
    }
    for (const [i, port] of receivers.entries()) {
      const offset = tick - receiveAt[i];
      if (offset >= 0 && offset < SPAN) measured[i][offset] = sim.lastIncoming[port];
    }
  }

  const [sum, difference, cRaw] = measured;
  const a = decode(sum.map((value, i) => (value + difference[i]) * Math.SQRT1_2), AMPLITUDE);
  const b = decode(sum.map((value, i) => (value - difference[i]) * Math.SQRT1_2), AMPLITUDE);
  assert.deepEqual(a, patternA);
  assert.deepEqual(b, patternB);
  const c = decode(cRaw, C_LEVEL);
  const energyError = sim.getStats().relativeError;
  assert.ok(energyError < 1e-12);
  const scores = Array.from(sim.alignment.accumulated);
  const { selectedTap, selectedQuarterTurns, selectionTick } = sim.alignment;
  return { patterns: { a, b, c }, selectedTap, selectionTick, receiveAt, scores,
    ...(allowRotation ? { selectedQuarterTurns } : {}),
    recoveredOriginals: true, energyError, readings: { sum: Array.from(sum), difference: Array.from(difference), c: Array.from(cRaw) } };
}

// The 2D name of the winning temporal delay is assigned by the observer only.
function labelTap(tap) {
  if (tap === null) return null;
  const delayTicks = tap - A_DELAY;
  const dy = Math.floor((delayTicks + SIZE - 1) / STRIDE);
  return { dx: delayTicks - dy * STRIDE, dy, delayTicks };
}

function runTranslationDemo() {
  const outputs = path.resolve(__dirname, '../outputs');
  const previous = JSON.parse(fs.readFileSync(path.join(outputs, 'sphere-shifted-patterns-measurements.json')));
  assert.equal(previous.size, SIZE);
  const { a, b, cAligned } = previous.patterns;
  const main = run(a, b);
  const swapped = run(b, a);
  const single = (xs) => Array.from({ length: SIZE ** 2 }, (_, i) => Number(xs.includes(i)));
  const tie = run(single([24]), single([22, 26]));
  const selection = { ...labelTap(main.selectedTap), tap: main.selectedTap,
    score: main.patterns.c.reduce((total, bit) => total + bit, 0) };
  assert.deepEqual(labelTap(main.selectedTap), { dx: -4, dy: -4, delayTicks: -56 });
  assert.deepEqual(main.patterns.c, cAligned);
  assert.deepEqual(labelTap(swapped.selectedTap), { dx: 4, dy: 4, delayTicks: 56 });
  assert.equal(tie.selectedTap, null);
  assert.ok(tie.patterns.c.every(bit => bit === 0));
  const output = { size: SIZE, patterns: main.patterns, selection, main,
    controls: { swapped: { ...swapped, selection: labelTap(swapped.selectedTap) }, tie },
    experiment: { seed: setup.seed, mixer: setup.mixer, station: setup.station.node, markerTick: START,
      tapCount: TAP_COUNT, aDelay: A_DELAY, span: SPAN, stride: STRIDE, cEmittedAt: START + EMIT_C,
      cTravelLinks: setup.nodes - setup.station.probes[1].hops, cActiveLevel: C_LEVEL,
      singleEmissionPerMotifAndRun: true, delaySearchInsideJunction: true } };
  fs.writeFileSync(path.join(outputs, 'sphere-internal-shift-measurements.json'), JSON.stringify(output, null, 2) + '\n');
  console.log(JSON.stringify({ selection, reversed: labelTap(swapped.selectedTap), tie: tie.selectedTap, energyError: main.energyError }));
}

if (require.main === module) runTranslationDemo();

module.exports = { AlignmentCircuit, encode, decode, run, labelTap, SIZE, STRIDE, SPAN, TAP_COUNT, START, C_LEVEL, setup };
