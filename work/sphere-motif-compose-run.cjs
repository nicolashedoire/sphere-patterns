'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation, createSetup } = require('./sphere-roundtrip-run.cjs');

const SIZE = 7;
const SYMBOLS = SIZE ** 2;
const AMPLITUDE = 0.5;
const C_LEVEL = AMPLITUDE * Math.sin(Math.PI * AMPLITUDE ** 2);
const PRECISION = 1e-13;
const setup = createSetup();
const mixer = setup.mixer;
const relay = setup.station.node;

function rotatePair(values, first, second, angle) {
  const a = values[first], b = values[second];
  const cosine = Math.cos(angle), sine = Math.sin(angle);
  values[first] = a * cosine - b * sine;
  values[second] = a * sine + b * cosine;
}

class MotifSimulation extends FileSimulation {
  constructor(steering) {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    this.steering = steering;
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    const node = this.nodeIndex;
    const [forward0, backward0, forward1, backward1] = this.lanePorts[node];
    if (node === mixer) {
      // Inputs use the reverse outgoing-port index in the original kernel.
      const a = values[backward0], b = values[backward1];
      const carrier = values[forward0], side = values[forward1];
      values[forward0] = (a + b) * Math.SQRT1_2;
      values[forward1] = (a - b) * Math.SQRT1_2;
      values[backward0] = carrier;
      values[backward1] = side;
      rotatePair(values, backward0, backward1, Math.PI * a * b);
      this.nodeIndex++;
      return;
    }

    const c = node === relay ? values[forward1] : 0;
    super.rotate(values, first, second);
    if (node === relay && this.steering) rotatePair(values, forward0, forward1, Math.PI * c);
  }
}

function runDemo() {
  const a = Array.from({ length: SYMBOLS }, (_, i) => Number(Math.abs(i % SIZE - 3) <= 1));
  const b = Array.from({ length: SYMBOLS }, (_, i) => Number(Math.abs(Math.floor(i / SIZE) - 3) <= 1));
  const empty = Array(SYMBOLS).fill(0);
  const carrier = Array(SYMBOLS).fill(AMPLITUDE);
  const start = setup.firstMixerTick;
  const signalFirstArrival = setup.station.probes.map(({ delay }) => start + delay);
  const cDelay = setup.loopDelays[1] - setup.station.probes[1].delay;
  const cArrival = start + cDelay;
  const relayArcPositions = new MotifSimulation(true);
  const { arcs, nodePorts } = relayArcPositions.network;
  const [mixerForward0] = relayArcPositions.lanePorts[mixer];
  const [relayForward0, , relayForward1] = relayArcPositions.lanePorts[relay];
  const carrierArc = arcs[nodePorts[mixer][mixerForward0]].reverse;
  const dSourceArc = setup.station.probes[0].arc;
  const dArcs = [nodePorts[relay][relayForward0], nodePorts[relay][relayForward1]];
  const probeFirstArrivals = dArcs.map(arc => cArrival + arcs[arc].delay);
  const lastTick = Math.max(...signalFirstArrival, ...probeFirstArrivals, cArrival) + SYMBOLS - 1;
  const cPort = nodePorts[relay][relayForward1];
  assert.ok(signalFirstArrival[0] + SYMBOLS <= cArrival && cArrival + SYMBOLS <= signalFirstArrival[1]);
  assert.ok(lastTick < start + Math.min(...setup.loopDelays));
  const relayToMixer = setup.loopDelays.map((lap, lane) => lap - setup.station.probes[lane].delay);
  assert.ok(lastTick < cArrival + Math.min(...relayToMixer));

  function run({ patternA, patternB, steering }) {
    const sim = new MotifSimulation(steering);
    const inputs = [
      { arc: setup.sourceArcs[0], arrival: start, samples: patternA.map(bit => bit * AMPLITUDE) },
      { arc: setup.sourceArcs[1], arrival: start, samples: patternB.map(bit => bit * AMPLITUDE) },
      { arc: carrierArc, arrival: start, samples: carrier },
      { arc: dSourceArc, arrival: cArrival, samples: carrier }
    ].map(input => ({ ...input, launch: input.arrival - arcs[input.arc].delay }));
    assert.ok(inputs.every(input => input.launch >= 1));
    const measured = { mixed: [[], []], c: [], original: [], diverted: [] };
    function sample(target, firstTick, receivingPort, tick) {
      if (tick >= firstTick && tick < firstTick + SYMBOLS) target.push(sim.lastIncoming[receivingPort]);
    }
    for (let tick = 1; tick <= lastTick; tick++) {
      sim.step();
      for (const { arc, launch, samples } of inputs) {
        const value = samples[tick - launch];
        if (value) sim.transmit(arc, value);
      }
      for (const [i, probe] of setup.station.probes.entries()) {
        sample(measured.mixed[i], signalFirstArrival[i], probe.receivingPort, tick);
      }
      sample(measured.c, cArrival, cPort, tick);
      sample(measured.original, probeFirstArrivals[0], arcs[dArcs[0]].reverse, tick);
      sample(measured.diverted, probeFirstArrivals[1], arcs[dArcs[1]].reverse, tick);
    }
    const stats = sim.getStats();
    assert.ok(stats.relativeError < 1e-12);
    assert.ok([measured.c, measured.original, measured.diverted, ...measured.mixed].every(trace => trace.length === SYMBOLS));
    return { ...measured, energyError: stats.relativeError };
  }

  function decode(trace, level) {
    return trace.map(value => {
      const bit = Number(value > level / 2);
      assert.ok(Math.abs(value - bit * level) < PRECISION);
      return bit;
    });
  }

  const withC = run({ patternA: a, patternB: b, steering: true });
  const withoutC = run({ patternA: a, patternB: b, steering: false });
  const aOnly = run({ patternA: a, patternB: empty, steering: true });
  const bOnly = run({ patternA: empty, patternB: b, steering: true });

  function readOriginals(result) {
    const [sum, difference] = result.mixed;
    return [decode(sum.map((value, i) => (value + difference[i]) * Math.SQRT1_2), AMPLITUDE),
      decode(sum.map((value, i) => (value - difference[i]) * Math.SQRT1_2), AMPLITUDE)];
  }

  const [recoveredA, recoveredB] = readOriginals(withC);
  assert.deepEqual([recoveredA, recoveredB], [a, b]);
  assert.deepEqual(readOriginals(withoutC), [a, b]);
  assert.deepEqual(readOriginals(aOnly), [a, empty]);
  assert.deepEqual(readOriginals(bOnly), [empty, b]);

  const c = decode(withC.c, C_LEVEL);
  assert.deepEqual(withC.c, withoutC.c);
  assert.deepEqual(decode(aOnly.c, C_LEVEL), empty);
  assert.deepEqual(decode(bOnly.c, C_LEVEL), empty);
  assert.deepEqual(c, a.map((bit, i) => bit * b[i]));

  const divertedTicks = withC.diverted.map(value => Number(Math.abs(value) > PRECISION));
  assert.deepEqual(divertedTicks, c);
  for (const control of [withoutC, aOnly, bOnly]) {
    assert.ok(control.original.every(value => value === AMPLITUDE));
    assert.ok(control.diverted.every(value => value === 0));
  }
  for (let i = 0; i < SYMBOLS; i++) {
    assert.ok(Math.abs(withC.original[i] ** 2 + withC.diverted[i] ** 2 - AMPLITUDE ** 2) < PRECISION);
    assert.ok(Math.abs(withC.original[i] - AMPLITUDE * Math.cos(Math.PI * withC.c[i])) < PRECISION);
    assert.ok(Math.abs(withC.diverted[i] - AMPLITUDE * Math.sin(Math.PI * withC.c[i])) < PRECISION);
  }

  const lit = c.flatMap((bit, i) => bit ? [[i % SIZE, Math.floor(i / SIZE)]] : []);
  const cols = lit.map(([col]) => col), rows = lit.map(([, row]) => row);
  const width = Math.max(...cols) - Math.min(...cols) + 1;
  const height = Math.max(...rows) - Math.min(...rows) + 1;
  const shape = { pixels: lit.length, width, height, filled: lit.length === width * height,
    centered: Math.min(...cols) + Math.max(...cols) === SIZE - 1 && Math.min(...rows) + Math.max(...rows) === SIZE - 1 };
  assert.deepEqual(shape, { pixels: 9, width: 3, height: 3, filled: true, centered: true });
  const divertedFraction = withC.diverted.reduce((total, sample) => total + sample * sample, 0) / (shape.pixels * AMPLITUDE ** 2);
  const byRoute = ({ original, diverted }) => ({ original, diverted });
  const summary = { size: SIZE, patterns: { a, b, c }, originalsRecovered: { a: recoveredA, b: recoveredB, exact: true },
    shape, divertedTicks: divertedTicks.reduce((sum, bit) => sum + bit, 0), divertedFraction,
    probe: { withC: byRoute(withC), withoutC: byRoute(withoutC), aOnly: byRoute(aOnly), bOnly: byRoute(bOnly) },
    logic: { aAndBAtMixingNode: 'rotation of independent carrier with angle π × a × b',
      actionAtRelay: 'rotation of two forward routes with angle π × c', polarity: '0 and +0.5' },
    initialWaveEnergy: 0,
    locations: { seed: setup.seed, nodes: setup.nodes, mixer, relay, start, signalFirstArrival, cArrival, cDelay,
      cHops: setup.nodes - setup.station.probes[1].hops, cPort, dInput: dSourceArc, dOutputs: dArcs,
      dDestinations: dArcs.map(arc => arcs[arc].to), probeFirstArrivals },
    measurements: { withC, withoutC, aOnly, bOnly } };
  const output = path.resolve(__dirname, '../outputs/sphere-croisement-motifs-mesures.json');
  fs.writeFileSync(output, JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify({ shape, divertedFraction, locations: summary.locations, energyError: withC.energyError }));
}

if (require.main === module) runDemo();

module.exports = { MotifSimulation, setup, AMPLITUDE, C_LEVEL };
