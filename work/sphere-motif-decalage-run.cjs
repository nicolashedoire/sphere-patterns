'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { MotifSimulation, setup, AMPLITUDE, C_LEVEL } = require('./sphere-motif-compose-run.cjs');

const size = 7;
const radius = size - 1;
// Horizontal differences, even after a trial shift, are at most 12. This
// stride makes a false coincidence across two different rows impossible.
const stride = 2 * size - 1;
const pixelTimes = Array.from({ length: size ** 2 }, (_, i) => Math.floor(i / size) * stride + i % size);
const span = pixelTimes.at(-1) + 1;
const maxShift = radius * (stride + 1);
const slotWidth = span + 2 * maxShift;
const trialsPerBatch = Math.floor((Math.min(...setup.loopDelays) - 1) / slotWidth);
const firstTick = setup.firstMixerTick;
assert.ok(trialsPerBatch >= 1);

function raster(points) {
  const pixels = Array(size ** 2).fill(0);
  for (const [x, y] of points) pixels[y * size + x] = 1;
  return pixels;
}

const a = raster([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [3, 0]]);
const b = raster([[4, 4], [4, 5], [4, 6], [5, 6], [6, 6], [6, 3]]);
const shifts = [];
for (let dy = -radius; dy <= radius; dy++) {
  for (let dx = -radius; dx <= radius; dx++) shifts.push({ dx, dy, delayTicks: dy * stride + dx });
}
assert.equal(new Set(shifts.map(shift => shift.delayTicks)).size, shifts.length);

function decodeBit(value, level) {
  const bit = Number(value > level / 2);
  assert.ok(Math.abs(value - bit * level) < 1e-13);
  return bit;
}

function runBatch(batch) {
  const sim = new MotifSimulation(false);
  const { arcs, nodePorts } = sim.network;
  const mixer = setup.mixer, station = setup.station.node;
  const [mixerForward0] = sim.lanePorts[mixer];
  const [, , stationForward1] = sim.lanePorts[station];
  const carrierArc = arcs[nodePorts[mixer][mixerForward0]].reverse;
  const sourceArcs = [...setup.sourceArcs, carrierArc];
  const receiverPorts = [...setup.station.probes.map(probe => probe.receivingPort), nodePorts[station][stationForward1]];
  const propagation = [...setup.station.probes.map(probe => probe.delay), setup.loopDelays[1] - setup.station.probes[1].delay];
  const length = batch.length * slotWidth;
  const outgoing = sourceArcs.map(() => new Float64Array(length));

  for (const [trial, shift] of batch.entries()) {
    const startA = trial * slotWidth + maxShift;
    const startB = startA + shift.delayTicks;
    for (const [pixel, offset] of pixelTimes.entries()) {
      outgoing[0][startA + offset] = AMPLITUDE * a[pixel];
      outgoing[1][startB + offset] = AMPLITUDE * b[pixel];
      outgoing[2][startA + offset] = AMPLITUDE;
    }
  }

  const measured = receiverPorts.map(() => new Float64Array(length));
  const endTick = firstTick + length - 1 + Math.max(...propagation);
  for (let tick = 1; tick <= endTick; tick++) {
    sim.step();
    for (const [stream, arc] of sourceArcs.entries()) {
      const sample = outgoing[stream][tick + arcs[arc].delay - firstTick];
      if (sample) sim.transmit(arc, sample);
    }
    for (const [channel, port] of receiverPorts.entries()) {
      const index = tick - propagation[channel] - firstTick;
      if (index >= 0 && index < length) measured[channel][index] = sim.lastIncoming[port];
    }
  }
  const energyError = sim.getStats().relativeError;
  assert.ok(energyError < 1e-12);

  const [sum, difference, signalC] = measured;
  const results = batch.map((shift, trial) => {
    const startA = trial * slotWidth + maxShift;
    const startB = startA + shift.delayTicks;
    const recoveredA = pixelTimes.map(offset => {
      const t = startA + offset;
      return decodeBit((sum[t] + difference[t]) * Math.SQRT1_2, AMPLITUDE);
    });
    const recoveredB = pixelTimes.map(offset => {
      const t = startB + offset;
      return decodeBit((sum[t] - difference[t]) * Math.SQRT1_2, AMPLITUDE);
    });
    assert.deepEqual(recoveredA, a);
    assert.deepEqual(recoveredB, b);
    const rawC = pixelTimes.map(offset => signalC[startA + offset]);
    const decoded = rawC.map(sample => decodeBit(sample, C_LEVEL));
    return { ...shift, score: decoded.reduce((total, bit) => total + bit, 0), rawC, decoded };
  });
  return { results, energyError };
}

const results = [];
let maxEnergyError = 0;
for (let start = 0; start < shifts.length; start += trialsPerBatch) {
  const batch = runBatch(shifts.slice(start, start + trialsPerBatch));
  results.push(...batch.results);
  maxEnergyError = Math.max(maxEnergyError, batch.energyError);
}
const direct = results.find(({ dx, dy }) => dx === 0 && dy === 0);
assert.ok(direct);
const ranked = [...results].sort((a, b) => b.score - a.score);
const [best, alternate] = ranked;
const tiedAtBest = ranked.filter(trial => trial.score === best.score).length;
const bestShift = { dx: best.dx, dy: best.dy, delayTicks: best.delayTicks, score: best.score,
  runnerUpScore: alternate.score, tiedAtBest };
const summary = { size, patterns: { a, b, cDirect: direct.decoded, cAligned: best.decoded }, bestShift,
  directScore: direct.score, testedShifts: results.length, recoveredOriginalsEveryTrial: true, maxEnergyError,
  experiment: { seed: setup.seed, mixer: setup.mixer, station: setup.station.node, stride, slotWidth, trialsPerBatch,
    carrierLevel: AMPLITUDE, cActiveLevel: C_LEVEL, cTravelLinks: setup.nodes - setup.station.probes[1].hops },
  trials: results };
const output = path.resolve(__dirname, '../outputs/sphere-motifs-decales-mesures.json');
fs.writeFileSync(output, JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ bestShift, directScore: direct.score, testedShifts: results.length, maxEnergyError }));
