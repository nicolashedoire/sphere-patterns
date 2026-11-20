'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { run, labelTap, SIZE, STRIDE, SPAN, TAP_COUNT, START, C_LEVEL, setup } =
  require('./sphere-pattern-internal-delay-run.cjs');

const outputs = path.resolve(__dirname, '../outputs');
const reference = JSON.parse(fs.readFileSync(path.join(outputs, 'sphere-shifted-patterns-measurements.json')));
assert.equal(reference.size, SIZE);
const a = reference.patterns.a;

// The transmitter builds the test image. The junction receives only its
// pulses. The L has been turned clockwise and placed at column 3, row 3.
const b = Array(SIZE ** 2).fill(0);
for (const [x, y] of [[5, 3], [4, 3], [3, 3], [3, 4], [3, 5], [6, 1]]) b[y * SIZE + x] = 1;

function summarize(result) {
  if (result.selectedTap === null) return null;
  return { rotationDegrees: result.selectedQuarterTurns * 90, ...labelTap(result.selectedTap),
    tap: result.selectedTap, score: result.patterns.c.reduce((total, pixel) => total + pixel, 0) };
}

const main = run(a, b, true);
const reversed = run(b, a, true);
const translationOnly = run(a, b);
const selection = summarize(main);
const reversedSelection = summarize(reversed);
assert.deepEqual(selection, { rotationDegrees: 270, dx: -3, dy: -1, delayTicks: -16, tap: 68, score: 5 });
assert.deepEqual(main.patterns.c, reference.patterns.cAligned);
assert.deepEqual(reversedSelection, { rotationDegrees: 90, dx: -1, dy: 3, delayTicks: 38, tap: 122, score: 5 });
const translationOnlyPeak = Math.round(Math.max(...translationOnly.scores) / C_LEVEL ** 2);
assert.ok(translationOnlyPeak < selection.score);
const ranked = [...main.scores].sort((a, b) => b - a);
const runnerUpScore = Math.round(ranked[1] / C_LEVEL ** 2);

const summary = { size: SIZE, patterns: main.patterns, selection, runnerUpScore, main,
  controls: { reversed: { ...reversed, selection: reversedSelection }, translationOnly: { ...translationOnly, peakScore: translationOnlyPeak } },
  experiment: { seed: setup.seed, mixer: setup.mixer, station: setup.station.node, inputMarkerTick: START,
    rotationDegreesTested: [0, 90, 180, 270], angleConvention: 'clockwise around grid center, then translation',
    tapsPerRotation: TAP_COUNT, candidates: main.scores.length, stride: STRIDE, span: SPAN, cActiveLevel: C_LEVEL,
    cEmittedAt: main.selectionTick + 1, cTravelLinks: setup.nodes - setup.station.probes[1].hops,
    singleEmissionPerMotifAndRun: true, searchInsideJunction: true } };
fs.writeFileSync(path.join(outputs, 'sphere-internal-rotation-measurements.json'), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ selection, reversedSelection, translationOnlyPeak, runnerUpScore, energyError: main.energyError }));
