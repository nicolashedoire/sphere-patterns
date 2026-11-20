'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation } = require('./sphere-roundtrip-run.cjs');
const { AlignmentCircuit, START, C_LEVEL, setup } = require('./sphere-pattern-internal-delay-run.cjs');

const SIZE = 25;
const STRIDE = 2 * SIZE - 1;
const CENTER = (SIZE - 1) / 2;
const PIXEL_TICKS = Array.from({ length: SIZE ** 2 }, (_, index) => Math.floor(index / SIZE) * STRIDE + index % SIZE);
const SPAN = PIXEL_TICKS.at(-1) + 1;
const AMPLITUDE = 0.5;
const RESULT_LEVEL = AMPLITUDE * Math.sin(Math.PI * AMPLITUDE * C_LEVEL);
const ANGLES = Array.from({ length: 24 }, (_, index) => index * 15);
const MATCH_THRESHOLD = 0.86;
const MIN_VISIBLE = 0.60;
const MAX_POSITION_ERROR = 2;
const MAX_ANGLE_ERROR = 15;

// Each mask rectangle uses inclusive grid-cell bounds.
const SCENES = [
  { id: 'intact', pose: [8, 9, 30], cover: null, expected: 'target' },
  { id: 'partial-30', pose: [8, 9, 30], cover: [10, 5, 12, 11], expected: 'target' },
  { id: 'partial-58', pose: [9, 10, 58], cover: [11, 8, 14, 15], expected: 'target' },
  { id: 'partial-107', pose: [8, 10, 107], cover: [7, 4, 13, 7], expected: 'target' },
  { id: 'heavily-covered', pose: [9, 10, 58], cover: [4, 6, 14, 14], expected: 'abstain' },
  { id: 'distractor-only', pose: null, cover: [10, 5, 12, 11], expected: 'abstain' },
  { id: 'lookalike', pose: null, lookalike: [8, 9, 30], cover: [6, 10, 10, 12], expected: 'abstain' }
];

function rotate(x, y, degrees) {
  const angle = degrees * Math.PI / 180;
  const cos = Math.cos(angle), sin = Math.sin(angle);
  return [x * cos - y * sin, x * sin + y * cos];
}

function silhouette(form, centerX, centerY, degrees) {
  return PIXEL_TICKS.map((_, index) => {
    const [x, y] = rotate(index % SIZE - centerX, Math.floor(index / SIZE) - centerY, -degrees);
    if (form === 'L') {
      return Number((x > -3.5 && x < -.5 && Math.abs(y) < 4.5) ||
        (x > -3.5 && x < 4.5 && y > 1.5 && y < 4.5));
    }
    const rightEdge = form === 'short-T' ? 2.5 : 5.5;
    return Number((x > -5.5 && x < rightEdge && y > -4.5 && y < -1.5) ||
      (Math.abs(x) < 1.5 && y >= -1.5 && y < 4.5));
  });
}

function visibility(cover) {
  if (cover === null) return PIXEL_TICKS.map(() => 1);
  const [left, top, right, bottom] = cover;
  return PIXEL_TICKS.map((_, index) => {
    const x = index % SIZE, y = Math.floor(index / SIZE);
    return Number(x < left || x > right || y < top || y > bottom);
  });
}

function encode(pixels) {
  const wave = new Float64Array(SPAN);
  PIXEL_TICKS.forEach((tick, index) => { wave[tick] = AMPLITUDE * pixels[index]; });
  return wave;
}

function decode(wave, level) {
  return PIXEL_TICKS.map(tick => {
    const bit = Number(wave[tick] > level / 2);
    assert.ok(Math.abs(wave[tick] - bit * level) < 1e-13);
    return bit;
  });
}

function addressBank(inverseDegrees) {
  return Int16Array.from(PIXEL_TICKS, (_, index) => {
    const [x, y] = rotate(index % SIZE - CENTER, Math.floor(index / SIZE) - CENTER, -inverseDegrees);
    const col = Math.round(CENTER + x), row = Math.round(CENTER + y);
    return col < 0 || row < 0 || col >= SIZE || row >= SIZE ? -1 : row * SIZE + col;
  });
}

function sortedBest(best, candidate, compare) {
  if (best === null) return { candidate, tied: false };
  const difference = compare(candidate, best.candidate);
  if (difference > 0) return { candidate, tied: false };
  if (difference === 0) return { ...best, tied: true };
  return best;
}

const compareOrdinary = (a, b) => a.matches - b.matches;
const compareVisible = (a, b) => a.matches * b.visible - b.matches * a.visible || a.visible - b.visible;

class VisibilityComparator {
  constructor(start, visibleCells, timing) {
    this.start = start;
    this.emitAt = timing;
    this.visibleCells = visibleCells;
    this.reference = new Float64Array(SPAN);
    this.scene = new Float64Array(SPAN);
    this.ordinary = null;
    this.aware = null;
    this.withoutVisibilityFloor = null;
    this.matchAtTick = new Float64Array(SPAN);
    this.selectedCellAtTick = new Int16Array(SPAN).fill(-1);
  }

  latch() {
    const referenceCells = PIXEL_TICKS.keys().filter(index => this.reference[PIXEL_TICKS[index]] > 0).toArray();
    if (!referenceCells.length) return;
    const columns = referenceCells.map(index => index % SIZE);
    const rows = referenceCells.map(index => Math.floor(index / SIZE));
    const minCol = Math.min(...columns), maxCol = Math.max(...columns);
    const minRow = Math.min(...rows), maxRow = Math.max(...rows);
    const minVisible = Math.ceil(referenceCells.length * MIN_VISIBLE);
    let baseline = null, adjusted = null, unguarded = null;

    for (const angle of ANGLES) {
      const addresses = addressBank(angle);
      this.searchOrientation(addresses, minCol, maxCol, minRow, maxRow, referenceCells, (dx, dy, matches, visible) => {
        if (matches === 0) return;
        const candidate = { angle, dx, dy, matches, visible };
        baseline = sortedBest(baseline, candidate, compareOrdinary);
        unguarded = sortedBest(unguarded, candidate, compareVisible);
        if (visible >= minVisible) adjusted = sortedBest(adjusted, candidate, compareVisible);
      });
    }

    this.ordinary = this.summarize(baseline, referenceCells.length, false);
    this.aware = this.summarize(adjusted, referenceCells.length, true);
    this.withoutVisibilityFloor = this.summarize(unguarded, referenceCells.length, true);
    if (!this.aware.accepted) return;

    const { angle, dx, dy } = this.aware.best;
    const addresses = addressBank(angle);
    for (const index of referenceCells) {
      const cell = addresses[(Math.floor(index / SIZE) - dy) * SIZE + index % SIZE - dx];
      if (!this.visibleCells[cell]) continue;
      this.matchAtTick[PIXEL_TICKS[index]] = this.scene[PIXEL_TICKS[cell]];
      this.selectedCellAtTick[PIXEL_TICKS[index]] = cell;
    }
  }

  searchOrientation(addresses, minCol, maxCol, minRow, maxRow, referenceCells, evaluate) {
    for (let dy = maxRow - SIZE + 1; dy <= minRow; dy++) {
      for (let dx = maxCol - SIZE + 1; dx <= minCol; dx++) {
        let matches = 0, visible = 0, outside = false;
        for (const index of referenceCells) {
          const cell = addresses[(Math.floor(index / SIZE) - dy) * SIZE + index % SIZE - dx];
          if (cell < 0) {
            outside = true;
            break;
          }
          if (!this.visibleCells[cell]) continue;
          visible++;
          if (this.scene[PIXEL_TICKS[cell]] > 0) matches++;
        }
        if (!outside) evaluate(dx, dy, matches, visible);
      }
    }
  }

  summarize(result, total, aware) {
    if (result === null) return { accepted: false, best: null, tied: false, score: 0 };
    const { candidate, tied } = result;
    const { angle, dx, dy, matches, visible } = candidate;
    const [x, y] = rotate(-dx, -dy, -angle);
    const score = matches / (aware ? visible : total);
    return {
      accepted: !tied && score >= MATCH_THRESHOLD,
      best: { ...candidate, center: [CENTER + x, CENTER + y], angleInScene: (360 - angle) % 360 },
      tied,
      score
    };
  }

  receive(tick, referenceSample, sceneSample) {
    const elapsed = tick - this.start;
    if (elapsed >= 0 && elapsed < SPAN) {
      this.reference[elapsed] = referenceSample;
      this.scene[elapsed] = sceneSample;
    }
    if (elapsed === this.emitAt - 1) this.latch();

    const readout = elapsed - this.emitAt;
    if (readout < 0 || readout >= SPAN) return 0;
    return Math.PI * this.reference[readout] * this.matchAtTick[readout];
  }
}

class OcclusionSimulation extends FileSimulation {
  constructor(visibleCells, disconnected) {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    this.learning = new AlignmentCircuit(true, START, { size: SIZE });
    this.referenceEmittedAt = START + this.learning.emitAt;
    this.referenceReturnedAt = this.referenceEmittedAt + setup.loopDelays[1];
    this.comparison = new VisibilityComparator(this.referenceReturnedAt, visibleCells, this.learning.emitAt);
    this.disconnected = disconnected;
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);

    const [forward0, backward0, forward1, backward1] = this.lanePorts[this.nodeIndex++];
    const a = values[backward0], b = values[backward1];
    const carrier = values[forward0], returned = values[forward1];
    const tick = this.tick + 1;
    // Open the reference input at the comparator without removing wave energy.
    const angle = tick < this.referenceReturnedAt ? this.learning.receive(tick, a, b) :
      this.comparison.receive(tick, this.disconnected ? 0 : returned, b);
    values[forward0] = (a + b) * Math.SQRT1_2;
    values[forward1] = (a - b) * Math.SQRT1_2;
    values[backward0] = carrier * Math.cos(angle) - returned * Math.sin(angle);
    values[backward1] = carrier * Math.sin(angle) + returned * Math.cos(angle);
  }
}

function run(scene, visibleCells, disconnected = false) {
  const sim = new OcclusionSimulation(visibleCells, disconnected);
  const { arcs, nodePorts } = sim.network;
  const carrier = arcs[nodePorts[setup.mixer][sim.lanePorts[setup.mixer][0]]].reverse;
  const probe = nodePorts[setup.station.node][sim.lanePorts[setup.station.node][2]];
  const loopTicks = setup.loopDelays[1];
  const probeDelay = loopTicks - setup.station.probes[1].delay;
  const resultEmittedAt = sim.referenceReturnedAt + sim.comparison.emitAt;
  const inputs = [
    [setup.sourceArcs[0], START, encode(silhouette('T', CENTER, CENTER, 0))],
    [setup.sourceArcs[1], START, encode(silhouette('T', CENTER, CENTER, 90))],
    [carrier, sim.referenceEmittedAt, new Float64Array(SPAN).fill(AMPLITUDE)],
    [setup.sourceArcs[1], sim.referenceReturnedAt, encode(scene)],
    [carrier, resultEmittedAt, new Float64Array(SPAN).fill(AMPLITUDE)]
  ].map(([arc, tick, samples]) => ({ arc, start: tick - arcs[arc].delay, samples }));

  const captures = [sim.referenceEmittedAt + probeDelay, resultEmittedAt + probeDelay]
    .map(start => ({ start, wave: new Float64Array(SPAN) }));
  const end = captures.at(-1).start + SPAN - 1;
  for (let tick = 1; tick <= end; tick++) {
    sim.step();
    for (const { arc, start, samples } of inputs) {
      const sample = samples[tick - start];
      if (sample) sim.transmit(arc, sample);
    }
    for (const { start, wave } of captures) {
      if (tick >= start && tick < start + SPAN) wave[tick - start] = sim.lastIncoming[probe];
    }
  }

  const comparator = sim.comparison;
  const [{ wave: referenceWave }, { wave: outputWave }] = captures;
  assert.deepEqual(comparator.reference, disconnected ? new Float64Array(SPAN) : referenceWave);
  assert.deepEqual(comparator.scene, encode(scene));
  const reference = decode(referenceWave, C_LEVEL);
  const matchedReference = decode(outputWave, RESULT_LEVEL);
  const sensedCount = matchedReference.filter(Boolean).length;
  assert.equal(sensedCount, comparator.aware?.accepted ? comparator.aware.best.matches : 0);
  const matchedScene = [...new Set(PIXEL_TICKS.map((tick, i) => matchedReference[i] ? comparator.selectedCellAtTick[tick] : -1))]
    .filter(index => index >= 0).sort((a, b) => a - b);
  const energyError = sim.getStats().relativeError;
  assert.ok(energyError < 1e-12);
  return {
    ordinary: comparator.ordinary,
    aware: comparator.aware,
    withoutVisibilityFloor: comparator.withoutVisibilityFloor,
    reference,
    matchedReference,
    matchedScene,
    probe: {
      referenceEmittedAt: sim.referenceEmittedAt,
      referenceObservedAt: captures[0].start,
      referenceReturnedAt: sim.referenceReturnedAt,
      resultObservedAt: captures[1].start,
      loopTicks,
      energyError
    }
  };
}

function judge(assessment, fixture) {
  if (fixture.expected === 'abstain') return !assessment.accepted;
  if (!assessment.accepted) return false;
  const [trueX, trueY, trueAngle] = fixture.pose;
  const { center, angleInScene } = assessment.best;
  const angleError = Math.abs((angleInScene - trueAngle + 540) % 360 - 180);
  return Math.hypot(center[0] - trueX, center[1] - trueY) <= MAX_POSITION_ERROR && angleError <= MAX_ANGLE_ERROR;
}

const distractor = silhouette('L', 19, 17, 37);
const results = SCENES.map(fixture => {
  const visible = visibility(fixture.cover);
  const target = fixture.pose ? silhouette('T', ...fixture.pose) : [];
  const lookalike = fixture.lookalike ? silhouette('short-T', ...fixture.lookalike) : [];
  const image = visible.map((value, index) => value * (target[index] | lookalike[index] | distractor[index]));
  const simulation = run(image, visible);
  const correct = {
    ordinary: judge(simulation.ordinary, fixture),
    aware: judge(simulation.aware, fixture)
  };
  return {
    fixture,
    visibleTargetPixels: target.filter((value, index) => value && visible[index]).length,
    targetPixels: target.filter(Boolean).length,
    image,
    visible,
    correct,
    ...simulation
  };
});
// Use a successful partial scene so the disconnected control can remove a signal.
const cutFixture = results.find(result => result.fixture.id.startsWith('partial-') && result.correct.aware);
assert.ok(cutFixture);
const cut = run(cutFixture.image, cutFixture.visible, true);
assert.equal(cut.ordinary, null);
assert.equal(cut.aware, null);
assert.ok(cut.matchedReference.every(bit => bit === 0));
assert.deepEqual(cut.reference, cutFixture.reference);

const report = {
  protocol: {
    size: SIZE,
    trainingAngles: [0, 90],
    searchAngles: ANGLES,
    overlapThreshold: MATCH_THRESHOLD,
    minVisibleFraction: MIN_VISIBLE,
    positionToleranceCells: MAX_POSITION_ERROR,
    angleToleranceDegrees: MAX_ANGLE_ERROR,
    visibilityInput: 'The junction receives the exact cells hidden by a synthetic rectangle. The mask is supplied locally and is not inferred or transmitted as a wave.',
    scoring: 'Ordinary overlap divides matched cells by all reference cells. Visibility-aware overlap divides by visible reference cells and requires at least 60% to remain visible. Equal winning poses cause abstention.',
    coverageControl: 'The same junction also scores visible overlap without the minimum-visibility gate as a read-only diagnostic.',
    physicalControl: 'Both scores use the same returning waveform C. Only the visibility-aware result is emitted on the carrier. The disconnected reference still circulates but is gated at the comparator input.',
    scope: 'Synthetic fixed-scale, top-down silhouettes only. The experiment does not test unknown occluders or photographic segmentation.'
  },
  results,
  disconnected: { scene: cutFixture.fixture.id, ...cut }
};
const output = process.argv[2] ?? path.join(__dirname, '../outputs/sphere-occlusion-measurements.json');
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
for (const result of results) {
  const { fixture, correct, ordinary, aware, visibleTargetPixels, targetPixels } = result;
  const mask = targetPixels ? `${visibleTargetPixels}/${targetPixels} visible` : 'no target';
  console.log(`${fixture.id}: ${mask}; ordinary ${ordinary.score.toFixed(3)} (${correct.ordinary}); visibility-aware ${aware.score.toFixed(3)} (${correct.aware})`);
}
console.log('Disconnected reference: no matching wave at the probe.');
