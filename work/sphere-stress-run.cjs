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
const SCALES = [0.75, 1, 1.25];
const THRESHOLD = 0.86;
const MIN_VISIBLE = 0.60;
const TOLERANCES = { centerCells: 2, angleDegrees: 15, scale: 0.13 };

const L = ['L', 19, 17, 37, 0.85];
const CLUTTER = [L, ['ring', 20, 5, 0, 0.75], ['stripe', 5, 18, 67, 1], ['dots', 17, 10, 0, 1]];
const SCALED_TARGET = ['T', 8, 9, 30, 1.25];
const SCENES = [
  { id: 'clean-target', stage: 'isolated', target: ['T', 8, 9, 30, 1], distractors: [] },
  { id: 'clean-negative', stage: 'isolated', target: null, distractors: [L] },
  { id: 'small-target', stage: 'scale', target: ['T', 9, 10, 58, 0.75], distractors: [] },
  { id: 'large-target', stage: 'scale', target: SCALED_TARGET, distractors: [] },
  { id: 'clutter-target', stage: 'clutter', target: SCALED_TARGET, distractors: CLUTTER },
  { id: 'intruding-stripe', stage: 'clutter', target: SCALED_TARGET,
    distractors: [...CLUTTER, ['stripe', 13, 10, 73, 0.8]] },
  { id: 'lookalike-negative', stage: 'clutter', target: null,
    distractors: [...CLUTTER, ['short-T', 8, 9, 30, 1.25]] },
  { id: 'solid-negative', stage: 'clutter', target: null,
    distractors: [L, ['solid', 9, 10, 30, 1]] },
  { id: 'covered-clutter-target', stage: 'occlusion', target: SCALED_TARGET, distractors: CLUTTER,
    cover: [10, 5, 12, 11] },
  { id: 'covered-oblique-target', stage: 'occlusion', target: ['T', 9, 10, 58, 1], distractors: [L],
    cover: [11, 8, 14, 15] },
  { id: 'heavily-covered-target', stage: 'occlusion', target: ['T', 9, 10, 58, 1], distractors: [L],
    cover: [4, 6, 14, 14] },
  { id: 'covered-negative', stage: 'occlusion', target: null,
    distractors: [...CLUTTER, ['short-T', 8, 9, 30, 1.25]], cover: [6, 10, 10, 12] }
];

function rotate(x, y, degrees) {
  const radians = degrees * Math.PI / 180;
  const cos = Math.cos(radians), sin = Math.sin(radians);
  return [x * cos - y * sin, x * sin + y * cos];
}

function silhouette([kind, centerX, centerY, angle, scale]) {
  return PIXEL_TICKS.map((_, index) => {
    const [x, y] = rotate((index % SIZE - centerX) / scale, (Math.floor(index / SIZE) - centerY) / scale, -angle);
    if (kind === 'L') {
      return Number((x > -3.5 && x < -.5 && Math.abs(y) < 4.5) ||
        (x > -3.5 && x < 4.5 && y > 1.5 && y < 4.5));
    }
    if (kind === 'ring') return Number(x * x + y * y >= 4 && x * x + y * y < 16);
    if (kind === 'stripe') return Number(Math.abs(x) < 4.5 && Math.abs(y) < 1);
    if (kind === 'dots') return Number(Math.abs(x) <= 3 && Math.abs(y) <= 2 && (x + 2 * y) % 3 === 0);
    if (kind === 'solid') return Number(Math.abs(x) < 5.5 && Math.abs(y) < 4.5);
    const rightEdge = kind === 'short-T' ? 2.5 : 5.5;
    return Number((x > -5.5 && x < rightEdge && y > -4.5 && y < -1.5) ||
      (Math.abs(x) < 1.5 && y >= -1.5 && y < 4.5));
  });
}

function render(fixture) {
  const target = fixture.target ? silhouette(fixture.target) : PIXEL_TICKS.map(() => 0);
  const clutter = fixture.distractors.map(silhouette);
  const visible = PIXEL_TICKS.map((_, index) => {
    if (!fixture.cover) return 1;
    const [left, top, right, bottom] = fixture.cover;
    const x = index % SIZE, y = Math.floor(index / SIZE);
    return Number(x < left || x > right || y < top || y > bottom);
  });
  const image = target.map((bit, index) => visible[index] && Number(bit || clutter.some(mask => mask[index])));
  return { target, visible, image };
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

function templates(reference) {
  const active = PIXEL_TICKS.flatMap((tick, index) => reference[tick] > 0 ? [index] : []);
  if (!active.length) return [];
  const reach = Math.ceil(Math.max(...active.map(index => Math.hypot(index % SIZE - CENTER, Math.floor(index / SIZE) - CENTER))) * SCALES.at(-1) + 2);

  return SCALES.flatMap(scale => ANGLES.map(angle => {
    const cells = [];
    for (let y = -reach; y <= reach; y++) {
      for (let x = -reach; x <= reach; x++) {
        const [sx, sy] = rotate(x / scale, y / scale, -angle);
        const col = Math.round(CENTER + sx), row = Math.round(CENTER + sy);
        if (col < 0 || col >= SIZE || row < 0 || row >= SIZE) continue;
        const refIndex = row * SIZE + col;
        if (reference[PIXEL_TICKS[refIndex]] > 0) cells.push({ x, y, refIndex });
      }
    }
    const xs = cells.map(cell => cell.x), ys = cells.map(cell => cell.y);
    const bounds = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    return { scale, angle, cells, bounds };
  }));
}

function foregroundSum(pixels) {
  const pitch = SIZE + 1;
  const sums = new Int16Array(pitch ** 2);
  for (let y = 1; y <= SIZE; y++) {
    let row = 0;
    for (let x = 1; x <= SIZE; x++) {
      row += pixels[(y - 1) * SIZE + x - 1];
      sums[y * pitch + x] = sums[(y - 1) * pitch + x] + row;
    }
  }
  return (left, top, right, bottom) => sums[(bottom + 1) * pitch + right + 1] -
    sums[top * pitch + right + 1] - sums[(bottom + 1) * pitch + left] + sums[top * pitch + left];
}

function choose(best, candidate) {
  if (best === null) return { candidate, tied: false };
  const order = candidate.matches * best.candidate.denominator - best.candidate.matches * candidate.denominator;
  if (order > 0) return { candidate, tied: false };
  if (order === 0) return { ...best, tied: true };
  return best;
}

function summarize(best) {
  if (best === null) return { score: 0, tied: false, accepted: false, best: null };
  const { candidate, tied } = best;
  const { template, x, y, matches, denominator } = candidate;
  const score = matches / denominator;
  return { score, tied, accepted: !tied && score >= THRESHOLD,
    best: { center: [x, y], angle: template.angle, scale: template.scale,
      matches, denominator, templateCells: template.cells.length } };
}

class StressComparator {
  constructor(start, emitAt, knownVisibility) {
    this.start = start;
    this.emitAt = emitAt;
    this.knownVisibility = knownVisibility;
    this.reference = new Float64Array(SPAN);
    this.scene = new Float64Array(SPAN);
    this.outgoing = new Float64Array(SPAN);
    this.matchedScene = [];
  }

  latch() {
    const scene = PIXEL_TICKS.map(tick => Number(this.scene[tick] > 0));
    const rectangle = foregroundSum(scene);
    const winners = { fixed: null, scaled: null, iou: null, knownMask: null };

    for (const template of templates(this.reference)) {
      const { bounds: [left, top, right, bottom], cells, scale } = template;
      const area = cells.length;
      for (let y = -top; y < SIZE - bottom; y++) {
        for (let x = -left; x < SIZE - right; x++) {
          let matches = 0, visible = 0;
          for (const cell of cells) {
            const index = (y + cell.y) * SIZE + x + cell.x;
            matches += scene[index];
            visible += this.knownVisibility[index];
          }
          if (!matches) continue;

          const candidate = { template, x, y, matches, denominator: area };
          if (scale === 1) winners.fixed = choose(winners.fixed, candidate);
          winners.scaled = choose(winners.scaled, candidate);
          const localForeground = rectangle(x + left, y + top, x + right, y + bottom);
          winners.iou = choose(winners.iou, { ...candidate, denominator: area + localForeground - matches });
          if (visible >= Math.ceil(area * MIN_VISIBLE)) {
            winners.knownMask = choose(winners.knownMask, { ...candidate, denominator: visible });
          }
        }
      }
    }

    this.assessments = Object.fromEntries(Object.entries(winners).map(([method, best]) => [method, summarize(best)]));
    if (!this.assessments.iou.accepted) return;
    const { template, x, y } = winners.iou.candidate;
    for (const cell of template.cells) {
      const index = (y + cell.y) * SIZE + x + cell.x;
      if (!scene[index]) continue;
      this.outgoing[PIXEL_TICKS[cell.refIndex]] = this.scene[PIXEL_TICKS[index]];
      this.matchedScene.push(index);
    }
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
    return Math.PI * this.reference[readout] * this.outgoing[readout];
  }
}

class StressSimulation extends FileSimulation {
  constructor(visibility, disconnected) {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    this.learning = new AlignmentCircuit(true, START, { size: SIZE });
    this.referenceEmittedAt = START + this.learning.emitAt;
    this.referenceReturnedAt = this.referenceEmittedAt + setup.loopDelays[1];
    this.comparison = new StressComparator(this.referenceReturnedAt, this.learning.emitAt, visibility);
    this.disconnected = disconnected;
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);

    const [forward0, backward0, forward1, backward1] = this.lanePorts[this.nodeIndex++];
    const a = values[backward0], b = values[backward1];
    const carrier = values[forward0], returned = values[forward1];
    const tick = this.tick + 1;
    // Disconnect only the comparator input. The pulse still contributes to physical energy.
    const angle = tick < this.referenceReturnedAt ? this.learning.receive(tick, a, b) :
      this.comparison.receive(tick, this.disconnected ? 0 : returned, b);
    values[forward0] = (a + b) * Math.SQRT1_2;
    values[forward1] = (a - b) * Math.SQRT1_2;
    values[backward0] = carrier * Math.cos(angle) - returned * Math.sin(angle);
    values[backward1] = carrier * Math.sin(angle) + returned * Math.cos(angle);
  }
}

function compareCached(referenceWave, scene, visibility) {
  const comparator = new StressComparator(0, SPAN, visibility);
  comparator.reference.set(referenceWave);
  comparator.scene.set(encode(scene));
  comparator.latch();
  return comparator;
}

function simulate(scene, visibility, disconnected = false) {
  const sim = new StressSimulation(visibility, disconnected);
  const { arcs, nodePorts } = sim.network;
  const carrier = arcs[nodePorts[setup.mixer][sim.lanePorts[setup.mixer][0]]].reverse;
  const probe = nodePorts[setup.station.node][sim.lanePorts[setup.station.node][2]];
  const loopTicks = setup.loopDelays[1];
  const probeDelay = loopTicks - setup.station.probes[1].delay;
  const resultEmittedAt = sim.referenceReturnedAt + sim.comparison.emitAt;
  const inputs = [
    [setup.sourceArcs[0], START, encode(silhouette(['T', CENTER, CENTER, 0, 1]))],
    [setup.sourceArcs[1], START, encode(silhouette(['T', CENTER, CENTER, 90, 1]))],
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

  const { reference, scene: seenScene, assessments, matchedScene, outgoing } = sim.comparison;
  const [{ wave: referenceWave }, { wave: outputWave }] = captures;
  assert.deepEqual(reference, disconnected ? new Float64Array(SPAN) : referenceWave);
  assert.deepEqual(seenScene, encode(scene));
  const sensedReference = decode(referenceWave, C_LEVEL);
  const matchAtProbe = decode(outputWave, RESULT_LEVEL);
  assert.deepEqual(matchAtProbe, PIXEL_TICKS.map(tick => Number(outgoing[tick] > 0)));
  assert.equal(matchedScene.length, assessments.iou.accepted ? assessments.iou.best.matches : 0);
  const directCacheParity = !disconnected;
  if (directCacheParity) {
    const cached = compareCached(referenceWave, scene, visibility);
    assert.deepEqual(cached.assessments, assessments);
    assert.deepEqual(cached.outgoing, outgoing);
    if (visibility.some(bit => !bit)) {
      const cacheWithoutMask = compareCached(referenceWave, scene, visibility.map(() => 1));
      for (const method of ['fixed', 'scaled', 'iou']) {
        assert.deepEqual(cacheWithoutMask.assessments[method], assessments[method]);
      }
      assert.deepEqual(cacheWithoutMask.outgoing, outgoing);
    }
  }
  const energyError = sim.getStats().relativeError;
  assert.ok(energyError < 1e-12);
  return { assessments, reference: sensedReference, matchAtProbe, matchedScene, directCacheParity, probe: {
    loopTicks, energyError, referenceEmittedAt: sim.referenceEmittedAt,
    referenceReturnedAt: sim.referenceReturnedAt, resultObservedAt: captures[1].start
  } };
}

function judge(assessment, target) {
  if (!assessment.accepted) return { status: target ? 'miss' : 'correct-abstention' };
  if (!target) return { status: 'false-positive' };
  const [, x, y, angle, scale] = target;
  const best = assessment.best;
  const angularDistance = Math.abs(best.angle - angle);
  const errors = {
    centerCells: Math.hypot(best.center[0] - x, best.center[1] - y),
    angleDegrees: Math.min(angularDistance, 360 - angularDistance),
    scale: Math.abs(best.scale - scale)
  };
  const correct = Object.entries(errors).every(([key, error]) => error <= TOLERANCES[key]);
  return { status: correct ? 'true-positive' : 'wrong-pose', errors };
}

const results = SCENES.map(fixture => {
  const { image, target, visible } = render(fixture);
  const measured = simulate(image, visible);
  const outcomes = Object.fromEntries(Object.entries(measured.assessments)
    .map(([name, assessment]) => [name, judge(assessment, fixture.target)]));
  return { fixture, image, target, visible, visibleTargetCells: target.filter((bit, index) => bit && visible[index]).length,
    targetCells: target.filter(Boolean).length, ...measured, outcomes };
});
for (const result of results) assert.deepEqual(result.reference, results[0].reference);
const selected = results.filter(result => result.outcomes.iou.status === 'true-positive').at(-1);
assert.ok(selected, 'The physical cut must have an accepted counterpart.');
const cut = simulate(selected.image, selected.visible, true);
assert.deepEqual(cut.reference, selected.reference);
assert.ok(Object.values(cut.assessments).every(assessment => !assessment.accepted));
assert.equal(cut.matchAtProbe.filter(Boolean).length, 0);
assert.ok(selected.matchAtProbe.some(Boolean));

const totals = Object.fromEntries(Object.keys(results[0].outcomes).map(method => {
  const counts = { 'true-positive': 0, 'wrong-pose': 0, miss: 0, 'false-positive': 0, 'correct-abstention': 0 };
  for (const result of results) counts[result.outcomes[method].status]++;
  return [method, counts];
}));
const report = {
  protocol: {
    size: SIZE, angles: ANGLES, scales: SCALES, threshold: THRESHOLD,
    minVisibleFraction: MIN_VISIBLE, tolerances: TOLERANCES,
    format: 'Controlled synthetic binary raster silhouettes at fixed lighting and view. No real photos or automatic segmentation.',
    trainedReference: 'Two intact, unit-scale T masks at 0 and 90 degrees. Each scene uses an independent run with the same seed and training input.',
    baseline: 'Fixed scale of 1. Score is the fraction of projected template cells occupied in the scene.',
    scaled: 'Identical foreground score, with all three scales available. The top alignment must be unique.',
    iou: 'Junction compares foreground intersection over union inside the tight bounding box of each projected template, without supplied visibility. Only this arm drives the downstream result probe.',
    knownMask: 'Separate oracle diagnostic. Uses foreground overlap on visible projected template cells with at least 60% of the template visible. The known mask is not used by any other arm.',
    cut: 'Pick the last correctly localized IoU-arm scene. Disable only the comparator input carrying C. The reference still travels and its energy is conserved.',
    cachedControl: 'Run the same comparator directly on the cached C measured at the distant probe and the exact input raster. Scores, selections, and the pulse drive are identical in every connected trial. For occluded scenes, replacing the oracle mask with an all-visible mask also leaves the three primary scores and the pulse drive identical.',
    judgment: 'A positive requires correct center, orientation, and scale. Accepted wrong poses are separate from negative-scene false positives.'
  },
  totals,
  results,
  disconnected: { scene: selected.fixture.id, ...cut }
};
const output = process.argv[2] ?? path.join(__dirname, '../outputs/sphere-stress-measurements.json');
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
for (const { fixture, assessments, outcomes } of results) {
  const row = Object.entries(outcomes).map(([method, { status }]) => `${method} ${status} ${assessments[method].score.toFixed(3)}`);
  console.log(`${fixture.id}: ${row.join(' | ')}`);
}
console.log(JSON.stringify({ totals, cut: selected.fixture.id }));
