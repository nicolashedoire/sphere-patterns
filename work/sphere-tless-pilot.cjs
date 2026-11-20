'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const out = path.resolve(__dirname, '../outputs');
const panel = JSON.parse(fs.readFileSync(path.join(out, 'sphere-tless-fixtures.json'), 'utf8'));
const SIZE = panel.size;
const CENTER = (SIZE - 1) / 2;
const CELLS = Array.from({ length: SIZE ** 2 }, (_, i) => [i % SIZE, Math.floor(i / SIZE)]);
const THRESHOLD = .86;
const DEGREES = Array.from({ length: 24 }, (_, i) => i * 15);
const SHIFT = [-2, -1, 0, 1, 2];
const BORDER = CELLS.flatMap(([x, y], i) => x === 0 || y === 0 || x === SIZE - 1 || y === SIZE - 1 ? [i] : []);

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) / 2)];
}

function foregroundContrast(luma) {
  const borderValue = median(BORDER.map(i => luma[i]));
  return luma.map(value => Math.abs(value - borderValue));
}

function chooseRgbThreshold(examples) {
  const training = examples.flatMap(item => foregroundContrast(item.rgb).map((value, i) =>
    ({ value, part: item.visibleMask[i] > 0 })));
  const positives = training.filter(item => item.part).length;
  const negatives = training.length - positives;
  let best = { cutoff: 1, balanced: -1 };
  for (let cutoff = 1; cutoff <= 255; cutoff++) {
    const tp = training.filter(item => item.part && item.value >= cutoff).length;
    const tn = training.filter(item => !item.part && item.value < cutoff).length;
    const balanced = (tp / positives + tn / negatives) / 2;
    if (balanced > best.balanced) best = { cutoff, balanced };
  }
  return best;
}

function depthInput(values) {
  const nearCenter = values.filter((value, i) => value > 0 &&
    Math.abs(CELLS[i][0] - CENTER) <= 3 && Math.abs(CELLS[i][1] - CENTER) <= 3);
  const depth = median(nearCenter);
  return values.map(value => Number(depth > 0 && value > 0 && Math.abs(value - depth) <= 25));
}

function transform(mask, angle, dx, dy) {
  const radians = angle * Math.PI / 180;
  const c = Math.cos(radians), s = Math.sin(radians);
  return CELLS.map(([x, y]) => {
    const rx = x - CENTER - dx, ry = y - CENTER - dy;
    const ix = Math.round(rx * c + ry * s + CENTER), iy = Math.round(-rx * s + ry * c + CENTER);
    return ix >= 0 && ix < SIZE && iy >= 0 && iy < SIZE ? mask[iy * SIZE + ix] : 0;
  });
}

function iou(left, right) {
  let intersection = 0, union = 0;
  for (let i = 0; i < left.length; i++) {
    intersection += left[i] & right[i];
    union += left[i] | right[i];
  }
  return union ? intersection / union : 0;
}

function searchBank(mask) {
  return DEGREES.flatMap(angle => SHIFT.flatMap(dx => SHIFT.map(dy =>
    ({ angle, dx, dy, mask: transform(mask, angle, dx, dy) }))));
}

function compare(poses, candidate) {
  let best = null;
  for (const pose of poses) {
    const score = iou(pose.mask, candidate);
    if (!best || score > best.score) best = { ...pose, score };
  }
  return best;
}

const [a, b] = panel.examples.map(item => item.visibleMask);
const bAligned = compare(searchBank(b), a);
const reference = a.map((bit, i) => bit & bAligned.mask[i]);
assert.ok(reference.some(Boolean));
const poses = searchBank(reference);
const rgbThreshold = chooseRgbThreshold(panel.examples);
const inputs = {
  rgb: item => foregroundContrast(item.rgb).map(value => Number(value >= rgbThreshold.cutoff)),
  depth: item => depthInput(item.depthMm),
  annotatedMask: item => item.visibleMask,
};

const samples = panel.cases.map(item => {
  const methods = Object.fromEntries(Object.entries(inputs).map(([name, fn]) => {
    const mask = fn(item);
    const best = compare(poses, mask);
    return [name, {
      maskIoU: iou(mask, item.visibleMask),
      score: best.score, accepted: best.score >= THRESHOLD,
      pose: { angle: best.angle, dx: best.dx, dy: best.dy },
    }];
  }));
  const { scene, frame, index, part, band, visibility, box } = item;
  return { scene, frame, index, part, band, visibility, box, target: part === panel.targetPart, methods };
});
const summary = Object.fromEntries(Object.keys(inputs).map(name => [name,
  Object.fromEntries(['all', 'high', 'partial'].map(band => {
    const rows = samples.filter(row => band === 'all' || row.band === band);
    const positives = rows.filter(row => row.target), negatives = rows.filter(row => !row.target);
    return [band, {
      positive: positives.length,
      matched: positives.filter(row => row.methods[name].accepted).length,
      negative: negatives.length,
      falseAccepts: negatives.filter(row => row.methods[name].accepted).length,
      medianTargetScore: median(positives.map(row => row.methods[name].score)),
      medianTargetMaskIoU: median(positives.map(row => row.methods[name].maskIoU)),
    }];
  }))
]));

const report = {
  source: panel.source, targetPart: panel.targetPart, photos: 12, physicalScenes: 6,
  inputs: 'The amodal bounding box for each proposed candidate is supplied by T-LESS. RGB and depth arms use only the camera crop at recognition time. The annotated-mask arm supplies the visible mask as an oracle comparison. This is fixed-candidate classification outside the wave network.',
  calibration: { rgbThreshold, depthToleranceMm: 25, acceptAt: THRESHOLD, degrees: DEGREES, shifts: SHIFT },
  reference: { mask: reference, cells: reference.filter(Boolean).length, trainingViews: [0, 12],
    trainingAgreement: bAligned.score, orientationDegrees: bAligned.angle },
  summary, samples,
};
const destination = process.argv[2] ?? path.join(out, 'sphere-tless-measurements.json');
fs.writeFileSync(destination, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ referenceCells: report.reference.cells, summary }, null, 2));
