'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { simulate } = require('./sphere-stress-run.cjs');

const directory = path.resolve(__dirname, '../outputs');
const data = JSON.parse(fs.readFileSync(path.join(directory, 'sphere-tless-wave-fixtures.json'), 'utf8'));
const visible = Array(data.size ** 2).fill(1);

function observations(run) {
  return {
    verdict: run.assessments.iou,
    referenceCells: run.reference.filter(Boolean).length,
    resultPulses: run.matchAtProbe.filter(Boolean).length,
    directCacheParity: run.directCacheParity,
    probe: run.probe,
  };
}

const results = data.candidates.map(({ supportMask, ...candidate }) => {
  const template = data.templates[candidate.reference];
  // The same training view is injected twice in alignment. This tests transport
  // and verification of a fixed real reference, independently of view learning.
  const examples = [template, template];
  const strict = simulate(supportMask, visible, false, examples);
  const local = simulate(supportMask, visible, false, examples, { locationOnly: true });
  for (const measured of [strict, local]) {
    assert.deepEqual(measured.reference, template);
    assert.ok(measured.directCacheParity);
  }
  assert.equal(local.assessments.iou.score, strict.assessments.iou.score);
  return { ...candidate, strictPose: observations(strict), locationOnly: observations(local) };
});

const accepted = results.filter(row => row.locationOnly.resultPulses > 0);
assert.ok(accepted.length);
const first = accepted.reduce((best, row) => row.locationOnly.verdict.score > best.locationOnly.verdict.score ? row : best);
const picked = data.candidates.find(row => row.id === first.id && row.box.every((n, i) => n === first.box[i]));
const template = data.templates[picked.reference];
const dropped = simulate(picked.supportMask, visible, true, [template, template], { locationOnly: true });
assert.equal(dropped.assessments.iou.accepted, false);
assert.equal(dropped.matchAtProbe.some(Boolean), false);

const report = {
  scope: 'The wave comparator receives depth support from fixed full-frame detector proposals and their selected training view. It does not search the original photo or receive test annotations.',
  reference: 'Two identical, prealigned copies of one training-visible mask form C. This checks transfer and shape verification, independently of learning across viewpoints.',
  protocol: 'The fixed overlap threshold remains 0.86. The location-only variant accepts tied poses if the rectangle containing their centers has a diagonal of at most two cells. Pose is left ambiguous on those matches. This variant was evaluated on the same panel after observing tied poses in the initial run.',
  results, cut: { id: first.id, box: first.box, connected: first.locationOnly.resultPulses, ...observations(dropped) },
};
const output = process.argv[2] ?? path.join(directory, 'sphere-tless-wave-measurements.json');
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
for (const row of results) console.log(row.id, row.box.join(','), row.locationOnly.verdict.score.toFixed(3),
  row.strictPose.resultPulses, row.locationOnly.resultPulses);
console.log('Disconnected control:', first.id, first.locationOnly.resultPulses, '->', report.cut.resultPulses);
