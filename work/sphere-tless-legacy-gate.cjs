'use strict';

const fs = require('node:fs');
const { verifyCached } = require('./sphere-stress-run.cjs');

const [inputs, rawFile, destination] = process.argv.slice(2);
const { candidates, templates } = JSON.parse(fs.readFileSync(inputs, 'utf8'));
const raw = JSON.parse(fs.readFileSync(rawFile, 'utf8'));
const frames = new Map(raw.frames.map(frame => [frame.id, []]));
for (const candidate of candidates) {
  const decision = verifyCached(templates[candidate.reference], candidate.supportMask, { locationOnly: true });
  frames.get(candidate.id).push({
    box: candidate.box, score: decision.score, accepted: decision.accepted, reference: candidate.reference,
  });
}
fs.writeFileSync(destination, JSON.stringify({ frames: [...frames].map(([id, detections]) =>
  ({ id, detections })) }, null, 2) + '\n');
