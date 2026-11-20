'use strict';

const fs = require('node:fs');
const path = require('node:path');

const source = process.argv[2] ?? path.join(__dirname, '../outputs/sphere-occlusion-measurements.json');
const target = process.argv[3] ?? path.join(__dirname, '../outputs/sphere-occlusion-figure.svg');
const report = JSON.parse(fs.readFileSync(source, 'utf8'));
const size = report.protocol.size;
const pictured = report.results.find(result => result.fixture.id === report.disconnected.scene);
const partial = report.results.filter(result => result.fixture.id.startsWith('partial-'));

const palette = {
  backdrop: '#F1EFE8',
  grid: '#E3E0D6',
  ink: '#203134',
  muted: '#536361',
  normal: '#596D78',
  occluder: '#D6AE84',
  match: '#115D56',
  accentWash: '#D8E7E0'
};

function grid(pixels, visible, matched, x, y) {
  const highlighted = new Set(matched);
  const cell = 8;
  const rectangles = [];
  for (const index of pixels.keys()) {
    const fill = highlighted.has(index) ? palette.match : pixels[index] ? palette.normal :
      visible[index] === 0 ? palette.occluder : palette.grid;
    const cellX = x + (index % size) * cell;
    const cellY = y + Math.floor(index / size) * cell;
    rectangles.push(`<rect x="${cellX}" y="${cellY}" width="${cell - 0.6}" height="${cell - 0.6}" fill="${fill}"/>`);
  }
  return rectangles.join('');
}

function label(x, y, text, style = 'body') {
  return `<text x="${x}" y="${y}" class="${style}">${text}</text>`;
}

const ordinaryFound = partial.filter(result => result.correct.ordinary).length;
const awareFound = partial.filter(result => result.correct.aware).length;
const view = [
  '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="484" viewBox="0 0 1120 484" role="img" aria-labelledby="title desc">',
  '<title id="title">Recognition with a supplied occlusion mask</title>',
  '<desc id="desc">Three panels show a learned T template, a scene with a hidden section, and the matching cells recovered by the probe. A table compares ordinary overlap with visibility-aware overlap.</desc>',
  `<style>text{font-family:system-ui,sans-serif}.title{font-size:38px;font-weight:700;fill:${palette.ink}}.subtitle{font-size:17px;fill:${palette.muted}}.heading{font-size:16px;font-weight:700;fill:${palette.ink}}.body{font-size:14px;fill:${palette.ink}}.muted{font-size:13px;fill:${palette.muted}}.success{font-size:14px;font-weight:700;fill:${palette.match}}</style>`,
  `<rect width="1120" height="484" fill="${palette.backdrop}"/>`,
  label(32, 56, 'Finding a part behind an occluder', 'title'),
  label(32, 88, 'The comparator receives a known visibility mask and the returning reference wave.', 'subtitle'),
  label(32, 133, 'Learned reference C', 'heading'),
  label(260, 133, `Occluded scene · ${pictured.fixture.pose[2]}°`, 'heading'),
  label(488, 133, 'Matching cells at the probe', 'heading'),
  grid(pictured.reference, [], [], 32, 149),
  grid(pictured.image, pictured.visible, [], 260, 149),
  grid(pictured.image, pictured.visible, pictured.matchedScene, 488, 149),
  label(32, 372, 'Formed from two intact views.', 'muted'),
  label(260, 372, 'Tan cells are hidden.', 'muted'),
  label(488, 372, `${pictured.aware.best.matches} of ${pictured.aware.best.visible} visible reference cells match.`, 'muted'),
  `<rect x="716" y="122" width="372" height="245" rx="12" fill="${palette.accentWash}"/>`,
  label(734, 153, 'Partially occluded targets', 'heading'),
  label(734, 181, 'Target angle', 'muted'),
  label(843, 181, 'Ordinary', 'muted'),
  label(957, 181, 'Mask-aware', 'muted')
];

for (const [index, result] of partial.entries()) {
  const y = 213 + 33 * index;
  view.push(
    label(734, y, `${result.fixture.pose[2]}°`),
    label(843, y, result.correct.ordinary ? 'Found' : 'Abstained'),
    label(957, y, result.correct.aware ? 'Found' : result.aware.tied ? 'Tie · abstained' : 'Abstained',
      result.correct.aware ? 'success' : 'body')
  );
}

view.push(
  `<path d="M734 306h336" stroke="${palette.muted}" opacity=".4"/>`,
  label(734, 332, 'Located'),
  label(843, 332, `${ordinaryFound} / ${partial.length}`),
  label(957, 332, `${awareFound} / ${partial.length}`, 'success'),
  label(32, 424, 'The intact scene passes both methods. Both abstain on heavy occlusion, an L distractor and a shortened-T lookalike.'),
  label(32, 450, `Gating C at the comparator silences the ${pictured.fixture.pose[2]}° match. The reference returns after ${pictured.probe.loopTicks.toLocaleString('en-US')} ticks.`, 'muted'),
  '</svg>'
);
fs.writeFileSync(target, view.join('\n') + '\n');
