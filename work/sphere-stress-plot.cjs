'use strict';

const fs = require('node:fs');
const path = require('node:path');

const input = process.argv[2] ?? path.join(__dirname, '../outputs/sphere-stress-measurements.json');
const output = process.argv[3] ?? path.join(__dirname, '../outputs/sphere-stress-figure.svg');
const { protocol, results } = JSON.parse(fs.readFileSync(input, 'utf8'));

const methods = [
  ['fixed', 'Fixed scale'],
  ['scaled', 'Scale bank'],
  ['iou', 'Local IoU'],
  ['knownMask', 'Known mask']
];
const stages = [
  ['isolated', 'Isolated silhouette'],
  ['scale', 'Scale change'],
  ['clutter', 'Clutter and lookalikes'],
  ['occlusion', 'Occlusion']
];
const outcomes = {
  'true-positive': ['Correct pose', '#167b60'],
  'wrong-pose': ['Wrong pose', '#a65215'],
  miss: ['Miss', '#66798a'],
  'false-positive': ['False alarm', '#ae3650'],
  'correct-abstention': ['Abstained', '#167b60']
};
const thumbnails = {
  'clean-target': 'Unit-scale T',
  'clean-negative': 'L distractor',
  'small-target': 'T at 75% scale',
  'large-target': 'T at 125% scale',
  'clutter-target': 'T amid four distractors',
  'intruding-stripe': 'Stripe overlaps the T',
  'lookalike-negative': 'Shortened T',
  'solid-negative': 'Solid block',
  'covered-clutter-target': 'Scaled T, partly covered',
  'covered-oblique-target': 'Oblique T, partly covered',
  'heavily-covered-target': 'Oblique T, heavily covered',
  'covered-negative': 'Shortened T, partly covered'
};
const width = 1280, cardWidth = 291, cardHeight = 165, stageHeight = 204;
const margin = 36, gap = 15, firstRow = 142, pixel = 4;
const height = firstRow + stages.length * stageHeight + 46;
const svg = [];
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
function label(x, y, text, size, color = '#143041', weight = 500) {
  return `<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-weight="${weight}">${escape(text)}</text>`;
}
function rect(x, y, w, h, color, radius = 0) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${color}"/>`;
}
function drawCard(result, x, y) {
  const pictureX = x + 13, pictureY = y + 51;
  const pictureSize = protocol.size * pixel;
  const marks = new Set(result.matchedScene);
  svg.push(rect(x, y, cardWidth, cardHeight, '#ffffff', 11));
  svg.push(label(x + 13, y + 23, thumbnails[result.fixture.id], 14, '#143041', 700));
  const note = result.fixture.target ? `${result.visibleTargetCells} of ${result.targetCells} target cells visible` : 'No target in the scene';
  svg.push(label(x + 13, y + 39, note, 11, '#526677'));
  svg.push(rect(pictureX, pictureY, pictureSize, pictureSize, '#ebf0f3', 3));
  for (let index = 0; index < result.image.length; index++) {
    if (!result.image[index] && result.visible[index]) continue;
    const color = !result.visible[index] ? '#e9b97a' : marks.has(index) ? '#087d8b' : '#243b49';
    svg.push(rect(pictureX + (index % protocol.size) * pixel,
      pictureY + Math.floor(index / protocol.size) * pixel, pixel, pixel, color));
  }
  for (const [row, [method, name]] of methods.entries()) {
    const line = y + 66 + 23 * row;
    const [outcome, color] = outcomes[result.outcomes[method].status];
    svg.push(label(x + 125, line, name, 10, '#526677'));
    svg.push(label(x + 194, line, outcome, 10, color, 700));
    svg.push(label(x + 262, line, result.assessments[method].score.toFixed(2), 10, '#526677', 500));
  }
}
svg.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title description">`);
svg.push('<title id="title">Complexity tests for part recognition in the wave sphere</title>');
svg.push('<desc id="description">Twelve synthetic scenes compare fixed-scale overlap, multiscale overlap, local intersection over union, and overlap with an oracle visibility mask.</desc>');
svg.push('<g font-family="Inter, Arial, sans-serif">');
svg.push(rect(0, 0, width, height, '#f2f5f7'));
svg.push(rect(0, 0, width, 116, '#143041'));
svg.push(label(margin, 48, 'Part recognition under increasing scene complexity', 31, '#ffffff', 750));
svg.push(label(margin, 79, '12 synthetic scenes on a 25 × 25 grid   ·   One returning reference   ·   Four scoring rules', 15, '#d9e8ec'));
for (const [row, [stage, name]] of stages.entries()) {
  const y = firstRow + stageHeight * row;
  svg.push(label(margin, y, name, 17, '#143041', 750));
  results.filter(result => result.fixture.stage === stage).forEach((result, column) =>
    drawCard(result, margin + (cardWidth + gap) * column, y + 15));
}
const legendY = height - 25;
svg.push(rect(margin, legendY - 10, 11, 11, '#087d8b', 2));
svg.push(label(margin + 17, legendY, 'Matched scene cells in the local IoU arm', 12, '#526677'));
svg.push(rect(margin + 285, legendY - 10, 11, 11, '#e9b97a', 2));
svg.push(label(margin + 302, legendY, 'Hidden cells, known only to the mask control', 12, '#526677'));
svg.push(label(margin + 685, legendY, 'A correct pose requires position, angle and scale within tolerance.', 12, '#526677'));
svg.push('</g></svg>');
fs.writeFileSync(output, svg.join('\n') + '\n');
