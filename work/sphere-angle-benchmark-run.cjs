'use strict';

const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const RAD = Math.PI / 180;
const KEY_STRIDE = 512;
const SCENE = { width: 142, height: 124 };
const TRAIN = { width: 58, height: 58 };
// The threshold, angles and scene groups are fixed before rendering. Target
// positions and orientations are available only to the scorer.
const protocol = {
  seed: 1257913,
  quarterTurns: [0, 90, 180, 270],
  stepDegrees: 2,
  thresholdIoU: 0.86,
  winnerMargin: 0.025,
  poseToleranceDegrees: 4,
  minComponentArea: 24,
  obliqueAngles: [9, 17, 33, 47, 63, 77, 109, 121, 132, 153, 171, 207, 223, 247, 281, 317, 339, 353],
  cardinalRepeats: 2,
  negativeOrdinary: 12,
  negativeLookalike: 12,
  trainingPoses: [{ angle: 0, x: 26.25, y: 30.5 }, { angle: 90, x: 29.25, y: 27.5 }]
};

function insideCircle(x, y, cx, cy, radius) {
  return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2;
}

// These silhouettes generate raster fixtures. The detector has no access to
// shape membership, fixture labels or target pose.
const silhouette = {
  t(x, y) {
    return ((Math.abs(x) <= 17 && y >= -14 && y <= -4) || (Math.abs(x) <= 5 && y >= -4 && y <= 16)) &&
      !insideCircle(x, y, -11, -9, 2.7) && !insideCircle(x, y, 11, -9, 2.7) && !insideCircle(x, y, 0, 9, 2.7);
  },
  lookalike(x, y) {
    return ((Math.abs(x) <= 17 && y >= -14 && y <= -4) || (Math.abs(x) <= 5 && y >= -4 && y <= 16)) &&
      !insideCircle(x, y, -11, -9, 2.7) && !insideCircle(x, y, 7, -9, 2.7) && !insideCircle(x, y, 0, 7, 2.7);
  },
  shortT(x, y) {
    return (x >= -15 && x <= 9 && y >= -14 && y <= -5) || (Math.abs(x) <= 6 && y >= -5 && y <= 14);
  },
  washer(x, y) {
    return insideCircle(x, y, 0, 0, 11) && !insideCircle(x, y, 0, 0, 6);
  },
  angle(x, y) {
    return (x >= -12 && x <= 12 && y >= 5 && y <= 12) || (x >= -12 && x <= -5 && Math.abs(y) <= 12);
  },
  link(x, y) {
    return (Math.abs(x) <= 16 && Math.abs(y) <= 5.5) &&
      !insideCircle(x, y, -11, 0, 2.6) && !insideCircle(x, y, 11, 0, 2.6);
  }
};

function render({ width, height }, objects) {
  const pixels = new Uint8Array(width * height);
  for (const { shape, x: cx, y: cy, angle } of objects) {
    const ink = silhouette[shape];
    const cos = Math.cos(angle * RAD), sin = Math.sin(angle * RAD);
    for (let y = Math.max(0, Math.floor(cy - 26)); y <= Math.min(height - 1, Math.ceil(cy + 26)); y++) {
      for (let x = Math.max(0, Math.floor(cx - 26)); x <= Math.min(width - 1, Math.ceil(cx + 26)); x++) {
        let covered = 0;
        for (const sy of [-0.25, 0.25]) for (const sx of [-0.25, 0.25]) {
          const dx = x + sx - cx, dy = y + sy - cy;
          covered += Number(ink(dx * cos + dy * sin, -dx * sin + dy * cos));
        }
        if (covered >= 2) pixels[y * width + x] = 1;
      }
    }
  }
  return { width, height, pixels };
}

function components(image) {
  const { width, height, pixels } = image;
  const visited = new Uint8Array(pixels.length), found = [];
  const neighbors = [-1, 0, 1];
  for (const [index, value] of pixels.entries()) {
    if (!value || visited[index]) continue;
    const indices = [index];
    visited[index] = 1;
    for (let head = 0; head < indices.length; head++) {
      const x = indices[head] % width, y = Math.floor(indices[head] / width);
      for (const dy of neighbors) for (const dx of neighbors) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const neighbor = ny * width + nx;
        if (pixels[neighbor] && !visited[neighbor]) {
          visited[neighbor] = 1;
          indices.push(neighbor);
        }
      }
    }
    if (indices.length < protocol.minComponentArea) continue;
    const absolute = indices.map(i => [i % width, Math.floor(i / width)]);
    const center = absolute.reduce(([sumX, sumY], [x, y]) => [sumX + x, sumY + y], [0, 0])
      .map(sum => sum / absolute.length);
    const [cx, cy] = center.map(Math.round);
    const local = new Set(absolute.map(([x, y]) => (y - cy) * KEY_STRIDE + x - cx));
    found.push({ absolute, center, local });
  }
  return found;
}

function largest(image) {
  return components(image).sort((a, b) => b.absolute.length - a.absolute.length)[0];
}

function coordinates(encoded) {
  const y = Math.round(encoded / KEY_STRIDE);
  return [encoded - y * KEY_STRIDE, y];
}

function rotate(mask, angle) {
  const points = [...mask].map(coordinates);
  const radius = Math.ceil(Math.max(...points.map(([x, y]) => Math.hypot(x, y)))) + 1;
  const cos = Math.cos(angle * RAD), sin = Math.sin(angle * RAD);
  const out = [];
  // Rasterize an oriented template by backwards resampling. The physical
  // fixture generator uses supersampled shape geometry instead.
  for (let y = -radius; y <= radius; y++) for (let x = -radius; x <= radius; x++) {
    const ox = Math.round(x * cos + y * sin);
    const oy = Math.round(-x * sin + y * cos);
    if (mask.has(oy * KEY_STRIDE + ox)) out.push(y * KEY_STRIDE + x);
  }
  return out;
}

function compare(candidate, orientations) {
  let best = { score: -1 };
  for (const { angle, points } of orientations) {
    if (Math.min(points.length, candidate.size) / Math.max(points.length, candidate.size) <= best.score) continue;
    for (const dy of [-1, 0, 1]) for (const dx of [-1, 0, 1]) {
      const offset = dy * KEY_STRIDE + dx;
      let shared = 0;
      for (const value of points) shared += Number(candidate.has(value + offset));
      const score = shared / (candidate.size + points.length - shared);
      if (score > best.score) best = { angle, dx, dy, score };
    }
  }
  return best;
}

function learn(a, b) {
  const first = largest(a), second = largest(b);
  const candidates = protocol.quarterTurns.map(angle => ({ angle, points: rotate(second.local, angle) }));
  const selected = compare(first.local, candidates);
  const shift = selected.dy * KEY_STRIDE + selected.dx;
  const common = candidates.find(item => item.angle === selected.angle).points
    .map(key => key + shift).filter(key => first.local.has(key));
  return { common: new Set(common), selected, areas: [first.local.size, second.local.size] };
}

function readScene(parts, orientations) {
  if (orientations === null) return { status: 'reference-disconnected', top: null, next: null, match: null };
  const ranked = parts.map(part => ({ ...compare(part.local, orientations),
    x: Number(part.center[0].toFixed(3)), y: Number(part.center[1].toFixed(3)) }))
    .sort((a, b) => b.score - a.score);
  const [top = null, next = null] = ranked;
  let status = 'recognized';
  if (!top || top.score < protocol.thresholdIoU) status = 'below-threshold';
  else if (next && top.score - next.score < protocol.winnerMargin) status = 'ambiguous';
  return { status, top, next, match: status === 'recognized' ? top : null };
}

function fixtureRng() {
  let state = protocol.seed >>> 0;
  return limit => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return Math.floor(state / 4294967296 * limit);
  };
}

function fixtures() {
  const random = fixtureRng();
  const cases = [
    ...protocol.quarterTurns.flatMap(angle => Array.from({ length: protocol.cardinalRepeats }, () => ({ group: 'orthogonal', angle }))),
    ...protocol.obliqueAngles.map(angle => ({ group: 'oblique', angle })),
    ...Array.from({ length: protocol.negativeOrdinary }, () => ({ group: 'absent' })),
    ...Array.from({ length: protocol.negativeLookalike }, () => ({ group: 'lookalike' }))
  ];
  return cases.map(({ group, angle }, index) => {
    const slots = [[34, 31], [106, 31], [34, 92], [106, 92]];
    const focal = random(slots.length);
    const distractors = ['washer', 'angle', 'link', 'shortT'];
    const objects = slots.map(([x, y], i) => ({
      shape: i === focal && angle !== undefined ? 't' : i === focal && group === 'lookalike' ? 'lookalike' : distractors[i],
      x: x + (random(25) - 12) / 4,
      y: y + (random(25) - 12) / 4,
      angle: i === focal && angle !== undefined ? angle : random(360)
    }));
    const image = render(SCENE, objects);
    const actual = angle === undefined ? null : largest(render(SCENE, [objects[focal]])).center;
    return { id: 'scene-' + String(index + 1).padStart(2, '0'), group, angle, actual, objects, image, parts: components(image) };
  });
}

function hash(data) {
  return createHash('sha256').update(data).digest('hex');
}

function audit(fixture, result) {
  const { actual, angle } = fixture;
  const { match } = result;
  if (!actual) return { localization: match === null, angleError: null, pose: null };
  const localization = !!match && Math.hypot(match.x - actual[0], match.y - actual[1]) < 2;
  const angleError = localization ? Math.abs((match.angle - angle + 540) % 360 - 180) : null;
  return { localization, angleError, pose: angleError !== null && angleError <= protocol.poseToleranceDegrees };
}

function count(rows, arm) {
  return Object.fromEntries(['orthogonal', 'oblique', 'absent', 'lookalike'].map(group => {
    const scoped = rows.filter(row => row.group === group);
    const negative = group === 'absent' || group === 'lookalike';
    const counts = { scenes: scoped.length, detections: scoped.filter(row => row[arm].result.match).length };
    if (negative) return [group, { ...counts, falsePositives: counts.detections }];
    const localized = scoped.filter(row => row[arm].audit.localization);
    return [group, { ...counts, localized: localized.length,
      poseWithinTolerance: localized.filter(row => row[arm].audit.pose).length,
      meanAngleError: localized.length ?
        localized.reduce((total, row) => total + row[arm].audit.angleError, 0) / localized.length : null }];
  }));
}

function drawMask(image, x, y, scale) {
  const runs = [];
  for (let row = 0; row < image.height; row++) {
    let start = -1;
    for (let col = 0; col <= image.width; col++) {
      if (col < image.width && image.pixels[row * image.width + col]) {
        if (start === -1) start = col;
      } else if (start !== -1) {
        runs.push(`M${start} ${row}h${col - start}v1h-${col - start}z`);
        start = -1;
      }
    }
  }
  return `<g transform="translate(${x} ${y}) scale(${scale})"><path fill="#becedc" d="${runs.join('')}"/></g>`;
}

function marker(result, x, y, scale, color) {
  if (!result.match) return '';
  const { x: cx, y: cy } = result.match;
  return `<circle cx="${x + cx * scale}" cy="${y + cy * scale}" r="50" fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="5 4"/>`;
}

function svg(scene, near, rows, stats) {
  const positive = rows.find(item => item.id === scene.id);
  const negative = rows.find(item => item.id === near.id);
  const small = (x, y, value, fill = '#adc4d3') => `<text x="${x}" y="${y}" fill="${fill}" font-size="20">${value}</text>`;
  const ti = (x, y, value) => `<text x="${x}" y="${y}" font-size="23" font-weight="700">${value}</text>`;
  const card = (x) => `<rect x="${x}" y="140" width="353" height="384" rx="16" fill="#162a3a" stroke="#314c5f"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="620" viewBox="0 0 1200 620" font-family="system-ui,sans-serif" fill="#f3f8fc" role="img" aria-labelledby="title">
<title id="title">Recognizing the T-shaped part at oblique angles</title>
<rect width="1200" height="620" fill="#0b1824"/>
<text x="44" y="64" font-size="42" font-weight="750">Recognizing a rotated part</text>
${small(46, 101, 'Geometric benchmark outside the sphere · synthetic raster scenes')}
${card(43)}${card(423)}${card(803)}
${ti(65, 181, 'Quarter turns')}${ti(445, 181, 'Search in 2° steps')}${ti(825, 181, 'Near-match decoy')}
${drawMask(scene.image, 76, 200, 2)}${drawMask(scene.image, 456, 200, 2)}${drawMask(near.image, 836, 200, 2)}
${marker(positive.quarters.result, 76, 200, 2, '#efbf82')}${marker(positive.dense.result, 456, 200, 2, '#61e5cf')}${marker(negative.dense.result, 836, 200, 2, '#efbf82')}
${small(65, 481, positive.quarters.result.match ? 'Candidate accepted' : 'No candidate accepted')}
${small(445, 481, positive.dense.result.match ? `Part found at ${positive.dense.result.match.angle}°` : 'No candidate accepted', '#61e5cf')}
${small(825, 481, negative.dense.result.match ? 'Mistaken for the target' : 'Rejected at the chosen threshold', '#efbf82')}
${small(46, 576, `${stats.dense.oblique.localized}/${stats.dense.oblique.scenes} oblique targets located · ${stats.dense.lookalike.falsePositives}/${stats.dense.lookalike.scenes} false positives among near matches`)}
</svg>\n`;
}

function run() {
  const views = protocol.trainingPoses.map(pose => render(TRAIN, [{ shape: 't', ...pose }]));
  const { common, selected, areas } = learn(...views);
  assert.ok(common.size > 300);
  const quarterTemplates = protocol.quarterTurns.map(angle => ({ angle, points: rotate(common, angle) }));
  const denseTemplates = Array.from({ length: 360 / protocol.stepDegrees }, (_, i) => i * protocol.stepDegrees)
    .map(angle => ({ angle, points: rotate(common, angle) }));
  const all = fixtures();
  const rows = all.map(fixture => {
    const { id, group, angle = null, actual, objects, image, parts } = fixture;
    const arms = { quarters: quarterTemplates, dense: denseTemplates, cut: null };
    const outcomes = Object.fromEntries(Object.entries(arms).map(([arm, orientations]) => {
      const result = readScene(parts, orientations);
      return [arm, { result, audit: audit(fixture, result) }];
    }));
    return { id, group, angle, targetCenter: actual, objects, imageSHA256: hash(image.pixels),
      components: parts.length, ...outcomes };
  });
  const counts = Object.fromEntries(['quarters', 'dense', 'cut'].map(arm => [arm, count(rows, arm)]));
  assert.ok(rows.every(row => row.components === 4), 'Objects must remain separate within each scene.');
  assert.ok(rows.every(row => row.cut.result.match === null));
  const report = {
    scope: 'Complementary geometric benchmark outside the sphere. Binary synthetic images use a fixed scale and viewpoint, with separate silhouettes.',
    method: 'Extract eight-connected components. Form the reference by intersecting two aligned views, then score each rotated and recentered component by intersection over union (IoU). The localizer receives only scene and reference pixels. Generated poses serve as ground truth.',
    cutControl: 'Run the same scenes with the reference disconnected from the software comparator.',
    protocol, scene: SCENE, training: TRAIN,
    reference: { trainingImageSHA256: views.map(view => hash(view.pixels)), areas, alignmentOfSecondView: selected,
      coordinates: [...common].sort((a, b) => a - b).map(coordinates), area: common.size },
    counts, rows
  };
  const output = name => path.join(__dirname, '../outputs/sphere-angle-benchmark-' + name);
  fs.writeFileSync(output('measurements.json'), JSON.stringify(report, null, 2) + '\n');
  const example = all.find(item => item.angle === 47);
  const decoy = all.find(item => item.group === 'lookalike');
  fs.writeFileSync(output('figure.svg'), svg(example, decoy, rows, counts));
  console.log(JSON.stringify({ referenceArea: common.size, trainingAlignment: selected, counts }, null, 2));
}

if (require.main === module) run();
