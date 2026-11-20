'use strict';

const offsets = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]];
const MIN_AREA = 5;
const BOLT_ELONGATION = 1.8;
const END_FRACTION = 0.28;

function components(mask, size) {
  const visited = new Uint8Array(mask.length);
  const result = [];
  for (let index = 0; index < mask.length; index++) {
    if (!mask[index] || visited[index]) continue;
    const pixels = [];
    const pending = [index];
    visited[index] = 1;
    while (pending.length) {
      const current = pending.pop();
      const x = current % size, y = Math.floor(current / size);
      pixels.push([x, y]);
      for (const [dx, dy] of offsets) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || xx >= size || yy < 0 || yy >= size) continue;
        const neighbor = yy * size + xx;
        if (mask[neighbor] && !visited[neighbor]) {
          pending.push(neighbor);
          visited[neighbor] = 1;
        }
      }
    }
    if (pixels.length >= MIN_AREA) result.push(pixels);
  }
  return result;
}

function moments(pixels) {
  let x = 0, y = 0, x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [px, py] of pixels) {
    x += px;
    y += py;
    x0 = Math.min(x0, px);
    x1 = Math.max(x1, px);
    y0 = Math.min(y0, py);
    y1 = Math.max(y1, py);
  }
  const area = pixels.length;
  x /= area;
  y /= area;
  let xx = 0, yy = 0, xy = 0;
  for (const [px, py] of pixels) {
    const dx = px - x, dy = py - y;
    xx += dx * dx;
    yy += dy * dy;
    xy += dx * dy;
  }
  const spread = Math.hypot(xx - yy, 2 * xy);
  // Include the area of each pixel so that a one-pixel-wide cylinder has
  // a finite minor axis.
  const elongation = Math.sqrt((xx + yy + spread + area / 6) / (xx + yy - spread + area / 6));
  const angle = Math.atan2(2 * xy, xx - yy) / 2;
  return { x, y, angle, elongation, bounds: { x0, y0, x1, y1 }, area };
}

function thickness(projections, edge, limit) {
  let min = Infinity, max = -Infinity;
  for (const [along, across] of projections) {
    if (Math.abs(along - edge) > limit) continue;
    min = Math.min(min, across);
    max = Math.max(max, across);
  }
  return max - min;
}

function boltPose(pixels, { x, y, angle, bounds, area }) {
  const dx = Math.cos(angle), dy = Math.sin(angle);
  let min = Infinity, max = -Infinity;
  const projections = pixels.map(([px, py]) => {
    const along = (px - x) * dx + (py - y) * dy;
    const across = (py - y) * dx - (px - x) * dy;
    min = Math.min(min, along);
    max = Math.max(max, along);
    return [along, across];
  });
  const extent = max - min;
  const end = Math.max(1, extent * END_FRACTION);
  const direction = thickness(projections, max, end) > thickness(projections, min, end) ? -1 : 1;
  const middle = (min + max) / 2;
  return { x: x + dx * middle, y: y + dy * middle, axis: [direction * dx, direction * dy],
    length: extent + 1, bounds, area };
}

function observe(mask, size) {
  if (!Number.isInteger(size) || size <= 0 || mask.length !== size * size) {
    throw new RangeError('Une grille carrée est nécessaire.');
  }
  const bolts = [], nuts = [];
  for (const pixels of components(mask, size)) {
    const pose = moments(pixels);
    if (pose.elongation >= BOLT_ELONGATION) bolts.push(boltPose(pixels, pose));
    else nuts.push({ x: pose.x, y: pose.y, bounds: pose.bounds, area: pose.area });
  }
  return { bolts, nuts };
}

module.exports = { observe };
