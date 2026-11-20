'use strict';

// The junction receives luminance samples from the scene and from the returned C.
// All positions, scales and rotations are searched on a fixed grid. Small local
// refinements use the same received samples around the strongest coarse peaks.
const DEFAULT_ANGLES = Array.from({ length: 24 }, (_, index) => 15 * index);
const DEFAULT_SCALES = [0.34, 0.43, 0.54, 0.67, 0.82, 1, 1.21];
const PEAKS_PER_TRANSFORM = 3;
const EPSILON = 1e-10;

function rasterValue(pixels, size, x, y) {
  return x >= 0 && y >= 0 && x < size && y < size ? pixels[y * size + x] : 0;
}

function interpolate(pixels, size, x, y) {
  const left = Math.floor(x), top = Math.floor(y);
  const horizontal = x - left, vertical = y - top;
  return (1 - vertical) * ((1 - horizontal) * rasterValue(pixels, size, left, top) +
    horizontal * rasterValue(pixels, size, left + 1, top)) + vertical * (
    (1 - horizontal) * rasterValue(pixels, size, left, top + 1) +
    horizontal * rasterValue(pixels, size, left + 1, top + 1));
}

function warpReference(pixels, size, angle, scale, cutoff) {
  const center = (size - 1) / 2;
  const reach = Math.ceil(Math.SQRT2 * center * scale) + 2;
  const side = reach * 2 + 1;
  const warped = new Float32Array(side * side);
  const cosine = Math.cos(angle * Math.PI / 180) / scale;
  const sine = Math.sin(angle * Math.PI / 180) / scale;
  let left = side, right = -1, top = side, bottom = -1;
  for (let y = 0; y < side; y++) {
    const dy = y - reach;
    for (let x = 0; x < side; x++) {
      const dx = x - reach;
      const value = interpolate(pixels, size, center + dx * cosine + dy * sine,
        center - dx * sine + dy * cosine);
      warped[y * side + x] = value;
      if (value <= cutoff) continue;
      left = Math.min(left, x);
      right = Math.max(right, x);
      top = Math.min(top, y);
      bottom = Math.max(bottom, y);
    }
  }
  if (right - left < 1 || bottom - top < 1) return null;

  const padding = scale < 0.5 ? 1 : 2;
  const width = right - left + 1 + padding * 2;
  const height = bottom - top + 1 + padding * 2;
  const crop = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      crop[y * width + x] = rasterValue(warped, side, left + x - padding, top + y - padding);
    }
  }
  return { pixels: crop, width, height, padding };
}

function gradient(pixels, width, height) {
  const gx = new Float32Array(pixels.length), gy = new Float32Array(pixels.length);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const index = y * width + x;
      gx[index] = pixels[index + 1] - pixels[index - 1];
      gy[index] = pixels[index + width] - pixels[index - width];
    }
  }
  return { gx, gy };
}

function measureReference(reference, sceneWidth) {
  const { pixels, width, height } = reference;
  const { gx, gy } = gradient(pixels, width, height);
  const offsets = new Int32Array(pixels.length);
  const mean = pixels.reduce((sum, value) => sum + value, 0) / pixels.length;
  const luminance = new Float32Array(pixels.length);
  const sparseIndices = [];
  const spacing = Math.max(1, Math.ceil(Math.sqrt(pixels.length / 28)));
  let luminanceEnergy = 0, edgeEnergy = 0;
  let brightest = 0, darkest = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      offsets[index] = y * sceneWidth + x;
      const centered = pixels[index] - mean;
      luminance[index] = centered;
      luminanceEnergy += centered * centered;
      edgeEnergy += gx[index] * gx[index] + gy[index] * gy[index];
      if (pixels[index] > pixels[brightest]) brightest = index;
      if (pixels[index] < pixels[darkest]) darkest = index;
      if (y % spacing === 0 && x % spacing === (Math.floor(y / spacing) % spacing)) {
        sparseIndices.push(index);
      }
    }
  }
  if (!sparseIndices.includes(brightest)) sparseIndices.push(brightest);
  if (!sparseIndices.includes(darkest)) sparseIndices.push(darkest);
  const sparseMean = sparseIndices.reduce((sum, index) => sum + pixels[index], 0) / sparseIndices.length;
  const sparse = {
    offsets: Int32Array.from(sparseIndices, index => offsets[index]),
    values: Float32Array.from(sparseIndices, index => pixels[index] - sparseMean),
  };
  sparse.energy = sparse.values.reduce((sum, value) => sum + value * value, 0);
  return { offsets, luminance, luminanceEnergy, gx, gy, edgeEnergy, sparse };
}

function normalizedLuminance(scene, start, offsets, weights, energy) {
  let sum = 0, sumSquares = 0, product = 0;
  for (let i = 0; i < offsets.length; i++) {
    const value = scene[start + offsets[i]];
    sum += value;
    sumSquares += value * value;
    product += value * weights[i];
  }
  const variance = sumSquares - sum * sum / offsets.length;
  return variance > EPSILON && energy > EPSILON ? product / Math.sqrt(energy * variance) : 0;
}

function denseScore(scene, imageGradient, start, measured) {
  const { offsets, gx, gy, edgeEnergy, luminance, luminanceEnergy } = measured;
  const correlation = normalizedLuminance(scene, start, offsets, luminance, luminanceEnergy);
  const { gx: sceneX, gy: sceneY } = imageGradient;
  let edgeProduct = 0, sceneEnergy = 0;
  for (let index = 0; index < offsets.length; index++) {
    if (gx[index] === 0 && gy[index] === 0) continue;
    const sample = start + offsets[index];
    const horizontal = sceneX[sample], vertical = sceneY[sample];
    edgeProduct += gx[index] * horizontal + gy[index] * vertical;
    sceneEnergy += horizontal * horizontal + vertical * vertical;
  }
  const edgeAgreement = sceneEnergy > EPSILON && edgeEnergy > EPSILON
    ? edgeProduct / Math.sqrt(edgeEnergy * sceneEnergy)
    : 0;
  return Math.max(0, Math.min(1, 0.7 * correlation + 0.3 * edgeAgreement));
}

function coarsePeaks(scene, sceneWidth, maxX, maxY, sparse) {
  const step = 2;
  const columns = Math.floor(maxX / step) + 1, rows = Math.floor(maxY / step) + 1;
  const scores = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < columns; col++) {
      scores[row * columns + col] = normalizedLuminance(scene, step * (row * sceneWidth + col),
        sparse.offsets, sparse.values, sparse.energy);
    }
  }

  const peaks = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < columns; col++) {
      const score = scores[row * columns + col];
      if (score <= 0 || (peaks.length === PEAKS_PER_TRANSFORM && score <= peaks.at(-1).score)) continue;
      let localMaximum = true;
      for (let nearbyRow = Math.max(0, row - 1); nearbyRow <= Math.min(rows - 1, row + 1); nearbyRow++) {
        for (let nearbyCol = Math.max(0, col - 1); nearbyCol <= Math.min(columns - 1, col + 1); nearbyCol++) {
          if (scores[nearbyRow * columns + nearbyCol] > score) localMaximum = false;
        }
      }
      if (!localMaximum) continue;
      peaks.push({ x: col * step, y: row * step, score });
      peaks.sort((a, b) => b.score - a.score);
      if (peaks.length > PEAKS_PER_TRANSFORM) peaks.pop();
    }
  }
  return peaks;
}

function overlap(first, second) {
  const [ax, ay, aw, ah] = first, [bx, by, bw, bh] = second;
  const width = Math.max(0, Math.min(ax + aw, bx + bw) - Math.max(ax, bx));
  const height = Math.max(0, Math.min(ay + ah, by + bh) - Math.max(ay, by));
  return (width * height) / (aw * ah + bw * bh - width * height);
}

function createMatcher(width, height, exampleSize, referenceCount, options = {}) {
  const { angles = DEFAULT_ANGLES, scales = DEFAULT_SCALES, limit = 5 } = options;
  if (![width, height, exampleSize, referenceCount, limit].every(Number.isInteger) ||
      Math.min(width, height, exampleSize, referenceCount, limit) < 1) {
    throw new RangeError('Positive integer raster dimensions, reference count and result limit are required.');
  }
  if (!angles.length || !angles.every(Number.isFinite) || !scales.length ||
      !scales.every(scale => Number.isFinite(scale) && scale > 0)) {
    throw new RangeError('Finite angles and positive scales are required.');
  }

  return function match(references, scene) {
    if (references.length !== referenceCount || scene.length !== width * height ||
        references.some(reference => reference.length !== exampleSize ** 2)) {
      throw new RangeError('Received rasters must have the configured dimensions.');
    }
    const imageGradient = gradient(scene, width, height);
    const candidates = [];
    for (const [index, reference] of references.entries()) {
      const maximum = reference.reduce((top, value) => Math.max(top, value), 0);
      if (maximum <= EPSILON) continue;
      for (const scale of scales) {
        for (const angle of angles) {
          const template = warpReference(reference, exampleSize, angle, scale, maximum * 0.2);
          if (!template || template.width > width || template.height > height) continue;
          const measured = measureReference(template, width);
          const maxX = width - template.width, maxY = height - template.height;
          for (const peak of coarsePeaks(scene, width, maxX, maxY, measured.sparse)) {
            let best = null;
            for (let y = Math.max(0, peak.y - 1); y <= Math.min(maxY, peak.y + 1); y++) {
              for (let x = Math.max(0, peak.x - 1); x <= Math.min(maxX, peak.x + 1); x++) {
                const score = denseScore(scene, imageGradient, y * width + x, measured);
                if (score > (best?.score ?? 0)) best = { x, y, score };
              }
            }
            if (best) {
              const { padding } = template;
              candidates.push({ box: [best.x + padding, best.y + padding,
                template.width - 2 * padding, template.height - 2 * padding],
              score: best.score, reference: index, angle });
            }
          }
        }
      }
    }
    candidates.sort((a, b) => b.score - a.score);
    const accepted = [];
    for (const candidate of candidates) {
      if (accepted.some(existing => overlap(existing.box, candidate.box) >= 0.35)) continue;
      accepted.push(candidate);
      if (accepted.length === limit) break;
    }
    return accepted;
  };
}

module.exports = { createMatcher };
