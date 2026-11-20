'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const { AMPLITUDE, GROUPS, createCircuit, encodeSpace, votesForGroup, directVote, waveVote, rank, iou } = require('./sphere-pose-consensus.cjs');

const MODES = ['direct', 'sphere', 'feedForward', 'cut'];
const SPLITS = ['control-dev', 'control-test', 'control-validation', 'development', 'heldout'];

function eligibleTargets(frame) {
  return frame.truth.filter(item => (item.visibility ?? 1) >= 0.20);
}

function checkCircuit(circuit) {
  const input = [new Map([[0, 7], [1, 9]]), new Map([[0, 5], [2, 12]]), new Map([[0, 3]])];
  const direct = directVote(input);
  assert.deepEqual(waveVote(input, circuit, 'sphere'), direct);
  assert.deepEqual(waveVote(input, circuit, 'feedForward'), direct);
  const cut = waveVote(input, circuit, 'cut');
  assert.deepEqual(cut.get(0), { amplitude: 8, votes: [0, 5, 3] });
  assert.ok(!cut.has(1) && cut.get(2).amplitude === 12);
  assert.deepEqual(waveVote(input, circuit, 'sphere', 1), cut);
  assert.ok(circuit.view.routes[0].hops.some(({ from, to }) => from === circuit.view.cut[0] && to === circuit.view.cut[1]));
  assert.ok(!circuit.view.feedRoutes[0].hops.some(({ from, to }) => from === circuit.view.cut[0] && to === circuit.view.cut[1]));
}

function runFrame(frame, references, circuit) {
  const space = encodeSpace(frame, references);
  const votes = Array.from({ length: GROUPS }, (_, group) => votesForGroup(references, frame.points, space, group));
  const vectors = { direct: directVote(votes), sphere: waveVote(votes, circuit, 'sphere'),
    feedForward: waveVote(votes, circuit, 'feedForward'), cut: waveVote(votes, circuit, 'cut') };
  // Maps are compared before ranking, suppression, thresholding or evaluation.
  assert.deepEqual(vectors.sphere, vectors.direct, frame.id);
  assert.deepEqual(vectors.feedForward, vectors.direct, frame.id);
  const delayed = waveVote(votes, circuit, 'sphere', 1);
  assert.deepEqual(delayed, vectors.cut, frame.id);
  const baseRank = rank(vectors.sphere, space);
  const best = baseRank[0] ?? null;
  const results = Object.fromEntries(MODES.map(mode => {
    const sample = best ? vectors[mode].get(best.slot) : null;
    return [mode, {
      raw: mode === 'cut' ? rank(vectors.cut, space) : baseRank,
      atSphereBest: (sample?.votes ?? [0, 0, 0]).map(value => value / AMPLITUDE),
      trackedScore: sample ? sample.amplitude / (GROUPS * AMPLITUDE) : null,
    }];
  }));
  const { points, ...metadata } = frame;
  const noFeatures = eligibleTargets(frame).filter(({ box: [x, y, w, h] }) =>
    !points.some(point => point.x >= x && point.x < x + w && point.y >= y && point.y < y + h)).length;
  return { ...metadata, groups: references[0].groups, ...results, signal: {
    nFeatures: points.length, slots: vectors.sphere.size,
    targetsWithoutFeatures: noFeatures,
    changedByCut: votes[0].size > 0,
    changedByDelay: votes[0].size > 0,
  } };
}

function match(frame, mode, threshold) {
  const remaining = eligibleTargets(frame);
  const visibleTargets = remaining.length;
  let hits = 0, falseAlarms = 0;
  for (const detection of frame[mode].raw) {
    if (detection.score < threshold) continue;
    let greatestOverlap = 0.50;
    let matchIndex = -1;
    remaining.forEach((target, index) => {
      if (detection.part !== target.part) return;
      const overlap = iou(target.box, detection.box);
      if (overlap < greatestOverlap) return;
      greatestOverlap = overlap;
      matchIndex = index;
    });
    if (matchIndex < 0) falseAlarms++;
    else {
      hits++;
      remaining.splice(matchIndex, 1);
    }
  }
  return { targets: visibleTargets, hits, falseAlarms, absentAlarms: Number(frame.truth.length === 0 && falseAlarms > 0) };
}

function totals(frames, mode, threshold) {
  const result = { targets: 0, hits: 0, falseAlarms: 0, absentAlarms: 0 };
  for (const frame of frames) {
    const counts = match(frame, mode, threshold);
    for (const key of Object.keys(result)) result[key] += counts[key];
  }
  return result;
}

function calibrate(frames, domain) {
  const candidates = new Set([Infinity]);
  for (const frame of frames) for (const { score } of frame.direct.raw) candidates.add(score);
  const [limitFalse, limitAbsent] = domain === 'control' ? [0, 0] : [2, 1];
  let threshold = Infinity;
  let best = { hits: 0, falseAlarms: 0 };
  for (const score of candidates) {
    const result = totals(frames, 'direct', score);
    if (result.falseAlarms > limitFalse || result.absentAlarms > limitAbsent) continue;
    if (result.hits > best.hits || (result.hits === best.hits && result.falseAlarms < best.falseAlarms)) {
      threshold = score;
      best = result;
    } else if (result.hits === best.hits && result.falseAlarms === best.falseAlarms && score > threshold) {
      threshold = score;
    }
  }
  // All cutoffs between adjacent observed scores give the same development
  // verdict. Use the middle of that interval, not a positive's exact score.
  if (!Number.isFinite(threshold)) return threshold;
  const next = Math.max(0, ...[...candidates].filter(score => score < threshold));
  return (threshold + next) / 2;
}

function summarizePanel(frames, threshold) {
  const { targets } = totals(frames, 'sphere', threshold);
  return { frames: frames.length, targets, threshold: Number.isFinite(threshold) ? threshold : null,
    targetsWithoutFeatures: frames.reduce((sum, row) => sum + row.signal.targetsWithoutFeatures, 0),
    ...Object.fromEntries(MODES.map(mode => {
      const { hits, falseAlarms, absentAlarms } = totals(frames, mode, threshold);
      const raw = totals(frames, mode, 0);
      return [mode, { hits, falseAlarms, absentAlarms, topFiveHits: raw.hits, topFiveFalse: raw.falseAlarms }];
    })),
  };
}

function main() {
  const [inputs, output, pageData] = process.argv.slice(2);
  if (!inputs || !output) throw new Error('Usage: sphere-pose-benchmark.cjs INPUTS.json MEASUREMENTS.json [LAB.json]');
  const { references, frames, metadata } = JSON.parse(fs.readFileSync(inputs, 'utf8'));
  const circuit = createCircuit();
  checkCircuit(circuit);
  const referencesByDomain = Object.fromEntries(['control', 'real'].map(domain => [domain, references.filter(reference => reference.domain === domain)]));
  assert.ok(referencesByDomain.control.length && referencesByDomain.real.length);
  const rows = frames.map(frame => runFrame(frame, referencesByDomain[frame.domain], circuit));
  const bySplit = Object.fromEntries(SPLITS.map(split => [split, rows.filter(row => row.split === split)]));
  const thresholds = { control: calibrate(bySplit['control-dev'], 'control'), real: calibrate(bySplit.development, 'real') };
  const summary = {
    mechanism: { frames: rows.length, sameDirect: rows.length, sameFeedForward: rows.length,
      changedByCut: rows.filter(row => row.signal.changedByCut).length,
      changedByDelay: rows.filter(row => row.signal.changedByDelay).length },
    panels: Object.fromEntries(SPLITS.map(split => [split, summarizePanel(bySplit[split], thresholds[split.startsWith('control') ? 'control' : 'real'])])),
  };
  for (const row of rows) for (const mode of MODES) {
    const { raw } = row[mode];
    row[mode].best = raw[0] ?? null;
    row[mode].detections = raw.filter(candidate => candidate.score >= thresholds[row.domain]);
  }
  const report = { version: 1, metadata, summary, frames: rows, circuit: circuit.view };
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
  if (pageData) {
    const framesWithPreviews = rows.filter(row => row.preview).map(row => ({ ...row, ...Object.fromEntries(MODES.map(mode => {
      const { raw, ...results } = row[mode];
      return [mode, results];
    })) }));
    fs.writeFileSync(pageData, JSON.stringify({ ...report, frames: framesWithPreviews }) + '\n');
  }
  console.log(JSON.stringify(summary, null, 2));
}

if (require.main === module) main();
