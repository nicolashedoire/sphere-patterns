'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createCircuit, iou } = require('./sphere-pose-consensus.cjs');

const PATCH_SIZE = 48;
const CANDIDATES = 5;
const EVIDENCE_WEIGHT = 0.16;
const RESOLUTION = 10000;
const ARMS = ['coarse', 'fixed', 'targeted', 'wave', 'cut'];
const LABELS = {
  coarse: 'Neural proposals', fixed: 'Fixed crops', targeted: 'Targeted crops',
  wave: 'Targeted crops via waves', cut: 'Cut request route',
};
const SPLITS = {
  'control-dev': 'Synthetic calibration', 'control-test': 'Synthetic initial evaluation',
  'control-validation': 'Synthetic validation', development: 'T-LESS calibration', heldout: 'T-LESS replay evaluation',
  confirmation: 'T-LESS extra replay views',
};
const PROTOCOL = 'Fixed and targeted policies expose the same number of native pixels to the verifier. T-LESS uses the existing calibration and replay panels.';

const read = path => JSON.parse(fs.readFileSync(path, 'utf8'));
const write = (path, value, indentation) => fs.writeFileSync(path, JSON.stringify(value, null, indentation) + '\n');
const center = ([x, y, w, h]) => [x + w / 2, y + h / 2];
const rounded = value => Math.round(value * RESOLUTION) / RESOLUTION;

function pathsFor(circuit) {
  const measurement = circuit.recurrent[0];
  return {
    measurement, request: measurement.slice().reverse().map(port => circuit.arcs[port].reverse),
    readout: circuit.paths[3], cut: circuit.arcs[circuit.cut].reverse,
  };
}

function transmit(circuit, path, word, tick, role, cut) {
  // One integer sample per tick. Dedicated links preserve amplitude and add
  // delay. The open endpoint only decodes a complete received word.
  const hops = [];
  for (const port of path) {
    if (port === cut) return { word: null, ready: tick, hops };
    const { from, to, delay } = circuit.arcs[port];
    hops.push({ from, to, start: tick, end: tick + delay, role });
    tick += delay;
  }
  return { word: Array.from(word), ready: tick + word.length, hops };
}

function sendRequest(circuit, paths, query, severed = false) {
  const word = [query.slot, ...query.box];
  const signal = transmit(circuit, paths.request, word, 0, 'request', severed ? paths.cut : null);
  if (!signal.word) return { query: null, signal };
  const [slot, ...box] = signal.word;
  assert.equal(slot, query.slot, 'Decode the request on its addressed comparison channel');
  return { query: { ...query, box }, signal };
}

function candidateCenter(candidate) {
  return Number.isFinite(candidate.x) && Number.isFinite(candidate.y) ? [candidate.x, candidate.y] : center(candidate.box);
}

function transform(point, pivot, anchor, scale, radians) {
  const dx = (point[0] - pivot[0]) * scale;
  const dy = (point[1] - pivot[1]) * scale;
  const cosine = Math.cos(radians), sine = Math.sin(radians);
  return [anchor[0] + dx * cosine - dy * sine, anchor[1] + dx * sine + dy * cosine];
}

function candidatesWithCues(candidates, references) {
  return candidates.map((candidate, index) => {
    const ref = references.get(candidate.key);
    assert.ok(ref?.box && ref.cues?.length, `Missing reference cues for ${candidate.key}`);
    const angle = candidate.angle ?? 0;
    const radians = angle * Math.PI / 180;
    const pivot = center(ref.box);
    const anchor = candidateCenter(candidate);
    const scale = candidate.scale ?? candidate.box[2] / ref.box[2];
    assert.ok(scale > 0);
    const cues = ref.cues.map(cue => ({ ...cue, at: candidate.cues?.find(item => item.id === cue.id)?.xy ??
      transform(cue.xy, pivot, anchor, scale, radians) }));
    return { ...candidate, id: candidate.id ?? `c${index + 1}`, score: rounded(candidate.score), angle, radians, pivot, anchor, scale, cues };
  });
}

function chooseCue(candidate, contenders, frame) {
  let best = null;
  let bestValue = -Infinity;
  for (const cue of candidate.cues) {
    const [x, y] = cue.at;
    if (x < 0 || x >= frame.width || y < 0 || y >= frame.height) continue;
    // A preferred point is both repeatable across the training pair and
    // discriminative in the current scene's competing coordinate systems.
    const distances = contenders.filter(other => other.id !== candidate.id).map(other => {
      const comparable = other.cues.find(item => item.id === cue.id)?.at ?? other.anchor;
      return Math.min(1, Math.hypot(x - comparable[0], y - comparable[1]) / (PATCH_SIZE * 2));
    });
    const separation = distances.length ? Math.min(...distances) : 1;
    const displacement = Math.min(1, Math.hypot(x - candidate.anchor[0], y - candidate.anchor[1]) / PATCH_SIZE);
    const quality = Number.isFinite(cue.quality) ? cue.quality : 1;
    const utility = quality * (0.45 + 0.40 * separation + 0.15 * displacement);
    if (utility > bestValue) {
      bestValue = utility;
      best = cue;
    }
  }
  return best ?? { id: 'center', at: candidate.anchor };
}

function request(candidate, cue, frame, slot) {
  const { width, height } = frame;
  assert.ok(width >= PATCH_SIZE && height >= PATCH_SIZE);
  const x = Math.max(0, Math.min(width - PATCH_SIZE, Math.round(cue.at[0] - PATCH_SIZE / 2)));
  const y = Math.max(0, Math.min(height - PATCH_SIZE, Math.round(cue.at[1] - PATCH_SIZE / 2)));
  const box = [x, y, PATCH_SIZE, PATCH_SIZE];
  const [cx, cy] = transform(center(box), candidate.anchor, candidate.pivot, 1 / candidate.scale, -candidate.radians);
  const side = PATCH_SIZE / candidate.scale;
  return { id: candidate.id, candidateId: candidate.id, slot, cue: cue.id, box,
    refBox: [cx - side / 2, cy - side / 2, side, side].map(rounded), angle: candidate.angle };
}

function planFrame(frame, references, circuit, paths) {
  const selected = candidatesWithCues(frame.candidates.slice(0, CANDIDATES), references);
  const fixed = selected.map((candidate, slot) => request(candidate, { id: 'center', at: candidate.anchor }, frame, slot));
  const targeted = selected.map((candidate, slot) => request(candidate, chooseCue(candidate, selected, frame), frame, slot));
  const wave = targeted.map(query => sendRequest(circuit, paths, query).query);
  const cut = targeted.map(query => sendRequest(circuit, paths, query, true).query).filter(Boolean);
  assert.deepEqual(wave, targeted, 'The fine reader must receive identical coordinates');
  assert.equal(cut.length, 0, 'The cut decoder must never issue a crop read');
  const candidates = selected.map(({ radians, pivot, anchor, cues, ...row }) => row);
  return { ...frame, candidates, plans: { fixed, targeted, wave, cut } };
}

function updateScore(score, delta) {
  return rounded(score + EVIDENCE_WEIGHT * delta);
}

function rerank(candidates, measurements) {
  const byId = new Map(measurements.map(row => [row.id, row]));
  return candidates.map(candidate => ({
    ...candidate, baseScore: candidate.score,
    score: byId.get(candidate.id)?.score ?? updateScore(candidate.score, byId.get(candidate.id)?.delta ?? 0),
  })).sort((left, right) => right.score - left.score || left.id.localeCompare(right.id));
}

function routeEvidence(circuit, paths, query, response, baseScore) {
  const requestSignal = sendRequest(circuit, paths, query).signal;
  const encoded = Math.round(response.delta * RESOLUTION);
  const responseSignal = transmit(circuit, paths.measurement, [encoded], requestSignal.ready + 1, 'measurement');
  const value = responseSignal.word[0] / RESOLUTION;
  const updated = updateScore(baseScore, value);
  const probe = transmit(circuit, paths.readout, [Math.round(updated * RESOLUTION)], responseSignal.ready, 'readout');
  return { value, updated: probe.word[0] / RESOLUTION, duration: probe.ready,
    hops: [...requestSignal.hops, ...responseSignal.hops, ...probe.hops] };
}

function measureFrame(frame, measurements, circuit, paths) {
  const decoded = measurements.responses;
  assert.deepEqual(frame.plans.targeted, frame.plans.wave);
  assert.deepEqual(decoded.targeted, decoded.wave, `${frame.id} wave and direct crop measurements differ`);
  assert.deepEqual(decoded.cut, [], `${frame.id} cut request unexpectedly read pixels`);
  const outcomes = Object.fromEntries(['fixed', 'targeted', 'wave'].map(arm => {
    const requests = frame.plans[arm];
    assert.equal(requests.length, decoded[arm].length);
    const responses = new Map(decoded[arm].map(response => [response.id, response]));
    const crops = requests.map(query => {
      const response = responses.get(query.id);
      assert.ok(response && [response.match, response.visibility, response.delta].every(Number.isFinite));
      const baseScore = frame.candidates[query.slot].score;
      const wave = arm === 'wave' ? routeEvidence(circuit, paths, query, response, baseScore) : null;
      return { ...query, match: rounded(response.match), visibility: rounded(response.visibility),
        signal: wave ? wave.value : rounded(response.delta), probeScore: wave?.updated, probeTick: wave?.duration };
    });
    const candidates = rerank(frame.candidates, crops.map(crop => ({ id: crop.candidateId, delta: crop.signal, score: crop.probeScore })));
    return [arm, { pixels: crops.reduce((sum, crop) => sum + crop.box[2] * crop.box[3], 0), crops, candidates }];
  }));
  assert.deepEqual(outcomes.targeted.candidates, outcomes.wave.candidates);
  assert.equal(outcomes.targeted.pixels, outcomes.fixed.pixels);
  const coarse = rerank(frame.candidates, []);
  const cold = { pixels: 0, crops: [], candidates: coarse };
  const { image, plans, candidates, truth, ...metadata } = frame;
  return { ...metadata, targets: truth ?? [], coarse, arms: { coarse: cold, ...outcomes, cut: cold },
    waveCheck: { sameRequests: true, sameScores: true, cutReads: plans.cut.length } };
}

function evaluation(frames, arm, cutoff, maxCandidates = CANDIDATES) {
  const values = { frames: frames.length, targetCount: 0, matches: 0, falseAlarms: 0, absentAlarms: 0, meanPixels: 0 };
  for (const frame of frames) {
    const remaining = frame.targets.filter(target => (target.visibility ?? 1) >= .20);
    const stream = frame.arms[arm];
    values.targetCount += remaining.length;
    values.meanPixels += stream.pixels;
    let falseInFrame = 0;
    for (const candidate of stream.candidates.slice(0, maxCandidates)) {
      if (candidate.score < cutoff) continue;
      let bestIndex = -1;
      let bestOverlap = .50;
      remaining.forEach((target, index) => {
        const overlap = iou(candidate.box, target.box);
        if (overlap >= bestOverlap && (target.part == null || candidate.part === target.part)) {
          bestIndex = index;
          bestOverlap = overlap;
        }
      });
      if (bestIndex < 0) falseInFrame++;
      else {
        values.matches++;
        remaining.splice(bestIndex, 1);
      }
    }
    values.falseAlarms += falseInFrame;
    values.absentAlarms += Number(frame.targets.length === 0 && falseInFrame > 0);
  }
  values.meanPixels = frames.length ? values.meanPixels / frames.length : 0;
  return values;
}

function cutoffFor(frames, arm, control) {
  const scores = [...new Set(frames.flatMap(frame => frame.arms[arm].candidates.map(row => row.score)))].sort((a, b) => b - a);
  let bestIndex = -1;
  let bestHits = 0;
  let bestFalse = 0;
  scores.forEach((score, index) => {
    const value = evaluation(frames, arm, score);
    if (value.falseAlarms > (control ? 0 : 2) || value.absentAlarms > (control ? 0 : 1)) return;
    if (value.matches > bestHits || (value.matches === bestHits && value.falseAlarms < bestFalse)) {
      bestIndex = index;
      bestHits = value.matches;
      bestFalse = value.falseAlarms;
    }
  });
  if (bestIndex < 0) return Infinity;
  const value = scores[bestIndex];
  return (value + (scores[bestIndex + 1] ?? value)) / 2;
}

function summarize(frames, baseline, gradient) {
  const panels = Object.entries(SPLITS).map(([split, label]) => ({ split, label, frames: frames.filter(frame => frame.split === split) }));
  const gradientFrames = gradient?.frames.map(({ id, split, truth, candidates }) => ({ id, split, targets: truth,
    arms: { coarse: { pixels: 0, candidates: rerank(candidates.slice(0, CANDIDATES).map((row, index) => ({ ...row, id: `c${index + 1}` })), []) } },
  }));
  const cutoffs = Object.fromEntries(['control', 'real'].map(domain => {
    const control = domain === 'control';
    const development = frames.filter(frame => frame.split === (control ? 'control-dev' : 'development'));
    const values = Object.fromEntries(ARMS.map(arm => [arm, cutoffFor(development, arm, control)]));
    assert.equal(values.targeted, values.wave);
    assert.equal(values.coarse, values.cut);
    return [domain, values];
  }));
  for (const frame of frames) for (const arm of ARMS) {
    const cutoff = cutoffs[frame.split.startsWith('control') ? 'control' : 'real'][arm];
    const stream = frame.arms[arm];
    stream.detectedIds = stream.candidates.filter(row => row.score >= cutoff).map(row => row.id);
  }
  return panels.filter(panel => panel.frames.length).map(({ split, label, frames: rows }) => {
    const cutoffsForSplit = cutoffs[split.startsWith('control') ? 'control' : 'real'];
    const summaries = ARMS.map(arm => {
      const raw = evaluation(rows, arm, -Infinity);
      return { arm: LABELS[arm], ...evaluation(rows, arm, cutoffsForSplit[arm]),
        cutoff: Number.isFinite(cutoffsForSplit[arm]) ? cutoffsForSplit[arm] : null,
        topOneHits: evaluation(rows, arm, -Infinity, 1).matches, topFiveHits: raw.matches, topFiveFalse: raw.falseAlarms };
    });
    if (baseline?.summary.panels[split] && baseline.summary.panels[split].frames === rows.length) {
      const previous = baseline.summary.panels[split];
      summaries.unshift({ arm: 'Previous RootSIFT proposals', frames: previous.frames, targetCount: previous.targets,
        matches: previous.direct.hits, falseAlarms: previous.direct.falseAlarms, absentAlarms: previous.direct.absentAlarms,
        meanPixels: 0, topFiveHits: previous.direct.topFiveHits, topFiveFalse: previous.direct.topFiveFalse });
    }
    if (gradientFrames) {
      const gradientPanel = gradientFrames.filter(frame => frame.split === split);
      assert.deepEqual(gradientPanel.map(row => row.id), rows.map(row => row.id));
      const control = split.startsWith('control');
      const calibration = gradientFrames.filter(frame => frame.split === (control ? 'control-dev' : 'development'));
      const cutoff = cutoffFor(calibration, 'coarse', control);
      const raw = evaluation(gradientPanel, 'coarse', -Infinity);
      summaries.splice(1, 0, { arm: 'Dense gradients, same search', ...evaluation(gradientPanel, 'coarse', cutoff),
        cutoff: Number.isFinite(cutoff) ? cutoff : null,
        topOneHits: evaluation(gradientPanel, 'coarse', -Infinity, 1).matches, topFiveHits: raw.matches, topFiveFalse: raw.falseAlarms });
    }
    return { split, label, rows: summaries };
  });
}

function controls(circuit, paths) {
  const refs = new Map([['demo', { box: [0, 0, 100, 100], cues: [
    { id: 'left', xy: [20, 30], quality: .9 }, { id: 'center', xy: [50, 50], quality: .5 },
  ] }]]);
  const frame = { id: 'control', width: 300, height: 240, candidates: [
    { id: 'c1', key: 'demo', box: [0, 0, 100, 100], score: .6, angle: 0, scale: 1 },
    { id: 'c2', key: 'demo', box: [20, 5, 100, 100], score: .58, angle: 0, scale: 1 },
  ] };
  const planned = planFrame(frame, refs, circuit, paths);
  assert.deepEqual(planned.plans, planFrame({ ...frame, truth: [{ box: [1, 2, 3, 4] }] }, refs, circuit, paths).plans);
  const query = planned.plans.wave[0];
  assert.deepEqual(query.box, [0, 6, PATCH_SIZE, PATCH_SIZE]);
  assert.deepEqual(query.refBox, [0, 6, PATCH_SIZE, PATCH_SIZE]);
  assert.notDeepEqual(query.box, planned.plans.fixed[0].box);
  assert.ok(paths.request.includes(paths.cut));
  const negative = routeEvidence(circuit, paths, query, { delta: -.75 }, .6);
  assert.equal(negative.value, -.75);
  assert.equal(negative.updated, .48);
  assert.ok(negative.duration > circuit.view.duration);
  assert.equal(rerank(planned.candidates, [{ id: 'c1', delta: negative.value }])[0].id, 'c2');
}

function main() {
  const [mode, input, second, reportPath, pagePath, baselinePath, gradientPath] = process.argv.slice(2);
  if (!input || !second || !['plan', 'score'].includes(mode)) {
    throw new Error('Usage: sphere-neural-feedback.cjs plan CANDIDATES PLAN | score PLAN READS REPORT PAGE [PREVIOUS_REPORT [GRADIENT_PROPOSALS]]');
  }
  const circuit = createCircuit();
  const paths = pathsFor(circuit);
  controls(circuit, paths);
  const data = read(input);
  if (mode === 'plan') {
    const references = new Map(data.references.map(ref => [ref.key, ref]));
    const metadata = { encoder: data.model, revision: data.revision, safetensorsSha256: data.safetensorsSha256,
      device: data.device, descriptorStride: data.descriptorStride, angles: data.angles, scales: data.scales,
      dataset: 'Procedural mechanical plates and T-LESS object 1', protocol: PROTOCOL };
    write(second, { metadata, references: data.references, frames: data.frames.map(frame => planFrame(frame, references, circuit, paths)) });
    return;
  }
  const responses = read(second);
  const indexed = new Map(responses.frames.map(frame => [frame.id, frame]));
  assert.equal(indexed.size, data.frames.length);
  const frames = data.frames.map(frame => measureFrame(frame, indexed.get(frame.id), circuit, paths));
  const gradient = gradientPath ? read(gradientPath) : null;
  const summaries = summarize(frames, baselinePath ? read(baselinePath) : null, gradient);
  const { verifier, modelPatchSize, neutralCosine, visibility, delta } = responses.metadata;
  const metadata = { ...data.metadata, localRead: { verifier, modelPatchSize, neutralCosine, visibility, delta }, patchSize: PATCH_SIZE,
    gradientControl: gradient?.model, candidates: CANDIDATES, wordParity: true, framesMatched: frames.length,
    cutReads: frames.reduce((total, frame) => total + frame.waveCheck.cutReads, 0), evidenceWeight: EVIDENCE_WEIGHT,
    differentCropFrames: frames.filter(frame => frame.arms.targeted.crops.some((crop, index) =>
      crop.box.some((value, axis) => value !== frame.arms.fixed.crops[index].box[axis]))).length };
  write(reportPath, { metadata, summaries, frames }, 2);
  const topology = { nodes: circuit.view.nodes, edges: circuit.view.edges };
  const shown = frames.filter(frame => frame.preview).map(frame => {
    const crop = frame.arms.wave.crops.find(row => row.candidateId === frame.arms.wave.candidates[0]?.id);
    const measured = crop ? routeEvidence(circuit, paths, crop, { delta: crop.signal }, frame.coarse.find(row => row.id === crop.candidateId).score) : null;
    const trace = measured ? { hops: measured.hops, duration: measured.duration } : null;
    const stopped = crop ? sendRequest(circuit, paths, crop, true).signal : null;
    const cutTrace = stopped ? { hops: stopped.hops, duration: stopped.ready } : null;
    return { ...frame, image: frame.preview, trace, arms: { ...frame.arms, cut: { ...frame.arms.cut, trace: cutTrace } } };
  });
  write(pagePath, { metadata, summaries, topology, frames: shown });
  console.log(JSON.stringify({ metadata, summaries }, null, 2));
}

if (require.main === module) main();
