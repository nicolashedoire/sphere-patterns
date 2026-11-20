'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createSetup, beginRecording, continueRecording } = require('../sphere-roundtrip-run.cjs');

function readout(setup, streams) {
  const arrival = setup.station.probes.map(probe => setup.firstMixerTick + probe.delay);
  const ticks = Math.max(...arrival) + streams[0].length - 1;
  assert.ok(ticks < setup.station.firstReturn, 'Record the direct arrivals before a return can interfere.');
  const recording = beginRecording(setup, ticks);
  const stats = continueRecording(setup, recording, streams, setup.firstMixerTick, ticks);
  return {
    recovered: recording.traces[0].slice(arrival[0] - 1, arrival[0] + streams[0].length - 1),
    firstProbeTick: arrival[0],
    error: stats.relativeError,
  };
}

function test(data) {
  const settings = data.settings;
  const { plans } = data;
  const streams = [[], []];
  for (const row of plans) {
    const first = row.hypotheses[0].initialFit - row.hypotheses[1].initialFit;
    const second = row.verdicts.active_probe.margin;
    streams[0].push(first, first);
    streams[1].push(0, (second ?? 0) * settings.second_weight);
  }

  const setup = createSetup();
  const withObservation = readout(setup, streams);
  const withheld = readout(setup, [streams[0], []]);
  let maxMarginError = 0;
  let recordedChanges = 0;
  for (const [index, row] of plans.entries()) {
    const first = streams[0][2 * index];
    const updated = withObservation.recovered[2 * index + 1] * Math.SQRT2;
    const initial = withObservation.recovered[2 * index] * Math.SQRT2;
    const control = withheld.recovered[2 * index + 1] * Math.SQRT2;
    const expected = row.verdicts.active_probe.decisionMargin;
    maxMarginError = Math.max(maxMarginError, Math.abs(initial - first), Math.abs(control - first),
      Math.abs(updated - expected));
    const initialPart = row.verdicts.one_view.part;
    const threshold = row.verdicts.active_probe.margin === null ? settings.first_threshold : settings.final_threshold;
    // The decision boundary is unchanged at machine precision.
    const wavePart = Math.abs(updated) + 1e-14 < threshold || Math.abs(updated) < 1e-14
      ? null : row.hypotheses[updated > 0 ? 0 : 1].part;
    assert.equal(wavePart, row.verdicts.active_probe.part);
    if (wavePart !== initialPart) recordedChanges++;
  }
  assert.ok(maxMarginError < 1e-13);
  return {
    cases: plans.length,
    recordedChanges,
    suppressedSecondStream: plans.length,
    firstProbeTick: withObservation.firstProbeTick,
    maxMarginError,
    maxRelativeEnergyError: Math.max(withObservation.error, withheld.error),
  };
}

const [predictions, output] = process.argv.slice(2);
const report = test(JSON.parse(fs.readFileSync(predictions, 'utf8')));
const json = JSON.stringify(report, null, 2) + '\n';
if (output) fs.writeFileSync(output, json);
process.stdout.write(json);
