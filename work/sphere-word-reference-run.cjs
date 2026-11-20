'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createNetwork } = require('./sphere-kernel.js');
const { FileSimulation } = require('./sphere-roundtrip-run.cjs');
const { AlignmentCircuit, START, C_LEVEL, setup } = require('./sphere-pattern-internal-delay-run.cjs');

const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const slots = 12;
const span = slots * letters.length;
const aDelay = (slots - 1) * letters.length;
const tapDelays = Array.from({ length: 2 * slots - 1 }, (_, index) => index * letters.length);
const timing = { span, aDelay, tapDelays };
const amplitude = 0.5;
const matchLevel = amplitude * Math.sin(Math.PI * C_LEVEL * amplitude);
const examples = ['SO LATE', 'LATE NIGHT'];
const emissionOffset = new AlignmentCircuit(false, START, timing).emitAt;
assert.equal(emissionOffset, span + aDelay);

const referenceEmission = START + emissionOffset;
const referenceReturn = referenceEmission + setup.loopDelays[1];
const matchEmission = referenceReturn + emissionOffset;
const probeDelay = setup.loopDelays[1] - setup.station.probes[1].delay;
const referenceAtProbe = referenceEmission + probeDelay;
const matchAtProbe = matchEmission + probeDelay;

function encode(text) {
  assert.ok(text.length <= slots && /^[A-Z ]*$/.test(text));
  const wave = new Float64Array(span);
  for (const [index, letter] of [...text].entries()) {
    if (letter !== ' ') wave[index * letters.length + letters.indexOf(letter)] = amplitude;
  }
  return wave;
}

function decode(wave, level) {
  const decoded = Array(slots).fill(' ');
  for (const [tick, sample] of wave.entries()) {
    if (Math.abs(sample) < 1e-13) continue;
    assert.ok(Math.abs(sample - level) < 1e-13);
    const slot = Math.floor(tick / letters.length);
    assert.equal(decoded[slot], ' ');
    decoded[slot] = letters[tick % letters.length];
  }
  return decoded.join('');
}

function shiftInLetters(tap) {
  return (tapDelays[tap] - aDelay) / letters.length;
}

class WordSimulation extends FileSimulation {
  constructor(readReturningReference) {
    super(createNetwork({ seed: setup.seed }), { writing: false });
    this.learn = new AlignmentCircuit(false, START, timing);
    this.match = new AlignmentCircuit(false, referenceReturn, timing);
    this.learning = null;
    this.readReturningReference = readReturningReference;
  }

  rotate(values, first, second) {
    if (first !== 0 || second !== 1) return;
    if (this.nodeIndex !== this.splitter) return super.rotate(values, first, second);

    const [forward0, backward0, forward1, backward1] = this.lanePorts[this.nodeIndex++];
    const a = values[backward0], b = values[backward1];
    const carrier = values[forward0], returning = values[forward1];
    const tick = this.tick + 1;
    if (tick === referenceEmission + span) {
      const { selectedTap, selectionTick, accumulated } = this.learn;
      assert.notEqual(selectedTap, null);
      this.learning = { shift: shiftInLetters(selectedTap), selectionTick,
        letters: Math.round(accumulated[selectedTap] / C_LEVEL ** 2) };
      this.learn = null;
    }
    const angle = this.learn ? this.learn.receive(tick, a, b) :
      this.match.receive(tick, this.readReturningReference ? returning : 0, b);
    values[forward0] = (a + b) * Math.SQRT1_2;
    values[forward1] = (a - b) * Math.SQRT1_2;
    values[backward0] = carrier * Math.cos(angle) - returning * Math.sin(angle);
    values[backward1] = carrier * Math.sin(angle) + returning * Math.cos(angle);
  }
}

function run(scene, readReturningReference = true) {
  const sim = new WordSimulation(readReturningReference);
  const { arcs, nodePorts } = sim.network;
  const [mixerForward] = sim.lanePorts[setup.mixer];
  const [, , probeForward] = sim.lanePorts[setup.station.node];
  const carrierArc = arcs[nodePorts[setup.mixer][mixerForward]].reverse;
  const portC = nodePorts[setup.station.node][probeForward];
  const carrier = new Float64Array(span).fill(amplitude);
  const sends = [
    { arc: setup.sourceArcs[0], mixerTick: START, samples: encode(examples[0]) },
    { arc: setup.sourceArcs[1], mixerTick: START, samples: encode(examples[1]) },
    { arc: carrierArc, mixerTick: referenceEmission, samples: carrier },
    { arc: setup.sourceArcs[1], mixerTick: referenceReturn, samples: encode(scene) },
    { arc: carrierArc, mixerTick: matchEmission, samples: carrier }
  ].map(send => ({ ...send, launch: send.mixerTick - arcs[send.arc].delay }));
  assert.ok(sends.every(send => send.launch >= 1));

  const capture = (tick, port) => ({ tick, port, wave: new Float64Array(span) });
  const firstC = capture(referenceAtProbe, portC);
  const secondC = capture(matchAtProbe, portC);
  const firstUtterances = setup.station.probes.map(probe => capture(START + probe.delay, probe.receivingPort));
  const sceneUtterance = setup.station.probes.map(probe => capture(referenceReturn + probe.delay, probe.receivingPort));
  const probes = [firstC, secondC, ...firstUtterances, ...sceneUtterance];
  const stop = Math.max(...probes.map(probe => probe.tick)) + span - 1;
  assert.ok(stop < referenceReturn + setup.loopDelays[1]);

  for (let tick = 1; tick <= stop; tick++) {
    sim.step();
    for (const { arc, launch, samples } of sends) {
      const sample = samples[tick - launch];
      if (sample) sim.transmit(arc, sample);
    }
    for (const { tick: firstTick, port, wave } of probes) {
      const index = tick - firstTick;
      if (index >= 0 && index < span) wave[index] = sim.lastIncoming[port];
    }
  }

  assert.equal(sim.learn, null);
  assert.deepEqual(sim.match.b, encode(scene));
  if (readReturningReference) assert.deepEqual(sim.match.a, firstC.wave);
  else assert.ok(sim.match.a.every(sample => sample === 0));
  const [sumFirst, differenceFirst] = firstUtterances.map(probe => probe.wave);
  assert.equal(decode(sumFirst.map((sample, index) => (sample + differenceFirst[index]) * Math.SQRT1_2), amplitude).trimEnd(), examples[0]);
  assert.equal(decode(sumFirst.map((sample, index) => (sample - differenceFirst[index]) * Math.SQRT1_2), amplitude).trimEnd(), examples[1]);
  const [sum, difference] = sceneUtterance.map(probe => probe.wave);
  assert.equal(decode(sum.map((sample, index) => (sample - difference[index]) * Math.SQRT1_2), amplitude).trimEnd(), scene);

  const referenceSlots = decode(firstC.wave, C_LEVEL);
  const matchedSlots = decode(secondC.wave, matchLevel);
  const { selectedTap, selectionTick, accumulated } = sim.match;
  const scores = Array.from(accumulated, (energy, tap) => ({ shift: shiftInLetters(tap), letters: Math.round(energy / matchLevel ** 2) }));
  const selection = selectedTap === null ? null : { ...scores[selectedTap],
    scenePositions: [...matchedSlots].flatMap((letter, index) => letter === ' ' ? [] : [index - shiftInLetters(selectedTap)]) };
  const energyError = sim.getStats().relativeError;
  assert.ok(energyError < 1e-12);
  return { scene, reference: referenceSlots.trim(), referenceSlots, match: matchedSlots.trim(), selection, selectionTick,
    scores, learning: sim.learning, energyError, traces: { learnedAtProbe: Array.from(firstC.wave),
      returningAtComparator: Array.from(sim.match.a), matchedAtProbe: Array.from(secondC.wave),
      sceneSumAtProbe: Array.from(sum), sceneDifferenceAtProbe: Array.from(difference) } };
}

const main = run('LAKE OR LATE');
const decoyOnly = run('LAKE');
const disconnected = run('LAKE OR LATE', false);
assert.equal(main.reference, 'LATE');
assert.equal(main.match, 'LATE');
assert.deepEqual(main.selection, { shift: -5, letters: 4, scenePositions: [8, 9, 10, 11] });
assert.equal(main.learning.shift, 3);
assert.equal(main.learning.letters, 4);
assert.equal(main.scores.find(score => score.shift === 3).letters, 3);
assert.equal(decoyOnly.selection?.letters, 3);
assert.equal(decoyOnly.match, 'LA E');
assert.equal(disconnected.selection, null);
assert.equal(disconnected.match, '');

const report = { examples, main, controls: { decoyOnly, disconnected }, experiment: { seed: setup.seed,
  mixer: setup.mixer, probe: setup.station.node, alphabet: letters, slots, span, tapDelays,
  referenceEmission, referenceAtProbe, referenceReturn, matchEmission, matchAtProbe,
  referenceLoop: setup.loopDelays[1], referenceFromReturningWaveOnly: true, sceneInjectionsPerTrial: 1 } };
fs.writeFileSync(path.resolve(__dirname, '../outputs/sphere-word-reference-measurements.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ reference: main.reference, match: main.match, selection: main.selection,
  decoy: decoyOnly.selection, disconnected: disconnected.selection, experiment: report.experiment }));
