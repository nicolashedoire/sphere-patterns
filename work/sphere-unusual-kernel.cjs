'use strict';

// Abstract conservative link scattering with local surprise-gated angle writing.
// Calibration uses only current local waves; it has no clock, word or label input.
const Deformable = require('./sphere-deformable-kernel.cjs');
const Rules = require('./sphere-rules-kernel.js');
const HABIT = Object.freeze({ updateRate: 1 / 256, coherenceScale: 0.012, deviationMultiplier: 2 });
const DEFAULTS = Object.freeze({ ...Deformable.DEFAULTS, ...HABIT });

function ownedCalibration(input, length, name, minimum, maximum) {
  if (input === undefined) return new Float64Array(length);
  if (!(input instanceof Float64Array)) throw new TypeError(name + ' must be a Float64Array.');
  if (input.length !== length) throw new RangeError(name + ' must contain one value per link.');
  for (const value of input) {
    if (!Number.isFinite(value) || value < minimum || value > maximum) {
      throw new RangeError(name + ' contains a non-finite or out-of-range value.');
    }
  }
  return Float64Array.from(input);
}

class UnusualSimulation extends Deformable.Simulation {
  constructor(network, options) {
    const checked = options === undefined ? {} : options;
    if (checked === null || typeof checked !== 'object' || Array.isArray(checked)) {
      throw new TypeError('Options must be an object.');
    }
    for (const key of Object.keys(checked)) {
      if (!['writing', 'angles', 'meanCoherence', 'meanDeviation'].includes(key)) {
        throw new TypeError('Unknown option: ' + key);
      }
    }
    const resolved = network || Rules.createNetwork();
    const mean = ownedCalibration(checked.meanCoherence, resolved.edges.length, 'meanCoherence', -1, 1);
    const deviation = ownedCalibration(checked.meanDeviation, resolved.edges.length, 'meanDeviation', 0, 2);
    super(resolved, { angles: checked.angles, writing: checked.writing });
    Object.defineProperty(this, 'meanCoherence', { value: mean, enumerable: true });
    Object.defineProperty(this, 'meanDeviation', { value: deviation, enumerable: true });
    Object.defineProperty(this, 'habituationConstants', { value: HABIT, enumerable: true });
  }

  step(count) {
    count = count == null ? 1 : Math.floor(count);
    if (!Number.isFinite(count) || count < 0 || count > 1000000) {
      throw new RangeError('Invalid step count.');
    }
    for (let tick = 0; tick < count; tick++) {
      // Bypass Deformable.step: perform the unchanged base transport exactly once,
      // then the same single link rotation, followed by this variant's update.
      Rules.Simulation.prototype.step.call(this, 1);
      for (const midpoint of this.midpoints) {
        const index = midpoint.edgeIndex;
        const ab = this.network.arcs[midpoint.arcAB], ba = this.network.arcs[midpoint.arcBA];
        const indexAB = (ab.head + midpoint.binAB) % ab.delay;
        const indexBA = (ba.head + midpoint.binBA) % ba.delay;
        this.lastMidpointIndices[index * 2] = indexAB;
        this.lastMidpointIndices[index * 2 + 1] = indexBA;
        const a = ab.buffer[indexAB], b = ba.buffer[indexBA];
        const angle = this.angles[index];
        if (angle !== 0) {
          const cosine = Math.cos(angle), sine = Math.sin(angle);
          ab.buffer[indexAB] = a * cosine - b * sine;
          ba.buffer[indexBA] = a * sine + b * cosine;
        }
        const coherence = (2 * a * b) / (HABIT.coherenceScale + a * a + b * b);
        const mean = this.meanCoherence[index], deviation = this.meanDeviation[index];
        const difference = coherence - mean;
        if (this.writing) {
          const unexpected = Math.max(0, Math.abs(difference) - HABIT.deviationMultiplier * deviation);
          const next = angle + Deformable.DEFAULTS.rate * Math.sign(difference) * unexpected;
          this.angles[index] = Math.max(-Deformable.DEFAULTS.cap, Math.min(Deformable.DEFAULTS.cap, next));
        }
        // Always calibrate, even while angle writing is disabled. Both use the
        // prior mean/deviation and the pre-rotation counterpropagating samples.
        this.meanCoherence[index] = mean + HABIT.updateRate * difference;
        this.meanDeviation[index] = deviation + HABIT.updateRate * (Math.abs(difference) - deviation);
      }
    }
    return this;
  }
}

module.exports = { Simulation: UnusualSimulation, UnusualSimulation,
  DEFAULTS, HABIT, createNetwork: Rules.createNetwork };
