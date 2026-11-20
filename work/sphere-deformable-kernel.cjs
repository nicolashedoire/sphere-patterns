'use strict';

/*
 * Opt-in abstract link scattering layered after the unchanged bounded-003 step.
 * Persistent link angles change counterpropagating wave amplitudes, not travel
 * times or buffer lengths. The rotations conserve the model's squared wave norm;
 * this is not a claim about material/actuator energy or a literal elastic link.
 */
const Rules = require('./sphere-rules-kernel.js');

const DEFAULTS = Object.freeze({
  baseRuleId: 'bounded-003',
  rate: 1 / 4096,
  scale: 0.012,
  cap: 0.10
});

function validateOptions(options) {
  if (options === undefined) return {};
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('Les options doivent être un objet.');
  }
  for (const key of Object.keys(options)) {
    if (key !== 'angles' && key !== 'writing') throw new TypeError('Option inconnue : ' + key);
  }
  if (options.writing !== undefined && typeof options.writing !== 'boolean') {
    throw new TypeError('writing doit être un booléen.');
  }
  return options;
}

function ownedAngleMap(input, length) {
  if (input === undefined) return new Float64Array(length);
  if (!(input instanceof Float64Array)) throw new TypeError('angles doit être un Float64Array.');
  if (input.length !== length) throw new RangeError('La carte doit contenir un angle par liaison.');
  for (const value of input) {
    if (!Number.isFinite(value) || value < -DEFAULTS.cap || value > DEFAULTS.cap) {
      throw new RangeError('Chaque angle doit être fini et compris entre -0,10 et +0,10 radian.');
    }
  }
  return Float64Array.from(input);
}

class DeformableSimulation extends Rules.Simulation {
  constructor(network, options) {
    const checked = validateOptions(options);
    const resolvedNetwork = network || Rules.createNetwork();
    const angles = ownedAngleMap(checked.angles, resolvedNetwork.edges.length);
    super(resolvedNetwork, DEFAULTS.baseRuleId);
    Object.defineProperty(this, 'angles', { value: angles, enumerable: true });
    Object.defineProperty(this, 'ruleConstants', { value: DEFAULTS, enumerable: true });
    this._writing = checked.writing === undefined ? false : checked.writing;
    this.midpoints = Object.freeze(this.network.edges.map((edge, edgeIndex) => {
      const ab = this.network.arcs[edge.ab], ba = this.network.arcs[edge.ba];
      if (!ab || !ba || ab.delay !== ba.delay || ab.reverse !== edge.ba || ba.reverse !== edge.ab) {
        throw new RangeError('Les deux directions de chaque liaison doivent être appariées et de même délai.');
      }
      const binAB = Math.floor((ab.delay - 1) / 2);
      // Complementary logical bins have coincident bin-center positions,
      // including even delays. Each physical ring index uses its own live head.
      const binBA = ab.delay - 1 - binAB;
      return Object.freeze({ edgeIndex, arcAB: edge.ab, arcBA: edge.ba, binAB, binBA });
    }));
    this.lastMidpointIndices = new Int32Array(this.midpoints.length * 2);
    this._registerMidpointIndices();
  }

  get writing() { return this._writing; }
  set writing(value) {
    if (typeof value !== 'boolean') throw new TypeError('writing doit être un booléen.');
    this._writing = value;
  }

  getAngles() { return Float64Array.from(this.angles); }

  setAngles(values) {
    const validated = ownedAngleMap(values, this.network.edges.length);
    this.angles.set(validated);
    return this;
  }

  _registerMidpointIndices() {
    for (const midpoint of this.midpoints) {
      const ab = this.network.arcs[midpoint.arcAB], ba = this.network.arcs[midpoint.arcBA];
      this.lastMidpointIndices[midpoint.edgeIndex * 2] = (ab.head + midpoint.binAB) % ab.delay;
      this.lastMidpointIndices[midpoint.edgeIndex * 2 + 1] = (ba.head + midpoint.binBA) % ba.delay;
    }
  }

  step(count) {
    // Match the base kernel's coercion, limits, default and invalid-count errors.
    count = count == null ? 1 : Math.floor(count);
    if (!Number.isFinite(count) || count < 0 || count > 1000000) {
      throw new RangeError('Nombre de pas invalide.');
    }
    if (count === 0) return this;
    if (!this.writing && this.angles.every(value => value === 0)) {
      // This branch is deliberately bit-identical to the unchanged base rule.
      super.step(count);
      this._registerMidpointIndices();
      return this;
    }
    for (let tick = 0; tick < count; tick++) {
      // Entire original transport, junction scattering and source accounting first.
      super.step(1);
      for (const midpoint of this.midpoints) {
        const index = midpoint.edgeIndex;
        const ab = this.network.arcs[midpoint.arcAB], ba = this.network.arcs[midpoint.arcBA];
        const indexAB = (ab.head + midpoint.binAB) % ab.delay;
        const indexBA = (ba.head + midpoint.binBA) % ba.delay;
        this.lastMidpointIndices[index * 2] = indexAB;
        this.lastMidpointIndices[index * 2 + 1] = indexBA;
        const a = ab.buffer[indexAB], b = ba.buffer[indexBA];
        const angle = this.angles[index];
        // Apply only the pre-existing angle. A zero angle performs no writes.
        if (angle !== 0) {
          const cosine = Math.cos(angle), sine = Math.sin(angle);
          ab.buffer[indexAB] = a * cosine - b * sine;
          ba.buffer[indexBA] = a * sine + b * cosine;
        }
        if (this.writing) {
          // The update uses pre-rotation amplitudes and never reads labels.
          const delta = DEFAULTS.rate * (2 * a * b) / (DEFAULTS.scale + a * a + b * b);
          this.angles[index] = Math.max(-DEFAULTS.cap, Math.min(DEFAULTS.cap, angle + delta));
        }
      }
    }
    return this;
  }
}

module.exports = {
  Simulation: DeformableSimulation,
  DeformableSimulation,
  DEFAULTS,
  createNetwork: Rules.createNetwork
};
