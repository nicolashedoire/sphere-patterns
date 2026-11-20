/* Fixed, generic conservative local laws. No glyph labels or recognition logic. */
(function (root) {
  'use strict';
  const base = root.SphereKernel || (typeof require === 'function' ? require('./sphere-kernel.js') : null);
  if (!base) throw new Error('Load SphereKernel before SphereRules.');
  const DEFAULTS = Object.freeze({ theta0: 0.43, energyScale: 0.012, pulseAmplitude: 0.5 });
  // Frozen before experiments. Strengths are shared by all nodes and all inputs.
  const RULES = Object.freeze([
    { id: 'original', label: 'Baseline', family: 'bounded-energy', strength: 0.72 },
    { id: 'linear', label: 'Linear control', family: 'linear', strength: 0 },
    { id: 'bounded-001', label: 'Bounded energy 0.01', family: 'bounded-energy', strength: 0.01 },
    { id: 'bounded-003', label: 'Bounded energy 0.03', family: 'bounded-energy', strength: 0.03 },
    { id: 'bounded-006', label: 'Bounded energy 0.06', family: 'bounded-energy', strength: 0.06 },
    { id: 'band-006', label: 'Energy band 0.06', family: 'band-energy', strength: 0.06 },
    { id: 'band-015', label: 'Energy band 0.15', family: 'band-energy', strength: 0.15 },
    { id: 'junction-003', label: 'Junction energy 0.03', family: 'junction-energy', strength: 0.03 },
    { id: 'junction-008', label: 'Junction energy 0.08', family: 'junction-energy', strength: 0.08 },
    { id: 'balanced-002', label: 'Balanced collision 0.02', family: 'balanced-collision', strength: 0.02 },
    { id: 'balanced-006', label: 'Balanced collision 0.06', family: 'balanced-collision', strength: 0.06 }
  ].map(rule => Object.freeze(rule)));
  function getRule(value) {
    const id = value == null ? 'original' : typeof value === 'string' ? value : value.id;
    const rule = RULES.find(candidate => candidate.id === id);
    if (!rule) throw new RangeError('Unknown local rule: ' + String(id));
    return rule;
  }
  function angleFor(rule, values, first, second) {
    if (rule.family === 'linear') return DEFAULTS.theta0;
    const a = values[first], b = values[second];
    const pairEnergy = a * a + b * b;
    let energy = pairEnergy;
    if (rule.family === 'junction-energy' || rule.family === 'balanced-collision') {
      energy = 0;
      // All four values are available in the existing local scattering call.
      // Their total norm is invariant throughout its four pair rotations.
      for (let i = 0; i < values.length; i++) energy += values[i] * values[i];
    }
    const scale = DEFAULTS.energyScale;
    let response;
    if (rule.family === 'balanced-collision') {
      // Only couples a pair to the energy present in the complementary pair.
      // Both quantities are unchanged by that pair rotation; total local norm
      // is unchanged by every rotation. max only suppresses subtraction roundoff.
      const otherEnergy = Math.max(0, energy - pairEnergy);
      response = (4 * pairEnergy * otherEnergy) / ((scale + energy) * (scale + energy));
    } else if (rule.family === 'band-energy') {
      response = (4 * energy * scale) / ((energy + scale) * (energy + scale));
    } else response = energy / (scale + energy);
    return DEFAULTS.theta0 + rule.strength * response;
  }
  class RuleSimulation extends base.Simulation {
    constructor(network, ruleId) {
      const rule = getRule(ruleId);
      super(network, { theta0: DEFAULTS.theta0, energyScale: DEFAULTS.energyScale,
        pulseAmplitude: DEFAULTS.pulseAmplitude, alpha: rule.strength });
      this.rule = rule;
    }
    rotate(values, first, second) {
      if (this.rule.id === 'original') {
        // Exact reference path, including the original arithmetic order.
        super.rotate(values, first, second);
        return;
      }
      const a = values[first], b = values[second];
      const theta = angleFor(this.rule, values, first, second);
      const cosine = Math.cos(theta), sine = Math.sin(theta);
      values[first] = a * cosine - b * sine;
      values[second] = a * sine + b * cosine;
    }
  }
  const api = { RULES, DEFAULTS, getRule, angleFor,
    createNetwork: base.createNetwork, Simulation: RuleSimulation,
    createSimulation: (network, ruleId) => new RuleSimulation(network, ruleId),
    version: '1.0.0' };
  root.SphereRules = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
