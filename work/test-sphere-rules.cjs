'use strict';
const assert = require('node:assert/strict');
const baseline = require('./sphere-kernel.js');
const rules = require('./sphere-rules-kernel.js');
assert.equal(rules.RULES.length, 11);
assert.ok(Object.isFrozen(rules.RULES));
for (const rule of rules.RULES) assert.ok(Object.isFrozen(rule));
assert.throws(() => rules.getRule('not-a-rule'), RangeError);
const seed = 41873, nodeCount = 64;
const raster = Float64Array.from({ length: 1024 }, (_, i) => (i % 13 === 0 || i % 19 === 0) ? 1 : 0);
const zero = new Float64Array(1024);
function norm(values) { let result = 0; for (const value of values) result += value * value; return result; }
function equalFields(a, b) {
  assert.equal(a.tick, b.tick);
  for (let i = 0; i < a.network.arcs.length; i++) {
    assert.equal(a.network.arcs[i].head, b.network.arcs[i].head);
    assert.deepEqual(a.network.arcs[i].buffer, b.network.arcs[i].buffer);
  }
}
function make(id) { return new rules.Simulation(rules.createNetwork({ seed, nodeCount }), id); }
const results = [];
for (const rule of rules.RULES) {
  const sim = make(rule.id);
  const values = Float64Array.from([0.31, -0.17, 0.27, -0.06]);
  const before = norm(values);
  sim.rotate(values, 0, 1); sim.rotate(values, 2, 3);
  sim.rotate(values, 1, 2); sim.rotate(values, 3, 0);
  assert.ok(Math.abs(norm(values) - before) < 1e-15);
  for (const [first, second] of [[3, 0], [1, 2], [2, 3], [0, 1]]) {
    const theta = rules.angleFor(rule, values, first, second);
    const a = values[first], b = values[second];
    values[first] = a * Math.cos(theta) + b * Math.sin(theta);
    values[second] = -a * Math.sin(theta) + b * Math.cos(theta);
  }
  [0.31, -0.17, 0.27, -0.06].forEach((original, i) => {
    assert.ok(Math.abs(values[i] - original) < 1e-15, 'Local scattering must be reversible.');
  });
  sim.inject(raster, 'first metadata');
  sim.step(2048);
  assert.equal(sim.getStats().queuedImages, 0);
  const injectionEnergy = sim.energy();
  sim.step(1000);
  const stats = sim.getStats();
  assert.ok(Math.abs(stats.energy - injectionEnergy) / injectionEnergy < 1e-11);
  assert.ok(stats.relativeError < 1e-11);
  const repeat = make(rule.id);
  repeat.inject(raster, 'completely different metadata'); repeat.step(3048);
  equalFields(sim, repeat);
  const empty = make(rule.id); empty.inject(zero, 'zero'); empty.step(3048);
  assert.equal(empty.energy(), 0);
  assert.equal(empty.getStats().work, 0);
  results.push({ id: rule.id, postInjectionRelativeEnergyDrift: Math.abs(stats.energy - injectionEnergy) / injectionEnergy,
    sourceAccountingRelativeError: stats.relativeError, metadataExactlyInert: true });
}
const reference = new baseline.Simulation(baseline.createNetwork({ seed, nodeCount }));
reference.inject(raster, 'baseline'); reference.step(3048);
const original = make('original'); original.inject(raster, 'reference'); original.step(3048);
equalFields(reference, original);
const linearReference = new baseline.Simulation(baseline.createNetwork({ seed, nodeCount }), { alpha: 0 });
linearReference.inject(raster); linearReference.step(3048);
const linear = make('linear'); linear.inject(raster); linear.step(3048);
equalFields(linearReference, linear);
const band = rules.getRule('band-015');
assert.ok(Math.abs(rules.angleFor(band, [Math.sqrt(0.012), 0], 0, 1) - (0.43 + 0.15)) < 1e-15);
const bounded = rules.getRule('bounded-003'), junction = rules.getRule('junction-003');
assert.equal(rules.angleFor(bounded, [0.1, 0.2, 0.3, 0.4], 0, 1), rules.angleFor(bounded, [0.1, 0.2, 0, 0], 0, 1));
assert.notEqual(rules.angleFor(junction, [0.1, 0.2, 0.3, 0.4], 0, 1), rules.angleFor(junction, [0.1, 0.2, 0, 0], 0, 1));
const balanced = rules.getRule('balanced-006');
assert.equal(rules.angleFor(balanced, [0.1, 0.2, 0, 0], 0, 1), 0.43);
assert.ok(rules.angleFor(balanced, [0.1, 0.2, 0.3, 0.4], 0, 1) > 0.43);
console.log(JSON.stringify({ passed: true, originalBitwiseMatchesBaseline: true,
  linearBitwiseMatchesBaseline: true, rules: results }, null, 2));
