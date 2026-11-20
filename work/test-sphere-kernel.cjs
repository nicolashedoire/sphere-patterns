'use strict';
const assert = require('node:assert/strict');
const { createNetwork, Simulation } = require('./sphere-kernel.js');

const network = createNetwork({ nodeCount: 1000, seed: 260617 });
assert.equal(network.edges.length, 2000);
assert.equal(network.arcs.length, 4000);
const seen = new Set([0]), todo = [0], pairs = new Set();
for (const ports of network.nodePorts) assert.equal(ports.length, 4);
while (todo.length) {
  for (const port of network.nodePorts[todo.pop()]) {
    const next = network.arcs[port].to;
    if (!seen.has(next)) { seen.add(next); todo.push(next); }
  }
}
assert.equal(seen.size, 1000);
for (const node of network.nodes) assert.ok(Math.hypot(...node) <= 1);
for (const edge of network.edges) {
  const key = [edge.a, edge.b].sort((a, b) => a - b).join(':');
  assert.ok(!pairs.has(key)); pairs.add(key);
  assert.notEqual(edge.a, edge.b);
  assert.equal(network.arcs[edge.ab].reverse, edge.ba);
}
const circle = Array.from({ length: 256 }, (_, i) => {
  const x = (i % 16) - 7.5, y = Math.floor(i / 16) - 7.5;
  return Math.abs(Math.hypot(x, y) - 4) < 1 ? 1 : 0;
});
const line = Array.from({ length: 256 }, (_, i) => i % 16 === 7 || i % 16 === 8 ? 1 : 0);
const zero = new Float64Array(256);
const sim = new Simulation(network);
sim.inject(circle, 'Cercle'); sim.step(512);
const afterInjection = sim.getStats();
assert.equal(afterInjection.queuedImages, 0);
assert.ok(afterInjection.energy > 0);
sim.step(10000);
const afterTravel = sim.getStats();
assert.ok(Math.abs(afterTravel.energy - afterInjection.energy) / afterInjection.energy < 1e-10);
assert.ok(afterTravel.relativeError < 1e-10);
const existing = network.arcs.map(arc => Array.from(arc.buffer));
sim.inject(line, 'Ligne');
assert.deepEqual(network.arcs.map(arc => Array.from(arc.buffer)), existing, 'Queueing an image must not clear or change old waves.');
sim.step(512);
assert.equal(sim.getStats().imagesCompleted, 2);
assert.ok(sim.getStats().relativeError < 1e-10);
let renderedEnergy = 0, samples = 0;
sim.sampleWaves(sample => { renderedEnergy += sample.energy; samples++; assert.ok(sample.position >= 0 && sample.position <= 1); });
assert.ok(Math.abs(renderedEnergy - sim.energy()) / sim.energy() < 1e-12);
sim.reset(); sim.inject(zero, 'Noir'); sim.step(700);
assert.equal(sim.energy(), 0);
sim.reset(); assert.equal(sim.tick, 0); assert.equal(sim.getStats().imagesInjected, 0);

function signature(pixels) {
  const model = new Simulation(createNetwork({ nodeCount: 64, seed: 21 }));
  model.inject(pixels); model.step(900);
  return model.network.arcs.reduce((sum, arc, i) => sum + arc.buffer[(arc.head + 2) % arc.delay] * (i + 1), 0);
}
const circleSignature = signature(circle), lineSignature = signature(line);
assert.ok(Math.abs(circleSignature - lineSignature) > 1e-6);
assert.equal(circleSignature, signature(circle), 'Seeded geometry and propagation must be reproducible.');

// The generalized encoder preserves the original256slot path and supports
// position-registered64×16word images without changing propagation.
const wide = new Float64Array(1024);
for (let i = 0; i < wide.length; i++) wide[i] = i % 7 === 0 ? 1 : 0;
const wideSim = new Simulation(createNetwork({ nodeCount: 64, seed: 21 }));
wideSim.inject(wide, '64 × 16');
assert.equal(wideSim.queue[0].samples.length, 2048);
for (let i = 0; i < wide.length; i++) {
  assert.equal(wideSim.queue[0].samples[i * 2], wide[i] * 0.5);
  assert.equal(wideSim.queue[0].samples[i * 2 + 1], -wide[i] * 0.5);
}
wideSim.step(2047);
assert.equal(wideSim.getStats().queuedImages, 1);
wideSim.step(1);
assert.equal(wideSim.getStats().imagesCompleted, 1);
const wideEnergy = wideSim.energy();
assert.ok(wideEnergy > 0 && Number.isFinite(wideEnergy));
wideSim.step(1000);
assert.ok(Math.abs(wideSim.energy() - wideEnergy) / wideEnergy < 1e-10);
assert.ok(wideSim.getStats().relativeError < 1e-10);
assert.throws(() => wideSim.inject([]), RangeError);
assert.throws(() => wideSim.inject(new Float64Array(16385)), RangeError);

console.log(JSON.stringify({
  passed: true, nodes: network.nodes.length, edges: network.edges.length,
  delayRange: [Math.min(...network.edges.map(e => e.delay)), Math.max(...network.edges.map(e => e.delay))],
  energyAfterInjection: afterInjection.energy, energyAfter10000Ticks: afterTravel.energy,
  relativeConservationError: afterTravel.relativeError, renderedSamples: samples,
  imageSignatures: { circle: circleSignature, line: lineSignature },
  generalizedEncoder: { samples: wide.length, injectionTicks: 2048,
    relativeConservationError: wideSim.getStats().relativeError }
}, null, 2));
