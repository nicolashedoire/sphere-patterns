'use strict';

const assert = require('node:assert/strict');
const { createNetwork } = require('./sphere-unusual-kernel.cjs');
const { Simulation, JUNCTION_DELAY } = require('./sphere-separated-lanes-kernel.cjs');

assert.equal(JUNCTION_DELAY, 2);
const spec = { seed: 41873, nodeCount: 32 };

function close(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-13, `${actual} differs from ${expected}`);
}

function portFor(network, node, lane, forward) {
  return network.nodePorts[node].find(port => {
    const edge = network.arcs[port].edge;
    return (edge < network.nodes.length) === (lane === 0) && (network.edges[edge].ab === port) === forward;
  });
}

function seedIncoming(sim, port, amplitude) {
  const { arcs } = sim.network;
  const arc = arcs[arcs[port].reverse];
  arc.buffer[arc.head] = amplitude;
  sim.work += amplitude ** 2;
  sim.workMagnitude = sim.work;
}

const split = new Simulation(createNetwork(spec), { writing: false });
const splitter = split.network.arcs[split.network.sourcePort].to;
const forward = [0, 1].map(lane => portFor(split.network, splitter, lane, true));
const backward = [0, 1].map(lane => portFor(split.network, splitter, lane, false));
seedIncoming(split, backward[0], 0.4);
seedIncoming(split, forward[1], 0.2);

for (const [f0, f1, b0, b1] of [[0.2, 0.2, 0.1, 0.1], [0, 0, 0, 0], [0.2, -0.2, -0.1, 0.1]]) {
  split.step();
  for (const [port, expected] of [[forward[0], f0], [forward[1], f1], [backward[0], b0], [backward[1], b1]]) {
    close(split.lastOutgoing[port], expected);
  }
  close(split.energy(), 0.2);
}

const through = new Simulation(createNetwork(spec), { writing: false });
const throughNode = (splitter + 1) % spec.nodeCount;
seedIncoming(through, portFor(through.network, throughNode, 1, true), 0.4);
through.step();
assert.equal(through.lastOutgoing[portFor(through.network, throughNode, 1, false)], 0.4);
assert.equal(through.lastOutgoing.filter(Boolean).length, 1);

split.reset();
assert.equal(split.energy(), 0);
seedIncoming(split, backward[0], 0.4);
split.step();
close(split.lastOutgoing[forward[0]], 0.2);
close(split.lastOutgoing[forward[1]], 0.2);

function loaded() {
  const network = createNetwork(spec);
  const sim = new Simulation(network, { writing: true });
  for (const [index, arc] of network.arcs.entries()) {
    for (let bin = 0; bin < arc.delay; bin++) arc.buffer[bin] = Math.sin(index * 3 + bin * 5) * 0.2;
  }
  sim.work = sim.workMagnitude = sim.energy();
  sim.inject(Float64Array.of(0));
  sim.queue[0].samples = Float64Array.of(0.21, -0.13, 0.05);
  return sim;
}

const batched = loaded().step(131);
const individual = loaded();
for (let tick = 0; tick < 131; tick++) individual.step();
assert.deepEqual(batched.network.arcs, individual.network.arcs);
assert.deepEqual(batched.junctionEchoes, individual.junctionEchoes);
assert.deepEqual(batched.angles, individual.angles);
assert.deepEqual(batched.getStats(), individual.getStats());
assert.ok(batched.getStats().relativeError < 1e-12);
console.log('Deux circuits vérifiés dans les deux sens, avec conservation de l’énergie et parité des pas groupés.');
