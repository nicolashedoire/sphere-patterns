'use strict';

const { Simulation: UnusualSimulation } = require('./sphere-unusual-kernel.cjs');

const JUNCTION_DELAY = 2;
const DIRECTIONS = 2;
const NORMALIZE = Math.SQRT1_2;

class SeparatedLanesSimulation extends UnusualSimulation {
  constructor(network, options) {
    super(network, options);
    const { arcs, edges, nodes, nodePorts, sourcePort } = this.network;
    this.splitter = arcs[sourcePort].to;
    this.lanePorts = nodePorts.map(ports => {
      const positions = Array(4).fill(-1);
      for (const [index, port] of ports.entries()) {
        const edge = arcs[port].edge;
        const lane = edge < nodes.length ? 0 : 1;
        const direction = edges[edge].ab === port ? 0 : 1;
        const position = lane * DIRECTIONS + direction;
        if (positions[position] !== -1) throw new Error('Circuit de jonction ambigu.');
        positions[position] = index;
      }
      if (positions.includes(-1)) throw new Error('Circuit de jonction incomplet.');
      return positions;
    });
    this.junctionEchoes = new Float64Array(JUNCTION_DELAY * DIRECTIONS);
    this.echoPhase = 0;
    this.nodeIndex = 0;
  }

  reset() {
    super.reset();
    this.junctionEchoes?.fill(0);
    this.echoPhase = 0;
    this.nodeIndex = 0;
    return this;
  }

  mixLanes(values, first, second, slot) {
    const a = values[first], b = values[second];
    const direct = (a + b) * NORMALIZE;
    const delayed = this.junctionEchoes[slot];
    this.junctionEchoes[slot] = (a - b) * NORMALIZE;
    values[first] = (direct + delayed) * NORMALIZE;
    values[second] = (direct - delayed) * NORMALIZE;
  }

  rotate(values, first, second) {
    // The base kernel calls (0, 1) first at each node. Handle all four ports
    // together there. Incoming samples occupy the reverse outgoing ports.
    if (first !== 0 || second !== 1) return;
    const node = this.nodeIndex++;
    const [forward0, backward0, forward1, backward1] = this.lanePorts[node];
    const a = values[forward0], b = values[forward1];
    values[forward0] = values[backward0];
    values[forward1] = values[backward1];
    values[backward0] = a;
    values[backward1] = b;

    if (node === this.splitter) {
      const slot = this.echoPhase * DIRECTIONS;
      this.mixLanes(values, forward0, forward1, slot);
      this.mixLanes(values, backward0, backward1, slot + 1);
    }
  }

  step(count) {
    count = count == null ? 1 : Math.floor(count);
    if (!Number.isFinite(count) || count < 0 || count > 1000000) {
      throw new RangeError('Nombre de pas invalide.');
    }
    for (let tick = 0; tick < count; tick++) {
      this.nodeIndex = 0;
      super.step(1);
      if (this.nodeIndex !== this.lanePorts.length) throw new Error('Le nombre de jonctions a changé.');
      this.echoPhase = (this.echoPhase + 1) % JUNCTION_DELAY;
    }
    return this;
  }

  energy() {
    let stored = 0;
    for (const value of this.junctionEchoes) stored += value * value;
    return super.energy() + stored;
  }
}

module.exports = { Simulation: SeparatedLanesSimulation, JUNCTION_DELAY };
