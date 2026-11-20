/* Closed wave network, exploratory model. No training, losses, pruning or rescaling. */
(function (root) {
  'use strict';

  function randomGenerator(seed) {
    let value = seed >>> 0;
    return function () {
      value += 0x6D2B79F5;
      let t = value;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function pairKey(a, b) { return a < b ? a + ':' + b : b + ':' + a; }
  function squaredDistance(a, b) {
    const x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    return x * x + y * y + z * z;
  }

  // Two edge-disjoint spatial Hamiltonian cycles give a connected 4-regular graph.
  // The cycles only construct the geometry; propagation uses every port equally.
  function spatialCycle(nodes, random, forbidden) {
    const count = nodes.length;
    for (let attempt = 0; attempt < 80; attempt++) {
      const used = new Uint8Array(count);
      const tour = [Math.floor(random() * count)];
      used[tour[0]] = 1;
      while (tour.length < count) {
        const current = tour[tour.length - 1];
        let best = -1, bestDistance = Infinity;
        for (let candidate = 0; candidate < count; candidate++) {
          if (used[candidate] || forbidden.has(pairKey(current, candidate))) continue;
          const distance = squaredDistance(nodes[current], nodes[candidate]);
          if (distance < bestDistance) { best = candidate; bestDistance = distance; }
        }
        if (best < 0) break;
        used[best] = 1;
        tour.push(best);
      }
      if (tour.length === count && !forbidden.has(pairKey(tour[count - 1], tour[0]))) return tour;
    }
    throw new Error('Could not construct both spatial cycles.');
  }

  function createNetwork(options) {
    options = options || {};
    const nodeCount = options.nodeCount == null ? 1000 : options.nodeCount;
    if (!Number.isInteger(nodeCount) || nodeCount < 16 || nodeCount > 4000) {
      throw new RangeError('nodeCount must be an integer between 16 and 4,000.');
    }
    const seed = options.seed == null ? 260617 : options.seed >>> 0;
    const random = randomGenerator(seed);
    const nodes = [];
    while (nodes.length < nodeCount) {
      const p = [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1];
      if (squaredDistance(p, [0, 0, 0]) <= 1) nodes.push(p);
    }
    const edges = [], arcs = [], ports = Array.from({ length: nodeCount }, () => []);
    const usedEdges = new Set();
    for (let cycle = 0; cycle < 2; cycle++) {
      const tour = spatialCycle(nodes, random, usedEdges);
      for (let i = 0; i < nodeCount; i++) {
        const a = tour[i], b = tour[(i + 1) % nodeCount];
        const length = Math.sqrt(squaredDistance(nodes[a], nodes[b]));
        const delay = Math.ceil(length * 18) + 5;
        const edgeIndex = edges.length, ab = arcs.length, ba = ab + 1;
        arcs.push({ from: a, to: b, edge: edgeIndex, reverse: ba, delay, buffer: new Float64Array(delay), head: 0 });
        arcs.push({ from: b, to: a, edge: edgeIndex, reverse: ab, delay, buffer: new Float64Array(delay), head: 0 });
        edges.push({ a, b, length, delay, ab, ba });
        ports[a].push(ab); ports[b].push(ba);
        usedEdges.add(pairKey(a, b));
      }
    }
    // Port ordering is purely geometric and reproducible, rather than cycle order.
    const nodePorts = ports.map((p, index) => Int32Array.from(p.sort((left, right) => {
      const l = nodes[arcs[left].to], r = nodes[arcs[right].to], c = nodes[index];
      const la = Math.atan2(l[2] - c[2], l[1] - c[1]);
      const ra = Math.atan2(r[2] - c[2], r[1] - c[1]);
      return la - ra || left - right;
    })));
    let sourceNode = 0;
    for (let i = 1; i < nodeCount; i++) if (nodes[i][0] < nodes[sourceNode][0]) sourceNode = i;
    // Enter along the most inward-pointing of the four source-node connections.
    let sourcePort = nodePorts[sourceNode][0], inward = -Infinity;
    for (const port of nodePorts[sourceNode]) {
      const to = nodes[arcs[port].to];
      const score = (to[0] - nodes[sourceNode][0]) / edges[arcs[port].edge].length;
      if (score > inward) { inward = score; sourcePort = port; }
    }
    return { seed, nodes, edges, arcs, nodePorts, sourceNode, sourcePort };
  }

  class Simulation {
    constructor(network, options) {
      this.network = network || createNetwork();
      options = options || {};
      this.theta0 = options.theta0 == null ? 0.43 : options.theta0;
      this.alpha = options.alpha == null ? 0.72 : options.alpha;
      this.energyScale = options.energyScale == null ? 0.012 : options.energyScale;
      this.pulseAmplitude = options.pulseAmplitude == null ? 0.5 : options.pulseAmplitude;
      if (![this.theta0, this.alpha, this.energyScale, this.pulseAmplitude].every(Number.isFinite) || this.energyScale <= 0) {
        throw new RangeError('Invalid simulation constants.');
      }
      const size = this.network.arcs.length;
      this.lastIncoming = new Float64Array(size);
      this.lastOutgoing = new Float64Array(size);
      this.reset();
    }

    reset() {
      for (const arc of this.network.arcs) { arc.buffer.fill(0); arc.head = 0; }
      this.lastIncoming.fill(0); this.lastOutgoing.fill(0);
      this.tick = 0;
      this.work = 0;
      this.workMagnitude = 0;
      this.queue = [];
      this.imagesInjected = 0;
      this.imagesCompleted = 0;
      this.lastLabel = '';
      return this;
    }

    inject(pixels, label) {
      if (!pixels || !Number.isInteger(pixels.length) || pixels.length < 1 || pixels.length > 16384) {
        throw new RangeError('The injected image must contain between 1 and 16,384 luminance values.');
      }
      const samples = new Float64Array(pixels.length * 2);
      for (let i = 0; i < pixels.length; i++) {
        const raw = Number(pixels[i]);
        if (!Number.isFinite(raw)) throw new TypeError('Non-finite luminance value.');
        const amplitude = Math.max(0, Math.min(1, raw)) * this.pulseAmplitude;
        // A positive and negative lobe, one tick each, for each raster-read pixel.
        samples[i * 2] = amplitude;
        samples[i * 2 + 1] = -amplitude;
      }
      const item = { samples, index: 0, label: String(label || 'Image'), id: ++this.imagesInjected };
      this.queue.push(item);
      return item.id;
    }

    // A nonlinear rotation is norm-preserving and invertible: its angle only
    // depends on the pair energy, which the rotation itself preserves.
    rotate(values, first, second) {
      const a = values[first], b = values[second];
      const energy = a * a + b * b;
      const theta = this.theta0 + this.alpha * energy / (this.energyScale + energy);
      const cosine = Math.cos(theta), sine = Math.sin(theta);
      values[first] = a * cosine - b * sine;
      values[second] = a * sine + b * cosine;
    }

    step(count) {
      count = count == null ? 1 : Math.floor(count);
      if (!Number.isFinite(count) || count < 0 || count > 1000000) throw new RangeError('Invalid step count.');
      const { arcs, nodePorts, sourcePort } = this.network;
      const incoming = this.lastIncoming, outgoing = this.lastOutgoing;
      const local = new Float64Array(4);
      for (let iteration = 0; iteration < count; iteration++) {
        // Remove each arrived wave before anything is scattered. An incoming
        // wave is indexed by the reverse (outgoing) port at its destination.
        for (let index = 0; index < arcs.length; index++) {
          const arc = arcs[index];
          incoming[arc.reverse] = arc.buffer[arc.head];
          arc.buffer[arc.head] = 0;
          arc.head = (arc.head + 1) % arc.delay;
        }
        for (const ports of nodePorts) {
          local[0] = incoming[ports[0]]; local[1] = incoming[ports[1]];
          local[2] = incoming[ports[2]]; local[3] = incoming[ports[3]];
          this.rotate(local, 0, 1); this.rotate(local, 2, 3);
          this.rotate(local, 1, 2); this.rotate(local, 3, 0);
          outgoing[ports[0]] = local[0]; outgoing[ports[1]] = local[1];
          outgoing[ports[2]] = local[2]; outgoing[ports[3]] = local[3];
        }
        if (this.queue.length) {
          const current = this.queue[0];
          const source = current.samples[current.index++];
          const oldValue = outgoing[sourcePort];
          const delta = source * source + 2 * oldValue * source;
          // Exact source work: an additive source can interfere destructively
          // with an old returning wave. Nothing is discarded or normalized.
          this.work += delta;
          this.workMagnitude += Math.abs(delta);
          outgoing[sourcePort] = oldValue + source;
          if (current.index === current.samples.length) {
            this.imagesCompleted++;
            this.lastLabel = current.label;
            this.queue.shift();
          }
        }
        for (let index = 0; index < arcs.length; index++) {
          const arc = arcs[index];
          const write = (arc.head + arc.delay - 1) % arc.delay;
          arc.buffer[write] = outgoing[index];
        }
        this.tick++;
      }
      return this;
    }

    energy() {
      // Kahan accumulation makes the independent measured energy meaningful
      // even after the wave has spread into many very small amplitudes.
      let total = 0, correction = 0;
      for (const arc of this.network.arcs) for (const value of arc.buffer) {
        const adjusted = value * value - correction;
        const sum = total + adjusted;
        correction = (sum - total) - adjusted;
        total = sum;
      }
      return total;
    }

    getStats() {
      const energy = this.energy();
      const item = this.queue[0];
      const progress = item ? item.index / item.samples.length : (this.imagesCompleted ? 1 : 0);
      return {
        tick: this.tick, energy, work: this.work, energyBudget: this.work,
        error: energy - this.work,
        relativeError: Math.abs(energy - this.work) / Math.max(1e-30, this.workMagnitude, energy),
        nodeCount: this.network.nodes.length, edgeCount: this.network.edges.length,
        arcCount: this.network.arcs.length, imagesInjected: this.imagesInjected,
        imagesCompleted: this.imagesCompleted, queuedImages: this.queue.length,
        injecting: !!item, progress, injectionProgress: progress,
        activeLabel: item ? item.label : this.lastLabel,
        sourceNode: this.network.sourceNode, sourcePort: this.network.sourcePort
      };
    }

    // Every emitted sample is a real stored wave. Display code may thin or
    // threshold these samples without modifying the numerical state.
    // head is next to arrive; the newest sample lies at head+delay-1.
    sampleWaves(callback, stride, phase) {
      stride = Math.max(1, Math.floor(stride || 1));
      phase = Math.max(0, Math.min(1, Number(phase) || 0));
      const arcs = this.network.arcs;
      for (let index = 0; index < arcs.length; index++) {
        const arc = arcs[index];
        for (let bin = 0; bin < arc.delay; bin += stride) {
          const amplitude = arc.buffer[(arc.head + bin) % arc.delay];
          if (amplitude === 0) continue;
          callback({ arcIndex: index, from: arc.from, to: arc.to,
            position: (arc.delay - 1 - bin + phase) / arc.delay,
            amplitude, energy: amplitude * amplitude, bin });
        }
      }
    }

    inspectNode(index) {
      if (!Number.isInteger(index) || index < 0 || index >= this.network.nodes.length) return null;
      let energy = 0;
      const ports = Array.from(this.network.nodePorts[index], arcIndex => {
        const incoming = this.lastIncoming[arcIndex], outgoing = this.lastOutgoing[arcIndex];
        energy += incoming * incoming;
        return { arcIndex, neighbor: this.network.arcs[arcIndex].to, incoming, outgoing };
      });
      return { index, position: this.network.nodes[index].slice(), ports, energy };
    }
  }

  const api = { createNetwork, Simulation, version: '1.0.0' };
  root.SphereKernel = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
