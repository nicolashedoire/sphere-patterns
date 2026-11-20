'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const HEADER_BYTES = 4;
const MAX_PAYLOAD_BYTES = 128;
const DEFAULT_AMPLITUDE = 0.5;
const OUTPUT_NAMES = ['recovered-text.txt', 'recovered-image.png'];

function encode(bytes, amplitude = DEFAULT_AMPLITUDE) {
  assert.ok(bytes instanceof Uint8Array && bytes.length <= MAX_PAYLOAD_BYTES, 'Invalid payload.');
  assert.ok(Number.isFinite(amplitude) && amplitude > 0, 'Invalid amplitude.');
  const frame = Buffer.alloc(HEADER_BYTES + bytes.length);
  frame.writeUInt32BE(bytes.length);
  frame.set(bytes, HEADER_BYTES);
  const symbols = new Float64Array(frame.length * 8);
  for (const [index, byte] of frame.entries()) {
    for (let bit = 0; bit < 8; bit++) {
      symbols[index * 8 + bit] = (byte & (0x80 >> bit)) ? amplitude : -amplitude;
    }
  }
  return symbols;
}

function unpack(symbols, amplitude) {
  const tolerance = amplitude / 4;
  let maxResidual = 0;
  function readBytes(count, start) {
    assert.ok((start + count) * 8 <= symbols.length, 'Truncated frame.');
    const result = Buffer.alloc(count);
    for (let index = 0; index < count; index++) {
      for (let bit = 0; bit < 8; bit++) {
        const sample = symbols[(start + index) * 8 + bit];
        const expected = sample > 0 ? amplitude : -amplitude;
        const residual = Math.abs(sample - expected);
        assert.ok(residual <= tolerance, 'Amplitude outside the symbol alphabet.');
        result[index] = (result[index] << 1) | Number(sample > 0);
        maxResidual = Math.max(maxResidual, residual);
      }
    }
    return result;
  }

  const count = readBytes(HEADER_BYTES, 0).readUInt32BE();
  assert.ok(count <= MAX_PAYLOAD_BYTES, 'Invalid payload length.');
  const bytes = readBytes(count, HEADER_BYTES);
  for (let tick = (HEADER_BYTES + count) * 8; tick < symbols.length; tick++) {
    const residual = Math.abs(symbols[tick]);
    assert.ok(residual <= tolerance, 'Unexpected signal after the frame.');
    maxResidual = Math.max(maxResidual, residual);
  }
  return { bytes, maxResidual };
}

function decode({ amplitude, firstTicks, matrix, traces }) {
  const pair = value => Array.isArray(value) && value.length === 2;
  assert.ok(Number.isFinite(amplitude) && amplitude > 0, 'Invalid amplitude.');
  assert.ok(pair(firstTicks) && firstTicks.every(tick => Number.isSafeInteger(tick) && tick >= 1), 'Invalid arrival ticks.');
  assert.ok(pair(matrix) && matrix.every(row => pair(row) && row.every(Number.isFinite)), 'Invalid calibration matrix.');
  assert.ok(pair(traces) && traces.every(trace => Array.isArray(trace) && trace.every(Number.isFinite)), 'Invalid probe traces.');

  const [[a, b], [c, d]] = matrix;
  const determinant = a * d - b * c;
  const scale = Math.max(Math.abs(a), Math.abs(b), Math.abs(c), Math.abs(d));
  assert.ok(Number.isFinite(determinant) && Math.abs(determinant) > 1e-12 * scale ** 2, 'The mixtures are insufficiently independent.');

  const starts = firstTicks.map(tick => tick - 1);
  const alignedSymbols = Math.min(traces[0].length - starts[0], traces[1].length - starts[1]);
  assert.ok(alignedSymbols >= HEADER_BYTES * 8, 'Incomplete header in the probe output.');
  const channels = [new Float64Array(alignedSymbols), new Float64Array(alignedSymbols)];
  for (let tick = 0; tick < alignedSymbols; tick++) {
    const y = traces[0][starts[0] + tick];
    const z = traces[1][starts[1] + tick];
    channels[0][tick] = (d * y - b * z) / determinant;
    channels[1][tick] = (a * z - c * y) / determinant;
  }
  return { files: channels.map(symbols => unpack(symbols, amplitude)), alignedSymbols };
}

if (require.main === module) {
  const [capturePath, outputDir] = process.argv.slice(2);
  try {
    assert.ok(capturePath && outputDir && process.argv.length === 4, 'Usage: node sphere-roundtrip-codec.cjs capture.json output-dir');
    const { files, alignedSymbols } = decode(JSON.parse(fs.readFileSync(capturePath, 'utf8')));
    fs.mkdirSync(outputDir, { recursive: true });
    const stats = files.map(({ bytes, maxResidual }, index) => {
      const name = OUTPUT_NAMES[index];
      fs.writeFileSync(path.join(outputDir, name), bytes);
      return { name, bytes: bytes.length, maxResidual };
    });
    console.log(JSON.stringify({ files: stats, alignedSymbols }));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = { encode, decode };
