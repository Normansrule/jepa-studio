// Write sample .safetensors, .npy and .zip files with the web writers into a directory given
// on the command line; tests/test_web_engine.py opens them with safetensors / numpy / zipfile.
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { safetensorsBytes, parseSafetensors } from '../../site/js/io/safetensors.js';
import { npyBytes } from '../../site/js/io/npy.js';
import { zipBytes } from '../../site/js/io/zip.js';
import { initParams } from '../../site/js/engine/mlp.js';

const dir = process.argv[2];
mkdirSync(dir, { recursive: true });

const params = initParams(3);
const tensors = [];
for (const p of params) {
  tensors.push({ name: `${p.name}.weight`, data: p.W, shape: [p.dout, p.din] });
  tensors.push({ name: `${p.name}.bias`, data: p.b, shape: [p.dout] });
}
tensors.push({ name: 'odd.shape', data: Float32Array.from([1, 2, 3]), shape: [3] });
const st = safetensorsBytes(tensors, { arch: 'mlp-tiny', step: 12, tier: 'web-demo' });
writeFileSync(join(dir, 'weights.safetensors'), st);
const back = parseSafetensors(st);
if (back.tensors['backbone.net.0.weight'].data[5] !== params[0].W[5]) throw new Error('roundtrip mismatch');

const emb = new Float32Array(7 * 5).map((_, i) => i * 0.5 - 3);
writeFileSync(join(dir, 'emb.npy'), npyBytes(emb, [7, 5]));
writeFileSync(join(dir, 'vec.npy'), npyBytes(Float32Array.from([1.5, -2, 3.25]), [3]));

const bin = new Uint8Array(1000).map((_, i) => (i * 37) & 0xff);
const zip = zipBytes([
  { name: 'config.json', data: '{"version": 1}\n' },
  { name: 'README.txt', data: 'héllo — unicode ✓\n' },
  { name: 'weights.safetensors', data: st },
  { name: 'data/bin.dat', data: bin },
  { name: 'empty.txt', data: '' },
]);
writeFileSync(join(dir, 'bundle.zip'), zip);

const reference = {};
for (const t of tensors) reference[t.name] = { shape: t.shape, head: Array.from(t.data.slice(0, 4)) };
writeFileSync(join(dir, 'expected.json'), JSON.stringify({ tensors: reference, emb: Array.from(emb) }));
console.log('ok');
