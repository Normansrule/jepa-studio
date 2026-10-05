// Minimal NumPy .npy writer (format version 1.0), float32 little-endian, C order.
// Header: magic "\x93NUMPY", version 1 0, uint16 header length, then an ASCII dict padded with
// spaces and ending in "\n" so that the data starts on a 64-byte boundary.

export function npyBytes(data, shape) {
  const f32 = data instanceof Float32Array ? data : Float32Array.from(data);
  const count = shape.reduce((a, b) => a * b, 1);
  if (count !== f32.length) throw new Error(`npy: shape ${shape} needs ${count} values, got ${f32.length}`);
  const shapeStr = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': ${shapeStr}, }`;
  const pre = 10;
  const total = Math.ceil((pre + header.length + 1) / 64) * 64;
  header = header + ' '.repeat(total - pre - header.length - 1) + '\n';
  const out = new Uint8Array(total + f32.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  out[8] = header.length & 0xff;
  out[9] = (header.length >> 8) & 0xff;
  for (let i = 0; i < header.length; i++) out[pre + i] = header.charCodeAt(i);
  const le = new DataView(out.buffer, total);
  for (let i = 0; i < f32.length; i++) le.setFloat32(i * 4, f32[i], true);
  return out;
}
