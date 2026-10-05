// Minimal safetensors writer: 8-byte little-endian header length N, N bytes of UTF-8 JSON
// {"__metadata__": {str: str}, name: {dtype: "F32", shape: [...], data_offsets: [start, end]}},
// then the raw little-endian tensor bytes. The header is space-padded to a multiple of 8 bytes
// (allowed by the format and what the reference implementation does).

/** tensors: [{name, data: Float32Array, shape: number[]}], metadata: {key: string}. */
export function safetensorsBytes(tensors, metadata = {}) {
  const header = {};
  const meta = {};
  for (const [k, v] of Object.entries(metadata)) meta[k] = String(v);
  if (Object.keys(meta).length) header.__metadata__ = meta;
  let offset = 0;
  const sorted = [...tensors].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  for (const t of sorted) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (n !== t.data.length) throw new Error(`safetensors: ${t.name} shape ${t.shape} != ${t.data.length} values`);
    if (t.name === '__metadata__') throw new Error('reserved tensor name');
    header[t.name] = { dtype: 'F32', shape: t.shape, data_offsets: [offset, offset + n * 4] };
    offset += n * 4;
  }
  let json = new TextEncoder().encode(JSON.stringify(header));
  const padded = Math.ceil(json.length / 8) * 8;
  if (padded !== json.length) {
    const p = new Uint8Array(padded).fill(0x20);
    p.set(json);
    json = p;
  }
  const out = new Uint8Array(8 + json.length + offset);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, json.length, true);
  dv.setUint32(4, 0, true);
  out.set(json, 8);
  let pos = 8 + json.length;
  for (const t of sorted) {
    for (let i = 0; i < t.data.length; i++, pos += 4) dv.setFloat32(pos, t.data[i], true);
  }
  return out;
}

/** Parse our own output back (used by Open weights and by tests). F32 only. */
export function parseSafetensors(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = dv.getUint32(0, true);
  if (dv.getUint32(4, true) !== 0 || n > 100 * 1024 * 1024 || 8 + n > bytes.length) throw new Error('not a safetensors file');
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + n)));
  const out = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    if (info.dtype !== 'F32') throw new Error(`${name}: only F32 supported in the browser`);
    const [s, e] = info.data_offsets;
    const data = new Float32Array((e - s) / 4);
    for (let i = 0; i < data.length; i++) data[i] = dv.getFloat32(8 + n + s + i * 4, true);
    out[name] = { shape: info.shape, data };
  }
  return { tensors: out, metadata: header.__metadata__ || {} };
}
