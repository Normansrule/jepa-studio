// Optional WebGPU acceleration for the two large products of the first layer:
//   forward   Y  = X W^T + b      X (n, din), W (dout, din)
//   gradient  dW = dY^T X         dY (n, dout)
// Everything else stays on the CPU. On init the kernels are checked against the JavaScript
// result on a random problem; if they disagree (or anything throws) the engine stays on CPU.
import { linearForward } from './tensor.js';

const TILE = 16;

const FORWARD_WGSL = /* wgsl */`
struct Dims { n: u32, din: u32, dout: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
var<workgroup> xs: array<array<f32, ${TILE}>, ${TILE}>;
var<workgroup> ws: array<array<f32, ${TILE}>, ${TILE}>;
@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let row = g.y; let col = g.x;
  var acc = 0.0;
  let tiles = (d.din + ${TILE}u - 1u) / ${TILE}u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let kx = t * ${TILE}u + l.x;
    xs[l.y][l.x] = select(0.0, X[row * d.din + kx], row < d.n && kx < d.din);
    let wr = ${TILE}u * (g.x / ${TILE}u) + l.y;
    ws[l.y][l.x] = select(0.0, W[wr * d.din + kx], wr < d.dout && kx < d.din);
    workgroupBarrier();
    for (var k = 0u; k < ${TILE}u; k = k + 1u) { acc = acc + xs[l.y][k] * ws[l.x][k]; }
    workgroupBarrier();
  }
  if (row < d.n && col < d.dout) { Y[row * d.dout + col] = acc + B[col]; }
}`;

// dW[o, i] = sum_r dY[r, o] * X[r, i]
const GRADW_WGSL = /* wgsl */`
struct Dims { n: u32, din: u32, dout: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> DW: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;
var<workgroup> ys: array<array<f32, ${TILE}>, ${TILE}>;
var<workgroup> xs: array<array<f32, ${TILE}>, ${TILE}>;
@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let o = g.y; let i = g.x;
  var acc = 0.0;
  let tiles = (d.n + ${TILE}u - 1u) / ${TILE}u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let r = t * ${TILE}u + l.x;
    let oo = ${TILE}u * (g.y / ${TILE}u) + l.y;
    ys[l.y][l.x] = select(0.0, DY[r * d.dout + oo], r < d.n && oo < d.dout);
    let r2 = t * ${TILE}u + l.y;
    xs[l.y][l.x] = select(0.0, X[r2 * d.din + i], r2 < d.n && i < d.din);
    workgroupBarrier();
    for (var k = 0u; k < ${TILE}u; k = k + 1u) { acc = acc + ys[l.y][k] * xs[k][l.x]; }
    workgroupBarrier();
  }
  if (o < d.dout && i < d.din) { DW[o * d.din + i] = acc; }
}`;

export class GpuMatmul {
  static async create(gpu) {
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const g = new GpuMatmul(device, adapter);
    g.fallbackAdapter = !!(adapter.isFallbackAdapter || (adapter.info && adapter.info.isFallbackAdapter));
    const ok = await g.selfTest();
    if (!ok) { device.destroy?.(); return { rejected: true, reason: `the WebGPU kernels failed their self-test (${g.selfTestError})`, info: g.info }; }
    const speed = await g.benchmark();
    g.speed = speed;
    if (speed.gpuMs >= speed.cpuMs) {
      device.destroy?.();
      return { rejected: true, reason: `WebGPU adapter "${[g.info.vendor, g.info.architecture].filter(Boolean).join(' ') || 'unnamed'}" works but was slower than JavaScript on this layer (${speed.gpuMs.toFixed(1)} ms vs ${speed.cpuMs.toFixed(1)} ms)${g.fallbackAdapter ? '; it is a software fallback adapter' : ''}`, info: g.info, speed };
    }
    return g;
  }

  /** Time the first-layer product (192 x 768 -> 256, a batch of 32 with 6 views) on both paths. */
  async benchmark() {
    const n = 192, din = 768, dout = 256;
    const X = Float32Array.from({ length: n * din }, (_, i) => Math.sin(i));
    const W = Float32Array.from({ length: dout * din }, (_, i) => Math.cos(i) * 0.03);
    const b = new Float32Array(dout);
    const now = () => performance.now();
    await this.linear(X, n, din, W, b, dout); // warm up pipelines
    let t = now();
    for (let i = 0; i < 3; i++) await this.linear(X, n, din, W, b, dout);
    const gpuMs = (now() - t) / 3;
    const Y = new Float32Array(n * dout);
    linearForward(X, n, din, W, b, dout, Y);
    t = now();
    for (let i = 0; i < 3; i++) linearForward(X, n, din, W, b, dout, Y);
    const cpuMs = (now() - t) / 3;
    return { gpuMs, cpuMs };
  }

  constructor(device, adapter) {
    this.device = device;
    const info = adapter.info || {};
    this.info = { vendor: info.vendor || '', architecture: info.architecture || '', description: info.description || '' };
    const mk = (code) => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    this.fwd = mk(FORWARD_WGSL);
    this.gw = mk(GRADW_WGSL);
  }

  buf(data, usage) {
    const b = this.device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  async run(pipeline, inputs, outFloats, dims, gx, gy) {
    const dev = this.device;
    const S = GPUBufferUsage.STORAGE;
    const bufs = inputs.map((a) => this.buf(a, S));
    const outSize = outFloats * 4;
    const out = dev.createBuffer({ size: Math.max(16, outSize), usage: S | GPUBufferUsage.COPY_SRC });
    const uni = this.buf(new Uint32Array(dims), GPUBufferUsage.UNIFORM);
    const read = dev.createBuffer({ size: Math.max(16, outSize), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const entries = [...bufs, out, uni].map((buffer, binding) => ({ binding, resource: { buffer } }));
    const bind = dev.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(gx / TILE), Math.ceil(gy / TILE));
    pass.end();
    enc.copyBufferToBuffer(out, 0, read, 0, Math.max(16, outSize));
    dev.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const res = new Float32Array(read.getMappedRange().slice(0, outSize));
    read.unmap();
    [...bufs, out, uni, read].forEach((b) => b.destroy());
    return res;
  }

  /** Y = X W^T + b */
  linear(X, n, din, W, b, dout) {
    return this.run(this.fwd, [X, W, b], n * dout, [n, din, dout, 0], dout, n);
  }

  /** dW += dY^T X */
  async gradW(dY, X, n, din, dout, dW) {
    const r = await this.run(this.gw, [dY, X], dout * din, [n, din, dout, 0], din, dout);
    for (let i = 0; i < r.length; i++) dW[i] += r[i];
  }

  async selfTest() {
    try {
      const n = 37, din = 70, dout = 29;
      const X = Float32Array.from({ length: n * din }, (_, i) => Math.sin(i * 0.37));
      const W = Float32Array.from({ length: dout * din }, (_, i) => Math.cos(i * 0.11));
      const b = Float32Array.from({ length: dout }, (_, i) => i * 0.01);
      const ref = linearForward(X, n, din, W, b, dout, new Float32Array(n * dout));
      const got = await this.linear(X, n, din, W, b, dout);
      let err = 0;
      for (let i = 0; i < ref.length; i++) err = Math.max(err, Math.abs(ref[i] - got[i]));
      const dY = Float32Array.from({ length: n * dout }, (_, i) => Math.sin(i * 0.05));
      const dW = new Float32Array(dout * din);
      await this.gradW(dY, X, n, din, dout, dW);
      let err2 = 0;
      for (let o = 0; o < dout; o++) for (let i = 0; i < din; i++) {
        let s = 0;
        for (let r = 0; r < n; r++) s += dY[r * dout + o] * X[r * din + i];
        err2 = Math.max(err2, Math.abs(s - dW[o * din + i]));
      }
      this.selfTestError = Math.max(err, err2);
      return this.selfTestError < 1e-3;
    } catch (e) {
      this.selfTestError = String(e);
      return false;
    }
  }
}
