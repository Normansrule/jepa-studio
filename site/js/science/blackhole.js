// Live black hole for the browser: a WebGL2 fragment shader traces one light ray per pixel through
// the curved space around a NON-SPINNING (Schwarzschild, a = 0) black hole, in real time.
//
// Physics (units G = c = M = 1; see docs/jev/science.md):
//   * Rays: the photon orbit equation u'' + u = 3 M u^2 (u = 1/r) is integrated in Cartesian form,
//     d^2x/dl^2 = -3 M h^2 x / r^5 with h = |x x dx/dl| conserved, by RK4 with steps proportional to r.
//     This reproduces Schwarzschild null geodesics exactly (only the parametrisation differs).
//   * Camera: a static observer at radius r; local directions are converted to coordinate directions
//     (cot psi_coord = sqrt(1 - 2M/r) cot psi_local), so the shadow has the exact finite-distance size
//     sin(alpha) = 3 sqrt(3) M sqrt(1 - 2M/r) / r.
//   * Disk: thin, opaque, in the plane y = 0 from the ISCO (6M) outwards, Page-Thorne temperature
//     profile for a = 0, Keplerian gas. g = nu_obs/nu_emit = [1/sqrt(1-2M/r_cam)] / [u^t (1 - Omega L_y)],
//     colour and brightness of a blackbody at g T (toggles: doppler, beaming, redshift).
//   * Sky: procedural stars (point sources whose brightness follows the lens magnification, from the
//     screen-space derivatives of the escape direction), a faint band, or a checker grid.
// Not in the browser version (the desktop renderer, jev/science/blackhole.py, has them): spin / frame
// dragging, light-travel-time delays, supersampling. The disk can still rotate (animated pattern).
//
// API: mountBlackHole(canvas, options) -> {setParams, getParams, render, snapshot, destroy, capabilities}
// Throws an Error mentioning WebGL2 when it is unavailable (or draws a message with options.fallback =
// 'message'). No DOM is created; the caller owns the canvas and its size (canvas.width / height).
import { blackbodyTable } from './color.js';

export const capabilities = Object.freeze({
  spin: false,                 // a = 0 only in the browser; spin is rendered by the desktop backend
  metric: 'schwarzschild',
  disk: true, doppler: true, beaming: true, redshift: true,
  backgrounds: ['stars', 'grid', 'none'],
  qualities: ['low', 'medium', 'high'],
  lightTravelTime: false,
  note: 'Live view: non-spinning black hole (Schwarzschild), traced per pixel on the GPU.',
});
export const BLACKHOLE_CAPABILITIES = capabilities;

export const DEFAULTS = Object.freeze({
  distance: 28, inclination: 85, azimuth: 0, fov: 55,
  disk: true, diskInner: 6, diskOuter: 16, temperature: 6000,
  doppler: true, beaming: true, redshift: true, interstellar: false,
  background: 'stars', quality: 'medium', exposure: 1.0,
  autoRotate: false, animateDisk: true, timeScale: 20, reducedMotion: false,
});

const QUALITY = {
  low: { scale: 0.5, steps: 220, h: 1.6 },
  medium: { scale: 0.75, steps: 360, h: 1.0 },
  high: { scale: 1.0, steps: 700, h: 0.5 },
};

const BB_N = 64, BB_LOGT0 = 3.0, BB_LOGT1 = 4.7;

const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform vec2 uRes;
uniform vec3 uCamPos;      // camera position (hole at the origin, disk normal +y)
uniform vec3 uRight, uUp, uFwd;
uniform float uTanHalf;    // tan(horizontal fov / 2)
uniform float uTime;       // coordinate time (units of M) for the disk pattern
uniform int uMaxSteps;
uniform float uStepScale;
uniform float uRIn, uROut, uRFar;
uniform int uDisk, uDoppler, uBeaming, uRedshift, uBackground;
uniform float uTemp, uFmax, uExposure;
uniform vec4 uBB[${BB_N}];

const float M = 1.0;

float fluxPT(float r) {
  // Page & Thorne (1974) flux for a = 0 (zero torque at r = 6M), up to a constant factor
  float x = sqrt(r), x0 = sqrt(6.0), s3 = sqrt(3.0);
  float br = x - x0 - 0.5 * s3 * log((x - s3) / (x0 - s3)) + 0.5 * s3 * log((x + s3) / (x0 + s3));
  return max(br, 0.0) / (x * x * x * x * (x * x * x - 3.0 * x));
}

vec4 bb(float T) {
  float u = clamp((log(T) / log(10.0) - ${BB_LOGT0.toFixed(1)}) / ${(BB_LOGT1 - BB_LOGT0).toFixed(2)}, 0.0, 1.0) * float(${BB_N - 1});
  int i = int(floor(u));
  i = min(i, ${BB_N - 2});
  float f = u - float(i);
  return mix(uBB[i], uBB[i + 1], f);
}

float hash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}

float vnoise(float x, float seed) {
  float i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(hash13(vec3(i, seed, 7.0)), hash13(vec3(i + 1.0, seed, 7.0)), f);
}

float diskPattern(float r, float psi) {
  float u = log(r);
  float ring = 0.0, amp = 0.5, fr = 9.0;
  for (int o = 0; o < 4; o++) { ring += amp * vnoise(u * fr + 17.3 * float(o), float(o)); amp *= 0.55; fr *= 2.3; }
  ring /= 0.9;
  float az = 0.55 * sin(2.0 * psi + 1.3 + 1.7 * u) + 0.4 * sin(3.0 * psi + 4.1 - 2.3 * u)
           + 0.25 * sin(5.0 * psi + 0.7 + 0.9 * u) + 0.15 * sin(8.0 * psi + 2.9 - 1.1 * u);
  return clamp(0.5 + 0.5 * ring * (1.0 + 0.3 * az), 0.05, 2.0);
}

// Stars are point sources: the offset of the star from this pixel's ray is expressed in pixel units by
// solving delta = alpha * dFdx(dir) + beta * dFdy(dir) (2x2 normal equations), so every star image is
// about one pixel wide however the lens stretches the sky; its brightness follows the magnification.
vec3 starLayer(vec3 d, vec3 ex, vec3 ey, float cells, float bright, float prob, float seed, float mag,
               float footprint, float pixAngle) {
  vec3 a = abs(d);
  float m = max(a.x, max(a.y, a.z));
  vec3 face; vec2 uv;
  if (m == a.x) { face = vec3(sign(d.x), 0.0, 0.0); uv = d.yz / a.x; }
  else if (m == a.y) { face = vec3(0.0, sign(d.y), 0.0); uv = d.zx / a.y; }
  else { face = vec3(0.0, 0.0, sign(d.z)); uv = d.xy / a.z; }
  vec2 cell = floor((uv * 0.5 + 0.5) * cells);
  vec3 key = vec3(cell, dot(face, vec3(1.0, 2.0, 3.0)) + seed * 7.0);
  float h1 = hash13(key + 0.11), h2 = hash13(key + 5.3), h3 = hash13(key + 9.7), h4 = hash13(key + 13.1);
  vec2 suv = ((cell + 0.2 + 0.6 * vec2(h1, h2)) / cells) * 2.0 - 1.0;
  vec3 sd;
  if (m == a.x) sd = vec3(face.x, suv.x, suv.y);
  else if (m == a.y) sd = vec3(suv.y, face.y, suv.x);
  else sd = vec3(suv.x, suv.y, face.z);
  sd = normalize(sd);
  vec3 dl = sd - d;
  float fl = 0.05 * pixAngle * pixAngle;                     // floor: magnification cap near caustics
  float g11 = dot(ex, ex) + fl, g22 = dot(ey, ey) + fl, g12 = dot(ex, ey);
  float det = max(g11 * g22 - g12 * g12, 1e-30);
  float r1 = dot(dl, ex), r2 = dot(dl, ey);
  float al = (g22 * r1 - g12 * r2) / det, be = (g11 * r2 - g12 * r1) / det;
  float px2 = al * al + be * be;                              // squared distance in pixels
  float flux = h3 > prob ? 0.0 : bright * (pow(h4, 6.0) * 30.0 + 0.15);
  float T = 2500.0 + 22500.0 * pow(hash13(key + 21.7), 2.5);
  float w = flux * exp(-0.5 * px2 / 0.36) * mag;
  // footprint >> cell (strong demagnification by the shadow): the layer's mean surface brightness
  float cellArea = 4.0 / (cells * cells);
  float mean = prob * bright * (30.0 / 7.0 + 0.15) * 6.2832 * 0.36 * pixAngle * pixAngle / cellArea;
  w = mix(w, mean, clamp((footprint / cellArea - 0.05) / 0.45, 0.0, 1.0));
  return w * bb(T).rgb;
}

vec3 sky(vec3 d, vec3 ex, vec3 ey, float footprint, float pixAngle) {
  if (uBackground == 2) return vec3(0.0);
  if (uBackground == 1) {
    float th = acos(clamp(d.y, -1.0, 1.0));
    float ph = atan(d.z, d.x);
    float s = radians(10.0);
    float chk = mod(floor(th / s) + floor(ph / s), 2.0);
    vec3 c = mix(vec3(0.08, 0.10, 0.16), vec3(0.20, 0.26, 0.40), chk);
    c *= ph > 0.0 ? vec3(1.0, 0.85, 0.6) : vec3(0.6, 0.85, 1.0);
    vec2 f = abs(vec2(th, ph) / s - floor(vec2(th, ph) / s + 0.5));
    if (min(f.x, f.y * max(sin(th), 0.05)) < 0.03) c = vec3(0.55, 0.6, 0.7);
    return c * 0.6;
  }
  float mag = clamp(pixAngle * pixAngle / max(footprint, 1e-12), 0.05, 20.0);
  vec3 c = starLayer(d, ex, ey, 120.0, 0.5, 0.35, 0.0, mag, footprint, pixAngle)
         + starLayer(d, ex, ey, 40.0, 2.0, 0.5, 1.0, mag, footprint, pixAngle)
         + starLayer(d, ex, ey, 13.0, 8.0, 0.6, 2.0, mag, footprint, pixAngle);
  vec3 n = normalize(vec3(0.35, 0.9, -0.25));
  float z = dot(d, n);
  float lon = atan(d.z, d.x);
  c += exp(-(z / 0.18) * (z / 0.18)) * (0.6 + 0.25 * sin(3.0 * lon + 1.3) + 0.15 * sin(7.0 * lon + 20.0 * z))
       * 0.005 * vec3(1.0, 0.86, 0.72);
  return c;
}

vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
vec3 srgb(vec3 x) {
  x = clamp(x, 0.0, 1.0);
  return mix(12.92 * x, 1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, x));
}

void main() {
  vec2 p = (vUv * 2.0 - 1.0) * vec2(1.0, uRes.y / uRes.x) * uTanHalf;
  vec3 n = normalize(uFwd + p.x * uRight + p.y * uUp);       // local look direction (static observer)
  float rc = length(uCamPos);
  vec3 rhat = uCamPos / rc;
  float nr = dot(n, rhat);
  vec3 nt = n - nr * rhat;
  float lapse = sqrt(1.0 - 2.0 * M / rc);
  vec3 v = normalize(nr * lapse * rhat + nt);                 // coordinate direction (aberration)
  float b = rc * length(nt) / lapse;                          // impact parameter of this ray
  vec3 x = uCamPos;
  vec3 L = cross(x, v);
  float h2 = dot(L, L);
  vec3 Lhat = h2 > 0.0 ? L / sqrt(h2) : vec3(0.0, 1.0, 0.0);
  float Ly = -b * Lhat.y;                                     // photon angular momentum about the disk axis

  vec3 col = vec3(0.0);
  bool escaped = false;
  vec3 skyDir = v;
  for (int i = 0; i < 4096; i++) {
    if (i >= uMaxSteps) break;
    float r = length(x);
    float dt = uStepScale * 0.07 * r * clamp((r - 2.0) / 1.5, 0.15, 1.0);
    // RK4 on (x, v) with a(x) = -3 M h^2 x / r^5
    vec3 k1v = -3.0 * M * h2 * x / pow(r, 5.0);
    vec3 x2 = x + 0.5 * dt * v;       vec3 v2 = v + 0.5 * dt * k1v;
    vec3 k2v = -3.0 * M * h2 * x2 / pow(length(x2), 5.0);
    vec3 x3 = x + 0.5 * dt * v2;      vec3 v3 = v + 0.5 * dt * k2v;
    vec3 k3v = -3.0 * M * h2 * x3 / pow(length(x3), 5.0);
    vec3 x4 = x + dt * v3;            vec3 v4 = v + dt * k3v;
    vec3 k4v = -3.0 * M * h2 * x4 / pow(length(x4), 5.0);
    vec3 xn = x + dt / 6.0 * (v + 2.0 * v2 + 2.0 * v3 + v4);
    vec3 vn = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    if (uDisk == 1 && x.y * xn.y < 0.0) {
      float t = x.y / (x.y - xn.y);
      vec3 q = mix(x, xn, t);
      float rq = length(q);
      if (rq >= uRIn && rq <= uROut) {
        float ut = 1.0 / sqrt(1.0 - 3.0 * M / rq);
        float Om = sqrt(M / (rq * rq * rq));
        float g = (1.0 / lapse) / (ut * (1.0 - Om * Ly));
        float gg = sqrt(1.0 - 2.0 * M / rq) / lapse;
        float gd = g / gg;
        float T = uTemp * pow(max(fluxPT(rq), 0.0) / uFmax, 0.25);
        float gc = (uRedshift == 1 ? gg : 1.0) * (uDoppler == 1 ? gd : 1.0);
        float gl = (uRedshift == 1 ? gg : 1.0) * (uBeaming == 1 ? gd : 1.0);
        vec4 cc = bb(max(T * gc, 500.0));
        float lum = pow(10.0, bb(max(T * gl, 500.0)).a);
        float phi = atan(-q.z, q.x);
        float psi = phi - Om * uTime;
        float edge = clamp((uROut - rq) / (0.15 * uROut), 0.0, 1.0);
        col = cc.rgb * lum * diskPattern(rq, psi) * edge * (T > 1.0 ? 1.0 : 0.0);
        break;
      }
    }
    x = xn; v = vn;
    float rn = length(x);
    if (rn < 2.0 * M * 1.005) break;                          // captured: black
    if (rn > uRFar && dot(x, v) > 0.0) { escaped = true; skyDir = normalize(v); break; }
  }
  // derivatives outside any branch: the pixel's footprint on the sky (lens magnification)
  vec3 ddx = dFdx(skyDir), ddy = dFdy(skyDir);
  float footprint = length(cross(ddx, ddy));
  float pixAngle = 2.0 * uTanHalf / uRes.x;
  if (escaped) col += sky(skyDir, ddx, ddy, footprint, pixAngle);
  outColor = vec4(srgb(aces(col * uExposure)), 1.0);
}`;

const BLIT_FRAG = `#version 300 es
precision mediump float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uTex;
void main() { outColor = texture(uTex, vUv); }`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error(`WebGL2 shader failed to compile: ${log}`);
  }
  return s;
}

function program(gl, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, 'aPos');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`WebGL2 program failed to link: ${gl.getProgramInfoLog(p)}`);
  return p;
}

/** Page-Thorne flux for a = 0 (same expression as the shader), for the temperature normalisation. */
export function fluxPT(r) {
  const x = Math.sqrt(r), x0 = Math.sqrt(6), s3 = Math.sqrt(3);
  const br = x - x0 - 0.5 * s3 * Math.log((x - s3) / (x0 - s3)) + 0.5 * s3 * Math.log((x + s3) / (x0 + s3));
  return Math.max(br, 0) / (x ** 4 * (x ** 3 - 3 * x));
}

/** Angular radius of the shadow seen by a static observer at r (units of M): exact for a = 0. */
export function shadowAngle(r) {
  return Math.asin(Math.min(1, (3 * Math.sqrt(3) * Math.sqrt(1 - 2 / r)) / r));
}

export function isSupported() {
  try {
    const c = (typeof OffscreenCanvas !== 'undefined') ? new OffscreenCanvas(1, 1) : document.createElement('canvas');
    return !!c.getContext('webgl2');
  } catch { return false; }
}

function normalize(p) {
  const q = { ...p };
  q.distance = Math.min(500, Math.max(4, Number(q.distance) || DEFAULTS.distance));
  q.inclination = Math.min(179, Math.max(1, Number(q.inclination)));
  if (!Number.isFinite(q.inclination)) q.inclination = DEFAULTS.inclination;
  q.azimuth = Number(q.azimuth) || 0;
  q.fov = Math.min(120, Math.max(5, Number(q.fov) || DEFAULTS.fov));
  q.diskInner = Math.max(6, Number(q.diskInner) || 6);          // ISCO of a = 0
  q.diskOuter = Math.max(q.diskInner + 0.5, Number(q.diskOuter) || DEFAULTS.diskOuter);
  q.temperature = Math.min(40000, Math.max(1000, Number(q.temperature) || DEFAULTS.temperature));
  if (!QUALITY[q.quality]) q.quality = 'medium';
  if (!['stars', 'grid', 'none'].includes(q.background)) q.background = 'stars';
  q.exposure = Math.max(0.01, Number(q.exposure) || 1);
  return q;
}

/**
 * Mount the live black hole on a canvas.
 * options: distance (M), inclination (deg from the disk axis; 90 = edge-on), azimuth (deg), fov (deg),
 *   disk, diskInner, diskOuter (M), temperature (K at the hottest ring), doppler, beaming, redshift,
 *   interstellar (true = no Doppler colour / beaming, the film's look), background ('stars'|'grid'|'none'),
 *   quality ('low'|'medium'|'high'), exposure, autoRotate, animateDisk, timeScale (M per second),
 *   reducedMotion (default: the prefers-reduced-motion media query), spin (accepted, ignored: a = 0 here),
 *   onChange(params) (called after the user orbits or zooms), fallback ('throw' | 'message').
 * Returns {setParams(p), getParams(), render(), snapshot() -> Promise<Blob>, destroy(), capabilities}.
 */
export function mountBlackHole(canvas, options = {}) {
  const mq = typeof matchMedia === 'function' ? matchMedia('(prefers-reduced-motion: reduce)') : null;
  let params = normalize({ ...DEFAULTS, reducedMotion: mq ? mq.matches : false, ...options });
  const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true, alpha: false, premultipliedAlpha: false });
  if (!gl) {
    if (options.fallback === 'message') return messageFallback(canvas);
    throw new Error('WebGL2 is not available in this browser (the live black hole needs it).');
  }
  const prog = program(gl, FRAG);
  const blit = program(gl, BLIT_FRAG);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const U = {};
  for (const name of ['uRes', 'uCamPos', 'uRight', 'uUp', 'uFwd', 'uTanHalf', 'uTime', 'uMaxSteps', 'uStepScale', 'uRIn',
    'uROut', 'uRFar', 'uDisk', 'uDoppler', 'uBeaming', 'uRedshift', 'uBackground', 'uTemp', 'uFmax', 'uExposure', 'uBB']) {
    U[name] = gl.getUniformLocation(prog, name);
  }
  const uTex = gl.getUniformLocation(blit, 'uTex');
  gl.useProgram(prog);
  gl.uniform4fv(U.uBB, blackbodyTable(BB_N, BB_LOGT0, BB_LOGT1));

  let fbo = null, tex = null, fboW = 0, fboH = 0;
  let raf = 0, last = 0, simTime = 0, destroyed = false, lost = false;
  const pointers = new Map();
  let pinch0 = 0, dist0 = 0;

  function fmax() {
    let m = 0;
    for (let r = params.diskInner; r <= Math.max(params.diskOuter, params.diskInner + 1); r += 0.02) m = Math.max(m, fluxPT(r));
    return m || 1;
  }
  let Fmax = fmax();

  function camera() {
    const inc = (params.inclination * Math.PI) / 180, az = (params.azimuth * Math.PI) / 180;
    const r = params.distance;
    const pos = [r * Math.sin(inc) * Math.cos(az), r * Math.cos(inc), r * Math.sin(inc) * Math.sin(az)];
    const fwd = pos.map((c) => -c / r);
    // right = fwd x worldUp, up = right x fwd (spin / disk axis appears up)
    let right = [fwd[1] * 0 - fwd[2] * 1, fwd[2] * 0 - fwd[0] * 0, fwd[0] * 1 - fwd[1] * 0];
    const rl = Math.hypot(...right) || 1;
    right = right.map((c) => c / rl);
    const up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]];
    return { pos, fwd, right, up };
  }

  function ensureTarget(w, h) {
    if (fbo && fboW === w && fboH === h) return;
    if (fbo) { gl.deleteFramebuffer(fbo); gl.deleteTexture(tex); }
    tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    fboW = w; fboH = h;
  }

  function render() {
    if (destroyed || lost) return;
    const W = canvas.width || 1, Hh = canvas.height || 1;
    const q = QUALITY[params.quality];
    const w = Math.max(1, Math.round(W * q.scale)), h = Math.max(1, Math.round(Hh * q.scale));
    const off = q.scale < 1;
    if (off) { ensureTarget(w, h); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); } else gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, off ? w : W, off ? h : Hh);
    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    const cam = camera();
    const noDop = params.interstellar;
    gl.uniform2f(U.uRes, off ? w : W, off ? h : Hh);
    gl.uniform3fv(U.uCamPos, cam.pos);
    gl.uniform3fv(U.uRight, cam.right);
    gl.uniform3fv(U.uUp, cam.up);
    gl.uniform3fv(U.uFwd, cam.fwd);
    gl.uniform1f(U.uTanHalf, Math.tan((params.fov * Math.PI) / 360));
    gl.uniform1f(U.uTime, simTime);
    gl.uniform1i(U.uMaxSteps, q.steps);
    gl.uniform1f(U.uStepScale, q.h);
    gl.uniform1f(U.uRIn, params.diskInner);
    gl.uniform1f(U.uROut, params.diskOuter);
    gl.uniform1f(U.uRFar, Math.max(1.2 * params.distance, 2 * params.diskOuter, 50));
    gl.uniform1i(U.uDisk, params.disk ? 1 : 0);
    gl.uniform1i(U.uDoppler, params.doppler && !noDop ? 1 : 0);
    gl.uniform1i(U.uBeaming, params.beaming && !noDop ? 1 : 0);
    gl.uniform1i(U.uRedshift, params.redshift ? 1 : 0);
    gl.uniform1i(U.uBackground, { stars: 0, grid: 1, none: 2 }[params.background]);
    gl.uniform1f(U.uTemp, params.temperature);
    gl.uniform1f(U.uFmax, Fmax);
    // exposure: the disk's visible luminance at the hottest ring sets the scale (as the desktop auto-exposure)
    gl.uniform1f(U.uExposure, params.exposure * 1.6);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    if (off) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, W, Hh);
      gl.useProgram(blit);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(uTex, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }

  const animating = () => !params.reducedMotion && (params.autoRotate || (params.animateDisk && params.disk));
  function frame(t) {
    raf = 0;
    if (destroyed) return;
    const dt = last ? Math.min(0.1, (t - last) / 1000) : 0;
    last = t;
    if (!params.reducedMotion) {
      if (params.animateDisk) simTime += dt * params.timeScale;
      if (params.autoRotate) params.azimuth = (params.azimuth + dt * 6) % 360;
    }
    render();
    if (animating()) raf = requestAnimationFrame(frame);
  }
  function kick() {
    if (destroyed) return;
    if (animating()) { if (!raf) { last = 0; raf = requestAnimationFrame(frame); } } else if (!raf) raf = requestAnimationFrame(frame);
  }

  function changed() { if (typeof params.onChange === 'function') params.onChange(getParams()); kick(); }

  // ------------------------------------------------------------------ interaction
  const onDown = (e) => { canvas.setPointerCapture && canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]); if (pointers.size === 2) { pinch0 = spread(); dist0 = params.distance; } };
  const spread = () => { const [a, b] = [...pointers.values()]; return Math.hypot(a[0] - b[0], a[1] - b[1]) || 1; };
  const onMove = (e) => {
    if (!pointers.has(e.pointerId)) return;
    const prev = pointers.get(e.pointerId);
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (pointers.size === 2) { params.distance = clampDist(dist0 * pinch0 / spread()); changed(); return; }
    params.azimuth = (params.azimuth + (e.clientX - prev[0]) * 0.4) % 360;
    params.inclination = Math.min(179, Math.max(1, params.inclination - (e.clientY - prev[1]) * 0.3));
    changed();
  };
  const onUp = (e) => { pointers.delete(e.pointerId); };
  const onWheel = (e) => { e.preventDefault(); params.distance = clampDist(params.distance * Math.exp(e.deltaY * 0.001)); changed(); };
  const onKey = (e) => {
    const k = e.key;
    if (k === 'ArrowLeft') params.azimuth -= 5;
    else if (k === 'ArrowRight') params.azimuth += 5;
    else if (k === 'ArrowUp') params.inclination = Math.max(1, params.inclination - 3);
    else if (k === 'ArrowDown') params.inclination = Math.min(179, params.inclination + 3);
    else if (k === '+' || k === '=') params.distance = clampDist(params.distance / 1.1);
    else if (k === '-' || k === '_') params.distance = clampDist(params.distance * 1.1);
    else if (k === ' ' && !params.reducedMotion) params.autoRotate = !params.autoRotate;
    else return;
    e.preventDefault();
    changed();
  };
  const clampDist = (d) => Math.min(500, Math.max(4, d));
  const onLost = (e) => { e.preventDefault(); lost = true; if (raf) cancelAnimationFrame(raf); raf = 0; };
  const onMq = () => { params.reducedMotion = mq.matches; kick(); };
  if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
  if (!canvas.style.touchAction) canvas.style.touchAction = 'none';
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('keydown', onKey);
  canvas.addEventListener('webglcontextlost', onLost);
  if (mq && !('reducedMotion' in options) && mq.addEventListener) mq.addEventListener('change', onMq);

  function setParams(p = {}) {
    const old = params;
    params = normalize({ ...params, ...p });
    if (old.diskInner !== params.diskInner || old.diskOuter !== params.diskOuter) Fmax = fmax();
    kick();
  }
  function getParams() {
    const { onChange, ...rest } = params;   // eslint-disable-line no-unused-vars
    return rest;
  }
  function snapshot() {
    render();
    return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('no image'))), 'image/png'));
  }
  function destroy() {
    destroyed = true;
    if (raf) cancelAnimationFrame(raf);
    canvas.removeEventListener('pointerdown', onDown);
    canvas.removeEventListener('pointermove', onMove);
    canvas.removeEventListener('pointerup', onUp);
    canvas.removeEventListener('pointercancel', onUp);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('keydown', onKey);
    canvas.removeEventListener('webglcontextlost', onLost);
    if (mq && mq.removeEventListener) mq.removeEventListener('change', onMq);
    if (fbo) { gl.deleteFramebuffer(fbo); gl.deleteTexture(tex); }
    gl.deleteBuffer(buf);
    gl.deleteVertexArray(vao);
    gl.deleteProgram(prog);
    gl.deleteProgram(blit);
  }

  render();
  kick();
  return { setParams, getParams, render, snapshot, destroy, capabilities, supported: true };
}

function messageFallback(canvas) {
  const ctx = canvas.getContext('2d');
  const draw = () => {
    if (!ctx) return;
    const css = getComputedStyle(document.documentElement);
    ctx.fillStyle = css.getPropertyValue('--surface-2').trim() || '#232e45';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = css.getPropertyValue('--ink-2').trim() || '#b4bccb';
    ctx.font = `${Math.max(12, canvas.width / 40)}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('This live view needs WebGL2, which this browser does not provide.', canvas.width / 2, canvas.height / 2);
  };
  draw();
  const none = () => {};
  return {
    setParams: none, getParams: () => ({}), render: draw, destroy: none, capabilities, supported: false,
    snapshot: () => new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('no image'))), 'image/png')),
  };
}
