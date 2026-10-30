// Accuracy of the browser physics in site/js/science/sims.js (run by tests/test_science_web.py).
// Prints one JSON object of measured numbers; the Python side asserts the bounds.
import { createNBody, nbodyStep, nbodyEnergy, pendulumRHS, pendulumEnergy, rk4, createWave, waveStep, waveEnergy,
  wavePulse, courant, keplerPosition, lorenzRHS } from '../../site/js/science/sims.js';
import { chromaticity, blackbodyTable } from '../../site/js/science/color.js';
import { fluxPT, shadowAngle } from '../../site/js/science/blackhole.js';

const out = {};

// figure-eight three-body orbit: one period, velocity Verlet dt = 1e-3
{
  const s = createNBody(3);
  const x0 = s.x.map((p) => [...p]);
  const E0 = nbodyEnergy(s);
  let worst = 0;
  const T = 6.32591398, dt = 1e-3, n = Math.round(T / dt);
  for (let i = 0; i < n; i++) { nbodyStep(s, dt); if (i % 50 === 0) worst = Math.max(worst, Math.abs((nbodyEnergy(s) - E0) / E0)); }
  out.figure8_energy_drift = worst;
  out.figure8_return_error = Math.max(...s.x.flatMap((p, i) => [Math.abs(p[0] - x0[i][0]), Math.abs(p[1] - x0[i][1])]));
  const c = createNBody(40, 3);
  const Ec = nbodyEnergy(c);
  for (let i = 0; i < 1500; i++) nbodyStep(c, 2e-3);
  out.cluster_energy_drift = Math.abs((nbodyEnergy(c) - Ec) / Ec);
}

// double pendulum: energy with RK4, divergence of a twin started 1e-9 rad away
{
  let a = [2.0944, -0.349, 0, 0], b = [2.0944 + 1e-9, -0.349, 0, 0];
  const E0 = pendulumEnergy(a);
  let worst = 0;
  const dt = 1e-3;
  for (let i = 0; i < 12000; i++) {
    a = rk4((y) => pendulumRHS(y), a, dt); b = rk4((y) => pendulumRHS(y), b, dt);
    if (i % 100 === 0) worst = Math.max(worst, Math.abs((pendulumEnergy(a) - E0) / E0));
  }
  out.pendulum_energy_drift = worst;
  out.pendulum_separation = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]);
}

// waves: CFL guard, exact discrete energy with fixed edges, front speed
{
  try { createWave(64, 64, { cfl: 1.2 }); out.cfl_guard = false; } catch (e) { out.cfl_guard = e instanceof RangeError; }
  const w = createWave(96, 96, { cfl: 0.5 });
  wavePulse(w, 48, 48, 3);
  waveStep(w);
  const E1 = waveEnergy(w);
  let worst = 0;
  for (let i = 0; i < 200; i++) { waveStep(w); worst = Math.max(worst, Math.abs(waveEnergy(w) - E1) / E1); }
  out.wave_energy_variation = worst;
  out.wave_courant = courant(1, w.dt, w.dx);
  const v = createWave(161, 161, { cfl: 0.5 });
  wavePulse(v, 80, 80, 3.2);
  const steps = Math.round(0.3 / v.dt);
  for (let i = 0; i < steps; i++) waveStep(v);
  let best = 0, at = 0;
  for (let i = 80; i < 161; i++) { const val = Math.abs(v.u[80 * 161 + i]); if (val > best) { best = val; at = i; } }
  out.wave_front_radius = (at - 80) * v.dx;
  out.wave_front_expected = v.t;
}

// Kepler: analytic solution sanity and Verlet against it
{
  const e = 0.6, dt = 1e-3;
  let x = [1 - e, 0], vv = [0, Math.sqrt((1 + e) / (1 - e))], t = 0, worst = 0;
  const acc = (p) => { const r = Math.hypot(...p); return [-p[0] / r ** 3, -p[1] / r ** 3]; };
  let a = acc(x);
  for (let i = 0; i < Math.round((2 * Math.PI) / dt); i++) {
    vv = [vv[0] + 0.5 * dt * a[0], vv[1] + 0.5 * dt * a[1]];
    x = [x[0] + dt * vv[0], x[1] + dt * vv[1]];
    a = acc(x);
    vv = [vv[0] + 0.5 * dt * a[0], vv[1] + 0.5 * dt * a[1]];
    t += dt;
    const k = keplerPosition(t, 1, e);
    worst = Math.max(worst, Math.hypot(x[0] - k[0], x[1] - k[1]));
  }
  out.kepler_position_error = worst;
  out.kepler_apoapsis = keplerPosition(Math.PI, 1, e);
}

// Lorenz runs and stays bounded on the attractor
{
  let s = [1, 1, 20];
  for (let i = 0; i < 4000; i++) s = rk4(lorenzRHS, s, 0.005);
  out.lorenz_state = s;
}

// colour + black-hole helpers shared with the shader
out.chromaticity = Object.fromEntries([3000, 5000, 6500, 10000].map((T) => [T, chromaticity(T)]));
const tab = blackbodyTable(64, 3.0, 4.7);
out.table_6500 = (() => { const i = Math.round(((Math.log10(6500) - 3) / 1.7) * 63); return Array.from(tab.slice(4 * i, 4 * i + 4)); })();
out.fluxPT = [7, 10, 20, 50].map((r) => fluxPT(r));
out.shadow_30 = shadowAngle(30);
process.stdout.write(JSON.stringify(out));
