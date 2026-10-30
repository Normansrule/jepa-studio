// Blackbody colours for the live black hole, the same method as jev/science/color.py:
// CIE 1931 colour-matching functions from the multi-lobe Gaussian fit of Wyman, Sloan & Shirley
// (JCGT 2(2), 2013), integrated against Planck's law, converted to linear sRGB (D65).
// No DOM access here, so Node tests can import it (tests/js/science_color.mjs).

const H = 6.62607015e-34, C = 2.99792458e8, K = 1.380649e-23;

function lobe(x, mu, s1, s2) {
  const t = (x - mu) / (x < mu ? s1 : s2);
  return Math.exp(-0.5 * t * t);
}

/** CIE 1931 2-degree colour matching functions at wavelength lam (nm). */
export function cieXYZBar(lam) {
  return [
    1.056 * lobe(lam, 599.8, 37.9, 31.0) + 0.362 * lobe(lam, 442.0, 16.0, 26.7) - 0.065 * lobe(lam, 501.1, 20.4, 26.2),
    0.821 * lobe(lam, 568.8, 46.9, 40.5) + 0.286 * lobe(lam, 530.9, 16.3, 31.1),
    1.217 * lobe(lam, 437.0, 11.8, 36.0) + 0.681 * lobe(lam, 459.0, 26.0, 13.8),
  ];
}

/** XYZ of a blackbody at temperature T (K), arbitrary but consistent scale (2 nm steps). */
export function blackbodyXYZ(T) {
  let X = 0, Y = 0, Z = 0;
  for (let lam = 360; lam <= 830; lam += 2) {
    const l = lam * 1e-9;
    const B = (2 * H * C * C) / (l ** 5) / Math.expm1((H * C) / (l * K * T));
    const [x, y, z] = cieXYZBar(lam);
    X += B * x; Y += B * y; Z += B * z;
  }
  return [X, Y, Z];
}

export const XYZ_TO_SRGB = [
  [3.2404542, -1.5371385, -0.4985314],
  [-0.9692660, 1.8760108, 0.0415560],
  [0.0556434, -0.2040259, 1.0572252],
];

/** Chromaticity (x, y) of a blackbody. */
export function chromaticity(T) {
  const [X, Y, Z] = blackbodyXYZ(T);
  const s = X + Y + Z;
  return [X / s, Y / s];
}

/**
 * Table of n entries over log10 T in [logTmin, logTmax]: [r, g, b, log10(Y / Y_6500K)], where (r, g, b)
 * is linear sRGB with luminance 1 (negative components clipped to 0). Returned as a Float32Array of 4n.
 */
export function blackbodyTable(n = 64, logTmin = 3.0, logTmax = 4.7) {
  const ref = blackbodyXYZ(6500)[1];
  const out = new Float32Array(4 * n);
  for (let i = 0; i < n; i++) {
    const T = 10 ** (logTmin + ((logTmax - logTmin) * i) / (n - 1));
    const [X, Y, Z] = blackbodyXYZ(T);
    for (let k = 0; k < 3; k++) {
      const m = XYZ_TO_SRGB[k];
      out[4 * i + k] = Math.max(0, (m[0] * X + m[1] * Y + m[2] * Z) / Y);
    }
    out[4 * i + 3] = Math.log10(Y / ref);
  }
  return out;
}
