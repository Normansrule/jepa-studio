"""The physics notes JEV quotes when asked how the black-hole renderer works (explain_black_hole_physics).

Plain text with simple formulas so a small local model can paraphrase them without inventing anything.
docs/jev/science.md is the long version with typeset equations and measured numbers.
"""
from __future__ import annotations

NOTES: dict[str, str] = {
    "overview": """\
JEV renders a spinning (Kerr) black hole the way the film Interstellar's images were made (James,
von Tunzelmann, Franklin & Thorne 2015): for every pixel a light ray is traced backwards from the camera
through curved spacetime until it falls into the hole (black), hits the accretion disk (coloured by
its temperature and frequency shift) or escapes to the distant sky (lensed stars). Units: G = c = 1,
lengths and times in units of the mass M. Luminet (1979) made the first such image of a thin disk
around a non-spinning hole.""",
    "geodesics": """\
Spacetime: Kerr metric in Boyer-Lindquist coordinates (t, r, theta, phi), spin a = J/M.
Sigma = r^2 + a^2 cos^2 theta, Delta = r^2 - 2Mr + a^2.
A photon has three conserved quantities: energy E, axial angular momentum L and the Carter constant Q
(Carter 1968). With b = L/E, q = Q/E^2 and Mino time (d sigma = d lambda / Sigma):
  (dr/dsigma)^2 = R(r) = (r^2 + a^2 - a b)^2 - Delta [(b - a)^2 + q]
  (dtheta/dsigma)^2 = Theta(theta) = q + a^2 cos^2 theta - b^2 cot^2 theta
  dphi/dsigma = a (r^2 + a^2 - a b)/Delta + b/sin^2 theta - a
  dt/dsigma = (r^2 + a^2)(r^2 + a^2 - a b)/Delta + a (b - a sin^2 theta)
The renderer integrates d^2r/dsigma^2 = R'(r)/2 and d^2theta/dsigma^2 = Theta'(theta)/2 (no square roots,
smooth through turning points) with an adaptive Dormand-Prince 5(4) method, vectorized over all
pixels, and keeps (dr/dsigma)^2 = R and (dtheta/dsigma)^2 = Theta exact away from turning points.""",
    "camera": """\
The camera is a zero-angular-momentum observer (ZAMO, also called the fiducial observer or FIDO): at
rest relative to the local frame dragging, as in James et al. (2015). A pixel's direction n in the
camera's orthonormal frame gives the photon's momenta p_r = -n_r sqrt(Sigma/Delta) E_cam,
p_theta = -n_theta sqrt(Sigma) E_cam, L = -n_phi varpi E_cam, E = alpha E_cam + omega L, where alpha is
the lapse, omega = 2 a M r / A the frame-dragging angular velocity and varpi the circumferential
radius. A camera moving along phi with speed beta is handled by special-relativistic aberration.""",
    "disk": """\
The accretion disk is geometrically thin, optically thick and lies in the equatorial plane between
r_in (default: the innermost stable circular orbit, ISCO) and r_out. Its temperature follows the
Novikov-Thorne model: flux F(r) from Page & Thorne (1974), T proportional to F^(1/4), zero at the ISCO
(no torque at the inner edge), peaking just outside it and falling as r^(-3/4) far out. (Option:
the simpler T ~ r^(-3/4).) ISCO (Bardeen, Press & Teukolsky 1972): 6M for a = 0, 1.237M for a = 0.998.
Rays that cross the plane inside the ISCO or outside r_out keep going, so the far side of the disk is
seen above the shadow (bent over the top), its underside below it, and thin higher-order images hug
the shadow edge: the photon ring.""",
    "redshift": """\
Gas orbits on prograde circular geodesics, Omega = 1/(r^(3/2) + a) (in units of M). The frequency ratio
of a photon is g = nu_obs/nu_emit = E_cam / (u^t (1 - Omega b)). Because I_nu / nu^3 is invariant along
a ray, a blackbody of temperature T is seen as a blackbody of temperature g T: its colour comes from the
CIE 1931 colour-matching functions integrated over Planck's law at g T, and its visible brightness
changes with it (beaming). Toggles: redshift (the gravitational part alpha(r_emit)/alpha(r_cam)),
doppler (colour shift from the motion), beaming (brightness change from the motion). The side of the
disk moving toward the camera is blueshifted and much brighter.""",
    "interstellar": """\
For the film, Christopher Nolan and Paul Franklin chose to leave out the Doppler shifts of the disk
(which would make one side blue-white and far brighter than the other) so the images would be
understandable for a mass audience; the spin was also lowered to about a = 0.6 M for the disk shots
(James et al. 2015). JEV's interstellar=true switches off the Doppler colour shift and the beaming to
reproduce that look; the physical default keeps them.""",
    "shadow": """\
Photons that come close to the unstable spherical photon orbits wind around the hole many times; the
boundary between captured and escaping rays is the critical curve (Bardeen 1973). For a = 0 it is a
circle of angular radius sin(alpha) = 3 sqrt(3) M sqrt(1 - 2M/r)/r for a static observer at r (the
photon sphere is at r = 3M). For a spinning hole seen edge-on the shadow is shifted toward the side
rotating away from the camera and flattened on the approaching side: for a = 0.99, Bardeen's
coordinates run from about -2.25 M to +6.98 M. The renderer reproduces these numbers to 1e-4 or
better (tests/test_science_kerr.py).""",
    "limits": """\
What is simplified: the disk is infinitely thin and stationary, with no radiative transfer through it
(optionally semi-transparent), no light from the plunging region inside the ISCO, no
self-irradiation, no polarisation; stars are idealised point sources on a procedural sky; the light
travel time is used only to animate the disk pattern; colours use an analytic fit of the CIE 1931
functions. The browser preview integrates Schwarzschild (a = 0) geodesics only.""",
    "references": """\
- O. James, E. von Tunzelmann, P. Franklin, K. S. Thorne, "Gravitational lensing by spinning black holes
  in astrophysics, and in the movie Interstellar", Class. Quantum Grav. 32, 065001 (2015),
  doi:10.1088/0264-9381/32/6/065001, arXiv:1502.03808.
- J.-P. Luminet, "Image of a spherical black hole with thin accretion disk", Astron. Astrophys. 75,
  228-235 (1979).
- B. Carter, "Global structure of the Kerr family of gravitational fields", Phys. Rev. 174, 1559 (1968).
- J. M. Bardeen, W. H. Press, S. A. Teukolsky, "Rotating black holes: locally nonrotating frames, energy
  extraction, and scalar synchrotron radiation", ApJ 178, 347 (1972), doi:10.1086/151796.
- J. M. Bardeen, "Timelike and null geodesics in the Kerr metric", in Black Holes (Les Houches 1972),
  eds. C. DeWitt & B. DeWitt, p. 215 (1973).
- D. N. Page, K. S. Thorne, "Disk-accretion onto a black hole. Time-averaged structure of accretion
  disk", ApJ 191, 499 (1974), doi:10.1086/152990.
- C. Wyman, P.-P. Sloan, P. Shirley, "Simple analytic approximations to the CIE XYZ color matching
  functions", JCGT 2(2), 1-11 (2013).""",
}

TOPICS = tuple(NOTES)


def explain(topic: str | None = None) -> str:
    if not topic or topic == "all":
        return "\n\n".join(f"## {k}\n{v}" for k, v in NOTES.items())
    if topic not in NOTES:
        raise KeyError(topic)
    return NOTES[topic] + ("\n\n## references\n" + NOTES["references"] if topic != "references" else "")
