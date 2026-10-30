"""Scientific rendering and physics simulations for JEV.

    jev.science.kerr        Kerr spacetime: horizon, ISCO, photon orbits, critical curve, ZAMO camera,
                            Page-Thorne disk flux and the vectorized null-geodesic integrator
    jev.science.color       blackbody colour (CIE 1931 fit -> linear sRGB), tone mapping
    jev.science.background  procedural, lensed celestial sphere (stars, checker grid)
    jev.science.blackhole   render(): stills (PNG) and animations (MP4) of a Kerr black hole + disk
    jev.science.sims        run(): N-body, double pendulum, 2-D wave, Kepler orbit, Lorenz
    jev.science.notes       the physics notes JEV can quote (explain_black_hole_physics)

Units: geometrized G = c = 1 and the black-hole mass M = 1, so lengths and times are in units of M.
The physics, approximations and references are written up in docs/jev/science.md.
"""
