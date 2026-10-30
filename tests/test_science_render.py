"""End-to-end checks of jev.science.blackhole.render (small sizes, a few seconds on a CPU) and of the
blackbody colour table in jev.science.color."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image

from jev.science import blackhole, color, kerr


def lum(img):
    return np.asarray(img, dtype=float)[..., :3] @ np.array([0.2126, 0.7152, 0.0722])


def test_planckian_locus_chromaticity():
    """Blackbody chromaticities from the CIE fit agree with the CIE 1931 Planckian locus (x, y)."""
    for T, (x, y) in ((3000, (0.4369, 0.4041)), (5000, (0.3451, 0.3516)), (6500, (0.3135, 0.3237)),
                      (10000, (0.2807, 0.2884))):
        xy = color.xy_chromaticity([T])[0]
        assert abs(xy[0] - x) < 2e-3 and abs(xy[1] - y) < 2e-3, (T, xy)
    rgb = color.blackbody_rgb([2000.0, 6500.0, 20000.0], luminance=False)
    assert rgb[0, 0] > rgb[0, 2]                 # cool = red
    assert np.allclose(rgb[1], 1.0, atol=0.06)   # ~D65 white
    assert rgb[2, 2] > rgb[2, 0]                 # hot = blue
    assert color.bb_luminance(8000.0) > color.bb_luminance(4000.0)


def test_still_has_shadow_photon_ring_and_disk(tmp_path):
    paths = blackhole.render({"width": 160, "height": 90, "seed": 1}, tmp_path)
    png, meta = paths
    assert png.suffix == ".png" and meta.suffix == ".json"
    img = Image.open(png)
    assert img.size == (160, 90)
    L = lum(img)
    m = json.loads(meta.read_text())
    assert m["physics"]["r_isco"] == pytest.approx(kerr.isco(0.6))
    # the shadow's centre is black; the disk (centre row band, just below the centre) is bright
    assert L[40:44, 76:84].mean() < 8
    assert L.max() > 200
    assert m["trace"]["unresolved"] == 0


def _hdr(params):
    p = blackhole.normalize({"spin": 0.0, "width": 160, "height": 90, "background": "none", "inclination_deg": 80.0,
                             "distance_M": 30.0, "fov_deg": 50.0, **params})
    cam, res, shape = blackhole.trace_view(p, p["distance_M"], p["inclination_deg"])
    return blackhole.shade(res, shape, p, cam) @ np.array([0.2126, 0.7152, 0.0722])


def test_doppler_beaming_asymmetry_and_interstellar_mode():
    """a = 0: with beaming the approaching (left) side is far brighter; in Interstellar mode
    (no Doppler, no beaming) the image is mirror symmetric."""
    L = _hdr({"disk": {"texture": False, "temperature_scale": 5000}})
    assert L[:, :80].sum() > 3 * L[:, 80:].sum()
    L = _hdr({"interstellar": True, "disk": {"texture": False, "temperature_scale": 5000}})
    assert np.abs(L - L[:, ::-1]).max() < 1e-6 * L.max()


def test_secondary_image_below_the_shadow(tmp_path):
    """Seen from just above the disk plane, the far side of the disk appears above the shadow and its
    underside below it (rays that pass under the hole), with the shadow dark in between."""
    paths = blackhole.render({"spin": 0.0, "width": 160, "height": 120, "background": "none",
                              "inclination_deg": 86.0, "distance_M": 40.0, "fov_deg": 40.0,
                              "disk": {"outer": 12.0}, "interstellar": True}, tmp_path)
    L = lum(Image.open(paths[0]))
    col = L[:, 78:82].mean(1)
    # from the top: dark sky, bright arch (far side over the top), dark shadow, bright lower arch
    above = col[: 60].argmax()
    below = 60 + col[60:].argmax()
    shadow = col[above:below].argmin() + above
    assert col[above] > 60 and col[below] > 30 and col[shadow] < 5
    assert above < shadow < below


def test_orbit_animation_mp4_and_gif(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    paths = blackhole.render({"width": 64, "height": 36, "frames": 4, "seconds": 1.0}, tmp_path)
    mp4 = paths[0]
    assert mp4.suffix == ".mp4" and mp4.stat().st_size > 1000
    r = imageio.get_reader(mp4)
    assert r.count_frames() == 4
    meta = json.loads(paths[1].read_text())
    assert meta["fps"] == pytest.approx(4.0)
    gif = blackhole.render({"width": 48, "height": 28, "frames": 3, "format": "gif", "animation": "zoom"},
                           tmp_path / "g")[0]
    assert gif.suffix == ".gif" and Image.open(gif).n_frames == 3


def test_validation_and_cancel(tmp_path):
    with pytest.raises(ValueError):
        blackhole.normalize({"spin": 1.2})
    with pytest.raises(ValueError):
        blackhole.normalize({"distance_M": 1.5})
    with pytest.raises(ValueError):
        blackhole.normalize({"disk": {"inner": 2.0}, "spin": 0.0})      # below the photon orbit
    p = blackhole.normalize({"width": 101, "height": 57, "bogus": 1, "disk": {"interstellar": True}})
    assert (p["width"], p["height"]) == (102, 58) and "bogus" not in p
    assert p["disk"]["doppler"] is False and p["disk"]["beaming"] is False
    assert p["disk"]["inner"] == pytest.approx(kerr.isco(0.6))
    assert blackhole.render({"width": 64, "height": 36, "frames": 3}, tmp_path, cancelled=lambda: True) == []
    assert not list(tmp_path.glob("*.mp4"))
    est = blackhole.estimate_seconds({"width": 640, "height": 360})
    assert 0 < est["seconds"] < 600 and est["rays"] == 640 * 360


def test_progress_is_reported(tmp_path):
    seen = []
    blackhole.render({"width": 48, "height": 28}, tmp_path, progress=lambda f, m: seen.append(f))
    assert seen and seen[-1] == 1.0 and all(0 <= f <= 1 for f in seen)


def test_shadow_size_in_rendered_image(tmp_path):
    """Disk off, a = 0, face-on: the black disc in the image has the analytic angular radius."""
    r_obs, fov, w = 30.0, 40.0, 200
    paths = blackhole.render({"spin": 0.0, "inclination_deg": 1.0, "distance_M": r_obs, "fov_deg": fov,
                              "width": w, "height": w, "background": "grid", "disk": {"enabled": False}, "bloom": 0.0},
                              tmp_path)
    L = lum(Image.open(paths[0]))
    row = L[w // 2]
    dark = np.nonzero(row < 1)[0]
    half_width_px = (dark.max() - dark.min() + 1) / 2
    tan_half = math.tan(math.radians(fov) / 2)
    ang = math.atan(half_width_px / (w / 2) * tan_half)
    exact = math.asin(3 * math.sqrt(3) * math.sqrt(1 - 2 / r_obs) / r_obs)
    assert ang == pytest.approx(exact, abs=math.atan(2 * tan_half / w) * 1.5)
