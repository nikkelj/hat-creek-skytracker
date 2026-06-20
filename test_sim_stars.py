"""
Phase 2 regression: the synthetic-image simulator renders real Hipparcos stars into
camera frames at catalogue-predicted positions, and the legacy random field still works
when the toggle is off.
"""

import unittest

import numpy as np
from skyfield.api import load

from config import ConfigState
from tracking_visuals import TrackingVisState
from simulator import HardwareSimulator


def _make_sim(use_real):
    cfg = ConfigState()
    cfg.lat_str, cfg.lon_str, cfg.alt_str = '34.874', '-120.446', '120'
    cfg.mount_mode = 'AltAz'
    cfg.alignment_azimuth_str = '0.0'
    cfg.star_limiting_magnitude = 6.5
    cfg.sim_config['enabled'] = True
    cfg.sim_config['sim_use_real_stars'] = use_real
    cfg.camera_configs['camera1']['exposure'] = 1000.0  # us, short -> point sources
    ts = load.timescale()
    tvs = TrackingVisState()
    return cfg, ts, HardwareSimulator(cfg, tvs, ts)


class TestSimRealStars(unittest.TestCase):
    def test_boresight_star_lands_at_center(self):
        cfg, ts, sim = _make_sim(use_real=True)
        cat = sim._get_star_catalog()
        res = cat.current_altaz(34.874, -120.446, 120.0, ts, ts.now().tt,
                                elevation_mask=30.0, max_count=50, limiting_mag=6.5)
        self.assertGreater(res['n_visible'], 0, "no bright stars up to aim at")
        i = int(np.argmin(res['mag']))
        # AltAz: cam_az = (AZM + align) % 360, cam_el = 90 - ALT
        sim.mount.az_true_deg = float(res['az'][i]) % 360.0
        sim.mount.el_true_deg = 90.0 - float(res['el'][i])

        frame = sim.render_frame(0)
        h, w = frame.shape
        cy, cx = h // 2, w // 2
        # The aimed star must render a bright source at the image center. (With a deep
        # catalogue many stars saturate, so the GLOBAL argmax may be a different star;
        # assert the central window has a bright peak instead.)
        win = frame[cy - 20:cy + 20, cx - 20:cx + 20]
        self.assertGreater(int(win.max()), 50, "no bright star at image center")

    def test_legacy_field_when_toggle_off(self):
        cfg, ts, sim = _make_sim(use_real=False)
        frame = sim.render_frame(0)
        # The legacy random field draws something above the background floor.
        self.assertGreater(int(frame.max()), int(cfg.sim_config['background_level']))

    def test_high_elevation_star_not_walled(self):
        """Regression: azimuth converges with elevation, so a star can be well inside the
        frame in pixels while its raw |d_az| exceeds the azimuth half-FOV. The cull must
        compare cross-elevation (|d_az|*cos(el)) -- otherwise stars wink out at a phantom
        vertical wall at column (w/2)*cos(el)."""
        import math
        from simulator import ifov_deg
        cfg, ts, sim = _make_sim(use_real=True)
        s = sim._sim()
        w = int(s.get('cam_width', 960))
        ifd = ifov_deg(float(cfg.get_camera_pixel_size('camera1')),
                       float(cfg.get_camera_focal_length('camera1')))
        fov_x = w * ifd / 2.0

        cam_el = 75.0                       # high elevation: strong azimuth convergence
        cos_el = math.cos(math.radians(cam_el))
        # Azimuth offset chosen between the old (buggy) wall fov_x and the true edge
        # fov_x/cos(el): inside the frame, but the un-corrected cull would have dropped it.
        d_az = fov_x * (1.0 / cos_el + 1.0) / 2.0
        self.assertGreater(d_az, fov_x)              # old gate would cull this
        star_az = 123.0
        cam_az = (star_az - d_az) % 360.0
        sim.mount.az_true_deg = cam_az               # AltAz: cam_az = AZM (+align 0)
        sim.mount.el_true_deg = 90.0 - cam_el        # cam_el = 90 - ALT

        # Inject a single bright star at the boresight elevation, offset in azimuth.
        sim._catalog_fov_stars = lambda *a, **k: (
            np.array([star_az]), np.array([cam_el]), np.array([250.0]))
        frame = sim.render_frame(0)

        self.assertGreater(int(frame.max()), 50, "high-elevation star was culled (phantom wall)")
        peak_col = int(np.unravel_index(int(np.argmax(frame)), frame.shape)[1])
        # The star sits in the previously-walled band (right of column (w/2)*cos(el)).
        self.assertGreater(peak_col, w * 0.5 + (w * 0.5) * cos_el,
                           "star did not render beyond the old azimuth wall")


if __name__ == '__main__':
    unittest.main()
