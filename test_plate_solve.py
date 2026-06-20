"""
Phase 3 end-to-end: render a synthetic star frame in the simulator, plate-solve it with
tetra3, convert the solution to topocentric az/el, and confirm it matches the boresight
truth -- and that the instantaneous-alignment derivation cancels an injected mount
misalignment.

Requires a tetra3 database covering the sim camera's FOV (e.g. db_fov12_mag7, built by
tools/build_tetra3_db.py). Skipped if tetra3 or the database is unavailable.
"""

import math
import unittest

import numpy as np
from skyfield.api import load

import plate_solver as ps_mod
from config import ConfigState
from tracking_visuals import TrackingVisState
from simulator import HardwareSimulator

LAT, LON, ELEV = 34.874, -120.446, 120.0


def _angular_sep_deg(az1, el1, az2, el2):
    a1, e1, a2, e2 = map(math.radians, (az1, el1, az2, el2))
    cos_sep = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


@unittest.skipUnless(ps_mod.tetra3_available(), "tetra3 not installed")
class TestPlateSolveSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = ConfigState()
        cls.cfg.lat_str, cls.cfg.lon_str, cls.cfg.alt_str = '34.874', '-120.446', '120'
        cls.cfg.mount_mode = 'AltAz'
        cls.cfg.alignment_azimuth_str = '0.0'
        cls.cfg.star_limiting_magnitude = 7.0
        cls.cfg.sim_config['enabled'] = True
        cls.cfg.sim_config['sim_use_real_stars'] = True
        # Match the sim rendering depth to the 8-deg Hipparcos mag-7 DB (db_fov12_mag7),
        # and force the Hipparcos path so the rendered field deterministically matches
        # the (Hipparcos) DB -- independent of whether tyc_main.dat happens to be present.
        cls.cfg.sim_config['sim_star_limit_mag'] = 7.0
        cls.cfg.sim_config['sim_use_deep_catalog'] = False
        cls.cfg.camera_configs['camera1']['focal_length'] = 25.0   # ~8 deg FOV
        cls.cfg.camera_configs['camera1']['pixel_size'] = 3.75
        cls.cfg.camera_configs['camera1']['exposure'] = 2000.0     # short -> point sources
        cls.ts = load.timescale()
        cls.tvs = TrackingVisState()
        cls.sim = HardwareSimulator(cls.cfg, cls.tvs, cls.ts)
        try:
            cls.solver = ps_mod.PlateSolver(cls.cfg, 'camera1')
            cls.solver.ensure_loaded()
        except Exception as e:
            raise unittest.SkipTest(f"tetra3 DB unavailable: {e}")

    def _aim_at_rich_field(self):
        """Point the mount at the densest catalogue field above the horizon and return
        its (az, el) -- a star-rich, reliably solvable field."""
        cat = self.sim._get_star_catalog()
        res = cat.current_altaz(LAT, LON, ELEV, self.ts, self.ts.now().tt,
                                elevation_mask=30.0, max_count=2000, limiting_mag=7.0)
        self.assertGreater(res['n_visible'], 0)
        az_all, el_all = res['az'], res['el']
        # star count within an ~8x6 deg box centred on each candidate
        best_i, best_n = 0, -1
        for j in range(len(az_all)):
            d_az = ((az_all - az_all[j] + 180.0) % 360.0) - 180.0
            d_el = el_all - el_all[j]
            n = int(((np.abs(d_az) < 4.0) & (np.abs(d_el) < 3.0)).sum())
            if n > best_n:
                best_n, best_i = n, j
        az, el = float(az_all[best_i]), float(el_all[best_i])
        # AltAz, align=0: cam_az = az_true, cam_el = 90 - alt_true
        self.sim.mount.az_true_deg = az % 360.0
        self.sim.mount.el_true_deg = 90.0 - el
        return az, el

    def test_solution_matches_boresight(self):
        truth_az, truth_el = self._aim_at_rich_field()
        frame = self.sim.render_frame(0)
        t = self.ts.now()
        result = self.solver.solve(frame)
        self.assertIsNotNone(result)
        self.assertTrue(result.solved, f"no solution (status {self.solver.db_name})")

        az, el = ps_mod.solved_azel(result.ra_deg, result.dec_deg, LAT, LON, ELEV,
                                    self.tvs.ephemeris, self.ts, t)
        sep = _angular_sep_deg(az, el, truth_az, truth_el)
        self.assertLess(sep, 0.3, f"solved {sep*60:.1f} arcmin from boresight")

    def test_instantaneous_alignment_cancels_misalignment(self):
        truth_az, truth_el = self._aim_at_rich_field()
        frame = self.sim.render_frame(0)
        t = self.ts.now()
        result = self.solver.solve(frame)
        self.assertTrue(result.solved)
        az, el = ps_mod.solved_azel(result.ra_deg, result.dec_deg, LAT, LON, ELEV,
                                    self.tvs.ephemeris, self.ts, t)

        # Inject a 3 deg azimuth encoder bias: the mount would *report* az_true + 3.
        injected = 3.0
        mount_reported_azm = (self.sim.mount.az_true_deg + injected) % 360.0
        align = ps_mod.instantaneous_alignment_azimuth(az, mount_reported_azm)
        # align should be ~ -injected so that (AZM_reported + align) recovers true az.
        self.assertAlmostEqual(align, -injected, delta=0.3)


if __name__ == '__main__':
    unittest.main()
