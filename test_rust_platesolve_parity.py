#!/usr/bin/env python
"""Live A/B parity: the Rust plate solver vs Python tetra3 on identical
synthetic star-field images.

Star fields are rendered by projecting stars from the tetra3 database's
own star table at random pointings (pinhole camera, Gaussian PSFs), so
both solvers face exactly the frames their shared pattern database
describes. Gates (from the Phase 2 plan): solutions within 30 arcsec
centre / 0.1 deg roll, and the Rust solver must solve at least the same
frames Python does.

Uses the FOV-matched sim database (db_fov12_mag7) when present, else
tetra3's bundled default_database.

Build first: cd rust/skytracker-ffi && maturin develop --release
Run: python -m pytest test_rust_platesolve_parity.py
"""

import math
import os
import unittest

import numpy as np

try:
    import skytracker_core
    _HAVE_PS = getattr(skytracker_core, "PLATESOLVE_AVAILABLE", False)
except ImportError:
    _HAVE_PS = False

try:
    import tetra3
    _HAVE_T3 = True
except Exception:
    _HAVE_T3 = False


def _db_path(name):
    import tetra3 as t3

    return os.path.join(os.path.dirname(t3.__file__), "data", name + ".npz")


def _pick_db():
    for name, fov in [("db_fov12_mag7", 10.0), ("default_database", 20.0)]:
        if os.path.exists(_db_path(name)):
            return name, fov
    return None, None


def _render_field(star_table, boresight, roll_rad, fov_deg, w, h, rng,
                  limit_mag=7.0, max_stars=40):
    """Project catalog stars around `boresight` into a synthetic frame."""
    fov = math.radians(fov_deg)
    # Camera basis: x = boresight; construct y/z via the roll.
    x = boresight / np.linalg.norm(boresight)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, x)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    y0 = np.cross(ref, x)
    y0 /= np.linalg.norm(y0)
    z0 = np.cross(x, y0)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    yb = cr * y0 + sr * z0
    zb = -sr * y0 + cr * z0
    rot = np.stack([x, yb, zb])  # world -> camera

    vecs = star_table[:, 2:5]
    mags = star_table[:, 5]
    cam = vecs @ rot.T
    infront = cam[:, 0] > 0.5
    cam = cam[infront]
    m = mags[infront]
    scale = -w / 2.0 / math.tan(fov / 2.0)
    yy = scale * cam[:, 2] / cam[:, 0] + h / 2.0
    xx = scale * cam[:, 1] / cam[:, 0] + w / 2.0
    inframe = (yy > 5) & (yy < h - 5) & (xx > 5) & (xx < w - 5) & (m < limit_mag)
    order = np.argsort(m[inframe])[:max_stars]
    ys, xs, ms = yy[inframe][order], xx[inframe][order], m[inframe][order]

    img = rng.normal(8.0, 2.0, size=(h, w))
    ygrid, xgrid = np.mgrid[0:h, 0:w]
    for yc, xc, mag in zip(ys, xs, ms):
        amp = 250.0 * 10 ** (-0.3 * (mag - 2.0))
        amp = min(max(amp, 30.0), 250.0)
        sig = 1.4
        img += amp * np.exp(-((xgrid - xc) ** 2 + (ygrid - yc) ** 2) / (2 * sig * sig))
    return np.clip(img, 0, 255).astype(np.uint8), len(ys)


@unittest.skipUnless(_HAVE_PS, "skytracker_core wheel lacks PlateSolver")
@unittest.skipUnless(_HAVE_T3, "python tetra3 not installed")
class PlateSolveParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_name, cls.fov_deg = _pick_db()
        if cls.db_name is None:
            raise unittest.SkipTest("no tetra3 database available")
        cls.t3 = tetra3.Tetra3(load_database=cls.db_name)
        cls.rust = skytracker_core.PlateSolver(_db_path(cls.db_name))
        with np.load(_db_path(cls.db_name)) as db:
            cls.star_table = db["star_table"]

    def test_solutions_match(self):
        from PIL import Image

        rng = np.random.default_rng(20260816)
        w, h = 960, 720
        n_fields = 12
        solved_pairs = 0
        worst_sep = 0.0
        worst_roll = 0.0
        for trial in range(n_fields):
            # Random pointing away from the poles for simplicity.
            ra = rng.uniform(0, 2 * math.pi)
            dec = rng.uniform(-1.0, 1.0)
            boresight = np.array([
                math.cos(dec) * math.cos(ra),
                math.cos(dec) * math.sin(ra),
                math.sin(dec),
            ])
            roll = rng.uniform(0, 2 * math.pi)
            img, n_stars = _render_field(
                self.star_table, boresight, roll, self.fov_deg, w, h, rng)
            if n_stars < 8:
                continue

            kwargs = dict(fov_estimate=self.fov_deg,
                          fov_max_error=self.fov_deg * 0.6,
                          pattern_checking_stars=12, match_radius=0.02,
                          match_threshold=1e-2)
            py = self.t3.solve_from_image(Image.fromarray(img), **kwargs)
            rs = self.rust.solve_from_image(img, **kwargs)

            py_ok = py is not None and py.get("RA") is not None
            rs_ok = rs is not None
            if py_ok:
                self.assertTrue(
                    rs_ok,
                    f"trial {trial}: python solved (RA {py['RA']:.3f}) but rust did not")
            if not (py_ok and rs_ok):
                continue
            solved_pairs += 1

            # Angular separation between solution centres.
            def unit(ra_d, dec_d):
                r, d = math.radians(ra_d), math.radians(dec_d)
                return np.array([
                    math.cos(d) * math.cos(r),
                    math.cos(d) * math.sin(r),
                    math.sin(d),
                ])

            sep = math.degrees(math.acos(np.clip(
                np.dot(unit(py["RA"], py["Dec"]), unit(rs["RA"], rs["Dec"])),
                -1, 1)))
            droll = abs((rs["Roll"] - py["Roll"] + 180.0) % 360.0 - 180.0)
            worst_sep = max(worst_sep, sep)
            worst_roll = max(worst_roll, droll)
            self.assertLess(sep, 30.0 / 3600.0,
                            f"trial {trial}: centre sep {sep * 3600:.1f} arcsec")
            self.assertLess(droll, 0.1, f"trial {trial}: roll diff {droll:.4f} deg")
            self.assertEqual(rs["Matches"], py["Matches"], f"trial {trial}: match count")

        self.assertGreaterEqual(solved_pairs, 8,
                                f"only {solved_pairs} of {n_fields} fields solved by both")
        print(f"\n[parity] platesolve: {solved_pairs} fields, worst centre "
              f"{worst_sep * 3600:.2f} arcsec, worst roll {worst_roll:.4f} deg")

    def test_flag_routes_plate_solver_wrapper(self):
        """plate_solver.PlateSolver with the env flag on must route to Rust
        and agree with the Python path on the same frame."""
        import time as _time

        from config import ConfigState
        import plate_solver as ps

        rng = np.random.default_rng(99)
        w, h = 960, 720
        img, n = _render_field(self.star_table, np.array([0.6, 0.64, 0.48]),
                               1.1, self.fov_deg, w, h, rng)
        if n < 8:
            self.skipTest("field too sparse")

        cfg = ConfigState()
        cfg.camera_configs.setdefault("camera1", {})["tetra3_db"] = self.db_name
        solver = ps.PlateSolver(cfg, "camera1")
        solver.fov_deg = self.fov_deg  # bypass optics-derived FOV for the test

        os.environ["SKYTRACKER_RUST_PLATESOLVE"] = "0"
        try:
            t0 = _time.perf_counter()
            py_res = solver.solve(img)
            t_py = _time.perf_counter() - t0
            os.environ["SKYTRACKER_RUST_PLATESOLVE"] = "1"
            t0 = _time.perf_counter()
            rs_res = solver.solve(img)
            t_rs = _time.perf_counter() - t0
        finally:
            os.environ.pop("SKYTRACKER_RUST_PLATESOLVE", None)

        self.assertIsNotNone(py_res)
        self.assertIsNotNone(rs_res)
        self.assertEqual(py_res.solved, rs_res.solved)
        if py_res.solved:
            self.assertLess(abs(py_res.ra_deg - rs_res.ra_deg), 30.0 / 3600.0)
            self.assertLess(abs(py_res.dec_deg - rs_res.dec_deg), 30.0 / 3600.0)
            self.assertEqual(py_res.n_matches, rs_res.n_matches)
        print(f"\n[parity] wrapper flag-on: solved={rs_res.solved} "
              f"py {t_py * 1e3:.1f} ms vs rust {t_rs * 1e3:.1f} ms "
              f"(x{t_py / max(t_rs, 1e-9):.1f})")

    def test_centroids_match(self):
        rng = np.random.default_rng(7)
        w, h = 960, 720
        boresight = np.array([1.0, 0.0, 0.0])
        img, _ = _render_field(self.star_table, boresight, 0.3, self.fov_deg, w, h, rng)
        py = tetra3.get_centroids_from_image(img)
        rs = self.rust.get_centroids(img)
        self.assertEqual(len(py), rs.shape[0], "centroid count")
        worst = float(np.max(np.linalg.norm(np.asarray(py) - rs, axis=1)))
        self.assertLess(worst, 0.3)
        print(f"[parity] centroids: {rs.shape[0]} spots, worst {worst:.4f} px")


if __name__ == "__main__":
    unittest.main(verbosity=2)
