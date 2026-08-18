"""Record golden vectors from the Python reference implementations.

Phase 0 of the Rust port (see rust/README.md). Dumps deterministic
input/output pairs from skyfield, tetra3, and OpenCV into tests/golden/ so
later Rust ports can be validated against the *current* Python behavior --
including long after the Python implementations are deleted. The golden
files are committed; Rust integration tests read them via the npyz crate.

Run under the `track` conda env from the repo root:

    C:\\Users\\nikke\\anaconda3\\envs\\track\\python.exe tools/record_golden.py

Everything here is deterministic: fixed epochs, fixed site, fixed seeds,
and the TLE lines actually used are stored inside the .npz so the golden
file is self-contained (no dependency on tle_cache.tle staying stable).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO, "tests", "golden")

# The observing site from config.json, frozen here so the goldens do not
# drift if the config changes.
SITE_LAT_DEG = 34.8740289
SITE_LON_DEG = -120.4461237
SITE_ALT_M = 100.0

# Fixed epoch grids (UTC). Satellite epochs sit near the TLE epochs in the
# cache (propagating a 2026 TLE years away is meaningless); star/body/GAST
# epochs span several years to exercise precession/nutation.
SAT_EPOCH_START = (2026, 8, 1)
SAT_EPOCH_DAYS = 14.0
SAT_EPOCH_COUNT = 200
WIDE_EPOCH_START = (2024, 1, 1)
WIDE_EPOCH_YEARS = 3.0
WIDE_EPOCH_COUNT = 50


def _timescale():
    from skyfield.api import load

    return load.timescale()


def _site():
    from skyfield.api import wgs84

    return wgs84.latlon(SITE_LAT_DEG, SITE_LON_DEG, elevation_m=SITE_ALT_M)


def _sat_times(ts):
    t0 = ts.utc(*SAT_EPOCH_START)
    return ts.tt_jd(np.linspace(t0.tt, t0.tt + SAT_EPOCH_DAYS, SAT_EPOCH_COUNT))


def _wide_times(ts):
    t0 = ts.utc(*WIDE_EPOCH_START)
    return ts.tt_jd(
        np.linspace(t0.tt, t0.tt + 365.25 * WIDE_EPOCH_YEARS, WIDE_EPOCH_COUNT)
    )


def _pick_tles():
    """Pick ISS + up to 5 more sats with well-spread inclinations from the
    local TLE cache. The chosen lines are stored in the golden file itself."""
    path = os.path.join(REPO, "tle_cache.tle")
    entries = []  # (name, l1, l2, incl_deg)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        # The cache has \r\r\n endings; universal newlines yields an empty
        # line after every real one -- drop blanks.
        lines = [ln.rstrip() for ln in f if ln.strip()]
    for i in range(0, len(lines) - 2):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            try:
                incl = float(lines[i + 2][8:16])
            except ValueError:
                continue
            entries.append((lines[i].strip(), lines[i + 1], lines[i + 2], incl))
    chosen = []
    for name, l1, l2, incl in entries:
        if "ISS (ZARYA)" in name or l1[2:7] == "25544":
            chosen.append((name, l1, l2, incl))
            break
    # Spread of inclination buckets, first-seen wins: deterministic.
    buckets = {}
    for name, l1, l2, incl in entries:
        b = int(incl // 15)
        if b not in buckets:
            buckets[b] = (name, l1, l2, incl)
    for b in sorted(buckets):
        cand = buckets[b]
        if all(c[1] != cand[1] for c in chosen):
            chosen.append(cand)
        if len(chosen) >= 6:
            break
    return chosen


def record_astro_sats():
    from skyfield.api import EarthSatellite

    ts = _timescale()
    site = _site()
    times = _sat_times(ts)
    tles = _pick_tles()

    names, l1s, l2s = [], [], []
    alt = np.zeros((len(tles), SAT_EPOCH_COUNT))
    az = np.zeros_like(alt)
    dist_km = np.zeros_like(alt)
    az_rate = np.zeros_like(alt)  # deg/s, finite difference over 1 s
    el_rate = np.zeros_like(alt)

    dt_s = 0.5
    t_minus = ts.tt_jd(times.tt - dt_s / 86400.0)
    t_plus = ts.tt_jd(times.tt + dt_s / 86400.0)

    for i, (name, l1, l2, _incl) in enumerate(tles):
        sat = EarthSatellite(l1, l2, name, ts)
        diff = sat - site
        a, z, d = diff.at(times).altaz()
        alt[i], az[i], dist_km[i] = a.degrees, z.degrees, d.km
        am, zm, _ = diff.at(t_minus).altaz()
        ap, zp, _ = diff.at(t_plus).altaz()
        daz = (zp.degrees - zm.degrees + 180.0) % 360.0 - 180.0
        az_rate[i] = daz / (2 * dt_s)
        el_rate[i] = (ap.degrees - am.degrees) / (2 * dt_s)
        names.append(name)
        l1s.append(l1)
        l2s.append(l2)

    # Plain-text TLE copy for the Rust tests (numpy unicode arrays are
    # awkward to read via npyz).
    with open(os.path.join(GOLDEN_DIR, "sat_tles.txt"), "w", newline="\n") as f:
        for name, l1, l2 in zip(names, l1s, l2s):
            f.write(f"{name}\n{l1}\n{l2}\n")

    np.savez_compressed(
        os.path.join(GOLDEN_DIR, "astro_sats.npz"),
        tle_name=np.array(names),
        tle_line1=np.array(l1s),
        tle_line2=np.array(l2s),
        time_tt_jd=times.tt,
        site_lat_lon_alt=np.array([SITE_LAT_DEG, SITE_LON_DEG, SITE_ALT_M]),
        alt_deg=alt,
        az_deg=az,
        dist_km=dist_km,
        az_rate_dps=az_rate,
        el_rate_dps=el_rate,
    )
    print(f"astro_sats.npz: {len(tles)} sats x {SAT_EPOCH_COUNT} epochs")


def record_astro_bodies():
    from skyfield.api import load

    ts = _timescale()
    site = _site()
    times = _wide_times(ts)
    eph = load(os.path.join(REPO, "de421.bsp"))
    earth = eph["earth"]
    observer = earth + _site()

    bodies = [
        ("sun", "sun"),
        ("moon", "moon"),
        ("mercury", "mercury"),
        ("venus", "venus"),
        ("mars", "mars"),
        ("jupiter", "jupiter barycenter"),
        ("saturn", "saturn barycenter"),
    ]
    names = []
    alt = np.zeros((len(bodies), WIDE_EPOCH_COUNT))
    az = np.zeros_like(alt)
    ra_h = np.zeros_like(alt)
    dec_deg = np.zeros_like(alt)
    for i, (name, key) in enumerate(bodies):
        app = observer.at(times).observe(eph[key]).apparent()
        a, z, _ = app.altaz()
        r, d, _ = app.radec()
        alt[i], az[i] = a.degrees, z.degrees
        ra_h[i], dec_deg[i] = r.hours, d.degrees
        names.append(name)

    np.savez_compressed(
        os.path.join(GOLDEN_DIR, "astro_bodies.npz"),
        body=np.array(names),
        time_tt_jd=times.tt,
        site_lat_lon_alt=np.array([SITE_LAT_DEG, SITE_LON_DEG, SITE_ALT_M]),
        alt_deg=alt,
        az_deg=az,
        ra_hours=ra_h,
        dec_deg=dec_deg,
    )
    print(f"astro_bodies.npz: {len(bodies)} bodies x {WIDE_EPOCH_COUNT} epochs")


def record_astro_stars():
    from skyfield.api import Star, load
    from skyfield.data import hipparcos

    ts = _timescale()
    times = _wide_times(ts)[::5]  # 10 epochs across the span
    eph = load(os.path.join(REPO, "de421.bsp"))
    observer = eph["earth"] + _site()

    with open(os.path.join(REPO, "hip_main.dat"), "rb") as f:
        df = hipparcos.load_dataframe(f)
    df = df[df["magnitude"].notna()].sort_values("magnitude")
    # 40 brightest + 10 highest proper-motion of the bright set: exercises
    # both the common path and the PM-dominated corner (e.g. 61 Cyg).
    bright = df.head(200)
    pm = np.hypot(
        bright["ra_mas_per_year"].fillna(0), bright["dec_mas_per_year"].fillna(0)
    )
    picks = list(bright.head(40).index) + [
        i for i in pm.sort_values(ascending=False).index if i not in bright.head(40).index
    ][:10]
    sel = df.loc[picks]

    hip_ids = np.array(sel.index, dtype=np.int64)
    n = len(sel)
    ra_app = np.zeros((n, len(times.tt)))
    dec_app = np.zeros_like(ra_app)
    alt = np.zeros_like(ra_app)
    az = np.zeros_like(ra_app)
    for i, (_, row) in enumerate(sel.iterrows()):
        star = Star.from_dataframe(row)
        app = observer.at(times).observe(star).apparent()
        r, d, _ = app.radec(epoch="date")
        ra_app[i], dec_app[i] = r.hours, d.degrees
        a, z, _ = app.altaz()
        alt[i], az[i] = a.degrees, z.degrees

    np.savez_compressed(
        os.path.join(GOLDEN_DIR, "astro_stars.npz"),
        hip_id=hip_ids,
        catalog_ra_deg=sel["ra_degrees"].to_numpy(),
        catalog_dec_deg=sel["dec_degrees"].to_numpy(),
        pm_ra_mas_yr=sel["ra_mas_per_year"].fillna(0).to_numpy(),
        pm_dec_mas_yr=sel["dec_mas_per_year"].fillna(0).to_numpy(),
        parallax_mas=sel["parallax_mas"].fillna(0).to_numpy(),
        time_tt_jd=times.tt,
        site_lat_lon_alt=np.array([SITE_LAT_DEG, SITE_LON_DEG, SITE_ALT_M]),
        ra_apparent_hours=ra_app,
        dec_apparent_deg=dec_app,
        alt_deg=alt,
        az_deg=az,
    )
    print(f"astro_stars.npz: {n} stars x {len(times.tt)} epochs")


def record_astro_time():
    ts = _timescale()
    times = _wide_times(ts)
    np.savez_compressed(
        os.path.join(GOLDEN_DIR, "astro_time.npz"),
        time_tt_jd=times.tt,
        time_ut1_jd=times.ut1,
        gast_hours=times.gast,
        gmst_hours=times.gmst,
        delta_t_s=times.delta_t,
    )
    print(f"astro_time.npz: {WIDE_EPOCH_COUNT} epochs (GAST/GMST/deltaT)")


def record_tetra3():
    import tetra3
    from tetra3.tetra3 import _key_to_index

    rng = np.random.default_rng(20260815)

    # --- hash-key goldens against the DB actually used by camera1 ---------
    db_path = os.path.join(
        os.path.dirname(tetra3.__file__), "data", "db_cam1_tyc.npz"
    )
    with np.load(db_path) as db:
        props = db["props_packed"]
        pattern_bins = int(props["pattern_bins"][()])
        catalog_length = int(db["pattern_catalog"].shape[0])
    n_keys = 1000
    # Edge-ratio keys: 5 ints in [0, 2*pattern_bins) as tetra3 quantizes them.
    keys = rng.integers(0, 2 * pattern_bins, size=(n_keys, 5), dtype=np.uint64)
    indices = np.array(
        [_key_to_index(tuple(k), pattern_bins, catalog_length) for k in keys],
        dtype=np.uint64,
    )
    # A second parameter set so the Rust port can't hardcode one geometry.
    indices_alt = np.array(
        [_key_to_index(tuple(k), 25, 1_000_003) for k in keys], dtype=np.uint64
    )

    # --- centroid goldens on synthetic Gaussian-spot frames ---------------
    imgs, cents = [], []
    for _ in range(5):
        img = rng.normal(8.0, 2.0, size=(512, 512))
        n_stars = 30
        xs = rng.uniform(20, 492, n_stars)
        ys = rng.uniform(20, 492, n_stars)
        amps = rng.uniform(40, 250, n_stars)
        sig = rng.uniform(1.0, 2.2, n_stars)
        yy, xx = np.mgrid[0:512, 0:512]
        for x, y, a, s in zip(xs, ys, amps, sig):
            img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))
        img = np.clip(img, 0, 255).astype(np.uint8)
        got = tetra3.get_centroids_from_image(img)
        imgs.append(img)
        cents.append(np.asarray(got, dtype=np.float64))
    max_c = max(c.shape[0] for c in cents)
    cent_pad = np.full((len(cents), max_c, 2), np.nan)
    for i, c in enumerate(cents):
        cent_pad[i, : c.shape[0]] = c

    np.savez_compressed(
        os.path.join(GOLDEN_DIR, "tetra3_hashes.npz"),
        keys=keys,
        pattern_bins=np.uint64(pattern_bins),
        catalog_length=np.uint64(catalog_length),
        indices=indices,
        alt_bins=np.uint64(25),
        alt_max_index=np.uint64(1_000_003),
        indices_alt=indices_alt,
    )
    np.savez_compressed(
        os.path.join(GOLDEN_DIR, "tetra3_centroids.npz"),
        images=np.stack(imgs),
        centroids_yx=cent_pad,  # (n_img, max_c, 2) NaN-padded, tetra3 (y, x) order
    )
    print(
        f"tetra3_hashes.npz: {n_keys} keys (bins={pattern_bins}, "
        f"catalog={catalog_length}); tetra3_centroids.npz: 5 frames"
    )


def record_spk_states():
    """Raw barycentric/relative states from every de421.bsp segment, for
    validating the Rust SPK (DAF type-2 Chebyshev) reader in isolation."""
    from jplephem.spk import SPK

    ts = _timescale()
    times = _wide_times(ts)[::2]  # 25 epochs
    # jplephem takes TDB Julian dates; skyfield's tdb = tt + tdb_minus_tt.
    jd_tdb = times.tdb

    spk = SPK.open(os.path.join(REPO, "de421.bsp"))
    pairs = [(s.center, s.target) for s in spk.segments]
    centers, targets = [], []
    pos = np.zeros((len(pairs), len(jd_tdb), 3))
    vel = np.zeros_like(pos)
    for i, (c, t) in enumerate(pairs):
        p, v = spk[c, t].compute_and_differentiate(jd_tdb)
        pos[i] = p.T
        vel[i] = (v / 86400.0).T  # km/day -> km/s
        centers.append(c)
        targets.append(t)
    spk.close()

    np.savez_compressed(
        os.path.join(GOLDEN_DIR, "spk_states.npz"),
        center=np.array(centers, dtype=np.int32),
        target=np.array(targets, dtype=np.int32),
        jd_tdb=jd_tdb,
        position_km=pos,
        velocity_km_s=vel,
    )
    print(f"spk_states.npz: {len(pairs)} segments x {len(jd_tdb)} epochs")


def record_cv2_ops():
    import cv2

    rng = np.random.default_rng(42)
    base = rng.normal(30, 6, size=(256, 256))
    yy, xx = np.mgrid[0:256, 0:256]
    for x, y, a in [(60, 50, 180), (190, 80, 220), (120, 200, 160), (40, 170, 140)]:
        base += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0**2))
    base = np.clip(base, 0, 255).astype(np.float32)

    # Phase correlation on known sub-pixel shifts, with Hanning window.
    shifts = [(3.25, -1.5), (-7.75, 4.5), (0.3, 0.7), (12.0, -9.25)]
    hann = cv2.createHanningWindow((256, 256), cv2.CV_32F)
    pc_measured = []
    shifted_imgs = []
    for dx, dy in shifts:
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        sh = cv2.warpAffine(
            base, m, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        # cv2.phaseCorrelate MUTATES its inputs in place (OpenCV 4.8 applies
        # the window into the caller's buffers) -- pass copies or every
        # later array in this recorder is silently computed from a
        # corrupted base. See LEARNINGS.md 2026-08-17.
        (mx, my), resp = cv2.phaseCorrelate(base.copy(), sh.copy(), hann)
        pc_measured.append([mx, my, resp])
        shifted_imgs.append(sh)

    # Filter kernels used by the stacking quality metrics.
    blur = cv2.GaussianBlur(base, (5, 5), 1.2)
    lap = cv2.Laplacian(base, cv2.CV_32F)
    sobx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)

    # Similarity-transform estimation with RANSAC on noisy correspondences.
    rng2 = np.random.default_rng(7)
    src = rng2.uniform(0, 256, size=(60, 2)).astype(np.float32)
    theta, scale, tx, ty = np.deg2rad(2.5), 1.02, 4.0, -3.0
    rot = scale * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    dst = (src @ rot.T + [tx, ty]).astype(np.float32)
    dst += rng2.normal(0, 0.3, dst.shape).astype(np.float32)
    dst[::10] += rng2.uniform(20, 40, dst[::10].shape).astype(np.float32)  # outliers
    mat, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0
    )

    # Pyramidal Lucas-Kanade on a known integer+subpixel translation.
    lk_shift = (5.5, -3.25)
    m = np.float32([[1, 0, lk_shift[0]], [0, 1, lk_shift[1]]])
    moved = cv2.warpAffine(
        base, m, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    p0 = cv2.goodFeaturesToTrack(
        base.astype(np.uint8), maxCorners=40, qualityLevel=0.05, minDistance=10
    )
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        base.astype(np.uint8), moved.astype(np.uint8), p0, None,
        winSize=(21, 21), maxLevel=3,
    )

    np.savez_compressed(
        os.path.join(GOLDEN_DIR, "cv2_ops.npz"),
        base=base,
        shifts_true=np.array(shifts),
        shifted=np.stack(shifted_imgs),
        phasecorr_measured=np.array(pc_measured),
        gaussian_blur_5x5_s1p2=blur,
        laplacian=lap,
        sobel_x=sobx,
        sobel_y=soby,
        affine_src=src,
        affine_dst=dst,
        affine_true=np.array([theta, scale, tx, ty]),
        affine_est=mat,
        affine_inliers=inliers.ravel(),
        lk_shift_true=np.array(lk_shift),
        lk_p0=p0.reshape(-1, 2),
        lk_p1=p1.reshape(-1, 2),
        lk_status=st.ravel(),
    )
    print("cv2_ops.npz: phasecorr/filters/RANSAC-affine/LK goldens")


def write_manifest():
    import cv2
    import skyfield
    import tetra3  # noqa: F401

    manifest = {
        "recorded_utc": None,  # filled by the committer; kept stable across reruns
        "site": {"lat": SITE_LAT_DEG, "lon": SITE_LON_DEG, "alt_m": SITE_ALT_M},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "skyfield": skyfield.__version__,
            "opencv": cv2.__version__,
        },
        "tolerances": {
            "sat_altaz_arcsec": 20,
            "body_star_altaz_arcsec": 60,
            "gast_seconds": 0.005,
            "tetra3_hash": "bit-exact",
            "centroid_px": 0.3,
            "phasecorr_px": 0.05,
            "affine_similarity": "0.1 px translation / 0.05 deg rotation",
        },
        "files": sorted(
            f for f in os.listdir(GOLDEN_DIR) if f.endswith(".npz")
        ),
    }
    with open(os.path.join(GOLDEN_DIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("MANIFEST.json written")


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    record_astro_sats()
    record_astro_bodies()
    record_astro_stars()
    record_astro_time()
    record_spk_states()
    record_tetra3()
    record_cv2_ops()
    write_manifest()
    total = sum(
        os.path.getsize(os.path.join(GOLDEN_DIR, f)) for f in os.listdir(GOLDEN_DIR)
    )
    print(f"total golden size: {total / 1e6:.1f} MB in {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
