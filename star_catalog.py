"""
Star catalogue for the skyplot and synthetic-image simulator.

Loads the Hipparcos catalogue (via Skyfield, already a project dependency) once and
serves the current topocentric (az, el) of the visible stars for a given observer and
time. The expensive part -- Skyfield's observe()/apparent()/altaz() over thousands of
stars -- is *anchored* infrequently; between anchors the whole sky is advanced by a
cheap rigid rotation about the celestial pole (diurnal motion is exactly Earth's
rotation), so callers may query every render frame without paying the Skyfield cost.

Conventions
-----------
Horizontal unit vectors are stored in an (East, North, Up) frame:
    E = cos(el) * sin(az)
    N = cos(el) * cos(az)
    U = sin(el)
with azimuth measured from North toward East. Positions are *geometric* (no
refraction), which keeps the sim end-to-end consistent with the unrefracted geometric
pointing model fitted in Phase 4.
"""

import math
import threading

import numpy as np

from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos

# Re-anchor (full Skyfield recompute) when the query time drifts this far from the
# anchor, or whenever the observer changes. The diurnal rotation is exact, so this
# bound is really about precession/nutation/aberration drift -- minutes are plenty.
ANCHOR_MAX_AGE_SEC = 600.0

# Hard cap on the bright subset that gets the full Skyfield treatment at each anchor,
# independent of the UI's max_rendered_star_count. Keeps anchor cost bounded.
ANCHOR_MAG_LIMIT = 7.5

# Common names for the brightest stars, keyed by Hipparcos (HIP) number. Used for the
# always-on labels on the brightest stars in view; everything else falls back to HIP id.
BRIGHT_STAR_NAMES = {
    32349: "Sirius", 30438: "Canopus", 71683: "Rigil Kent.", 69673: "Arcturus",
    91262: "Vega", 24608: "Capella", 24436: "Rigel", 37279: "Procyon",
    7588: "Achernar", 27989: "Betelgeuse", 68702: "Hadar", 97649: "Altair",
    21421: "Aldebaran", 65474: "Spica", 80763: "Antares", 37826: "Pollux",
    113368: "Fomalhaut", 102098: "Deneb", 49669: "Regulus", 33579: "Adhara",
    85927: "Shaula", 25336: "Bellatrix", 109268: "Alnair", 25428: "Elnath",
    27366: "Saiph", 26727: "Alnitak", 26311: "Alnilam", 8903: "Mirach",
    9884: "Hamal", 14576: "Algol", 11767: "Polaris", 62434: "Mimosa",
    72607: "Kochab", 60718: "Acrux", 59747: "Gacrux", 86032: "Rasalhague",
    100751: "Peacock", 677: "Alpheratz", 3419: "Diphda", 15863: "Mirfak",
    54061: "Dubhe", 67301: "Alkaid", 95947: "Tarazed", 28360: "Menkalinan",
}


class StarCatalog:
    """Loaded Hipparcos catalogue with a cached, time-advanced horizontal projection."""

    def __init__(self, ephemeris=None):
        # Earth ephemeris -- reuse the app's already-loaded de421 if provided.
        self._earth = (ephemeris if ephemeris is not None else load('de421.bsp'))['earth']

        with load.open(hipparcos.URL) as f:
            df = hipparcos.load_dataframe(f)
        df = df[np.isfinite(df['magnitude'].values)]
        self._df = df  # kept for Star.from_dataframe at anchor time

        self.hip = df.index.to_numpy()
        self.mag = df['magnitude'].to_numpy(dtype=np.float64)
        if 'ra_degrees' in df.columns:
            ra_deg = df['ra_degrees'].to_numpy(dtype=np.float64)
        else:
            ra_deg = df['ra_hours'].to_numpy(dtype=np.float64) * 15.0
        dec_deg = df['dec_degrees'].to_numpy(dtype=np.float64)
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg

        # Index of the bright subset that we actually project/anchor.
        self._bright_idx = np.where(self.mag <= ANCHOR_MAG_LIMIT)[0]
        self._bright_star = Star.from_dataframe(df.iloc[self._bright_idx])
        self.bright_mag = self.mag[self._bright_idx]
        self.bright_hip = self.hip[self._bright_idx]

        # Anchor cache (guarded; the render thread and main thread both call in).
        self._lock = threading.Lock()
        self._anchor_xyz = None       # (M, 3) horizontal unit vectors at anchor
        self._anchor_tt = None        # anchor time (TT Julian date)
        self._anchor_lst_rad = None   # local apparent sidereal time at anchor (radians)
        self._anchor_key = None       # (lat, lon, elev) the anchor was built for
        self._pole = None             # celestial-pole axis in (E,N,U) for this latitude

    # ---- public API ---------------------------------------------------------

    def current_altaz(self, lat_deg, lon_deg, elev_m, ts, t_tt,
                      elevation_mask=0.0, max_count=2000, limiting_mag=6.5):
        """Return the brightest visible stars for the given observer/time.

        Returns a dict with numpy arrays (parallel, length = number rendered):
            az, el, mag, hip  and scalars  cutoff_mag, n_visible, top_label_mask.
        top_label_mask is a boolean array marking the brightest ~5% (always labelled).
        """
        az, el = self._project(lat_deg, lon_deg, elev_m, ts, t_tt)
        mag = self.bright_mag

        visible = (el >= elevation_mask) & (mag <= limiting_mag)
        vidx = np.where(visible)[0]
        if vidx.size == 0:
            return {'az': np.empty(0), 'el': np.empty(0), 'mag': np.empty(0),
                    'hip': np.empty(0, dtype=np.int64), 'cutoff_mag': float('nan'),
                    'n_visible': 0, 'top_label_mask': np.empty(0, dtype=bool)}

        vmag = mag[vidx]
        if vidx.size > max_count:
            # brightest max_count by magnitude (smallest mag = brightest)
            keep = np.argpartition(vmag, max_count)[:max_count]
            vidx = vidx[keep]
            vmag = mag[vidx]

        cutoff_mag = float(vmag.max())

        # Top 5% brightest of the rendered set get permanent labels.
        n_label = max(1, int(math.ceil(vidx.size * 0.05)))
        thresh = np.partition(vmag, n_label - 1)[n_label - 1]
        top_label_mask = vmag <= thresh

        return {
            'az': az[vidx], 'el': el[vidx], 'mag': vmag,
            'hip': self.bright_hip[vidx],
            'cutoff_mag': cutoff_mag, 'n_visible': int(vidx.size),
            'top_label_mask': top_label_mask,
        }

    def stars_in_fov(self, lat_deg, lon_deg, elev_m, ts, t_tt,
                     boresight_az, boresight_el, half_fov_az, half_fov_el,
                     limiting_mag=9.0):
        """Catalogue stars within a rectangular FOV box around a boresight.

        Returns (az, el, mag) numpy arrays. Used by the synthetic-image simulator.
        """
        az, el = self._project(lat_deg, lon_deg, elev_m, ts, t_tt)
        mag = self.bright_mag
        d_az = ((az - boresight_az + 180.0) % 360.0) - 180.0
        d_el = el - boresight_el
        # Azimuth converges with elevation: a column of pixels spans half_fov_az/cos(el)
        # of azimuth. Comparing raw |d_az| to half_fov_az would drop stars that are
        # actually inside the frame (a phantom vertical wall at high elevation), so test
        # cross-elevation (|d_az|*cos(el)) against the azimuth half-FOV instead.
        cos_el = max(math.cos(math.radians(boresight_el)), 1e-3)
        in_box = ((np.abs(d_az) * cos_el <= half_fov_az) & (np.abs(d_el) <= half_fov_el)
                  & (mag <= limiting_mag))
        idx = np.where(in_box)[0]
        return az[idx], el[idx], mag[idx]

    def name_for(self, hip):
        """Human-friendly label for a star: common name if known, else 'HIP n'."""
        return BRIGHT_STAR_NAMES.get(int(hip), f"HIP {int(hip)}")

    # ---- internals ----------------------------------------------------------

    def _project(self, lat_deg, lon_deg, elev_m, ts, t_tt):
        """Current (az, el) of the bright subset, re-anchoring via Skyfield as needed."""
        with self._lock:
            key = (round(lat_deg, 6), round(lon_deg, 6), round(elev_m, 1))
            t = ts.tt(jd=t_tt)
            need_anchor = (
                self._anchor_xyz is None
                or self._anchor_key != key
                or abs((t_tt - self._anchor_tt) * 86400.0) > ANCHOR_MAX_AGE_SEC
            )
            if need_anchor:
                self._build_anchor(lat_deg, lon_deg, elev_m, ts, t, key)

            # Advance the anchored sky by the change in local sidereal time. Diurnal
            # motion is a rigid rotation of the celestial sphere about the pole.
            lst_rad = self._lst_rad(t, lon_deg)
            dtheta = lst_rad - self._anchor_lst_rad
            xyz = _rotate_about_axis(self._anchor_xyz, self._pole, dtheta)
            return _altaz_from_xyz(xyz)

    def _build_anchor(self, lat_deg, lon_deg, elev_m, ts, t, key):
        topos = wgs84.latlon(lat_deg, lon_deg, elevation_m=elev_m)
        observer = self._earth + topos
        alt, az, _ = observer.at(t).observe(self._bright_star).apparent().altaz()
        az_rad = az.radians
        el_rad = alt.radians
        self._anchor_xyz = _xyz_from_altaz(az_rad, el_rad)
        self._anchor_tt = t.tt
        self._anchor_lst_rad = self._lst_rad(t, lon_deg)
        self._anchor_key = key
        phi = math.radians(lat_deg)
        # Celestial pole direction in (E, N, U). For the southern hemisphere phi<0
        # points the axis below the horizon toward the south pole, which is correct.
        self._pole = np.array([0.0, math.cos(phi), math.sin(phi)])

    @staticmethod
    def _lst_rad(t, lon_deg):
        """Local apparent sidereal time in radians for the given Skyfield time."""
        lst_hours = (t.gast + lon_deg / 15.0) % 24.0
        return math.radians(lst_hours * 15.0)


# ---- vector helpers (vectorised, no Skyfield) -------------------------------

def _xyz_from_altaz(az_rad, el_rad):
    """(az, el) radians -> (N,3) unit vectors in (East, North, Up)."""
    cos_el = np.cos(el_rad)
    return np.stack([cos_el * np.sin(az_rad), cos_el * np.cos(az_rad), np.sin(el_rad)], axis=-1)


def _altaz_from_xyz(xyz):
    """(N,3) (East, North, Up) unit vectors -> (az_deg, el_deg) arrays."""
    e, n, u = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el = np.degrees(np.arcsin(np.clip(u, -1.0, 1.0)))
    az = np.degrees(np.arctan2(e, n)) % 360.0
    return az, el


# Sign of the sidereal advance about the pole axis. Stars rise in the east and set in
# the west; this constant orients the rotation to match Skyfield (verified by
# test_star_catalog.py comparing the fast path to a full Skyfield recompute).
_SIDEREAL_SIGN = -1.0


def _rotate_about_axis(vectors, axis, angle_rad):
    """Rodrigues rotation of (N,3) vectors about a unit axis by angle_rad."""
    theta = _SIDEREAL_SIGN * angle_rad
    k = axis / (np.linalg.norm(axis) + 1e-12)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dot = vectors @ k
    cross = np.cross(np.broadcast_to(k, vectors.shape), vectors)
    return (vectors * cos_t
            + cross * sin_t
            + np.outer(dot * (1.0 - cos_t), k))


def _xyz_from_radec_deg(ra_deg, dec_deg):
    """(ra, dec) degrees -> ICRF unit vector(s)."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)], axis=-1)


# Default location of the Tycho catalogue (downloaded from CDS I/239). Deep enough
# (complete to mag ~10) for the narrow-FOV cameras Hipparcos can't populate.
TYCHO_PATH = "tyc_main.dat"


class DeepStarCatalog:
    """Tycho-backed deep catalogue for narrow-FOV synthetic imagery and solving.

    Avoids a full-sky altaz pass: for each request it converts the boresight to RA/Dec
    (Skyfield), masks the catalogue to a cone around it (a cheap dot product), and runs
    Skyfield altaz on only that small subset. So it stays fast even at ~1M stars.
    """

    def __init__(self, ephemeris=None, path=TYCHO_PATH, mag_limit=11.0):
        import pandas as pd
        self._earth = (ephemeris if ephemeris is not None else load('de421.bsp'))['earth']
        # tyc_main.dat is pipe-delimited; col 5 = VT mag, 8 = RA deg, 9 = Dec deg.
        df = pd.read_csv(path, sep='|', header=None, usecols=[5, 8, 9],
                         names=['mag', 'ra', 'dec'], engine='c', dtype=str)
        mag = pd.to_numeric(df['mag'], errors='coerce').to_numpy()
        ra = pd.to_numeric(df['ra'], errors='coerce').to_numpy()
        dec = pd.to_numeric(df['dec'], errors='coerce').to_numpy()
        ok = np.isfinite(mag) & np.isfinite(ra) & np.isfinite(dec) & (mag <= mag_limit)
        self.mag = mag[ok].astype(np.float64)
        self.ra_deg = ra[ok].astype(np.float64)
        self.dec_deg = dec[ok].astype(np.float64)
        self.xyz = _xyz_from_radec_deg(self.ra_deg, self.dec_deg)
        self._observers = {}
        self._lock = threading.Lock()

    def _observer(self, lat_deg, lon_deg, elev_m):
        key = (round(lat_deg, 6), round(lon_deg, 6), round(elev_m, 1))
        o = self._observers.get(key)
        if o is None:
            o = self._earth + wgs84.latlon(lat_deg, lon_deg, elevation_m=elev_m)
            self._observers[key] = o
        return o

    def stars_in_fov(self, lat_deg, lon_deg, elev_m, ts, t_tt,
                     boresight_az, boresight_el, half_fov_az, half_fov_el,
                     limiting_mag=11.0, max_stars=1500):
        """(az, el, mag) of catalogue stars near the boresight (cone-masked)."""
        with self._lock:
            observer = self._observer(lat_deg, lon_deg, elev_m)
            t = ts.tt(jd=t_tt)
            # Boresight -> apparent RA/Dec -> ICRF unit vector for the cone mask.
            ra_b, dec_b, _ = observer.at(t).from_altaz(
                alt_degrees=boresight_el, az_degrees=boresight_az).radec()
            rb, db = ra_b.radians, dec_b.radians
            b = np.array([math.cos(db) * math.cos(rb),
                          math.cos(db) * math.sin(rb), math.sin(db)])
            cone = math.radians(max(half_fov_az, half_fov_el) * 1.8 + 0.3)
            mask = (self.xyz @ b > math.cos(cone)) & (self.mag <= limiting_mag)
            idx = np.where(mask)[0]
            if idx.size == 0:
                return np.empty(0), np.empty(0), np.empty(0)
            if idx.size > max_stars:
                idx = idx[np.argsort(self.mag[idx])[:max_stars]]  # keep brightest
            sub = Star(ra_hours=self.ra_deg[idx] / 15.0, dec_degrees=self.dec_deg[idx])
            alt, az, _ = observer.at(t).observe(sub).apparent().altaz()
            return az.degrees, alt.degrees, self.mag[idx]


# ---- module singletons ------------------------------------------------------

_CATALOG = None
_DEEP_CATALOG = None
_DEEP_TRIED = False
_CATALOG_LOCK = threading.Lock()


def get_catalog(ephemeris=None):
    """Lazily build and return the shared StarCatalog singleton."""
    global _CATALOG
    if _CATALOG is None:
        with _CATALOG_LOCK:
            if _CATALOG is None:
                _CATALOG = StarCatalog(ephemeris=ephemeris)
    return _CATALOG


def _load_deep_catalog_blocking(ephemeris, path):
    global _DEEP_CATALOG
    try:
        cat = DeepStarCatalog(ephemeris=ephemeris, path=path)
        _DEEP_CATALOG = cat
        print(f"Deep star catalogue (Tycho) loaded: {cat.mag.size} stars")
    except Exception as e:
        print(f"Deep star catalogue load failed: {e}")


def get_deep_catalog(ephemeris=None, path=TYCHO_PATH, blocking=False):
    """Return the Tycho deep catalogue, or None if not present / not loaded yet.

    Parsing ~1M Tycho rows takes a few seconds, so by default the first call kicks off
    a BACKGROUND load and returns None immediately -- callers (the render thread) must
    never block on it. Subsequent calls return the catalogue once it's ready. Pass
    blocking=True to wait (e.g. for plate-solving / tests).
    """
    global _DEEP_CATALOG, _DEEP_TRIED
    if _DEEP_CATALOG is not None:
        return _DEEP_CATALOG
    import os
    with _CATALOG_LOCK:
        if _DEEP_CATALOG is None and not _DEEP_TRIED:
            _DEEP_TRIED = True
            if os.path.exists(path):
                if blocking:
                    _load_deep_catalog_blocking(ephemeris, path)
                else:
                    threading.Thread(target=_load_deep_catalog_blocking,
                                     args=(ephemeris, path), daemon=True,
                                     name="DeepStarCatalogLoad").start()
    return _DEEP_CATALOG
