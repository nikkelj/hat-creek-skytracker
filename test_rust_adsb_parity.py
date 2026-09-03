#!/usr/bin/env python
"""Live A/B parity: Rust Mode-S decode vs pyModeS through the real
adsb_receiver.decode_adsb_message entry point.

The corpus is generated: random ident/position/velocity DF17 payloads
with valid CRCs (computed via pyModeS's own encoder mode), so both
decoders face hundreds of distinct frames, plus the classic reference
messages. Geometry (geodetic->az/el/range) is A/B'd against the numpy
implementation.

Build first: cd rust/skytracker-ffi && maturin develop --release
"""

import os
import unittest

import numpy as np

try:
    import skytracker_core
    _HAVE = getattr(skytracker_core, "ADSB_AVAILABLE", False)
except ImportError:
    _HAVE = False

try:
    import pyModeS as pms
    _HAVE_PMS = True
except Exception:
    _HAVE_PMS = False

import adsb_receiver

REF_LAT, REF_LON = 34.874, -120.446


def _finish(payload22):
    """22 hex chars of DF17 header+ME -> full 28-char message with valid CRC."""
    crc = pms.crc(payload22 + "000000", encode=True)
    return (payload22 + f"{crc:06X}").upper()


def _make_corpus(rng, n_each=80):
    msgs = []
    for _ in range(n_each):
        # ident: TC 1-4, random category + callsign chars
        tc = rng.integers(1, 5)
        cat = rng.integers(0, 8)
        chars = rng.integers(0, 64, 8)
        me = (int(tc) << 51) | (int(cat) << 48)
        for i, c in enumerate(chars):
            me |= int(c) << (42 - i * 6)
        icao = rng.integers(0, 1 << 24)
        msgs.append(_finish(f"8D{int(icao):06X}{me:014X}"))
    for _ in range(n_each):
        # airborne position: TC 9-18, Q-bit altitudes, CPR near the ref
        tc = rng.integers(9, 19)
        alt_ft = rng.integers(500, 45000)
        n = (int(alt_ft) + 1000) // 25
        altcode12 = ((n >> 4) << 5) | (1 << 4) | (n & 0xF)
        odd = rng.integers(0, 2)
        cprlat = rng.integers(0, 1 << 17)
        cprlon = rng.integers(0, 1 << 17)
        me = (int(tc) << 51) | (int(altcode12) << 36) | (int(odd) << 34) \
            | (int(cprlat) << 17) | int(cprlon)
        icao = rng.integers(0, 1 << 24)
        msgs.append(_finish(f"8D{int(icao):06X}{me:014X}"))
    for _ in range(n_each):
        # velocity: TC 19 subtype 1 (and some 3), random components
        sub = 1 if rng.random() < 0.8 else 3
        vew_sign = rng.integers(0, 2)
        vew = rng.integers(1, 1024)
        vns_sign = rng.integers(0, 2)
        vns = rng.integers(1, 1024)
        vr_sign = rng.integers(0, 2)
        vr = rng.integers(0, 512)
        me = (19 << 51) | (int(sub) << 48) \
            | (int(vew_sign) << 42) | (int(vew) << 32) \
            | (int(vns_sign) << 31) | (int(vns) << 21) \
            | (int(vr_sign) << 19) | (int(vr) << 10)
        icao = rng.integers(0, 1 << 24)
        msgs.append(_finish(f"8D{int(icao):06X}{me:014X}"))
    # The classic pyModeS reference messages.
    msgs += ["8D40621D58C382D690C8AC2863A7", "8D4840D6202CC371C32CE0576098"]
    return msgs


@unittest.skipUnless(_HAVE, "skytracker_core wheel lacks ADS-B decode")
@unittest.skipUnless(_HAVE_PMS, "pyModeS not installed")
class AdsbParity(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("SKYTRACKER_RUST_ADSB", None)

    def test_decode_corpus_matches(self):
        rng = np.random.default_rng(20260818)
        corpus = _make_corpus(rng)
        n_decoded = 0
        kinds = {"ident": 0, "position": 0, "velocity": 0}
        for msg in corpus:
            os.environ["SKYTRACKER_RUST_ADSB"] = "0"
            py = adsb_receiver.decode_adsb_message(msg, REF_LAT, REF_LON)
            os.environ["SKYTRACKER_RUST_ADSB"] = "1"
            rs = adsb_receiver.decode_adsb_message(msg, REF_LAT, REF_LON)
            if py is None:
                self.assertIsNone(rs, msg)
                continue
            self.assertIsNotNone(rs, f"{msg}: python decoded, rust did not: {py}")
            n_decoded += 1
            kinds[py["kind"]] += 1
            self.assertEqual(py["kind"], rs["kind"], msg)
            self.assertEqual(py["icao"].upper(), rs["icao"].upper(), msg)
            for k in py:
                if k in ("kind", "icao"):
                    continue
                a, b = py[k], rs[k]
                if a is None or b is None:
                    self.assertEqual(a, b, f"{msg} field {k}: {a} vs {b}")
                elif isinstance(a, str):
                    self.assertEqual(a, b, f"{msg} field {k}")
                else:
                    self.assertLess(abs(float(a) - float(b)), 1e-9,
                                    f"{msg} field {k}: {a} vs {b}")
        self.assertGreater(kinds["ident"], 30)
        self.assertGreater(kinds["position"], 30)
        self.assertGreater(kinds["velocity"], 30)
        print(f"\n[parity] adsb decode: {n_decoded}/{len(corpus)} decoded "
              f"identically ({kinds})")

    def test_geometry_matches(self):
        rng = np.random.default_rng(5)
        worst = 0.0
        for _ in range(200):
            lat = REF_LAT + rng.uniform(-3, 3)
            lon = REF_LON + rng.uniform(-3, 3)
            alt = rng.uniform(0, 15000)
            py = adsb_receiver.geodetic_to_azel_range(
                lat, lon, alt, REF_LAT, REF_LON, 100.0)
            rs = skytracker_core.adsb_geodetic_to_azel_range(
                lat, lon, alt, REF_LAT, REF_LON, 100.0)
            for a, b in zip(py, rs):
                worst = max(worst, abs(a - b))
        self.assertLess(worst, 1e-9)
        print(f"[parity] adsb geometry: worst diff {worst:.2e} over 200 targets")


if __name__ == "__main__":
    unittest.main(verbosity=2)
