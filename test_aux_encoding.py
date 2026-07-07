#!/usr/bin/env python
"""
Golden-byte regression tests for NexStar AUX command *encoding*.

These exist because the simulator overrides the hand-controller methods at the
Python level, so a broken command builder can pass every sim test and still
crash (or mis-command the mount) the first time it touches real hardware.
Two such bugs motivated this file:

  * hc_set_guide_rate's variable-rate branch formatted raw bytes with
    '{:06x}' -> TypeError on every non-sidereal call (the continuous-rate
    tracking / alignment fine-slew path),
  * dms2f treated arcminutes/arcseconds as fractions of a rotation instead of
    fractions of a degree (dms2f(0, 30, 0) came out as 180 degrees).

Every test here drives the real NexstarHandController against a byte-capturing
fake device -- no sim overrides anywhere in the path.
Run: python test_aux_encoding.py
"""

import unittest

from lib.auxstar import (
    NexstarHandController,
    Targets,
    dms2f,
    f2dms,
    pack_int3,
)


class ByteCaptureSerial:
    """Fake serial device that records every request and acks with '#'."""

    def __init__(self):
        self.writes = []

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def read(self, n):
        return b'#' if n else b''

    def reset_input_buffer(self):
        pass


class GuideRateEncodingTests(unittest.TestCase):

    def setUp(self):
        self.dev = ByteCaptureSerial()
        self.ctrl = NexstarHandController(self.dev)

    def test_variable_rate_positive_golden_bytes(self):
        # 0.001 rev/s -> pack_int3 gives the 3 data bytes on the wire.
        ok = self.ctrl.hc_set_guide_rate(Targets.AZM, 0.001)
        self.assertTrue(ok)
        (msg,) = self.dev.writes
        expected_data = pack_int3(0.001)
        # 'P', len=4, target=AZM(0x10), MC_SET_POS_GUIDERATE(0x06), 3 data
        # bytes, expected-response-count 0.
        self.assertEqual(
            msg, bytes([0x50, 0x04, 0x10, 0x06]) + expected_data + bytes([0x00]))

    def test_variable_rate_negative_uses_neg_command_and_magnitude(self):
        self.ctrl.hc_set_guide_rate(Targets.ALT, -0.002)
        (msg,) = self.dev.writes
        self.assertEqual(msg[3], 0x07, "negative rate must use MC_SET_NEG_GUIDERATE")
        self.assertEqual(msg[4:7], pack_int3(0.002),
                         "magnitude (not the sign-corrupted value) is packed")

    def test_sidereal_golden_bytes(self):
        self.ctrl.hc_set_guide_rate(Targets.AZM, 1, sidereal=True)
        (msg,) = self.dev.writes
        self.assertEqual(msg, bytes([0x50, 0x04, 0x10, 0x06, 0xff, 0xff, 0x00, 0x00]))

    def test_set_rate_dps_scale_convention(self):
        # hc_set_rate_dps packs dps/360 (rev/sec) -- the convention shared with
        # the RATES table. The on-wire *scale* still needs bench_guiderate.py
        # on real hardware; this pins the code's intended convention.
        self.ctrl.hc_set_rate_dps(Targets.AZM, 0.36)
        (msg,) = self.dev.writes
        self.assertEqual(msg[4:7], pack_int3(0.36 / 360.0))

    def test_every_rate_magnitude_encodes_without_error(self):
        # The old '{:06x}'.format(bytes) crashed on every call; sweep a range
        # of realistic tracking rates to keep the whole branch exercised.
        for i, dps in enumerate([0.0001, 0.01, 0.1, 1.0, 4.0, -0.05, -2.5]):
            self.assertTrue(self.ctrl.hc_set_rate_dps(Targets.AZM, dps))
        self.assertEqual(len(self.dev.writes), 7)
        for msg in self.dev.writes:
            self.assertEqual(len(msg), 8)
            self.assertEqual(msg[0], 0x50)

    def test_set_position_encodes_without_error(self):
        # Regression: an extra {:02x} placeholder made every call raise
        # IndexError before it touched the wire.
        self.assertTrue(self.ctrl.hc_set_position(Targets.AZM, 45, 0, 0))
        (msg,) = self.dev.writes
        self.assertEqual(msg[3], 0x04)  # MC_SET_POSITION
        self.assertEqual(msg[4:7], pack_int3(45 / 360.0))


class Dms2fTests(unittest.TestCase):

    def test_minutes_are_fractions_of_a_degree(self):
        # 0 deg 30' = half a degree, NOT half a rotation.
        self.assertAlmostEqual(dms2f(0, 30, 0), 0.5 / 360.0, places=12)

    def test_seconds_are_fractions_of_a_minute(self):
        self.assertAlmostEqual(dms2f(0, 0, 36), (36 / 3600.0) / 360.0, places=12)

    def test_whole_degrees_unchanged(self):
        # The (dd, 0, 0) form every current caller uses must be unaffected.
        self.assertAlmostEqual(dms2f(45, 0, 0), 45 / 360.0, places=12)
        self.assertAlmostEqual(dms2f(-45, 0, 0), -45 / 360.0, places=12)

    def test_roundtrip_with_f2dms(self):
        # Note: f2dms carries the sign on the degrees term, so a negative angle
        # smaller than 1 degree (dd == 0) loses its sign in the DMS triple --
        # a long-standing quirk of the display formatter, not of dms2f. Keep
        # the negative cases at >= 1 degree.
        for f in [0.125, 0.7431, -0.25, -0.0035, 0.999]:
            d, m, s = f2dms(f)
            self.assertAlmostEqual(dms2f(d, m, s), f, places=9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
