#!/usr/bin/env python3
"""
Sign-convention regression pins for the *general* quaternion transform
(AzAlt2AzEl). Converted from a print-and-eyeball script to real asserts.

NOTE: this transform is NOT in the control path (control.py uses the
AltAz/Passthrough/Eq branches); it survives in rendering and debug tooling.
Its observed behavior, pinned here so a refactor can't change it silently:

  * (AZM=0, ALT=0) maps exactly to the alignment point;
  * at AZM=0, +ALT raises sky elevation and -ALT lowers it;
  * the ALT sign FLIPS with azimuth (at AZM=90 the same +ALT lowers
    elevation), and AZM alone does not move the output at ALT=0. That is a
    long-standing quirk of this two-to-one inverse -- the original eyeball
    script would have shown "INVERTED" for half its azimuth sweep. If this
    transform is ever promoted into a pointing path, fix the convention and
    update these pins deliberately.
"""

import unittest

from transformations import AzAlt2AzEl

ALIGN_AZ = 45.0
ALIGN_EL = 30.0


class AltSignConventionTests(unittest.TestCase):

    def test_zero_maps_to_alignment(self):
        az0, el0 = AzAlt2AzEl(0.0, 0.0, ALIGN_AZ, ALIGN_EL)
        self.assertAlmostEqual(az0 % 360.0, ALIGN_AZ, places=6)
        self.assertAlmostEqual(el0, ALIGN_EL, places=6)

    def test_alt_sign_at_azimuth_zero(self):
        _, el_base = AzAlt2AzEl(0.0, 0.0, ALIGN_AZ, ALIGN_EL)
        _, el_up = AzAlt2AzEl(0.0, 10.0, ALIGN_AZ, ALIGN_EL)
        _, el_dn = AzAlt2AzEl(0.0, -10.0, ALIGN_AZ, ALIGN_EL)
        self.assertGreater(el_up, el_base, "+ALT must raise elevation at AZM=0")
        self.assertLess(el_dn, el_base, "-ALT must lower elevation at AZM=0")

    def test_known_quirk_alt_sign_flips_with_azimuth(self):
        # Pin (not endorse) the quirk: at AZM=90 the elevation response to
        # +ALT is inverted relative to AZM=0. A deliberate fix should flip
        # this assertion, not trip over it by surprise.
        _, el_base = AzAlt2AzEl(90.0, 0.0, ALIGN_AZ, ALIGN_EL)
        _, el_up = AzAlt2AzEl(90.0, 10.0, ALIGN_AZ, ALIGN_EL)
        self.assertLess(el_up, el_base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
