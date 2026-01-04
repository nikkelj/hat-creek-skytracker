#!/usr/bin/env python3
"""
Test script to verify ALT axis rotation sign convention
"""

import numpy as np
from transformations import AzAlt2AzEl

def test_alt_sign_convention():
    """Test to verify the sign convention for ALT axis rotation"""

    # Assuming a typical telescope alignment (pointed slightly north-east)
    alignment_az = 45.0  # degrees
    alignment_el = 30.0  # degrees

    print("Testing ALT axis sign convention")
    print(f"Telescope aligned to AZ={alignment_az}°, EL={alignment_el}°")
    print("When AZM=0°, ALT=0°:")
    az0, el0 = AzAlt2AzEl(0.0, 0.0, alignment_az, alignment_el)
    print(f"  Sky coordinates: AZ={az0:.1f}°, EL={el0:.1f}°")
    print()

    # Test positive ALT rotation
    print("Testing positive ALT rotation (AXIS=0°, ALT=+10°):")
    az_pos10, el_pos10 = AzAlt2AzEl(0.0, 10.0, alignment_az, alignment_el)
    delta_el_pos = el_pos10 - el0
    print(f"  Sky coordinates: AZ={az_pos10:.1f}°, EL={el_pos10:.1f}°")
    print(f"  Elevation change: {delta_el_pos:.2f}°")
    print("  Expected: Positive ALT should increase elevation ↑")
    print(f"  Result: {'✓ CORRECT' if delta_el_pos > 0 else '✗ INVERTED'}")
    print()

    # Test negative ALT rotation
    print("Testing negative ALT rotation (AXIS=0°, ALT=-10°):")
    az_neg10, el_neg10 = AzAlt2AzEl(0.0, -10.0, alignment_az, alignment_el)
    delta_el_neg = el_neg10 - el0
    print(f"  Sky coordinates: AZ={az_neg10:.1f}°, EL={el_neg10:.1f}°")
    print(f"  Elevation change: {delta_el_neg:.2f}°")
    print("  Expected: Negative ALT should decrease elevation ↓")
    print(f"  Result: {'✓ CORRECT' if delta_el_neg < 0 else '✗ INVERTED'}")
    print()

    # Test at different azimuth positions
    print("Testing at different azimuth angles...")
    for test_az in [0, 90, 180, 270]:
        az_at_azm, el_at_azm = AzAlt2AzEl(test_az, 0.0, alignment_az, alignment_el)
        az_at_azm_plus10, el_at_azm_plus10 = AzAlt2AzEl(test_az, 10.0, alignment_az, alignment_el)
        az_at_azm_minus10, el_at_azm_minus10 = AzAlt2AzEl(test_az, -10.0, alignment_az, alignment_el)

        delta_el_plus = el_at_azm_plus10 - el_at_azm
        delta_el_minus = el_at_azm_minus10 - el_at_azm

        print(".1f")
        print(f"    +10° ALT: {delta_el_plus:+.2f}° {'✓' if delta_el_plus > 0 else '✗'}")
        print(f"    -10° ALT: {delta_el_minus:+.2f}° {'✓' if delta_el_minus < 0 else '✗'}")
        print()

if __name__ == "__main__":
    test_alt_sign_convention()
