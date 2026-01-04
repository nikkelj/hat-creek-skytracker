#!/usr/bin/env python3
"""
Debug script to understand the transformation behavior
"""

import numpy as np
from transformations import AzAlt2AzEl, AzEl2AzAlt

# Test the original problematic case
alignment_azimuth = 266.0
alignment_elevation = 34.874

print("=" * 60)
print("DEBUGGING ORIGINAL TEST CASE")
print("=" * 60)

# Test AZM=0, ALT=0 - should map to alignment position
azm, alt = 0.0, 0.0
result_az, result_el = AzAlt2AzEl(azm, alt, alignment_azimuth, alignment_elevation)
print("Original test case:")
print(".0f")
print(".6f")

# Test a few other points
test_points = [(30, 10), (-45, 25), (180, -15), (90, 40)]

print("\n" + "=" * 60)
print("FORWARD TRANSFORMATION (AzAlt2AzEl)")
print("=" * 60)

for azm, alt in test_points:
    result_az, result_el = AzAlt2AzEl(azm, alt, alignment_azimuth, alignment_elevation)
    print(".0f")

print("\n" + "=" * 60)
print("ROUND-TRIP TESTS (AzAlt2AzEl → AzEl2AzAlt → AzAlt2AzEl)")
print("=" * 60)

for azm, alt in test_points:
    # Forward transformation
    az1, el1 = AzAlt2AzEl(azm, alt, alignment_azimuth, alignment_elevation)

    # Inverse transformation
    azm2, alt2 = AzEl2AzAlt(az1, el1, alignment_azimuth, alignment_elevation)

    # Round-trip back
    az3, el3 = AzAlt2AzEl(azm2, alt2, alignment_azimuth, alignment_elevation)

    print(".0f")
    print("           Inverse: AZM={azm2:.2f}°, ALT={alt2:.2f}°".format(azm2=azm2, alt2=alt2))
    print("        Round-trip: AZ={az3:.3f}°, EL={el3:.3f}°".format(az3=az3, el3=el3))
    print()

# Test inverse directly on alignment position - azm, alt should be 0, 0
print("=" * 60)
print("TESTING INVERSE ON ALIGNMENT POSITION")
print("=" * 60)

test_alignments = [
    (alignment_azimuth, alignment_elevation, "Original case"),
    (0, 90, "North zenith"),
    (90, 45, "East-mid elevation"),
    (270, 60, "West-high elevation"),
]

for test_az, test_el, name in test_alignments:
    azm_inv, alt_inv = AzEl2AzAlt(test_az, test_el, alignment_azimuth, alignment_elevation)
    print("29"
    print("                            Inverse: AZM={azm_inv:.2f}°, ALT={alt_inv:.2f}°".format(azm_inv=azm_inv, alt_inv=alt_inv))
