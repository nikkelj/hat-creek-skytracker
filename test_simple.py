#!/usr/bin/env python3

from transformations import AzAlt2AzEl

# Test the problematic case
AZM = 0.0
ALT = 0.0
alignment_azimuth = 266.0
alignment_elevation = 34.874

result_az, result_el = AzAlt2AzEl(AZM, ALT, alignment_azimuth, alignment_elevation)

print(f"Input: AZM=0°, ALT=0°, alignment_azimuth=266°, alignment_elevation=34.874°")
print(f"Output: az={result_az:.3f}°, el={result_el:.3f}°")
print(f"Expected: az≈266°, el≈34.874°")
print(f"Error: Δaz={abs(result_az-266):.3f}°, Δel={abs(result_el-34.874):.3f}°")
