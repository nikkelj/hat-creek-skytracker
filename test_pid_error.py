#!/usr/bin/env python3
"""
Test script to verify PID error calculation for automatic tracking
"""

import numpy as np
from transformations import AzAlt2AzEl, AzEl2AzAlt
from control import compute_mount_position_error

class MockConfigState:
    def __init__(self, alignment_az=45.0, alignment_el=30.0):
        self.alignment_azimuth_str = str(alignment_az)
        self.alignment_elevation_str = str(alignment_el)

def test_error_calculation():
    """Test that error calculation produces correct signs for PID control"""

    config = MockConfigState(alignment_az=45.0, alignment_el=30.0)

    # Test case 1: Telescope pointing east of alignment (AZM > 0), target is west
    print("=== Test Case 1: AZM correction ===")
    current_azm = 10.0  # Telescope pointing 10° east of alignment (AZM=10°)
    current_alt = 0.0   # Level (ALT=0°)

    # Convert to sky coordinates
    current_az, current_el = AzAlt2AzEl(current_azm, current_alt, 45.0, 30.0)
    print(f"Current mount: AZM={current_azm}°, ALT={current_alt}°")
    print(f"Current sky: AZ={current_az:.1f}°, EL={current_el:.1f}°")

    # Target: Alignment position (AZ=45°, EL=30°)
    target_az, target_el = 45.0, 30.0
    print(f"Target sky: AZ={target_az}°, EL={target_el}°")

    # Compute error using corrected function
    azm_error, alt_error = compute_mount_position_error(
        config, current_azm, current_alt, target_az, target_el
    )
    print(f"PID errors: AZM={azm_error:.2f}°, ALT={alt_error:.2f}°")
    print("Expected: AZM negative (should move telescope left/west)")
    print()

    # Test case 2: ALT correction
    print("=== Test Case 2: ALT correction ===")
    current_azm = 0.0   # At alignment AZM
    current_alt = -5.0  # Telescope pointing 5° below alignment (ALT=-5°)

    # Convert to sky coordinates
    current_az, current_el = AzAlt2AzEl(current_azm, current_alt, 45.0, 30.0)
    print(f"Current mount: AZM={current_azm}°, ALT={current_alt}°")
    print(f"Current sky: AZ={current_az:.1f}°, EL={current_el:.1f}°")

    # Target: Alignment position (AZ=45°, EL=30°)
    target_az, target_el = 45.0, 30.0
    print(f"Target sky: AZ={target_az}°, EL={target_el}°")

    # Compute error
    azm_error, alt_error = compute_mount_position_error(
        config, current_azm, current_alt, target_az, target_el
    )
    print(f"PID errors: AZM={azm_error:.2f}°, ALT={alt_error:.2f}°")
    print("Expected: ALT positive (should move telescope up)")
    print()

    # Test case 3: Verify that target -> mount -> back conversion works
    print("=== Test Case 3: Round-trip consistency ===")
    target_mount_azm, target_mount_alt = AzEl2AzAlt(45.0, 30.0, 45.0, 30.0)
    print("Target sky (45°, 30°) -> Mount coordinates:", target_mount_azm, target_mount_alt)

    # Test that we compute zero error when telescope is already at target
    azm_error, alt_error = compute_mount_position_error(
        config, target_mount_azm, target_mount_alt, 45.0, 30.0
    )
    print(f"Zero error test: AZM={azm_error:.6f}°, ALT={alt_error:.6f}°")

if __name__ == "__main__":
    test_error_calculation()
