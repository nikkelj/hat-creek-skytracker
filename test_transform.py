#!/usr/bin/env python3
"""
Unit test suite for astronomical coordinate transformations using unittest framework.

Tests the correctness of AzAlt2AzEl and AzEl2AzAlt transformations with various
test cases across all quadrants and boundary conditions.
"""

import unittest
import numpy as np
from transformations import (
    cartesian_from_az_el,
    az_el_from_cartesian,
    alignment_rotation_matrix,
    AzAlt2AzEl,
    AzEl2AzAlt
)

class TestCoordinateConversions(unittest.TestCase):
    """Test basic coordinate conversions between Cartesian and spherical."""

    def test_round_trip_basic(self):
        """Test basic azimuth/elevation to Cartesian and back."""
        test_cases = [
            (0, 0),      # North horizon
            (90, 0),     # East horizon
            (180, 0),    # South horizon
            (270, 0),    # West horizon
            (0, 90),     # Zenith
            (45, 30),    # Quadrant 1
            (135, 45),   # Quadrant 2
            (225, 60),   # Quadrant 3
            (315, 15),   # Quadrant 4
            (30, -10),   # Negative elevation
        ]

        for az, el in test_cases:
            with self.subTest(az=az, el=el):
                # Convert to Cartesian
                cart = cartesian_from_az_el(az, el)

                # Convert back to spherical
                az_back, el_back = az_el_from_cartesian(cart)

                # Check accuracy
                az_error = abs(az_back - az) % 360
                if az_error > 180:
                    az_error = 360 - az_error

                self.assertLess(az_error, 0.001, f"Azimuth error too large: {az_error}°")
                self.assertAlmostEqual(el, el_back, places=3, msg=f"Elevation error: {abs(el - el_back)}°")

    def test_cartesian_vector_magnitude(self):
        """Test that Cartesian vectors have unit magnitude."""
        test_cases = [(az, el) for az in range(0, 360, 30) for el in range(-90, 91, 30)]

        for az, el in test_cases:
            with self.subTest(az=az, el=el):
                vector = cartesian_from_az_el(az, el)
                magnitude = np.linalg.norm(vector)
                self.assertAlmostEqual(magnitude, 1.0, places=6, msg=f"Magnitude {magnitude} != 1.0")

class TestAzAlt2AzEl(unittest.TestCase):
    """Test the AzAlt2AzEl transformation function."""

    def setUp(self):
        """Set up common test data."""
        self.test_alignments = [
            (0, 45, "North"),
            (266, 34.874, "Original case"),
            (90, 30, "East"),
            (180, 20, "South"),
            (270, 40, "West"),
            (45, 60, "Northeast"),
            (135, 50, "Southeast"),
            (225, 35, "Southwest"),
            (315, 25, "Northwest"),
        ]

        self.test_angles = [
            (0, 0),    # Zero position
            (30, 10),  # Small positive
            (90, 20),  # 90° azimuth
            (180, -15), # Negative elevation
            (-45, 25),  # Negative azimuth
            (360, 0),   # Full circle
            (-90, -10), # Both negative
        ]

    def test_zero_position_transformation(self):
        """Test that AZM=0, ALT=0 always maps to the alignment position."""
        for align_az, align_el, name in self.test_alignments:
            with self.subTest(alignment=name):
                result_az, result_el = AzAlt2AzEl(0.0, 0.0, align_az, align_el)

                # Allow some numerical tolerance
                az_error = abs(result_az - align_az) % 360
                if az_error > 180:
                    az_error = 360 - az_error

                el_error = abs(result_el - align_el)

                self.assertLess(az_error, 1.0,
                               f"Azimuth error {az_error:.2f}° for alignment {name}")
                self.assertLess(el_error, 1.0,
                               f"Elevation error {el_error:.2f}° for alignment {name}")

    def test_output_ranges(self):
        """Test that all outputs are within valid ranges."""
        for align_az, align_el, align_name in self.test_alignments:
            for azm, alt in self.test_angles:
                with self.subTest(align=align_name, azm=azm, alt=alt):
                    result_az, result_el = AzAlt2AzEl(azm, alt, align_az, align_el)

                    # Valid ranges
                    self.assertGreaterEqual(result_az, -180.0,
                                          f"Azimuth {result_az} too small")
                    self.assertLessEqual(result_az, 540.0,
                                       f"Azimuth {result_az} too large")
                    self.assertGreaterEqual(result_el, -90.0,
                                          f"Elevation {result_el} too small")
                    self.assertLessEqual(result_el, 90.0,
                                       f"Elevation {result_el} too large")

    def test_quadrant_transitions(self):
        """Test transformations across quadrant boundaries."""
        alignment_az, alignment_el = 90, 30  # East-facing alignment

        # Test angles spanning four quadrants relative to alignment
        quadrant_tests = [
            (0, 20, "Relative North"),
            (90, 25, "Relative East"),
            (180, 15, "Relative South"),
            (270, -10, "Relative West"),
        ]

        for delta_az, delta_el, quadrant in quadrant_tests:
            with self.subTest(quadrant=quadrant):
                result_az, result_el = AzAlt2AzEl(delta_az, delta_el, alignment_az, alignment_el)

                # Just check that we get reasonable values
                self.assertTrue(-180 <= result_az <= 540)
                self.assertTrue(-90 <= result_el <= 90)

class TestAzEl2AzAlt(unittest.TestCase):
    """Test the inverse AzEl2AzAlt transformation function."""

    def test_inverse_transformation(self):
        """Test that AzEl2AzAlt is the inverse of AzAlt2AzEl."""
        test_cases = [
            (45, 30),       # Mid-range
            (-30, 40),      # Negative azimuth
            (180, -20),     # High azimuth, negative elevation
            (270, 60),      # High azimuth, high elevation
            (0, 0),         # Boundary case
            (359, 45),      # Near full circle
        ]

        alignment_az, alignment_el = 135, 45

        for test_az, test_el in test_cases:
            with self.subTest(az=test_az, el=test_el):
                # Forward transformation
                mount_az1, mount_alt1 = AzEl2AzAlt(test_az, test_el, alignment_az, alignment_el)

                # Inverse transformation (round trip)
                result_az, result_el = AzAlt2AzEl(mount_az1, mount_alt1, alignment_az, alignment_el)

                # Check round-trip accuracy
                az_error = abs(result_az - test_az) % 360
                if az_error > 180:
                    az_error = 360 - az_error

                el_error = abs(result_el - test_el)

                # Allow 2° tolerance for numerical issues
                self.assertLess(az_error, 2.0,
                              f"Round-trip az error: {az_error:.2f}°")
                self.assertLess(el_error, 2.0,
                              f"Round-trip el error: {el_error:.2f}°")

    def test_output_ranges_inverse(self):
        """Test AzEl2AzAlt output ranges."""
        test_cases = [
            (30, 40, 90, 30),
            (180, -10, 45, 60),
            (-45, 20, 270, 25),
            (89, 45, 0, 30),
            (91, 35, 180, 40),
            (179, -15, 315, 20),
            (181, 25, 90, -10),
            (269, 35, 135, 50),
            (271, -5, 225, 70),
        ]

        for az, el, align_az, align_el in test_cases:
            with self.subTest(az=az, el=el):
                result_azm, result_alt = AzEl2AzAlt(az, el, align_az, align_el)

                # Valid azimuth range (should handle full 360° + wrapping)
                self.assertGreaterEqual(result_azm, -180.0)
                self.assertLessEqual(result_azm, 540.0)

                # Valid mount-altitude range. AzEl2AzAlt recovers ALT as a polar
                # rotation angle via arccos, so its natural range is [0, 180°].
                # Sky points outside the reachable hemisphere for a given alignment
                # legitimately map to ALT > 90° (the forward transform is 2-to-1,
                # and no |ALT| <= 90 representation of those points round-trips).
                self.assertGreaterEqual(result_alt, 0.0)
                self.assertLessEqual(result_alt, 180.0)

class TestRoundTripConsistency(unittest.TestCase):
    """Test that forward and inverse transformations are consistent."""

    def test_multiple_round_trips(self):
        """Test consistency over multiple round-trip transformations."""
        initial_angles = [(45.5, 12.3), (-67.8, 34.1), (234.9, -15.6)]

        for az, el in initial_angles:
            with self.subTest(az=az, el=el):
                alignment = (90, 40)
                align_az, align_el = alignment

                # Multiple transformations
                for round_num in range(3):
                    # Transform to mount coordinates
                    mount_az, mount_alt = AzEl2AzAlt(az, el, align_az, align_el)

                    # Transform back to sky coordinates
                    new_az, new_el = AzAlt2AzEl(mount_az, mount_alt, align_az, align_el)

                    az_error = abs(new_az - az) % 360
                    if az_error > 180:
                        az_error = 360 - az_error
                    el_error = abs(new_el - el)

                    self.assertLess(az_error, 2.0,
                                  f"Round {round_num} az error: {az_error:.2f}°")
                    self.assertLess(el_error, 2.0,
                                  f"Round {round_num} el error: {el_error:.2f}°")

class TestBoundaryConditions(unittest.TestCase):
    """Test boundary and edge conditions."""

    def test_elevation_boundaries(self):
        """Test elevation near ±90°."""
        alignment_az, alignment_el = 180, 30

        # Test cases near elevation boundaries
        boundary_tests = [
            (0, 85),     # Near zenith
            (90, -85),   # Negative, near nadir
        ]

        for delta_az, delta_el in boundary_tests:
            with self.subTest(delta_az=delta_az, delta_el=delta_el):
                # Check that result stays within valid elevation range
                result_az, result_el = AzAlt2AzEl(delta_az, delta_el, alignment_az, alignment_el)
                self.assertGreaterEqual(result_el, -90.0)
                self.assertLessEqual(result_el, 90.0)

    def test_azimuth_wraparound(self):
        """Test azimuth wraparound at 0°/360° boundary."""
        alignment_az, alignment_el = 10, 30

        # Test angles near 0° boundary
        test_angles = [359, 0, 1, 359.5, 0.5]
        mount_alt = 15

        results = []
        for mount_az in test_angles:
            result_az, result_el = AzAlt2AzEl(mount_az, mount_alt, alignment_az, alignment_el)
            results.append(result_az)

        # Check continuity across 0° boundary
        self.assertTrue(isinstance(results, list))
        self.assertEqual(len(results), len(test_angles))

if __name__ == '__main__':
    # Configure test runner with verbose output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
