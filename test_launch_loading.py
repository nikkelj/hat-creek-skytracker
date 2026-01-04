#!/usr/bin/env python3
"""
Simple test script to verify launch trajectory loading functionality.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from trajectory import read_launch_trajectories

def test_launch_loading():
    """Test the launch trajectory loading functionality."""
    print("Testing launch trajectory loading...")

    # Test with existing launches directory
    launches_dir = "./launches"

    try:
        # Read launch trajectories
        launch_trajectories = read_launch_trajectories(launches_dir)

        print(f"Found {len(launch_trajectories)} launch trajectories:")

        for launch_name in sorted(launch_trajectories.keys()):
            if not launch_name.endswith('_arcs'):
                trajectory_data, times_array = launch_trajectories[launch_name]
                arc_data = launch_trajectories.get(launch_name + '_arcs', [])

                print(f"  {launch_name}:")
                print(f"    Trajectory points: {len(trajectory_data)}")
                print(f"    Time range: {times_array[0]:.2f} to {times_array[-1]:.2f}")
                print(f"    Arc segments: {len(arc_data)}")

                # Check first few data points
                if trajectory_data:
                    first_point = trajectory_data[0]
                    print(f"    First point: time={first_point[0]:.2f}, alt={first_point[1]:.2f}°, az={first_point[2]:.2f}°, range={first_point[3]:.2f}km")
                    print(f"    Pixel coords: ({first_point[4]:.1f}, {first_point[5]:.1f})")
                    if len(first_point) > 6:
                        print(f"    Rates: az={first_point[6]:.4f} °/s, el={first_point[7]:.4f} °/s")

        print("\nLaunch trajectory loading test completed successfully!")
        return True

    except Exception as e:
        print(f"Error testing launch trajectory loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_launch_loading()
    sys.exit(0 if success else 1)
