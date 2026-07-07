"""
Capture Manager for Hat Creek Skytracker
Handles image and metadata capture with multithreaded dumping
"""

import threading
import time
import pygame
import os
import json
import csv
from datetime import datetime, timezone
from collections import deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from camera_buffer import CircularBuffer
from trajectory import interpolate_position

class CaptureDumpThread(threading.Thread):
    """Thread for processing and dumping captured frame sequences"""

    def __init__(self, capture_info, buffer, trajectory_state, selected_satellite, config_state, tracking_surface=None, update_status_callback=None, dump_dir="./data"):
        super().__init__()
        self.capture_info = capture_info
        self.buffer = buffer  # list snapshot of frame dicts (taken at stop_capture)
        self.trajectory_state = trajectory_state
        self.selected_satellite = selected_satellite
        self.config_state = config_state
        self.tracking_surface = tracking_surface
        self.dump_dir = dump_dir
        self.update_status_callback = update_status_callback
        self.running = True
        self.progress = 0.0
        self.status = "Starting dump..."

    def run(self):
        """Process capture and save to disk with parallel image saving"""
        try:
            # Create output directory
            os.makedirs(self.dump_dir, exist_ok=True)

            # Generate timestamp for capture folder
            start_time = self.capture_info['capture_start_time']
            timestamp_str = start_time.strftime("%Y_%m_%d_%H_%M_%S")
            timestamp_filename = f"{timestamp_str}"

            # Determine satellite name or use 'manual'
            if self.selected_satellite:
                satellite_name = self.selected_satellite.name.replace(' ', '_')
                satellite_id = self.selected_satellite.model.satnum_str if hasattr(self.selected_satellite, 'model') else str(self.selected_satellite)
                folder_name = f"{satellite_name}_{satellite_id}_{timestamp_filename}"
            else:
                folder_name = f"manual_{timestamp_filename}"

            capture_dir = os.path.join(self.dump_dir, folder_name)
            os.makedirs(capture_dir, exist_ok=True)

            self.status = f"Created directory: {folder_name}"
            self.progress = 0.02  # Small progress for setup
            # Yield briefly to allow UI updates
            time.sleep(0.01)

            # Extract frame range from capture info
            start_idx = self.capture_info['start_idx']
            end_idx = self.capture_info['end_idx']
            buffer_length = len(self.buffer)

            # Use the captured frame count instead of index calculation
            if 'captured_frame_count' in self.capture_info:
                total_frames = self.capture_info['captured_frame_count']
            else:
                # Fallback to index calculation
                total_frames = min(1000, buffer_length)  # Reasonable limit

            self.status = f"Preparing to save {total_frames} frames..."
            self.progress = 0.05

            # Extract the captured frames - work backwards from end_idx for the captured_frame_count
            frame_indices = []
            if total_frames > 0:
                for i in range(total_frames):
                    buffer_idx = end_idx - i
                    if buffer_idx < 0:
                        buffer_idx += buffer_length  # Wrap around
                    frame_indices.append(buffer_idx)
                # Reverse to get chronological order
                frame_indices.reverse()

            self.status = f"Processing {total_frames} frames..."
            self.progress = 0.1

            # Prepare frame data for parallel saving
            save_tasks = []
            for frame_idx in frame_indices:
                frame_data = self.buffer[frame_idx]
                timestamp_microseconds = frame_data['datetime_utc_microseconds']

                # Create filename with microsecond precision (Windows-compatible)
                camera_name = f"Camera{frame_data['camera_index'] + 1}"
                sequence_num = frame_data['sequence_in_capture']
                # Replace colons and other unsafe characters with underscores for Windows
                safe_timestamp = timestamp_microseconds.replace('T', '_').replace('Z', '').replace(':', '_').replace('-', '_')

                # Use image format from config (BMP for capture, PNG for tracking vis)
                ext = getattr(self.config_state, 'image_format', 'PNG').lower()
                filename = f"{camera_name}_{sequence_num:06d}__{safe_timestamp}.{ext}"

                filepath = os.path.join(capture_dir, filename)
                save_tasks.append((frame_data['frame'], filepath))

            # Use ThreadPoolExecutor for parallel image saving (4 threads instead of 8 for better UI responsiveness)
            processed_frames = 0
            batch_size = max(1, len(save_tasks) // 30)  # Update progress every ~3% or at least once for maximum UI responsiveness
            batch_count = 0

            with ThreadPoolExecutor(max_workers=8) as executor:
                # Submit all tasks at once
                futures = [executor.submit(self._save_image_task, surface, filepath) for surface, filepath in save_tasks]

                # Process completed tasks with progress updates
                for future in as_completed(futures):
                    try:
                        future.result()  # This will raise any exceptions that occurred
                        processed_frames += 1

                        # Update progress more frequently for UI responsiveness
                        if processed_frames % batch_size == 0 or processed_frames == len(save_tasks):
                            progress_ratio = processed_frames / total_frames
                            # Image saving = 0.1 to 0.8 (70% of total progress)
                            image_progress = 0.1 + (progress_ratio * 0.7)
                            self.progress = image_progress
                            self.status = f"Saved {processed_frames}/{total_frames} images ({progress_ratio:.1%})..."

                            # Use callback instead of print for UI updates
                            if self.update_status_callback:
                                self.update_status_callback(self.status)

                            # Longer yield for UI responsiveness during parallel saving
                            time.sleep(0.01)  # Increased from 0.001 to 0.01

                    except Exception as e:
                        error_msg = f"Failed to save image: {str(e)}"
                        print(error_msg)
                        if self.update_status_callback:
                            self.update_status_callback(error_msg)

            self.progress = 0.8
            self.status = "Completing file operations..."

            # Extract frame data for CSV generation (done in main thread)
            image_files = []
            frame_timestamps = []
            for frame_idx in frame_indices:
                frame_data = self.buffer[frame_idx]
                image_files.append(frame_data)
                frame_timestamps.append(frame_data['datetime_utc_microseconds'])

            self.status = "Generating CSV trajectory data..."
            self.progress = 0.85

            # Generate trajectory CSV with interpolated satellite positions
            self._generate_trajectory_csv(capture_dir, frame_timestamps, folder_name)

            self.status = "Saving tracking visualization..."
            self.progress = 0.90

            # Save final tracking visualization image
            self._save_tracking_visualization(capture_dir)

            self.status = "Copying configuration..."
            self.progress = 0.95

            # Copy config.json
            self._copy_config(capture_dir)

            self.progress = 1.0
            self.status = f"Dump complete! {total_frames} frames saved to {folder_name}"

        except Exception as e:
            self.status = f"Dump failed: {str(e)}"
            print(f"Capture dump error: {e}")
            import traceback
            traceback.print_exc()

    def _save_image_task(self, surface, filepath):
        """Task function for saving individual images in parallel threads"""
        try:
            pygame.image.save(surface, filepath)
        except Exception as e:
            raise Exception(f"Failed to save {filepath}: {e}")

    def _generate_trajectory_csv(self, capture_dir, frame_timestamps, folder_name):
        """Generate CSV with full trajectory data + samples at camera frame capture times"""
        csv_path = os.path.join(capture_dir, "trajectory.csv")

        # Import here to avoid circular imports
        from skyfield.api import load
        from trajectory import interpolate_position

        # Get timescale
        ts = load.timescale()

        # Determine satellite name safely
        if self.selected_satellite:
            satellite_name = getattr(self.selected_satellite, 'name', 'Unknown Satellite')
            try:
                satellite_id = self.selected_satellite.model.satnum_str
                satellite_name = f"{satellite_name} ({satellite_id})"
            except (AttributeError, TypeError):
                pass
        else:
            satellite_name = "Manual Capture"

        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'satellite_name', 'altitude_deg', 'azimuth_deg',
                         'distance_km', 'pixel_x', 'pixel_y', 'sequence_in_capture', 'camera_index']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Generate trajectory data at camera frame capture times
            if (self.selected_satellite and
                hasattr(self.trajectory_state, 'satellite_trajectories') and
                self.selected_satellite in self.trajectory_state.satellite_trajectories):

                trajectory_data, times_array = self.trajectory_state.satellite_trajectories[self.selected_satellite]

                # Step 1: First write the full pre-computed trajectory data
                print("Writing full trajectory data...")

                # Create observer for azimuth calculation
                observer = None
                if hasattr(self.config_state, 'lat_str') and hasattr(self.config_state, 'lon_str'):
                    from skyfield.api import wgs84
                    lat = float(self.config_state.lat_str or 0)
                    lon = float(self.config_state.lon_str or 0)
                    alt_m = float(getattr(self.config_state, 'alt_str', 0))
                    observer = wgs84.latlon(lat, lon, elevation_m=alt_m)

                for i, (time_val, alt, az, dist, px, py) in enumerate(trajectory_data):
                    try:
                        # Convert TT (Terrestrial Time) to UTC timestamp correctly
                        skyfield_time = ts.tt(jd=time_val)
                        utc_dt = skyfield_time.utc_datetime()
                        timestamp_str = utc_dt.isoformat() + "Z"

                        writer.writerow({
                            'timestamp': timestamp_str,
                            'satellite_name': satellite_name,
                            'altitude_deg': alt,
                            'azimuth_deg': az,  # Azimuth from trajectory data
                            'distance_km': dist,
                            'pixel_x': px,
                            'pixel_y': py,
                            'sequence_in_capture': 0,  # Mark as trajectory data (not capture data)
                            'camera_index': -1  # Mark as full trajectory data
                        })
                    except Exception as e:
                        print(f"Warning: Could not convert trajectory time {time_val}: {e}")

                # Step 2: Now add samples interpolated to camera frame capture times
                print(f"Adding interpolated samples at frame capture times...")

                # Get all captured frames (any frame with sequence_in_capture > 0)
                captured_frame_timestamps = []
                captured_frame_data = []
                frame_counts = {'total': 0, 'captured': 0, 'by_camera': {}}

                print(f"Scanning {len(self.buffer)} frames in buffer...")
                for idx in range(len(self.buffer)):
                    frame_data = self.buffer[idx]
                    frame_counts['total'] += 1
                    sequence_num = frame_data.get('sequence_in_capture', 0)
                    camera_idx = frame_data.get('camera_index', 'unknown')

                    # Track frame counts by camera
                    if camera_idx not in frame_counts['by_camera']:
                        frame_counts['by_camera'][camera_idx] = {'total': 0, 'captured': 0}

                    frame_counts['by_camera'][camera_idx]['total'] += 1

                    if sequence_num > 0:  # Frame was captured during active capture
                        frame_counts['captured'] += 1
                        frame_counts['by_camera'][camera_idx]['captured'] += 1
                        captured_frame_timestamps.append(frame_data['datetime_utc_microseconds'])
                        captured_frame_data.append(frame_data)

                print(f"Found {len(captured_frame_data)} captured frames")

                # Process each captured frame and interpolate satellite position
                for frame_data in captured_frame_data:
                    camera_utc_microseconds = frame_data['datetime_utc_microseconds']
                    sequence_num = frame_data['sequence_in_capture']
                    camera_index = frame_data['camera_index']

                    try:
                        # Convert camera timestamp to Skyfield time
                        utc_dt = datetime.fromisoformat(camera_utc_microseconds.replace('Z', '+00:00'))
                        skyfield_time = ts.utc(utc_dt.year, utc_dt.month, utc_dt.day,
                                               utc_dt.hour, utc_dt.minute,
                                               utc_dt.second + utc_dt.microsecond / 1e6)

                        # Interpolate satellite position at this exact time
                        px, py, alt, dist = interpolate_position(
                            self.trajectory_state.satellite_trajectories[self.selected_satellite],
                            skyfield_time.tt
                        )

                        if px is not None:
                            # Try to compute azimuth at this specific time
                            az_deg = 0.0  # Default value
                            if observer is not None:
                                try:
                                    difference = self.selected_satellite - observer
                                    topocentric = difference.at(skyfield_time)
                                    alt_az = topocentric.altaz()
                                    az_deg = alt_az[1].degrees
                                except Exception as e:
                                    print(f"Warning: Could not calculate azimuth for {camera_utc_microseconds}: {e}")

                            # Write interpolated point with camera-specific data.
                            # NOTE: this must stay at the `if px` level, NOT inside
                            # `if observer` -- without an observer (no site lat/lon
                            # configured) the row is still written, with the az_deg
                            # fallback above. An indentation slip here once silently
                            # dropped every per-frame row of the labeled data product.
                            writer.writerow({
                                'timestamp': camera_utc_microseconds,
                                'satellite_name': satellite_name,
                                'altitude_deg': alt,
                                'azimuth_deg': az_deg,
                                'distance_km': dist,
                                'pixel_x': px,
                                'pixel_y': py,
                                'sequence_in_capture': sequence_num,
                                'camera_index': camera_index
                            })

                            # Use callback for significant progress updates
                            if self.update_status_callback and len(captured_frame_data) > 100:
                                if captured_frame_data.index(frame_data) % 50 == 0:  # Update every 50 frames
                                    cam_counts = frame_counts['by_camera']
                                    cam_status = []
                                    for cam_idx in sorted(cam_counts.keys()):
                                        if cam_idx != 'unknown':
                                            captured_count = cam_counts[cam_idx]['captured']
                                            cam_status.append(f"C{cam_idx+1}: {captured_count}")
                                    status_str = f"Processing CSV {len(captured_frame_data)-1}-{frame_data['sequence_in_capture']}/{total_frames} - {', '.join(cam_status)}"
                                    self.update_status_callback(status_str)

                    except (ValueError, KeyError, AttributeError) as e:
                        print(f"Warning: Could not process frame timestamp {camera_utc_microseconds}: {e}")

                print(f"Generated trajectory CSV with {len(trajectory_data)} trajectory points + {len(captured_frame_data)} frame samples")
            else:
                print(f"Note: No trajectory data available for satellite {satellite_name}")

    def _save_tracking_visualization(self, capture_dir):
        """Save final tracking visualization image"""
        try:
            # Use the passed tracking_surface instead of importing main
            if self.tracking_surface:
                # Create filename
                satellite_name = self.selected_satellite.name if self.selected_satellite else 'manual'
                timestamp_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                filename = f"{satellite_name}_trackingvis_{timestamp_str}.png"
                filepath = os.path.join(capture_dir, filename)

                # Save the tracking visualization surface
                pygame.image.save(self.tracking_surface, filepath)
                print(f"Saved tracking visualization to: {filepath}")
            else:
                print("Warning: Could not capture tracking visualization - no surface available")
        except Exception as e:
            print(f"Failed to save tracking visualization: {e}")
            import traceback
            traceback.print_exc()

    def _copy_config(self, capture_dir):
        """Copy config.json to capture directory"""
        try:
            config_source = "config.json"
            timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
            config_dest = os.path.join(capture_dir, f"config_{timestamp_str}.json")

            if hasattr(self.config_state, 'save_to_file'):
                # Save current config to the capture directory
                original_config_state = self.config_state.__dict__.copy()
                self.config_state.config_file = config_dest
                self.config_state.save_to_file()
                # Restore original state
                self.config_state.__dict__.update(original_config_state)
            else:
                # Fallback: copy existing file
                import shutil
                if os.path.exists(config_source):
                    shutil.copy2(config_source, config_dest)

        except Exception as e:
            print(f"Failed to copy config: {e}")

class CaptureManager:
    """Manages capture state and coordinates with camera threads"""

    def __init__(self):
        self.dump_threads = []
        self.current_dumps = []
        self.max_concurrent_dumps = 1  # Only allow one dump at a time

        # Progress tracking for UI
        self.current_dump_progress = 0.0
        self.current_dump_status = ""
        self.is_dumping = False

    def start_capture(self, camera_index=None):
        """Start capture on specified camera or all connected cameras"""
        from camera_manager import camera_manager

        if camera_index is not None:
            # Start capture on specified camera
            camera = camera_manager.get_camera(camera_index)
            if camera and camera.thread:
                camera.thread.start_capture()
        else:
            # Start capture on all connected cameras
            for idx in range(len(camera_manager.cameras)):
                camera = camera_manager.get_camera(idx)
                if camera and camera.connected and camera.thread:
                    camera.thread.start_capture()

    def stop_capture(self, camera_index=None, trajectory_state=None, selected_satellite=None, config_state=None, tracking_surface=None, update_status_callback=None):
        """Stop capture and begin dump process for specified camera or all cameras"""
        from camera_manager import camera_manager
        started_dumps = []

        if camera_index is not None:
            # Stop capture on specified camera
            camera = camera_manager.get_camera(camera_index)
            if camera and camera.thread:
                capture_info, buffer = camera.thread.stop_capture()

                if capture_info and buffer:
                    # Start dump thread
                    dump_thread = CaptureDumpThread(
                        capture_info=capture_info,
                        buffer=buffer,
                        trajectory_state=trajectory_state,
                        selected_satellite=selected_satellite,
                        config_state=config_state,
                        tracking_surface=tracking_surface,
                        update_status_callback=update_status_callback
                    )

                    self.dump_threads.append(dump_thread)
                    dump_thread.start()
                    started_dumps.append(camera_index)
                    self.is_dumping = True
        else:
            # Stop capture on all connected cameras
            for idx in range(len(camera_manager.cameras)):
                camera = camera_manager.get_camera(idx)
                if camera and camera.connected and camera.thread:
                    capture_info, buffer = camera.thread.stop_capture()

                    if capture_info and buffer:
                        # Start dump thread for this camera
                        dump_thread = CaptureDumpThread(
                            capture_info=capture_info,
                            buffer=buffer,
                            trajectory_state=trajectory_state,
                            selected_satellite=selected_satellite,
                            config_state=config_state,
                            tracking_surface=tracking_surface,
                            update_status_callback=update_status_callback
                        )

                        self.dump_threads.append(dump_thread)
                        dump_thread.start()
                        started_dumps.append(idx)

            if started_dumps:
                self.is_dumping = True

        return len(started_dumps) > 0

    def get_dump_progress(self):
        """Get current dump progress for UI (handles multiple concurrent dumps)"""
        if self.dump_threads:
            # Check if any dumps are still running
            running_dumps = [t for t in self.dump_threads if t.is_alive()]
            if running_dumps:
                # Handle multiple dumps: show combined progress
                total_progress = sum(dump.progress for dump in running_dumps)
                combined_progress = total_progress / len(running_dumps)

                # Get status from first alive dump (for simplicity and UI responsiveness)
                status = "Unknown status"
                for thread in self.dump_threads:
                    if thread.is_alive():
                        status = thread.status
                        break

                return combined_progress, f"{len(running_dumps)} cameras dumping - {status}"
            else:
                # All dumps completed - this check should be fast
                if self.is_dumping:
                    self.current_dump_status = "All camera dumps completed!"
                    self.current_dump_progress = 1.0
                    self.is_dumping = False
                return 1.0, "No active dumps"

        return 0.0, "No dumps in progress"

    def cleanup_completed_dumps(self):
        """Clean up completed dump threads"""
        self.dump_threads = [t for t in self.dump_threads if t.is_alive()]

# Create global capture manager instance
capture_manager = CaptureManager()
