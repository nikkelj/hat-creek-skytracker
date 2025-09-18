import numpy as np
import math

def cartesian_from_az_el(az_deg, el_deg):
    """Convert az/el coordinates to cartesian unit vector."""
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    return np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad)
    ])

def az_el_from_cartesian(vec):
    """Convert cartesian vector to az/el coordinates."""
    vec = np.asarray(vec)
    r = np.linalg.norm(vec)
    if r == 0:
        return 0.0, 0.0
    az = np.degrees(np.arctan2(vec[1], vec[0])) % 360
    el = np.degrees(np.arcsin(vec[2] / r))
    return az, el

def rotation_matrix_around_axis(axis, angle_rad):
    """Create rotation matrix around axis by angle using Rodrigues formula."""
    axis = np.asarray(axis)
    axis /= np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.eye(3) * cos_a + (1 - cos_a) * np.outer(axis, axis) + sin_a * K
    return R

def alignment_rotation_matrix(alignment_azimuth, alignment_elevation):
    """Compute rotation matrix that aligns telescope to alignment direction."""
    # Standard north (pointing up)
    north = np.array([0.0, 0.0, 1.0])

    # Alignment direction
    alignment_vector = cartesian_from_az_el(alignment_azimuth, alignment_elevation)

    # Compute rotation axis (cross product)
    axis = np.cross(north, alignment_vector)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-10:
        # Already aligned, return identity matrix
        return np.eye(3)

    # Normalize axis
    axis /= axis_norm

    # Compute angle
    cos_theta = np.dot(north, alignment_vector)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    # Return rotation matrix
    return rotation_matrix_around_axis(axis, angle_rad)

def AzAlt2AzEl(AZM, ALT, alignment_azimuth, alignment_elevation):
    """
    Transform from mount coordinates (AZM, ALT) to true az/el (azimuth, elevation).

    Uses quaternion-based rotations for accurate, invertible transformations.

    Args:
        AZM: Mount azimuth angle (degrees)
        ALT: Mount altitude angle (degrees)
        alignment_azimuth: Alignment azimuth offset (degrees)
        alignment_elevation: Alignment elevation offset (degrees)

    Returns:
        tuple: (az, el) in degrees
    """
    from scipy.spatial.transform import Rotation as R

    # Convert angles to radians
    azm_rad = np.radians(AZM)
    alt_rad = np.radians(ALT)
    align_az_rad = np.radians(alignment_azimuth)
    align_el_rad = np.radians(alignment_elevation)

    # Build the transformation as a composition of rotations:

    # 1. First: Rotate the telescope mount to point at the alignment direction
    # This is equivalent to rotating from zenith (0,90) to alignment (align_az, align_el)
    # We need to rotate the sky coordinate system so that (align_az, align_el) becomes (0,90)

    # Define the standard zenith direction (top of sky)
    zenith = np.array([0, 0, 1])  # [x, y, z] = [north, east, up]

    # Define the alignment direction
    align_vec = np.array([
        np.cos(align_el_rad) * np.cos(align_az_rad),  # north component
        np.cos(align_el_rad) * np.sin(align_az_rad),  # east component
        np.sin(align_el_rad)                         # up component
    ])

    # Create rotation that maps zenith to alignment direction
    # This represents the misalignment of the telescope mount
    alignment_rotation = R.align_vectors([align_vec], [zenith])[0]

    # 2. AZM rotation: Rotation around the vertical axis (around zenith/up)
    # This moves the telescope azimuthally around the sky
    azm_rotation = R.from_rotvec(azm_rad * np.array([0, 0, 1]))

    # 3. ALT rotation: Rotation around the east-west horizontal axis
    # The rotation axis depends on current azimuth position
    # For ALT, we rotate around an axis perpendicular to both zenith and meridians
    # Current position of the "right" axis after AZM rotation is [-sin(azm), cos(azm), 0]
    right_axis = np.array([-np.sin(azm_rad), np.cos(azm_rad), 0])
    alt_rotation = R.from_rotvec(alt_rad * right_axis)

    # Apply transformations in sequence:
    # Start with zenith-directed telescope and apply rotations

    # The total transformation is:
    # final_position = alignment_rotation @ alt_rotation @ azm_rotation @ zenith
    # But we need to think about the coordinate frame transformation

    # For the forward transformation, we want:
    # Mount coordinates (AZM, ALT) → Sky coordinates (az, el)

    # The mount's zero position points to the alignment direction
    # We want to find where the current mount position points in sky coordinates

    # Start with the alignment direction and apply the AZM and ALT rotations to it
    final_rotation = alignment_rotation * alt_rotation * azm_rotation

    # Apply the combined rotation to the sky coordinate frame
    # The mount's current pointing direction in sky coordinates
    current_pointing = final_rotation.apply(zenith)

    # Convert back to az/el
    az, el = az_el_from_cartesian(current_pointing)

    return az, el

def _AzAlt2AzEl_fallback(AZM, ALT, alignment_azimuth, alignment_elevation):
    """
    Fallback implementation using direct mathematical transformations.
    """
    # Convert to radians
    azm_rad = np.radians(AZM)
    alt_rad = np.radians(ALT)
    align_az_rad = np.radians(alignment_azimuth)
    align_el_rad = np.radians(alignment_elevation)

    # Start from the alignment direction
    d_az = np.cos(align_el_rad) * np.cos(align_az_rad)
    d_el = np.cos(align_el_rad) * np.sin(align_az_rad)
    d_vert = np.sin(align_el_rad)

    # Apply AZM rotation around vertical axis
    cos_azm = np.cos(azm_rad)
    sin_azm = np.sin(azm_rad)

    d_az_new = d_az * cos_azm - d_el * sin_azm
    d_el_new = d_az * sin_azm + d_el * cos_azm
    d_vert_new = d_vert

    # Apply ALT rotation around the east-west axis
    cos_alt = np.cos(alt_rad)
    sin_alt = np.sin(alt_rad)

    # The axis of rotation for ALT is perpendicular to the current pointing direction
    # For ALT, we rotate around the axis [-sin(azm), cos(azm), 0] (east direction)
    right_axis = np.array([-np.sin(azm_rad), np.cos(azm_rad), 0])
    horiz_length = np.sqrt(d_az_new**2 + d_el_new**2)

    if horiz_length > 1e-10:
        # Rotate the pointing vector around the right axis
        rotation_axis = right_axis
        rotation_angle = alt_rad

        # Rodrigues formula for rotation
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        rot_matrix = np.array([
            [cos_a + rotation_axis[0]**2 * (1 - cos_a),
             rotation_axis[0]*rotation_axis[1]*(1 - cos_a) - rotation_axis[2]*sin_a,
             rotation_axis[0]*rotation_axis[2]*(1 - cos_a) + rotation_axis[1]*sin_a],
            [rotation_axis[1]*rotation_axis[0]*(1 - cos_a) + rotation_axis[2]*sin_a,
             cos_a + rotation_axis[1]**2 * (1 - cos_a),
             rotation_axis[1]*rotation_axis[2]*(1 - cos_a) - rotation_axis[0]*sin_a],
            [rotation_axis[2]*rotation_axis[0]*(1 - cos_a) - rotation_axis[1]*sin_a,
             rotation_axis[2]*rotation_axis[1]*(1 - cos_a) + rotation_axis[0]*sin_a,
             cos_a + rotation_axis[2]**2 * (1 - cos_a)]
        ])

        final_vec = rot_matrix @ np.array([d_az_new, d_el_new, d_vert_new])
        d_az_final = final_vec[0]
        d_el_final = final_vec[1]
        d_vert_final = final_vec[2]

    else:
        # Special case: pointing straight up/down
        d_az_final = d_az_new
        d_el_final = d_el_new
        d_vert_final = d_vert_new * np.cos(alt_rad)

    # Convert to az/el
    az, el = az_el_from_cartesian(np.array([d_az_final, d_el_final, d_vert_final]))

    return az, el

def AzEl2AzAlt(az, el, alignment_azimuth, alignment_elevation):
    """
    Transform from true az/el (azimuth, elevation) to mount coordinates (AZM, ALT).

    Uses numerical optimization with multiple initial guesses to find the inverse transformation accurately.

    Args:
        az: True azimuth angle (degrees)
        el: True elevation angle (degrees)
        alignment_azimuth: Alignment azimuth offset (degrees)
        alignment_elevation: Alignment elevation offset (degrees)

    Returns:
        tuple: (AZM, ALT) in degrees
    """
    from scipy.optimize import minimize
    import random

    target_vec = cartesian_from_az_el(az, el)

    # Objective function: error between target and forward transformation
    def error_function(params):
        azm, alt = params
        computed_az, computed_el = AzAlt2AzEl(azm, alt, alignment_azimuth, alignment_elevation)
        computed_vec = cartesian_from_az_el(computed_az, computed_el)
        return np.linalg.norm(target_vec - computed_vec)

    # Generate multiple initial guesses
    initial_guesses = []

    # Simple approximation
    initial_azm = (az - alignment_azimuth) % 360
    initial_alt = np.clip(el - alignment_elevation, -90, 90)
    initial_guesses.append([initial_azm, initial_alt])

    # Additional random guesses
    for i in range(20):
        rand_azm = random.uniform(0, 360)
        rand_alt = random.uniform(-90, 90)
        initial_guesses.append([rand_azm, rand_alt])

    # Perform optimization from each initial guess
    results = []
    for guess in initial_guesses:
        result = minimize(error_function, guess, bounds=[(0, 360), (-90, 90)],
                         method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-6})
        if result.success:
            results.append((result.fun, result.x))

    # Find the best result
    if results:
        best_result = min(results, key=lambda x: x[0])
        best_error, best_params = best_result
        # If error is too large, there might be issues
        if best_error > 1e-3:
            print(f"Warning: Large optimization error {best_error} for az={az}, el={el}")
        return best_params[0] % 360, best_params[1]

    # Fallback to original method if all optimizations fail
    print(f"Warning: All optimizations failed for az={az}, el={el}, using simple approximation")
    initial_azm = (az - alignment_azimuth) % 360
    initial_alt = np.clip(el - alignment_elevation, -90, 90)
    return initial_azm, initial_alt

def _AzEl2AzAlt_numerical(az, el, alignment_azimuth, alignment_elevation):
    """
    Fallback numerical solution for AzEl2AzAlt when scipy not available.
    """
    # Simple numerical solution - iteratively refine
    current_azm = (az - alignment_azimuth) % 360
    if current_azm > 180:
        current_azm -= 360

    current_alt = el - alignment_elevation
    current_alt = np.clip(current_alt, -90, 90)

    # Simple gradient descent approach
    learning_rate = 0.1
    max_iterations = 50

    for _ in range(max_iterations):
        # Compute current output
        test_az, test_el = AzAlt2AzEl(current_azm, current_alt, alignment_azimuth, alignment_elevation)

        # Compute errors
        az_error = az - test_az
        el_error = el - test_el

        # Handle azimuth wraparound
        if az_error > 180:
            az_error -= 360
        elif az_error < -180:
            az_error += 360

        # Update estimates
        if abs(az_error) > 1.0:  # Only update if error is significant
            current_azm += az_error * learning_rate
        if abs(el_error) > 1.0:
            current_alt += el_error * learning_rate * 2  # Higher gain for elevation

        # Keep in bounds
        current_azm = (current_azm + 360) % 360
        current_alt = np.clip(current_alt, -90, 90)

        # Check convergence
        if abs(az_error) < 0.1 and abs(el_error) < 0.1:
            break

    return current_azm, current_alt

def apply_rotation_to_az_el(az, el, rotation_angle):
    """
    Apply additional rotation around the pointing direction.

    Args:
        az, el: Azimuth and elevation (degrees)
        rotation_angle: Rotation angle around the pointing vector (degrees)

    Returns:
        tuple: New (az, el) after rotation
    """
    if abs(rotation_angle) < 1e-10:
        return az, el

    # Get pointing vector
    vec = cartesian_from_az_el(az, el)

    # Create rotation matrix around the pointing vector
    rot_matrix = rotation_matrix_around_axis(vec, np.radians(rotation_angle))

    # Rotate pointing vector
    rotated_vec = rot_matrix @ vec

    # Convert back to az/el
    new_az, new_el = az_el_from_cartesian(rotated_vec)
    return new_az, new_el

def compute_fov_for_camera(pixel_size_um, focal_length_mm, roi_width_pct, roi_height_pct,
                          camera_width_pixels, camera_height_pixels):
    """
    Compute FOV parameters for a camera.

    Args:
        pixel_size_um: Pixel size in micrometers
        focal_length_mm: Focal length in millimeters
        roi_width_pct: ROI width percentage (0.0 to 1.0)
        roi_height_pct: ROI height percentage (0.0 to 1.0)
        camera_width_pixels: Camera sensor width in pixels
        camera_height_pixels: Camera sensor height in pixels

    Returns:
        dict: FOV parameters including spot_size, fov_width_deg, fov_height_deg,
              roi_pixel_widths, roi_pixel_heights
    """
    # Calculate spot size (arcseconds per pixel)
    spot_size_arcsec_per_pixel = 206 * pixel_size_um / focal_length_mm

    # Calculate ROI pixel dimensions
    roi_pixel_width = camera_width_pixels * roi_width_pct
    roi_pixel_height = camera_height_pixels * roi_height_pct

    # Calculate FOV in degrees
    fov_width_deg = (spot_size_arcsec_per_pixel * roi_pixel_width) / 3600.0
    fov_height_deg = (spot_size_arcsec_per_pixel * roi_pixel_height) / 3600.0

    return {
        'spot_size_arcsec_per_pixel': spot_size_arcsec_per_pixel,
        'fov_width_deg': fov_width_deg,
        'fov_height_deg': fov_height_deg,
        'roi_pixel_width': roi_pixel_width,
        'roi_pixel_height': roi_pixel_height
    }
