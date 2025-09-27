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

    # Define the standard zenith direction (top of sky)
    zenith = np.array([0, 0, 1])  # [x, y, z] = [north, east, up]

    # Define the alignment direction
    align_vec = np.array([
        np.cos(align_el_rad) * np.cos(align_az_rad),  # north component
        np.cos(align_el_rad) * np.sin(align_az_rad),  # east component
        np.sin(align_el_rad)                         # up component
    ])

    # Create rotation that maps zenith to alignment direction
    alignment_rotation = R.align_vectors([align_vec], [zenith])[0]

    # Apply rotations in correct order:
    # 1. First apply AZM rotation around vertical axis
    azm_rotation = R.from_rotvec(azm_rad * np.array([0, 0, 1]))

    # 2. Then apply ALT rotation around the east-west axis
    # The rotation axis for ALT is perpendicular to the vertical axis
    # This axis points in the direction of increasing azimuth at the current AZM position
    alt_axis = np.array([np.sin(azm_rad), -np.cos(azm_rad), 0])
    alt_rotation = R.from_rotvec(alt_rad * alt_axis)

    # Combine rotations: final = align @ azm @ alt
    final_rotation = alignment_rotation * azm_rotation * alt_rotation

    # Apply the combined rotation to find where the telescope is pointing
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

    Analytical inverse of the corrected AzAlt2AzEl transformation.

    Args:
        az: True azimuth angle (degrees)
        el: True elevation angle (degrees)
        alignment_azimuth: Alignment azimuth offset (degrees)
        alignment_elevation: Alignment elevation offset (degrees)

    Returns:
        tuple: (AZM, ALT) in degrees
    """
    from scipy.spatial.transform import Rotation as R
    from scipy.optimize import minimize

    # Convert target angles to radians
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    align_az_rad = np.radians(alignment_azimuth)
    align_el_rad = np.radians(alignment_elevation)

    # Create target pointing vector in sky coordinates
    target_vec = cartesian_from_az_el(az, el)

    # Define the alignment direction (mount zero position)
    align_vec = np.array([
        np.cos(align_el_rad) * np.cos(align_az_rad),  # north component
        np.cos(align_el_rad) * np.sin(align_az_rad),  # east component
        np.sin(align_el_rad)                         # up component
    ])

    # Define the zenith direction
    zenith = np.array([0, 0, 1])

    # Create alignment rotation that maps zenith to alignment direction
    alignment_rotation = R.align_vectors([align_vec], [zenith])[0]

    # We need to find AZM and ALT such that when we apply the sequence:
    # align @ azm_rotation @ alt_rotation to zenith, we get target_vec

    # Solve by applying inverse rotations in reverse order
    # Start from target_vec and apply inverse rotations

    # First, apply inverse of alignment rotation
    temp_vec = alignment_rotation.inv().apply(target_vec)

    # Now temp_vec represents the direction relative to alignment
    # We need to find AZM and ALT that would rotate zenith to temp_vec

    # The temp_vec should be the result of: azm_rotation @ alt_rotation @ zenith_rel
    # where zenith_rel is [0, 0, 1] (zenith relative to alignment)

    # For finding the inverse, we'll use numerical optimization as before
    # but with better error function and constraints

    def error_function(params):
        azm_test, alt_test = params

        # Test the forward transformation
        az_test, el_test = AzAlt2AzEl(azm_test, alt_test, alignment_azimuth, alignment_elevation)
        test_vec = cartesian_from_az_el(az_test, el_test)

        # Return error distance
        return np.linalg.norm(target_vec - test_vec)

    # Generate better initial guesses
    initial_guesses = []

    # Best guess: simple offset from alignment
    initial_azm = (az - alignment_azimuth) % 360
    initial_alt = el - alignment_elevation
    initial_alt = np.clip(initial_alt, -90, 90)
    initial_guesses.append([initial_azm, initial_alt])

    # Alternative: zero position for alignment target
    if abs(az - alignment_azimuth) < 1 and abs(el - alignment_elevation) < 1:
        initial_guesses.append([0.0, 0.0])

    # Quadrant-based guesses
    for quad in [0, 90, 180, 270]:
        quad_azm = (az - alignment_azimuth - quad) % 360
        if quad_azm > 180:
            quad_azm -= 360
        quad_alt = np.clip(el - alignment_elevation, -90, 90)
        initial_guesses.append([quad_azm, quad_alt])

    # Perform optimization from each initial guess
    results = []
    for guess in initial_guesses[:5]:  # Use only first 5 guesses
        try:
            result = minimize(error_function, guess, bounds=[(-180, 540), (-90, 90)],
                             method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
            if result.success:
                error = error_function(result.x)
                results.append((error, result.x))
        except:
            continue

    # Find the best result
    if results:
        best_result = min(results, key=lambda x: x[0])
        best_error, best_params = best_result
        # If error is acceptable, return the result
        if best_error < 1e-4:
            return best_params[0] % 360, best_params[1]

    # If optimization fails, try fallback with better tolerances
    print("Using fallback inverse transformation")
    fallback_azm = (az - alignment_azimuth) % 360
    fallback_alt = np.clip(el - alignment_elevation, -90, 90)
    return fallback_azm, fallback_alt

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

def AzAlt2AzEl_AltAz(AZM, ALT, alignment_azimuth):
    """
    Transform from mount coordinates (AZM, ALT) to true az/el (azimuth, elevation) for AltAz mount.

    Simplified transformation assuming gravity-aligned AltAz mount where 0 ALT = 90° elevation.
    In this mode, alignment_elevation is assumed to be zero and ignored.

    Args:
        AZM: Mount azimuth angle (degrees)
        ALT: Mount altitude angle (degrees)
        alignment_azimuth: Alignment azimuth offset (degrees)

    Returns:
        tuple: (az, el) in degrees
    """
    # Simplified transformations for AltAz mount
    az = (AZM + alignment_azimuth) % 360  # Azimuth is simply AZM plus offset
    el = 90.0 - ALT  # Elevation is 90 degrees minus ALT

    return az, el

def AzEl2AzAlt_AltAz(az, el, alignment_azimuth):
    """
    Transform from true az/el (azimuth, elevation) to mount coordinates (AZM, ALT) for AltAz mount.

    Simplified inverse transformation for AltAz mount.

    Args:
        az: True azimuth angle (degrees)
        el: True elevation angle (degrees)
        alignment_azimuth: Alignment azimuth offset (degrees)

    Returns:
        tuple: (AZM, ALT) in degrees
    """
    # Simplified inverse transformations for AltAz mount
    AZM = (az - alignment_azimuth) % 360  # AZM is azimuth minus offset
    ALT = 90.0 - el  # ALT is 90 degrees minus elevation

    return AZM, ALT

def AzAlt2AzEl_Passthrough(AZM, ALT):
    """
    Transform from mount coordinates (AZM, ALT) to true az/el (azimuth, elevation) in Passthrough mode.

    In passthrough mode, telescope coordinates are treated as sky coordinates.
    No transformation is applied - mount coordinates are directly used as sky coordinates.

    Args:
        AZM: Mount azimuth angle (degrees)
        ALT: Mount altitude angle (degrees)

    Returns:
        tuple: (az, el) in degrees - same as input
    """
    # Passthrough: mount coordinates are sky coordinates
    return AZM, ALT

def AzEl2AzAlt_Passthrough(az, el):
    """
    Transform from true az/el (azimuth, elevation) to mount coordinates (AZM, ALT) in Passthrough mode.

    In passthrough mode, sky coordinates are treated as mount coordinates.
    No transformation is applied - sky coordinates are directly used as mount coordinates.

    Args:
        az: True azimuth angle (degrees)
        el: True elevation angle (degrees)

    Returns:
        tuple: (AZM, ALT) in degrees - same as input
    """
    # Passthrough: sky coordinates are mount coordinates
    return az, el

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
