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

    Args:
        AZM: Mount azimuth angle (degrees)
        ALT: Mount altitude angle (degrees)
        alignment_azimuth: Alignment azimuth offset (degrees)
        alignment_elevation: Alignment elevation offset (degrees)

    Returns:
        tuple: (az, el) in degrees
    """
    # Compute direction in standard local coordinates (before alignment)
    d = cartesian_from_az_el(AZM, ALT)

    # Apply inverse alignment rotation (negative angle)
    R_align = alignment_rotation_matrix(alignment_azimuth, alignment_elevation)
    # For the transformation from mount to sky, we apply the inverse alignment
    R_inv = R_align.T  # Transpose for rotation matrix inverse

    # Also apply the alignment rotation for the coordinate system
    d_aligned = R_inv @ d

    # Convert back to az/el
    az, el = az_el_from_cartesian(d_aligned)
    return az, el

def AzEl2AzAlt(az, el, alignment_azimuth, alignment_elevation, max_iter=10, tolerance=1e-6):
    """
    Transform from true az/el (azimuth, elevation) to mount coordinates (AZM, ALT).

    Uses numerical approximation to solve the inverse transformation.

    Args:
        az: True azimuth angle (degrees)
        el: True elevation angle (degrees)
        alignment_azimuth: Alignment azimuth offset (degrees)
        alignment_elevation: Alignment elevation offset (degrees)
        max_iter: Maximum iterations for numerical solution
        tolerance: Tolerance for convergence

    Returns:
        tuple: (AZM, ALT) in degrees
    """
    # Initial guess: simple linear correction
    AZM_guess = (az - alignment_azimuth) % 360
    ALT_guess = el - alignment_elevation

    for i in range(max_iter):
        # Compute current transformation
        az_guess, el_guess = AzAlt2AzEl(AZM_guess, ALT_guess, alignment_azimuth, alignment_elevation)

        # Compute error
        daz = az - az_guess
        dele = el - el_guess

        # Normalize az difference to [-180, 180]
        daz = (daz + 180) % 360 - 180

        # Check convergence
        if abs(daz) < tolerance and abs(dele) < tolerance:
            break

        # Jacobian approximation (estimated partial derivatives)
        jac_az_azm = 1.0 + 0.01 * abs(ALT_guess - 90)  # Approximates coupling effect
        jac_el_alt = 1.0

        # Update guess using Newton's method
        AZM_guess += daz / jac_az_azm
        ALT_guess += dele / jac_el_alt

        # Keep within bounds
        AZM_guess %= 360
        ALT_guess = np.clip(ALT_guess, -90, 90)

        # Handle pole crossing
        if ALT_guess > 80 or ALT_guess < -80:
            # Near singularity, flip azimuth
            AZM_guess = (AZM_guess + 180) % 360
            ALT_guess = 90 - np.abs(ALT_guess) if ALT_guess > 0 else -90 + np.abs(ALT_guess)

    return AZM_guess, ALT_guess

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
