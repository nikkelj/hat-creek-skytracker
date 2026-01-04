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

def AzEl2AzAlt_AltAz(az, el, alignment_azimuth, alignment_elevation):
    """
    Transform from true az/el (azimuth, elevation) to mount coordinates (AZM, ALT) for AltAz mount.

    Simplified inverse transformation for AltAz mount.

    Args:
        az: True azimuth angle (degrees)
        el: True elevation angle (degrees)
        alignment_azimuth: Alignment azimuth offset (degrees)
        alignment_elevation: Alignment elevation offset (degrees)

    Returns:
        tuple: (AZM, ALT) in degrees
    """
    # Simplified inverse transformations for AltAz mount
    AZM = (az - alignment_azimuth) % 360  # AZM is azimuth minus offset
    ALT = el - alignment_elevation  # ALT is 90 degrees minus elevation

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
    
# Forward: Telescope alt/az (wedge frame) → Local horizontal elev/az
def telescope_to_local_elev_az(alt_tel_deg, az_tel_deg, lat_deg=45.0):
    alt_t = np.deg2rad(alt_tel_deg)
    az_t = np.deg2rad(az_tel_deg)
    lat = np.deg2rad(lat_deg)
    
    sin_el = np.sin(alt_t) * np.cos(lat) + np.cos(alt_t) * np.sin(lat) * np.cos(az_t)
    el_local_deg = np.rad2deg(np.arcsin(sin_el))
    
    cos_el = np.cos(np.deg2rad(el_local_deg))
    cos_el = np.maximum(cos_el, 1e-8)
    
    sin_az = np.cos(alt_t) * np.sin(az_t) / cos_el
    cos_az = (np.cos(alt_t) * np.cos(az_t) * np.cos(lat) - np.sin(alt_t) * np.sin(lat)) / cos_el
    
    az_local_deg = np.rad2deg(np.arctan2(sin_az, cos_az)) % 360
    return el_local_deg, az_local_deg

# Reverse: Local horizontal elev/az → Telescope alt/az (wedge frame)
def local_elev_az_to_telescope(el_local_deg, az_local_deg, lat_deg=45.0):
    el = np.deg2rad(el_local_deg)
    az = np.deg2rad(az_local_deg)
    lat = np.deg2rad(lat_deg)
    
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    
    sin_el = np.sin(el)
    cos_el = np.cos(el)
    sin_az = np.sin(az)
    cos_az = np.cos(az)
    
    D = cos_lat * sin_el - sin_lat * cos_el * cos_az
    A = sin_lat * sin_el + cos_lat * cos_el * cos_az
    B = cos_el * sin_az
    
    alt_tel_deg = np.rad2deg(np.arcsin(D))
    cos_alt = np.sqrt(1 - D**2)
    cos_alt = np.maximum(cos_alt, 1e-8)
    
    sin_az_tel = B / cos_alt
    cos_az_tel = A / cos_alt
    az_tel_deg = np.rad2deg(np.arctan2(sin_az_tel, cos_az_tel)) % 360
    
    return alt_tel_deg, az_tel_deg

# Equatorial coordinates from local horizontal
def altaz_local_to_radec(el_deg, az_deg, lat_deg=45.0, lst_hours=17.89):
    el = np.deg2rad(el_deg)
    az = np.deg2rad(az_deg)
    lat = np.deg2rad(lat_deg)
    
    sin_dec = np.sin(lat) * np.sin(el) + np.cos(lat) * np.cos(el) * np.cos(az)
    dec_deg = np.rad2deg(np.arcsin(sin_dec))
    
    cos_dec = np.cos(np.deg2rad(dec_deg))
    cos_dec = np.maximum(cos_dec, 1e-10)
    sin_ha = -np.cos(el) * np.sin(az) / cos_dec
    cos_ha = (np.cos(lat) * np.sin(el) - np.sin(lat) * np.cos(el) * np.cos(az)) / cos_dec
    ha_deg = np.rad2deg(np.arctan2(sin_ha, cos_ha))
    ha_hours = ha_deg / 15
    ra_hours = (lst_hours - ha_hours) % 24
    
    return ra_hours, dec_deg
