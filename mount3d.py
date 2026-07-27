"""3D mount visualization ("Mount 3D" tab).

A software-3D scene of the telescope mount, posed from the live mount axis
angles through the SAME forward transforms the tracker uses, surrounded by
the sky (stars, satellites, trajectories, keepout) so the operator can build
alignment intuition -- especially in AltAz-Side mode, where the AZM axis lies
on the horizon and the axis/sky relationship is otherwise hard to picture.

World frame: ENU right-handed -- X=East, Y=North, Z=Up. This is deliberately
NOT transformations.cartesian_from_az_el's (North, East, Up); the two frames
differ by a reflection, so cross products change sign between them. This
module never imports that function -- conversions happen only through az/el
degrees at the boundaries, and test_mount3d pins the basis.

Kinematics: every supported mount mode is the SAME two-axis equatorial chain
(rotate about a polar axis by an hour-angle-like angle H, then about the
declination axis by d) with a different pole and (H, d) <- (AZM, ALT) map:

    mode        pole                          H            d
    AltAz       zenith                        AZM+align_az 90-ALT
    Passthrough zenith                        AZM          ALT
    AltAz-Side  (align_az, 0)  horizon pole   AZM+/-90     90-ALT
    Eq          (align_az, align_el or lat)   AZM          ALT

matching transformations.eq_mount_to_azel / AzAlt2AzEl_* exactly. The parity
test (model boresight vs the app's forward transform) pins every sign.

Scene scale: 1 unit = 1 meter (mount head ~1.3 m up); sky content is drawn
as directions on a far sphere R_SKY.
"""

import math

import numpy as np
import pygame


# ------------------------------------------------------------------ frames

UP = np.array([0.0, 0.0, 1.0])
NORTH = np.array([0.0, 1.0, 0.0])
EAST = np.array([1.0, 0.0, 0.0])

R_SKY = 50.0          # sky-sphere radius (scene units = meters)
HEAD_HEIGHT = 1.25    # mount head (axis intersection) above ground
BASE_DIST = 6.0       # orbit-view camera distance at zoom 1


def unit_from_azel(az_deg, el_deg):
    """Sky direction as an ENU unit vector. Valid for any el (past-zenith
    el>90 folds through the pole exactly like the physical pointing)."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    ce = math.cos(e)
    return np.array([ce * math.sin(a), ce * math.cos(a), math.sin(e)])


def azel_from_unit(v):
    """Inverse of unit_from_azel (el in [-90, 90], az in [0, 360))."""
    az = math.degrees(math.atan2(v[0], v[1])) % 360.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, float(v[2])))))
    return az, el


def rot_about(axis, angle_deg):
    """Rodrigues rotation matrix about a unit axis (right-hand rule)."""
    x, y, z = axis
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _eq_basis_enu(pole_az_deg, pole_alt_deg):
    """ENU mirror of transformations._eq_basis: (p, x, y) with p the pole,
    x the upper-meridian equator direction, y chosen so the boresight formula
    v = cos(d)(cos(H) x + sin(H) y) + sin(d) p reproduces eq_mount_to_azel.
    NEU->ENU is a reflection, so the NEU 'y = p cross x' becomes
    'y = x cross p' here (cross products flip sign under reflection)."""
    p = unit_from_azel(pole_az_deg, pole_alt_deg)
    x = UP - np.dot(UP, p) * p
    nx = np.linalg.norm(x)
    if nx < 1e-9:                       # pole at zenith: pick the north meridian
        x = NORTH - np.dot(NORTH, p) * p
        x /= np.linalg.norm(x)
    else:
        x /= nx
    y = np.cross(x, p)
    return p, x, y


# ------------------------------------------------------------- mount modes

def _config_floats(config_state):
    def f(name, default=0.0):
        try:
            return float(getattr(config_state, name, default) or default)
        except (TypeError, ValueError):
            return default
    return f


def _eq_params(config_state, azm_deg, alt_deg):
    """(pole_az, pole_alt, H, d) for the generic chain, per mount mode.
    Mirrors the tracker's conventions (control.py sky_target_to_mount /
    transformations.AzAlt2AzEl_*), including the Eq pole-altitude fallback
    to site latitude and the AltAz-Side index-home H offset."""
    f = _config_floats(config_state)
    mode = getattr(config_state, 'mount_mode', 'AltAz')
    align_az = f('alignment_azimuth_str')
    if mode == 'AltAz':
        return 0.0, 90.0, azm_deg + align_az, 90.0 - alt_deg
    if mode == 'Passthrough':
        return 0.0, 90.0, azm_deg, alt_deg
    if mode == 'AltAz-Side':
        from transformations import ALTAZ_SIDE_H0_DEG
        h0 = -ALTAZ_SIDE_H0_DEG if bool(getattr(config_state, 'altaz_side_flip', False)) \
            else ALTAZ_SIDE_H0_DEG
        return align_az, 0.0, azm_deg + h0, 90.0 - alt_deg
    # Eq: pole altitude falls back to site latitude when the alignment
    # elevation is unset -- the tracker's convention (control.py Eq branch).
    pole_alt = f('alignment_elevation_str')
    if pole_alt == 0.0:
        pole_alt = f('lat_str')
    return align_az, pole_alt, azm_deg, alt_deg


def mount_forward(config_state, azm_deg, alt_deg):
    """Mount axes -> sky (az, el) via the app's OWN forward transforms (the
    tracker's truth), with an explicit branch for every mode -- unlike the
    older display dispatches, Passthrough does not leak into the Eq path."""
    f = _config_floats(config_state)
    mode = getattr(config_state, 'mount_mode', 'AltAz')
    align_az = f('alignment_azimuth_str')
    if mode == 'AltAz':
        from transformations import AzAlt2AzEl_AltAz
        return AzAlt2AzEl_AltAz(azm_deg, alt_deg, align_az)
    if mode == 'Passthrough':
        from transformations import AzAlt2AzEl_Passthrough
        return AzAlt2AzEl_Passthrough(azm_deg, alt_deg)
    if mode == 'AltAz-Side':
        from transformations import AzAlt2AzEl_AltAzSide
        return AzAlt2AzEl_AltAzSide(
            azm_deg, alt_deg, align_az,
            flip=bool(getattr(config_state, 'altaz_side_flip', False)))
    from transformations import eq_mount_to_azel
    pole_az, pole_alt, _h, _d = _eq_params(config_state, azm_deg, alt_deg)
    return eq_mount_to_azel(azm_deg, alt_deg, pole_az, pole_alt)


def _chain_frame(p, x, y, H_deg, d_deg):
    """Orientation matrix M(H, d) of the generic chain, with boresight
    M @ x_home semantics: O1 rotates about the pole (H increases from x
    toward y, i.e. rot(p, -H) since y = x cross p here), then O2 rotates
    about the instantaneous dec axis (e(H) cross p) by +d."""
    O1 = rot_about(p, -H_deg)
    e = O1 @ x                              # equator direction at hour angle H
    a2 = np.cross(e, p)                     # dec axis at this H
    O2 = rot_about(a2, d_deg)
    return O2 @ O1, a2


def mount_pose(config_state, azm_deg, alt_deg):
    """Articulated pose for the 3D model.

    Returns dict:
      p            : axis-1 (polar/azimuth) direction, world
      axis2_world  : axis-2 (dec/alt) direction at the current pose, world
      boresight    : boresight unit vector, world (== unit_from_azel of
                     mount_forward -- pinned by the parity test)
      R_ax1        : 3x3 posing axis-1-stage geometry from its home pose
      R_ax2        : 3x3 posing axis-2-stage geometry from its home pose
      home         : dict with the same fields at mount (0, 0), for sculpting
    Geometry contract: model parts are built in WORLD coordinates at the
    mount-home pose (AZM=ALT=0); the renderer applies R_ax1/R_ax2 (rotations
    about the head point) to pose them.
    """
    pole_az, pole_alt, H, d = _eq_params(config_state, azm_deg, alt_deg)
    _pa, _pl, H0, d0 = _eq_params(config_state, 0.0, 0.0)
    p, x, y = _eq_basis_enu(pole_az, pole_alt)

    M, a2 = _chain_frame(p, x, y, H, d)
    M0, a2_0 = _chain_frame(p, x, y, H0, d0)

    boresight = M @ x
    bore0 = M0 @ x

    # Stage rotations relative to home: parts sculpted at the home pose are
    # posed by R = M_now @ M_home^T (a pure rotation about the head point).
    O1_now = rot_about(p, -(H - H0))
    R_ax1 = O1_now                                  # axis-1 stage: about pole only
    R_ax2 = M @ M0.T                                # axis-2 stage: full chain

    return {
        'p': p,
        'axis2_world': a2,
        'boresight': boresight,
        'R_ax1': R_ax1,
        'R_ax2': R_ax2,
        'home': {'boresight': bore0, 'axis2_world': a2_0, 'p': p},
    }


# ------------------------------------------------------------------- state

class Mount3DState:
    """Per-session view/interaction state for the Mount 3D screen."""

    def __init__(self):
        self.view_mode = 'orbit'      # 'orbit' | 'operator'
        # Orbit view: camera position direction from the mount head.
        self.view_az = 150.0
        self.view_el = 18.0
        self.zoom = 1.0               # orbit distance divisor, clamp [0.3, 4]
        # Operator view: absolute look direction (initialized toward the head
        # on first entry / seat change).
        self.look_az = None
        self.look_el = None
        self.eye_zoom = 1.0           # focal multiplier in operator view
        # Interaction
        self.dragging_view = False
        self.dragging_slider = None   # 'azm' | 'alt' | None
        # Pose source
        self.follow_live = True
        self.manual_azm = 0.0
        self.manual_alt = 0.0
        # HUD hit rects, rebuilt each frame by the renderer.
        self.ui_rects = {}


# ------------------------------------------------------------- view/project

def _camera(m3d, config_state, viewport_w, viewport_h):
    """(R rows [right, up, forward], cam_pos, focal_px). Orbit view circles
    the mount head; operator view sits at the configured seat with true eye
    height, so foreground parallax matches what the operator actually sees."""
    head = np.array([0.0, 0.0, HEAD_HEIGHT])
    f = _config_floats(config_state)
    if m3d.view_mode == 'operator':
        bearing = f('mount3d_observer_bearing_deg', 200.0)
        dist = max(0.5, f('mount3d_observer_distance_m', 2.5))
        eye_h = f('mount3d_eye_height_m', 1.2)
        seat = unit_from_azel(bearing, 0.0) * dist
        cam = np.array([seat[0], seat[1], eye_h])
        if m3d.look_az is None:
            to_head = head - cam
            m3d.look_az, m3d.look_el = azel_from_unit(
                to_head / np.linalg.norm(to_head))
        fwd = unit_from_azel(m3d.look_az, max(-89.0, min(89.0, m3d.look_el)))
        focal = 0.9 * viewport_h * m3d.eye_zoom
    else:
        # Full +/-89 orbit range: diving below the ground plane and looking
        # straight up through the mount is a legitimate (and requested) view.
        cam_dir = unit_from_azel(m3d.view_az,
                                 max(-89.0, min(89.0, m3d.view_el)))
        cam = head + cam_dir * (BASE_DIST / m3d.zoom)
        fwd = -cam_dir
        focal = 0.9 * viewport_h
    right = np.cross(fwd, UP)
    n = np.linalg.norm(right)
    right = EAST if n < 1e-6 else right / n
    up = np.cross(right, fwd)
    R = np.stack([right, up, fwd])
    return R, cam, focal


_COORD_CLAMP = 20000.0


def project_points(pts, R, cam_pos, cx, cy, focal):
    """Vectorized perspective projection of Nx3 world points.
    Returns (sx, sy, z_cam); a point is drawable when z_cam > 0."""
    pc = (np.atleast_2d(pts) - cam_pos) @ R.T
    z = pc[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    sx = np.clip(cx + focal * pc[:, 0] / safe_z, -_COORD_CLAMP, _COORD_CLAMP)
    sy = np.clip(cy - focal * pc[:, 1] / safe_z, -_COORD_CLAMP, _COORD_CLAMP)
    return sx, sy, z


def project_dirs(dirs, R, cx, cy, focal):
    """Project sky DIRECTIONS (points at infinity): translation-free, correct
    for both view modes. Returns (sx, sy, z_cam)."""
    pc = np.atleast_2d(dirs) @ R.T
    z = pc[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    sx = np.clip(cx + focal * pc[:, 0] / safe_z, -_COORD_CLAMP, _COORD_CLAMP)
    sy = np.clip(cy - focal * pc[:, 1] / safe_z, -_COORD_CLAMP, _COORD_CLAMP)
    return sx, sy, z


# ---------------------------------------------------------------- geometry

_HEAD = np.array([0.0, 0.0, HEAD_HEIGHT])
AXIS1_COLOR = (80, 210, 255)     # cyan: the AZM (polar) axis arrow
AXIS2_COLOR = (255, 120, 235)    # magenta: the ALT (dec) axis arrow
FOV_COLORS = ((255, 0, 0), (255, 165, 0))   # cam1 red, cam2 orange (skyplot match)


def _ortho_pair(axis):
    """Two unit vectors perpendicular to `axis` (and each other)."""
    ref = UP if abs(float(np.dot(axis, UP))) < 0.9 else NORTH
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _box(center, half_vecs, color, stage):
    """Parallelepiped part: center + Σ ±half_vecs; 6 quad faces."""
    hx, hy, hz = half_vecs
    verts = np.array([center + sx * hx + sy * hy + sz * hz
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return {'verts': verts, 'faces': faces, 'color': color, 'stage': stage}


def _tube(base, axis, radius, length, color, stage, nsides=8):
    """Prism (octagonal by default) from `base` along unit `axis`."""
    u, v = _ortho_pair(axis)
    ring = [base + radius * (math.cos(2 * math.pi * i / nsides) * u
                             + math.sin(2 * math.pi * i / nsides) * v)
            for i in range(nsides)]
    verts = np.array(ring + [q + axis * length for q in ring])
    faces = []
    for i in range(nsides):
        j = (i + 1) % nsides
        faces.append((i, j, nsides + j, nsides + i))
    faces.append(tuple(range(nsides)))                       # end caps
    faces.append(tuple(range(2 * nsides - 1, nsides - 1, -1)))
    return {'verts': verts, 'faces': faces, 'color': color, 'stage': stage}


def _arrow(base, axis, length, color, stage):
    """Axis arrow: thin square shaft + pyramid head, along unit `axis`."""
    u, v = _ortho_pair(axis)
    s = 0.02
    tip_base = base + axis * (length * 0.82)
    shaft = _box(base + axis * (length * 0.41),
                 (u * s, v * s, axis * (length * 0.41)), color, stage)
    hs = 0.055
    ring = [tip_base + hs * (math.cos(a) * u + math.sin(a) * v)
            for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]
    verts = np.array(ring + [base + axis * length])
    faces = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (3, 2, 1, 0)]
    head = {'verts': verts, 'faces': faces, 'color': color, 'stage': stage}
    return [shaft, head]


# Fork-arm sculpt dimensions (model metres). Rigid attachment chain:
# tripod/pier -> center of the AXIS-1 (AZ) tube -> its top end carries one
# end of the AXIS-2 (ALT) tube -> whose far end carries the OTA by its side
# (single fork arm, NexStar/AVX alt-az style). The OTA centerline is offset
# from the head along axis 2 by OTA_ARM_OFFSET -- the boresight ray and FOV
# cone apex use the same offset so the optics emanate from the tube, not
# from inside the arm.
AZ_TUBE_R, AZ_TUBE_LEN = 0.10, 0.40
ALT_TUBE_R, ALT_TUBE_LEN = 0.085, 0.32
OTA_R, OTA_LEN = 0.14, 1.00
OTA_ARM_OFFSET = ALT_TUBE_LEN + OTA_R * 0.85   # OTA side flush on the arm end


def build_mount_geometry(pose):
    """Model parts in WORLD coordinates at the mount-home pose (AZM=ALT=0).
    Stages: 'base' fixed, 'ax1' rotates with axis 1, 'ax2' with the full
    chain. The renderer poses ax1/ax2 parts about the head point via
    pose['R_ax1'] / pose['R_ax2']."""
    p = pose['home']['p']
    b0 = pose['home']['boresight']
    a2 = pose['home']['axis2_world']
    saddle_up = np.cross(a2, b0)
    n = np.linalg.norm(saddle_up)
    saddle_up = UP if n < 1e-6 else saddle_up / n

    parts = []
    # BASE: pier + three tripod legs.
    pier_h = HEAD_HEIGHT - 0.18
    parts.append(_box(np.array([0.0, 0.0, pier_h / 2]),
                      (EAST * 0.06, NORTH * 0.06, UP * (pier_h / 2)),
                      (70, 70, 80), 'base'))
    for leg_az in (90.0, 210.0, 330.0):
        foot = unit_from_azel(leg_az, 0.0) * 0.6
        top = np.array([0.0, 0.0, pier_h * 0.85])
        mid = (foot + top) / 2
        leg = top - foot
        leg_len = np.linalg.norm(leg)
        leg_dir = leg / leg_len
        lu, lv = _ortho_pair(leg_dir)
        parts.append(_box(mid, (lu * 0.025, lv * 0.025, leg_dir * (leg_len / 2)),
                          (55, 55, 65), 'base'))

    # AX1 stage: the AZ-axis tube, its CENTER sitting on the pier top (the
    # tripod attachment), its top end reaching the head where the ALT tube
    # bolts on. Plus the cyan axis-1 arrow.
    az_base = _HEAD - p * (0.18 + AZ_TUBE_LEN / 2)   # center at pier top
    parts.append(_tube(az_base, p, AZ_TUBE_R, AZ_TUBE_LEN, (110, 112, 125), 'ax1'))
    parts.extend(_arrow(_HEAD + p * 0.06, p, 0.62, AXIS1_COLOR, 'ax1'))

    # AX1 stage: the ALT-axis tube (fork arm). One end cap rigidly at the AZ
    # tube's top end (the head); it extends sideways along axis 2. It swings
    # with azimuth but not with altitude, hence 'ax1'.
    parts.append(_tube(_HEAD, a2, ALT_TUBE_R, ALT_TUBE_LEN, (120, 122, 138), 'ax1'))

    # AX2 stage: the OTA, fatter, side-mounted -- its side surface rides the
    # far end of the ALT tube, centerline on the axis-2 line so it pivots in
    # place. Guide-scope stub on top, magenta axis-2 arrow poking out past
    # the far side of the OTA.
    ota_center = _HEAD + a2 * OTA_ARM_OFFSET
    ota_base = ota_center - b0 * (OTA_LEN * 0.45)
    parts.append(_tube(ota_base, b0, OTA_R, OTA_LEN, (168, 172, 185), 'ax2', nsides=10))
    parts.append(_tube(ota_base + saddle_up * (OTA_R + 0.032) + b0 * (OTA_LEN * 0.3),
                       b0, 0.032, 0.42, (140, 150, 170), 'ax2'))
    parts.extend(_arrow(_HEAD + a2 * (OTA_ARM_OFFSET + OTA_R + 0.02), a2,
                        0.45, AXIS2_COLOR, 'ax2'))
    return parts


def pose_part_verts(part, pose):
    """Pose a part's home-pose vertices for the current mount angles."""
    stage = part['stage']
    if stage == 'base':
        return part['verts']
    R = pose['R_ax1'] if stage == 'ax1' else pose['R_ax2']
    return (part['verts'] - _HEAD) @ R.T + _HEAD


# --------------------------------------------------------------- FOV cones

def camera_fov_specs(config_state):
    """Per-camera FOV parameters for the cones, self-contained from config
    (+ live ROI when a camera is connected). Safe with no cameras at all."""
    from transformations import compute_fov_for_camera
    specs = []
    for idx in (0, 1):
        cam_cfg = getattr(config_state, 'camera_configs', {}).get(f"camera{idx + 1}", {})
        try:
            pixel_size = float(cam_cfg.get('pixel_size', 2.9))
            focal_mm = float(cam_cfg.get('focal_length', 162.0))
            rotation = float(cam_cfg.get('alignment_rotation', 0.0))
        except (TypeError, ValueError):
            continue
        width_px, height_px, roi_pct = 1920, 1280, 1.0
        try:
            from camera_manager import camera_manager
            cam_obj = camera_manager.get_camera(idx)
            if cam_obj is not None and getattr(cam_obj, 'connected', False):
                width_px = cam_obj.width_res
                height_px = cam_obj.height_res
                if getattr(cam_obj, 'roi_size', 0) > 0:
                    roi_pct = 0.5 ** cam_obj.roi_size
        except Exception:
            pass
        fov = compute_fov_for_camera(pixel_size, focal_mm, roi_pct, roi_pct,
                                     width_px, height_px)
        specs.append({'camera_id': idx,
                      'fov_width_deg': fov['fov_width_deg'],
                      'fov_height_deg': fov['fov_height_deg'],
                      'rotation': rotation,
                      'color': FOV_COLORS[idx]})
    return specs


def fov_corner_dirs(boresight, axis2_world, fov_width_deg, fov_height_deg,
                    rotation_deg):
    """Unit directions of the 4 FOV corners: pinhole corners built on the
    tube's (axis2, up) frame, then rolled about the boresight by the camera's
    alignment rotation. Corner order: (-w,-h), (+w,-h), (+w,+h), (-w,+h)."""
    b = boresight / np.linalg.norm(boresight)
    right = axis2_world - np.dot(axis2_world, b) * b
    n = np.linalg.norm(right)
    if n < 1e-9:
        right, _ = _ortho_pair(b)
    else:
        right = right / n
    up_t = np.cross(b, right)
    tw = math.tan(math.radians(fov_width_deg / 2.0))
    th = math.tan(math.radians(fov_height_deg / 2.0))
    roll = rot_about(b, rotation_deg)
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        d = b + right * (sx * tw) + up_t * (sy * th)
        d = roll @ d
        corners.append(d / np.linalg.norm(d))
    return corners


# ------------------------------------------------------------ scene caches

_GEOM_CACHE = {'key': None, 'parts': None}
_KEEPOUT_CACHE = {'key': None, 'dirs': None}
_FONTS = {}


def _font(size):
    if size not in _FONTS:
        pygame.font.init()
        _FONTS[size] = pygame.font.Font(None, size)
    return _FONTS[size]


def _geometry_for(config_state):
    """Home-pose model parts, cached on everything that shapes the home pose."""
    f = _config_floats(config_state)
    key = (getattr(config_state, 'mount_mode', 'AltAz'),
           f('alignment_azimuth_str'), f('alignment_elevation_str'),
           f('lat_str'), bool(getattr(config_state, 'altaz_side_flip', False)))
    if _GEOM_CACHE['key'] != key:
        _GEOM_CACHE['parts'] = build_mount_geometry(
            mount_pose(config_state, 0.0, 0.0))
        _GEOM_CACHE['key'] = key
    return _GEOM_CACHE['parts']


def _keepout_dirs(config_state):
    """Unit directions of keepout sky cells (5-deg grid): sky directions with
    NO in-limits mount-axis solution -- the same semantics as the skyplot's
    keepout wash (rendering_threads). Cached until the shaping config changes.
    Returns an Nx3 array (possibly empty) or None when limits don't parse."""
    from control import mount_target_for
    f = _config_floats(config_state)
    key = (getattr(config_state, 'mount_mode', 'AltAz'),
           bool(getattr(config_state, 'altaz_side_flip', False)),
           str(getattr(config_state, 'azm_limit_min_str', '')),
           str(getattr(config_state, 'azm_limit_max_str', '')),
           str(getattr(config_state, 'alt_limit_min_str', '')),
           str(getattr(config_state, 'alt_limit_max_str', '')),
           f('alignment_azimuth_str'), f('alignment_elevation_str'), f('lat_str'))
    if _KEEPOUT_CACHE['key'] == key:
        return _KEEPOUT_CACHE['dirs']
    try:
        azm_min = float(config_state.azm_limit_min_str)
        azm_max = float(config_state.azm_limit_max_str)
        alt_min = float(config_state.alt_limit_min_str)
        alt_max = float(config_state.alt_limit_max_str)
    except (TypeError, ValueError, AttributeError):
        _KEEPOUT_CACHE['key'] = key
        _KEEPOUT_CACHE['dirs'] = None
        return None
    mode = getattr(config_state, 'mount_mode', 'AltAz')
    flips = (False, True) if mode in ('AltAz', 'Passthrough') else (False,)
    out = []
    for el in range(3, 90, 5):
        for az in range(0, 360, 5):
            reachable = False
            for flip in flips:
                try:
                    azm, alt = mount_target_for(config_state, float(az), float(el), flip)
                except Exception:
                    continue
                if azm_min <= azm <= azm_max and alt_min <= alt <= alt_max:
                    reachable = True
                    break
            if not reachable:
                out.append(unit_from_azel(float(az), float(el)))
    dirs = np.array(out) if out else np.zeros((0, 3))
    _KEEPOUT_CACHE['key'] = key
    _KEEPOUT_CACHE['dirs'] = dirs
    return dirs


# ----------------------------------------------------------------- render

def _draw_sky_polyline(surface, samples_azel, R, cx, cy, focal, color, width):
    """Polyline through sky (az, el) samples, broken at the view boundary."""
    run = []
    for az, el in samples_azel:
        d = unit_from_azel(az, el)
        sx, sy, z = project_dirs(d[None, :], R, cx, cy, focal)
        if z[0] > 0.08:
            run.append((int(sx[0]), int(sy[0])))
        else:
            if len(run) >= 2:
                pygame.draw.lines(surface, color, False, run, width)
            run = []
    if len(run) >= 2:
        pygame.draw.lines(surface, color, False, run, width)


def _current_pose_angles(m3d, joystick_mode_state):
    """(azm, alt, live) driving the model this frame."""
    live_ok = (m3d.follow_live and joystick_mode_state is not None
               and getattr(joystick_mode_state, 'telescope_connected', False))
    if live_ok:
        return (float(getattr(joystick_mode_state, 'current_azm', 0.0)),
                float(getattr(joystick_mode_state, 'current_alt', 0.0)), True)
    return m3d.manual_azm, m3d.manual_alt, False


def render_mount3d_on_surface(surface, config_state, tracking_vis_state,
                              joystick_mode_state, m3d, ts):
    """Render the full Mount 3D scene onto `surface` (any size)."""
    w, h = surface.get_size()
    cx, cy = w // 2, int(h * 0.52)
    surface.fill((8, 10, 16))
    R, cam, focal = _camera(m3d, config_state, w, h)
    f = _config_floats(config_state)

    # Shared tracking-clock policy (slider / paused / live, with the
    # unset-guard) -- one implementation, so this can't drift from the
    # render threads' notion of scene time.
    current_tt = None
    if ts is not None:
        try:
            from rendering_threads import render_time_tt
            current_tt = render_time_tt(tracking_vis_state, ts)
        except Exception:
            current_tt = None

    # Two translucent layers: sky-level tint (keepout dots, ground disc)
    # composites UNDER the mount model so it can't wash over the hardware;
    # the top layer (boresight ray, FOV cones) composites over everything.
    overlay_sky = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)

    # --- Sky pass -----------------------------------------------------------
    # Elevation rings + horizon + cardinals (under everything else).
    az_samples = [(a, 0.0) for a in range(0, 361, 5)]
    for el_ring in (30.0, 60.0):
        _draw_sky_polyline(surface, [(a, el_ring) for a in range(0, 361, 10)],
                           R, cx, cy, focal, (38, 38, 46), 1)
    mask = f('elevation_mask_str')
    if mask > 0.0:
        _draw_sky_polyline(surface, [(a, mask) for a in range(0, 361, 5)],
                           R, cx, cy, focal, (140, 30, 30), 1)
    _draw_sky_polyline(surface, az_samples, R, cx, cy, focal, (200, 200, 210), 2)
    for az_c, lbl in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        d = unit_from_azel(float(az_c), 2.0)
        sx, sy, z = project_dirs(d[None, :], R, cx, cy, focal)
        if z[0] > 0.08:
            col = (255, 130, 130) if lbl == "N" else (225, 225, 235)
            t = _font(20).render(lbl, True, col)
            surface.blit(t, (int(sx[0]) - t.get_width() // 2,
                             int(sy[0]) - t.get_height() // 2))

    # Keepout dome tint (light red dots on unreachable sky).
    ko = _keepout_dirs(config_state)
    if ko is not None and len(ko):
        sx, sy, z = project_dirs(ko, R, cx, cy, focal)
        vis = z > 0.08
        for X, Y in zip(sx[vis], sy[vis]):
            if -20 <= X <= w + 20 and -20 <= Y <= h + 20:
                pygame.draw.circle(overlay_sky, (255, 70, 70, 46),
                                   (int(X), int(Y)), 3)

    # Stars.
    if current_tt is not None and getattr(config_state, 'starfield_enabled', True):
        try:
            from star_catalog import get_catalog
            data = get_catalog(getattr(tracking_vis_state, 'ephemeris', None)
                               if tracking_vis_state is not None else None
                               ).current_altaz(
                f('lat_str'), f('lon_str'), f('alt_str'), ts, current_tt,
                elevation_mask=0.0,
                max_count=int(getattr(config_state, 'max_rendered_star_count', 2000)),
                limiting_mag=float(getattr(config_state, 'star_limiting_magnitude', 6.5)))
            az_r = np.radians(data['az'])
            el_r = np.radians(data['el'])
            ce = np.cos(el_r)
            dirs = np.stack([ce * np.sin(az_r), ce * np.cos(az_r),
                             np.sin(el_r)], axis=1)
            sx, sy, z = project_dirs(dirs, R, cx, cy, focal)
            mags = data['mag']
            lo, hi = (float(np.min(mags)), float(np.max(mags))) if len(mags) else (0, 1)
            span = max(1e-6, hi - lo)
            for i in np.nonzero(z > 0.08)[0]:
                X, Y = sx[i], sy[i]
                if not (-5 <= X <= w + 5 and -5 <= Y <= h + 5):
                    continue
                norm = (float(mags[i]) - lo) / span
                rad = max(1, int(round(2.6 - 2.0 * norm)))
                shade = int(np.clip(255 - 150 * norm, 90, 255))
                pygame.draw.circle(surface, (shade, shade, shade),
                                   (int(X), int(Y)), rad)
        except Exception as e:
            print(f"Mount3D starfield error: {e}")

    # Satellites (+ aircraft): interpolated trajectory position; cols 1/2 are
    # sky el/az (cols 4/5 are baked polar-plot pixels -- never those).
    def draw_traj_markers(positions, trajectories, selected, color):
        if not positions or not trajectories or current_tt is None:
            return
        drawn = 0
        for key_obj in positions:
            traj = trajectories.get(key_obj)
            if not traj:
                continue
            rows, times = traj[0], traj[1]
            if len(rows) == 0 or len(times) == 0:
                continue
            # searchsorted returns the sample AFTER current_tt; every other
            # interpolation site subtracts 1 and blends. Snapping to the next
            # sample made Mount 3D lead the skyplot by up to one 30 s sample.
            i = int(np.clip(np.searchsorted(times, current_tt) - 1, 0, len(rows) - 1))
            if i < len(rows) - 1 and times[i + 1] > times[i]:
                frac = (current_tt - times[i]) / (times[i + 1] - times[i])
                frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
                el_s = float(rows[i][1]) + frac * (float(rows[i + 1][1]) - float(rows[i][1]))
                d_az = ((float(rows[i + 1][2]) - float(rows[i][2]) + 180.0) % 360.0) - 180.0
                az_s = (float(rows[i][2]) + frac * d_az) % 360.0
            else:
                el_s, az_s = float(rows[i][1]), float(rows[i][2])
            if el_s <= 0.0:
                continue
            d = unit_from_azel(az_s, el_s)
            sx, sy, z = project_dirs(d[None, :], R, cx, cy, focal)
            if z[0] <= 0.08:
                continue
            X, Y = int(sx[0]), int(sy[0])
            if not (0 <= X < w and 0 <= Y < h):
                continue
            is_sel = key_obj == selected
            pygame.draw.circle(surface, (255, 255, 100) if is_sel else color,
                               (X, Y), 4 if is_sel else 2)
            if is_sel:
                pygame.draw.circle(surface, (255, 255, 100), (X, Y), 8, 1)
            drawn += 1
            if drawn > 500:
                break

    tvs = tracking_vis_state
    if tvs is not None:
        if getattr(config_state, 'satellites_enabled', True):
            draw_traj_markers(getattr(tvs, 'satellite_positions', None),
                              getattr(tvs, 'satellite_trajectories', None),
                              getattr(tvs, 'selected_satellite', None), (170, 190, 140))
        if getattr(config_state, 'aircraft_enabled', True):
            draw_traj_markers(getattr(tvs, 'aircraft_positions', None),
                              getattr(tvs, 'aircraft_trajectories', None),
                              getattr(tvs, 'selected_aircraft', None), (90, 200, 220))

    # Celestial layer on the sky dome: sun/moon/planets always (pinned dimmed
    # on the horizon when below it, like the skyplot), Messier/NGC per the
    # toggles. Stars come from the starfield above; 'star' anchor entries are
    # selection-only on the skyplot and are skipped here.
    if current_tt is not None:
        try:
            from celestial import get_celestial
            cat = get_celestial(getattr(tvs, 'ephemeris', None)
                                if tvs is not None else None)
            objs = cat.compute_plot_objects(config_state, ts, current_tt,
                                            elevation_mask=0.0)
            sel = getattr(tvs, 'selected_celestial', None) if tvs is not None else None
            for obj in objs:
                kind = obj['kind']
                if kind == 'star':
                    continue
                el = obj['el']
                body_like = kind in ('sun', 'moon', 'planet')
                below = el < 0.0
                if below and not body_like:
                    continue
                d = unit_from_azel(obj['az'], max(el, 0.0))
                sx, sy, z = project_dirs(d[None, :], R, cx, cy, focal)
                if z[0] <= 0.08:
                    continue
                X, Y = int(sx[0]), int(sy[0])
                if not (-10 <= X < w + 10 and -10 <= Y < h + 10):
                    continue
                color = obj['color'] or ((205, 130, 255) if kind == 'messier'
                                         else (90, 200, 180))
                if below:
                    color = tuple(c // 2 for c in color)
                if body_like:
                    pygame.draw.circle(surface, color, (X, Y), obj['radius'] + 1)
                    if obj['key'] == 'sun':
                        pygame.draw.circle(surface, (255, 245, 160), (X, Y),
                                           obj['radius'] + 4, 1)
                    name = obj['name'] + (" (below hrz)" if below else "")
                    t = _font(16).render(name, True, (235, 235, 245))
                    surface.blit(t, (X + obj['radius'] + 5, Y - 8))
                elif kind == 'messier':
                    pygame.draw.rect(surface, color,
                                     pygame.Rect(X - 3, Y - 3, 7, 7), 1)
                    t = _font(14).render(obj['name'].split()[0], True,
                                         (205, 150, 245))
                    surface.blit(t, (X + 6, Y - 7))
                else:  # ngc
                    pygame.draw.circle(surface, color, (X, Y), 3, 1)
                if obj['key'] == sel:
                    pygame.draw.circle(surface, (255, 255, 0), (X, Y),
                                       obj['radius'] + 7, 1)
        except Exception as e:
            print(f"Mount3D celestial error: {e}")

    if tvs is not None:

        # Selected-target trajectory polyline (navball color rules).
        try:
            from joystick_panels import _navball_active_target
            target = _navball_active_target(tvs)
        except Exception:
            target = None
        if target is not None:
            traj, cur_tt, sunlit, tgt_az, tgt_el = target
            rows, _times = traj
            run = []
            for i, row in enumerate(rows):
                t_el, t_az = float(row[1]), float(row[2])
                d = unit_from_azel(t_az, t_el)
                sx, sy, z = project_dirs(d[None, :], R, cx, cy, focal)
                ok = t_el > 0.0 and z[0] > 0.08
                if ok:
                    is_past = float(row[0]) < (cur_tt or 0.0)
                    if is_past:
                        col = (128, 128, 128)
                    elif sunlit is not None and i < len(sunlit) and not sunlit[i]:
                        col = (255, 0, 0)
                    else:
                        col = (255, 255, 0)
                    run.append(((int(sx[0]), int(sy[0])), col))
                else:
                    for (p1, c1), (p2, _c2) in zip(run, run[1:]):
                        pygame.draw.line(surface, c1, p1, p2, 2)
                    run = []
            for (p1, c1), (p2, _c2) in zip(run, run[1:]):
                pygame.draw.line(surface, c1, p1, p2, 2)
            if tgt_az is not None and tgt_el is not None and tgt_el > 0:
                d = unit_from_azel(tgt_az, tgt_el)
                sx, sy, z = project_dirs(d[None, :], R, cx, cy, focal)
                if z[0] > 0.08:
                    X, Y = int(sx[0]), int(sy[0])
                    for dx, dy in ((-7, 0), (7, 0), (0, -7), (0, 7)):
                        pygame.draw.line(surface, (205, 95, 255),
                                         (X + dx, Y + dy), (X + dx // 7, Y + dy // 7), 2)

    # --- Ground + seat marker ----------------------------------------------
    ring = [np.array([3.0 * math.sin(t), 3.0 * math.cos(t), 0.0])
            for t in np.linspace(0, 2 * math.pi, 36, endpoint=False)]
    gsx, gsy, gz = project_points(np.array(ring), R, cam, cx, cy, focal)
    pts = [(int(x), int(y)) for x, y, zz in zip(gsx, gsy, gz) if zz > 0.05]
    if len(pts) >= 3:
        pygame.draw.polygon(overlay_sky, (30, 60, 34, 90), pts)
    # Sky-level tint (keepout dots + ground disc) composites here, UNDER the
    # seat marker and mount model, so translucency can't wash over hardware.
    surface.blit(overlay_sky, (0, 0))
    seat_bearing = f('mount3d_observer_bearing_deg', 200.0)
    seat_dist = max(0.5, f('mount3d_observer_distance_m', 2.5))
    seat = unit_from_azel(seat_bearing, 0.0) * seat_dist
    if m3d.view_mode == 'orbit':
        ssx, ssy, sz = project_points(
            np.array([[seat[0], seat[1], 0.25]]), R, cam, cx, cy, focal)
        if sz[0] > 0.05:
            X, Y = int(ssx[0]), int(ssy[0])
            pygame.draw.circle(surface, (240, 210, 90), (X, Y), 5)
            t = _font(16).render("operator", True, (240, 210, 90))
            surface.blit(t, (X - t.get_width() // 2, Y + 7))

    # --- Mount model (painter's algorithm) ---------------------------------
    azm, alt, live = _current_pose_angles(m3d, joystick_mode_state)
    pose = mount_pose(config_state, azm, alt)
    parts = _geometry_for(config_state)
    light = np.array([0.35, 0.5, 0.79])
    light /= np.linalg.norm(light)
    faces = []
    for part in parts:
        verts = pose_part_verts(part, pose)
        vsx, vsy, vz = project_points(verts, R, cam, cx, cy, focal)
        for face in part['faces']:
            if any(vz[i] <= 0.05 for i in face):
                continue
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            n = np.cross(v1 - v0, v2 - v0)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            shade = 0.55 + 0.45 * abs(float(np.dot(n, light)))
            col = tuple(min(255, int(c * shade)) for c in part['color'])
            zmean = float(np.mean([vz[i] for i in face]))
            faces.append((zmean, [(int(vsx[i]), int(vsy[i])) for i in face], col))
    faces.sort(key=lambda t: -t[0])
    for _z, poly, col in faces:
        pygame.draw.polygon(surface, col, poly)

    # Boresight ray (thin, from the tube out toward the sky sphere). The OTA
    # centerline sits OTA_ARM_OFFSET out along the posed axis 2 (fork arm),
    # so the ray/cones originate there, not at the head inside the arm.
    ota_origin = _HEAD + pose['axis2_world'] * OTA_ARM_OFFSET
    tip = ota_origin + pose['boresight'] * (OTA_LEN * 0.58)
    far = ota_origin + pose['boresight'] * (R_SKY * 0.5)
    bsx, bsy, bz = project_points(np.array([tip, far]), R, cam, cx, cy, focal)
    if bz[0] > 0.05 and bz[1] > 0.05:
        pygame.draw.line(overlay, (255, 255, 255, 110),
                         (int(bsx[0]), int(bsy[0])), (int(bsx[1]), int(bsy[1])), 1)

    # --- FOV cones ----------------------------------------------------------
    apex = ota_origin + pose['boresight'] * (OTA_LEN * 0.56)
    for spec in camera_fov_specs(config_state):
        corners = fov_corner_dirs(pose['boresight'], pose['axis2_world'],
                                  spec['fov_width_deg'], spec['fov_height_deg'],
                                  spec['rotation'])
        cone_len = R_SKY * 0.55
        far_pts = np.array([apex + c * cone_len for c in corners])
        pts_all = np.vstack([apex[None, :], far_pts])
        csx, csy, czz = project_points(pts_all, R, cam, cx, cy, focal)
        p2 = [(int(x), int(y)) for x, y in zip(csx, csy)]
        ok = czz > 0.05
        rC, gC, bC = spec['color']
        # Per-face culling: a cone half-behind the camera keeps its visible
        # side triangles instead of popping out entirely.
        for i in range(4):
            j = 1 + (i + 1) % 4
            if ok[0] and ok[1 + i] and ok[j]:
                pygame.draw.polygon(overlay, (rC, gC, bC, 26),
                                    [p2[0], p2[1 + i], p2[j]])
        if np.all(ok[1:]):
            pygame.draw.polygon(overlay, (rC, gC, bC, 130), p2[1:], 1)

    surface.blit(overlay, (0, 0))

    # --- HUD ----------------------------------------------------------------
    m3d.ui_rects = {}
    sky_az, sky_el = mount_forward(config_state, azm, alt)
    mode = getattr(config_state, 'mount_mode', 'AltAz')
    hud = [
        (f"Mount 3D - {mode}" + ("" if not getattr(config_state, 'altaz_side_flip', False)
                                 or mode != 'AltAz-Side' else " (flipped)"), (255, 255, 255)),
        (f"AZM {azm:7.2f}°   ALT {alt:7.2f}°   {'LIVE' if live else 'MANUAL'}",
         (170, 220, 255) if live else (255, 210, 120)),
        (f"sky az {sky_az % 360.0:6.2f}°   el {sky_el:6.2f}°", (200, 200, 210)),
        ("axis colors: AZM axis cyan, ALT axis magenta", (150, 155, 170)),
        ("drag = orbit/look   wheel = zoom", (120, 125, 140)),
    ]
    y = 8
    for text, col in hud:
        surface.blit(_font(18).render(text, True, col), (10, y))
        y += 20

    def button(name, label, x, by, active):
        t = _font(16).render(label, True, (255, 255, 255))
        rect = pygame.Rect(x, by, t.get_width() + 14, 20)
        pygame.draw.rect(surface, (0, 120, 60) if active else (60, 62, 75), rect)
        pygame.draw.rect(surface, (160, 160, 175), rect, 1)
        surface.blit(t, (x + 7, by + 3))
        m3d.ui_rects[name] = rect
        return rect.right + 8

    bx = 10
    bx = button('view_toggle',
                "VIEW: OPERATOR" if m3d.view_mode == 'operator' else "VIEW: ORBIT",
                bx, y + 4, m3d.view_mode == 'operator')
    bx = button('follow_toggle', "FOLLOW LIVE" if m3d.follow_live else "MANUAL POSE",
                bx, y + 4, m3d.follow_live)
    y += 30

    # Object-type show/hide toggles: the same config flags as the skyplot
    # (tracking_visuals.OBJECT_TOGGLES), so the two views always agree.
    # Sun/moon/planets and the named stars stay always-on by design.
    from tracking_visuals import OBJECT_TOGGLES
    bx = 10
    for attr, label, default in OBJECT_TOGGLES:
        on = bool(getattr(config_state, attr, default))
        bx = button('toggle_' + attr, f"{label} {'On' if on else 'Off'}",
                    bx, y + 4, on)
    y += 26

    # Operator-seat controls (always visible; they also move the marker).
    seat_y = y + 4
    surface.blit(_font(16).render(
        f"seat: bearing {seat_bearing:5.0f}°  dist {seat_dist:4.1f} m  "
        f"eye {f('mount3d_eye_height_m', 1.2):4.2f} m", True, (185, 190, 205)),
        (10, seat_y))
    sy2 = seat_y + 18
    bx = 10
    for name, label in (('seat_bearing_minus', 'brg -'), ('seat_bearing_plus', 'brg +'),
                        ('seat_dist_minus', 'dist -'), ('seat_dist_plus', 'dist +'),
                        ('seat_eye_minus', 'eye -'), ('seat_eye_plus', 'eye +')):
        bx = button(name, label, bx, sy2, False)

    # Manual pose sliders (interactable when not following live).
    if not (m3d.follow_live and joystick_mode_state is not None
            and getattr(joystick_mode_state, 'telescope_connected', False)):
        sl_y = sy2 + 30
        for name, label, value, lo, hi in (
                ('slider_azm', 'AZM', m3d.manual_azm, 0.0, 360.0),
                ('slider_alt', 'ALT', m3d.manual_alt, -90.0, 270.0)):
            surface.blit(_font(16).render(f"{label} {value:7.1f}°", True,
                                          (255, 210, 120)), (10, sl_y))
            track = pygame.Rect(110, sl_y + 6, 180, 5)
            pygame.draw.rect(surface, (90, 92, 105), track)
            frac = (value - lo) / (hi - lo)
            hx = track.x + int(frac * track.width)
            pygame.draw.rect(surface, (255, 210, 120), (hx - 3, sl_y + 2, 6, 13))
            m3d.ui_rects[name] = pygame.Rect(track.x - 6, sl_y, track.width + 12, 18)
            sl_y += 24


def draw_mount3d(display, config_state, tracking_vis_state,
                 joystick_mode_state, m3d, ts):
    """Main-thread draw entry: renders into a persistent surface sized to the
    content sub-area and blits it (the render clears its own background)."""
    size = (display.sub_width, display.sub_height)
    surf = getattr(m3d, '_surface', None)
    if surf is None or surf.get_size() != size:
        surf = pygame.Surface(size)
        m3d._surface = surf
    render_mount3d_on_surface(surf, config_state, tracking_vis_state,
                              joystick_mode_state, m3d, ts)
    display.menu_screen.blit(surf, (display.sub_x, display.sub_y))


# ------------------------------------------------------------------ events

_SEAT_STEPS = {
    'seat_bearing_minus': ('mount3d_observer_bearing_deg', -15.0, 200.0),
    'seat_bearing_plus': ('mount3d_observer_bearing_deg', 15.0, 200.0),
    'seat_dist_minus': ('mount3d_observer_distance_m', -0.25, 2.5),
    'seat_dist_plus': ('mount3d_observer_distance_m', 0.25, 2.5),
    'seat_eye_minus': ('mount3d_eye_height_m', -0.1, 1.2),
    'seat_eye_plus': ('mount3d_eye_height_m', 0.1, 1.2),
}


def _local(display, pos):
    return pos[0] - display.sub_x, pos[1] - display.sub_y


def _slider_value(m3d, name, local_x):
    rect = m3d.ui_rects[name]
    frac = min(1.0, max(0.0, (local_x - (rect.x + 6)) / (rect.width - 12)))
    if name == 'slider_azm':
        m3d.manual_azm = round(frac * 360.0, 1)
    else:
        m3d.manual_alt = round(-90.0 + frac * 360.0, 1)


def handle_mount3d_mouse_down(pos, button, display, m3d, config_state):
    """HUD buttons / sliders first, else start a view drag. Returns True if
    the event was consumed."""
    if button not in (1,):
        return False
    lp = _local(display, pos)
    for name, rect in m3d.ui_rects.items():
        if not rect.collidepoint(lp):
            continue
        if name == 'view_toggle':
            m3d.view_mode = 'operator' if m3d.view_mode == 'orbit' else 'orbit'
            m3d.look_az = None            # re-aim at the mount on entry
            return True
        if name == 'follow_toggle':
            m3d.follow_live = not m3d.follow_live
            return True
        if name.startswith('toggle_'):
            # Object-type visibility flags shared with the skyplot views.
            attr = name[len('toggle_'):]
            from tracking_visuals import OBJECT_TOGGLES
            default = next((d for a, _n, d in OBJECT_TOGGLES if a == attr), True)
            setattr(config_state, attr,
                    not bool(getattr(config_state, attr, default)))
            return True
        if name in _SEAT_STEPS:
            attr, step, default = _SEAT_STEPS[name]
            cur = _config_floats(config_state)(attr, default)
            if attr == 'mount3d_observer_bearing_deg':
                val = (cur + step) % 360.0
            elif attr == 'mount3d_observer_distance_m':
                val = min(15.0, max(0.5, cur + step))
            else:
                val = min(2.5, max(0.2, cur + step))
            setattr(config_state, attr, round(val, 3))
            m3d.look_az = None            # seat moved: re-aim operator view
            return True
        if name in ('slider_azm', 'slider_alt'):
            m3d.dragging_slider = name
            _slider_value(m3d, name, lp[0])
            return True
    m3d.dragging_view = True
    return True


def handle_mount3d_mouse_motion(pos, rel, buttons, display, m3d):
    if m3d.dragging_slider and buttons[0]:
        lp = _local(display, pos)
        _slider_value(m3d, m3d.dragging_slider, lp[0])
        return
    if m3d.dragging_view and buttons[0]:
        if m3d.view_mode == 'operator':
            if m3d.look_az is not None:
                m3d.look_az = (m3d.look_az + rel[0] * 0.15) % 360.0
                m3d.look_el = min(89.0, max(-89.0, m3d.look_el - rel[1] * 0.15))
        else:
            m3d.view_az = (m3d.view_az - rel[0] * 0.4) % 360.0
            m3d.view_el = min(89.0, max(-89.0, m3d.view_el + rel[1] * 0.4))


def handle_mount3d_mouse_up(pos, m3d):
    m3d.dragging_view = False
    m3d.dragging_slider = None


def handle_mount3d_wheel(dy, m3d):
    if m3d.view_mode == 'operator':
        m3d.eye_zoom = min(3.0, max(0.5, m3d.eye_zoom * (1.15 ** dy)))
    else:
        m3d.zoom = min(4.0, max(0.3, m3d.zoom * (1.15 ** dy)))
