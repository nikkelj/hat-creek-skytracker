//! Mount 3D: a software-3D instrument view of the telescope mount, drawn
//! with the egui `Painter` (no GPU mesh, no extra crates). Port of the
//! pygame `mount3d.py` screen, redrawn in the app's visual language.
//!
//! What it shows: a ground disc with compass rose and azimuth ticks, the
//! pier + tripod, the azimuth (axis-1) assembly turned by the mount's AZM
//! readout, the fork arm and the tube tilted by ALT, the amber boresight
//! ray out to a 50 m sky dome, the camera FOV cone, the target direction
//! (green when tracking) with a reticle on the dome, soft-limit arcs around
//! each axis, and a small HUD (mode, AZM/ALT, sky az/el, target separation).
//!
//! World frame: ENU right-handed, X = East, Y = North, Z = Up, 1 unit = 1 m.
//!
//! Kinematics mirror `mount3d.py` exactly: every mount mode is the same
//! two-axis chain (rotate about a pole by an hour-angle-like H, then about
//! the instantaneous dec axis by d) with a per-mode pole and (H, d) map:
//!
//! | mode        | pole              | H        | d        |
//! |-------------|-------------------|----------|----------|
//! | AltAz       | zenith            | AZM      | 90 - ALT |
//! | Passthrough | zenith            | AZM      | ALT      |
//! | AltAzSide   | (north, horizon)  | AZM + 90 | 90 - ALT |
//! | Eq          | (north, lat)      | AZM      | ALT      |
//!
//! `MountPose::az_deg` / `el_deg` are therefore the mount AXIS readouts
//! (AZM, ALT -- what the mount worker publishes), not sky az/el; the sky
//! direction is derived through the chain and shown in the HUD.
//!
//! Interaction: drag = orbit (yaw/pitch around the mount head), wheel =
//! zoom, double-click = reset view.

use egui::{pos2, Align2, Color32, PointerButton, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, Vec2};

use crate::theme;

// ---------------------------------------------------------------- constants

const R_SKY: f64 = 50.0; // sky-dome radius (m), centred on the mount head
const HEAD_HEIGHT: f64 = 1.25; // axis intersection above ground (m)
const BASE_DIST: f64 = 6.0; // orbit distance at zoom 1 (m)
const GROUND_R: f64 = 3.0; // ground disc radius (m)
const NEAR: f64 = 0.05; // camera-space near plane (m)

const AZ_TUBE_R: f64 = 0.10;
const AZ_TUBE_LEN: f64 = 0.40;
const ALT_TUBE_R: f64 = 0.085;
const ALT_TUBE_LEN: f64 = 0.32;
const OTA_R: f64 = 0.14;
const OTA_LEN: f64 = 1.00;
const OTA_ARM_OFFSET: f64 = ALT_TUBE_LEN + OTA_R * 0.85;

const ALTAZ_SIDE_H0_DEG: f64 = 90.0; // transformations.ALTAZ_SIDE_H0_DEG

// ------------------------------------------------------------------ vec/mat

#[derive(Clone, Copy, Debug, PartialEq)]
struct V3 {
    x: f64,
    y: f64,
    z: f64,
}

const fn v3(x: f64, y: f64, z: f64) -> V3 {
    V3 { x, y, z }
}

const UP: V3 = v3(0.0, 0.0, 1.0);
const NORTH: V3 = v3(0.0, 1.0, 0.0);
const EAST: V3 = v3(1.0, 0.0, 0.0);
const HEAD: V3 = v3(0.0, 0.0, HEAD_HEIGHT);

impl V3 {
    fn dot(self, o: V3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(self, o: V3) -> V3 {
        v3(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
    /// Unit vector; returns `self` unchanged when degenerate.
    fn normalized(self) -> V3 {
        let n = self.norm();
        if n < 1e-12 {
            self
        } else {
            self * (1.0 / n)
        }
    }
}

impl std::ops::Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl std::ops::Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl std::ops::Mul<f64> for V3 {
    type Output = V3;
    fn mul(self, s: f64) -> V3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}
impl std::ops::Neg for V3 {
    type Output = V3;
    fn neg(self) -> V3 {
        v3(-self.x, -self.y, -self.z)
    }
}

/// Row-major 3x3 matrix.
#[derive(Clone, Copy, Debug)]
struct M3([[f64; 3]; 3]);

impl M3 {
    fn identity() -> M3 {
        M3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }
    /// Rodrigues rotation about a unit axis (right-hand rule), degrees.
    fn rot_about(axis: V3, deg: f64) -> M3 {
        let (x, y, z) = (axis.x, axis.y, axis.z);
        let (s, c) = deg.to_radians().sin_cos();
        let cc = 1.0 - c;
        M3([
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ])
    }
    fn mul_v(&self, v: V3) -> V3 {
        let m = &self.0;
        v3(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        )
    }
    fn mul_m(&self, o: &M3) -> M3 {
        let mut r = [[0.0; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| self.0[i][k] * o.0[k][j]).sum();
            }
        }
        M3(r)
    }
    fn transpose(&self) -> M3 {
        let m = &self.0;
        M3([[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]])
    }
}

/// Sky direction (az clockwise from north, el above horizon) as an ENU unit
/// vector. Valid for any el (past-zenith folds through the pole).
fn unit_from_azel(az_deg: f64, el_deg: f64) -> V3 {
    let (sa, ca) = az_deg.to_radians().sin_cos();
    let (se, ce) = el_deg.to_radians().sin_cos();
    v3(ce * sa, ce * ca, se)
}

/// Inverse of `unit_from_azel`: (az in [0, 360), el in [-90, 90]).
fn azel_from_unit(v: V3) -> (f64, f64) {
    let az = v.x.atan2(v.y).to_degrees().rem_euclid(360.0);
    let el = v.z.clamp(-1.0, 1.0).asin().to_degrees();
    (az, el)
}

/// Two unit vectors perpendicular to `axis` (and to each other).
fn ortho_pair(axis: V3) -> (V3, V3) {
    let r = if axis.dot(UP).abs() < 0.9 { UP } else { NORTH };
    let u = axis.cross(r).normalized();
    (u, axis.cross(u))
}

// -------------------------------------------------------------- kinematics

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModeKind {
    AltAz,
    AltAzSide,
    Passthrough,
    Eq,
}

impl ModeKind {
    fn parse(s: &str) -> ModeKind {
        let k: String = s.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
        match k.as_str() {
            "altazside" => ModeKind::AltAzSide,
            "passthrough" => ModeKind::Passthrough,
            "eq" | "equatorial" => ModeKind::Eq,
            _ => ModeKind::AltAz,
        }
    }
    fn label(self) -> &'static str {
        match self {
            ModeKind::AltAz => "ALT-AZ",
            ModeKind::AltAzSide => "ALT-AZ SIDE",
            ModeKind::Passthrough => "PASSTHROUGH",
            ModeKind::Eq => "EQUATORIAL",
        }
    }
}

/// (pole_az, pole_alt, H, d) of the generic chain for one mode; mirrors
/// `mount3d._eq_params` with alignment az/el = 0 (Eq pole altitude = site
/// latitude, the tracker's fallback) and no side flip.
fn eq_params(mode: ModeKind, lat_deg: f64, azm: f64, alt: f64) -> (f64, f64, f64, f64) {
    match mode {
        ModeKind::AltAz => (0.0, 90.0, azm, 90.0 - alt),
        ModeKind::Passthrough => (0.0, 90.0, azm, alt),
        ModeKind::AltAzSide => (0.0, 0.0, azm + ALTAZ_SIDE_H0_DEG, 90.0 - alt),
        ModeKind::Eq => (0.0, lat_deg, azm, alt),
    }
}

/// ENU equatorial basis (p, x, y): p the pole, x the upper-meridian equator
/// direction, y = x cross p (the ENU mirror of transformations._eq_basis).
fn eq_basis(pole_az: f64, pole_alt: f64) -> (V3, V3, V3) {
    let p = unit_from_azel(pole_az, pole_alt);
    let mut x = UP - p * UP.dot(p);
    if x.norm() < 1e-9 {
        x = NORTH - p * NORTH.dot(p);
    }
    let x = x.normalized();
    let y = x.cross(p);
    (p, x, y)
}

/// Orientation M(H, d) of the chain (boresight = M x) and the posed axis 2.
fn chain_frame(p: V3, x: V3, h_deg: f64, d_deg: f64) -> (M3, V3) {
    let o1 = M3::rot_about(p, -h_deg);
    let e = o1.mul_v(x);
    let a2 = e.cross(p);
    let o2 = M3::rot_about(a2, d_deg);
    (o2.mul_m(&o1), a2)
}

/// Articulated pose of the mount for one (AZM, ALT).
#[derive(Clone, Copy, Debug)]
struct Pose {
    p: V3,          // axis 1 (polar / azimuth) direction
    axis2: V3,      // axis 2 (dec / alt) direction at this pose
    boresight: V3,  // boresight unit vector
    r_ax1: M3,      // poses axis-1-stage parts from their home sculpt
    r_ax2: M3,      // poses axis-2-stage parts from their home sculpt
    home_bore: V3,  // boresight at AZM = ALT = 0
    home_axis2: V3, // axis 2 at AZM = ALT = 0
}

fn mount_pose(mode: ModeKind, lat_deg: f64, azm: f64, alt: f64) -> Pose {
    let (pole_az, pole_alt, h, d) = eq_params(mode, lat_deg, azm, alt);
    let (_, _, h0, d0) = eq_params(mode, lat_deg, 0.0, 0.0);
    let (p, x, _y) = eq_basis(pole_az, pole_alt);
    let (m, a2) = chain_frame(p, x, h, d);
    let (m0, a2_0) = chain_frame(p, x, h0, d0);
    Pose {
        p,
        axis2: a2,
        boresight: m.mul_v(x),
        r_ax1: M3::rot_about(p, -(h - h0)),
        r_ax2: m.mul_m(&m0.transpose()),
        home_bore: m0.mul_v(x),
        home_axis2: a2_0,
    }
}

/// Unit directions of the 4 FOV corners (pinhole), built on the tube's
/// (axis2, up) frame. Order: (-w,-h), (+w,-h), (+w,+h), (-w,+h).
fn fov_corner_dirs(boresight: V3, axis2: V3, fov_w_deg: f64, fov_h_deg: f64) -> [V3; 4] {
    let b = boresight.normalized();
    let mut right = axis2 - b * axis2.dot(b);
    if right.norm() < 1e-9 {
        right = ortho_pair(b).0;
    }
    let right = right.normalized();
    let up_t = b.cross(right);
    let tw = (fov_w_deg.to_radians() / 2.0).tan();
    let th = (fov_h_deg.to_radians() / 2.0).tan();
    let c = |sx: f64, sy: f64| (b + right * (sx * tw) + up_t * (sy * th)).normalized();
    [c(-1.0, -1.0), c(1.0, -1.0), c(1.0, 1.0), c(-1.0, 1.0)]
}

// ---------------------------------------------------------------- geometry

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Base,
    Ax1,
    Ax2,
}

struct Part {
    verts: Vec<V3>,
    faces: Vec<Vec<usize>>,
    fill: Color32,
    edge: Color32,
    stage: Stage,
}

fn part_box(center: V3, hx: V3, hy: V3, hz: V3, fill: Color32, edge: Color32, stage: Stage) -> Part {
    let mut verts = Vec::with_capacity(8);
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                verts.push(center + hx * sx + hy * sy + hz * sz);
            }
        }
    }
    let faces = vec![
        vec![0, 1, 3, 2],
        vec![4, 6, 7, 5],
        vec![0, 4, 5, 1],
        vec![2, 3, 7, 6],
        vec![0, 2, 6, 4],
        vec![1, 5, 7, 3],
    ];
    Part { verts, faces, fill, edge, stage }
}

#[allow(clippy::too_many_arguments)]
fn part_tube(base: V3, axis: V3, radius: f64, length: f64, nsides: usize, fill: Color32, edge: Color32, stage: Stage) -> Part {
    let (u, v) = ortho_pair(axis);
    let mut verts = Vec::with_capacity(2 * nsides);
    for i in 0..nsides {
        let a = std::f64::consts::TAU * i as f64 / nsides as f64;
        verts.push(base + u * (radius * a.cos()) + v * (radius * a.sin()));
    }
    for i in 0..nsides {
        verts.push(verts[i] + axis * length);
    }
    let mut faces = Vec::with_capacity(nsides + 2);
    for i in 0..nsides {
        let j = (i + 1) % nsides;
        faces.push(vec![i, j, nsides + j, nsides + i]);
    }
    faces.push((0..nsides).collect());
    faces.push((nsides..2 * nsides).rev().collect());
    Part { verts, faces, fill, edge, stage }
}

fn part_arrow(base: V3, axis: V3, length: f64, color: Color32, stage: Stage) -> Vec<Part> {
    let (u, v) = ortho_pair(axis);
    let s = 0.02;
    let shaft = part_box(base + axis * (length * 0.41), u * s, v * s, axis * (length * 0.41), color, color, stage);
    let tip_base = base + axis * (length * 0.82);
    let hs = 0.055;
    let mut verts: Vec<V3> = [0.0, 0.5, 1.0, 1.5]
        .iter()
        .map(|k| tip_base + u * (hs * (k * std::f64::consts::PI).cos()) + v * (hs * (k * std::f64::consts::PI).sin()))
        .collect();
    verts.push(base + axis * length);
    let faces = vec![vec![0, 1, 4], vec![1, 2, 4], vec![2, 3, 4], vec![3, 0, 4], vec![3, 2, 1, 0]];
    vec![shaft, Part { verts, faces, fill: color, edge: color, stage }]
}

/// Model parts in WORLD coordinates at the mount-home pose (AZM = ALT = 0).
/// Base parts are fixed; Ax1 parts turn with axis 1; Ax2 with the chain.
fn build_mount_geometry(home: &Pose) -> Vec<Part> {
    let p = home.p;
    let b0 = home.home_bore;
    let a2 = home.home_axis2;
    let mut saddle_up = a2.cross(b0);
    if saddle_up.norm() < 1e-6 {
        saddle_up = UP;
    }
    let saddle_up = saddle_up.normalized();

    let pier_fill = Color32::from_rgb(38, 43, 55);
    let pier_edge = theme::with_alpha(theme::TEXT_2, 70);
    let leg_fill = Color32::from_rgb(32, 36, 47);
    let tube_fill = Color32::from_rgb(60, 68, 86);
    let tube_edge = theme::with_alpha(theme::TEXT_2, 90);
    let ota_fill = Color32::from_rgb(88, 92, 106);
    let ota_edge = theme::with_alpha(theme::AMBER, 120);
    let stub_fill = Color32::from_rgb(72, 76, 90);

    let mut parts = Vec::new();
    // BASE: pier + three tripod legs.
    let pier_h = HEAD_HEIGHT - 0.18;
    parts.push(part_box(v3(0.0, 0.0, pier_h / 2.0), EAST * 0.06, NORTH * 0.06, UP * (pier_h / 2.0), pier_fill, pier_edge, Stage::Base));
    for leg_az in [90.0, 210.0, 330.0] {
        let foot = unit_from_azel(leg_az, 0.0) * 0.6;
        let top = v3(0.0, 0.0, pier_h * 0.85);
        let mid = (foot + top) * 0.5;
        let leg = top - foot;
        let len = leg.norm();
        let dir = leg * (1.0 / len);
        let (lu, lv) = ortho_pair(dir);
        parts.push(part_box(mid, lu * 0.025, lv * 0.025, dir * (len / 2.0), leg_fill, pier_edge, Stage::Base));
    }
    // AX1: azimuth tube (centre on the pier top), axis-1 arrow, fork arm.
    let az_base = HEAD - p * (0.18 + AZ_TUBE_LEN / 2.0);
    parts.push(part_tube(az_base, p, AZ_TUBE_R, AZ_TUBE_LEN, 8, tube_fill, tube_edge, Stage::Ax1));
    parts.extend(part_arrow(HEAD + p * 0.06, p, 0.62, theme::ACCENT, Stage::Ax1));
    parts.push(part_tube(HEAD, a2, ALT_TUBE_R, ALT_TUBE_LEN, 8, tube_fill, tube_edge, Stage::Ax1));
    // AX2: the OTA, side-mounted on the arm end; guide stub; axis-2 arrow.
    let ota_center = HEAD + a2 * OTA_ARM_OFFSET;
    let ota_base = ota_center - b0 * (OTA_LEN * 0.45);
    parts.push(part_tube(ota_base, b0, OTA_R, OTA_LEN, 10, ota_fill, ota_edge, Stage::Ax2));
    parts.push(part_tube(
        ota_base + saddle_up * (OTA_R + 0.032) + b0 * (OTA_LEN * 0.3),
        b0,
        0.032,
        0.42,
        8,
        stub_fill,
        tube_edge,
        Stage::Ax2,
    ));
    parts.extend(part_arrow(HEAD + a2 * (OTA_ARM_OFFSET + OTA_R + 0.02), a2, 0.45, theme::VIOLET, Stage::Ax2));
    parts
}

fn pose_vertex(v: V3, stage: Stage, pose: &Pose) -> V3 {
    match stage {
        Stage::Base => v,
        Stage::Ax1 => pose.r_ax1.mul_v(v - HEAD) + HEAD,
        Stage::Ax2 => pose.r_ax2.mul_v(v - HEAD) + HEAD,
    }
}

// ------------------------------------------------------------------ camera

/// Orbit camera around the mount head. Camera space: x right, y up, z
/// forward (depth); screen y grows downward.
struct Camera {
    right: V3,
    up: V3,
    fwd: V3,
    pos: V3,
    focal: f64,
    cx: f64,
    cy: f64,
    /// Frustum half-slopes (|x| <= kx z, |y| <= ky z) with a pixel margin,
    /// so clipped geometry never projects to absurd coordinates.
    kx: f64,
    ky: f64,
}

/// Number of clip planes: near + 4 frustum sides.
const N_PLANES: usize = 5;

impl Camera {
    fn orbit(yaw_deg: f64, pitch_deg: f64, zoom: f64, rect: Rect) -> Camera {
        let cam_dir = unit_from_azel(yaw_deg, pitch_deg.clamp(-89.0, 89.0));
        let pos = HEAD + cam_dir * (BASE_DIST / zoom.max(1e-3));
        let fwd = -cam_dir;
        let mut right = fwd.cross(UP);
        if right.norm() < 1e-6 {
            right = EAST;
        }
        let right = right.normalized();
        let up = right.cross(fwd);
        let focal = (0.9 * rect.height() as f64).max(1.0);
        let margin = 60.0;
        Camera {
            right,
            up,
            fwd,
            pos,
            focal,
            cx: rect.min.x as f64 + rect.width() as f64 / 2.0,
            cy: rect.min.y as f64 + rect.height() as f64 * 0.52,
            kx: (rect.width() as f64 / 2.0 + margin) / focal,
            ky: (rect.height() as f64 * 0.52 + margin) / focal,
        }
    }

    /// Signed "inside" distance of a camera-space point to clip plane `i`
    /// (>= 0 means inside).
    fn plane(&self, c: V3, i: usize) -> f64 {
        match i {
            0 => c.z - NEAR,
            1 => self.kx * c.z - c.x,
            2 => self.kx * c.z + c.x,
            3 => self.ky * c.z - c.y,
            _ => self.ky * c.z + c.y,
        }
    }

    fn to_cam(&self, p: V3) -> V3 {
        let d = p - self.pos;
        v3(d.dot(self.right), d.dot(self.up), d.dot(self.fwd))
    }

    fn screen(&self, c: V3) -> Pos2 {
        let z = if c.z.abs() < 1e-9 { 1e-9 } else { c.z };
        let sx = (self.cx + self.focal * c.x / z).clamp(-20000.0, 20000.0);
        let sy = (self.cy - self.focal * c.y / z).clamp(-20000.0, 20000.0);
        pos2(sx as f32, sy as f32)
    }

    /// Camera-space depth of a world point (positive = in front).
    fn depth(&self, p: V3) -> f64 {
        self.to_cam(p).z
    }

    /// Screen position of a world point, `None` when at/behind the near plane.
    fn project(&self, p: V3) -> Option<Pos2> {
        let c = self.to_cam(p);
        (c.z > NEAR).then(|| self.screen(c))
    }

    /// World segment -> screen segment, clipped against the near plane and
    /// the (margin-padded) view frustum. `None` when nothing is visible.
    fn segment(&self, a: V3, b: V3) -> Option<[Pos2; 2]> {
        let (ca, cb) = (self.to_cam(a), self.to_cam(b));
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        for i in 0..N_PLANES {
            let (fa, fb) = (self.plane(ca, i), self.plane(cb, i));
            if fa < 0.0 && fb < 0.0 {
                return None;
            }
            if fa < 0.0 {
                t0 = t0.max(fa / (fa - fb));
            } else if fb < 0.0 {
                t1 = t1.min(fa / (fa - fb));
            }
            if t0 > t1 {
                return None;
            }
        }
        let d = cb - ca;
        Some([self.screen(ca + d * t0), self.screen(ca + d * t1)])
    }

    /// World polygon -> screen polygon, Sutherland-Hodgman clipped against
    /// the near plane and the view frustum (may return fewer than 3 points
    /// when nothing is visible).
    fn polygon(&self, pts: &[V3]) -> Vec<Pos2> {
        let mut poly: Vec<V3> = pts.iter().map(|&p| self.to_cam(p)).collect();
        for i in 0..N_PLANES {
            if poly.is_empty() {
                break;
            }
            let mut out: Vec<V3> = Vec::with_capacity(poly.len() + 2);
            for k in 0..poly.len() {
                let a = poly[k];
                let b = poly[(k + 1) % poly.len()];
                let (fa, fb) = (self.plane(a, i), self.plane(b, i));
                if fa >= 0.0 {
                    out.push(a);
                }
                if (fa >= 0.0) != (fb >= 0.0) {
                    out.push(a + (b - a) * (fa / (fa - fb)));
                }
            }
            poly = out;
        }
        poly.into_iter().map(|c| self.screen(c)).collect()
    }
}

/// First intersection of the ray (origin, dir) with the sky dome.
fn dome_hit(origin: V3, dir: V3) -> V3 {
    let oc = origin - HEAD;
    let bq = oc.dot(dir);
    let disc = (bq * bq - oc.dot(oc) + R_SKY * R_SKY).max(0.0);
    origin + dir * (-bq + disc.sqrt())
}

fn sky_point(az: f64, el: f64) -> V3 {
    HEAD + unit_from_azel(az, el) * R_SKY
}

// -------------------------------------------------------------- public API

/// Live inputs for one frame of the Mount 3D view.
///
/// `az_deg` / `el_deg` are the mount AXIS angles (AZM, ALT) as published by
/// the mount worker, in the mount's own convention (AltAz: ALT 0 = zenith,
/// el = 90 - ALT). `target` is a sky (az, el) direction; `fov_deg` is the
/// camera's horizontal FOV (a 3:2 frame is drawn). Limits are in mount axis
/// degrees: `(min, max)` allowed spans for AZM and ALT.
/// A sky object painted on the dome.
#[derive(Clone, Debug)]
pub struct SkyMark {
    pub az: f64,
    pub el: f64,
    pub kind: SkyKind,
}

#[derive(Clone, Debug)]
pub enum SkyKind {
    Star { mag: f32 },
    Body { name: String },
    Sat { selected: bool, geo: bool, name: String },
}

pub struct MountPose<'a> {
    /// Stars / planets / satellites to paint on the dome (may be empty).
    pub sky: &'a [SkyMark],
    pub az_deg: f64,
    pub el_deg: f64,
    /// "AltAz" | "AltAzSide" | "Passthrough" | "Eq" (case/punctuation-insensitive).
    pub mount_mode: &'a str,
    pub lat_deg: f64,
    /// Sky (az, el) of the selected/tracked target.
    pub target: Option<(f64, f64)>,
    pub tracking: bool,
    pub fov_deg: f64,
    pub az_limits: Option<(f64, f64)>,
    pub el_limits: Option<(f64, f64)>,
}

/// Persistent orbit-view state (drag = orbit, wheel = zoom).
pub struct Mount3dView {
    /// Camera bearing from the mount head, degrees clockwise from north.
    pub yaw_deg: f32,
    /// Camera elevation above the ground plane, degrees (-89..89).
    pub pitch_deg: f32,
    /// Orbit distance divisor (1 = 6 m from the head), clamped to 0.3..4.
    pub zoom: f32,
    geom_key: Option<(ModeKind, u64)>,
    geom: Vec<Part>,
}

impl Default for Mount3dView {
    fn default() -> Self {
        Mount3dView { yaw_deg: 150.0, pitch_deg: 18.0, zoom: 1.0, geom_key: None, geom: Vec::new() }
    }
}

/// One sorted face of the painter's-algorithm pass.
struct FaceDraw {
    depth: f64,
    pts: Vec<Pos2>,
    fill: Color32,
    edge: Color32,
}

impl Mount3dView {
    fn geometry(&mut self, mode: ModeKind, lat_deg: f64) -> &[Part] {
        let key = (mode, lat_deg.to_bits());
        if self.geom_key != Some(key) {
            self.geom = build_mount_geometry(&mount_pose(mode, lat_deg, 0.0, 0.0));
            self.geom_key = Some(key);
        }
        &self.geom
    }

    /// Draw the view into `size` points of the current `ui` and handle the
    /// orbit/zoom interaction. Returns the canvas response.
    pub fn ui(&mut self, ui: &mut Ui, size: Vec2, pose: &MountPose) -> Response {
        let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
        let rect = response.rect;
        let painter = painter.with_clip_rect(rect);

        // ---- interaction -------------------------------------------------
        if response.dragged_by(PointerButton::Primary) {
            let d = response.drag_delta();
            // Drag right -> the scene turns with the hand (yaw increases).
            self.yaw_deg = (self.yaw_deg + d.x * 0.4).rem_euclid(360.0);
            self.pitch_deg = (self.pitch_deg + d.y * 0.4).clamp(-89.0, 89.0);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                self.zoom = (self.zoom * 1.15f32.powf(scroll / 50.0)).clamp(0.3, 4.0);
            }
        }
        if response.double_clicked() {
            let d = Mount3dView::default();
            self.yaw_deg = d.yaw_deg;
            self.pitch_deg = d.pitch_deg;
            self.zoom = d.zoom;
        }
        let cursor = if response.dragged() { egui::CursorIcon::Grabbing } else { egui::CursorIcon::Grab };
        let response = response.on_hover_cursor(cursor);

        // ---- scene inputs ------------------------------------------------
        let mode = ModeKind::parse(pose.mount_mode);
        let cam = Camera::orbit(self.yaw_deg as f64, self.pitch_deg as f64, self.zoom as f64, rect);
        let mp = mount_pose(mode, pose.lat_deg, pose.az_deg, pose.el_deg);
        let ota_origin = HEAD + mp.axis2 * OTA_ARM_OFFSET;
        let (sky_az, sky_el) = azel_from_unit(mp.boresight);

        // ---- ground ------------------------------------------------------
        painter.rect_filled(rect, 4.0, theme::BG);
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, theme::HAIRLINE));

        let hair = |a: u8| Stroke::new(1.0, theme::with_alpha(theme::HAIRLINE, a));
        let ground = |az: f64, r: f64| unit_from_azel(az, 0.0) * r;
        let n_ring = 72;
        let ring_pts: Vec<V3> = (0..n_ring).map(|i| ground(360.0 * i as f64 / n_ring as f64, GROUND_R)).collect();
        let disc = cam.polygon(&ring_pts);
        if disc.len() >= 3 {
            painter.add(Shape::convex_polygon(disc, theme::with_alpha(theme::RAISED, 110), Stroke::new(1.0, theme::with_alpha(theme::TEXT_2, 80))));
        }
        for r in [1.0, 2.0] {
            let pts: Vec<V3> = (0..n_ring).map(|i| ground(360.0 * i as f64 / n_ring as f64, r)).collect();
            for i in 0..n_ring {
                if let Some(seg) = cam.segment(pts[i], pts[(i + 1) % n_ring]) {
                    painter.line_segment(seg, hair(200));
                }
            }
        }
        for az in (0..360).step_by(30) {
            let a = az as f64;
            let cardinal = az % 90 == 0;
            if let Some(seg) = cam.segment(ground(a, 0.35), ground(a, GROUND_R)) {
                painter.line_segment(seg, hair(if cardinal { 255 } else { 140 }));
            }
            let tick_out = if cardinal { GROUND_R + 0.28 } else { GROUND_R + 0.16 };
            if let Some(seg) = cam.segment(ground(a, GROUND_R), ground(a, tick_out)) {
                painter.line_segment(seg, Stroke::new(1.0, if cardinal { theme::TEXT_2 } else { theme::DIM }));
            }
            if cardinal {
                let label = match az {
                    0 => "N",
                    90 => "E",
                    180 => "S",
                    _ => "W",
                };
                if let Some(p) = cam.project(ground(a, GROUND_R + 0.62)) {
                    let col = if az == 0 { theme::TEXT } else { theme::TEXT_2 };
                    painter.text(p, Align2::CENTER_CENTER, label, theme::mono(11.5), col);
                }
            }
        }

        // ---- sky dome: horizon + elevation rings ---------------------------
        let dome_ring = |el: f64, stroke: Stroke| {
            let pts: Vec<V3> = (0..n_ring).map(|i| sky_point(360.0 * i as f64 / n_ring as f64, el)).collect();
            for i in 0..n_ring {
                if let Some(seg) = cam.segment(pts[i], pts[(i + 1) % n_ring]) {
                    painter.line_segment(seg, stroke);
                }
            }
        };
        dome_ring(0.0, Stroke::new(1.0, theme::with_alpha(theme::TEXT_2, 120)));
        for el in [30.0, 60.0] {
            dome_ring(el, hair(230));
            // Label on the far side of the dome (straight ahead of the camera).
            if let Some(p) = cam.project(sky_point(self.yaw_deg as f64 + 180.0, el)) {
                painter.text(p + Vec2::new(0.0, -3.0), Align2::CENTER_BOTTOM, format!("{el:.0}°"), theme::mono(10.0), theme::DIM);
            }
        }

        // ---- sky objects on the dome ---------------------------------------
        for m in pose.sky {
            if m.el < -0.5 {
                continue;
            }
            let Some(p) = cam.project(sky_point(m.az, m.el)) else { continue };
            match &m.kind {
                SkyKind::Star { mag } => {
                    let r = (2.4 - mag * 0.4).clamp(0.7, 2.4);
                    let g = (205.0 - mag * 18.0).clamp(90.0, 220.0) as u8;
                    painter.circle_filled(p, r, Color32::from_rgb(g, g, (g as u16 + 12).min(255) as u8));
                }
                SkyKind::Body { name } => {
                    let (col, r) = match name.as_str() {
                        "sun" => (Color32::from_rgb(255, 225, 130), 5.5),
                        "moon" => (Color32::from_rgb(215, 218, 228), 4.5),
                        _ => (Color32::from_rgb(235, 205, 140), 3.0),
                    };
                    painter.circle_filled(p, r, col);
                    painter.text(p + Vec2::new(7.0, -5.0), Align2::LEFT_CENTER, name, theme::sans(10.5), theme::with_alpha(col, 220));
                }
                SkyKind::Sat { selected, geo, name } => {
                    let col = if *geo { theme::VIOLET } else { Color32::from_rgb(255, 236, 200) };
                    painter.circle_filled(p, if *selected { 3.0 } else { 2.0 }, if *selected { theme::ACCENT } else { theme::with_alpha(col, 200) });
                    if *selected {
                        painter.circle_stroke(p, 8.0, Stroke::new(1.4, theme::ACCENT));
                        painter.text(p + Vec2::new(11.0, 0.0), Align2::LEFT_CENTER, name, theme::sans(11.0), theme::TEXT);
                    }
                }
            }
        }

        // ---- soft-limit rings ----------------------------------------------
        // Axis 1: ring around the azimuth tube; forbidden AZM span in red.
        if let Some((lo, hi)) = pose.az_limits {
            let c1 = HEAD - mp.p * 0.42;
            let ring = |a: f64| c1 + M3::rot_about(mp.p, -a).mul_v(mp.home_axis2) * 0.5;
            draw_limit_ring(&painter, &cam, &ring, lo, hi);
        }
        // Axis 2: ring around the tube's swing at the current AZM.
        if let Some((lo, hi)) = pose.el_limits {
            let ring = |t: f64| ota_origin + mount_pose(mode, pose.lat_deg, pose.az_deg, t).boresight * 0.8;
            draw_limit_ring(&painter, &cam, &ring, lo, hi);
        }

        // ---- mount model (painter's algorithm) -----------------------------
        let light = v3(0.35, 0.5, 0.79).normalized();
        let mut faces: Vec<FaceDraw> = Vec::new();
        for part in self.geometry(mode, pose.lat_deg) {
            let verts: Vec<V3> = part.verts.iter().map(|&v| pose_vertex(v, part.stage, &mp)).collect();
            for face in &part.faces {
                if face.len() < 3 || face.iter().any(|&i| cam.depth(verts[i]) <= NEAR) {
                    continue;
                }
                let (v0, v1, v2) = (verts[face[0]], verts[face[1]], verts[face[2]]);
                let n = (v1 - v0).cross(v2 - v0);
                if n.norm() < 1e-12 {
                    continue;
                }
                let shade = 0.55 + 0.45 * n.normalized().dot(light).abs();
                let fill = theme::lerp(Color32::BLACK, part.fill, shade as f32);
                let depth = face.iter().map(|&i| cam.depth(verts[i])).sum::<f64>() / face.len() as f64;
                let pts: Vec<Pos2> = face.iter().map(|&i| cam.screen(cam.to_cam(verts[i]))).collect();
                faces.push(FaceDraw { depth, pts, fill, edge: part.edge });
            }
        }
        faces.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal));
        for f in faces {
            painter.add(Shape::convex_polygon(f.pts, f.fill, Stroke::new(0.8, f.edge)));
        }

        // ---- boresight ray + FOV cone ----------------------------------------
        let tip = ota_origin + mp.boresight * (OTA_LEN * 0.58);
        let bore_far = dome_hit(ota_origin, mp.boresight);
        if let Some(seg) = cam.segment(tip, bore_far) {
            painter.line_segment(seg, Stroke::new(3.0, theme::with_alpha(theme::AMBER, 28)));
            painter.line_segment(seg, Stroke::new(1.2, theme::with_alpha(theme::AMBER, 210)));
        }
        if let Some(p) = cam.project(bore_far) {
            painter.circle_stroke(p, 3.5, Stroke::new(1.2, theme::AMBER));
        }
        if pose.fov_deg > 0.0 {
            let apex = ota_origin + mp.boresight * (OTA_LEN * 0.56);
            let corners = fov_corner_dirs(mp.boresight, mp.axis2, pose.fov_deg, pose.fov_deg * 2.0 / 3.0);
            let far: Vec<V3> = corners.iter().map(|&c| apex + c * (R_SKY * 0.97)).collect();
            // Faint glass fill on the side faces, hairline edges, far frame.
            for i in 0..4 {
                let tri = cam.polygon(&[apex, far[i], far[(i + 1) % 4]]);
                if tri.len() >= 3 {
                    painter.add(Shape::convex_polygon(tri, theme::with_alpha(theme::AMBER, 6), Stroke::NONE));
                }
                if let Some(seg) = cam.segment(apex, far[i]) {
                    painter.line_segment(seg, Stroke::new(0.8, theme::with_alpha(theme::AMBER, 70)));
                }
            }
            let quad = cam.polygon(&far);
            if quad.len() >= 3 {
                painter.add(Shape::closed_line(quad, Stroke::new(1.0, theme::with_alpha(theme::AMBER, 150))));
            }
        }

        // ---- target direction -------------------------------------------------
        let mut sep_deg: Option<f64> = None;
        if let Some((taz, tel)) = pose.target {
            let tdir = unit_from_azel(taz, tel);
            let col = if pose.tracking { theme::GREEN } else { theme::TEXT_2 };
            let tfar = HEAD + tdir * R_SKY;
            if let Some(seg) = cam.segment(ota_origin, tfar) {
                painter.extend(Shape::dashed_line(&seg, Stroke::new(1.0, theme::with_alpha(col, 190)), 7.0, 5.0));
            }
            if let Some(p) = cam.project(tfar) {
                painter.circle_stroke(p, 7.0, Stroke::new(1.2, col));
                for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                    painter.line_segment([p + Vec2::new(dx * 4.0, dy * 4.0), p + Vec2::new(dx * 11.0, dy * 11.0)], Stroke::new(1.2, col));
                }
            }
            // Great-circle arc from boresight to target on the dome.
            let b = mp.boresight;
            let cosang = b.dot(tdir).clamp(-1.0, 1.0);
            let ang = cosang.acos();
            sep_deg = Some(ang.to_degrees());
            if ang > 1e-4 && ang < std::f64::consts::PI - 1e-4 {
                let n = 24;
                let sa = ang.sin();
                let pts: Vec<V3> = (0..=n)
                    .map(|i| {
                        let t = i as f64 / n as f64;
                        let d = b * (((1.0 - t) * ang).sin() / sa) + tdir * ((t * ang).sin() / sa);
                        HEAD + d * R_SKY
                    })
                    .collect();
                for w in pts.windows(2) {
                    if let Some(seg) = cam.segment(w[0], w[1]) {
                        painter.line_segment(seg, Stroke::new(1.0, theme::with_alpha(col, 90)));
                    }
                }
            }
        }

        // ---- HUD ---------------------------------------------------------------
        let pad = 10.0;
        let mut y = rect.min.y + pad;
        let x = rect.min.x + pad;
        // Backing plate reserved first (filled once the text extent is known).
        let plate = painter.add(Shape::Noop);
        let mut hud_rect = Rect::NOTHING;
        let mut line = |y: f32, text: String, font: egui::FontId, col: Color32| -> f32 {
            let r = painter.text(pos2(x, y), Align2::LEFT_TOP, text, font, col);
            hud_rect = hud_rect.union(r);
            r.height() + 3.0
        };
        let header = if pose.tracking { format!("MOUNT · {}  ·  TRACKING", mode.label()) } else { format!("MOUNT · {}", mode.label()) };
        y += line(y, header, theme::sans(10.5), if pose.tracking { theme::GREEN } else { theme::DIM });
        y += line(y, format!("AZM {:8.2}°   ALT {:8.2}°", pose.az_deg, pose.el_deg), theme::mono(12.5), theme::TEXT);
        y += line(y, format!("sky az {:7.2}°  el {:6.2}°", sky_az, sky_el), theme::mono(11.0), theme::TEXT_2);
        if let (Some((taz, tel)), Some(sep)) = (pose.target, sep_deg) {
            let col = if pose.tracking { theme::GREEN } else { theme::TEXT_2 };
            line(y, format!("target {:7.2}°  {:6.2}°   Δ {:.2}°", taz.rem_euclid(360.0), tel, sep), theme::mono(11.0), col);
        }
        if hud_rect.is_positive() {
            painter.set(plate, Shape::rect_filled(hud_rect.expand2(Vec2::new(7.0, 5.0)), 4.0, theme::with_alpha(theme::PANEL, 200)));
        }
        // Axis legend (bottom-left) and controls hint (bottom-right).
        let by = rect.max.y - pad;
        let lx = rect.min.x + pad;
        painter.circle_filled(pos2(lx + 4.0, by - 6.0), 3.0, theme::ACCENT);
        let r1 = painter.text(pos2(lx + 11.0, by), Align2::LEFT_BOTTOM, "axis 1", theme::sans(10.5), theme::DIM);
        painter.circle_filled(pos2(r1.max.x + 12.0, by - 6.0), 3.0, theme::VIOLET);
        let r2 = painter.text(pos2(r1.max.x + 19.0, by), Align2::LEFT_BOTTOM, "axis 2", theme::sans(10.5), theme::DIM);
        painter.text(
            pos2(r2.max.x + 16.0, by),
            Align2::LEFT_BOTTOM,
            format!("view {:.0}° / {:+.0}°  ×{:.2}", self.yaw_deg, self.pitch_deg, self.zoom),
            theme::mono(10.0),
            theme::DIM,
        );
        painter.text(pos2(rect.max.x - pad, by), Align2::RIGHT_BOTTOM, "drag · orbit    wheel · zoom    dbl-click · reset", theme::sans(10.5), theme::DIM);

        response
    }
}

/// A soft-limit ring around one axis: the full travel circle as a dim
/// hairline, the ALLOWED span `lo..hi` (axis degrees) as a brighter arc,
/// red end-stop ticks at the two limits and a short red overrun arc past
/// each. `ring(t)` gives the world point for axis angle `t`.
fn draw_limit_ring(painter: &egui::Painter, cam: &Camera, ring: &dyn Fn(f64) -> V3, lo: f64, hi: f64) {
    let n = 72;
    let full: Vec<V3> = (0..n).map(|i| ring(360.0 * i as f64 / n as f64)).collect();
    for i in 0..n {
        if let Some(seg) = cam.segment(full[i], full[(i + 1) % n]) {
            painter.line_segment(seg, Stroke::new(1.0, theme::with_alpha(theme::HAIRLINE, 230)));
        }
    }
    if hi <= lo {
        return;
    }
    let arc = |from: f64, span: f64, stroke: Stroke| {
        let steps = ((span / 4.0).ceil() as usize).max(1);
        let mut prev = ring(from);
        for i in 1..=steps {
            let p = ring(from + span * i as f64 / steps as f64);
            if let Some(seg) = cam.segment(prev, p) {
                painter.line_segment(seg, stroke);
            }
            prev = p;
        }
    };
    let span = (hi - lo).min(360.0);
    arc(lo, span, Stroke::new(1.3, theme::with_alpha(theme::TEXT_2, 150)));
    if span < 360.0 {
        let over = 12.0f64.min((360.0 - span) / 2.0);
        let red = Stroke::new(1.4, theme::with_alpha(theme::RED, 190));
        arc(hi, over, red);
        arc(lo - over, over, red);
        // End-stop ticks: short radial marks at the two limits.
        let centre = full.iter().fold(v3(0.0, 0.0, 0.0), |a, &b| a + b) * (1.0 / n as f64);
        for t in [lo, hi] {
            let p = ring(t);
            let out = centre + (p - centre) * 1.18;
            let inn = centre + (p - centre) * 0.86;
            if let Some(seg) = cam.segment(inn, out) {
                painter.line_segment(seg, Stroke::new(1.5, theme::RED));
            }
        }
    }
}

// ------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: V3, b: V3, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    #[test]
    fn enu_basis_and_round_trip() {
        assert!(close(unit_from_azel(0.0, 0.0), NORTH, 1e-12));
        assert!(close(unit_from_azel(90.0, 0.0), EAST, 1e-12));
        assert!(close(unit_from_azel(123.0, 90.0), UP, 1e-9));
        for az in [0.0, 45.0, 190.0, 359.0] {
            for el in [-45.0, 0.0, 30.0, 89.0] {
                let (a2, e2) = azel_from_unit(unit_from_azel(az, el));
                assert!((a2 - az).abs() < 1e-8 && (e2 - el).abs() < 1e-8, "{az} {el} -> {a2} {e2}");
            }
        }
    }

    #[test]
    fn rot_about_right_hand_rule() {
        // +90 about Up takes East -> North.
        assert!(close(M3::rot_about(UP, 90.0).mul_v(EAST), NORTH, 1e-12));
    }

    #[test]
    fn head_on_plus_z_projects_to_screen_centre_at_default_camera() {
        let v = Mount3dView::default();
        let rect = Rect::from_min_size(pos2(0.0, 0.0), Vec2::new(800.0, 600.0));
        let cam = Camera::orbit(v.yaw_deg as f64, v.pitch_deg as f64, v.zoom as f64, rect);
        let p = cam.project(HEAD).expect("head in front of camera");
        assert!((p.x - cam.cx as f32).abs() < 1e-3, "x {} vs {}", p.x, cam.cx);
        assert!((p.y - cam.cy as f32).abs() < 1e-3, "y {} vs {}", p.y, cam.cy);
        assert!((cam.cx - 400.0).abs() < 1e-9 && (cam.cy - 312.0).abs() < 1e-9);
        // A point behind the camera is culled by sign.
        let behind = cam.pos + unit_from_azel(v.yaw_deg as f64, v.pitch_deg as f64) * 2.0;
        assert!(cam.project(behind).is_none());
    }

    #[test]
    fn zoom_moves_orbit_camera_closer() {
        let rect = Rect::from_min_size(pos2(0.0, 0.0), Vec2::new(800.0, 600.0));
        let c1 = Camera::orbit(150.0, 18.0, 1.0, rect);
        let c2 = Camera::orbit(150.0, 18.0, 2.0, rect);
        assert!((c2.pos - HEAD).norm() < (c1.pos - HEAD).norm());
    }

    #[test]
    fn az_rotation_90_moves_tube_tip_from_north_to_east() {
        // Passthrough: (az, el) are sky angles directly.
        let tip = |mode, azm, alt| {
            let p = mount_pose(mode, 34.87, azm, alt);
            HEAD + p.axis2 * OTA_ARM_OFFSET + p.boresight * (OTA_LEN * 0.58)
        };
        let n = tip(ModeKind::Passthrough, 0.0, 0.0);
        let e = tip(ModeKind::Passthrough, 90.0, 0.0);
        assert!(n.y > 0.4 && n.y > n.x.abs(), "tip should point north: {n:?}");
        assert!(e.x > 0.4 && e.x > e.y.abs(), "tip should point east: {e:?}");
        assert!(close(mount_pose(ModeKind::Passthrough, 0.0, 0.0, 0.0).boresight, NORTH, 1e-12));
        assert!(close(mount_pose(ModeKind::Passthrough, 0.0, 90.0, 0.0).boresight, EAST, 1e-12));
        // AltAz: ALT 90 = horizon, AZM is sky azimuth.
        assert!(close(mount_pose(ModeKind::AltAz, 0.0, 0.0, 90.0).boresight, NORTH, 1e-12));
        assert!(close(mount_pose(ModeKind::AltAz, 0.0, 90.0, 90.0).boresight, EAST, 1e-12));
        assert!(close(mount_pose(ModeKind::AltAz, 0.0, 33.0, 0.0).boresight, UP, 1e-9));
    }

    #[test]
    fn forward_transform_parity() {
        // AltAz: az = AZM, el = 90 - ALT (transformations.AzAlt2AzEl_AltAz).
        for azm in [0.0, 30.0, 137.0, 270.0] {
            for alt in [-20.0, 0.0, 30.0, 60.0, 120.0] {
                let b = mount_pose(ModeKind::AltAz, 34.87, azm, alt).boresight;
                assert!(close(b, unit_from_azel(azm, 90.0 - alt), 1e-9), "AltAz {azm} {alt}");
                let b = mount_pose(ModeKind::Passthrough, 34.87, azm, alt).boresight;
                assert!(close(b, unit_from_azel(azm, alt), 1e-9), "Passthrough {azm} {alt}");
            }
        }
        // Eq: pole = (north, lat); at (0,0) the boresight is the upper-meridian
        // equator point (az 180, el 90 - lat); HA 90 west -> boresight west.
        let lat = 34.87;
        let p = mount_pose(ModeKind::Eq, lat, 0.0, 0.0);
        assert!(close(p.p, unit_from_azel(0.0, lat), 1e-12));
        assert!(close(p.boresight, unit_from_azel(180.0, 90.0 - lat), 1e-9));
        assert!(close(mount_pose(ModeKind::Eq, lat, 0.0, 90.0).boresight, p.p, 1e-9));
        // AltAzSide: axis 1 on the horizon at north; index home points along it.
        let s = mount_pose(ModeKind::AltAzSide, lat, 0.0, 0.0);
        assert!(s.p.z.abs() < 1e-12 && close(s.p, NORTH, 1e-12));
        assert!(s.boresight.dot(s.p) > 1.0 - 1e-9);
    }

    #[test]
    fn stage_rotations_track_the_boresight() {
        for mode in [ModeKind::AltAz, ModeKind::AltAzSide, ModeKind::Passthrough, ModeKind::Eq] {
            for (azm, alt) in [(0.0, 0.0), (50.0, 30.0), (200.0, -15.0), (123.0, 45.0)] {
                let p = mount_pose(mode, 34.87, azm, alt);
                let moved = p.r_ax2.mul_v(p.home_bore);
                assert!(moved.dot(p.boresight) > 1.0 - 1e-9, "{mode:?} {azm} {alt}");
                assert!(p.axis2.dot(p.p).abs() < 1e-9);
                for r in [p.r_ax1, p.r_ax2] {
                    let rrt = r.mul_m(&r.transpose());
                    let id = M3::identity();
                    for i in 0..3 {
                        for j in 0..3 {
                            assert!((rrt.0[i][j] - id.0[i][j]).abs() < 1e-9);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fov_corners_match_pinhole_geometry() {
        let b = unit_from_azel(40.0, 25.0);
        let a2 = unit_from_azel(130.0, 0.0);
        let (w, h) = (6.0f64, 4.0f64);
        let expect = ((w / 2.0).to_radians().tan().hypot((h / 2.0).to_radians().tan())).atan().to_degrees();
        for c in fov_corner_dirs(b, a2, w, h) {
            let ang = c.dot(b).clamp(-1.0, 1.0).acos().to_degrees();
            assert!((ang - expect).abs() < 1e-9);
        }
    }

    #[test]
    fn near_plane_clipping() {
        let rect = Rect::from_min_size(pos2(0.0, 0.0), Vec2::new(800.0, 600.0));
        let cam = Camera::orbit(150.0, 18.0, 1.0, rect);
        let behind = cam.pos - cam.fwd * 2.0;
        assert!(cam.segment(HEAD, behind).is_some());
        assert!(cam.segment(behind, behind - cam.fwd).is_none());
        let tri = cam.polygon(&[HEAD, HEAD + EAST, behind]);
        assert!(tri.len() >= 3);
    }

    #[test]
    fn mode_parsing_is_lenient() {
        assert_eq!(ModeKind::parse("AltAz-Side"), ModeKind::AltAzSide);
        assert_eq!(ModeKind::parse("altaz_side"), ModeKind::AltAzSide);
        assert_eq!(ModeKind::parse("Eq"), ModeKind::Eq);
        assert_eq!(ModeKind::parse("Passthrough"), ModeKind::Passthrough);
        assert_eq!(ModeKind::parse("whatever"), ModeKind::AltAz);
    }
}
