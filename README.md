# Hat Creek Skytracker

An optical tracker for select telescopes — a single-operator application for
finding, acquiring, tracking, and collecting imagery of space and airborne
objects (satellites, rockets, aircraft) from stationary or mobile platforms.

It pairs a Celestron NexStar mount with one or two cameras (a wide "finder/guide"
camera and a narrow long-focal-length camera) and provides a fast, game-like UI
with selectable tracking modes, real-time closed-loop control, and labeled data
capture.

![Main menu](doc/screenshots/main_menu.png)

## Current Capabilities

**Tracking modes** (cycle through them from the Joystick Loop):
- **STANDBY** — poll/display mount position only.
- **RATE_CONTROL** — manual joystick slewing with hardware safety limits,
  through an **adaptive speed gearbox**: full stick deflection is capped at a
  gentle base rate (~0.5°/s) so a slammed stick can't fling the target out of
  frame; holding the stick pinned deliberately winds the ceiling up toward the
  mount's full 10°/s, and backing off, releasing, or reversing direction winds
  it back down. A segmented **Slew speed** indicator beside the stick display
  shows the current gear. Kid-tested design goal: you have to *earn* the fast
  rates, and letting go always lands you back in gentle mode.
- **PROGRAM** — automatically follow a TLE satellite pass, an imported launch
  trajectory, a tracked ADS-B aircraft, **or any celestial object** — sun,
  moon, planets (Pluto included, because every kid asks), the 100 brightest
  named stars, and the Messier/NGC deep-sky catalogues — using interpolated
  angular position + rates. Celestial targets ride a sliding 90-minute
  Skyfield-apparent trajectory window that refreshes automatically, so the
  mount slews to the object and then follows it across the sky indefinitely.
  Selecting the sun posts a loud solar-filter warning.
- **HOTSPOT** — closed-loop *optical* tracker: detect the brightest ("hot")
  object in the camera frame and drive the mount to keep it centered. Intended
  as an operator hand-off once PROGRAM track has the object in frame (rockets,
  aircraft). Coasts briefly on a dropout, then falls back to PROGRAM.
- **HANDOFF / MTI** — reserved for future work (e.g. dim satellites amid
  streaking stars, which need a different detection approach than HOTSPOT).

**Real-time control architecture**
- A dedicated **mount control thread** runs the read → PID → command cycle at a
  fixed cadence, fully decoupled from the pygame render loop, so rendering jitter
  and blocking serial I/O can't stall mount commands.
- **Hardened serial layer**: every NexStar AUX transaction is locked and uses a
  short timeout; a missing/short response is reported and the cycle is skipped
  instead of freezing the loop (which previously let an axis run open-loop).
- **PID controller** with trajectory **feed-forward**, **derivative-on-measurement**
  (no derivative kick), and **conditional-integration anti-windup**.
- **One-click PID auto-tune** (`autotune.py`): while tracking in PROGRAM or
  HOTSPOT, the PID pane's **AUTOTUNE** button runs an online coordinate-descent
  optimizer over all six gains — no injected test signals, just the live
  tracking error. It probes each gain up/down in log space, keeps measured
  improvements, converges in a few minutes (the sliders visibly follow along),
  and pauses safely on STOP or a lost target. Stopping keeps the best gains
  found; works identically under the Python and Rust control loops.
- **Per-mode gain profiles**: PROGRAM (encoder loop) and HOTSPOT (optical loop)
  are different plants, so each keeps its own gain set in
  `config.pid_mode_profiles` — swapped into the live gains **automatically** on
  mode transitions (HANDOFF shares PROGRAM's). Each profile is stamped with the
  **target it was tuned on** and the date, shown in the PID pane, so you know
  whose tune you're flying.

**Imaging & visualization**
- Threaded camera capture (ZWO ASI) with a large circular buffer and
  microsecond-precision UTC timestamps; labeled `.png` + per-frame metadata capture.
- Real-time annotated polar sky plot, satellite pass tables, and camera FOV
  overlays. Render threads publish complete, double-buffered frames (no flicker).
- A KSP-style **navball** shows live mount attitude with the target's trajectory,
  a target crosshair, and ADS-B aircraft markers overlaid.
- **ADS-B aircraft tracking** (RTL-SDR / Nooelec): nearby aircraft appear on the
  skyplot and navball and can be selected, slewed to, and tracked like a satellite.

**Rendering performance.** The render threads invariant-cache their expensive
per-frame work so they don't starve the GIL the camera capture threads need:
the navball's hemisphere/grid is cached on quantized pointing, each processed
camera feed is cached on its capture sequence, and the skyplot's static
background (grid/labels/keepout), starfield, and selected-satellite arc are
cached layers rebuilt on a seconds-scale quantum instead of drawn per frame
(~14× cheaper). Camera frames are buffered into the capture ring **only while
a capture is armed** — idle memory stays flat instead of accumulating up to
1000 full-resolution frames per camera. See [`LEARNINGS.md`](LEARNINGS.md).

**Hardware simulator** (see below) — run the entire tracking loop without any
physical hardware present.

## Screens

**Tracking Vis** — annotated polar sky plot with the selected object's orbit,
orbital elements, camera FOV footprints, and a scrollable pass table.

The plot carries a full **celestial layer**: the sun, moon and planets are
always drawn (dimmed on the horizon rim when below it, so you can see where
they'll rise), the **100 brightest stars** get gold spike markers and their
IAU proper names (Sirius, Vega, Betelgeuse...), and the **Messier** and
**NGC** catalogues overlay as violet squares / teal circles (NGC additionally
magnitude-limited) to keep the noise manageable. A **Show objects** toggle
column on the left panel switches each object type on/off — satellites, their
labels, aircraft, stars, Messier, NGC — with the sun/moon/planets and named
stars deliberately always-on. Every object — plus every satellite and
aircraft — is click-selectable, and PROGRAM mode will slew to and track it.
Shown here with the moon selected (yellow ring):

![Tracking Vis](doc/screenshots/tracking_vis.png)

Both polar plots shade the **mount keepout** in light red: every sky direction
whose mount-axis solutions all fall outside the configured safety limits,
computed through the active mount mode's command transform (including the
over-the-zenith flip the tracking loop may use in AltAz) — so you can see at a
glance which targets and passes are flyable before picking one. Shown here
with an azimuth keepout wedge around north and a mount-ALT ceiling that
forbids the lowest elevations:

![Keepout overlay](doc/screenshots/keepout_overlay.png)

**Joystick Loop** — the operational screen: live camera feeds (with boresight
crosshairs), a polar-plot quadrant, an attitude navball, tracking-rate/error
strip charts, mount connection/position status, tracking mode, and PID
diagnostics. The PID pane's log sliders tune the gains live, and its
**AUTOTUNE** button hands them to the online auto-tuner while tracking. The
**Slew speed** gear bar next to the stick displays shows the adaptive
RATE_CONTROL ceiling (green = base range, orange = boost gears earned by
pinning the stick). The skyplot quadrant carries the same celestial layer and
click-selection as the full-screen plot, with **M** and **NGC** catalogue
toggles in the Targets strip. Shown here in simulation with the mount and
both cameras connected.

![Joystick Loop](doc/screenshots/joystick_loop.png)

**Sensor Calibration** — both camera feeds side by side with gain / exposure /
alignment-rotation / ROI controls for co-boresighting the guide and main cameras.

![Sensor Calibration](doc/screenshots/sensor_calib.png)

**Mount 3D** — a software-3D view of the mount for building alignment
intuition: the articulated single-fork-arm model (tripod → AZ-axis tube →
ALT-axis arm → side-mounted OTA, each rigidly attached like the real
hardware) poses from the live axis angles through the
**same forward transforms the tracker uses** (pinned by a parity test suite),
per mount mode — shown here in AltAz-Side, where the cyan AZM-axis arrow lies
on the horizon at the alignment azimuth. Both cameras' FOV cones sweep a real
star field with satellites, aircraft, the sun/moon/planets and the
Messier/NGC overlays, the selected trajectory, and the mount keepout tinted
on the sky dome — with the same object-type toggle buttons as the skyplot in
its HUD (the two views share one set of config flags). Two view cameras: free **orbit** (drag/wheel)
and an **operator view** rendered from your configured seat position (bearing
/ distance / eye height) so the perspective matches what you actually see.
Manual AZM/ALT sliders pose the model while disconnected. The orbit camera
dives the full ±89° — below the ground plane you look straight up through
the translucent ground at the mount and the whole sky dome. The star field
advances in real time with the tracking clock (sidereal rotation, pinned by
a regression test).

![Mount 3D](doc/screenshots/mount3d.png)

![Mount 3D from below](doc/screenshots/mount3d_below_horizon.png)

## Hardware Interfaces

- **Celestron NexStar AUX** (primary; implemented in `lib/auxstar.py`)
- **Meade LX200 classic** (legacy interface)

## Technologies

- **Python 3.11** (conda environment `track`)
- **pygame** — UI, input, and rendering
- **NumPy / SciPy** — control math, detection, geometry
- **OpenCV** — image handling
- **Skyfield / sgp4** — TLE propagation and astrometry
- **pandas** — pass/data tables
- **pyserial** — NexStar AUX serial protocol
- **zwoasi** — ZWO ASI camera SDK bindings
- **plotly** — post-processing/plots

## Hardware Simulator

To hone the user experience and develop the control/acquisition algorithms
without the bulky hardware, the app includes software simulators for the mount
and both cameras. Enable it from the **HW Sim** screen (off by default — when
off, real-hardware behavior is unchanged):

![Hardware Sim screen](doc/screenshots/hw_sim.png)

- **Sim mount** integrates the same rate commands the control loop issues into a
  true pointing, and reports an encoder position with injectable **initial
  misalignment** and **noise** — so offset calibration and the closed loop have
  something real to fight.
- **Sim cameras** render synthetic imagery of whatever is being tracked, given
  the mount's true pointing and the camera geometry, with injectable
  **inter-camera rotation + image-plane misalignment** (what co-boresight
  calibration is meant to solve). Frames flow through the *same* capture pipeline
  the real cameras use.
- A **launch** renders as a hot plume; a **live-TLE satellite** renders as a small
  dot in the wide cam and a rough **Starlink V2 Mini** in the narrow cam, over a
  **star field that streaks** with mount motion.

| Wide / guide cam (satellite dot among streaking stars) | Narrow cam (Starlink V2 Mini) |
| --- | --- |
| ![wide cam](doc/screenshots/sim_wide_satellite.png) | ![narrow cam](doc/screenshots/sim_narrow_v2mini.png) |

The simulator closes the loop entirely in software: sim mount pointing → sim
camera render → hot-spot detection → PID → rate command → sim mount moves. This
makes it possible to demonstrate a full hand-off-and-track sequence at a desk.

## Configuration

Site, optics, PID gains, safety limits, hot-spot, and simulator parameters live
in `config.json` and are editable from the **Config Options** and **HW Sim**
screens.

![Config options](doc/screenshots/config_options.png)

## Testing

The control/detection/simulation logic is covered by headless unit tests,
including a sim-in-the-loop test that asserts the hot-spot loop drives a
simulated object to frame center. The whole suite runs under pytest:

```
pip install -r requirements.txt
pytest                                 # ~250 tests, headless, < 1 minute
```

Suites with missing prerequisites are skipped automatically with a reason
(see `conftest.py`): the Rust parity tests need the `skytracker_core` wheel
(`maturin build rust/skytracker_core`), the star-catalog tests need
`hip_main.dat` (CDS I/239) in the repo root, and plate-solve tests need
`tetra3`. `de421.bsp` is auto-provisioned from the `skyfield-data` package.
Individual files still run standalone, e.g. `python test_simulator.py`.

## Project Goals & Roadmap

The longer-term vision is one set of cooperating, automation-friendly tools that
make running an optical pass fun and fast — so even a child could run it — rather
than alt-tabbing between many GUI programs.

- **Visibility & pass setup** — rich, annotated live sky plots; extract relevant
  facts for a known opportunity; overlay optical-system capability; ingest a PVT
  ephemeris + covariance to augment TLE; annotate TLE-vs-ephemeris differences.
- **Sensor-to-mount calibration** — guided two-sensor co-boresight against a
  guide star; solve sensor rotations and step calibrations into config.
- **Joystick loop** — manual track, program track, moving-object search,
  acquisition, tracked-object handoff, click-in-image selection, and labeled
  capture (UTC, mount az/el, rate, centroid per frame).
- **Post-processing** — quick-look and product generation to get a good shot
  posted fast. A PIPP-style prep stage (sharpness grading, "lucky" frame
  culling, target centring/cropping) feeds an AutoStakkert-style stacker that
  aligns and averages the best frames — optionally with alignment-point local
  warping — into one high-SNR master ready for wavelet sharpening.

We deliberately don't try to re-create everything existing tools (Heavens-Above,
SkyTrack, KStars, etc.) already do well — the focus is the niche capabilities
that go beyond them.

## License

See [LICENSE](LICENSE).
