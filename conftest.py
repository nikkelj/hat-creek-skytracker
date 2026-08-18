"""Repo-wide pytest configuration.

Everything here exists so `pytest` passes on a fresh checkout / CI runner with
no display, no hardware, and no network:

  * headless SDL (the UI tests self-set this too, but collection order varies),
  * de421.bsp auto-provisioned from the `skyfield-data` package if missing
    (the JPL download host is blocked on many networks),
  * suites with unbuildable-here prerequisites (Rust wheel, star catalogues)
    are skipped with an actionable reason instead of erroring.
"""

import importlib.util
import os
import shutil

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _have_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _ensure_de421():
    """skyfield code loads 'de421.bsp' from the cwd; JPL's download host is
    blocked on many networks, but the skyfield-data PyPI package bundles the
    same kernel -- copy it into place when available."""
    dst = os.path.join(_ROOT, "de421.bsp")
    if os.path.exists(dst):
        return True
    if _have_module("skyfield_data"):
        import skyfield_data

        src = os.path.join(skyfield_data.get_skyfield_data_path(), "de421.bsp")
        if os.path.exists(src):
            shutil.copy(src, dst)
            return True
    return False


_HAVE_DE421 = _ensure_de421()
_HAVE_RUST = _have_module("skytracker_core")
_HAVE_HIPPARCOS = os.path.exists(os.path.join(_ROOT, "hip_main.dat"))

collect_ignore = []

if not _HAVE_RUST:
    collect_ignore += [
        "test_rust_pid_parity.py",
        "test_rust_hotspot_parity.py",
        "test_rust_mount_parity.py",
        "test_rust_transforms_parity.py",
        "test_rust_core_loop.py",
        "test_rust_core_loop_bridge.py",
        "test_rust_loop_adapter.py",
        "test_rust_perf.py",
        "test_rust_astro_parity.py",
        "test_rust_platesolve_parity.py",
        "test_rust_pointing_parity.py",
        "test_rust_imaging_parity.py",
        "test_rust_adsb_parity.py",
        "test_loop_ab_parity.py",
        # Live-fidelity tracking-quality metrics harness: drives the Rust
        # loop through the real adapter (imports skytracker_core at module
        # scope via rust_loop_adapter). Wall-clock based (~85 s); run it
        # locally / in the rust-core job environment.
        "test_tracking_quality.py",
    ]

if not _HAVE_HIPPARCOS:
    # These render/verify against the real Hipparcos catalogue, which must be
    # fetched once from CDS (I/239 hip_main.dat) into the repo root.
    collect_ignore += ["test_star_catalog.py", "test_sim_stars.py"]

if not _HAVE_DE421:
    # Launch trajectories convert ECEF->topocentric through skyfield.
    collect_ignore += ["test_launch_loading.py"]


def pytest_report_header(config):
    return (
        f"skytracker prerequisites: rust_core={_HAVE_RUST} "
        f"de421={_HAVE_DE421} hipparcos={_HAVE_HIPPARCOS} "
        f"(skipped suites are listed in collect_ignore, see conftest.py)"
    )
