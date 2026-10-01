"""Record the closed-loop tracking-quality baseline for the Rust port.

Phase 0 of the Rust port. Runs the full-wiring tracking-quality suite
(test_tracking_quality.py: SimMount + SimCap + real CameraThread +
RustCoreLoopAdapter + JoystickModeState) and captures every `[metrics]`
line it prints into tests/golden/loop_baseline.json. Later port phases
(1, 4, 6, 7) gate on these numbers: quality must stay at or better than
this snapshot.

The baseline reflects the current flag-on configuration (Rust core loop),
i.e. what the instrument actually flies with today.

Run under the `track` conda env from the repo root:

    C:\\Users\\nikke\\anaconda3\\envs\\track\\python.exe tools/record_loop_baseline.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tests", "golden", "loop_baseline.json")

# e.g. "[metrics] PROGRAM: bias=12.3 jitter(std)=8.1 rms=14.2 p95=25.0 peak=31.9 arcsec"
LINE_RE = re.compile(r"\[metrics\]\s+([^:]+):\s+(.*)")
KV_RE = re.compile(r"([\w()/]+)=([-+.\deE]+)")


def main() -> int:
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_tracking_quality.py", "-s", "-q"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    sys.stdout.write(proc.stdout[-2000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        print("\ntracking-quality suite FAILED; baseline not recorded")
        return 1

    metrics: dict[str, dict[str, float]] = {}
    for line in proc.stdout.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        scenario = m.group(1).strip()
        rest = m.group(2)
        vals = {k: float(v) for k, v in KV_RE.findall(rest)}
        if not vals:
            # Prose format, e.g. "HANDOFF latency: 0.87s; target
            # false-reject episodes: 0".
            lat = re.search(r"([\d.]+)s", rest)
            rej = re.search(r"episodes:\s*(\d+)", rest)
            if lat:
                vals["latency_s"] = float(lat.group(1))
            if rej:
                vals["false_reject_episodes"] = float(rej.group(1))
            if not vals:
                vals["raw"] = rest.strip()
        # Units: the suite prints arcsec for angle metrics, seconds for times.
        metrics.setdefault(scenario, {}).update(vals)

    if not metrics:
        print("no [metrics] lines captured; baseline not recorded")
        return 1

    baseline = {
        "source": "test_tracking_quality.py full-wiring suite",
        "loop": "rust (RustCoreLoopAdapter, use_rust_core_loop flag-on config)",
        "units": {"angles": "arcsec", "times": "s"},
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
    print(f"\nbaseline written: {OUT}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
