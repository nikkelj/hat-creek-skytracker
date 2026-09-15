#!/usr/bin/env python
"""
Cross-language byte-parity tests for the Rust Mount port (skytracker_core).

Asserts that the Rust request encoders emit byte-for-byte the same packets that
the real `lib.auxstar.NexstarHandController` writes to the wire, and that the
Rust byte-level SimMount closes a tracking loop (slew -> position advances).

Build the extension first, then run:
    cd rust/skytracker-ffi && maturin develop --release
    python test_rust_mount_parity.py

Skips cleanly (does not fail) if skytracker_core isn't built yet, so the rest of
the Python suite is unaffected on the experiment branch.
"""

import time
import unittest

try:
    import skytracker_core as rc
    _HAVE_CORE = True
except ImportError:
    _HAVE_CORE = False

from lib.auxstar import NexstarHandController, Targets


class _Recorder:
    """Fake serial that records written bytes and returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.written = b""

    def write(self, data):
        self.written += bytes(data)
        return len(data)

    def read(self, n):
        return self.response[:n]

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


def _py_request(method, response, *args):
    """Return the bytes NexstarHandController.<method>(*args) writes."""
    dev = _Recorder(response)
    ctrl = NexstarHandController(dev)
    getattr(ctrl, method)(*args)
    return dev.written


@unittest.skipUnless(_HAVE_CORE, "skytracker_core not built (run `maturin develop`)")
class RustMountParity(unittest.TestCase):

    def test_get_position_bytes(self):
        for py_target, rs_target in [(Targets.AZM, rc.AZM), (Targets.ALT, rc.ALT)]:
            py = _py_request("hc_get_position", b"\x00\x00\x00#", py_target)
            self.assertEqual(py, rc.encode_get_position(rs_target))

    def test_get_version_bytes(self):
        py = _py_request("hc_get_version", b"\x06\x0d#", Targets.HC)
        self.assertEqual(py, rc.encode_get_version(rc.HC))

    def test_slew_fixed_bytes(self):
        for rate in (0, 1, 5, 9, -1, -5, -9):
            py = _py_request("hc_slew_fixed", b"#", Targets.AZM, rate)
            self.assertEqual(py, rc.encode_slew_fixed(rc.AZM, rate), f"rate={rate}")

    def test_goto_fast_bytes(self):
        for dd, mm, ss in [(0, 0, 0), (45, 30, 0), (123, 45, 30)]:
            py = _py_request("hc_goto_fast", b"#", Targets.AZM, dd, mm, ss)
            self.assertEqual(
                py, rc.encode_goto_fast(rc.AZM, dd, mm, ss), f"dms={dd},{mm},{ss}"
            )

    def test_pack_int3_matches(self):
        from lib.auxstar import pack_int3 as py_pack
        for f in (0.0, 0.25, 0.5, 0.123456):
            self.assertEqual(py_pack(f), rc.pack_int3(f), f"f={f}")

    def test_sim_closed_loop(self):
        m = rc.SimMount()
        self.assertTrue(m.hc_slew_fixed(rc.AZM, 9))
        time.sleep(0.2)
        pos = m.hc_get_position(rc.AZM)
        self.assertGreater(pos, 0.0)
        self.assertLess(pos, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
