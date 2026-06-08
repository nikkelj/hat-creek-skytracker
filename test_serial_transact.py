#!/usr/bin/env python
"""
Regression tests for the hardened NexStar serial transaction layer.

Covers the behavior introduced to fix the tracking-runaway bug:
  * _transact returns a full response and the high-level commands parse it,
  * a short/timed-out read flushes the input buffer and raises SerialCommError
    (instead of hanging or crashing on a partial struct unpack),
  * write+read round-trips are atomic under concurrent callers (the lock keeps
    the control thread and UI thread from interleaving bytes on the wire).

Uses a fake serial device, so it needs only pyserial (pure Python) and no
hardware. Run: python test_serial_transact.py
"""

import threading
import time
import unittest

from lib.auxstar import NexstarHandController, SerialCommError, Targets


class FakeSerial:
    """Minimal stand-in for serial.Serial with a programmable read queue."""

    def __init__(self):
        self.responses = []      # FIFO of bytes objects returned by read()
        self.reset_count = 0
        self.closed = False
        self.read_delay = 0.0
        self.log = []            # ordered ('write'|'read') events for atomicity
        self._loglock = threading.Lock()

    def write(self, data):
        with self._loglock:
            self.log.append('write')
        return len(data)

    def read(self, n):
        if self.read_delay:
            time.sleep(self.read_delay)
        with self._loglock:
            self.log.append('read')
        return self.responses.pop(0) if self.responses else b''

    def reset_input_buffer(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


class SerialTransactTests(unittest.TestCase):

    def test_full_response_acked(self):
        dev = FakeSerial()
        dev.responses = [b'#']  # MC_MOVE_* expects 1 byte
        ctrl = NexstarHandController(dev)
        self.assertTrue(ctrl.hc_slew_fixed(Targets.AZM, 5))
        self.assertEqual(dev.reset_count, 0)

    def test_position_parsed(self):
        dev = FakeSerial()
        # 3 data bytes (0x400000 -> quarter turn) + trailing hash
        dev.responses = [b'\x40\x00\x00#']
        ctrl = NexstarHandController(dev)
        pos = ctrl.hc_get_position(Targets.AZM)
        self.assertAlmostEqual(pos, 0.25, places=6)

    def test_short_read_raises_and_flushes(self):
        dev = FakeSerial()
        dev.responses = [b'']  # timeout -> empty read
        ctrl = NexstarHandController(dev)
        with self.assertRaises(SerialCommError):
            ctrl.hc_get_position(Targets.ALT)
        self.assertEqual(dev.reset_count, 1, "input buffer should be flushed")

    def test_partial_read_raises(self):
        dev = FakeSerial()
        dev.responses = [b'\x40\x00']  # only 2 of 4 expected bytes
        ctrl = NexstarHandController(dev)
        with self.assertRaises(SerialCommError):
            ctrl.hc_get_position(Targets.ALT)
        self.assertEqual(dev.reset_count, 1)

    def test_concurrent_transactions_do_not_interleave(self):
        dev = FakeSerial()
        n = 40
        dev.responses = [b'#'] * n
        dev.read_delay = 0.001  # widen the window for a race to show up
        ctrl = NexstarHandController(dev)

        def worker():
            ctrl.hc_slew_fixed(Targets.AZM, 1)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With the lock, the log must be a clean repetition of write,read pairs.
        self.assertEqual(len(dev.log), 2 * n)
        for i in range(0, len(dev.log), 2):
            self.assertEqual(dev.log[i], 'write')
            self.assertEqual(dev.log[i + 1], 'read')

    def test_close_is_locked(self):
        dev = FakeSerial()
        ctrl = NexstarHandController(dev)
        ctrl.close()
        self.assertTrue(dev.closed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
