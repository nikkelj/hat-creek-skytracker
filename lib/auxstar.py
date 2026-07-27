"""
This file implements an API for the Nextar AUX command set, for use with a hand controller
over a serial port directly.

Some inspiration and re-use from:
https://raw.githubusercontent.com/jochym/nexstar-evo/master/nsevo/nexstarevo.py
"""

import argparse
import serial
import struct
import threading
import time
from enum import Enum


class SerialCommError(IOError):
    """Raised when a mount transaction returns a short/timed-out response.

    Callers (notably the control thread) should treat this as 'skip this
    cycle' rather than letting the failure stall the loop.
    """
    pass

class Targets(Enum):
    ANY = 0x00
    MB = 0x01
    HC = 0x04
    UKN1 = 0x05
    HCPLUS = 0x0d
    AZM = 0x10
    ALT = 0x11
    FOCUS = 0x12
    APP = 0x20
    GPS = 0xb0
    UKN2 = 0xb4
    WIFI = 0xb5
    BAT = 0xb6
    CHG = 0xb7
    LIGHT = 0xbf
    
class Control(Enum):
    HC = 0x04
    HCPLUS = 0x0d
    APP = 0x20
    
# Command set, defined by command:(id, expected command bytes including msgid, expected response bytes)
# Focus motor: https://www.cloudynights.com/topic/750060-celestron-mototfocus-command-interface/
# https://indilib.org/forum/general/4470-celestron-motorised-focuser-is-it-supported-in-ekos-yet.html
COMMANDS={
          'MC_GET_POSITION':(0x01, 1, 3),
          'MC_GOTO_FAST':(0x02, 4, 0),
          'MC_SET_POSITION':(0x04, 4, 0),
          'MC_UNKNONW_1':(0x05, 1, 0),
          'MC_SET_POS_GUIDERATE':(0x06, 4, 0),
          'MC_SET_NEG_GUIDERATE':(0x07, 4, 0),
          'MC_LEVEL_START':(0x0b, 1, 0),
          'MC_PEC_RECORD_START': (0x0c, 1, 0),
          'MC_PEC_PLAYBACK': (0x0d, 2, 0),
          'MC_SET_POS_BACKLASH':(0x10, 2, 0),
          'MC_SET_NEG_BACKLASH':(0x11, 2, 0),
          'MC_LEVEL_DONE': (0x12, 1, 1),
          'MC_SLEW_DONE':(0x13, 1, 1),
          'MC_UNKNOWN_2': (0x14, 1, 0),
          'MC_PEC_RECORD_DONE': (0x15, 1, 1),
          'MC_PEC_RECORD_STOP': (0x16, 1, 0),
          'MC_GOTO_SLOW':(0x17, 3, 0),
          'MC_AT_INDEX':(0x18, 1, 1),
          'MC_SEEK_INDEX':(0x19, 1, 0),
          'MC_SET_MAXRATE':(0x20, 2, 0),
          'MC_GET_MAXRATE':(0x21, 1, 1),
          'MC_ENABLE_MAXRATE':(0x22, 1, 0),
          'MC_MAXRATE_ENABLED':(0x23, 1, 0),
          'MC_MOVE_POS':(0x24, 2, 0),
          'MC_MOVE_NEG':(0x25, 2, 0),
          'FOC_GET_HS_POSITIONS':(0x2c, 2, 8), # returns 2 32-bit uints containing low and high limits
          'MC_ENABLE_CORDWRAP':(0x38, 1, 0),
          'MC_DISABLE_CORDWRAP':(0x39, 1, 0),
          'MC_SET_CORDWRAP_POS':(0x3a, 4, 0),
          'MC_POLL_CORDWRAP':(0x3b, 1, 1),
          'MC_GET_CORDWRAP_POS':(0x3c, 3),
          'MC_GET_POS_BACKLASH':(0x40, 1, 1),
          'MC_GET_NEG_BACKLASH':(0x41, 1, 1),
          'MC_SET_AUTOGUIDE_RATE':(0x46, 2, 0),
          'MC_GET_AUTOGUIDE_RATE':(0x47, 1, 1),
          'MC_GET_APPROACH':(0xfc, 1, 1),
          'MC_SET_APPROACH':(0xfd, 2, 1),
          'MC_GET_VER':(0xfe, 1, 2),
         }
COMMAND_NAMES={value:key for key, value in COMMANDS.items()}

# Rates are defined as fractions of a full revolution per second
RATES = {
    0 : 0.0,
    1 : 1/(360*60),
    2 : 2/(360*60),
    3 : 5/(360*60),
    4 : 15/(360*60),
    5 : 30/(360*60),
    6 : 1/360,
    7 : 2/360,
    8 : 5/360,
    9 : 10/360
}
# WARNING: this MC_MOVE step table is SUSPECT on the real AVX. Bench
# measurement 2026-07-25: rate 4 moved at 0.0335 dps = 8.0x sidereal, not the
# 0.25 dps listed here (the classic Celestron HC progression is 0.5/1/4/8/16/
# 64x sidereal then deg/s steps). The sim and the discrete-rate PID path use
# this table, so until it is re-measured (bench_guiderate.py --survey) treat
# discrete-mode rate expectations as approximate; continuous guide-rate
# tracking (hc_set_rate_dps, calibrated below) does not depend on it.

# Firmware guide-rate unit for MC_SET_POS/NEG_GUIDERATE, CALIBRATED on the
# real AVX (bench_guiderate.py, 2026-07-25): the 24-bit value is arcseconds
# per second in Q10 fixed point -- value = arcsec_per_sec * 1024. The LSB is
# ~0.001"/s and full scale (2^24 - 1) is 4.551 deg/s, matching the AVX max
# slew. (The previous rev/sec * 2^24 assumption ran 79.1x slow -- commanded
# rates produced "no movement".)
GUIDE_COUNTS_PER_DPS = 3600.0 * 1024.0
GUIDE_RATE_MAX_DPS = (2 ** 24 - 1) / GUIDE_COUNTS_PER_DPS  # 4.551 deg/s

# Utility functions
def checksum(msg):
    return ((~sum([c for c in bytes(msg)]) + 1) ) & 0xFF

def f2dms(f):
    '''
    Convert fraction of the full rotation to DMS triple (degrees).
    '''
    s= 1 if f>0 else -1
    d=360*abs(f)
    dd=int(d)
    mm=int((d-dd)*60)
    ss=(d-dd-mm/60)*3600
    return s*dd,mm,ss

def dms2f(dd,mm,ss, sign=1):
    """Convert degrees, minutes, seconds to floating point fraction of full rotation

    Args:
        dd (float): Degrees
        mm (float): Minutes
        ss (float): Seconds
        sign (int): Sign of the input angle (-1 or 1)
        
    Returns:
        float: signed fraction of full rotation
    """
    assert abs(dd) < 360
    assert mm < 60
    assert ss < 60
    # mm/ss are arcminutes/arcseconds, i.e. fractions of a *degree*, so the
    # whole angle in degrees is |dd| + mm/60 + ss/3600; divide by 360 for the
    # fraction of a rotation. (The sign convention matches f2dms, which puts
    # the sign on the degrees term only.)
    axis_sign = -1 if dd < 0 else 1
    return sign * axis_sign * (abs(dd) + mm / 60 + ss / 3600) / 360

def parse_pos(d):
    '''
    Parse first three bytes into the DMS string
    '''
    if len(d)>=3 :
        pos=struct.unpack('!i',b'\x00'+d[:3])[0]/2**24
        return u'%03d°%02d\'%04.1f"' % f2dms(pos)
    else :
        return u''

def repr_pos(alt,azm):
    return u'(%03d°%02d\'%04.1f", %03d°%02d\'%04.1f")' % (f2dms(alt) + f2dms(azm))

def repr_angle(a):
    return u'%03d°%02d\'%04.1f"' % f2dms(a)

def format_angle_compact(angle):
    """Format angle as compact DMS string without degree symbol"""
    d, m, s = f2dms(angle)
    return f"{d:3d}°{m:02d}'{s:04.1f}\""

def unpack_int3(d):
    return struct.unpack('!i',b'\x00'+d[:3])[0]/2**24

def pack_int3(f):
    return struct.pack('!i',int(f*(2**24)))[1:]
    
def unpack_int2(d):
    return struct.unpack('!i',b'\x00\x00'+d[:2])[0]

def pack_int2(v):
    return struct.pack('!i',int(v))[-2:]

def dprint(m):
    m=bytes(m)
    for c in m:
        if c==0x3b :
            print()
        print("%02x" % c, end=':')
    print()

class NexstarHandController:

    def __init__(self, device):

        if isinstance(device, str):
            # For now, if we're passed a string, assume it's a serial device name.
            # We may add support for TCP ports later.
            device = serial.Serial(
                    port             = device,
                    baudrate         = 9600,
                    bytesize         = serial.EIGHTBITS,
                    parity           = serial.PARITY_NONE,
                    stopbits         = serial.STOPBITS_ONE,
                    # Short timeouts: a missing response must never stall the
                    # control loop for seconds. A timed-out read is reported as
                    # a SerialCommError and the cycle is skipped/stopped.
                    timeout          = 0.25,
                    xonxoff          = False,
                    rtscts           = False,
                    writeTimeout     = 0.25,
                    dsrdtr           = False,
                    interCharTimeout = None
                )

        self._device = device
        # Serializes every write+read transaction so the control thread and the
        # UI thread can never interleave bytes on the wire.
        self._lock = threading.RLock()
        self.alt = 0
        self.azm = 0
        self.focus = 0

    @property
    def device(self):
        return self._device

    def close(self):
        with self._lock:
            return self._device.close()

    def _write_binary(self, request):
        return self._device.write(request)

    def _read_binary(self, expected_response_length, check_and_remove_trailing_hash = True):

        response = self._device.read(expected_response_length)

        return response

    def _transact(self, request, expected_response_length):
        """Atomically write a request and read its full response.

        Holds the device lock for the whole round-trip and validates that the
        complete response arrived. On a short/timed-out read the input buffer
        is flushed (so the next command starts on a clean stream) and a
        SerialCommError is raised.
        """
        with self._lock:
            self._device.write(request)
            response = self._device.read(expected_response_length)
            if len(response) < expected_response_length:
                try:
                    self._device.reset_input_buffer()
                except Exception:
                    pass
                raise SerialCommError(
                    f"Short read: got {len(response)} of "
                    f"{expected_response_length} bytes"
                )
            return response

    ################################## Public API ##########################################
    def hc_get_version(self, target):
        """Get firmware version

        Args:
            target (int): Target device id for command

        Returns:
            int: firmware version
        """
        # HC, expected command bytes including msgid, target, id, data, expected response bytes
        request = '50{:02x}{:02x}{:02x}000000{:02x}'.format(COMMANDS['MC_GET_VER'][1], target.value, COMMANDS['MC_GET_VER'][0], COMMANDS['MC_GET_VER'][2])
        binary = bytearray.fromhex(request)
        binary_response = self._transact(binary, COMMANDS['MC_GET_VER'][2]+1)
        result = ''.join(format(x, '02x') for x in binary_response[0:-1])
        return result

    def hc_get_position(self, target):
        """Get current position

        Args:
            target (int): Target device id for command

        Returns:
            int: position
        """
        # HC, expected command bytes including msgid, target, id, data, expected response bytes
        request = '50{:02x}{:02x}{:02x}000000{:02x}'.format(COMMANDS['MC_GET_POSITION'][1], target.value, COMMANDS['MC_GET_POSITION'][0], COMMANDS['MC_GET_POSITION'][2])
        binary = bytearray.fromhex(request)
        binary_response = self._transact(binary, COMMANDS['MC_GET_POSITION'][2]+1)
        result = unpack_int3(binary_response)
        if target == Targets.ALT:
            self.alt = result
        if target == Targets.AZM:
            self.azm = result
        if target == Targets.FOCUS:
            self.focus = result
        return result

    def hc_goto_fast(self, target, dd, mm, ss):
        """Goto position at high slew rate

        Args:
            target (int): Target device id for command
            dd (float): Target position degrees (can be signed)
            mm (float): Target position minutes (can be signed)
            ss (float): Target position seconds (can be signed)

        Returns:
            boolean: True for success, False for failure
        """
        fofr = pack_int3(dms2f(dd,mm,ss, 1)) # Signed fraction of full rotation
        # HC, expected command bytes including msgid, target, id, data, expected response bytes
        request = '50{:02x}{:02x}{:02x}'.format(COMMANDS['MC_GOTO_FAST'][1], target.value, COMMANDS['MC_GOTO_FAST'][0]) + ''.join(['%02x' % c for c in fofr]) + '{:02x}'.format(COMMANDS['MC_GOTO_FAST'][2])
        binary = bytearray.fromhex(request)
        binary_response = self._transact(binary, COMMANDS['MC_GOTO_FAST'][2]+1)
        return binary_response == b'#'

    def hc_set_position(self, target, dd, mm, ss):
        """Goto position at normal rate

        Args:
            target (int): Target device id for command
            dd (float): Target position degrees (can be signed)
            mm (float): Target position minutes (can be signed)
            ss (float): Target position seconds (can be signed)

        Returns:
            boolean: True for success, False for failure
        """
        fofr = pack_int3(dms2f(dd,mm,ss, 1))
        # HC, expected command bytes including msgid, target, id, data, expected response bytes
        request = '50{:02x}{:02x}{:02x}'.format(COMMANDS['MC_SET_POSITION'][1], target.value, COMMANDS['MC_SET_POSITION'][0]) + ''.join(['%02x' % c for c in fofr]) + '{:02x}'.format(COMMANDS['MC_SET_POSITION'][2])
        binary = bytearray.fromhex(request)
        binary_response = self._transact(binary, COMMANDS['MC_SET_POSITION'][2]+1)
        return binary_response == b'#'
    
    def hc_set_guide_rate(self, target, rate, sidereal=False, solar=False, lunar=False):
        """Set guide rate

        Args:
            target (int): Target device id for command
            rate (float): Guide rate (TODO: units???), sign-only if sidereal/solar/lunar

        Returns:
            boolean: True for success, False for failure
        """
        cmd = 'MC_SET_POS_GUIDERATE' if rate > 0 else 'MC_SET_NEG_GUIDERATE'
        # HC, expected command bytes including msgid, target, id, data, expected response bytes
        if sidereal:
            request = '50{:02x}{:02x}{:02x}ffff00{:02x}'.format(COMMANDS[cmd][1], target.value, COMMANDS[cmd][0], COMMANDS[cmd][2])
        elif solar:
            request = '50{:02x}{:02x}{:02x}fffe00{:02x}'.format(COMMANDS[cmd][1], target.value, COMMANDS[cmd][0], COMMANDS[cmd][2])
        elif lunar:
            request = '50{:02x}{:02x}{:02x}fffd00{:02x}'.format(COMMANDS[cmd][1], target.value, COMMANDS[cmd][0], COMMANDS[cmd][2])
        else:
            # Direction is carried by POS vs NEG command id, so the magnitude is
            # what gets packed (packing a negative through pack_int3 would corrupt
            # the 3-byte value).
            packed_rate = pack_int3(abs(rate))
            # packed_rate is 3 raw bytes -- hex-encode them like every other
            # command builder ('{:06x}' on bytes raises TypeError).
            request = '50{:02x}{:02x}{:02x}'.format(COMMANDS[cmd][1], target.value, COMMANDS[cmd][0]) + packed_rate.hex() + '{:02x}'.format(COMMANDS[cmd][2])
        binary = bytearray.fromhex(request)
        binary_response = self._transact(binary, COMMANDS[cmd][2]+1)
        return binary_response == b'#'

    def hc_set_rate_dps(self, target, dps):
        """Command a continuous axis rate in degrees/second via the fine 24-bit
        variable-rate (MC_SET_POS/NEG_GUIDERATE) primitive, instead of the coarse
        10-step MC_MOVE. This is the smooth-tracking path that avoids the
        rate-quantization sawtooth.

        On-wire scale CALIBRATED on the real AVX (bench_guiderate.py,
        2026-07-25): value = arcsec_per_sec * 1024 (GUIDE_COUNTS_PER_DPS).
        Rates beyond the 24-bit full scale (GUIDE_RATE_MAX_DPS, 4.551 dps) are
        clamped; callers gate via config.guide_rate_max_dps and fall back to
        MC_MOVE above it anyway.
        """
        # Round+clamp in COUNT space so full scale encodes exactly 0xFFFFFF
        # (clamping in deg/s and converting back can land one LSB short).
        counts = min(int(round(abs(dps) * GUIDE_COUNTS_PER_DPS)), 2 ** 24 - 1)
        fraction = counts / 2 ** 24  # exact: pack_int3 recovers `counts`
        return self.hc_set_guide_rate(target, fraction if dps >= 0 else -fraction)

    def hc_slew_fixed(self, target, rate):
        """Move axis. Axis will keep moving until a stop is sent!
        
        Args:
            target (int): Target device id for command
            rate (int): Rate step, where 0 = stop, 1 to 9 = positive, -1 to -9 = negative

        Returns:
            boolean: True for success, False for failure
        """
        cmd = 'MC_MOVE_POS' if rate >= 0 else 'MC_MOVE_NEG'
        request = '50{:02x}{:02x}{:02x}{:02x}0000{:02x}'.format(COMMANDS[cmd][1], target.value, COMMANDS[cmd][0], abs(rate), COMMANDS[cmd][2])
        binary = bytearray.fromhex(request)
        binary_response = self._transact(binary, COMMANDS[cmd][2]+1)
        return binary_response == b'#'
    
    def hc_set_backlash(self, target, backlash):
        """Set backlash, +/- 0-99
        
        Args:
            target (int): Target device id for command
            backlash (int): Backlash setting, from +/- 0-99

        Returns:
            boolean: True for success, False for failure
        """
        cmd = 'MC_SET_POS_BACKLASH' if backlash >= 0 else 'MC_SET_NEG_BACKLASH'
        request = '50{:02x}{:02x}{:02x}{:02x}0000{:02x}'.format(COMMANDS[cmd][1], target.value, COMMANDS[cmd][0], backlash, COMMANDS[cmd][2])
        binary = bytearray.fromhex(request)
        binary_response = self._transact(binary, COMMANDS[cmd][2]+1)
        return binary_response == b'#'

def status_report(controller):

    alt_version = controller.hc_get_version(target=Targets.ALT)
    azm_version = controller.hc_get_version(target=Targets.AZM)
    hc_version = controller.hc_get_version(target=Targets.HC)
    print(f"ALT version ............................. : {alt_version}")
    print(f"AZM version ............................. : {azm_version}")
    print(f"HC version ............................. : {hc_version}")
    
    alt = controller.hc_get_position(target=Targets.ALT)
    azm = controller.hc_get_position(target=Targets.AZM)
    focus = controller.hc_get_position(target=Targets.FOCUS)
    print(f"ALT ............................. : {repr_angle(alt)}")
    print(f"AZM ............................. : {repr_angle(azm)}")
    print(f"FOCUS ............................. : {repr_angle(focus)}")

def main():
    """Provide a basic CLI"""
    parser = argparse.ArgumentParser(
                    prog='auxstar.py',
                    description='Test Auxstar Functionality')
    parser.add_argument("--port", type=str, default=None, help='Serial port to communicate on')
    parser.add_argument("--test", action="store_true", help="Execute test wiggle")
    args = parser.parse_args()

    port = args.port

    controller = NexstarHandController(port)

    if args.test:
        status_report(controller)
        print('Slewing ALT...')
        controller.hc_slew_fixed(Targets.ALT, 9)
        time.sleep(1)
        print('Slewing AZM...')
        controller.hc_slew_fixed(Targets.AZM, 9)
        print('Slewing FOCUS...')
        
        #controller.slew(NexstarDeviceId.ALT_DEC_MOTOR, +0.001)
        time.sleep(3)
        print('Stopping...')
        controller.hc_slew_fixed(Targets.ALT, 0)
        controller.hc_slew_fixed(Targets.AZM, 0)
        status_report(controller)
        

    controller.close()

if __name__ == "__main__":
    main()

# Below are some low-level HC <-> MC commands:
#
# Standard boot sequence:

# 3b 03 0d 11 05 da    3b 05 11 0d 05 0f 87 42
# 3b 03 0d 10 05 db    3b 05 10 0d 05 0f 87 43
# 3b 03 0d 10 fe e2    3b 05 10 0d fe 06 0d cd
# 3b 03 0d 10 fc e4    3b 04 10 0d fc 00 e3
# 3b 03 0d 11 fc e3    3b 04 11 0d fc 01 e1

# Slew left (hand controller)
#
# 3b 04 0d 10 25 09 b1 3b 04 10 0d 25 01 b9
# 3b 04 0d 10 24 00 bb 3b 04 10 0d 24 01 ba
#
# Slew right (hand controller)
#
# 3b 04 0d 10 24 09 b2 3b 04 10 0d 24 01 ba
# 3b 04 0d 10 24 00 bb 3b 04 10 0d 24 01 ba
#
# Slew up (hand controller)
#
# 3b 04 0d 11 24 09 b1 3b 04 11 0d 24 01 b9
# 3b 04 0d 11 24 00 ba 3b 04 11 0d 24 01 b9
#
# Slew down (hand controller)
#
# 3b 04 0d 11 25 09 b0 3b 04 11 0d 25 01 b8
# 3b 04 0d 11 24 00 ba 3b 04 11 0d 24 01 b9
