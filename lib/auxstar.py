"""
This file implements an API for the Nextar AUX command set, for use with a hand controller
over a serial port directly.

Some inspiration and re-use from:
https://raw.githubusercontent.com/jochym/nexstar-evo/master/nsevo/nexstarevo.py
"""

import argparse
import sys
import serial
import warnings
import datetime
import pytz
import struct
import time
from enum import Enum

class Targets(Enum):
    ANY = 0x00
    MB = 0x01
    HC = 0x04
    UKN1 = 0x05
    HCPLUS = 0x0d
    AZM = 0x10
    ALT = 0x11
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

def dms2f(dd,mm,ss, sign):
    """Convert degrees, minutes, seconds to floating point fraction of full rotation

    Args:
        dd (float): Degrees
        mm (float): Minutes
        ss (float): Seconds
        
    Returns:
        float: fraction of full rotation
    """
    sign = 1 if dd > 360 else -1
    assert dd < 360
    assert mm < 60
    assert ss < 60
    return sign * ( ss/3600 + mm/60 + dd/360 )

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
                    timeout          = 3.500,
                    xonxoff          = False,
                    rtscts           = False,
                    writeTimeout     = 3.500,
                    dsrdtr           = False,
                    interCharTimeout = None
                )

        self._device = device
        self.alt = 0
        self.azm = 0

    @property
    def device(self):
        return self._device

    def close(self):
        return self._device.close()

    def _write_binary(self, request):
        return self._device.write(request)

    def _read_binary(self, expected_response_length, check_and_remove_trailing_hash = True):

        response = self._device.read(expected_response_length)

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
        self._write_binary(binary)
        binary_response = self._read_binary(COMMANDS['MC_GET_VER'][2]+1)
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
        self._write_binary(binary)
        binary_response = self._read_binary(COMMANDS['MC_GET_POSITION'][2]+1)
        result = unpack_int3(binary_response)
        if target == Targets.ALT:
            self.alt = result
        if target == Targets.AZM:
            self.azm = result
        return result

    def hc_goto_fast(self, target, dd, mm, ss):
        """Goto position at high slew rate

        Args:
            target (int): Target device id for command
            dd (float): Target position degrees
            mm (float): Target position minutes
            ss (float): Target position seconds

        Returns:
            int: ack
        """
        fofr = pack_int3(dms2f(dd,mm,ss))
        # HC, expected command bytes including msgid, target, id, data, expected response bytes
        request = '50{:02x}{:02x}{:02x}{:06x}{:02x}'.format(COMMANDS['MC_GOTO_FAST'][1], target.value, COMMANDS['MC_GOTO_FAST'][0], fofr, COMMANDS['MC_GOTO_FAST'][2])
        binary = bytearray.fromhex(request)
        self._write_binary(binary)
        binary_response = self._read_binary(COMMANDS['MC_GOTO_FAST'][2]+1)
        result = unpack_int3(binary_response)
        return result

    def hc_set_position(self, target, dd, mm, ss):
        """Goto position at normal rate

        Args:
            target (int): Target device id for command
            dd (float): Target position degrees
            mm (float): Target position minutes
            ss (float): Target position seconds

        Returns:
            int: ack
        """
        fofr = pack_int3(dms2f(dd,mm,ss))
        # HC, expected command bytes including msgid, target, id, data, expected response bytes
        request = '50{:02x}{:02x}{:02x}{:06x}{:02x}'.format(COMMANDS['MC_SET_POSITION'][1], target.value, COMMANDS['MC_SET_POSITION'][0], fofr, COMMANDS['MC_SET_POSITION'][2])
        binary = bytearray.fromhex(request)
        self._write_binary(binary)
        binary_response = self._read_binary(COMMANDS['MC_SET_POSITION'][2]+1)
        result = unpack_int3(binary_response)
        return result
    
    def hc_set_guide_rate(self, target, rate, sidereal=False, solar=False, lunar=False):
        """Set guide rate

        Args:
            target (int): Target device id for command
            rate (float): Guide rate (TODO: units???), sign-only if sidereal/solar/lunar

        Returns:
            int: ack
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
            packed_rate = pack_int3(rate)
            request = '50{:02x}{:02x}{:02x}{:06x}{:02x}'.format(COMMANDS[cmd][1], target.value, COMMANDS[cmd][0], packed_rate, COMMANDS[cmd][2])
        binary = bytearray.fromhex(request)
        self._write_binary(binary)
        binary_response = self._read_binary(COMMANDS[cmd][2]+1)
        result = unpack_int3(binary_response)
        return result
    
    def hc_slew_fixed(self, target, rate):
        """Move axis. Axis will keep moving until a stop is sent!
        
        Args:
            target (int): Target device id for command
            rate (int): Rate step, where 0 = stop, 1 to 9 = positive, -1 to -9 = negative

        Returns:
            int: ack
        """
        cmd = 'MC_MOVE_POS' if rate >= 0 else 'MC_MOVE_NEG'
        request = '50{:02x}{:02x}{:02x}{:02x}0000{:02x}'.format(COMMANDS[cmd][1], target.value, COMMANDS[cmd][0], abs(rate), COMMANDS[cmd][2])
        binary = bytearray.fromhex(request)
        self._write_binary(binary)
        binary_response = self._read_binary(COMMANDS[cmd][2]+1)
        result = ''.join(format(x, '02x') for x in binary_response)
        return result
    
    def hc_set_backlash(self, target, backlash):
        """Set backlash, +/- 0-99
        
        Args:
            target (int): Target device id for command
            backlash (int): Backlash setting, from +/- 0-99

        Returns:
            int: ack
        """
        cmd = 'MC_SET_POS_BACKLASH' if backlash >= 0 else 'MC_SET_NEG_BACKLASH'
        request = '50{:02x}{:02x}{:02x}{:02x}0000{:02x}'.format(COMMANDS[cmd][1], target.value, COMMANDS[cmd][0], backlash, COMMANDS[cmd][2])
        binary = bytearray.fromhex(request)
        self._write_binary(binary)
        binary_response = self._read_binary(COMMANDS[cmd][2]+1)
        result = ''.join(format(x, '02x') for x in binary_response)
        return result

def status_report(controller):

    tz = pytz.timezone('US/Pacific')

    alt_version = controller.hc_get_version(target=Targets.ALT)
    azm_version = controller.hc_get_version(target=Targets.AZM)
    hc_version = controller.hc_get_version(target=Targets.HC)
    print("ALT version ............................. : {}".format(alt_version))
    print("AZM version ............................. : {}".format(azm_version))
    print("HC version ............................. : {}".format(hc_version))
    
    alt = controller.hc_get_position(target=Targets.ALT)
    azm = controller.hc_get_position(target=Targets.AZM)
    print("ALT ............................. : {}".format(repr_angle(alt)))
    print("AZM ............................. : {}".format(repr_angle(azm)))

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
        controller.hc_slew_fixed(Targets.AZM, 9)
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