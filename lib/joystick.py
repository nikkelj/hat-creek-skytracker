"""
Joystick CLI provides a pygame instance that displays joystick status
and handles joystick input. It provides a tare functionality, and allows
us to map the telescope rate commands proportionally onto each joystick axis.
"""
import argparse
import ctypes
import cv2 #opencv library
import h5py
import math as m
import numpy as np #numpy math library
import pygame
import time 

from config import CAM1_XSIZE, CAM1_YSIZE, CAM2_XSIZE, CAM2_YSIZE

from auxstar import NexstarHandController, status_report, RATES, Targets
from asi_python import ASI_CAMERA_INFO, ASI_CONTROL_CAPS, _errorcodes, _exposurecodes, _imgtypes

libasi = ctypes.cdll["ASICamera2"]

numberOfCameras = libasi.ASIGetNumOfConnectedCameras()
print("Num Connected Cameras:" + str(numberOfCameras))

pygame.init()

stopped = False

class TextPrint:
    """
    This is a simple class that will help us print to the screen.
    It has nothing to do with the joysticks, just outputting the
    information.
    """
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 25)

    def tprint(self, screen, text):
        text_bitmap = self.font.render(text, True, (0, 0, 0))
        screen.blit(text_bitmap, (self.x, self.y))
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10
        
class JoystickConfig:
    """
    This class stores config for select joystick axes
    """
    def __init__(self, tare_button=0, stop_button=1):
        self.reset()
        self.tare_button=tare_button
        self.stop_button=stop_button
        
    def reset(self):
        self.tare = [0] * 10

    def tare4(self, joystick):
        """Tare the first 4 axes"""
        for i in range(0,4):
            axis = joystick.get_axis(i)
            print(f"Tared Axis {i} value: {axis:>6.3f}")
            self.tare[i] = axis
        
def render_joystick_status(joystick, text_print, screen, joystick_config):
    """Render status of a joystick

    Args:
        joystick (Joystick): Joystick instance
        text_print (TextPrint): TextPrint instance
        screen (Screen): Screen instnace
        joystick_config (JoystickConfig): Joystick config instance
    """
    jid = joystick.get_instance_id()

    text_print.tprint(screen, f"Joystick {jid}")
    text_print.indent()

    # Get the name from the OS for the controller/joystick.
    name = joystick.get_name()
    text_print.tprint(screen, f"Joystick name: {name}")

    guid = joystick.get_guid()
    text_print.tprint(screen, f"GUID: {guid}")

    power_level = joystick.get_power_level()
    text_print.tprint(screen, f"Joystick's power level: {power_level}")

    # Usually axis run in pairs, up/down for one, and left/right for
    # the other. Triggers count as axes.
    axes = joystick.get_numaxes()
    text_print.tprint(screen, f"Number of axes: {axes}")
    text_print.indent()

    for i in range(axes):
        axis = joystick.get_axis(i)
        text_print.tprint(screen, f"Axis {i} value (tared): {(axis-joystick_config.tare[i]):>6.3f}")
    text_print.unindent()

    buttons = joystick.get_numbuttons()
    text_print.tprint(screen, f"Number of buttons: {buttons}")
    text_print.indent()

    for i in range(buttons):
        button = joystick.get_button(i)
        text_print.tprint(screen, f"Button {i:>2} value: {button}")
    text_print.unindent()

    hats = joystick.get_numhats()
    text_print.tprint(screen, f"Number of hats: {hats}")
    text_print.indent()

    # Hat position. All or nothing for direction, not a float like
    # get_axis(). Position is a tuple of int values (x, y).
    for i in range(hats):
        hat = joystick.get_hat(i)
        text_print.tprint(screen, f"Hat {i} value: {str(hat)}")
    text_print.unindent()

    text_print.unindent()
    
def process_events(event, joysticks, joystick_config, controller):
    """Process Events

    Args:
        event (Event): Event instance
        joysticks (dict): Dict of joystick instnaces
        joystick_tare (JoystickConfig): Joystick config instance
        controller (NexstarHandController): HC instance
    """
    global stopped
    if event.type == pygame.QUIT:
        done = True  # Flag that we are done so we exit this loop.

    if event.type == pygame.JOYBUTTONDOWN:
        print(f"Joystick button pressed: {event.button}")
        # Joystick Tare Event (X button)
        if event.button == joystick_config.tare_button:
            joystick = joysticks[event.instance_id]
            joystick_config.tare4(joystick)
            
            #if joystick.rumble(0, 0.7, 500):
            #    print(f"Rumble effect played on joystick {event.instance_id}")
        # Joystick Stop Event (Circle Button)
        if event.button == joystick_config.stop_button:
            stopped = not stopped
            if not stopped:
                print("LOOSED!")
            else:
                print("STOPPED!")
                
            
    if event.type == pygame.JOYBUTTONUP:
        print(f"Joystick button released: {event.button}")

    # Handle hotplugging
    if event.type == pygame.JOYDEVICEADDED:
        # This event will be generated when the program starts for every
        # joystick, filling up the list without needing to create them manually.
        joy = pygame.joystick.Joystick(event.device_index)
        joysticks[joy.get_instance_id()] = joy
        print(f"Joystick {joy.get_instance_id()} connencted")

    if event.type == pygame.JOYDEVICEREMOVED:
        del joysticks[event.instance_id]
        print(f"Joystick {event.instance_id} disconnected")
        
def rate_control(joystick, controller, screen, joystick_config):
    """Joystick driven rate controller
    
    Args:
        joystick (Joystick): Joystick instance
        controller (NextstarHandController): HC controller instance
        screen (Screen): Screen instnace
        joystick_config (JoystickConfig): Joystick config instance
    """
    for i in range(0,4):
        axis = joystick.get_axis(i)
        # Azimuth controller
        if i == 2:
            if stopped:
                controller.hc_slew_fixed(Targets.AZM, 0)
            else:
                azm_axis_value_mapped = int(m.floor((axis-joystick_config.tare[i])*(list(RATES.keys())[-1]+3)))
                if azm_axis_value_mapped > list(RATES.keys())[-1]:
                    azm_axis_value_mapped = list(RATES.keys())[-1]
                if azm_axis_value_mapped < -list(RATES.keys())[-1]:
                    azm_axis_value_mapped = -list(RATES.keys())[-1]
                controller.hc_slew_fixed(Targets.AZM, azm_axis_value_mapped)
                print(f"Azimuth {i} rate (mapped): {azm_axis_value_mapped:>6.3f}")
        # Altitude controller
        if i == 3:
            if stopped:
                controller.hc_slew_fixed(Targets.ALT, 0)
            else:
                alt_axis_value_mapped =int(m.floor((axis-joystick_config.tare[i])*(list(RATES.keys())[-1]+3)))
                if alt_axis_value_mapped > list(RATES.keys())[-1]:
                    alt_axis_value_mapped = list(RATES.keys())[-1] 
                if alt_axis_value_mapped < -list(RATES.keys())[-1]:
                    alt_axis_value_mapped = -list(RATES.keys())[-1] 
                controller.hc_slew_fixed(Targets.ALT, alt_axis_value_mapped)
                print(f"Altitude {i} rate (mapped): {alt_axis_value_mapped:>6.3f}")
        
def main():
    """Provide a basic joystick CLI for a NexStar Telescope using the AUX HC Interface"""
    
    parser = argparse.ArgumentParser(
                    prog='joystick.py',
                    description='Test Joystick Functionality')
    parser.add_argument("--port", type=str, default="COM4", help='HC serial port to communicate on')
    args = parser.parse_args()

    # Initialize the telescope
    controller = NexstarHandController(args.port)
    status_report(controller)
        
    # Set the width and height of the screen (width, height), and name the window.
    joystick_screen = pygame.display.set_mode((500, 700))
    pygame.display.set_caption("AuxStar Joystick CLI")
    camera1_screen = pygame.display.set_mode((1548/2.5, 1040/2.5))

    # Used to manage how fast the screen updates.
    clock = pygame.time.Clock()

    # Get ready to print.
    text_print = TextPrint()
    
    # Create the joystick tare object
    joystick_config = JoystickConfig()

    # This dict can be left as-is, since pygame will generate a
    # pygame.JOYDEVICEADDED event for every joystick connected
    # at the start of the program.
    joysticks = {}

    done = False
    while not done:
        # Event processing step.
        # Possible joystick events: JOYAXISMOTION, JOYBALLMOTION, JOYBUTTONDOWN,
        # JOYBUTTONUP, JOYHATMOTION, JOYDEVICEADDED, JOYDEVICEREMOVED
        for event in pygame.event.get():
            process_events(event, joysticks, joystick_config, controller)

        # Drawing step
        # First, clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.
        joystick_screen.fill((255, 255, 255))
        text_print.reset()

        # Get count of joysticks.
        joystick_count = pygame.joystick.get_count()

        text_print.tprint(joystick_screen, f"Number of joysticks: {joystick_count}")
        text_print.indent()

        # For each joystick, render the status of all joystick buttons onto the display
        for joystick in joysticks.values():
            render_joystick_status(joystick, text_print, joystick_screen, joystick_config)
            rate_control(joystick, controller, joystick_screen, joystick_config)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # Limit to 30 frames per second.
        clock.tick(30)


if __name__ == "__main__":
    main()
    # If you forget this line, the program will 'hang'
    # on exit if running from IDLE.
    pygame.quit()