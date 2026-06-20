#!/usr/bin/env python
"""
Headless tests for the HW Sim UI panel logic (hw_sim_ui.py): the draw pass lays
out control rects, and clicks toggle the enable flag and step parameter values
with the right clamping/integer rules. No real window required.

Run: python test_hw_sim_ui.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import unittest

import pygame
pygame.init()
pygame.font.init()

from config import ConfigState
from hw_sim_ui import draw_hw_sim_options, handle_hw_sim_click


class _Display:
    def __init__(self):
        self.menu_screen = pygame.Surface((1600, 1000))
        self.sub_x, self.sub_y = 300, 0
        self.sub_width, self.sub_height = 1300, 1000
        f = pygame.font.SysFont(None, 20)
        self.small_font = self.font = self.large_font = f


def _center(rect):
    return (rect.x + rect.width // 2, rect.y + rect.height // 2)


class HwSimUiTests(unittest.TestCase):

    def setUp(self):
        self.display = _Display()
        self.cfg = ConfigState()
        self.msgs = []
        draw_hw_sim_options(self.display, self.cfg, None)  # lays out rects

    def test_enable_toggle(self):
        self.assertFalse(self.cfg.sim_config["enabled"])
        handle_hw_sim_click(_center(self.display.hw_sim_enable_rect), self.display,
                            self.cfg, None, self.msgs)
        self.assertTrue(self.cfg.sim_config["enabled"])
        handle_hw_sim_click(_center(self.display.hw_sim_enable_rect), self.display,
                            self.cfg, None, self.msgs)
        self.assertFalse(self.cfg.sim_config["enabled"])

    def test_stepper_increments(self):
        minus, plus, step = self.display.hw_sim_rects["mount_misalignment_az_deg"]
        before = self.cfg.sim_config["mount_misalignment_az_deg"]
        handle_hw_sim_click(_center(plus), self.display, self.cfg, None, self.msgs)
        self.assertAlmostEqual(self.cfg.sim_config["mount_misalignment_az_deg"], before + step)
        handle_hw_sim_click(_center(minus), self.display, self.cfg, None, self.msgs)
        self.assertAlmostEqual(self.cfg.sim_config["mount_misalignment_az_deg"], before)

    def test_non_negative_clamp(self):
        self.cfg.sim_config["read_noise"] = 0.0
        minus, plus, step = self.display.hw_sim_rects["read_noise"]
        handle_hw_sim_click(_center(minus), self.display, self.cfg, None, self.msgs)
        self.assertEqual(self.cfg.sim_config["read_noise"], 0.0)  # clamped at 0

    def test_integer_key_stays_int(self):
        minus, plus, step = self.display.hw_sim_rects["star_density"]
        handle_hw_sim_click(_center(plus), self.display, self.cfg, None, self.msgs)
        self.assertIsInstance(self.cfg.sim_config["star_density"], int)

    def test_click_miss_returns_false(self):
        self.assertFalse(handle_hw_sim_click((5, 5), self.display, self.cfg, None, self.msgs))

    def test_rust_loop_toggle_auto_saves(self):
        saves = []
        self.cfg.save_to_file = lambda *a, **k: saves.append(True)  # don't touch disk
        self.assertFalse(self.cfg.use_rust_core_loop)
        handle_hw_sim_click(_center(self.display.hw_sim_rust_loop_rect), self.display,
                            self.cfg, None, self.msgs)
        self.assertTrue(self.cfg.use_rust_core_loop)
        self.assertTrue(saves, "toggle should auto-save so it persists for restart")
        handle_hw_sim_click(_center(self.display.hw_sim_rust_loop_rect), self.display,
                            self.cfg, None, self.msgs)
        self.assertFalse(self.cfg.use_rust_core_loop)

    def test_rust_loop_flag_round_trips(self):
        self.cfg.use_rust_core_loop = True
        self.cfg.rust_core_loop_hz = 20
        loaded = ConfigState()
        loaded.load_from_dict(self.cfg.get_config_dict())
        self.assertTrue(loaded.use_rust_core_loop)
        self.assertEqual(loaded.rust_core_loop_hz, 20)


if __name__ == '__main__':
    unittest.main(verbosity=2)
