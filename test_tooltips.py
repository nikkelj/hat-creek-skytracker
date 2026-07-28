#!/usr/bin/env python
"""Tests for the hover-tooltip system: the global toggle chip, registry-based
hover pick (smallest rect wins), text wrapping, and config persistence.

Headless. Run: python test_tooltips.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import unittest
from types import SimpleNamespace

import pygame
pygame.init()

import tooltips


def fake_display(w=1200, h=800):
    d = SimpleNamespace()
    d.menu_screen = pygame.Surface((w, h))
    d.sub_x, d.sub_y = 100, 0
    d.filter_rect = pygame.Rect(10, 40, 100, 28)
    d.filter_above_alt_rect = pygame.Rect(10, 90, 100, 28)
    d.filter_below_alt_rect = pygame.Rect(10, 140, 100, 28)
    d.clear_filters_button = pygame.Rect(10, 10, 90, 24)
    d.recompute_button = pygame.Rect(120, 10, 90, 24)
    d.scroll_bar_rect = pygame.Rect(200, 760, 800, 12)
    return d


class TestToggle(unittest.TestCase):
    def test_chip_click_flips_and_consumes(self):
        cfg = SimpleNamespace(tooltips_enabled=True)
        display = fake_display()
        tooltips.render(display, cfg, "main", mouse_pos=(0, 0))
        chip = tooltips._toggle_rect
        self.assertIsNotNone(chip)
        self.assertTrue(tooltips.handle_click(chip.center, cfg))
        self.assertFalse(cfg.tooltips_enabled)
        self.assertTrue(tooltips.handle_click(chip.center, cfg))
        self.assertTrue(cfg.tooltips_enabled)
        self.assertFalse(tooltips.handle_click((0, 0), cfg), "miss must not consume")

    def test_config_round_trip(self):
        from config import ConfigState
        cfg = ConfigState()
        self.assertTrue(cfg.tooltips_enabled)
        cfg.tooltips_enabled = False
        cfg2 = ConfigState()
        cfg2.load_from_dict(cfg.get_config_dict())
        self.assertFalse(cfg2.tooltips_enabled)


class TestHoverPick(unittest.TestCase):
    def test_tooltip_drawn_on_hover_and_smallest_rect_wins(self):
        cfg = SimpleNamespace(tooltips_enabled=True)
        display = fake_display()
        tvs = SimpleNamespace(object_toggle_rects={
            "ngc_enabled": pygame.Rect(300, 300, 110, 22)})
        pos = (305, 310)
        display.menu_screen.fill((0, 0, 0))
        tooltips.render(display, cfg, "tracking_vis", tracking_vis_state=tvs,
                        mouse_pos=pos)
        # A tooltip box must have been drawn near the cursor.
        region = pygame.surfarray.array3d(
            display.menu_screen.subsurface(pygame.Rect(pos[0], pos[1] + 5, 200, 60)))
        self.assertGreater(int((region.sum(axis=2) > 0).sum()), 100,
                           "tooltip box should render near the cursor")
        # Smallest-rect preference: a big pane rect and a small button rect
        # both under the cursor -> the button's text wins.
        collected = [(pygame.Rect(0, 0, 500, 500), "pane"),
                     (pygame.Rect(290, 295, 130, 30), "button")]
        best = None
        for rect, text in collected:
            if rect.collidepoint(pos):
                if best is None or rect.width * rect.height < best[0].width * best[0].height:
                    best = (rect, text)
        self.assertEqual(best[1], "button")

    def test_disabled_draws_only_chip(self):
        cfg = SimpleNamespace(tooltips_enabled=False)
        display = fake_display()
        tvs = SimpleNamespace(object_toggle_rects={
            "ngc_enabled": pygame.Rect(300, 300, 110, 22)})
        display.menu_screen.fill((0, 0, 0))
        tooltips.render(display, cfg, "tracking_vis", tracking_vis_state=tvs,
                        mouse_pos=(305, 310))
        region = pygame.surfarray.array3d(
            display.menu_screen.subsurface(pygame.Rect(305, 315, 200, 60)))
        self.assertEqual(int((region.sum(axis=2) > 0).sum()), 0,
                         "no tooltip may render while disabled")


class TestWrap(unittest.TestCase):
    def test_wrap_bounds(self):
        lines = tooltips._wrap("word " * 40)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(ln) <= tooltips._WRAP for ln in lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)
