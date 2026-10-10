# -*- coding: utf-8 -*-
"""Smoke tests for the current NAV+MPPI module (pure A* + MPPI).

These tests intentionally stay small and deterministic.  They verify:
  - the A* global planner returns a usable reference on a free map;
  - the MPPI controller emits one finite command within control limits;
  - the project-root log/visual paths work from any cwd.

Run:
  python tests/test_nav_mppi_smoke.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from nav_return_dev.config import NavConfig
from nav_return_dev.costmap import RollingCostmap
from nav_return_dev.global_planner import GlobalPathPlanner
from nav_return_dev.mppi_controller import MPPIController


class NavMppiSmokeTest(unittest.TestCase):
    def setUp(self):
        self.cfg = NavConfig()
        self.costmap = RollingCostmap(self.cfg)
        self.costmap._shift_grid_to_origin(-4.0, -4.0)
        self.planner = GlobalPathPlanner(self.cfg)
        self.mppi = MPPIController(self.cfg)

    def test_astar_free_map_plans(self):
        pose = (0.0, 0.0, 0.0)
        goal = (1.5, 0.0, 0.0)
        ref = self.planner.plan(pose, goal, self.costmap)
        self.assertIsNotNone(ref)
        self.assertGreaterEqual(len(ref), 2)
        end = ref[-1]
        self.assertAlmostEqual(end[0], 1.5, delta=0.2)
        self.assertAlmostEqual(end[1], 0.0, delta=0.2)

    def test_astar_far_goal_stays_inside_rolling_map(self):
        pose = (0.0, 0.0, 0.0)
        goal = (0.0, -5.4, 0.0)
        ref = self.planner.plan(pose, goal, self.costmap)
        self.assertIsNotNone(ref)
        local_limit = self.cfg.MAP_SIZE_Y / 2.0 - self.cfg.LOCAL_GOAL_MAP_MARGIN
        self.assertLessEqual(np.hypot(ref[-1, 0] - pose[0],
                                     ref[-1, 1] - pose[1]), local_limit + 0.1)

    def test_mppi_free_map_command_limits(self):
        pose = (0.0, 0.0, 0.0)
        goal = (1.5, 0.0, 0.0)
        ref = self.planner.plan(pose, goal, self.costmap)
        self.assertIsNotNone(ref)
        v, w, meta = self.mppi.compute_velocity_command(pose, ref, self.costmap)
        self.assertTrue(np.isfinite(v))
        self.assertTrue(np.isfinite(w))
        self.assertGreaterEqual(v, self.cfg.MIN_VX - 1e-6)
        self.assertLessEqual(v, self.cfg.MAX_VX + 1e-6)
        self.assertLessEqual(abs(w), self.cfg.MAX_WZ + 1e-6)
        self.assertIn("S_min", meta)

    def test_nav_root_paths(self):
        from nav_return_dev.navigator import _project_root
        root = _project_root()
        self.assertTrue((root / "nav_return_dev" / "config.py").exists())
        self.assertTrue((root / "nav_return_dev" / "visual").parent.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
