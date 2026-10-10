# -*- coding: utf-8 -*-
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


class RollingCostmapTest(unittest.TestCase):
    def test_grid_shift_keeps_world_anchored(self):
        costmap = RollingCostmap(NavConfig())
        costmap.origin_x = -4.0
        costmap.origin_y = -4.0
        costmap.grid[120, 80] = 300.0

        costmap._shift_grid_to_origin(-3.9, -1.9)

        self.assertAlmostEqual(costmap.origin_x, -3.9)
        self.assertAlmostEqual(costmap.origin_y, -1.9)
        self.assertEqual(costmap.grid[78, 78], 300.0)
        self.assertEqual(costmap.grid[120, 80], 0.0)

    def test_footprint_rejects_center_free_box_contact(self):
        costmap = RollingCostmap(NavConfig())
        costmap._shift_grid_to_origin(-4.0, -4.0)
        box_cell = costmap.world_to_cell(0.20, 0.15)
        costmap.lethal[box_cell[1], box_cell[0]] = True

        self.assertFalse(costmap.footprint_is_clear(0.0, 0.0, 0.0))

    def test_backup_narrow_band_remains_soft_at_fifteen_centimeters(self):
        cfg = NavConfig()
        costmap = RollingCostmap(cfg)
        costmap.lethal[80, 80] = True

        cost = costmap.inflate()

        self.assertGreater(cost[80, 83], 0.0)
        self.assertLess(cost[80, 83], cfg.LETHAL_COST)

    def test_astar_allows_diagonal_around_blocked_corners(self):
        cfg = NavConfig()
        planner = GlobalPathPlanner(cfg)
        costmap = RollingCostmap(cfg)
        costmap._shift_grid_to_origin(-4.0, -4.0)

        start = (80, 80)
        goal = (81, 81)
        cost = np.zeros_like(costmap.grid)
        blocked = np.zeros_like(costmap.lethal)
        blocked[80, 81] = True
        blocked[81, 80] = True
        clearance = np.full_like(cost, 1.0)

        path = planner._astar(start, goal, cost, blocked, clearance)

        self.assertIsNotNone(path)
        self.assertEqual(path, [start, goal])


if __name__ == "__main__":
    unittest.main()
