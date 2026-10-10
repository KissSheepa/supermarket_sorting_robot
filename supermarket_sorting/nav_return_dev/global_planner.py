# -*- coding: utf-8 -*-
"""A* global planner over the soft-inflated costmap.

A* is allowed to traverse any cell with cost < LETHAL_COST(253); it only
refuses the truly lethal band (>=253). The cost of traversing a soft cell is
its inflated cost, so A* naturally prefers to stay on the centerline while
still being able to route through a tight gap that is inside the soft band.
"""
import heapq
import math
from typing import Optional, Tuple

import numpy as np

from nav_return_dev.config import NavConfig
from nav_return_dev.costmap import _signed_distance_to_lethal


class GlobalPathPlanner:
    def __init__(self, cfg: NavConfig):
        self.cfg = cfg
        self.last_diagnostics = {}

    # ------------------------------------------------------------------
    def plan(self, start_pose, goal_pose, costmap,
             avoid_regions=None) -> Optional[np.ndarray]:
        cfg = self.cfg
        cost = costmap.inflate()              # [0,253]
        blocked = cost >= cfg.LETHAL_COST     # only truly lethal is blocked
        self.last_diagnostics = {
            "astar_avoid_cells": 0,
            "astar_path_min_clearance": 0.0,
        }
        self._apply_avoid_regions(blocked, costmap, avoid_regions)
        # Distance (in cells) from each cell to the nearest real lethal obstacle.
        # Remains a soft centerline preference; hard collision is enforced by
        # MPPI with the true rectangular footprint.
        _clear_cells = (_signed_distance_to_lethal(costmap.lethal) *
                        cfg.MAP_RESOLUTION)

        # Keep the local goal inside the rolling 8m map with enough room for the
        # full chassis. GOAL_RAY_CLAMP alone may exceed the map half-width.
        local_goal_clamp = min(
            cfg.GOAL_RAY_CLAMP,
            cfg.MAP_SIZE_X / 2.0 - cfg.LOCAL_GOAL_MAP_MARGIN,
            cfg.MAP_SIZE_Y / 2.0 - cfg.LOCAL_GOAL_MAP_MARGIN,
        )
        goal_distance = math.hypot(float(goal_pose[0]) - float(start_pose[0]),
                                   float(goal_pose[1]) - float(start_pose[1]))
        self.last_diagnostics["astar_effective_goal_clamp"] = float(local_goal_clamp)
        self.last_diagnostics["astar_goal_was_clamped"] = int(
            goal_distance > local_goal_clamp + 1e-6)
        goal_pose = self._clamp_goal_to_window(start_pose, goal_pose,
                                               local_goal_clamp)

        n = blocked.shape[0]
        start_cell = costmap.world_to_cell(start_pose[0], start_pose[1])
        goal_cell = costmap.world_to_cell(goal_pose[0], goal_pose[1])

        # free the start/goal if they land inside the soft/lethal band
        # direction for goal snapping: prefer free cells toward the real goal,
        # never snap hundreds of cm sideways/backwards into the shelf rows.
        dgx = float(goal_pose[0]) - float(start_pose[0])
        dgy = float(goal_pose[1]) - float(start_pose[1])
        start_cell = self._free_or_softest(start_cell, cost, blocked,
                                           (dgx, dgy), prefer_positive=True)
        goal_cell = self._free_or_softest(goal_cell, cost, blocked,
                                          (dgx, dgy), prefer_positive=True)
        if start_cell is None or goal_cell is None:
            return None

        path_cells = self._astar(start_cell, goal_cell, cost, blocked,
                                 _clear_cells)
        if path_cells is None:
            return None

        if len(path_cells):
            self.last_diagnostics["astar_path_min_clearance"] = float(
                min(_clear_cells[cy, cx] for cx, cy in path_cells))

        pts = [costmap.cell_center(cx, cy) for cx, cy in path_cells]
        return self._resample(pts)

    # ------------------------------------------------------------------
    def _apply_avoid_regions(self, blocked, costmap, avoid_regions):
        if not avoid_regions:
            return
        yy, xx = np.indices(blocked.shape)
        for x, y, radius in avoid_regions:
            cx, cy = costmap.world_to_cell(x, y)
            radius_cells = max(1, int(math.ceil(float(radius) / costmap.res)))
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius_cells ** 2
            newly_blocked = mask & ~blocked
            self.last_diagnostics["astar_avoid_cells"] += int(np.count_nonzero(newly_blocked))
            blocked[mask] = True

    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_goal_to_window(start_pose, goal_pose, clamp_dist):
        """Clamp goal to clamp_dist meters from the robot along the exact
        start->goal direction vector.  If already within clamp_dist, return as
        is.  Preserves the goal yaw (used only for arrival check upstream)."""
        sx, sy = start_pose[0], start_pose[1]
        gx, gy = goal_pose[0], goal_pose[1]
        dx = gx - sx; dy = gy - sy
        d = math.hypot(dx, dy)
        if d <= clamp_dist or d < 1e-9:
            return goal_pose
        f = clamp_dist / d
        new_g = (sx + dx * f, sy + dy * f)
        return (new_g[0], new_g[1], goal_pose[2])

    # ------------------------------------------------------------------
    def _free_or_softest(self, cell, cost, blocked, direction=None,
                          prefer_positive=True):
        """Find the nearest non-blocked cell.

        If the exact cell is free, return it. Otherwise search outward rings.
        When a goal point is blocked, prefer candidates lying in the direction
        from robot to the real goal so the planner does not snap sideways/back
        into shelf rows or the wrong half-plane.
        """
        n = cost.shape[0]
        cx, cy = cell
        if 0 <= cx < n and 0 <= cy < n and not blocked[cy, cx]:
            return (cx, cy)

        if direction is not None:
            dir_x, dir_y = float(direction[0]), float(direction[1])
            dlen = math.hypot(dir_x, dir_y)
            if dlen > 1e-9:
                dir_x /= dlen
                dir_y /= dlen
            else:
                direction = None

        best = None
        best_score = float('inf')
        for radius in range(1, n):
            found_any = False
            ring_best = None
            ring_score = float('inf')
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < n and 0 <= ny < n and not blocked[ny, nx]:
                        found_any = True
                        # cost term
                        cscore = float(cost[ny, nx])
                        # direction term: favor cells along the goal direction
                        ndir = 0.0
                        if direction is not None:
                            dd = math.hypot(dx, dy)
                            if dd > 1e-9:
                                ndir = (dx / dd) * direction[0] + (dy / dd) * direction[1]
                            if prefer_positive and ndir < -0.3:
                                # strongly discourage moving opposite the goal
                                ndir = -3.0
                        score = cscore + (-20.0 * max(0.0, ndir))
                        if score < ring_score:
                            ring_score = score
                            ring_best = (nx, ny)
            if found_any and ring_best is not None:
                # per-ring: if it has a valid candidate, return this ring's best
                # immediately to keep snapping local.
                return ring_best
        return best

    # ------------------------------------------------------------------
    def _astar(self, start_cell, goal_cell, cost, blocked, clear_cells):
        for key in ("astar_avoid_cells", "astar_path_min_clearance"):
            self.last_diagnostics.setdefault(key, 0)
        n = cost.shape[0]
        sx, sy = start_cell
        gx, gy = goal_cell
        cfg = self.cfg
        clear_goal = getattr(cfg, "A_STAR_CLEARANCE_GOAL", 0.30)
        narrow_w = getattr(cfg, "A_STAR_NARROW_WEIGHT", 5.0)

        def h(cx, cy):
            return math.hypot(cx - gx, cy - gy)

        open_heap = [(h(sx, sy), 0.0, sx, sy)]
        came = {}
        gscore = {(sx, sy): 0.0}
        closed = set()
        while open_heap:
            f, g, cx, cy = heapq.heappop(open_heap)
            if (cx, cy) in closed:
                continue
            closed.add((cx, cy))
            if (cx, cy) == (gx, gy):
                path = [(gx, gy)]
                cur = (gx, gy)
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < n and 0 <= ny < n):
                    continue
                if blocked[ny, nx] or (nx, ny) in closed:
                    continue
                step = 1.41421356 if dx != 0 and dy != 0 else 1.0
                # traversal cost = step + scaled inflated cost: prefer low cost
                # cells (centerline) but allow soft cells for narrow gaps.
                inflate_w = getattr(cfg, "A_STAR_INFLATE_WEIGHT", 3.0)
                extra = inflate_w * float(cost[ny, nx]) / 253.0
                # Narrow-passage penalty: if clearance to the nearest obstacle is
                # below the preferred per-side clearance, add cost proportional
                # to the shortfall. This biases A* toward wider lanes (e.g. the
                # robot's left corridor) when a narrower gap cannot fit
                # the real MPPI footprint.
                _clr = float(clear_cells[ny, nx])
                if _clr < clear_goal:
                    extra += narrow_w * (clear_goal - _clr)
                ng = g + step + extra
                if ng < gscore.get((nx, ny), float("inf")):
                    gscore[(nx, ny)] = ng
                    came[(nx, ny)] = (cx, cy)
                    heapq.heappush(open_heap, (ng + h(nx, ny), ng, nx, ny))
        return None

    # ------------------------------------------------------------------
    def _resample(self, pts):
        cfg = self.cfg
        ds = cfg.PATH_RESAMPLE_DS
        if len(pts) < 2:
            p = pts[0]
            return np.array([[p[0], p[1], 0.0]], dtype=np.float64)

        segs = np.diff(np.asarray(pts, dtype=np.float64), axis=0)
        seg_len = np.hypot(segs[:, 0], segs[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1]

        out = []
        s = 0.0
        while s <= total + 1e-9:
            i = np.searchsorted(cum, s, side='right') - 1
            i = max(0, min(i, len(pts) - 2))
            seg_s = s - cum[i]
            seg_l = seg_len[i]
            if seg_l < 1e-12:
                x, y = pts[i]
            else:
                frac = seg_s / seg_l
                x = pts[i][0] + frac * (pts[i+1][0] - pts[i][0])
                y = pts[i][1] + frac * (pts[i+1][1] - pts[i][1])
            out.append((x, y))
            s += ds

        gx, gy = pts[-1]
        out.append((gx, gy))

        ref = []
        for idx in range(len(out)):
            if idx < len(out) - 1:
                dx = out[idx+1][0] - out[idx][0]
                dy = out[idx+1][1] - out[idx][1]
            else:
                dx = out[idx][0] - out[idx-1][0]
                dy = out[idx][1] - out[idx-1][1]
            ref.append((out[idx][0], out[idx][1], math.atan2(dy, dx)))

        return np.asarray(ref, dtype=np.float64)
