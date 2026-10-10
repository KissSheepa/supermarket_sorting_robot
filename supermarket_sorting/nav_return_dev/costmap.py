# -*- coding: utf-8 -*-
"""Rolling 8m x 8m occupancy costmap with exact LiDAR->world projection and
gradient soft-inflation.

The deadly (lethal) band is the physical robot half-width (ROBOT_HALF_WIDTH),
cost = LETHAL_COST(253). Between SOFT_BAND_START and INFLATION_RADIUS the cost
decays from 253 to 0, forming a graduated soft standoff. A* is allowed to pass
through any cell with cost < 253, so narrow gaps remain solvable.
"""
import math
import numpy as np

from nav_return_dev.config import NavConfig
from nav_return_dev.footprint import rectangle_footprint_offsets, transform_footprint


def _world_to_cell(x, y, origin_x, origin_y, res):
    return int(round((x - origin_x) / res)), int(round((y - origin_y) / res))


def _in_bounds(cx, cy, n):
    return 0 <= cx < n and 0 <= cy < n


class RollingCostmap:
    def __init__(self, cfg: NavConfig):
        self.cfg = cfg
        self.res = cfg.MAP_RESOLUTION
        self.size_x = int(round(cfg.MAP_SIZE_X / self.res))   # 160
        self.size_y = int(round(cfg.MAP_SIZE_Y / self.res))   # 160
        self.grid = np.zeros((self.size_y, self.size_x), dtype=np.float32)
        self.lethal = np.zeros((self.size_y, self.size_x), dtype=bool)
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.max_range = cfg.SCAN_MAX

    # ------------------------------------------------------------------
    @staticmethod
    def _raycast_free(grid, x0, y0, x1, y1):
        """Bresenham: mark cells strictly between (x0,y0) and (x1,y1)."""
        n = grid.shape[0]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0); dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        cx, cy = x0, y0
        guard = 0
        while True:
            if cx == x1 and cy == y1:
                break
            guard += 1
            if guard > 100000:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy; cx += sx
            if e2 <= dx:
                err += dx; cy += sy
            if cx == x1 and cy == y1:
                break
            if 0 <= cx < n and 0 <= cy < n:
                grid[cy, cx] += 1.0

    # ------------------------------------------------------------------
    def update(self, robot_pose, scan_ranges, angle_min, angle_inc):
        cfg = self.cfg
        xr, yr, yaw = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
        new_origin_x = xr - cfg.MAP_SIZE_X / 2.0
        new_origin_y = yr - cfg.MAP_SIZE_Y / 2.0
        self._shift_grid_to_origin(new_origin_x, new_origin_y)

        # persistent occupancy: decay, never whole-clear
        self.grid *= cfg.OCC_DECAY

        r = np.asarray(scan_ranges, dtype=np.float64)
        valid = np.isfinite(r) & (r > cfg.SCAN_MIN)
        hit = valid & (r < self.max_range)          # real echo

        idx = np.flatnonzero(valid)
        if idx.size == 0:
            np.clip(self.grid, 0.0, cfg.LETHAL_COST * 3.0, out=self.grid)
            self.lethal[:] = self.grid >= cfg.LETHAL_COST
            return

        dist = r[idx]
        ang = angle_min + idx.astype(np.float64) * angle_inc
        cos_t, sin_t = np.cos(yaw), np.sin(yaw)
        lx = dist * np.cos(ang)
        ly = dist * np.sin(ang)
        wx = xr + cos_t * lx - sin_t * ly
        wy = yr + sin_t * lx + cos_t * ly
        cx = np.round((wx - self.origin_x) / self.res).astype(np.int32)
        cy = np.round((wy - self.origin_y) / self.res).astype(np.int32)

        rx = int(round((xr - self.origin_x) / self.res))
        ry = int(round((yr - self.origin_y) / self.res))
        n = self.size_y

        # ---- ray-cast free space (clear what rays pass through) ----
        # For a max-range (no-return) ray, clamp the clear length to
        # FREE_RAY_CLEAR_DIST so a real wall sitting exactly AT the sensor's max
        # range is not erroneously wiped out as free space.
        free_mask = np.zeros_like(self.grid)
        for i in range(idx.size):
            if hit[idx[i]]:
                # real echo trace free up to (but excluding) the hit endpoint
                ex = int(np.clip(cx[i], 0, self.size_x - 1))
                ey = int(np.clip(cy[i], 0, n - 1))
            else:
                # no-return ray: clamp endpoint to FREE_RAY_CLEAR_DIST
                d_clamp = min(float(dist[i]), cfg.FREE_RAY_CLEAR_DIST)
                a = ang[i]
                lxi = d_clamp * np.cos(a)
                lyi = d_clamp * np.sin(a)
                wxi = xr + cos_t * lxi - sin_t * lyi
                wyi = yr + sin_t * lxi + cos_t * lyi
                ex = int(np.clip(np.round((wxi - self.origin_x) / self.res), 0, self.size_x - 1))
                ey = int(np.clip(np.round((wyi - self.origin_y) / self.res), 0, n - 1))
            self._raycast_free(free_mask, rx, ry, ex, ey)
        clear = (free_mask > 0.0) & (self.grid > 0.0)
        if clear.any():
            self.grid[clear] *= (cfg.OCC_DECAY * cfg.OCC_DECAY)

        # ---- write hits ----
        hit_i = np.flatnonzero(hit)
        if hit_i.size:
            hx = cx[hit_i].astype(np.int32)
            hy = cy[hit_i].astype(np.int32)
            inb = (hx >= 0) & (hx < self.size_x) & (hy >= 0) & (hy < n)
            hx, hy = hx[inb], hy[inb]
            if hx.size:
                np.add.at(self.grid, (hy, hx), cfg.OCC_INCR)

        np.clip(self.grid, 0.0, cfg.LETHAL_COST * 3.0, out=self.grid)
        self.lethal[:] = self.grid >= cfg.LETHAL_COST

    def _shift_grid_to_origin(self, new_origin_x, new_origin_y):
        shift_x = int(round((new_origin_x - self.origin_x) / self.res))
        shift_y = int(round((new_origin_y - self.origin_y) / self.res))
        if shift_x == 0 and shift_y == 0:
            return
        if abs(shift_x) >= self.size_x or abs(shift_y) >= self.size_y:
            self.grid.fill(0.0)
        else:
            if shift_x:
                self.grid = np.roll(self.grid, -shift_x, axis=1)
                if shift_x > 0:
                    self.grid[:, -shift_x:] = 0.0
                else:
                    self.grid[:, :-shift_x] = 0.0
            if shift_y:
                self.grid = np.roll(self.grid, -shift_y, axis=0)
                if shift_y > 0:
                    self.grid[-shift_y:, :] = 0.0
                else:
                    self.grid[:-shift_y, :] = 0.0
        self.origin_x = new_origin_x
        self.origin_y = new_origin_y

    # ------------------------------------------------------------------
    def normalized_cost(self):
        return self.inflate()

    def inflate(self):
        """Gradient soft inflation (vectorized distance transform).

        - distance < SOFT_BAND_START  => cost 253 (lethal)
        - SOFT_BAND_START..INFLATION_RADIUS => linear decay 253 -> 0
        - beyond INFLATION_RADIUS => 0
        """
        cfg = self.cfg
        if not self.lethal.any():
            return np.zeros_like(self.grid, dtype=np.float32)

        dist_m = _signed_distance_to_lethal(self.lethal) * self.res  # meters
        start = cfg.SOFT_BAND_START
        radius = cfg.INFLATION_RADIUS
        # inside hard band => full cost
        hard = dist_m <= start
        # soft band => linear ramp from LETHAL_COST to 0
        span = max(radius - start, 1e-6)
        ramp = np.clip(1.0 - (dist_m - start) / span, 0.0, 1.0)
        cost = ramp * float(cfg.LETHAL_COST)
        cost[hard] = float(cfg.LETHAL_COST)
        # cells that are lethal (occupied) always full cost
        cost[self.lethal] = float(cfg.LETHAL_COST)
        return cost.astype(np.float32)

    # ------------------------------------------------------------------
    def world_to_cell(self, x, y):
        return _world_to_cell(x, y, self.origin_x, self.origin_y, self.res)

    def cell_in_bounds(self, cx, cy):
        return _in_bounds(cx, cy, self.size_y)

    def cell_center(self, cx, cy):
        return (cx * self.res + self.origin_x, cy * self.res + self.origin_y)

    def query_cost_xy(self, x, y):
        cost = self.inflate()
        fx = (x - self.origin_x) / self.res
        fy = (y - self.origin_y) / self.res
        x0 = int(np.floor(fx)); y0 = int(np.floor(fy))
        if not (0 <= x0 < self.size_x - 1 and 0 <= y0 < self.size_y - 1):
            cx = int(round(fx)); cy = int(round(fy))
            if _in_bounds(cx, cy, self.size_y):
                return float(cost[cy, cx])
            return 0.0
        tx = fx - x0; ty = fy - y0
        c00 = cost[y0, x0]; c01 = cost[y0, x0 + 1]
        c10 = cost[y0 + 1, x0]; c11 = cost[y0 + 1, x0 + 1]
        return float((c00 * (1 - tx) + c01 * tx) * (1 - ty) +
                     (c10 * (1 - tx) + c11 * tx) * ty)

    def footprint_is_clear(self, x, y, yaw, offsets=None):
        """Check the physical body samples against raw LiDAR obstacle cells."""
        if offsets is None:
            offsets = rectangle_footprint_offsets(self.cfg)
        px, py = transform_footprint(float(x), float(y), float(yaw), offsets)
        cx = np.round((px - self.origin_x) / self.res).astype(np.int32)
        cy = np.round((py - self.origin_y) / self.res).astype(np.int32)
        out_of_bounds = ((cx < 0) | (cx >= self.size_x) |
                         (cy < 0) | (cy >= self.size_y))
        if out_of_bounds.any():
            return False
        return not bool(np.any(self.lethal[cy, cx]))

    def footprint_clearance(self, x, y, yaw, offsets=None):
        """Return minimum raw-obstacle clearance at sampled body points."""
        if offsets is None:
            offsets = rectangle_footprint_offsets(self.cfg)
        px, py = transform_footprint(float(x), float(y), float(yaw), offsets)
        cx = np.round((px - self.origin_x) / self.res).astype(np.int32)
        cy = np.round((py - self.origin_y) / self.res).astype(np.int32)
        out_of_bounds = ((cx < 0) | (cx >= self.size_x) |
                         (cy < 0) | (cy >= self.size_y))
        if out_of_bounds.any():
            return 0.0
        if not self.lethal.any():
            return float("inf")
        dist = _signed_distance_to_lethal(self.lethal) * self.res
        return float(np.min(dist[cy, cx]))


def _signed_distance_to_lethal(lethal):
    """Two-pass chamfer distance (cells) from nearest lethal cell (vectorized)."""
    import numpy as _np
    n, m = lethal.shape
    INF = _np.float32(1e6)
    d = _np.where(lethal, 0.0, INF).astype(_np.float32)

    for i in range(n):
        up = d[i-1] if i > 0 else None
        row = d[i]
        left = _np.concatenate([[INF], row[:-1]])
        if up is not None:
            ul = _np.concatenate([[INF], up[:-1]])
            ur = _np.concatenate([up[1:], [INF]])
            d[i] = _np.minimum(_np.minimum(row, left + 1.0),
                               _np.minimum(ul + 1.41421356, ur + 1.41421356))
            d[i] = _np.minimum(d[i], up + 1.0)
        else:
            d[i] = _np.minimum(row, left + 1.0)

    for i in range(n-1, -1, -1):
        dn = d[i+1] if i < n-1 else None
        row = d[i]
        right = _np.concatenate([row[1:], [INF]])
        if dn is not None:
            dl = _np.concatenate([[INF], dn[:-1]])
            dr = _np.concatenate([dn[1:], [INF]])
            d[i] = _np.minimum(_np.minimum(row, right + 1.0),
                               _np.minimum(dl + 1.41421356, dr + 1.41421356))
            d[i] = _np.minimum(d[i], dn + 1.0)
        else:
            d[i] = _np.minimum(row, right + 1.0)

    d[d >= INF] = INF
    return d
