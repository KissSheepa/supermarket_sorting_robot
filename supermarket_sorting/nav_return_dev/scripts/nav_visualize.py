# -*- coding: utf-8 -*-
"""God-view visualization for NAV + MPPI.

Pure diagnostic top-down renderer based on live local costmap, live A* path,
robot pose and goal(s). It does NOT read known shelf/box/wall layout from the
server; everything is generated from the rolling occupancy costmap that is
built online from the current LiDAR scan.

Expected usage (inside nav/navigator.py):
    from nav_return_dev.scripts.nav_visualize import save_visualization
    save_visualization(navigator, robot_pose, goal_pose)

Image contents:
  - grey/white background with 1m world grid (coordinate grid only)
  - soft-inflated rolling costmap (dark = obstacle, light = free)
  - A* full reference / forward window (yellow)
  - robot pose (red heading triangle)
  - final delivery goal (green circle + star)
  - clamped local A* goal (orange circle/plus)
  - temporary failed-corridor blocks (purple circles)
  - costmap 8m x 8m boundary (blue frame)
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False


# ---------------------------------------------------------------- paths
def _project_root():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(here))
    except Exception:
        return os.getcwd()


def _nav_visual_dir():
    return os.path.join(_project_root(), "nav_return_dev", "visual", "delivery")


# ---------------------------------------------------------------- costmap
def _cost_array(costmap):
    try:
        cost = costmap.inflate()
        return np.asarray(cost, dtype=np.float32)
    except Exception:
        try:
            return np.asarray(costmap.grid, dtype=np.float32)
        except Exception:
            return np.zeros((160, 160), dtype=np.float32)


def _cost_layers_rgb(costmap):
    """Render occupancy, body exclusion, and soft cost as separate layers."""
    cost = _cost_array(costmap)
    try:
        raw = np.asarray(costmap.lethal, dtype=bool)
        if raw.shape != cost.shape:
            raw = np.zeros_like(cost, dtype=bool)
    except Exception:
        raw = np.zeros_like(cost, dtype=bool)

    lethal_cost = float(getattr(getattr(costmap, "cfg", None),
                                 "LETHAL_COST", 253.0))
    hard = (cost >= lethal_cost) & ~raw
    soft = (cost > 0.0) & ~hard & ~raw
    strength = np.clip(cost / max(lethal_cost, 1.0), 0.0, 1.0)

    image = np.zeros((*cost.shape, 3), dtype=np.uint8)
    image[soft, 0] = (18.0 + 22.0 * strength[soft]).astype(np.uint8)
    image[soft, 1] = (28.0 + 34.0 * strength[soft]).astype(np.uint8)
    image[soft, 2] = (48.0 + 80.0 * strength[soft]).astype(np.uint8)
    image[hard] = (105, 105, 105)
    image[raw] = (230, 65, 65)
    return np.flipud(image)


# ---------------------------------------------------------------- helpers
def _w2pix(x, y, ox, oy, res, dpi, height_m=8.0):
    u = int(round((x - ox) / res)) * dpi
    v = int(round((oy + height_m - y) / res)) * dpi
    return u, v


def _pix_to_px2(px, py):
    return float(px), float(py)


def _nearest_path_window(ref, pose, window_m=2.0):
    if ref is None or len(ref) < 2:
        return None
    ref = np.asarray(ref, dtype=np.float64)
    d = np.hypot(ref[:, 0] - float(pose[0]), ref[:, 1] - float(pose[1]))
    i0 = int(np.argmin(d))
    out = [ref[i0]]
    acc = 0.0
    for i in range(i0 + 1, len(ref)):
        seg = float(np.hypot(ref[i][0] - ref[i - 1][0],
                             ref[i][1] - ref[i - 1][1]))
        acc += seg
        if acc > window_m:
            out.append(ref[i])
            break
        out.append(ref[i])
    return np.asarray(out, dtype=np.float64)


def _draw_robot(draw, pose, ox, oy, res, dpi, color=(255, 0, 0)):
    px, py = float(pose[0]), float(pose[1])
    yaw = float(pose[2])
    hlen = 0.26
    wid = 0.11
    hx = px + hlen * math.cos(yaw)
    hy = py + hlen * math.sin(yaw)
    lax = px + wid * math.cos(yaw + math.pi / 2)
    lay = py + wid * math.sin(yaw + math.pi / 2)
    rax = px + wid * math.cos(yaw - math.pi / 2)
    ray = py + wid * math.sin(yaw - math.pi / 2)
    pts = [_w2pix(hx, hy, ox, oy, res, dpi),
           _w2pix(lax, lay, ox, oy, res, dpi),
           _w2pix(rax, ray, ox, oy, res, dpi)]
    draw.polygon(pts, fill=color)
    # robot body disk
    cx, cy = _w2pix(px, py, ox, oy, res, dpi)
    r = int(round(0.16 / res * dpi))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=max(2, int(dpi / 2)))


def _draw_goal(draw, goal, ox, oy, res, dpi, color=(0, 220, 0), label=None):
    gx, gy = float(goal[0]), float(goal[1])
    gu, gv = _w2pix(gx, gy, ox, oy, res, dpi)
    r = int(round(0.20 / res * dpi))
    rr = max(r, int(2 * dpi))
    draw.ellipse([gu - rr, gv - rr, gu + rr, gv + rr],
                 outline=color, width=max(2, int(1.4 * dpi)))
    # star shape
    s = int(round(0.08 / res * dpi))
    cx, cy = gu, gv
    pts = []
    for i in range(10):
        rad = s if i % 2 == 0 else s * 0.45
        ang = -math.pi / 2 + i * math.pi / 5
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    draw.polygon(pts, fill=color)
    if label:
        draw.text((gu + rr + 4, gv - rr - 4), label, fill=color)


def _draw_local_goal(draw, goal, ox, oy, res, dpi, color=(255, 140, 0), label="A* local goal"):
    gx, gy = float(goal[0]), float(goal[1])
    gu, gv = _w2pix(gx, gy, ox, oy, res, dpi)
    r = int(round(0.15 / res * dpi))
    rr = max(r, int(2 * dpi))
    draw.ellipse([gu - rr, gv - rr, gu + rr, gv + rr],
                 outline=color, width=max(2, int(1.2 * dpi)))
    # plus marker
    ln = int(round(0.10 / res * dpi))
    draw.line([gu - ln, gv, gu + ln, gv], fill=color, width=max(2, int(1.2 * dpi)))
    draw.line([gu, gv - ln, gu, gv + ln], fill=color, width=max(2, int(1.2 * dpi)))
    if label:
        draw.text((gu + rr + 4, gv - rr - 4), label, fill=color)


def _obstacle_clusters(costmap, min_cells=4):
    """Extract live obstacle clusters from the rolling lethal mask.

    Uses only real-time occupancy data stored in the costmap. This is a
    diagnostic helper; it never reads a static/real-world field layout.
    """
    try:
        lethal = np.asarray(getattr(costmap, "lethal", None), dtype=bool)
    except Exception:
        return []
    if lethal is None or lethal.size == 0:
        return []
    h, w = lethal.shape
    visited = np.zeros_like(lethal, dtype=bool)
    clusters = []
    for y in range(h):
        for x in range(w):
            if not lethal[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < h and 0 <= nx < w and lethal[ny,nx] and not visited[ny,nx]:
                        visited[ny,nx] = True
                        stack.append((ny,nx))
            if len(cells) >= min_cells:
                ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
                cy = int(np.mean(ys)); cx = int(np.mean(xs))
                clusters.append({
                    "cx": cx,
                    "cy": cy,
                    "x0": min(xs), "x1": max(xs),
                    "y0": min(ys), "y1": max(ys),
                    "n": len(cells),
                })
    return clusters


def _draw_obstacles(draw, clusters, ox, oy, res, dpi):
    """Draw integrated-occupancy cluster bounds for diagnostics."""
    for i, cl in enumerate(clusters):
        x0, y0 = cl["x0"], cl["y0"]
        x1, y1 = cl["x1"], cl["y1"]
        p0 = _w2pix(x0 * res + ox, y0 * res + oy, ox, oy, res, dpi)
        p1 = _w2pix((x1 + 1) * res + ox, (y1 + 1) * res + oy, ox, oy, res, dpi)
        if p1[1] < p0[1]:
            p0, p1 = (p0[0], p1[1]), (p1[0], p0[1])
        cx, cy = _w2pix(cl["cx"] * res + ox, cl["cy"] * res + oy, ox, oy, res, dpi)
        draw.rectangle([p0[0], p0[1], p1[0], p1[1]],
                       outline=(255, 145, 0), width=max(1, int(0.8 * dpi)))
        r = max(2, int(1.2 * dpi))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 145, 0))
        label = "O%d" % (i + 1)
        tx = min(max(cx + r + 2, 2), int((len(clusters) and p1[0]) or 2))
        draw.text((tx, cy - r - 2), label, fill=(255, 145, 0))


def _draw_current_scan(draw, navigator, pose, ox, oy, res, dpi):
    """Draw only the most recent LiDAR echoes, without affecting navigation."""
    try:
        scan = np.asarray(navigator._scan, dtype=np.float64)
        angle_min = float(navigator._angle_min)
        angle_inc = float(navigator._angle_inc)
        scan_max = float(navigator.cfg.SCAN_MAX)
        scan_min = float(navigator.cfg.SCAN_MIN)
    except Exception:
        return
    hit = np.isfinite(scan) & (scan > scan_min) & (scan < scan_max)
    if not hit.any():
        return
    angles = angle_min + np.flatnonzero(hit).astype(np.float64) * angle_inc
    ranges = scan[hit]
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    world_x = x + np.cos(yaw) * ranges * np.cos(angles) - np.sin(yaw) * ranges * np.sin(angles)
    world_y = y + np.sin(yaw) * ranges * np.cos(angles) + np.cos(yaw) * ranges * np.sin(angles)
    radius = max(1, int(0.35 * dpi))
    for px, py in zip(world_x, world_y):
        u, v = _w2pix(px, py, ox, oy, res, dpi)
        draw.ellipse([u - radius, v - radius, u + radius, v + radius],
                     fill=(80, 235, 255))


def _draw_avoid_regions(draw, navigator, ox, oy, res, dpi):
    """Show local regions rejected after an MPPI blocked-stop recovery."""
    for x, y, radius, _expiry in getattr(navigator, "_avoid_regions", []):
        cx, cy = _w2pix(float(x), float(y), ox, oy, res, dpi)
        pixels = max(2, int(round(float(radius) / res * dpi)))
        draw.ellipse([cx - pixels, cy - pixels, cx + pixels, cy + pixels],
                     outline=(190, 0, 255), width=max(2, int(dpi)))
        draw.text((cx + pixels + 2, cy - pixels - 2), "REPLAN", fill=(190, 0, 255))


def _draw_grid(draw, n_cells, dpi, cell_m=0.05, major_m=1.0):
    major = int(round(major_m / cell_m))
    for i in range(0, n_cells + 1, major):
        x = i * dpi
        w = max(1, int(dpi * 0.6))
        draw.line([(x, 0), (x, n_cells * dpi)], fill=(65, 65, 80), width=w)
        y = i * dpi
        draw.line([(0, y), (n_cells * dpi, y)], fill=(65, 65, 80), width=w)


def _draw_legend(draw, dpi):
    rows = [
        ((230, 65, 65), "integrated lidar occupancy"),
        ((80, 235, 255), "current lidar echo"),
        ((105, 105, 105), "body-exclusion band"),
        ((40, 55, 95), "soft clearance cost"),
    ]
    x0 = 8
    y0 = 28
    side = max(7, int(1.6 * dpi))
    gap = max(3, int(0.8 * dpi))
    for color, text in rows:
        draw.rectangle([x0, y0, x0 + side, y0 + side], fill=color)
        draw.text((x0 + side + gap, y0 - 1), text, fill=(235, 235, 235))
        y0 += side + gap


# ---------------------------------------------------------------- render
def render_visualization(navigator, robot_pose, goal_pose, title="",
                          out_dir=None):
    """Render a 1:1 top-down PNG using PIL. Returns path or None."""
    if not PIL_OK:
        return None
    costmap = getattr(navigator, "costmap", None)
    if costmap is None:
        return None

    cost = _cost_array(costmap)
    res = float(getattr(costmap, "res", 0.05))
    ox = float(getattr(costmap, "origin_x", 0.0))
    oy = float(getattr(costmap, "origin_y", 0.0))
    h, w = cost.shape
    nx, ny = int(round(w / res)), int(round(h / res))
    n_cells_x = int(round(w / res)) if res else int(w)
    n_cells_y = int(round(h / res)) if res else int(h)
    dpi = 5  # pixels per 0.05m cell -> 800px for 8m map

    layers = _cost_layers_rgb(costmap)
    img = Image.fromarray(layers, mode="RGB")
    img = img.resize((int(w) * dpi, int(h) * dpi), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    # 1m world grid
    _draw_grid(draw, n_cells_y, dpi, cell_m=res, major_m=1.0)

    # costmap boundary
    bx0, by0 = 0, 0
    bx1, by1 = int(w) * dpi, int(h) * dpi
    draw.rectangle([bx0, by0, bx1 - 1, by1 - 1], outline=(0, 60, 200),
                   width=max(2, int(0.8 * dpi)))

    # Live occupancy clusters supplement the raster layers above.
    clusters = _obstacle_clusters(costmap, min_cells=4)
    _draw_obstacles(draw, clusters, ox, oy, res, dpi)
    _draw_current_scan(draw, navigator, robot_pose, ox, oy, res, dpi)
    _draw_avoid_regions(draw, navigator, ox, oy, res, dpi)

    # A* path
    ref = getattr(navigator, "_ref", None)
    if ref is not None and len(ref) >= 2:
        full_xy = np.asarray(ref, dtype=np.float64)[:, :2]
        full_pts = [_w2pix(float(x), float(y), ox, oy, res, dpi)
                    for x, y in full_xy]
        draw.line(full_pts, fill=(255, 220, 0), width=max(2, int(1.5 * dpi)))
        win = _nearest_path_window(ref, robot_pose, 2.0)
        if win is not None and len(win) >= 2:
            win_pts = [_w2pix(float(pt[0]), float(pt[1]), ox, oy, res, dpi)
                       for pt in win]
            draw.line(win_pts, fill=(255, 255, 0), width=max(3, int(2.5 * dpi)))

    # local goal = end of A* ref (clamped within costmap)
    if ref is not None and len(ref):
        _draw_local_goal(draw, ref[-1], ox, oy, res, dpi)

    # ultimate delivery goal
    _draw_goal(draw, goal_pose, ox, oy, res, dpi, label="GOAL")

    # robot pose
    _draw_robot(draw, robot_pose, ox, oy, res, dpi, color=(255, 0, 0))

    _draw_legend(draw, dpi)

    # title
    if title:
        try:
            font = ImageFont.load_default()
            draw.text((8, 8), title, fill=(10, 10, 10), font=font)
        except Exception:
            draw.text((8, 8), title, fill=(10, 10, 10))

    out_path = None
    try:
        if out_dir is None:
            out_dir = _nav_visual_dir()
        os.makedirs(out_dir, exist_ok=True)
        px, py = float(robot_pose[0]), float(robot_pose[1])
        out_path = os.path.join(
            out_dir,
            "navview_%s_%.3f_%.3f.png" % (time.strftime("%Y%m%d_%H%M%S"), px, py))
        img.save(out_path)
    except Exception as exc:
        try:
            import os as _os
            _os.makedirs(os.path.join(_project_root(), "nav_return_dev", "logs"), exist_ok=True)
            with open(os.path.join(_project_root(), "nav_return_dev", "logs", "visual_error.log"), "a", encoding="utf-8") as f:
                f.write("render_exception %r\n" % (exc,))
        except Exception:
            pass
        out_path = None
    return out_path


# ---------------------------------------------------------------- public
def save_visualization(navigator, robot_pose, goal_pose, title="", out_dir=None):
    """Periodic snapshot helper. Returns path or None."""
    try:
        if out_dir is None:
            out_dir = _nav_visual_dir()
        os.makedirs(out_dir, exist_ok=True)
        path = render_visualization(navigator, robot_pose, goal_pose, title,
                                   out_dir=out_dir)
        if path is not None:
            return path
        try:
            os.makedirs(os.path.join(_project_root(), "nav_return_dev", "logs"), exist_ok=True)
            with open(os.path.join(_project_root(), "nav_return_dev", "logs", "visual_error.log"), "a", encoding="utf-8") as f:
                f.write("render_return_none costmap=%r ref=%r\n" % (
                    hasattr(navigator, "costmap"),
                    len(getattr(navigator, "_ref", []) or []),
                ))
        except Exception:
            pass
        return None
    except Exception:
        import traceback
        try:
            os.makedirs(os.path.join(_project_root(), "nav_return_dev", "logs"), exist_ok=True)
            with open(os.path.join(_project_root(), "nav_return_dev", "logs", "visual_error.log"), "a", encoding="utf-8") as f:
                f.write("save_exception traceback\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
        return None


if __name__ == "__main__":
    print("NAV visualization helper (PIL)")
    print("Use it inside nav/navigator.py; standalone mode needs a live navigator object.")
