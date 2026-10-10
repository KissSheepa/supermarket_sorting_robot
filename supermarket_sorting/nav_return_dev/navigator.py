# -*- coding: utf-8 -*-
"""Top-level NAV + MPPI driver (pure A* global + MPPI local, no hand rules).

Pipeline (20Hz):
    arrival -> costmap(8m soft-inflated) -> A* reference -> MPPI command

Normal avoidance and speed control emerge from A* topology plus MPPI. A
blocked-stop recovery can safely retreat or turn, then temporarily rejects the
failed local corridor before the next replan.
"""
import math
import time
from typing import Tuple

import numpy as np

from nav_return_dev.config import NavConfig
from nav_return_dev.costmap import RollingCostmap
from nav_return_dev.global_planner import GlobalPathPlanner
from nav_return_dev.mppi_controller import MPPIController

from pathlib import Path

# Diagnostic visualization is optional; a missing PIL/matplotlib must not break control.
try:
    from nav_return_dev.scripts.nav_visualize import save_visualization
except Exception:
    save_visualization = None


def _project_root() -> Path:
    """Return the baseline project root (parent of this nav/ package)."""
    here = Path(__file__).resolve()
    try:
        return here.parent.parent
    except Exception:
        return Path.cwd()


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class NavMppiNavigator:
    def __init__(self, cfg: NavConfig = None):
        self.cfg = cfg or NavConfig()
        self.costmap = RollingCostmap(self.cfg)
        self.planner = GlobalPathPlanner(self.cfg)
        self.mppi = MPPIController(self.cfg)
        self.last_cmd = (0.0, 0.0)
        self._ref = None
        self._scan_ready = False
        self.stuck_counter = 0      # consecutive near-zero MPPI frames
        self.unwedge_remaining = 0  # forced-reverse frames left (>0 = unwedging)
        self.recovery_turn_remaining = 0
        self.recovery_turn_sign = 0.0
        self._avoid_regions = []
        # --- return-to-shelf maneuver state ---
        self._return_active = False
        self._return_stage = ""
        self._return_backup_start = None
        self._return_open_goal = None
        self._return_final_goal = None
        self._visual_stage = "delivery"   # delivery | return
        self._visual_out_dir = None

        # --- diagnostics / visualization state (non-control) ---
        self._last_pose = None
        self._last_goal = None
        self._vis_t = 0.0
        self._vis_count = 0
        self._vis_interval = 3.0   # seconds between snapshots
        self._nav_log_path = None
        self._nav_log_fh = None
        self._nav_log_started = False

    # ------------------------------------------------------------------
    def _scan_sectors_diag(self):
        """Diagnostic-only min distance per sector (for logging; NOT used in
        any control decision). 0 = front, +angle = left."""
        r = self._scan
        n = r.size
        if n == 0:
            return {}
        a = self._angle_min + np.arange(n) * self._angle_inc
        finite = np.isfinite(r) & (r > self.cfg.SCAN_MIN) & (r <= self.cfg.SCAN_MAX)
        rr = np.where(finite, r, self.cfg.SCAN_MAX)
        def sec(lo, hi):
            m = (a >= lo) & (a <= hi)
            return float(rr[m].min()) if m.any() else self.cfg.SCAN_MAX
        return {
            "front_min": sec(-math.radians(25), math.radians(25)),
            "left_front_min": sec(math.radians(25), math.radians(80)),
            "right_front_min": sec(-math.radians(80), -math.radians(25)),
            "rear_min": min(sec(math.radians(120), math.pi),
                            sec(-math.pi, -math.radians(120))),
        }

    # ------------------------------------------------------------------
    def set_visual_stage(self, stage):
        """Select which `visual/<stage>` directory receives snapshots.

        stage must be 'delivery' or 'return'.  The value is used only by
        diagnostic visualization; it does not affect control.
        """
        if stage not in ("delivery", "return"):
            stage = "delivery"
        self._visual_stage = stage
        self._visual_out_dir = str(
            _project_root() / "nav_return_dev" / "visual" / stage)

    # ------------------------------------------------------------------
    def push_scan(self, ranges, angle_min=None, angle_inc=None):
        self._scan = np.asarray(list(ranges), dtype=np.float64)
        self._angle_min = -math.pi if angle_min is None else float(angle_min)
        self._angle_inc = (2.0 * math.pi / max(1, self._scan.size)
                           if angle_inc is None else float(angle_inc))
        self._scan_ready = True

    # ------------------------------------------------------------------
    def begin_return(self, current_pose, final_goal, open_goal=None):
        """Enter the return-to-shelf safety sequence.

        After delivery arrives at the table the robot is facing the table, so a
        raw turn-in-place may strike items placed there.  This method arms a
        short reverse backup, then NAV+MPPI drives directly toward the real
        return goal (no separate "open area" waypoint).
        """
        cfg = self.cfg
        px, py, pyaw = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
        gx, gy, gyaw = float(final_goal[0]), float(final_goal[1]), float(final_goal[2])
        self._return_active = True
        self._return_stage = "backup"
        self._return_backup_start = (px, py)
        self._return_final_goal = (gx, gy, gyaw)
        # No open-area waypoint anymore: after the reverse backup, go straight
        # to the real return goal.  Keep _return_open_goal as a safe alias to
        # the final goal so stale references never dereference None.
        self._return_open_goal = self._return_final_goal
        # Clear stale local planner/reflex state without dropping the accumulated
        # costmap, so the return path still benefits from walls seen on the way in.
        self._ref = None
        self.stuck_counter = 0
        self.unwedge_remaining = 0
        self.recovery_turn_remaining = 0
        self.recovery_turn_sign = 0.0
        self._avoid_regions = []
        self.mppi.u_mean = np.zeros_like(self.mppi.u_mean)
        self.mppi.last_cmd = np.zeros_like(self.mppi.last_cmd)

    def _return_backup_step(self, pose):
        """Reverse away from the table.  Uses only a short fixed retreat."""
        cfg = self.cfg
        sx, sy = self._return_backup_start
        moved = math.hypot(pose[0] - sx, pose[1] - sy)
        if moved >= cfg.RETURN_BACKUP_DIST:
            self._return_stage = "nav_final"
            return 0.0, 0.0, {
                "reason": "return_backup_done",
                "mode": "return-maneuver",
                "return_stage": self._return_stage,
                "pose_x": pose[0], "pose_y": pose[1], "pose_yaw": pose[2],
                "d_final": cfg.RETURN_BACKUP_DIST,
            }
        # Safety check the planned rearward corridor before commanding reverse.
        distance = abs(cfg.RETURN_BACKUP_SPEED) * cfg.CONTROL_DT
        samples = max(1, int(math.ceil(distance / max(cfg.MAP_RESOLUTION / 2.0, 0.01))))
        clear = True
        for progress in np.linspace(distance / samples, distance, samples):
            px = pose[0] - progress * math.cos(pose[2])
            py = pose[1] - progress * math.sin(pose[2])
            if not self.costmap.footprint_is_clear(px, py, pose[2],
                                                   self.mppi.footprint_offsets):
                clear = False
                break
        if not clear:
            return 0.0, 0.0, {
                "reason": "return_blocked",
                "mode": "return-maneuver",
                "return_stage": "backup",
                "pose_x": pose[0], "pose_y": pose[1], "pose_yaw": pose[2],
                "d_final": max(0.0, cfg.RETURN_BACKUP_DIST - moved),
            }
        return cfg.RETURN_BACKUP_SPEED, 0.0, {
            "reason": "return_backup",
            "mode": "return-maneuver",
            "return_stage": "backup",
            "pose_x": pose[0], "pose_y": pose[1], "pose_yaw": pose[2],
            "d_final": max(0.0, cfg.RETURN_BACKUP_DIST - moved),
        }

    # ------------------------------------------------------------------
    def step(self, current_pose, final_goal, scan_ranges=None,
             angle_min=None, angle_inc=None) -> Tuple[float, float, dict]:
        cfg = self.cfg
        if scan_ranges is not None:
            self._scan = np.asarray(list(scan_ranges), dtype=np.float64)
            if angle_min is not None:
                self._angle_min = float(angle_min)
            if angle_inc is not None:
                self._angle_inc = float(angle_inc)
            elif not hasattr(self, "_angle_inc"):
                self._angle_inc = 2.0 * math.pi / max(1, self._scan.size)
            if not hasattr(self, "_angle_min"):
                self._angle_min = -math.pi
            self._scan_ready = True
        if not self._scan_ready or self._scan.size == 0:
            return 0.0, 0.0, {"reason": "waiting_scan", "mode": "no-scan"}

        pose = (float(current_pose[0]), float(current_pose[1]), float(current_pose[2]))
        goal = (float(final_goal[0]), float(final_goal[1]), float(final_goal[2]))
        now = time.monotonic()
        self.costmap.update(pose, self._scan, self._angle_min, self._angle_inc)
        self._prune_avoid_regions(now)

        # ---- return-to-shelf stage routing ----
        if self._return_active:
            if self._return_stage == "backup":
                v, w, m = self._return_backup_step(pose)
                self.last_cmd = (v, w)
                return v, w, m
            if self._return_stage == "nav_open":
                goal = self._return_open_goal
            elif self._return_stage == "nav_final":
                goal = self._return_final_goal

        # ---- 1. arrival ----
        d_final = math.hypot(goal[0] - pose[0], goal[1] - pose[1])

        # A temporary open-area goal is not the real shelf-A arrival.  When the
        # robot reaches it, switch to the final shelf return and continue.
        if (self._return_active and self._return_stage == "nav_open"):
            yaw_err = _wrap_pi(self._return_open_goal[2] - pose[2])
            if (d_final <= cfg.RETURN_OPEN_GOAL_TOL and
                    abs(yaw_err) <= cfg.RETURN_OPEN_GOAL_YAW_TOL):
                self._return_stage = "nav_final"
                self.last_cmd = (0.0, 0.0)
                return 0.0, 0.0, {
                    "reason": "return_open_reached",
                    "mode": "return-maneuver",
                    "return_stage": self._return_stage,
                    "pose_x": pose[0], "pose_y": pose[1], "pose_yaw": pose[2],
                    "d_goal": float(d_final),
                }

        if self.recovery_turn_remaining > 0:
            self.recovery_turn_remaining -= 1
            if self.recovery_turn_sign == 0.0:
                self.last_cmd = (0.0, 0.0)
                return 0.0, 0.0, {
                    "reason": "recovery_blocked",
                    "mode": "recovery_turn",
                    "d_final": float(d_final),
                }
            turn = self.recovery_turn_sign * cfg.RECOVERY_TURN_WZ
            self.last_cmd = (0.0, turn)
            return 0.0, turn, {
                "reason": "recovery_turning",
                "mode": "recovery_turn",
                "d_final": float(d_final),
                "recovery_turn_frames_left": int(self.recovery_turn_remaining),
                "recovery_turn_sign": int(self.recovery_turn_sign),
            }

        # ---- micro-unwedge reflex: break absolute local minima by a forced
        # reverse burst. During unwedging we skip MPPI entirely. ----
        if self.unwedge_remaining > 0:
            if not self._reverse_corridor_is_clear(pose):
                self.unwedge_remaining = 0
                self.recovery_turn_sign = self._choose_recovery_turn(pose)
                self.recovery_turn_remaining = cfg.RECOVERY_TURN_FRAMES
                self.last_cmd = (0.0, 0.0)
                return 0.0, 0.0, {
                    "reason": "recovery_reverse_blocked",
                    "mode": "recovery_turn",
                    "d_final": float(d_final),
                    "recovery_turn_sign": int(self.recovery_turn_sign),
                }
            self.unwedge_remaining -= 1
            self.last_cmd = (cfg.UNWEDGE_REVERSE_V, 0.0)
            return cfg.UNWEDGE_REVERSE_V, 0.0, {
                "reason": "unwedge_reversing",
                "mode": "unwedge_reversing",
                "unwedge_frames_left": int(self.unwedge_remaining),
                "d_final": float(d_final),
            }

        if d_final <= cfg.GOAL_TOLERANCE:
            yaw_err = _wrap_pi(goal[2] - pose[2])
            if abs(yaw_err) <= cfg.GOAL_YAW_TOL:
                self.last_cmd = (0.0, 0.0)
                if self._return_active:
                    self._return_active = False
                    self._return_stage = "done"
                return 0.0, 0.0, {"reason": "arrived", "mode": "arrived"}
            wz = float(np.clip(1.0 * yaw_err, -0.6, 0.6))
            self.last_cmd = (0.0, wz)
            return 0.0, wz, {"reason": "align_yaw", "mode": "arriving"}

        # ---- 2. costmap + A* reference ----
        avoid_regions = self._active_avoid_regions()
        self._ref = self.planner.plan(pose, goal, self.costmap,
                                      avoid_regions=avoid_regions)
        if self._ref is None or len(self._ref) < 1:
            self.last_cmd = (0.0, 0.0)
            no_path_meta = {
                "reason": "no_safe_path",
                "mode": "no-path",
                "recovery_avoid_regions": len(avoid_regions),
                **self.planner.last_diagnostics,
            }
            no_path_meta.update(self._scan_sectors_diag())
            no_path_meta.update({
                "pose_x": pose[0], "pose_y": pose[1], "pose_yaw": pose[2],
                "goal_x": goal[0], "goal_y": goal[1], "goal_yaw": goal[2],
                "d_goal": float(d_final),
                "scan_n": int(self._scan.size),
                "angle_min": float(self._angle_min),
                "angle_inc": float(self._angle_inc),
                "cost_lethal_cells": int(np.count_nonzero(self.costmap.lethal)),
                "cost_mean": float(np.mean(self.costmap.inflate())),
            })
            self._last_pose = pose
            self._last_goal = goal
            self._maybe_visualize(pose, goal)
            return 0.0, 0.0, no_path_meta
        ref = self._forward_window(pose, self._ref)

        # ---- 3. MPPI ----
        if len(ref) < 2:
            v = 0.0
            dx = goal[0] - pose[0]; dy = goal[1] - pose[1]
            w = float(np.clip(0.8 * _wrap_pi(math.atan2(dy, dx) - pose[2]),
                              -cfg.MAX_WZ, cfg.MAX_WZ))
            self.last_cmd = (v, w)
            return v, w, {"reason": "mppi_fallback", "mode": "mppi"}

        v, w, meta = self.mppi.compute_velocity_command(pose, ref, self.costmap)
        self.last_cmd = (v, w)
        meta["d_final"] = float(d_final)
        meta["reason"] = meta.get("reason", "mppi")
        meta.update(self._scan_sectors_diag())   # observation only
        meta.update(self.planner.last_diagnostics)
        meta["recovery_avoid_regions"] = len(avoid_regions)

        # ---- navigation diagnostics (observation only, no control use) ----
        # d_goal: robot -> current target (pre-clamp) straight-line distance.
        # goal_clamped: A* clamped the far goal to GOAL_RAY_CLAMP radius (1/0).
        ref_all = self._ref if self._ref is not None else ref
        try:
            if ref_all is not None and len(ref_all) >= 2:
                segs = np.hypot(np.diff(ref_all[:, 0]), np.diff(ref_all[:, 1]))
                path_len = float(np.sum(segs))
                path_n = int(len(ref_all))
            else:
                path_len = 0.0
                path_n = int(len(ref_all)) if ref_all is not None else 0
            # nearest reference point to robot
            dr = np.hypot(ref_all[:, 0] - pose[0], ref_all[:, 1] - pose[1])
            i0 = int(np.argmin(dr))
            path_dev0 = float(dr[i0])
            ref_hdg = float(ref_all[i0, 2])
            ref_hdg_err = float(_wrap_pi(ref_hdg - pose[2]))
            # carrot distance from robot to final reference (lookahead proxy)
            lookahead_dist = float(np.hypot(ref_all[-1, 0] - pose[0],
                                            ref_all[-1, 1] - pose[1]))
        except Exception:
            path_len = 0.0
            path_n = 0
            path_dev0 = 0.0
            ref_hdg_err = 0.0
            lookahead_dist = 0.0
        goal_clamped = int(self.planner.last_diagnostics.get(
            "astar_goal_was_clamped", 0))
        meta["d_goal"] = float(d_final)
        meta["goal_clamped"] = int(goal_clamped)
        meta["path_len"] = path_len
        meta["path_n"] = path_n
        meta["path_dev0"] = path_dev0
        meta["ref_hdg_err"] = ref_hdg_err
        meta["lookahead_dist"] = lookahead_dist

        # extended diagnostics: pose/goal/scan/costmap/A* snapshot
        meta["pose_x"] = float(pose[0])
        meta["pose_y"] = float(pose[1])
        meta["pose_yaw"] = float(pose[2])
        meta["goal_x"] = float(goal[0])
        meta["goal_y"] = float(goal[1])
        meta["goal_yaw"] = float(goal[2])
        meta["scan_n"] = int(self._scan.size) if getattr(self, "_scan", None) is not None else 0
        meta["angle_min"] = float(getattr(self, "_angle_min", -math.pi))
        meta["angle_inc"] = float(getattr(self, "_angle_inc", 0.0))
        meta["cost_null"] = 0
        meta["cost_lethal_cells"] = 0
        meta["cost_mean"] = 0.0
        meta["ref_exists"] = int(self._ref is not None and len(self._ref) > 0)
        if self._ref is not None and len(self._ref):
            meta["ref_start_x"] = float(self._ref[0, 0])
            meta["ref_start_y"] = float(self._ref[0, 1])
            meta["ref_end_x"] = float(self._ref[-1, 0])
            meta["ref_end_y"] = float(self._ref[-1, 1])
            # first points of A* path: reveal whether the planner starts by
            # moving toward the true goal or toward the shelf/back.
            _n = min(5, len(self._ref))
            meta["ref_first_x"] = ",".join("%.2f" % float(self._ref[i, 0]) for i in range(_n))
            meta["ref_first_y"] = ",".join("%.2f" % float(self._ref[i, 1]) for i in range(_n))
            # cost sampled from robot toward the real goal at increasing range
            _gx, _gy = float(goal[0]), float(goal[1])
            _dx = _gx - pose[0]; _dy = _gy - pose[1]
            _gd = math.hypot(_dx, _dy)
            _cost_series = []
            if _gd > 1e-9:
                _ux = _dx / _gd; _uy = _dy / _gd
                for _r in (0.2, 0.4, 0.6, 0.8, 1.0):
                    try:
                        _cost_series.append("%.0f" % self.costmap.query_cost_xy(
                            pose[0] + _ux * _r, pose[1] + _uy * _r))
                    except Exception:
                        _cost_series.append("-")
            meta["ref_goal_dir_cost"] = ",".join(_cost_series)
            try:
                meta["cost_lethal_cells"] = int(np.count_nonzero(self.costmap.lethal))
                meta["cost_mean"] = float(np.mean(self.costmap.inflate()))
            except Exception:
                pass
        self._last_pose = pose
        self._last_goal = goal
        self._maybe_visualize(pose, goal)

        # ---- stuck detection: far from goal but MPPI outputs ~zero motion ----
        # A blocked_stop flag means nearly every MPPI sample predicted a lethal
        # collision, so the controller deliberately issued v=0.  Treat that as
        # stuck so the unwedge reflex fires instead of the robot pushing a wall.
        blocked_stop = bool(meta.get("blocked_stop", False))
        if (d_final > cfg.UNWEDGE_DIST_MIN and
                (blocked_stop or
                 (abs(v) < cfg.UNWEDGE_V_EPS and abs(w) < cfg.UNWEDGE_W_EPS))):
            self.stuck_counter += 1
            if self.stuck_counter >= cfg.UNWEDGE_STUCK_FRAMES:
                self.stuck_counter = 0
                self._remember_failed_corridor(pose, now)
                self.unwedge_remaining = cfg.UNWEDGE_REVERSE_FRAMES
                self.mppi.u_mean = np.zeros_like(self.mppi.u_mean)
                self.mppi.last_cmd = np.zeros_like(self.mppi.last_cmd)
        else:
            self.stuck_counter = 0
        meta["stuck_counter"] = int(self.stuck_counter)
        meta["recovery_avoid_regions"] = len(self._active_avoid_regions())
        return v, w, meta

    # ------------------------------------------------------------------
    def _prune_avoid_regions(self, now):
        self._avoid_regions = [region for region in self._avoid_regions
                               if region[3] > now]

    def _active_avoid_regions(self):
        return [(x, y, radius) for x, y, radius, _expiry in self._avoid_regions]

    def _remember_failed_corridor(self, pose, now):
        if self._ref is None or len(self._ref) < 2:
            return
        cfg = self.cfg
        distances = np.hypot(self._ref[:, 0] - pose[0], self._ref[:, 1] - pose[1])
        start = int(np.argmin(distances))
        expiry = now + cfg.RECOVERY_AVOID_TTL
        for point in self._ref[start + 1:]:
            ahead = math.hypot(float(point[0]) - pose[0], float(point[1]) - pose[1])
            if ahead < cfg.RECOVERY_AVOID_MIN_AHEAD:
                continue
            if ahead > cfg.RECOVERY_AVOID_MAX_AHEAD:
                break
            self._avoid_regions.append((float(point[0]), float(point[1]),
                                        cfg.RECOVERY_AVOID_RADIUS, expiry))
            break

    def _reverse_corridor_is_clear(self, pose):
        cfg = self.cfg
        distance = abs(cfg.UNWEDGE_REVERSE_V) * cfg.CONTROL_DT * cfg.UNWEDGE_REVERSE_FRAMES
        samples = max(1, int(math.ceil(distance / max(cfg.MAP_RESOLUTION / 2.0, 0.01))))
        for progress in np.linspace(distance / samples, distance, samples):
            x = pose[0] - progress * math.cos(pose[2])
            y = pose[1] - progress * math.sin(pose[2])
            if not self.costmap.footprint_is_clear(x, y, pose[2],
                                                   self.mppi.footprint_offsets):
                return False
        return True

    def _choose_recovery_turn(self, pose):
        cfg = self.cfg
        best_sign = 0.0
        best_clearance = -1.0
        for sign in (1.0, -1.0):
            min_clearance = float("inf")
            is_clear = True
            for ratio in (0.25, 0.5, 0.75, 1.0):
                yaw = pose[2] + sign * cfg.RECOVERY_TURN_CHECK_ANGLE * ratio
                if not self.costmap.footprint_is_clear(
                        pose[0], pose[1], yaw, self.mppi.footprint_offsets):
                    is_clear = False
                    break
                min_clearance = min(
                    min_clearance,
                    self.costmap.footprint_clearance(
                        pose[0], pose[1], yaw, self.mppi.footprint_offsets))
            if is_clear and min_clearance > best_clearance:
                best_clearance = min_clearance
                best_sign = sign
        return best_sign

    # ------------------------------------------------------------------
    def _fallback_ref(self, pose, goal):
        ds = self.cfg.PATH_RESAMPLE_DS
        dx = goal[0] - pose[0]; dy = goal[1] - pose[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return np.array([[pose[0], pose[1], pose[2]]], dtype=np.float64)
        n = max(2, int(dist / ds) + 2)
        xs = np.linspace(pose[0], goal[0], n)
        ys = np.linspace(pose[1], goal[1], n)
        return np.stack([xs, ys, np.full(n, math.atan2(dy, dx))], axis=1)

    def _forward_window(self, pose, ref):
        cfg = self.cfg
        px, py = pose[0], pose[1]
        d = np.hypot(ref[:, 0] - px, ref[:, 1] - py)
        i0 = int(np.argmin(d))
        out = [ref[i0]]
        acc = 0.0
        for i in range(i0 + 1, len(ref)):
            seg = math.hypot(ref[i][0] - ref[i-1][0], ref[i][1] - ref[i-1][1])
            acc += seg
            if acc > cfg.HORIZON_LENGTH:
                out.append(ref[i]); break
            out.append(ref[i])
        return np.asarray(out, dtype=np.float64)

    def _maybe_visualize(self, pose, goal):
        """Save a god-view PNG every VIS_INTERVAL seconds. Diagnostic only."""
        try:
            now = time.time()
            if self._vis_t == 0.0 or (now - self._vis_t) >= self._vis_interval:
                self._vis_t = now
                self._vis_count += 1
                if save_visualization is not None:
                    title = "NAV+MPPI god view #%d (t=%.3f)" % (
                        self._vis_count, now)
                    out_dir = self._visual_out_dir
                    if out_dir is None:
                        out_dir = str(_project_root() /
                                      "nav_return_dev" / "visual" /
                                      self._visual_stage)
                    path = save_visualization(self, pose, goal, title=title,
                                              out_dir=out_dir)
                    if path is None:
                        self._write_visual_diag("visual returned None")
        except Exception as exc:
            try:
                self._write_visual_diag("visual exception: %r" % (exc,))
            except Exception:
                pass

    def _write_visual_diag(self, msg):
        try:
            import os
            os.makedirs(str(_project_root() / "nav_return_dev" / "logs"), exist_ok=True)
            with open(str(_project_root() / "nav_return_dev" / "logs" / "visual_error.log"), "a", encoding="utf-8") as f:
                f.write("ts=%.3f %s\n" % (time.time(), msg))
        except Exception:
            pass

    def start_nav_log(self, log_root=None):
        """Open nav/logs/nav_<timestamp>.log for stage-4 diag lines.

        The default is anchored to the project root, so logs land in the
        baseline nav/logs directory regardless of the current process cwd.
        """
        try:
            import os
            import time as _t
            if self._nav_log_fh is not None:
                return self._nav_log_path
            if log_root is None:
                log_root = str(_project_root() / "nav_return_dev" / "logs")
            os.makedirs(log_root, exist_ok=True)
            ts = _t.strftime("%Y%m%d_%H%M%S")
            self._nav_log_path = os.path.join(log_root, "nav_%s.log" % ts)
            self._nav_log_fh = open(self._nav_log_path, "a", encoding="utf-8")
            self._nav_log_started = True
            return self._nav_log_path
        except Exception:
            return None

    def write_nav_log(self, text):
        """Append one diagnostic line. Never raises into control."""
        if self._nav_log_fh is None:
            return
        try:
            self._nav_log_fh.write(text + "\n")
            self._nav_log_fh.flush()
        except Exception:
            pass

    def close_nav_log(self):
        if self._nav_log_fh is not None:
            try:
                self._nav_log_fh.close()
            except Exception:
                pass
            self._nav_log_fh = None

    def reset(self):
        self.close_nav_log()
        self._ref = None
        self._scan_ready = False
        self.last_cmd = (0.0, 0.0)
        self.stuck_counter = 0
        self.unwedge_remaining = 0
        self.recovery_turn_remaining = 0
        self.recovery_turn_sign = 0.0
        self._avoid_regions = []
        # --- clear return-to-shelf maneuver state ---
        self._return_active = False
        self._return_stage = ""
        self._return_backup_start = None
        self._return_open_goal = None
        self._return_final_goal = None
        self._visual_stage = "delivery"
        self._visual_out_dir = None
        self._vis_t = 0.0
        self._last_pose = None
        self._last_goal = None
        self.mppi.u_mean = np.zeros_like(self.mppi.u_mean)
        self.mppi.last_cmd = np.zeros_like(self.mppi.last_cmd)
