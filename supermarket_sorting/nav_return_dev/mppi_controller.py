# -*- coding: utf-8 -*-
"""Vectorized MPPI controller with true rectangular-footprint collision check.

K samples are propagated by exact midpoint (RK2) unicycle kinematics. Hard
collision uses the real rectangle footprint against raw LiDAR obstacle cells;
soft inflation is scored at the vehicle center for standoff and speed shaping.
WEIGHT_OBSTACLE(300) is the sole speed-modulating obstacle term; there is NO
laser-distance if/else that hard-codes velocity.

MIN_VX = -0.10: a small reverse range is kept for the trapped-facing-obstacle
case. Forward progress cost still dominates, and the forced reverse reflex may
also use it as an emergency backup.
"""
import math
from typing import Tuple

import numpy as np

from nav_return_dev.config import NavConfig
from nav_return_dev.footprint import rectangle_footprint_offsets


class MPPIController:
    def __init__(self, cfg: NavConfig):
        self.cfg = cfg
        self.K = cfg.MPPI_NUM_SAMPLES
        self.T = cfg.MPPI_HORIZON_STEPS
        self.dt = cfg.CONTROL_DT
        self.lambda_ = cfg.TEMPERATURE_LAMBDA

        # Warm-start the first MPPI solution with a small forward velocity.
        # Without this, all samples start around v=0 and the optimizer prefers
        # turning in place while misaligned, causing delivery start circling.
        self.u_mean = np.zeros((self.T, 2), dtype=np.float32)
        self.u_mean[:, 0] = cfg.NAV_WARMSTART_V
        self.noise_sigma = np.array([cfg.NOISE_SIGMA_V, cfg.NOISE_SIGMA_W],
                                    dtype=np.float32)
        self.footprint_offsets = rectangle_footprint_offsets(cfg)
        self.last_cmd = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    def _score_components(self, states, controls, ref_xy, ref_hdg, lookahead, costmap):
        """Diagnostic-only component costs for the best few samples."""
        cfg = self.cfg
        off = self.footprint_offsets
        inflated_cost = costmap.inflate()
        raw_lethal_cost = np.where(costmap.lethal, cfg.LETHAL_COST, 0.0)
        K = states.shape[0]
        obs_total = np.zeros(K)
        obs_soft = np.zeros(K)
        path_total = np.zeros(K)
        head_total = np.zeros(K)
        twirl_total = np.zeros(K)
        smooth_total = np.zeros(K)
        collide_step = np.full(K, -1, dtype=np.int32)
        active = np.ones(K, dtype=bool)
        gx, gy = lookahead
        for t in range(0, self.T + 1):
            x = states[:, t, 0]; y = states[:, t, 1]; th = states[:, t, 2]
            cos_t = np.cos(th)[:, None]; sin_t = np.sin(th)[:, None]
            fx = x[:, None] + cos_t * off[None, :, 0] - sin_t * off[None, :, 1]
            fy = y[:, None] + sin_t * off[None, :, 0] + cos_t * off[None, :, 1]
            hard_obs = self._costmap_query_batch(fx, fy, costmap, raw_lethal_cost)
            collide = active & (hard_obs.max(axis=1) >= cfg.LETHAL_COST - 1e-3)
            center_soft = self._costmap_query_batch(
                x[:, None], y[:, None], costmap, inflated_cost)[:, 0]
            if t < cfg.COLLIDE_START_STEP:
                collide = np.zeros_like(collide)
            if collide.any():
                remain = int(np.clip(self.T - t, 1, self.T))
                obs_total[collide] += cfg.WEIGHT_OBSTACLE * remain
                collide_step[collide] = t
                active &= ~collide
            _obs_soft_diag = center_soft
            if t < cfg.COLLIDE_START_STEP:
                _obs_soft_diag = np.zeros_like(_obs_soft_diag)
            obs_soft += np.where(active, _obs_soft_diag, 0.0)
            path_dev = self._dist_to_path_batch(x, y, ref_xy)
            path_total += np.where(active, path_dev, 0.0)
            ref_i = self._nearest_ref_index_batch(x, y, ref_xy)
            hdg_err = self._wrap_pi_batch(th - ref_hdg[ref_i])
            head_total += np.where(active, hdg_err ** 2, 0.0)
            ci = max(0, t - 1)
            twirl_total += np.where(active, controls[:, ci, 1] ** 2, 0.0)
            du = controls[:, ci, :] - self.u_mean[ci]
            smooth_total += np.where(active, np.sum(du * du, axis=1), 0.0)
        _pi = int(np.argmin(np.hypot(ref_xy[:, 0] - states[0, 0, 0],
                                      ref_xy[:, 1] - states[0, 0, 1])))
        _pe = _pi + 1 if _pi + 1 < len(ref_xy) else _pi - 1
        _dsx, _dsy = ref_xy[_pe, 0] - ref_xy[_pi, 0], ref_xy[_pe, 1] - ref_xy[_pi, 1]
        _dl = math.hypot(_dsx, _dsy)
        if _dl > 1e-6:
            _dsx, _dsy = _dsx / _dl, _dsy / _dl
        else:
            _dsx, _dsy = math.cos(ref_hdg[_pi]), math.sin(ref_hdg[_pi])
        progress = ((states[:, -1, 0] - states[:, 0, 0]) * _dsx
                    + (states[:, -1, 1] - states[:, 0, 1]) * _dsy)
        # Mirror the optimizer's lower-is-better progress cost, not raw signed
        # forward displacement, so diagnostic S_progress truly matches the
        # control objective.
        progress_cost = np.maximum(0.0, max(cfg.LOOKAHEAD_DIST, 0.05) - progress)
        return {
            "S_obs_penalty": obs_total,
            "S_obs_soft": cfg.WEIGHT_OBSTACLE * obs_soft,
            "S_path": cfg.WEIGHT_PATH_ALIGN * path_total,
            "S_head": cfg.WEIGHT_HEADING * head_total,
            "S_twirl": cfg.WEIGHT_TWIRL * twirl_total,
            "S_smooth": 0.5 * smooth_total,
            "S_progress": cfg.WEIGHT_PATH_PROGRESS * progress_cost,
            "collide_step": collide_step,
        }

    def compute_velocity_command(self, robot_pose, reference_path, costmap):
        """robot_pose=(x,y,yaw); reference_path=(N,3). Returns (v,w,meta)."""
        cfg = self.cfg
        x0, y0, yaw0 = robot_pose

        ref_xy = reference_path[:, :2]
        ref_hdg = reference_path[:, 2]
        # "Carrot" lookahead: the single forward attractor is a point ~1.2m
        # ahead of the robot along the (already 0.05m-resampled) reference path.
        # MPPI is a pure path tracker: it minimizes lateral path deviation and
        # the Euclidean distance from each trajectory's FINAL pose to this
        # lookahead point. No unstable vector-projection rewards.
        lookahead = self._lookahead_point(x0, y0, ref_xy, cfg.LOOKAHEAD_DIST)
        gx, gy = lookahead

        # ---- 1. sample controls (v may be negative: weak reverse allowed) ----
        noise = np.random.normal(0.0, 1.0, size=(self.K, self.T, 2)).astype(np.float32)
        controls = self.u_mean[None, :, :] + noise * self.noise_sigma[None, None, :]
        v = np.clip(controls[:, :, 0], cfg.MIN_VX, cfg.MAX_VX)
        w = np.clip(controls[:, :, 1], -cfg.MAX_WZ, cfg.MAX_WZ)
        controls = np.stack([v, w], axis=-1)

        # ---- 2. parallel kinematics (midpoint RK2) ----
        states = np.empty((self.K, self.T + 1, 3), dtype=np.float64)
        states[:, 0, 0] = x0
        states[:, 0, 1] = y0
        states[:, 0, 2] = yaw0
        yaw = yaw0
        for t in range(self.T):
            vt = controls[:, t, 0]
            wt = controls[:, t, 1]
            yaw_next = yaw + wt * self.dt
            yaw_mid = yaw + 0.5 * wt * self.dt
            states[:, t + 1, 0] = states[:, t, 0] + vt * np.cos(yaw_mid) * self.dt
            states[:, t + 1, 1] = states[:, t, 1] + vt * np.sin(yaw_mid) * self.dt
            states[:, t + 1, 2] = yaw_next
            yaw = yaw_next

        # ---- 3. footprint projection & cost ----
        off = self.footprint_offsets
        M = off.shape[0]
        S = np.zeros(self.K, dtype=np.float64)
        inflated_cost = costmap.inflate()
        costmap._cached_cost = inflated_cost
        raw_lethal_cost = np.where(costmap.lethal, cfg.LETHAL_COST, 0.0)

        # Time-To-Collision gradient: for each trajectory, the FIRST step whose
        # footprint touches a lethal cell adds WEIGHT_OBSTACLE * remaining_horizon
        # and then that trajectory stops accumulating any further cost. This gives
        # Softmax a smooth gradient (prefer the LATEST collision / none) instead of
        # a single flat 1e9 for every colliding sample, which flattened all weights
        # to zero when every sample collided in a tight dead-end.
        active = np.ones(self.K, dtype=bool)
        for t in range(0, self.T + 1):
            x = states[:, t, 0]
            y = states[:, t, 1]
            th = states[:, t, 2]

            cos_t = np.cos(th)[:, None]
            sin_t = np.sin(th)[:, None]
            fx = x[:, None] + cos_t * off[None, :, 0] - sin_t * off[None, :, 1]
            fy = y[:, None] + sin_t * off[None, :, 0] + cos_t * off[None, :, 1]

            hard_obs = self._costmap_query_batch(fx, fy, costmap, raw_lethal_cost)
            max_hard_obs = hard_obs.max(axis=1)                # (K,)
            center_soft = self._costmap_query_batch(
                x[:, None], y[:, None], costmap, inflated_cost)[:, 0]

            # first-ever lethal touch for still-active trajectories.
            # Skip the robot's current footprint (t=0): when the chassis is
            # already touching an obstacle, penalizing every sample at t=0
            # flattens the cost landscape to the same 9000 and MPPI loses all
            # directional gradient, so it cannot back out of the box corner.
            collide = active & (max_hard_obs >= cfg.LETHAL_COST - 1e-3)
            if t < cfg.COLLIDE_START_STEP:
                collide = np.zeros_like(collide)
            if collide.any():
                remain = int(np.clip(self.T - t, 1, self.T))
                S[collide] += cfg.WEIGHT_OBSTACLE * float(remain)
                active &= ~collide

            path_dev = self._dist_to_path_batch(x, y, ref_xy)

            ref_i = self._nearest_ref_index_batch(x, y, ref_xy)
            target_h = ref_hdg[ref_i]
            hdg_err = self._wrap_pi_batch(th - target_h)
            heading_cost = cfg.WEIGHT_HEADING * (hdg_err ** 2)

            ci = max(0, t - 1)
            twirl = controls[:, ci, 1] ** 2
            prev_u = self.u_mean[ci]
            du = controls[:, ci, :] - prev_u
            smooth = np.sum(du * du, axis=1)

            # Per-step forward progress: reward driving along the path heading
            # instead of only the final pose. This is essential when the start
            # is misaligned; otherwise MPPI can turn in place at v=0 for the
            # first action and only accelerate later, causing circling.
            # forward component = v * cos(heading_error). Cost decreases as
            # forward speed approaches the target cruise speed, so v=0 is
            # penalized while actual forward motion is always preferred.
            _fwd_step = controls[:, ci, 0] * np.cos(hdg_err)
            _target_v = min(cfg.MAX_VX, 0.35)
            _prog_step = np.maximum(0.0, _target_v - _fwd_step)

            # During an initial large misalignment, allow "turn while moving
            # forward" by gently ramping heading/path penalties up over the
            # first ALIGN_RAMP_STEPS. Without this, MPPI prefers spinning/
            # reversing because near-zero v keeps heading/path deviation low.
            _align = min(1.0, float(t) / max(1, cfg.ALIGN_RAMP_STEPS))
            _align = 0.25 + 0.75 * _align
            # t=0 is the robot's current footprint.  Leave its (identical,
            # often already-lethal) soft-obstacle cost out of the optimization
            # so a chassis touching a box does not flatten every sample's cost
            # with the same huge constant and destroy the directional gradient.
            _obs_stage = cfg.WEIGHT_OBSTACLE * center_soft
            if t < cfg.COLLIDE_START_STEP:
                _obs_stage = np.zeros_like(_obs_stage)
            stage = (cfg.WEIGHT_PATH_ALIGN * _align * path_dev +
                     _obs_stage +
                     cfg.WEIGHT_TWIRL * twirl +
                     _align * heading_cost +
                     cfg.WEIGHT_PATH_PROGRESS * _prog_step +
                     0.5 * smooth)
            # only still-active (not-yet-collided) trajectories accumulate stage
            S += np.where(active, stage, 0.0)

        # ---- progress: signed forward projection along the reference path ----
        # Reward (lower cost for) samples that move forward along the local path
        # direction, not raw distance to a single carrot point. This prevents
        # MPPI from choosing reverse/turn-in-place just to "get closer" to a
        # lookahead point while the chassis is initially misaligned.
        final_x = states[:, -1, 0]
        final_y = states[:, -1, 1]
        start_i = int(np.argmin(np.hypot(ref_xy[:, 0] - x0, ref_xy[:, 1] - y0)))
        if start_i < len(ref_xy) - 1:
            _dsx = ref_xy[start_i + 1, 0] - ref_xy[start_i, 0]
            _dsy = ref_xy[start_i + 1, 1] - ref_xy[start_i, 1]
        else:
            _dsx = ref_xy[-1, 0] - ref_xy[-2, 0]
            _dsy = ref_xy[-1, 1] - ref_xy[-2, 1]
        _dl = math.hypot(_dsx, _dsy)
        if _dl > 1e-9:
            _dsx, _dsy = _dsx / _dl, _dsy / _dl
        else:
            _dsx, _dsy = math.cos(ref_hdg[start_i]), math.sin(ref_hdg[start_i])
        # fwd > 0 means the final pose advanced toward the path/goal direction.
        # Progress cost decreases linearly as the sample gets closer to the
        # lookahead length; this is what makes forward motion actually win over
        # turning in place (a flat "backward-only" penalty does not).
        fwd = (final_x - x0) * _dsx + (final_y - y0) * _dsy
        _progress_target = max(cfg.LOOKAHEAD_DIST, 0.05)
        path_progress = np.maximum(0.0, _progress_target - fwd)

        # ---- 4. softmax path integral ----
        Smin = S.min()
        w = np.exp(-(S - Smin) / self.lambda_)
        wsum = w.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            w = np.ones(self.K) / self.K
        else:
            w /= wsum

        u_exp = np.einsum('k,ktd->td', w, controls)

        # ---- 5. warm-start shift ----
        self.u_mean[:-1] = u_exp[1:]
        self.u_mean[-1] = u_exp[-1]

        # ---- 6. smoothing (accel limits only; no laser if/else) ----
        v_cmd = self._rate_limit_v(self.last_cmd[0], float(u_exp[0, 0]))
        w_cmd = self._rate_limit_w(self.last_cmd[1], float(u_exp[0, 1]))

        meta = {
            "reason": "mppi",
            "mode": "mppi",
            "S_min": float(Smin),
            "collide_count": int(np.count_nonzero(~active)),
        }

        # ---- 6b. all-collision safety gate ----
        # If almost every sampled trajectory predicts a lethal collision in the
        # first few steps, the softmax still picks a "least bad" forward sample.
        # That is why the robot kept pushing into a wall while MPPI kept sending
        # v>0.  Stop the chassis so the higher-level unwedge reflex can fire.
        cfg = self.cfg
        blocked_ratio = float(np.count_nonzero(~active)) / max(1, self.K)
        blocked_threshold = getattr(cfg, "MPPI_BLOCKED_STOP_RATIO", 0.90)
        blocked_s_min = getattr(cfg, "MPPI_BLOCKED_STOP_S_MIN", 5000.0)
        if blocked_ratio >= blocked_threshold and Smin >= blocked_s_min:
            v_cmd = 0.0
            w_cmd = 0.0
            self.last_cmd[0] = v_cmd
            self.last_cmd[1] = w_cmd
            meta["blocked_stop"] = True
            meta["blocked_ratio"] = float(blocked_ratio)
        else:
            self.last_cmd[0] = v_cmd
            self.last_cmd[1] = w_cmd
            meta["blocked_stop"] = False
            meta["blocked_ratio"] = float(blocked_ratio)
        # ---- diagnostic: top best sample component breakdown ----
        try:
            idx = np.argsort(S)[:5]
            comp = self._score_components(states[idx], controls[idx], ref_xy,
                                          ref_hdg, (gx, gy), costmap)
            meta["top_v"] = ",".join(["%.3f" % float(np.mean(controls[i, :, 0]))
                                      for i in idx])
            meta["top_w"] = ",".join(["%.3f" % float(np.mean(controls[i, :, 1]))
                                      for i in idx])
            meta["top_v0"] = ",".join(["%.3f" % float(controls[i, 0, 0])
                                       for i in idx])
            meta["top_w0"] = ",".join(["%.3f" % float(controls[i, 0, 1])
                                       for i in idx])
            for key in ("S_obs_penalty", "S_obs_soft", "S_path", "S_head",
                        "S_twirl", "S_smooth", "S_progress"):
                vals = comp[key]
                meta["top_" + key] = ",".join(["%.1f" % float(v) for v in vals])
            meta["top_collide_step"] = ",".join([str(int(v)) for v in comp["collide_step"]])
            meta["top_S"] = ",".join(["%.1f" % float(S[i]) for i in idx])
        except Exception:
            pass
        return v_cmd, w_cmd, meta

    # ------------------------------------------------------------------
    def _costmap_query_batch(self, fx, fy, costmap, cost=None):
        if cost is None:
            cost = getattr(costmap, "_cached_cost", None)
            if cost is None:
                cost = costmap.inflate()
        res = costmap.res
        ox = costmap.origin_x
        oy = costmap.origin_y
        sx = costmap.size_x
        sy = costmap.size_y

        cx = np.round((fx - ox) / res).astype(np.int32)
        cy = np.round((fy - oy) / res).astype(np.int32)
        oob = (cx < 0) | (cx >= sx) | (cy < 0) | (cy >= sy)
        cx = np.clip(cx, 0, sx - 1)
        cy = np.clip(cy, 0, sy - 1)
        out = cost[cy, cx].astype(np.float64)
        out = np.where(oob, float(self.cfg.LETHAL_COST), out)  # unknown = conservative
        return out

    @staticmethod
    def _nearest_ref_index_batch(x, y, ref_xy):
        dx = x[:, None] - ref_xy[None, :, 0]
        dy = y[:, None] - ref_xy[None, :, 1]
        return np.argmin(dx * dx + dy * dy, axis=1)

    @staticmethod
    def _wrap_pi_batch(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _lookahead_point(x0, y0, ref_xy, dist):
        """Point on the reference path at arc-length ~dist ahead of (x0,y0)."""
        # nearest index to robot
        dx = ref_xy[:, 0] - x0
        dy = ref_xy[:, 1] - y0
        d2 = dx * dx + dy * dy
        i0 = int(np.argmin(d2))
        # walk forward accumulating arc length until dist
        acc = 0.0
        i = i0
        while i < len(ref_xy) - 1:
            seg = float(np.hypot(ref_xy[i+1, 0] - ref_xy[i, 0],
                                 ref_xy[i+1, 1] - ref_xy[i, 1]))
            if acc + seg >= dist:
                frac = (dist - acc) / seg if seg > 1e-9 else 0.0
                px = ref_xy[i, 0] + frac * (ref_xy[i+1, 0] - ref_xy[i, 0])
                py = ref_xy[i, 1] + frac * (ref_xy[i+1, 1] - ref_xy[i, 1])
                return (float(px), float(py))
            acc += seg
            i += 1
        return (float(ref_xy[-1, 0]), float(ref_xy[-1, 1]))

    @staticmethod
    def _dist_to_path_batch(x, y, ref_xy):
        dx = x[:, None] - ref_xy[None, :, 0]
        dy = y[:, None] - ref_xy[None, :, 1]
        return np.hypot(dx, dy).min(axis=1)

    def _rate_limit_v(self, prev, cur):
        lim = self.cfg.MAX_ACC_VX * self.dt
        return float(np.clip(cur, prev - lim, prev + lim))

    def _rate_limit_w(self, prev, cur):
        lim = self.cfg.MAX_ACC_WZ * self.dt
        return float(np.clip(cur, prev - lim, prev + lim))
