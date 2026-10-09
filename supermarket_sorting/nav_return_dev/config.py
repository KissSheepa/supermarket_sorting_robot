# -*- coding: utf-8 -*-
"""Centralized physics/geometry parameters for NAV + MPPI (pure A* + MPPI).

Architecture notes (strict):
  * 8.0m x 8.0m rolling local map @ 0.05m, so the far goal is never truncated.
  * Gradient soft-inflation: lethal(253) <= ROBOT_HALF_WIDTH; a soft band from
    LETHAL_COST down to 0 gives A* a graduated standoff. A* MAY pass through any
    cell with cost < 253 (soft band) to solve narrow gaps.
  * MIN_VX = -0.10 allows a small reverse backup when the chassis is trapped
    facing an obstacle and needs a short retreat to align with A*.
  * NAV_WARMSTART_V biases the initial MPPI control toward forward motion,
    preventing a zero-speed spin at delivery start.
  * All speed modulation comes from WEIGHT_OBSTACLE on the true rectangular
    footprint; there are NO laser-distance if/else speed rules anywhere.
"""
import math


class NavConfig:
    # ---- robot body (exact geometry) ----
    ROBOT_LENGTH = 0.42            # m, front-back
    ROBOT_WIDTH = 0.30             # m, left-right
    FOOTPRINT_PADDING = 0.01       # m, tighter footprint margin for narrow corridor
    ROBOT_HALF_WIDTH = 0.15        # m, half width (physical; used for lethal band)
    PLANNER_CLEARANCE_MARGIN = 0.02  # m, LiDAR/grid margin outside the body

    # ---- control & timing ----
    CONTROL_DT = 0.05              # s, 20Hz
    MAX_VX = 0.55                  # m/s
    MIN_VX = -0.10                 # m/s, small reverse only; forward bias still dominates
    NAV_WARMSTART_V = 0.15          # initial MPPI forward mean bias (stop zero-speed spin)
    MAX_WZ = 0.8                  # rad/s
    MAX_ACC_VX = 1.5               # m/s^2
    MAX_ACC_WZ = 3.5               # rad/s^2

    # ---- rolling costmap grid (8m x 8m) ----
    MAP_SIZE_X = 8.0               # m
    MAP_SIZE_Y = 8.0               # m
    MAP_RESOLUTION = 0.05          # m/cell (160x160)
    LOCAL_GOAL_MAP_MARGIN = ((ROBOT_LENGTH + 2.0 * FOOTPRINT_PADDING) / 2.0 +
                             MAP_RESOLUTION)
    INFLATION_RADIUS = 0.25        # m, soft-inflation outer bound (keeps real gaps traversable)
    LETHAL_COST = 253              # normalized cost >= this is impassable by A*
    SOFT_BAND_START = 0.12         # backup baseline: keep narrow passages soft, not lethal
    # The hard band is a center-point clearance for A*. MPPI separately checks
    # the same physical rectangle against raw obstacle cells.
    OCC_DECAY = 0.9                # per-frame occupancy decay
    OCC_INCR = 254.0               # occupancy increment (still immediate, controlled by decay+raycast)

    # ---- LiDAR validity ----
    SCAN_MIN = 0.05                # m
    SCAN_MAX = 8.0                 # m
    FREE_RAY_CLEAR_DIST = 7.5      # m, max-range rays clear only up to here

    # ---- MPPI sampling ----
    MPPI_NUM_SAMPLES = 1500
    MPPI_HORIZON_STEPS = 30        # 30*0.05=1.5s
    TEMPERATURE_LAMBDA = 0.1
    NOISE_SIGMA_V = 0.40           # m/s
    NOISE_SIGMA_W = 0.80           # rad/s

    # ---- cost weights ----
    WEIGHT_PATH_ALIGN = 50.0       # track A* reference line
    WEIGHT_OBSTACLE = 300.0        # footprint/soft-inflation penalty (dominant)
    WEIGHT_TWIRL = 2.0             # suppress high-frequency shake
    WEIGHT_PATH_PROGRESS = 30.0     # carrot: end-pose distance to lookahead
    WEIGHT_HEADING = 15.0          # align chassis with local path heading (lower: turn+forward wins)
    ALIGN_RAMP_STEPS = 10         # first N MPPI steps soften heading/path alignment (turn+forward phase)
    COLLIDE_START_STEP = 2         # first MPPI step checked for lethal collision (give turn/fwd gradient near tight gaps)
    MPPI_BLOCKED_STOP_RATIO = 0.90  # fraction of lethal-colliding samples that triggers a safety stop
    MPPI_BLOCKED_STOP_S_MIN = 5000.0  # cost threshold that distinguishes genuine blocked state

    # ---- global planner ----
    PATH_RESAMPLE_DS = 0.05        # m
    HORIZON_LENGTH = 2.0           # m, forward reference window
    LOOKAHEAD_DIST = 1.2           # m, carrot point ahead of robot on path
    A_STAR_GOAL_TOL = 0.10         # m
    GOAL_RAY_CLAMP = 6.0           # m, upper bound before local-map safe clamp
    # Desired center-to-obstacle clearance. It is a soft preference only; hard
    # feasibility comes from the matching rectangular footprint check in A*.
    A_STAR_CLEARANCE_GOAL = 0.50   # m
    A_STAR_NARROW_WEIGHT = 12.0    # extra A* traversal cost per m of clearance shortfall
    A_STAR_INFLATE_WEIGHT = 2.0    # extra weight on soft-inflated cost so A* stays centered
    # ---- micro-unwedge reflex (deadlock breaker) ----
    UNWEDGE_DIST_MIN = 0.5        # m; only intervene when still far from goal
    UNWEDGE_STUCK_FRAMES = 15     # consecutive near-zero MPPI frames to trigger
    UNWEDGE_V_EPS = 0.02         # m/s; below this with small w counts as stuck
    UNWEDGE_W_EPS = 0.05         # rad/s
    UNWEDGE_REVERSE_FRAMES = 10  # forced reverse duration (0.5s @ 20Hz)
    UNWEDGE_REVERSE_V = -0.15    # m/s forced reverse speed
    RECOVERY_TURN_FRAMES = 10    # turn-in-place frames if reverse space is unsafe
    RECOVERY_TURN_WZ = 0.80      # rad/s
    RECOVERY_TURN_CHECK_ANGLE = 0.35  # rad, candidate turn clearance probe
    RECOVERY_AVOID_TTL = 4.0     # s, keep a failed corridor out of the next A*
    RECOVERY_AVOID_RADIUS = 0.25 # m, temporary forbidden corridor radius
    RECOVERY_AVOID_MIN_AHEAD = 0.20  # m, never forbid the robot's own cell
    RECOVERY_AVOID_MAX_AHEAD = 0.85  # m, only block the failed local neck
    # ---- return-to-shelf maneuver (after delivery, before replanning back) ----
    RETURN_BACKUP_DIST = 0.40        # m, reverse away from the table first
    RETURN_BACKUP_SPEED = -0.40      # m/s, rearward retreat
    RETURN_SLIDE_LEFT_DIST = 0.80    # m, robot-left offset toward open floor
    RETURN_OPEN_GOAL_TOL = 0.25      # m, arrival tolerance for temporary open goal
    RETURN_OPEN_GOAL_YAW_TOL = 0.35  # rad, yaw tolerance at temporary open goal
    # ---- arrival ----
    GOAL_TOLERANCE = 0.25          # m
    GOAL_YAW_TOL = 0.15            # rad

    # ---- footprint sampling ----
    FOOTPRINT_EDGE_SAMPLES = 12    # points on rectangle edges + center line

    def validate(self):
        assert self.ROBOT_LENGTH > 0 and self.ROBOT_WIDTH > 0
        assert self.CONTROL_DT > 0
        assert self.MAP_RESOLUTION > 0
        assert self.MPPI_NUM_SAMPLES > 0 and self.MPPI_HORIZON_STEPS > 0
        assert 0.0 < self.OCC_DECAY < 1.0
        assert self.LETHAL_COST == 253
        return True


cfg = NavConfig()
cfg.validate()
