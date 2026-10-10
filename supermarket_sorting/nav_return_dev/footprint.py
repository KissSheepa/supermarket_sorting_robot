# -*- coding: utf-8 -*-
"""Shared rectangular footprint helpers for planning and control."""
import numpy as np


def rectangle_footprint_offsets(cfg):
    """Return the same body samples for A* feasibility and MPPI collision."""
    half_length = (cfg.ROBOT_LENGTH + 2.0 * cfg.FOOTPRINT_PADDING) / 2.0
    half_width = (cfg.ROBOT_WIDTH + 2.0 * cfg.FOOTPRINT_PADDING) / 2.0
    corners = np.array([
        [half_length, half_width],
        [half_length, -half_width],
        [-half_length, -half_width],
        [-half_length, half_width],
    ], dtype=np.float64)
    points = []
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    per_edge = max(2, cfg.FOOTPRINT_EDGE_SAMPLES // 4)
    for start, end in edges:
        for ratio in np.linspace(0.0, 1.0, per_edge, endpoint=True):
            points.append(corners[start] * (1.0 - ratio) + corners[end] * ratio)
    for longitudinal in np.linspace(-half_length, half_length, 4, endpoint=True):
        points.append([longitudinal, 0.0])
    return np.asarray(points, dtype=np.float64)


def transform_footprint(x, y, yaw, offsets):
    """Transform local footprint offsets into world coordinates."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    px = x + cos_yaw * offsets[:, 0] - sin_yaw * offsets[:, 1]
    py = y + sin_yaw * offsets[:, 0] + cos_yaw * offsets[:, 1]
    return px, py
