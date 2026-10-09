# -*- coding: utf-8 -*-
"""arm_stow：机械臂“收纳 / 展开”姿态（纯常量 + 命令构造，无 ROS 依赖）。

根因（2026-08-11 实机确认）：
  机器人双臂在默认 0 位时，端点在 footprint 前向 0.41m 处（MMK2Kdl FK 实测）；
  从 D 货架去送货区会经过 corridor_right_board（x≈0.50~0.56、高 1.5m 的竖板）北角，
  前伸 0.41m 的手臂会挂住挡板，导致机器人在同一位置反复卡死 / 反复微调姿态浪费时间。

解决方案：运输段把双臂收到车身投影内（端点 |x|<=0.13m、|y|≈0.29m、z≈1.0m），
  到送货区再展开右臂放货；放完货返回货架前再次收纳。

收纳姿态由 MMK2Kdl FK 暴力搜索选出（满足关节限位、端点落在车身投影内、左右对称）：
  STOW_L = [1.6, -0.4, 0.16, 0.0, 0.3, 0.0]   -> 左端点 (0.124, +0.291, 0.994)
  STOW_R = [-1.6, -0.4, 0.16, 0.0, -0.3, 0.0] -> 右端点 (0.124, -0.291, 0.994)
  展开姿态 = client 抓取/放货用的 INIT_ARM_R（与旧流程运输姿态一致，放货行为不变）。
"""

from __future__ import annotations

from typing import List, Sequence

# ---- 收纳姿态（6 关节，与 /left|right_arm_forward_position_controller/commands 前 6 维对应） ----
STOW_L: List[float] = [1.6, -0.4, 0.16, 0.0, 0.3, 0.0]
STOW_R: List[float] = [-1.6, -0.4, 0.16, 0.0, -0.3, 0.0]

# ---- 放货 / 抓取展开姿态（右臂，与 client_task_1.INIT_ARM_R 一致） ----
UNSTOW_R: List[float] = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223]

# 夹爪闭合目标（与 client GRIP_CLOSE 一致；运输时保持夹紧防掉货）
STOW_GRIP: float = 0.08

# 收纳 / 展开允许的关节收敛误差（rad），与 client DEPLOY_JOINT_TOL 同级
ARM_JOINT_TOL: float = 0.08


def stow_cmd(grip: float = STOW_GRIP) -> tuple[List[float], List[float]]:
    """返回 (left_cmd7, right_cmd7)，7 维 = 6 关节 + 夹爪。"""
    return list(STOW_L) + [grip], list(STOW_R) + [grip]


def unstow_cmd(grip: float = STOW_GRIP) -> List[float]:
    """返回右臂展开命令 7 维（放货 / 抓取姿态）。"""
    return list(UNSTOW_R) + [grip]


def apply_stow(tc, grip: float = STOW_GRIP) -> None:
    """就地修改 19 维目标指令 tc：双臂收纳 + 夹爪闭合。

    tc 布局与 client_task_1 一致：
      [0,1]=base, [2]=slide, [3,4]=head, [5:11]=左臂6关节, [11]=左夹爪,
      [12:18]=右臂6关节, [18]=右夹爪
    """
    tc[5:11] = STOW_L
    tc[11] = grip
    tc[12:18] = STOW_R
    tc[18] = grip


def apply_unstow_right(tc, grip: float = STOW_GRIP) -> None:
    """就地修改 tc：右臂展开到放货 / 抓取姿态（左臂保持收纳）。"""
    tc[12:18] = UNSTOW_R
    tc[18] = grip


def joint_error(meas: Sequence[float], target: Sequence[float]) -> float:
    """6 关节测量与目标的逐关节最大误差（rad）。"""
    return max(abs(a - b) for a, b in zip(meas, target))


def stow_fk_positions() -> dict:
    """用 MMK2Kdl FK 计算收纳姿态下左右端点位置（供单元测试 / 调试）。"""
    import numpy as np
    from kinematics.mmk2_kdl import MMK2Kdl
    kdl = MMK2Kdl()

    def ee(q6, index):
        q = np.concatenate([[0.0], q6])
        res = kdl.forward_kinematics(q, index=index)
        T = res[0] if index == "left" else res[1]
        return np.asarray(T[:3, 3])

    return {"left": ee(STOW_L, "left"), "right": ee(STOW_R, "right")}
