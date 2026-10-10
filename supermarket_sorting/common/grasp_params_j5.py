#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-5 grasp parameter table (j5): 按商品类别微调夹爪张开大小与抓取高度.

纯 python 模块(无 rclpy), 可在任意机器单测.

官方评分点(DG-202606 比赛方案 V2.0, 技术评价分第 2 项):
  * 抓取动作是否精准、高效, 能否应对不同物品的抓取点差异;
  * 放置物品的姿态是否正确(如瓶子正立);
  * ArUco 码是否深度用于抓取前的最终位姿微调.

本模块借鉴 Warehouse_handling_robot/poses.yaml 的"按物体配置抓取参数"组织方式,
把 9 类商品的夹爪开合/深度/高度/放置参数做成一张表, 供抓取状态机查询。

默认参数沿用固定 baseline 的已验证值(kele 抓取流程), 每类只调整与物体几何
相关的差异项; 最终数值需要在服务器仿真中按类实测校准(每类 5 件随机分布)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# ---- 官方货架层高(米) ----
LEVEL_Z_BANDS = {
    "L1": (0.45, 0.70),
    "L2": (0.75, 1.02),
    "L3": (1.05, 1.35),
}
# ArUco 贴在货位下沿, 商品中心约在 marker z 上方该偏移处
GRASP_MARKER_Z_OFFSET = 0.06
# 3D 到 2D 表面点补偿(与 client_task.py 的表面点->中心点补偿一致)
VISION_SURFACE_TO_CENTER_FWD = 0.0265

# 9 类商品(顺序与 yolo_backend.CLASS_NAMES 一致)
KINDS = [
    "sanmingzhi", "heweidao", "shupian", "zhijin", "maidong",
    "kouxiangtang", "pingguo", "chengzi", "kele",
]


def level_of_z(z: float) -> Optional[str]:
    for level, (lo, hi) in LEVEL_Z_BANDS.items():
        if lo <= float(z) <= hi:
            return level
    return None


def slide_for_z(z: float) -> float:
    """由商品高度选推荐 slide(躯干升降)值(阶段二实测档位)."""
    if float(z) >= 1.10:
        return 0.35        # L3 顶层: 躯干下降
    if float(z) >= 0.75:
        return 0.11        # L2 中层(已验证值)
    return 0.35            # L1 底层: 躯干下降


def slide_from_marker_z(marker_z: float) -> Tuple[float, Optional[str]]:
    """ArUco 高度 -> 抓取高度 -> slide; 返回 (slide, level).

    根据 ArUco 码的 z 控制爪子高低(评分点), 供抓取前位姿微调用。
    """
    grasp_z = float(marker_z) + GRASP_MARKER_Z_OFFSET
    level = level_of_z(grasp_z)
    if level is None:
        # 带外兜底: 用最近档位
        level = "L2" if grasp_z < 1.05 else "L3"
    return slide_for_z(grasp_z), level


@dataclass
class KindGraspParams:
    """一类商品的抓取/放置参数(所有长度单位为米, 夹爪 0..1 开度)."""

    kind: str
    # 夹爪: 1.0 = 全开(接近前), close = 夹紧目标开度(按物体粗细微调)
    grip_open: float = 1.00
    grip_close: float = 0.08
    # 部署位相对物体中心的偏移(FOOTPRINT 前向/侧向/高度)
    deploy_offset: Tuple[float, float, float] = (-0.011, -0.220, -0.010)
    # creep 停止时夹爪深入物体中心线后的余量(越窄的物体越小)
    creep_stop_dy: float = 0.035
    lift_amount: float = 0.05
    slide_grasp: float = 0.11
    place_slide: float = 0.11
    level_hint: str = "L2"
    place_upright: bool = True      # 放置姿态: 瓶子正立等
    note: str = ""

    @property
    def deploy_offset_arr(self):
        return list(self.deploy_offset)


DEFAULT_PARAMS = KindGraspParams(kind="default", note="fallback: 固定 baseline 值")

# 每类商品的微调表。数值为初始工程值, 需在服务器仿真按类实测校准。
GRASP_PARAMS = {
    "kele":    KindGraspParams("kele", grip_close=0.08, creep_stop_dy=0.035,
                               slide_grasp=0.11, place_slide=0.11, level_hint="L2",
                               note="细瓶, 夹紧+正立放置"),
    "maidong": KindGraspParams("maidong", grip_close=0.07, creep_stop_dy=0.032,
                               slide_grasp=0.11, place_slide=0.11, level_hint="L2",
                               note="饮料瓶, 略宽于可乐"),
    "zhijin":  KindGraspParams("zhijin", grip_close=0.10, creep_stop_dy=0.010,
                               slide_grasp=0.11, place_slide=0.11, level_hint="L2",
                               note="纸巾软盒, 防压"),
    "kouxiangtang": KindGraspParams("kouxiangtang", grip_close=0.09, creep_stop_dy=0.038,
                                    slide_grasp=0.35, place_slide=0.20, level_hint="L3",
                                    note="口香糖方盒"),
    "shupian": KindGraspParams("shupian", grip_close=0.12, creep_stop_dy=0.045,
                               slide_grasp=0.35, place_slide=0.20, level_hint="L1",
                               note="薯片袋, 加宽"),
    "heweidao": KindGraspParams("heweidao", grip_close=0.10, creep_stop_dy=0.040,
                                slide_grasp=0.35, place_slide=0.20, level_hint="L1",
                                note="盒装"),
    "sanmingzhi": KindGraspParams("sanmingzhi", grip_close=0.11, creep_stop_dy=0.042,
                                  slide_grasp=0.35, place_slide=0.20, level_hint="L1",
                                  note="三明治袋装"),
    "pingguo": KindGraspParams("pingguo", grip_close=0.08, creep_stop_dy=0.035,
                               slide_grasp=0.35, place_slide=0.20, level_hint="L3",
                               note="苹果圆果"),
    "chengzi": KindGraspParams("chengzi", grip_close=0.07, creep_stop_dy=0.033,
                               slide_grasp=0.35, place_slide=0.20, level_hint="L3",
                               note="橙子圆果"),
}


def resolve(kind: str) -> KindGraspParams:
    return GRASP_PARAMS.get(kind, DEFAULT_PARAMS)


def selftest() -> int:
    """校验: 9 类参数完整且数值在合理范围, 高度映射符合层带."""
    ok = True
    for kind in KINDS:
        p = resolve(kind)
        issues = []
        if not (0.0 < p.grip_close < p.grip_open <= 1.0 + 1e-9):
            issues.append(f"grip range bad ({p.grip_open}, {p.grip_close})")
        if p.slide_grasp not in (0.11, 0.35):
            issues.append(f"slide_grasp={p.slide_grasp}")
        if not (0.0 < p.creep_stop_dy < 0.10):
            issues.append(f"creep_stop_dy={p.creep_stop_dy}")
        if not (0.0 < p.lift_amount < 0.20):
            issues.append(f"lift_amount={p.lift_amount}")
        if issues:
            ok = False
            print(f"[grasp_params_j5] {kind}: FAIL {issues}")
        else:
            print(f"[grasp_params_j5] {kind}: grip_close={p.grip_close:.2f} "
                  f"slide={p.slide_grasp:.2f} level={p.level_hint} {p.note}")
    # 高度控制: marker z -> slide
    for mz, expect in ((0.50, 0.35), (0.84, 0.11), (1.17, 0.35)):
        slide, level = slide_from_marker_z(mz)
        print(f"[grasp_params_j5] marker_z={mz:.2f} -> slide={slide:.2f} level={level}")
        if slide != expect:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())