#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-3 mapping core: product kind <-> shelf slot (ArUco) association.

Pure-python module (no rclpy import) so it can be unit-tested on any machine.
The ROS2 wrapper lives in ``client_task.py``.

Official rules (DG-202606 双镜像说明 V2.0):
  * ArUco 码固定绑定货位(45 个, DICT_4X4_50, id 0..44), 不随商品随机化移动;
  * 商品与货位的对应关系每局启动时随机变化;
  * 参赛程序需自主扫描货架, 结合 ArUco 码识别货位,
    建立"商品类别—货位"映射后再执行取货;
  * 不得读取 Server 内部布局文件或其它未公开真值。

因此本模块只依据机器人自身传感器观测来建图:
  /product/detections  -> (kind, world xyz)          YOLO + RGB-D
  /aruco/detections    -> (marker_id, world xyz)     head camera ArUco

关联模型(借鉴 ros2-aruco-object-estimator 的 ROI 归属思想):
  一个商品观测点归属到"同一层(z 带)内水平距离最近"的 ArUco 货位码,
  并对 (kind, marker_id) 累计加权投票; 建图后给出:
    kind -> best marker / 置信度 / 货位标签 (shelf/level/column)
    marker -> 投票商品类别(反向索引, 抓取前复核用)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ARUCO_JSON = _HERE / "aruco_slots.json"


def _load_aruco_slots(path: Optional[Path] = None) -> Dict[int, dict]:
    """Load the published ID->货位 table (common/aruco_slots.json)."""
    p = Path(path) if path else _ARUCO_JSON
    if not p.is_file():
        return {}
    rows = json.loads(p.read_text(encoding="utf-8"))
    return {int(r["aruco_id"]): r for r in rows}


ARUCO_SLOTS = _load_aruco_slots()

# 官方货架层高约为 0.5m / 0.85m / 1.19m; 商品检测 z 落在对应层带内。
LEVEL_Z_BANDS = {
    "L1": (0.45, 0.70),
    "L2": (0.75, 1.02),
    "L3": (1.05, 1.35),
}

# 货位列横向间距 0.22 m; ArUco 在货位下方, 与商品水平偏差 < 0.35 m 才归属。
DEFAULT_MAX_HORIZ_DIST = 0.35
# 相邻层高差约 0.345 m; z 差 > 0.30 m 视为不同层, 防止跨层误归属。
DEFAULT_MAX_Z_DIFF = 0.30
@dataclass
class SlotEntry:
    """Resolved mapping for one product kind."""

    kind: str
    marker_id: Optional[int] = None
    votes: float = 0.0
    confidence: float = 0.0
    slot: Optional[dict] = None
    samples: int = 0

    @property
    def label(self) -> str:
        if not self.slot:
            return "?"
        return f"{self.slot.get('shelf', '?')}/{self.slot.get('level', '?')}/{self.slot.get('column', '?')}"

    @property
    def mapped(self) -> bool:
        return self.marker_id is not None


def slot_label(marker_id: Optional[int]) -> str:
    """shelf/level/column label of a marker id, or '?'."""
    if marker_id is None:
        return "?"
    s = ARUCO_SLOTS.get(int(marker_id))
    if not s:
        return "?"
    return f"{s['shelf']}/{s['level']}/{s['column']}"


def level_of_z(z: float) -> Optional[str]:
    """Map a world z (m) to the official shelf level, or None."""
    for level, (lo, hi) in LEVEL_Z_BANDS.items():
        if lo <= float(z) <= hi:
            return level
    return None


class SlotMap:
    """Build and query the kind->ArUco货位 mapping from vision observations."""

    def __init__(
        self,
        max_horiz_dist: float = DEFAULT_MAX_HORIZ_DIST,
        max_z_diff: float = DEFAULT_MAX_Z_DIFF,
    ) -> None:
        self.max_horiz_dist = float(max_horiz_dist)
        self.max_z_diff = float(max_z_diff)
        # (kind, marker_id) -> accumulated weight (1/(1+dist))
        self.votes: Counter = Counter()
        # marker_id -> list of associated world points
        self.marker_points: Dict[int, List[np.ndarray]] = defaultdict(list)
        # kind -> list of (world point, marker_id, dist)
        self.observations: Dict[str, List[Tuple[np.ndarray, int, float]]] = defaultdict(list)
        self.n_observations = 0

    # ---- association -------------------------------------------------
    def associate(
        self, kind: str, world_xyz: Iterable[float], markers: Iterable[Tuple[int, Iterable[float]]]
    ) -> Optional[Tuple[int, float]]:
        """Associate one product observation to the nearest SAME-LEVEL marker.

        Returns (marker_id, horizontal_dist) or None when no marker qualifies.
        先按"层带"限定候选(防止同列相邻层误配), 再用 z 加权距离打破同层并列。
        """
        pw = np.asarray(world_xyz, dtype=float)
        prod_level = level_of_z(pw[2])
        best_mid, best_d = None, float("inf")
        for mid, mxyz in markers:
            pm = np.asarray(mxyz, dtype=float)
            d_h = float(np.hypot(pw[0] - pm[0], pw[1] - pm[1]))
            if d_h > self.max_horiz_dist:
                continue
            d_z = abs(pw[2] - pm[2])
            # ArUco 的"层"用固定货位真值表(ARUCO_SLOTS)判定; 检测坐标的 z 不可靠,
            # 会造成跨层错配(pingguo L3 -> L1 marker).
            slot_level = ARUCO_SLOTS.get(int(mid), {}).get("level")
            if prod_level is not None:
                if slot_level is not None:
                    if slot_level != prod_level:
                        continue  # 货位真值层与商品层不同, 不归属
                elif level_of_z(pm[2]) != prod_level:
                    continue  # 真值表缺失时的检测 z 兜底
            elif d_z > self.max_z_diff:
                continue  # 层带未知时的兜底 z 门限
            d = d_h + 0.30 * d_z          # z 加权, 同层内优先更近
            if d < best_d:
                best_d, best_mid = d, int(mid)
        return (best_mid, best_d) if best_mid is not None else None

    def observe(
        self,
        kind: str,
        world_xyz: Iterable[float],
        markers: Iterable[Tuple[int, Iterable[float]]],
    ) -> Optional[Tuple[int, float]]:
        """Record one product observation and vote for its slot."""
        hit = self.associate(kind, world_xyz, markers)
        if hit is None:
            return None
        mid, dist = hit
        weight = 1.0 / (1.0 + dist)
        self.votes[(kind, mid)] += weight
        pw = np.asarray(world_xyz, dtype=float)
        self.marker_points[mid].append(pw)
        self.observations[kind].append((pw, mid, dist))
        self.n_observations += 1
        return hit

    # ---- queries ------------------------------------------------------
    def kind_to_slot(self, kind: str) -> SlotEntry:
        """Best marker for a kind with confidence = votes[best] / total."""
        total = 0.0
        best_mid, best_votes = None, 0.0
        for (k, mid), w in self.votes.items():
            if k == kind:
                total += w
                if w > best_votes:
                    best_votes, best_mid = w, mid
        if best_mid is None or total <= 0.0:
            return SlotEntry(kind=kind)
        return SlotEntry(
            kind=kind,
            marker_id=best_mid,
            votes=best_votes,
            confidence=best_votes / total,
            slot=ARUCO_SLOTS.get(int(best_mid)),
            samples=len(self.observations.get(kind, [])),
        )

    def slot_to_kinds(self, marker_id: int) -> Counter:
        """Product kinds currently voting for a marker (grasp re-check)."""
        out: Counter = Counter()
        for (k, mid), w in self.votes.items():
            if mid == marker_id:
                out[k] += w
        return out

    def mapped_kinds(self) -> List[str]:
        return sorted({k for (k, _mid) in self.votes})

    def markers_for_kind(self, kind: str) -> List[Tuple[int, float]]:
        """All (marker_id, votes) for a kind, best first."""
        items = [(mid, w) for (k, mid), w in self.votes.items() if k == kind]
        return sorted(items, key=lambda kv: kv[1], reverse=True)

    # ---- reporting ----------------------------------------------------
    def summary(self) -> dict:
        return {
            "n_observations": self.n_observations,
            "n_markers_seen": len(self.marker_points),
            "kinds_mapped": len(self.mapped_kinds()),
            "mapping": {
                k: {
                    "marker_id": e.marker_id,
                    "slot": e.label,
                    "confidence": round(e.confidence, 3),
                    "samples": e.samples,
                }
                for k, e in ((k, self.kind_to_slot(k)) for k in self.mapped_kinds())
            },
        }

    def report(self) -> str:
        lines = ["[slot_map_j3] kind->ArUco mapping (from robot vision only):"]
        for kind in self.mapped_kinds():
            e = self.kind_to_slot(kind)
            lines.append(
                f"  {kind:14s} -> marker={e.marker_id} slot={e.label} "
                f"conf={e.confidence:.2f} samples={e.samples}"
            )
        if not self.mapped_kinds():
            lines.append("  (no mapping yet)")
        return "\n".join(lines)

    def reset(self) -> None:
        self.votes.clear()
        self.marker_points.clear()
        self.observations.clear()
        self.n_observations = 0

# ---------------------------------------------------------------------------
# synthetic scene + offline validation (used by --selftest, no server layout)
# ---------------------------------------------------------------------------
SHELF_C2_X = {"A": -1.735, "B": -0.850, "C": 0.035, "D": 0.920, "E": 1.805}
SHELF_Y = 3.243
COL_X = {"C1": -0.22, "C2": 0.0, "C3": 0.22}
LEVEL_Z = {"L1": 0.55, "L2": 0.895, "L3": 1.229}
MARKER_Z_OFFSET = -0.05          # ArUco 贴在货位下方
KINDS = [
    "sanmingzhi", "heweidao", "shupian", "zhijin", "maidong",
    "kouxiangtang", "pingguo", "chengzi", "kele",
]


def make_synthetic_scene(seed: int = 7):
    """A random 45-slot scene (kind per slot). Only for offline validation."""
    rng = np.random.default_rng(seed)
    slots = []
    for si, shelf in enumerate("ABCDE"):
        for li, level in enumerate(("L1", "L2", "L3")):
            for ci, col in enumerate(("C1", "C2", "C3")):
                marker_id = si * 9 + li * 3 + ci
                x = SHELF_C2_X[shelf] + COL_X[col]
                slots.append(
                    {
                        "marker_id": marker_id,
                        "shelf": shelf,
                        "level": level,
                        "column": col,
                        "kind": KINDS[rng.integers(0, len(KINDS))],
                        "product_xyz": np.array([x, SHELF_Y, LEVEL_Z[level]]),
                        "marker_xyz": np.array([x, SHELF_Y, LEVEL_Z[level] + MARKER_Z_OFFSET]),
                    }
                )
    return slots


def selftest(seed: int = 7, noise_xy: float = 0.02, noise_z: float = 0.01) -> dict:
    """Build the map from noisy synthetic observations and score accuracy.

    两种指标(同一 kind 有 5 个货位, 不能按货位逐槽判定):
      1. obs_acc   每条商品观测点是否关联到"自己所在货位"的 ArUco 码;
      2. kind_acc  kind_to_slot 解析出的最优货位是否为该 kind 的真值货位之一。
    """
    rng = np.random.default_rng(seed)
    slots = make_synthetic_scene(seed)
    sm = SlotMap()
    obs_correct, obs_total = 0, 0
    # 模拟"逐架扫描": 每架每次只看到本架 3x3 的标记与商品(带测量噪声)
    for shelf in "ABCDE":
        shelf_slots = [s for s in slots if s["shelf"] == shelf]
        markers = [
            (s["marker_id"], s["marker_xyz"] + rng.normal(0.0, noise_xy * 0.5, 3))
            for s in shelf_slots
        ]
        for _ in range(10):  # 多帧观测积累投票
            for s in shelf_slots:
                p = s["product_xyz"] + np.array(
                    [rng.normal(0.0, noise_xy), rng.normal(0.0, noise_xy), rng.normal(0.0, noise_z)]
                )
                hit = sm.associate(s["kind"], p, markers)
                obs_total += 1
                if hit is not None and hit[0] == s["marker_id"]:
                    obs_correct += 1
                sm.observe(s["kind"], p, markers)

    kind_ok, kind_total = 0, len(KINDS)
    for kind in KINDS:
        e = sm.kind_to_slot(kind)
        if e.mapped:
            gt = {s["marker_id"] for s in slots if s["kind"] == kind}
            if e.marker_id in gt:
                kind_ok += 1
    obs_acc = obs_correct / obs_total if obs_total else 0.0
    kind_acc = kind_ok / kind_total if kind_total else 0.0
    print(f"[slot_map_j3] selftest: observation-assoc {obs_correct}/{obs_total} = {obs_acc:.1%}")
    print(f"[slot_map_j3] selftest: kind-level mapping {kind_ok}/{kind_total} = {kind_acc:.1%}")
    print(sm.report())
    return {"obs_correct": obs_correct, "obs_total": obs_total,
            "obs_acc": obs_acc, "kind_acc": kind_acc}


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="slot_map_j3 offline validation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-xy", type=float, default=0.02)
    parser.add_argument("--noise-z", type=float, default=0.01)
    args = parser.parse_args()
    res = selftest(seed=args.seed, noise_xy=args.noise_xy, noise_z=args.noise_z)
    if res["obs_acc"] >= 0.95 and res["kind_acc"] >= 0.95:
        print("PASS: observation-assoc >= 95% and kind-level mapping >= 95%")
        raise SystemExit(0)
    print("FAIL: thresholds not met")
    raise SystemExit(1)


if __name__ == "__main__":
    _main()
