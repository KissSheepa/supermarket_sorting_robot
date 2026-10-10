# -*- coding: utf-8 -*-
"""arm_stow 单元测试（纯逻辑，无 ROS）：收纳姿态 FK 落点、关节限位、命令构造。

运行：
  python tests/test_arm_stow.py
"""

import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np

from common.arm_stow import (ARM_JOINT_TOL, STOW_GRIP, STOW_L, STOW_R,
                             UNSTOW_R, apply_stow, apply_unstow_right,
                             joint_error, stow_cmd, unstow_cmd,
                             stow_fk_positions)
from kinematics.mmk2_kdl import MMK2Kdl

_PASS = 0
_FAIL = []


def check(name, cond, detail=""):
    global _PASS
    if cond:
        _PASS += 1
        print(f"  PASS {name}")
    else:
        _FAIL.append(name)
        print(f"  FAIL {name} {detail}")


# 关节限位（来自 arm_kdl.py，实机 FK 同源）
JL = np.array([[-3.151, 2.08], [-2.963, 0.181], [-0.094, 3.161],
               [-3.012, 3.012], [-1.859, 1.859], [-3.017, 3.017]])


def in_limits(q):
    return bool(np.all(q >= JL[:, 0]) and np.all(q <= JL[:, 1]))


def main():
    print("== 1. 关节限位 ==")
    check("STOW_L in limits", in_limits(STOW_L), str(STOW_L))
    check("STOW_R in limits", in_limits(STOW_R), str(STOW_R))
    check("UNSTOW_R in limits", in_limits(UNSTOW_R), str(UNSTOW_R))

    print("== 2. 收纳姿态 FK 落点（车身投影内） ==")
    pos = stow_fk_positions()
    pL, pR = pos["left"], pos["right"]
    print(f"    L={np.round(pL, 3)} R={np.round(pR, 3)}")
    # 前向 <= 0.15m（默认 0 位前伸 0.41m，收纳必须大幅收回）
    check("L fwd <= 0.15", float(pL[0]) <= 0.15, str(pL[0]))
    check("R fwd <= 0.15", float(pR[0]) <= 0.15, str(pR[0]))
    # 横向贴车身两侧（底盘半宽约 0.20，收纳端点在 0.25~0.35 之间可接受）
    check("L |y| in [0.20,0.40]", 0.20 <= abs(float(pL[1])) <= 0.40, str(pL[1]))
    check("R |y| in [0.20,0.40]", 0.20 <= abs(float(pR[1])) <= 0.40, str(pR[1]))
    # 高度在底盘与货架板之间（0.6~1.3m），不触地也不顶货架
    check("L z in [0.6,1.3]", 0.6 <= float(pL[2]) <= 1.3, str(pL[2]))
    check("R z in [0.6,1.3]", 0.6 <= float(pR[2]) <= 1.3, str(pR[2]))
    # 左右对称（x/z 相同，y 相反）
    check("symmetry", (abs(pL[0] - pR[0]) < 1e-3 and abs(pL[2] - pR[2]) < 1e-3
                       and abs(pL[1] + pR[1]) < 1e-3))

    print("== 3. 命令构造 ==")
    lc, rc = stow_cmd()
    check("stow_cmd left len7", len(lc) == 7 and lc[:6] == STOW_L)
    check("stow_cmd right len7", len(rc) == 7 and rc[:6] == STOW_R)
    check("stow_cmd grip closed", lc[6] == STOW_GRIP and rc[6] == STOW_GRIP)
    uc = unstow_cmd()
    check("unstow_cmd right", uc[:6] == UNSTOW_R and uc[6] == STOW_GRIP)

    tc = np.zeros(19)
    apply_stow(tc)
    check("apply_stow left", list(tc[5:11]) == STOW_L)
    check("apply_stow right", list(tc[12:18]) == STOW_R)
    check("apply_stow grips", tc[11] == STOW_GRIP and tc[18] == STOW_GRIP)
    apply_unstow_right(tc)
    check("apply_unstow_right", list(tc[12:18]) == UNSTOW_R and tc[18] == STOW_GRIP)
    check("left stays stowed", list(tc[5:11]) == STOW_L)

    print("== 4. 收敛判定 ==")
    check("joint_error 0", joint_error(STOW_L, STOW_L) == 0.0)
    check("joint_error tol", joint_error(np.array(STOW_L) + 0.05, STOW_L) < ARM_JOINT_TOL)

    print(f"\nRESULT: {_PASS} passed, {len(_FAIL)} failed")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
