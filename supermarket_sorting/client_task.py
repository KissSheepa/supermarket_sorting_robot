#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-3 + Stage-5 client (j35): 先自主扫描建图, 再按最近货位抓取.

与 client_task_1.py / client_task_2.py 完全独立, 不改动任何原始文件.
参考官方建议流程(DG-202606 双镜像说明 V2.0):
  对货架进行图像检测, 结合 ArUco 码识别货位, 建立"商品类别—货位"映射;
  保存本局已识别货位, 抓取前再复核 ArUco ID 与商品类别.

流程:
  WAIT_TASK(原地等待手动下发订单) -> E架"从上往下"扫一遍 -> D架"从下往上"扫一遍
  -> C架"从上往下" -> B架"从下往上" -> A架"从上往下"(每个货架只扫一遍, 不回头)
  -> REPORT(订单商品 世界坐标+ArUco货位 汇总) -> 阶段五最近货位迭代抓取
  -> 抓取后转身背离货架并松开(阶段四接入点) -> DONE
扫描方式: 在货架前方较远处(SCAN_APPROACH_Y≈1.04m)停车, 头部俯仰随 slide
          联动扫层(L3=0.16 / L2=-0.35 / L1=-0.55), 摆头区域覆盖 3 层商品与 ArUco;
          货架间导航时头部保持不动, 到下一个货架前再摆头.
扫描到订单里的商品时, 立即把"商品世界坐标 + 对应ArUco货位"写入验证日志文件.

依赖话题:
  /supermarket_sorting/task   std_msgs/String        订单 JSON
  /product/detections         Detection3DArray       商品检测(世界系)
  /aruco/detections           Detection3DArray       ArUco 检测(世界系)
  /slamware_ros_sdk_server_node/odom + /joint_states + /scan  导航/避障

运行(与 scripts/run_stage3_j3.sh 等价):
  python3 client_task_5_j35.py
离线验证(无需 ROS):
  python3 client_task_5_j35.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common.slot_map_j3 import (  # noqa: E402
    DEFAULT_MAX_HORIZ_DIST, SlotMap, ARUCO_SLOTS, level_of_z, slot_label,
)
from common.task_parser import TaskParseError, parse_task  # noqa: E402
from common.grasp_params_j5 import (  # noqa: E402
    resolve as resolve_grasp_params, slide_for_z,
)
from common.arm_stow import (  # noqa: E402
    STOW_L, STOW_R, UNSTOW_R, apply_stow,
)

try:  # ROS2 只在服务器端可用; 本地 --selftest 无需这些导入
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile,
        ReliabilityPolicy,
        DurabilityPolicy,
        qos_profile_sensor_data,
    )
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float64MultiArray, String
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState, LaserScan
    from vision_msgs.msg import Detection3DArray
    from scipy.spatial.transform import Rotation
    from common.control import step_func
    from kinematics.mmk2_kdl import MMK2Kdl
    _ROS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ROS_AVAILABLE = False

# ---- scene constants (world frame, +X east / +Y north). ----
YELLOW_MID_Y = 2.475            # 走廊黄线(导航基准 y)
SCAN_APPROACH_Y = 2.20          # 扫描停车位 y: 距货架(y=3.243)约 1.04m, 让摆头覆盖 3 层
PICK_APPROACH_Y = YELLOW_MID_Y  # 抓取接泊位: 与已验证 j5 一致, 保证机械臂 IK 可达
SCAN_YAW_EAST_BIAS_DEG = 5.0    # 11 偏右 / 0 偏左, 折中取 5 使车头正对货架
SCAN_YAW = math.pi / 2.0 - math.radians(SCAN_YAW_EAST_BIAS_DEG)
SHELF_C2_X = {"A": -1.735, "B": -0.850, "C": 0.035, "D": 0.920, "E": 1.805}
APPROACH_BASE_X = 0.852
APPROACH_X_OFFSET = APPROACH_BASE_X - SHELF_C2_X["D"]   # -0.068
# 扫描顺序: 从启动区直走先到 E 架, 再依次 D/C/B/A
SEARCH_ORDER = ["E", "D", "C", "B", "A"]
PICK_SEARCH_ORDER = ["A", "B", "C", "D", "E"]
# 头部俯仰(rad). 符号约定(见 scripts/look_at_shelf.py 与 head_pitch_joint axis=0 0 -1):
#   负 pitch = 抬头, 正 pitch = 低头. 躯干升到高层后, 头保持接近水平只做小幅对准.
PITCH_BY_LEVEL = {"L1": -0.20, "L2": -0.05, "L3": 0.05}
# 每架只向下扫一次: L3 -> L2 -> L1 (扫完直接输出, 不再二遍验证).
SCAN_LEVELS = ["L3", "L2", "L1"]
# slide(躯干升降, spine FK: 高度 z = 1.406 - slide, 即数值越大越低). 整趟下移 0.2.
# SLIDE_BY_LEVEL = {"L1": 0.50, "L2": 0.31, "L3": 0.16}
SLIDE_BY_LEVEL = {"L1": 0.60, "L2": 0.40, "L3": 0.20}

# 每层驻留时长(放大放缓, 让上下移动充分到位后再采集)
DWELL_PER_LEVEL = 3.0          # s; 单层驻留
LEVEL_SETTLE = 2.5             # s; 每层到达后的稳定等待(在此之后才采信检测)
# ---- 固定货位真值几何(ArUco 码固定绑定货位, 位置固定不变, 官方允许据此识别) ----
SHELF_Y_FIX = 3.243
SHELF_LEVEL_Z = {"L1": 0.548, "L2": 0.895, "L3": 1.229}   # 商品层中心 z
SHELF_C1_X = {"A": -1.955, "B": -1.07, "C": -0.185, "D": 0.700, "E": 1.585}
COL_X_OFF = {"C1": 0.0, "C2": 0.22, "C3": 0.44}           # 列间横向偏移(真值列中心)
ARUCO_Y_FIX = 3.243                                        # 货位(货架面) y
ARUCO_Z_FIX = SHELF_LEVEL_Z                               # ArUco 贴在货位下方, x/z 用真值
SCAN_DURATION = len(SCAN_LEVELS) * DWELL_PER_LEVEL          # s; 单架逐层驻留总时长
NAV_TIMEOUT = 300.0         # s; 单段导航超时
RUN_TIMEOUT = 25.0 * 60.0   # s; 全流程超时
# 服务器固定基线自动任务是单件(count=1), 真实订单>=2件才启动
MIN_ORDER_COUNT = 2
RANDOM_ORDER_COUNT = 5     # 每次随机抽取 5 件订单(剔除纸巾), 便于完整闭环验证
# 商品检测最低置信度(过滤 YOLO 误检, 如 E 架上不存在的"kele")
DET_SCORE_MIN = 0.5
# 单架扫描中, 某商品的样本帧数低于此下限时, 视为偶发误检, 不参与最终"商品->货位"择优
MIN_MAP_SAMPLES = 30
SAME_LEVEL_X_SPLIT = 0.15

# ---- nav/避障参数(阶段四的完整规划不在此文件范围). ----
AVOID_TRIGGER_DIST = 0.55
AVOID_FWD_SECTOR = 0.45
AVOID_SIDE_SECTOR = 1.2
AVOID_SPEED = 0.30
AVOID_ANG = 0.7

# ---- Stage-5 grasp parameters (reuse the verified j5 interface). ----
GRASP_YAW_EAST_BIAS_DEG = 5.0
GRASP_YAW = math.pi / 2.0 - math.radians(GRASP_YAW_EAST_BIAS_DEG)
GRASP_ROT = np.eye(3)
GRASP_ROT_BY_KIND = {
    "sanmingzhi": np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]),
}
GRASP_X_OFFSET_BY_KIND = {
    "chengzi": 0.03,
}
GRIP_OPEN = 1.0
GRIP_FALLBACK_CLOSE = 0.08
GRIP_OPEN_BY_KIND = {
    "kele": 0.90,
    "maidong": 0.90,
    "chengzi": 1.00,
    "pingguo": 0.92,
    "kouxiangtang": 0.95,
    "zhijin": 1.00,
    "heweidao": 0.98,
    "sanmingzhi": 1.00,
    "shupian": 1.00,
}
GRASP_SLIDE_BY_LEVEL = {
    "L1": 0.60,
    "L2": 0.40,
    "L3": 0.20,
}
ARM_READY_TOL = 0.12
DEPLOY_ROT_TOL = 0.35
RETRACT_SPEED = 0.25
DEPLOY_TIMEOUT = 60.0
DEPLOY_CART_TOL = 0.065
CREEP_SPEED = 0.25
CREEP_YAW_KP = 4.0
CLOSE_DWELL = 0.8
LIFT_DWELL = 0.8
TURN_AWAY_DWELL = 1.0
RELEASE_OPEN_DWELL = 1.0

# 持物抓取闭环: 夹爪命令到位后保持 CLOSE_CMD_DWELL, 确保物理咬合;
# 缺反馈或超时则容错继续, 避免假闭合就收臂/导航导致松手.
CLOSE_CMD_TOL = 0.05        # rad; 夹爪命令到位容差
CLOSE_CMD_DWELL = 0.5       # s; 命令到位后再保持
GRIP_CLOSE_TIMEOUT = 8.0    # s; 夹爪闭合等待超时(容错继续)
HOLD_SLEW = 0.7             # rad/s; 持物收臂/运输时降低关节速度, 防惯性甩掉

# ---- 阶段4 送货/返程 常量(配送台位置可用环境变量覆盖). ----
def _env_vec(name, default):
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    vals = [float(x.strip()) for x in raw.split(",")]
    if len(vals) != len(default):
        raise ValueError(f"{name} must be 'x,y', got {raw!r}")
    return vals


def _fmt(x):
    """Format a scalar for navdiag lines; '-'-safe when the field is empty."""
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "-"


# 单排错开摆放: 左右两列 x 分两档, 前后(南北)轻微错开成棋盘, 避免同排挤在一起.
# 槽位 y 是导航停点(基座)目标, 比落点略偏北(车头前伸后落到桌面内).
DELIVERY_SLOTS = [
    [-1.62, -3.11],
    [-1.76, -3.05],
    [-1.90, -2.99],
    [-2.18, -2.87],
    [-2.04, -2.93],
]
DELIVERY_POINT = _env_vec("SUPERMARKET_DELIVERY_POINT", DELIVERY_SLOTS[0])
DELIVERY_YAW = float(os.environ.get("SUPERMARKET_DELIVERY_YAW", str(-math.pi / 2.0)))
RETURN_SHELF_POINT = _env_vec(
    "SUPERMARKET_RETURN_SHELF_POINT",
    [SHELF_C2_X["A"] + APPROACH_X_OFFSET, SCAN_APPROACH_Y])
RETURN_SHELF_YAW = float(
    os.environ.get("SUPERMARKET_RETURN_SHELF_YAW", str(SCAN_YAW)))

# 抓取失败安全网(基座被抵住/末端够不到商品时判定失败并重规划)
CREEP_MOVE_EPS = 0.03          # m; 位移低于此值视为"没动"
CREEP_STUCK_TIMEOUT = 6.0      # s; 无有效位移超过此时长判定为卡住
CARRY_HOLD_TIMEOUT = 600.0     # s; 持物收臂超时(保护, 正常远小于此)

# 放置(配送台桌面上方): 车头正前方 PLACE_FWD_DIST, 高度 PLACE_Z
PLACE_FWD_DIST = float(os.environ.get("SUPERMARKET_PLACE_FWD_DIST", "0.42"))
# 桌面 top 面 z=0.767(见 retail_competition.xml delivery_table_top z=0.742+0.025),
# 官方 place 侧 site delivery_target 标在 z=0.807. 这里把末端(夹爪)目标高度设为
# 0.81, 使夹住的商品底面刚好贴/略高于桌面, 既不悬浮也不穿模.
PLACE_Z = float(os.environ.get("SUPERMARKET_PLACE_Z", "0.85"))
# 放置时身体升降高度(桌面水平面对应值): 抓取货架层不同会留下不同 slide(0.2/0.4/0.6),
# 若保持抓取 slide 去放置, IK 用错误 target_height 无法把末端降到桌面(z=0.62).
# 统一抬升/下降到该值, 让末端与夹爪对齐桌面水平面(略高于桌面, 避免碰撞).
# 与官方已验证放置几何保持一致: 停在桌北侧, slide=0.17 时商品贴近桌面且不与台面初始重叠.
PLACE_SLIDE = float(os.environ.get("SUPERMARKET_PLACE_SLIDE", "0.17"))
PLACE_OPEN_DWELL = 1.2         # s; 夹爪张开后停留, 物品落到桌面
PLACE_SETTLE_DWELL = 1.5       # s; 商品稳定后再允许底盘/手臂运动
PLACE_CLEAR_LIFT = 0.38        # m; 收臂前先抬升身体, 把展开的右臂抬高离桌面商品
PLACE_CLEAR_MIN = -0.04        # slide 下限(官方 SLIDE_LIFT=-0.04), 抬升上限
PLACE_RETREAT_DIST = 0.20      # m; 收臂后倒车离开桌面
PLACE_RETREAT_SPEED = 0.20     # m/s; 收臂后倒车离开桌面(放慢防失衡/防止挂桌)
PLACE_STOW_DWELL = 1.0         # s; 放置完成后收臂停留
PLACE_STOW_TOL = 0.15          # rad; 收臂到位容差
PLACE_ARM_TOL = 0.10           # m; 末端到位容差
PLACE_TIMEOUT = 30.0           # s; 单次放置超时
# 放置前先把车头对准桌面(DELIVERY_YAW), 否则后退返程会斜着别到墙
PLACE_ALIGN_TOL = 0.03         # rad; 对准容差(约1.7°)
PLACE_ALIGN_KP = 2.0           # 角速度比例增益
PLACE_ALIGN_WZ = 1.0           # rad/s; 对准最大角速度

# 抓不住的商品: 下单时从订单中剔除为空(纸巾 zhijin 抓不住)
GRASP_UNREACHABLE_KINDS = {"zhijin"}

# 目标控制向量: [base_lin, base_ang, slide, head_yaw, head_pitch, arms(0)]
# j3 只做识别/建图/验证, 不控制手臂(抓取), 手臂段恒为 0.
# 初始姿态: 躯干升到最高(slide=L3), 头部接近水平(0), 再按层序降扫.
INIT_SLIDE = SLIDE_BY_LEVEL["L3"]
INIT_HEAD_PITCH = 0.0


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi
# top-level phases
WAIT_TASK, NAV_TO_SHELF, SCAN_SHELF, NEXT_SHELF, REPORT, PLAN_PICK, \
    NAV_PICK, ARM_READY, DEPLOY_ARM, CREEP, CLOSE_GRIP, LIFT, RETRACT, \
    STOW_GRIP_HOLD, NAV_TO_DELIVERY, PLACE_ITEM, NAV_RETURN_TO_NEXT, \
    DONE = range(18)
PHASE_NAME = {
    WAIT_TASK: "wait-task", NAV_TO_SHELF: "nav->shelf", SCAN_SHELF: "scan-shelf",
    NEXT_SHELF: "next-shelf", REPORT: "report", PLAN_PICK: "plan-nearest-pick",
    NAV_PICK: "nav->pick", ARM_READY: "arm-ready", DEPLOY_ARM: "deploy-arm",
    CREEP: "creep-in", CLOSE_GRIP: "close-grip", LIFT: "lift",
    RETRACT: "retract-out", STOW_GRIP_HOLD: "stow-grip-hold",
    NAV_TO_DELIVERY: "nav->delivery", PLACE_ITEM: "place-item",
    NAV_RETURN_TO_NEXT: "nav->return-next", DONE: "done",
}

JOINT_NAMES = [
    "slide_joint", "head_yaw_joint", "head_pitch_joint",
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
    "left_arm_eef_gripper_joint",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
    "right_arm_eef_gripper_joint",
]
# ROS 可用时才以 rclpy Node 为基类; 本地 --selftest 仅导入不做实例化。
_NodeBase = Node if _ROS_AVAILABLE else object


class Stage35Client(_NodeBase):
    """ROS2 节点: 阶段三逐架建图 -> 阶段五按最近货位抓取."""

    def __init__(self):
        super().__init__("stage35_mapping_grasp_client_j35")
        self.slot_map = SlotMap()
        self.kdl = MMK2Kdl() if _ROS_AVAILABLE else None

        self.tc = np.zeros(19)
        apply_stow(self.tc, GRIP_OPEN)
        self.tc[2] = INIT_SLIDE
        self.tc[4] = INIT_HEAD_PITCH
        self.action = self.tc.copy()
        self.joint_move_ratio = np.ones(19)
        self.tc_prev = self.tc.copy()
        self.joint_slew = 1.2

        self.base_xy = None
        self.base_yaw = 0.0
        self.jpos = None
        self.laser = None
        self.laser_angle_min = -math.pi
        self.laser_angle_inc = 2.0 * math.pi / 360.0

        # 订单状态
        self.task_ready = False
        self.run_prefix = ""
        self.targets = []
        self.pending_counts = Counter()

        # 感知缓冲
        self.det_buf = deque(maxlen=200)     # (kind, world xyz)
        self.aruco_buf = deque(maxlen=400)   # (marker_id, world xyz)
        # 阶段2式"订单商品坐标": kind -> deque[(world xyz, t)]
        self.order_points = defaultdict(deque)
        # 当前货架扫描期间的订单商品检测样本(用于"扫到即保存"日志)
        self.level_samples = defaultdict(list)
        # 每个订单商品最终确认的发现: kind -> dict(shelf/coords/marker/slot/samples)
        self.final = {}
        self.findings = []
        self.verify_log_path = None
        self.all_log_path = None
        self.order_log_path = None

        # Stage-5 state
        self.current_finding = None
        self.current_kind = None
        self.cur_params = None
        self.used_markers = set()
        self.object_world = None
        self.deploy_world = None
        self.creep_stop_y = None
        self.arm_target_set = False
        self.pick_route = []
        self.state_t0 = self.now()
        self.last_deploy_log = 0.0

        # 抓取失败安全网 + 持物/送货/放置/返程状态
        self.grab_failed = False
        self.creep_last_xy = None
        self.creep_last_move_t = 0.0
        self.carry_grip_close = GRIP_FALLBACK_CLOSE
        self.grip_closed_t0 = None     # 夹爪命令到位起始时间(用于 CLOSE_CMD_DWELL)
        self.total_det_count = 0
        self.mppi_steer = None
        self.mppi_last = (0.0, 0.0)
        self.mppi_last_t = 0.0
        self.mppi_diag_t = 0.0
        self._return_next_goal = None
        self.place_stage = "align"    # align->deploy->open->settle->clear->stow->retreat
        self.place_world = None
        self.place_rot = None
        self.place_start_xy = None
        self.place_armed = False      # 是否已根据当前末端高度反推出放置 slide
        self.place_slide = None       # 放置时身体升降目标(保持右臂不动按末端高度反推)
        self.place_clear_slide = None # 收臂前抬升身体的目标 slide(让右臂离桌商品)
        self.delivered_count = 0

        # 状态机
        self.phase = WAIT_TASK
        self.nav_t0 = self.now()
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.avoid_dir = 0
        self.shelf_idx = 0
        self.cur_shelf = SEARCH_ORDER[0]
        self.sweep_levels = list(SCAN_LEVELS)         # 每架向下扫一次 L3->L2->L1
        self.scan_lvl_idx = 0          # 当前扫描到第几层
        self.scan_t0 = 0.0             # 本层稳定后的采集起始时间
        self.scan_dets = 0             # 本遍消费的商品检测数(诊断)
        self.level_t0 = 0.0            # 当前层进入时间
        self.level_settled = False     # 当前层是否已稳定(稳定后才采信检测)
        self.done_ts = None
        self.run_start = None
        self.last_log = 0.0

        self.pos_tol, self.turn_tol = 0.06, 0.03
        self.max_lin, self.max_ang = 1.10, 2.2
        self.rate_hz = 50.0
        self.dt = 1.0 / self.rate_hz
        self.max_lin_acc, self.max_ang_acc = 2.0, 9.0
        self.des_lin = self.des_ang = 0.0
        self.cur_lin = self.cur_ang = 0.0

        # 话题
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.spine_pub = self.create_publisher(
            Float64MultiArray, "/spine_forward_position_controller/commands", 5)
        self.head_pub = self.create_publisher(
            Float64MultiArray, "/head_forward_position_controller/commands", 5)
        self.larm_pub = self.create_publisher(
            Float64MultiArray, "/left_arm_forward_position_controller/commands", 5)
        self.rarm_pub = self.create_publisher(
            Float64MultiArray, "/right_arm_forward_position_controller/commands", 5)

        task_qos = QoSProfile(depth=1,
                              reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/supermarket_sorting/task",
                                 self.task_cb, task_qos)
        self.create_subscription(LaserScan, "/slamware_ros_sdk_server_node/scan",
                                 self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self.odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        self.create_subscription(Detection3DArray, "/product/detections",
                                 self.det_cb, 10)
        self.create_subscription(Detection3DArray, "/aruco/detections",
                                 self.aruco_cb, 10)
        self.create_timer(self.dt, self.tick)
        self.get_logger().info("stage35 client up; waiting for manual task")

    # ---- ROS 回调 ----
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_xy = np.array([p.x, p.y])
        self.base_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]

    def js_cb(self, msg):
        self.jpos = {n: msg.position[i] for i, n in enumerate(msg.name)
                     if i < len(msg.position)}

    def scan_cb(self, msg):
        self.laser = np.asarray(msg.ranges, dtype=float)
        self.laser_angle_min = float(msg.angle_min)
        self.laser_angle_inc = float(msg.angle_increment)

    def task_cb(self, msg):
        """阶段2式订单解析: 复用 common.task_parser.parse_task."""
        try:
            task = parse_task(msg.data)
        except TaskParseError as exc:
            self.get_logger().error(f"[task] json parse failed: {exc}")
            return
        # 服务器固定基线会在启动时自动发布"单件任务"(count=1);
        # 忽略它并就地等待, 收到真实的多件订单(count>=2)才启动.
        # 纸巾(zhijin)抓不住: 下单时直接剔除, 不计入待抓列表.
        self.targets = [t for t in task.targets
                        if t["kind"] not in GRASP_UNREACHABLE_KINDS]
        if len(self.targets) > RANDOM_ORDER_COUNT:
            # 原服务器一次下发全部商品(每类 5 件); 这里先按"种类"去重, 再随机抽取
            # 5 种不同商品, 确保每轮订单的 5 件商品种类互不相同, 避免反复抓同一商品.
            _by_kind = {}
            for _t in self.targets:
                _by_kind.setdefault(_t["kind"], _t)
            _kinds = list(_by_kind.keys())
            if len(_kinds) >= RANDOM_ORDER_COUNT:
                self.targets = [_by_kind[k] for k in random.sample(
                    _kinds, RANDOM_ORDER_COUNT)]
            else:
                self.targets = list(_by_kind.values())
        self.pending_counts = Counter(t["kind"] for t in self.targets)
        if len(self.targets) < MIN_ORDER_COUNT:
            self.get_logger().info(
                f"[task] ignored auto/single task (count={task.count}, "
                f"{dict(task.pending_counts)}); waiting for manual order")
            return
        self.run_prefix = task.run_prefix
        self.order_points.clear()
        self.task_ready = True
        self.run_start = self.now()
        self.final.clear()
        self.findings.clear()
        self.used_markers.clear()
        self.all_log_path = None
        self.order_log_path = None
        self._open_verify_log()
        self.get_logger().info(
            f"[task] parsed {task.count} targets via task_parser; "
            f"random order={len(self.targets)}: {dict(self.pending_counts)}")

    def det_cb(self, msg):
        if self.base_xy is None:
            return
        self.total_det_count += len(msg.detections)
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0].hypothesis
            if hyp.score < DET_SCORE_MIN:
                continue
            kind = str(hyp.class_id)
            pos = det.results[0].pose.pose.position
            xyz = np.array([pos.x, pos.y, pos.z])
            self.det_buf.append((kind, xyz))
            if self.phase != SCAN_SHELF or not self.level_settled:
                continue
            scan_level = self.sweep_levels[self.scan_lvl_idx]
            self.order_points[kind].append((xyz, self.now()))
            if len(self.order_points[kind]) > 50:
                self.order_points[kind].popleft()
            self.level_samples[(kind, scan_level)].append(xyz.copy())
            if len(self.level_samples[(kind, scan_level)]) > 500:
                del self.level_samples[(kind, scan_level)][:100]

    def aruco_cb(self, msg):
        if self.base_xy is None:
            return
        for det in msg.detections:
            if not det.results:
                continue
            try:
                mid = int(det.results[0].hypothesis.class_id)
            except (TypeError, ValueError):
                continue
            pos = det.results[0].pose.pose.position
            self.aruco_buf.append((mid, np.array([pos.x, pos.y, pos.z])))

    # ---- 控制输出 ----
    def set_twist(self, lin, ang):
        self.des_lin = float(np.clip(lin, -self.max_lin, self.max_lin))
        self.des_ang = float(np.clip(ang, -self.max_ang, self.max_ang))

    def ramp_twist(self):
        dl = np.clip(self.des_lin - self.cur_lin, -self.max_lin_acc * self.dt, self.max_lin_acc * self.dt)
        da = np.clip(self.des_ang - self.cur_ang, -self.max_ang_acc * self.dt, self.max_ang_acc * self.dt)
        self.cur_lin += dl
        self.cur_ang += da
        self.tc[0], self.tc[1] = self.cur_lin, self.cur_ang

    def smooth_step(self):
        if not np.allclose(self.tc[2:19], self.tc_prev[2:19]):
            dif = np.abs(self.action[2:19] - self.tc[2:19])
            self.joint_move_ratio[2:19] = dif / (np.max(dif) + 1e-6)
            self.joint_move_ratio[2] *= 0.3
            self.tc_prev[:] = self.tc
        step = self.joint_slew * self.dt
        for i in range(2, 19):
            if i in (11, 18):
                # 夹爪直通: 不完全受归一化 ratio 压制, 用固定步长尽快逼近目标,
                # 防止收臂/升降期间夹爪命令被 ratio 拖慢导致松手(抓手命令实际偏松).
                self.action[i] += float(np.sign(self.tc[i] - self.action[i]) *
                                        min(abs(self.tc[i] - self.action[i]), step))
            else:
                self.action[i] = min(max(self.action[i], self.tc[i] - step * self.joint_move_ratio[i]),
                                     self.tc[i] + step * self.joint_move_ratio[i])

    def publish(self):
        tw = Twist()
        tw.linear.x = float(self.tc[0])
        tw.angular.z = float(self.tc[1])
        self.cmd_vel_pub.publish(tw)
        self.spine_pub.publish(Float64MultiArray(data=[float(self.action[2])]))
        self.head_pub.publish(Float64MultiArray(data=[float(self.action[3]), float(self.action[4])]))
        self.larm_pub.publish(Float64MultiArray(
            data=[float(x) for x in self.action[5:11]] + [float(self.action[11])]))
        self.rarm_pub.publish(Float64MultiArray(
            data=[float(x) for x in self.action[12:18]] + [float(self.action[18])]))

    # ---- 导航(带基础避障; 阶段四负责完整规划) ----
    def _laser_clearance(self, a0, a1):
        best = float("inf")
        if self.laser is None:
            return best
        for i, r in enumerate(self.laser):
            a = wrap_to_pi(self.laser_angle_min + i * self.laser_angle_inc)
            if a0 <= a <= a1 and math.isfinite(r) and 0.02 < r < 12.0:
                best = min(best, float(r))
        return best

    def _obstacle_ahead(self):
        return self._laser_clearance(-AVOID_FWD_SECTOR, AVOID_FWD_SECTOR) < AVOID_TRIGGER_DIST

    def _avoid_step(self):
        left = self._laser_clearance(0.2, AVOID_SIDE_SECTOR)
        right = self._laser_clearance(-AVOID_SIDE_SECTOR, -0.2)
        if left == float("inf") and right == float("inf"):
            self.set_twist(-0.15, 0.0)
            return
        fwd = self._laser_clearance(-AVOID_FWD_SECTOR, AVOID_FWD_SECTOR)
        if abs(left - right) > 0.15:
            self.avoid_dir = 1 if left > right else -1
        elif self.avoid_dir == 0:
            self.avoid_dir = 1 if left >= right else -1
        speed = AVOID_SPEED if fwd > 0.35 else AVOID_SPEED * 0.4
        self.set_twist(speed, self.avoid_dir * AVOID_ANG)

    def reset_nav(self):
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.avoid_dir = 0

    def follow_route(self, route, final_yaw):
        if self.nav_idx < len(route):
            target = np.array(route[self.nav_idx], dtype=float)
            delta = target - self.base_xy
            dist = float(np.linalg.norm(delta))
            if dist < self.pos_tol:            # 已在航点: 直接前进(避免原地转向卡死)
                self.nav_idx += 1
                self.nav_mode = "turn"
                self.set_twist(0.0, 0.0)
                return False
            yaw_err = wrap_to_pi(math.atan2(delta[1], delta[0]) - self.base_yaw)
            if self.nav_mode == "turn":
                self.set_twist(0.0, 4.2 * yaw_err)
                if abs(yaw_err) < self.turn_tol:
                    self.nav_mode = "drive"
            else:
                if dist < self.pos_tol:
                    self.nav_idx += 1
                    self.nav_mode = "turn"
                    self.set_twist(0.0, 0.0)
                elif self._obstacle_ahead():
                    self._avoid_step()
                else:
                    ang = 0.0 if (abs(yaw_err) < 0.05 or dist < 0.25) else 4.2 * yaw_err
                    align = max(0.0, math.cos(yaw_err))
                    self.set_twist(1.9 * dist * align, ang)
            return False
        yaw_err = wrap_to_pi(final_yaw - self.base_yaw)
        self.set_twist(0.0, 3.0 * yaw_err)
        if abs(yaw_err) < self.turn_tol:
            self.set_twist(0.0, 0.0)
            return True
        return False

    def _shelf_route(self):
        """直接前往当前货架前方(SCAN_APPROACH_Y, 停远一点让摆头覆盖 3 层),
        E->D->C->B->A 依次前进不回头."""
        x = SHELF_C2_X[self.cur_shelf] + APPROACH_X_OFFSET
        return [[x, SCAN_APPROACH_Y]]

    # ---- Stage-5: nearest-item planning, IK and grasp helpers ----
    def world_to_footprint(self, p_world):
        d = np.asarray(p_world, dtype=float) - np.array(
            [self.base_xy[0], self.base_xy[1], 0.0])
        c, s = math.cos(-self.base_yaw), math.sin(-self.base_yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    def footprint_to_world(self, fp):
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        return np.array([self.base_xy[0] + c * fp[0] - s * fp[1],
                         self.base_xy[1] + s * fp[0] + c * fp[1], fp[2]])

    @property
    def slide_meas(self):
        return self.jpos.get("slide_joint", self.tc[2])

    @property
    def rarm_meas(self):
        return np.array([self.jpos.get(f"right_arm_joint{i + 1}", self.tc[12 + i])
                         for i in range(6)])

    def arm_to(self, world_pos, rot=None, ref_pos=None):
        """逆解右臂到世界位姿; 默认以当前实测关节作 seed, 失败时回退中性 seed.

        IK 多解且对 seed 敏感: 抓取时当前姿态 seed 可解;
        从货架保持的抓取姿势去放置到较低桌面时, 当前 seed 可能无解,
        因此再尝试 UNSTOW_R/STOW_R seed 命中可达解, 避免 place 卡 IK unreachable.
        """
        if self.kdl is None:
            return False
        if rot is None:
            rot = GRASP_ROT_BY_KIND.get(self.current_kind or "", GRASP_ROT)
        fp = self.world_to_footprint(world_pos)
        target = np.eye(4)
        target[:3, :3] = rot
        target[:3, 3] = fp

        def _solve(seed):
            ref = np.zeros(7)
            ref[0] = float(self.tc[2])
            ref[1:] = np.asarray(seed, dtype=float)
            return self.kdl.inverse_kinematics(
                T_left=None, T_right=target, ref_pos=ref,
                target_height=float(self.tc[2]))

        seeds = [ref_pos if ref_pos is not None else self.rarm_meas,
                 UNSTOW_R, STOW_R]
        solutions = None
        for seed in seeds:
            solutions = _solve(seed)
            if solutions:
                break
        if not solutions:
            self.get_logger().warn(
                f"[stage5] IK unreachable: world={np.round(world_pos, 3)}")
            return False
        self.tc[12:18] = np.asarray(solutions[0])[1:7]
        self.arm_target_set = True
        return True

    def ee_footprint_pose(self):
        _, target = self.kdl.forward_kinematics(
            np.concatenate([[float(self.slide_meas)], self.rarm_meas]), index="right")
        return target

    def ee_world(self):
        target = self.ee_footprint_pose()
        return self.footprint_to_world(target[:3, 3])

    def _candidate_key(self, finding):
        return (finding["kind"], int(finding["marker"]))

    def _format_finding(self, finding):
        coords = finding["coords"]
        return (f"[{finding['shelf']}] found {finding['kind']} "
                f"level={finding['level']} "
                f"coords=({coords[0]:.2f},{coords[1]:.2f},{coords[2]:.2f}) "
                f"samples={finding['samples']} -> marker={finding['marker']} "
                f"slot={finding['slot']} dist={finding['dist']:.2f}")

    def _plan_nearest_pick(self):
        pending_kinds = {kind for kind, count in self.pending_counts.items() if count > 0}
        candidates = [f for f in self.findings
                      if f["kind"] in pending_kinds
                      and self._candidate_key(f) not in self.used_markers]
        if not candidates:
            self.get_logger().error(
                f"[stage5] no usable mapped candidates; pending={dict(self.pending_counts)}")
            self.done_ts = self.now()
            self.phase = DONE
            return
        shelf_priority = {shelf: idx for idx, shelf in enumerate(PICK_SEARCH_ORDER)}
        candidates.sort(key=lambda f: (
            shelf_priority.get(f["shelf"], len(PICK_SEARCH_ORDER)),
            float(np.linalg.norm(np.asarray(f["coords"])[:2] - self.base_xy))))
        finding = candidates[0]
        distance = float(np.linalg.norm(
            np.asarray(finding["coords"])[:2] - self.base_xy))
        self.current_finding = finding
        self.current_kind = finding["kind"]
        self.cur_params = resolve_grasp_params(self.current_kind)
        self.cur_shelf = finding["shelf"]
        self.pick_route = [[float(finding["coords"][0]), PICK_APPROACH_Y]]
        self.reset_nav()
        self.nav_t0 = self.now()
        self.phase = NAV_PICK
        target_line = self._format_finding(finding)
        self._append_log("[stage5-pick] " + target_line)
        self.get_logger().info(
            f"[stage5] selected by A->E order kind={self.current_kind} "
            f"slot={finding['slot']} marker={finding['marker']} "
            f"dist={distance:.3f} "
            f"remaining={sum(self.pending_counts.values())}")
        self.get_logger().info("[stage5-pick] " + target_line)

    def _prepare_arm_target(self):
        finding = self.current_finding
        obj = np.asarray(finding["coords"], dtype=float).copy()
        marker = int(finding["marker"])
        anchor = self._aruco_anchor_xyz().get(marker)
        if anchor is not None:
            obj[0] = float(anchor[3][0])
            obj[1] = float(anchor[3][1])
        obj[0] += float(GRASP_X_OFFSET_BY_KIND.get(self.current_kind, 0.0))
        self.object_world = obj
        self.tc[2] = self._grasp_slide_for_finding(finding)
        self.tc[3] = 0.0
        marker_level = str(ARUCO_SLOTS.get(int(finding["marker"]), {}).get(
            "level", finding.get("level", "L2")))
        self.tc[4] = PITCH_BY_LEVEL.get(marker_level, 0.0)
        self.tc[18] = self._grip_open_for(self.current_kind)
        self.deploy_world = obj + np.asarray(self.cur_params.deploy_offset_arr)
        self.creep_stop_y = float(obj[1]) + self.cur_params.creep_stop_dy
        if not self.arm_to(self.deploy_world):
            self.get_logger().warn("[stage5] deploy IK failed; skip this candidate")
            self.used_markers.add(self._candidate_key(finding))
            self.phase = PLAN_PICK
            return
        self.arm_target_set = True
        self.state_t0 = self.now()
        self.phase = DEPLOY_ARM
        self.get_logger().info(
            f"[stage5] deploy kind={self.current_kind} slot={finding['slot']} "
            f"object={np.round(obj, 3)} deploy={np.round(self.deploy_world, 3)} "
            f"slide={self.tc[2]:.2f} grip_open={self.tc[18]:.2f}")

    def _grip_open_for(self, kind):
        return float(GRIP_OPEN_BY_KIND.get(kind, GRIP_OPEN))

    def _grasp_slide_for_finding(self, finding):
        marker = int(finding["marker"])
        level = str(ARUCO_SLOTS.get(marker, {}).get("level", finding.get("level", "L2")))
        return float(GRASP_SLIDE_BY_LEVEL.get(level, slide_for_z(float(finding["coords"][2]))))

    def _start_arm_ready(self):
        self.tc[2] = self._grasp_slide_for_finding(self.current_finding)
        self.tc[12:18] = UNSTOW_R
        self.tc[18] = self._grip_open_for(self.current_kind)
        self.state_t0 = self.now()
        self.last_deploy_log = 0.0
        self.phase = ARM_READY
        self.get_logger().info(
            f"[stage5] arm ready kind={self.current_kind} "
            f"slide={self.tc[2]:.2f} grip_open={self.tc[18]:.2f}")

    def arm_ready_done(self):
        joint_error = float(np.max(np.abs(self.rarm_meas - np.asarray(UNSTOW_R))))
        slide_error = abs(self.slide_meas - self.tc[2])
        done = joint_error < ARM_READY_TOL and slide_error < 0.04
        if self.now() - self.last_deploy_log > 2.0:
            self.last_deploy_log = self.now()
            self.get_logger().info(
                f"[stage5] arm-ready joint_err={joint_error:.3f} "
                f"slide_err={slide_error:.3f} grip_open={self.tc[18]:.2f}")
        return done

    def deploy_done(self, dwell=0.4):
        if self.now() - self.state_t0 < dwell:
            return False
        slide_err = abs(self.slide_meas - self.tc[2])
        cart_err = float(np.linalg.norm(self.ee_world() - self.deploy_world))
        rot = self.ee_footprint_pose()[:3, :3]
        rot_delta = rot.T @ GRASP_ROT_BY_KIND.get(self.current_kind or "", GRASP_ROT)
        rot_err = math.acos(float(np.clip(
            (np.trace(rot_delta) - 1.0) / 2.0, -1.0, 1.0)))
        if self.now() - self.last_deploy_log > 2.0:
            self.last_deploy_log = self.now()
            self.get_logger().info(
                f"[stage5] deploy wait slide_err={slide_err:.3f} "
                f"cart_err={cart_err:.3f} rot_err={rot_err:.3f} "
                f"arm={np.round(self.rarm_meas, 3)}")
        return (slide_err < 0.03 and cart_err < DEPLOY_CART_TOL
                and rot_err < DEPLOY_ROT_TOL)

    def _grip_hold_done(self):
        """持物保持姿态就绪判定: 不要求收臂, 只要夹爪命令真正闭合即可送货."""
        cmd_ok = abs(self.action[18] - self.tc[18]) <= CLOSE_CMD_TOL
        grip_meas = self._gripper_meas()
        if self.now() - self.last_deploy_log > 2.0:
            self.last_deploy_log = self.now()
            self.get_logger().info(
                f"[stage5] grip-hold grip={self.tc[18]:.2f} "
                f"grip_meas={'-' if grip_meas is None else round(float(grip_meas), 3)} "
                f"cmd_ok={cmd_ok}")
        # 注意: j5 底座不用 grip_meas 判断是否闭合(只用 slot-clear 验证), 该反馈可能失真,
        # 因此这里仅以命令到位为准, 避免被失真实测长期卡死.
        return cmd_ok and self.tc[18] <= GRIP_FALLBACK_CLOSE + 0.05

    def _gripper_meas(self):
        """右夹爪实测关节值(0~1, 0=闭合); 未发布时返回 None."""
        if self.jpos is None:
            return None
        try:
            return float(self.jpos.get("right_arm_eef_gripper_joint"))
        except (TypeError, ValueError):
            return None

    def _finish_pick_hold(self):
        """抓取成功: 右臂持物收拢(STOW), 夹爪闭合, 进入配送台导航."""
        finding = self.current_finding
        kind = finding["kind"]
        self.used_markers.add(self._candidate_key(finding))
        for item in self.findings:
            if (item["shelf"] == finding["shelf"]
                    and int(item["marker"]) == int(finding["marker"])):
                item["kind"] = "HELD"
        self.carry_grip_close = (
            float(self.cur_params.grip_close)
            if self.cur_params is not None else GRIP_FALLBACK_CLOSE)
        self.joint_slew = HOLD_SLEW     # 持物收臂/运输降速, 防惯性把商品甩出
        self.grip_closed_t0 = None
        # 不持物收臂: 保留刚抓起商品时的右臂 IK 姿态, 只保持夹爪闭合.
        # 若改成 STOW_R 收拢, 末端翻转/惯性会把商品甩掉; 保持伸向商品的姿势直接送货.
        self.tc[18] = self.carry_grip_close
        self.tc[3] = 0.0
        self.tc[4] = -0.35
        # 抓取完成、退出货架时, 立即把身体抬到放置高度(PLACE_SLIDE=0.12, 最高位).
        # 否则握持手臂会停留在抓取层 slide(底层货架 L1≈0.60, 身体最低),
        # 送货途中前伸手臂低位撞到配送桌, 导致 nav->delivery 卡死/超时.
        # 抬高后手臂越过桌面高度, 可正常走到桌边, 再由 PLACE 逻辑放到桌面.
        self.tc[2] = PLACE_SLIDE
        self.arm_target_set = False
        self._append_log(
            f"[stage5] PICKED-HOLD kind={kind} slot={finding['slot']} "
            f"marker={finding['marker']} remaining={sum(self.pending_counts.values())}")
        self.get_logger().info(
            f"[stage5] PICKED-HOLD kind={kind} slot={finding['slot']} "
            f"marker={finding['marker']} remaining={sum(self.pending_counts.values())}")
        # 先确认夹爪仍闭合(命令到位), 再开始送货导航, 防止起步丢物.
        self.state_t0 = self.now()
        self.phase = STOW_GRIP_HOLD

    def _finish_pick_failed(self):
        """抓取失败(基座被抵住/夹爪未够到商品): 收起手臂、张开夹爪, 标记并重规划."""
        finding = self.current_finding
        self.used_markers.add(self._candidate_key(finding))
        self.grab_failed = False
        apply_stow(self.tc, GRIP_OPEN)
        self.tc[3] = 0.0
        self.tc[4] = 0.0
        self.arm_target_set = False
        self.object_world = self.deploy_world = self.creep_stop_y = None
        self._append_log(
            f"[stage5] GRAB-FAILED kind={finding['kind']} slot={finding['slot']} "
            f"marker={finding['marker']}; skip -> replan")
        self.get_logger().warn(
            f"[stage5] GRAB-FAILED kind={finding['kind']} slot={finding['slot']} "
            f"marker={finding['marker']}; skip -> replan")
        self.phase = PLAN_PICK

    # ---- 阶段四 NAV+MPPI 送货 / 返程 ----
    def _prepare_delivery_artifacts(self):
        """送货前清理旧的 delivery/return 图片并确保 logs/visual 目录存在."""
        import shutil
        root = _HERE / "nav_return_dev"
        logs_dir = root / "logs"
        try:
            if logs_dir.exists():
                for f in logs_dir.iterdir():
                    if f.is_file() and f.name.startswith("nav_"):
                        f.unlink()
        except Exception as exc:
            self.get_logger().warn(f"[stage4] cleanup nav log failed: {exc}")
        for sub in ("visual/delivery", "visual/return"):
            d = root / sub
            try:
                if d.exists():
                    for f in d.iterdir():
                        if f.is_file() or f.is_symlink():
                            f.unlink()
                        elif f.is_dir():
                            shutil.rmtree(f, ignore_errors=True)
            except Exception as exc:
                self.get_logger().warn(f"[stage4] cleanup {d} failed: {exc}")
            else:
                self.get_logger().info(
                    f"[stage4] cleaned nav_return_dev/{sub} before delivery")
        for sub in ("logs", "visual", "visual/delivery", "visual/return"):
            try:
                (root / sub).mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.get_logger().warn(f"[stage4] mkdir {sub} failed: {exc}")

    def _navigate_to_delivery(self):
        """阶段4送货: 用真实终点(DELIVERY_POINT)与实时感知在线规划.

        NAV+MPPI 根据激光+里程计增量构建占用栅格, 由内置 A* 在线生成动态子目标,
        再执行局部避障/跟踪; 不使用人工中间航点.
        """
        if self.mppi_steer is not None:
            self.mppi_steer.set_visual_stage("delivery")
        if self.mppi_steer is None:
            self.tc[3] = 0.0
            self.tc[4] = -0.35
            try:
                from nav_return_dev.navigator import NavMppiNavigator
            except Exception as exc:
                self.get_logger().error(
                    f"[stage4] NAV+MPPI unavailable ({exc}); stop delivery")
                self.set_twist(0.0, 0.0)
                return False
            self.mppi_steer = NavMppiNavigator()
            self._prepare_delivery_artifacts()
            self.mppi_steer.set_visual_stage("delivery")
            self.get_logger().info("[stage4] NAV+MPPI controller ready")
            _logp = self.mppi_steer.start_nav_log(
                str(_HERE / "nav_return_dev" / "logs"))
            if _logp:
                self.get_logger().info(f"[stage4] navdiag log -> {_logp}")

        if self.laser is not None:
            self.mppi_steer.push_scan(
                self.laser, angle_min=self.laser_angle_min,
                angle_inc=self.laser_angle_inc)

        now = self.now()
        if now - self.mppi_last_t >= 0.050:
            try:
                _delivery_idx = min(self.delivered_count, len(DELIVERY_SLOTS) - 1)
                _delivery_goal = DELIVERY_SLOTS[_delivery_idx]
                _pose = (float(self.base_xy[0]), float(self.base_xy[1]),
                         float(self.base_yaw))
                _goal = (float(_delivery_goal[0]), float(_delivery_goal[1]),
                         float(DELIVERY_YAW))
                _vx, _wz, meta = self.mppi_steer.step(
                    _pose, _goal, self.laser,
                    angle_min=self.laser_angle_min,
                    angle_inc=self.laser_angle_inc)
                self.mppi_last = (float(_vx), float(_wz))
            except Exception as exc:
                self.get_logger().error(f"[stage4] NAV+MPPI step err: {exc}")
                self.mppi_last = (0.0, 0.0)
                meta = {"reason": "err"}
            self.mppi_last_t = now
        else:
            meta = {"reason": "reuse"}

        if now - self.mppi_diag_t >= 0.2:
            self.mppi_diag_t = now
            _diag_line = (
                "[navdiag] ts={:.3f} reason={} mode={} "
                "vx={:.3f} wz={:.3f} "
                "pose_x={} pose_y={} pose_yaw={} "
                "goal_x={} goal_y={} goal_yaw={} "
                "d_goal={} goal_clamped={} "
                "path_n={} path_len={} lookahead={} path_dev0={} ref_hdg_err={} "
                "ref_exists={} ref_start=({},{}) ref_end=({},{}) "
                "ref_first=({},{}) ref_goal_dir_cost={} "
                "scan_n={} angle_min={} angle_inc={} "
                "front_min={} left_front={} right_front={} rear_min={} "
                "cost_lethal_cells={} cost_mean={} "
                "stuck_counter={} unwedge_left={} collide={} S_min={} "
                "top_v={} top_w={} ".format(
                    now,
                    meta.get("reason", "-"), meta.get("mode", "-"),
                    self.mppi_last[0], self.mppi_last[1],
                    _fmt(meta.get("pose_x", "-")),
                    _fmt(meta.get("pose_y", "-")),
                    _fmt(meta.get("pose_yaw", "-")),
                    _fmt(meta.get("goal_x", "-")),
                    _fmt(meta.get("goal_y", "-")),
                    _fmt(meta.get("goal_yaw", "-")),
                    _fmt(meta.get("d_goal", "-")),
                    meta.get("goal_clamped", "-"),
                    meta.get("path_n", "-"),
                    _fmt(meta.get("path_len", "-")),
                    _fmt(meta.get("lookahead_dist", "-")),
                    _fmt(meta.get("path_dev0", "-")),
                    _fmt(meta.get("ref_hdg_err", "-")),
                    meta.get("ref_exists", "-"),
                    _fmt(meta.get("ref_start_x", "-")),
                    _fmt(meta.get("ref_start_y", "-")),
                    _fmt(meta.get("ref_end_x", "-")),
                    _fmt(meta.get("ref_end_y", "-")),
                    meta.get("ref_first_x", "-"), meta.get("ref_first_y", "-"),
                    meta.get("ref_goal_dir_cost", "-"),
                    meta.get("scan_n", "-"),
                    _fmt(meta.get("angle_min", "-")),
                    _fmt(meta.get("angle_inc", "-")),
                    _fmt(meta.get("front_min", "-")),
                    _fmt(meta.get("left_front_min", "-")),
                    _fmt(meta.get("right_front_min", "-")),
                    _fmt(meta.get("rear_min", "-")),
                    meta.get("cost_lethal_cells", "-"),
                    _fmt(meta.get("cost_mean", "-")),
                    meta.get("stuck_counter", "-"),
                    meta.get("unwedge_frames_left", "-"),
                    meta.get("collide_count", "-"),
                    _fmt(meta.get("S_min", "-")),
                    meta.get("top_v", "-"), meta.get("top_w", "-"),
                )
            )
            self.get_logger().info(_diag_line)
            if self.mppi_steer is not None:
                self.mppi_steer.write_nav_log(_diag_line)

        self.set_twist(self.mppi_last[0], self.mppi_last[1])
        if meta.get("reason") == "arrived":
            self.set_twist(0.0, 0.0)
            return True
        if meta.get("reason") == "err":
            return False
        if self.now() - self.nav_t0 > NAV_TIMEOUT:
            return False
        return None

    def _emit_return_diag(self, now, meta):
        """返程诊断行, 追加到同一 nav 日志."""
        line = "[navdiag] ts={:.3f} reason={} mode={} return_stage={} " \
               "vx={:.3f} wz={:.3f} " \
               "pose_x={} pose_y={} pose_yaw={} " \
               "goal_x={} goal_y={} goal_yaw={} " \
               "d_goal={} path_n={} path_len={} lookahead={} " \
               "front_min={} left_front={} right_front={} rear_min={} " \
               "collide={} S_min={} stuck_counter={}".format(
                   now,
                   meta.get("reason", "-"), meta.get("mode", "-"),
                   meta.get("return_stage", "-"),
                   self.mppi_last[0], self.mppi_last[1],
                   _fmt(meta.get("pose_x", "-")),
                   _fmt(meta.get("pose_y", "-")),
                   _fmt(meta.get("pose_yaw", "-")),
                   _fmt(meta.get("goal_x", "-")),
                   _fmt(meta.get("goal_y", "-")),
                   _fmt(meta.get("goal_yaw", "-")),
                   _fmt(meta.get("d_goal", "-")),
                   meta.get("path_n", "-"),
                   _fmt(meta.get("path_len", "-")),
                   _fmt(meta.get("lookahead_dist", "-")),
                   _fmt(meta.get("front_min", "-")),
                   _fmt(meta.get("left_front_min", "-")),
                   _fmt(meta.get("right_front_min", "-")),
                   _fmt(meta.get("rear_min", "-")),
                   meta.get("collide_count", "-"),
                   _fmt(meta.get("S_min", "-")),
                   meta.get("stuck_counter", "-"),
               )
        self.get_logger().info(line)
        if self.mppi_steer is not None:
            self.mppi_steer.write_nav_log(line)

    def _navigate_return_to_shelf(self):
        """返程: 从配送台回到下一件目标货位. 用 begin_return 处理安全后退/左移."""
        if self.mppi_steer is None:
            self.set_twist(0.0, 0.0)
            return False
        self.mppi_steer.set_visual_stage("return")
        if self.laser is not None:
            self.mppi_steer.push_scan(
                self.laser, angle_min=self.laser_angle_min,
                angle_inc=self.laser_angle_inc)

        now = self.now()
        if now - self.mppi_last_t >= 0.050:
            try:
                _pose = (float(self.base_xy[0]), float(self.base_xy[1]),
                         float(self.base_yaw))
                _goal = tuple(float(x) for x in self._return_next_goal)
                _vx, _wz, meta = self.mppi_steer.step(
                    _pose, _goal, self.laser,
                    angle_min=self.laser_angle_min,
                    angle_inc=self.laser_angle_inc)
                self.mppi_last = (float(_vx), float(_wz))
            except Exception as exc:
                self.get_logger().error(f"[return] NAV+MPPI step err: {exc}")
                self.mppi_last = (0.0, 0.0)
                meta = {"reason": "err"}
            self.mppi_last_t = now
        else:
            meta = {"reason": "reuse"}

        if now - self.mppi_diag_t >= 0.2:
            self.mppi_diag_t = now
            self._emit_return_diag(now, meta)

        self.set_twist(self.mppi_last[0], self.mppi_last[1])
        if meta.get("reason") == "arrived":
            self.set_twist(0.0, 0.0)
            self.get_logger().info("[return] arrived at next pick goal; stop")
            return True
        if meta.get("reason") in ("err", "return_blocked", "no_safe_path"):
            self.set_twist(0.0, 0.0)
            self.get_logger().error(
                "[return] failed reason=%s; stop" % meta.get("reason"))
            return False
        if self.now() - self.nav_t0 > NAV_TIMEOUT:
            return False
        return None

    def _finish_place(self):
        """配送台放置成功: 标记订单完成, 进入返程取下一件或 DONE."""
        finding = self.current_finding
        kind = finding["kind"]
        self.used_markers.add(self._candidate_key(finding))
        if self.pending_counts.get(kind, 0) > 0:
            self.pending_counts[kind] -= 1
        for item in self.findings:
            if (item["shelf"] == finding["shelf"]
                    and int(item["marker"]) == int(finding["marker"])):
                item["kind"] = "NULL"
        ok_line = f"{kind} ---DELIVERED OK!!!"
        self._append_log(ok_line)
        self.get_logger().info(ok_line)
        self._write_all_and_order_logs(picked_kind=kind)
        self._append_log(
            f"[stage5] DELIVERED kind={kind} slot={finding['slot']} "
            f"marker={finding['marker']} remaining={sum(self.pending_counts.values())}")
        self.get_logger().info(
            f"[stage5] DELIVERED kind={kind} slot={finding['slot']} "
            f"remaining={sum(self.pending_counts.values())}")
        self.current_finding = None
        self.current_kind = None
        self.cur_params = None
        self.object_world = self.deploy_world = self.creep_stop_y = None
        self.arm_target_set = False
        self.delivered_count += 1
        apply_stow(self.tc, GRIP_OPEN)
        self.joint_slew = 1.2           # 商品已释放, 恢复正常关节速度
        self.tc[3] = 0.0
        self.tc[4] = 0.0
        remaining = sum(self.pending_counts.values())
        if remaining <= 0 or self.delivered_count >= RANDOM_ORDER_COUNT:
            self.done_ts = self.now()
            self.phase = DONE
            self.get_logger().info("ALL ORDER ITEMS DELIVERED (finish at table)")
        else:
            self._set_next_return_goal()
            try:
                if self.mppi_steer is not None:
                    self.mppi_steer.begin_return(
                        (float(self.base_xy[0]),
                         float(self.base_xy[1]),
                         float(self.base_yaw)),
                        tuple(float(x) for x in self._return_next_goal))
            except Exception as exc:
                self.get_logger().error(
                    f"[return] begin_return failed: {exc}; stop at table")
                self.done_ts = None
                self.phase = DONE
                return
            self.nav_t0 = self.now()
            self.mppi_last = (0.0, 0.0)
            self.mppi_last_t = 0.0
            self.mppi_diag_t = 0.0
            self.phase = NAV_RETURN_TO_NEXT

    def _place_item_at_delivery(self):
        """到达配送台执行放置: 右臂展开到桌面上方 -> 夹爪张开放下 -> 收臂回 STOW_R.

        多阶段子状态机; 返回 True=放置完成, False=失败, None=进行中.
        """
        if self.current_finding is None:
            self.get_logger().error("[place] no current item; stop")
            return False
        self.set_twist(0.0, 0.0)
        if self.place_world is None:
            self.place_start_xy = np.asarray(self.base_xy, dtype=float).copy()
            fx = math.cos(self.base_yaw)
            fy = math.sin(self.base_yaw)
            _slot_idx = min(self.delivered_count, len(DELIVERY_SLOTS) - 1)
            _slot_x, _slot_y = DELIVERY_SLOTS[_slot_idx]
            self.place_world = np.array([
                _slot_x,
                _slot_y - PLACE_FWD_DIST,
                PLACE_Z,
            ])
            self.place_stage = "align"
            self.place_armed = False
            self.place_slide = None
            self.place_clear_slide = None
            self.get_logger().info(
                f"[place] slot={_slot_idx + 1}/{len(DELIVERY_SLOTS)} "
                f"target world={np.round(self.place_world, 3)}")

        now = self.now()
        if self.place_stage == "align":
            # 放置前原地把车头对准桌面(法向), 让后退返程走直线不别墙.
            yaw_err = wrap_to_pi(DELIVERY_YAW - self.base_yaw)
            if abs(yaw_err) < PLACE_ALIGN_TOL:
                self.set_twist(0.0, 0.0)
                self.state_t0 = now
                self.place_stage = "deploy"
                self.get_logger().info(
                    f"[place] yaw aligned; yaw_err={yaw_err:.3f}")
                return None
            self.set_twist(0.0, float(np.clip(
                PLACE_ALIGN_KP * yaw_err, -PLACE_ALIGN_WZ, PLACE_ALIGN_WZ)))
            if now - self.state_t0 > PLACE_TIMEOUT:
                self.get_logger().error("[place] yaw align timeout")
                return False
            return None
        if self.place_stage == "deploy":
            # 与官方示例(PickPlaceClient)保持一致: 放置时右臂关节不动, 只升降身体,
            # 松爪让商品竖直落到桌面. 不要在这里重新做 IK —— 那会让手臂快速下摆,
            # 惯性把商品甩飞/穿桌.
            # 但因为抓取层 L1/L2/L3 不同, 固定 PLACE_SLIDE 会让末端高度每次不一致.
            # 这里用 slide(竖直升降) 与末端 z 近似 1:1 的关系, 按当前实测末端高度
            # 反推出一个 per-item 的目标 slide, 使任何抓取层下商品都停在同一桌面高度.
            self.tc[18] = self.carry_grip_close
            if not self.place_armed:
                self.place_armed = True
                ee = self.ee_world()
                self.place_slide = float(self.slide_meas) + (float(ee[2]) - PLACE_Z)
                self.place_slide = float(np.clip(self.place_slide, 0.0, 0.60))
                # 收臂前先抬升身体, 让展开的右臂离开桌面(高于刚放的商品),
                # 再把手臂收拢, 最后才倒车 —— 避免倒车时展开的右臂扫倒桌面上商品.
                self.place_clear_slide = float(np.clip(
                    self.place_slide - PLACE_CLEAR_LIFT,
                    PLACE_CLEAR_MIN, 0.60))
                self.get_logger().info(
                    f"[place] keep arm pose; ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                    f"slide={self.slide_meas:.3f} -> target_slide={self.place_slide:.3f} "
                    f"clear_slide={self.place_clear_slide:.3f} place_z={PLACE_Z}")
            self.tc[2] = self.place_slide
            slide_err = abs(self.slide_meas - self.place_slide)
            if now - self.state_t0 > PLACE_TIMEOUT:
                self.get_logger().error("[place] deploy timeout")
                return False
            if self.now() - self.last_deploy_log > 2.0:
                self.last_deploy_log = self.now()
                self.get_logger().info(
                    f"[place] deploy slide_err={slide_err:.3f} "
                    f"slide={self.slide_meas:.3f} target={self.place_slide:.3f}")
            # 身体升/降到目标 slide, 商品末端即落在桌面高度, 再张爪让商品落桌
            if slide_err < 0.03:
                self.state_t0 = now
                self.place_stage = "open"
                self.get_logger().info(
                    "[place] body at table height; opening gripper")
            return None
        elif self.place_stage == "open":
            self.tc[18] = GRIP_OPEN
            if now - self.state_t0 > PLACE_OPEN_DWELL:
                self.state_t0 = now
                self.place_stage = "settle"
                self.get_logger().info("[place] gripper opened; waiting item settle")
            return None
        elif self.place_stage == "settle":
            self.set_twist(0.0, 0.0)
            self.tc[18] = GRIP_OPEN
            if now - self.state_t0 > PLACE_SETTLE_DWELL:
                self.state_t0 = now
                self.place_stage = "clear"
                self.get_logger().info("[place] item settled; lift arm clear of table")
            return None
        elif self.place_stage == "clear":
            # 抬升身体(slide 减小), 把展开的右臂/夹爪抬高到桌面商品上方
            self.set_twist(0.0, 0.0)
            self.tc[18] = GRIP_OPEN
            self.tc[2] = self.place_clear_slide
            clear_err = abs(self.slide_meas - self.place_clear_slide)
            if now - self.state_t0 > PLACE_TIMEOUT:
                self.get_logger().error("[place] clear timeout")
                return False
            if self.now() - self.last_deploy_log > 2.0:
                self.last_deploy_log = self.now()
                self.get_logger().info(
                    f"[place] clear slide_err={clear_err:.3f} "
                    f"slide={self.slide_meas:.3f} target={self.place_clear_slide:.3f}")
            if clear_err < 0.03:
                self.state_t0 = now
                self.place_stage = "stow"
                self.get_logger().info("[place] arm lifted; retracting")
            return None
        elif self.place_stage == "stow":
            self.tc[18] = GRIP_OPEN
            self.tc[12:18] = list(STOW_R)
            if now - self.state_t0 > PLACE_STOW_DWELL and \
                    float(np.max(np.abs(self.rarm_meas - np.asarray(STOW_R)))) < PLACE_STOW_TOL:
                self.state_t0 = now
                self.place_stage = "retreat"
                self.get_logger().info("[place] arm stowed; back away from table")
                return None
            if now - self.state_t0 > PLACE_TIMEOUT:
                self.get_logger().error("[place] stow timeout")
                return False
            return None
        elif self.place_stage == "retreat":
            self.tc[18] = GRIP_OPEN
            self.tc[12:18] = list(STOW_R)
            moved = float(np.linalg.norm(
                np.asarray(self.base_xy, dtype=float) - self.place_start_xy))
            if moved >= PLACE_RETREAT_DIST:
                self.set_twist(0.0, 0.0)
                self._finish_place()
                return True
            self.set_twist(-PLACE_RETREAT_SPEED, 0.0)
            return None
        return False

    def _make_ee_target(self, world_pos, rot):
        """构造末端 4x4 目标位姿(右手)."""
        target = np.eye(4)
        target[:3, :3] = rot
        target[:3, 3] = self.world_to_footprint(world_pos)
        return target

    def _set_next_return_goal(self):
        """动态选择下一件目标货位作为返程终点."""
        pending_kinds = {kind for kind, count in self.pending_counts.items() if count > 0}
        candidates = [f for f in self.findings
                      if f["kind"] in pending_kinds
                      and self._candidate_key(f) not in self.used_markers]
        if not candidates:
            self._return_next_goal = (
                float(RETURN_SHELF_POINT[0]),
                float(RETURN_SHELF_POINT[1]),
                float(RETURN_SHELF_YAW))
            return
        shelf_priority = {shelf: idx for idx, shelf in enumerate(PICK_SEARCH_ORDER)}
        candidates.sort(key=lambda f: (
            shelf_priority.get(f["shelf"], len(PICK_SEARCH_ORDER)),
            float(np.linalg.norm(np.asarray(f["coords"])[:2] - self.base_xy))))
        f = candidates[0]
        self._return_next_goal = (
            float(f["coords"][0]),
            float(PICK_APPROACH_Y),
            float(GRASP_YAW))
        self.get_logger().info(
            f"[return] next target kind={f['kind']} shelf={f['shelf']} "
            f"marker={f['marker']} goal=({self._return_next_goal[0]:.3f},"
            f"{self._return_next_goal[1]:.3f})")

    # ---- 固定货位真值几何: id -> (shelf/level/column, 真值世界坐标) ----
    def _aruco_anchor_xyz(self):
        """每个 ArUco 货位码的固定世界坐标(ArUco 码固定绑定货位)."""
        out = {}
        for mid, slot in ARUCO_SLOTS.items():
            sh, lv, col = slot["shelf"], slot["level"], slot["column"]
            ax = SHELF_C1_X[sh] + COL_X_OFF[col]
            out[int(mid)] = (sh, lv, col, np.array([ax, ARUCO_Y_FIX, ARUCO_Z_FIX[lv]]))
        return out

    # ---- 建图: 把感知缓冲喂给 SlotMap ----
    def _shelf_fixed_anchors(self):
        """????????????????(ArUco ??/ID ????, ??????????).

        L1/L2/L3 ???? head ?? FOV ????, ??????????(marker=??).
        ??????????: ?? YOLO ???????, ??????????????????.
        """
        anchors = self._aruco_anchor_xyz()
        return [(mid, anchors[mid][3]) for mid in sorted(ARUCO_SLOTS)
                if ARUCO_SLOTS[mid].get("shelf") == self.cur_shelf]


    def _consume_perception(self):
        markers = list(self.aruco_buf)
        if not markers:
            return
        # 只使用"当前货架"的 ArUco 码, 防止跨架串扰; 关联用固定真值几何
        anchors = self._aruco_anchor_xyz()
        detected_shelf = [(mid, anchors[mid][3]) for (mid, xyz) in markers
                          if ARUCO_SLOTS.get(int(mid), {}).get("shelf") == self.cur_shelf]
        if not detected_shelf:
            self.get_logger().warn(
                f"[stage3] no aruco of current shelf {self.cur_shelf}; "
                f"skipping dets this scan (no anchor)")
            return
        fixed_anchors = self._shelf_fixed_anchors()
        while self.det_buf:
            kind, xyz = self.det_buf.popleft()
            self.slot_map.observe(kind, xyz, fixed_anchors)

    def _start_sweep(self):
        """到货架前: 清空导航途中积压的移动检测, 按 L3->L2->L1 向下扫一次并输出."""
        self.det_buf.clear()
        self.aruco_buf.clear()
        self.order_points.clear()
        self.level_samples.clear()
        self.scan_dets = 0
        self.sweep_levels = list(SCAN_LEVELS)
        self.scan_lvl_idx = 0
        self.level_t0 = self.now()
        self.level_settled = False
        self._enter_scan_level()
        self.tc[3] = 0.0
        self.scan_t0 = self.now()
        self.get_logger().info(
            f"[stage3] shelf {self.cur_shelf} sweep=down-once "
            f"levels={'/'.join(self.sweep_levels)} dwell={DWELL_PER_LEVEL:.1f}s "
            f"pitch={PITCH_BY_LEVEL[self.sweep_levels[0]]:.2f}")

    def _slide_for(self, lvl):
        """返回该层 scan 时的 slide 目标(表中数值整体已下移 0.2)."""
        return SLIDE_BY_LEVEL[lvl]

    def _enter_scan_level(self):
        """进入下一层驻留: 设定 slide 高度 + 俯仰角."""
        lvl = self.sweep_levels[self.scan_lvl_idx]
        self.tc[2] = self._slide_for(lvl)
        self.tc[4] = PITCH_BY_LEVEL[lvl]
        self.level_t0 = self.now()
        self.level_settled = False
        self.get_logger().info(
            f"[stage3] scan level {lvl} slide={self._slide_for(lvl):.2f} "
            f"pitch={PITCH_BY_LEVEL[lvl]:.2f}")

    def _finish_shelf_scan(self):
        """本架向下扫一遍结束: 写入结果并去下一架(不再二遍验证)."""
        self.get_logger().info(
            f"[stage3] shelf {self.cur_shelf} sweep done; "
            f"scan_dets={self.scan_dets}")
        self._record_shelf_findings()
        self.phase = NEXT_SHELF

    def _nearest_shelf_marker(self, world_xyz, shelf_markers):
        """在当前货架已检出的 ArUco 中, 找"同层带内水平距离最近"的货位码.
        用固定真值锚点几何计算距离(消除检测坐标偏差)."""
        pw = np.asarray(world_xyz, dtype=float)
        prod_level = level_of_z(pw[2])
        best_mid, best_d = None, float("inf")
        anchors = self._aruco_anchor_xyz()
        for mid, _mxyz in shelf_markers:
            slot = ARUCO_SLOTS.get(int(mid))
            if not slot:
                continue
            if prod_level is not None and slot["level"] != prod_level:
                continue
            axyz = anchors[mid][3]
            d = float(np.hypot(pw[0] - axyz[0], pw[1] - axyz[1]))
            if d <= DEFAULT_MAX_HORIZ_DIST and d < best_d:
                best_d, best_mid = d, int(mid)
        return best_mid, best_d

    @staticmethod
    def _split_x_clusters(points):
        """????? x ???????????????????"""
        ordered = sorted(points, key=lambda point: float(point[0]))
        if not ordered:
            return []
        clusters = [[ordered[0]]]
        for point in ordered[1:]:
            if float(point[0]) - float(clusters[-1][-1][0]) > SAME_LEVEL_X_SPLIT:
                clusters.append([point])
            else:
                clusters[-1].append(point)
        return clusters

    def _record_shelf_findings(self):
        """????????? (kind, marker) ???????????????"""
        markers = list(self.aruco_buf)
        detected_shelf = [(mid, xyz) for (mid, xyz) in markers
                          if ARUCO_SLOTS.get(int(mid), {}).get("shelf") == self.cur_shelf]
        seen_ids = sorted({int(mid) for mid, _ in detected_shelf})
        self.get_logger().info(
            f"[stage3] shelf {self.cur_shelf} aruco seen ids={seen_ids}")
        fixed_anchors = self._shelf_fixed_anchors()
        if not detected_shelf:
            self.get_logger().warn(
                f"[stage3] shelf {self.cur_shelf}: no shelf ArUco detected this scan "
                f"-> fallback to fixed geometry anchors")

        scanned_kinds = {sample_kind for (sample_kind, _level) in self.level_samples}
        shelf_candidates = {}
        for kind in sorted(scanned_kinds):
            for (sample_kind, scan_level), points in sorted(self.level_samples.items()):
                if sample_kind != kind or not points:
                    continue
                for cluster in self._split_x_clusters(points):
                    med = np.median(np.asarray(cluster, dtype=float), axis=0)
                    mid, dist = self._nearest_shelf_marker(med, fixed_anchors)
                    if mid is None:
                        continue
                    candidate = {
                        "kind": kind,
                        "shelf": self.cur_shelf,
                        "level": ARUCO_SLOTS[int(mid)]["level"],
                        "coords": [float(med[0]), float(med[1]), float(med[2])],
                        "marker": int(mid),
                        "slot": "{}/{}/{}".format(
                            ARUCO_SLOTS[int(mid)]["shelf"],
                            ARUCO_SLOTS[int(mid)]["level"],
                            ARUCO_SLOTS[int(mid)]["column"],
                        ),
                        "samples": len(cluster),
                        "dist": float(dist),
                    }
                    prior = shelf_candidates.get(candidate["marker"])
                    if (prior is None or candidate["samples"] > prior["samples"] or
                            (candidate["samples"] == prior["samples"] and
                             candidate["dist"] < prior["dist"])):
                        shelf_candidates[candidate["marker"]] = candidate

        for candidate in sorted(shelf_candidates.values(), key=lambda item: item["marker"]):
            line = (f"[{self.cur_shelf}] found {candidate['kind']} "
                    f"level={candidate['level']} "
                    f"coords=({candidate['coords'][0]:.2f},"
                    f"{candidate['coords'][1]:.2f},{candidate['coords'][2]:.2f}) "
                    f"samples={candidate['samples']} -> marker={candidate['marker']} "
                    f"slot={candidate['slot']} dist={candidate['dist']:.2f}")
            if candidate["samples"] < MIN_MAP_SAMPLES:
                line += " (low samples, ignored for final)"
            else:
                current = self.final.get(candidate["kind"])
                if (current is None or candidate["samples"] > current["samples"] or
                        (candidate["samples"] == current["samples"] and
                         candidate["dist"] < current["dist"])):
                    self.final[candidate["kind"]] = candidate
                self.findings.append(candidate)
            self._append_log(line)
            self.get_logger().info("[stage3] " + line)

    def _open_verify_log(self):
        """订单下发时创建验证日志文件(后续逐条追加)."""
        logs_dir = _HERE / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.verify_log_path = logs_dir / f"stage35_verify_{ts}.log"
        self.all_log_path = logs_dir / "All.log"
        self.order_log_path = logs_dir / "order.log"
        header = [
            "== Stage-3+5 verification log ==",
            f"run_prefix={self.run_prefix}",
            f"targets={len(self.targets)} pending={dict(self.pending_counts)}",
            "scan order: E -> D -> C -> B -> A (each shelf: down-once L3->L2->L1, no verify)",
            "",
        ]
        self.verify_log_path.write_text("\n".join(header) + "\n", encoding="utf-8")
        self.get_logger().info(f"[stage3] verification log -> {self.verify_log_path}")
        self.get_logger().info(f"[stage5] all log -> {self.all_log_path}")
        self.get_logger().info(f"[stage5] order log -> {self.order_log_path}")

    def _append_log(self, line: str):
        if self.verify_log_path is None:
            return
        with open(self.verify_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _write_all_and_order_logs(self, picked_kind=None):
        if self.all_log_path is None or self.order_log_path is None:
            return
        shelf_priority = {shelf: idx for idx, shelf in enumerate(PICK_SEARCH_ORDER)}
        findings = sorted(
            self.findings,
            key=lambda item: (
                shelf_priority.get(item["shelf"], len(PICK_SEARCH_ORDER)),
                int(item["marker"])))
        lines = [
            "== All recognized product slots ==",
            f"run_prefix={self.run_prefix}",
            "",
        ]
        if findings:
            lines.extend(self._format_finding(item) for item in findings)
        else:
            lines.append("(empty)")
        self.all_log_path.write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

        pending_kinds = sorted(
            kind for kind, count in self.pending_counts.items() if count > 0)
        picked_kinds = []
        for target in self.targets:
            kind = target["kind"]
            if kind not in pending_kinds and kind not in picked_kinds:
                picked_kinds.append(kind)
        order_lines = [
            "== Order pending slots ==",
            f"run_prefix={self.run_prefix}",
            f"pick order: A -> B -> C -> D -> E",
            "",
        ]
        for kind in pending_kinds:
            kind_slots = [item for item in findings if item["kind"] == kind]
            if kind_slots:
                order_lines.extend(self._format_finding(item) for item in kind_slots)
            else:
                order_lines.append(f"[MISS] {kind}")
        order_lines.extend(["", "== Order status =="])
        for kind in picked_kinds:
            order_lines.append(f"[picked] 订单{kind}已抓取")
        for kind in pending_kinds:
            count = self.pending_counts[kind]
            order_lines.append(f"[pending] 订单{kind}还有{count}个待抓取")
        self.order_log_path.write_text(
            "\n".join(order_lines) + "\n", encoding="utf-8")

    def _final_summary(self):
        """全部货架扫完后, 汇总每个订单商品的最终 世界坐标+ArUco货位."""
        lines = ["[final] 订单商品最终映射 (kind -> 世界坐标 -> ArUco货位):"]
        mapped = 0
        for t in self.targets:
            kind = t["kind"]
            f = self.final.get(kind)
            if not f:
                lines.append(f"  {t['id']:28s} {kind:14s} -> MISS")
                continue
            mapped += 1
            c = f["coords"]
            lines.append(
                f"  {t['id']:28s} {kind:14s} -> shelf={f['shelf']} "
                f"coords≈({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) "
                f"marker={f['marker']} slot={f['slot']} samples={f['samples']}")
        lines.append(f"  mapped {mapped}/{len(self.targets)}")
        return "\n".join(lines)

    # ---- 主循环 ----
    def tick(self):
        if self.base_xy is None or self.jpos is None:
            return
        if self.run_start is not None and self.now() - self.run_start > RUN_TIMEOUT:
            self.phase = DONE

        if self.phase == WAIT_TASK:
            self.set_twist(0.0, 0.0)
            if self.task_ready:
                self.reset_nav()
                self.nav_t0 = self.now()
                self.phase = NAV_TO_SHELF
                self.get_logger().info("[stage3] order received -> patrol shelves")
        elif self.phase == NAV_TO_SHELF:
            if self.now() - self.nav_t0 > NAV_TIMEOUT:
                self.get_logger().warn("[stage3] nav timeout; skip to next shelf")
                self.phase = NEXT_SHELF
            elif self.follow_route(self._shelf_route(), SCAN_YAW):
                self.phase = SCAN_SHELF
                self._start_sweep()
                self.get_logger().info(
                    f"[stage3] parked at shelf {self.cur_shelf}; head up & sweep")
        elif self.phase == SCAN_SHELF:
            self.set_twist(0.0, 0.0)
            self.tc[3] = 0.0
            # 逐层驻留: 每层先稳定(LEVEL_SETTLE)再采集(DWELL_PER_LEVEL); pass1 向下 / pass2 向上.
            lvl = self.sweep_levels[self.scan_lvl_idx]
            # 持续下发本层 slide/pitch 目标(升降到位由控制器完成)
            self.tc[2] = self._slide_for(lvl)
            self.tc[4] = PITCH_BY_LEVEL[lvl]
            since = self.now() - self.level_t0
            if not self.level_settled and since >= LEVEL_SETTLE:
                self.level_settled = True
                # 稳定后清掉过渡期间的移动检测, 进入纯采集窗口
                self.det_buf.clear()
                self.scan_t0 = self.now()
            if self.level_settled:
                n_before = len(self.det_buf)
                self._consume_perception()
                self.scan_dets += n_before - len(self.det_buf)
                if self.now() - self.scan_t0 >= DWELL_PER_LEVEL:
                    if self.scan_lvl_idx + 1 < len(self.sweep_levels):
                        self.scan_lvl_idx += 1
                        self._enter_scan_level()
                    else:
                        self._finish_shelf_scan()
        elif self.phase == NEXT_SHELF:
            self.set_twist(0.0, 0.0)
            self.shelf_idx += 1
            if self.shelf_idx >= len(SEARCH_ORDER):
                self.phase = REPORT
            else:
                self.cur_shelf = SEARCH_ORDER[self.shelf_idx]
                self.reset_nav()
                self.nav_t0 = self.now()
                self.phase = NAV_TO_SHELF
        elif self.phase == REPORT:
            self.set_twist(0.0, 0.0)
            summary = self._final_summary() if self.targets else \
                "[final] no order targets"
            self.get_logger().info("\n" + summary)
            if self.total_det_count == 0:
                self.get_logger().warn(
                    "[stage3] PERCEPTION-ALERT: 0 product detections received "
                    "across all shelves; product_detect is not publishing. "
                    "Check camera depth/camera_info/odom and GPU. "
                    "(aruco detections ARE flowing, so RGB is up)")
            self._append_log("")
            for ln in summary.splitlines():
                self._append_log(ln)
            self._write_all_and_order_logs()
            self.get_logger().info("[stage5] stage3 mapping finished; start A->E ordered picking")
            self.phase = PLAN_PICK
        elif self.phase == PLAN_PICK:
            self.set_twist(0.0, 0.0)
            self._plan_nearest_pick()
        elif self.phase == NAV_PICK:
            if self.now() - self.nav_t0 > NAV_TIMEOUT:
                self.get_logger().warn(
                    f"[stage5] nav timeout at slot={self.current_finding['slot']}; skip")
                self.used_markers.add(self._candidate_key(self.current_finding))
                self.phase = PLAN_PICK
            elif self.follow_route(self.pick_route, GRASP_YAW):
                self.set_twist(0.0, 0.0)
                self._start_arm_ready()
        elif self.phase == ARM_READY:
            self.set_twist(0.0, 0.0)
            self.tc[12:18] = UNSTOW_R
            self.tc[18] = self._grip_open_for(self.current_kind)
            if self.arm_ready_done():
                self._prepare_arm_target()
            elif self.now() - self.state_t0 > DEPLOY_TIMEOUT:
                self.get_logger().warn("[stage5] arm-ready timeout; skip candidate")
                self.used_markers.add(self._candidate_key(self.current_finding))
                self.phase = PLAN_PICK
        elif self.phase == DEPLOY_ARM:
            self.set_twist(0.0, 0.0)
            if self.deploy_done():
                self.state_t0 = self.now()
                self.phase = CREEP
            elif self.now() - self.state_t0 > DEPLOY_TIMEOUT:
                self.get_logger().warn("[stage5] deploy timeout; skip candidate")
                self.used_markers.add(self._candidate_key(self.current_finding))
                self.phase = PLAN_PICK
        elif self.phase == CREEP:
            if self.creep_last_xy is None:
                self.creep_last_xy = self.base_xy.copy()
                self.creep_last_move_t = self.now()
            moved = float(np.linalg.norm(self.base_xy - self.creep_last_xy))
            if moved >= CREEP_MOVE_EPS:
                self.creep_last_xy = self.base_xy.copy()
                self.creep_last_move_t = self.now()
            stuck = (self.now() - self.creep_last_move_t) > CREEP_STUCK_TIMEOUT
            ee_y = float(self.ee_world()[1])
            if ee_y < self.creep_stop_y:
                if stuck:
                    # 基座被货架抵住且末端仍没够到商品 -> 抓取失败, 退离后重规划.
                    self.get_logger().warn(
                        f"[stage5] grasp did not reach object (ee_y={ee_y:.3f} "
                        f"< stop_y={self.creep_stop_y:.3f}); FAILED, retract & replan")
                    self.set_twist(0.0, 0.0)
                    self.grab_failed = True
                    self.state_t0 = self.now()
                    self.phase = RETRACT
                else:
                    self.set_twist(
                        CREEP_SPEED,
                        CREEP_YAW_KP * wrap_to_pi(GRASP_YAW - self.base_yaw))
            else:
                self.set_twist(0.0, 0.0)
                self.grab_failed = False
                self.state_t0 = self.now()
                self.phase = CLOSE_GRIP
        elif self.phase == CLOSE_GRIP:
            self.set_twist(0.0, 0.0)
            self.tc[18] = self.cur_params.grip_close if self.cur_params else GRIP_FALLBACK_CLOSE
            cmd_ok = abs(self.action[18] - self.tc[18]) <= CLOSE_CMD_TOL
            if cmd_ok:
                if self.grip_closed_t0 is None:
                    self.grip_closed_t0 = self.now()
                elif self.now() - self.grip_closed_t0 >= CLOSE_CMD_DWELL:
                    self.state_t0 = self.now()
                    if self.cur_params is not None:
                        self.tc[2] = max(0.0, self.tc[2] - self.cur_params.lift_amount)
                    self.phase = LIFT
            else:
                # 命令未到位或实测仍张开: 继续等待, 超时容错进入抬升.
                self.grip_closed_t0 = None
                if self.now() - self.state_t0 > GRIP_CLOSE_TIMEOUT:
                    self.get_logger().warn("[stage5] grip close timeout; proceed")
                    self.grip_closed_t0 = None
                    self.state_t0 = self.now()
                    if self.cur_params is not None:
                        self.tc[2] = max(0.0, self.tc[2] - self.cur_params.lift_amount)
                    self.phase = LIFT
        elif self.phase == LIFT:
            self.set_twist(0.0, 0.0)
            if self.now() - self.state_t0 > LIFT_DWELL:
                self.state_t0 = self.now()
                self.phase = RETRACT
        elif self.phase == RETRACT:
            if self.base_xy[1] > SCAN_APPROACH_Y + 0.06:
                self.set_twist(
                    -RETRACT_SPEED,
                    1.0 * wrap_to_pi(GRASP_YAW - self.base_yaw))
            else:
                self.set_twist(0.0, 0.0)
                if self.grab_failed:
                    self._finish_pick_failed()
                else:
                    self._finish_pick_hold()
        elif self.phase == STOW_GRIP_HOLD:
            # 保持抓取姿势送货: 不收臂, 只维持夹爪闭合, 确认后起步.
            self.set_twist(0.0, 0.0)
            self.tc[18] = self.carry_grip_close
            if self._grip_hold_done():
                self.nav_t0 = self.now()
                self.mppi_last = (0.0, 0.0)
                self.mppi_last_t = 0.0
                self.mppi_diag_t = 0.0
                self.phase = NAV_TO_DELIVERY
                self.get_logger().info("[stage4] grip hold; start delivery nav")
            elif self.now() - self.state_t0 > CARRY_HOLD_TIMEOUT:
                self.get_logger().error("[stage4] grip-hold timeout; stop")
                self.done_ts = None
                self.phase = DONE
        elif self.phase == NAV_TO_DELIVERY:
            if self.now() - self.nav_t0 > NAV_TIMEOUT:
                self.get_logger().warn("[stage4] delivery nav timeout; stop")
                self.done_ts = None
                self.phase = DONE
            else:
                r = self._navigate_to_delivery()
                if r is True:
                    self.get_logger().info(
                        "[stage4] arrived at delivery table; place item")
                    self.place_stage = "align"
                    self.place_world = None
                    self.place_rot = None
                    self.place_start_xy = None
                    self.state_t0 = self.now()
                    self.phase = PLACE_ITEM
                elif r is False:
                    self.get_logger().error(
                        "[stage4] delivery route failed to complete")
                    self.done_ts = None
                    self.phase = DONE
        elif self.phase == PLACE_ITEM:
            if self.now() - self.state_t0 > PLACE_TIMEOUT:
                # 兜底: 即使放置未完全到位, 也松爪收臂并按成功计数, 继续返程送下一件.
                self.get_logger().warn(
                    "[place] place timeout; force finish & continue return")
                self.set_twist(0.0, 0.0)
                self.tc[18] = GRIP_OPEN
                self.tc[12:18] = list(STOW_R)
                self._finish_place()
            else:
                r = self._place_item_at_delivery()
                if r is False:
                    self.get_logger().warn(
                        "[place] placement failed; force finish & continue")
                    self.set_twist(0.0, 0.0)
                    self.tc[18] = GRIP_OPEN
                    self.tc[12:18] = list(STOW_R)
                    self._finish_place()
                elif r is True:
                    pass  # _finish_place 已切换 phase(NAV_RETURN_TO_NEXT 或 DONE)
        elif self.phase == NAV_RETURN_TO_NEXT:
            if self.now() - self.nav_t0 > NAV_TIMEOUT:
                self.get_logger().error("[return] return nav timeout; stop")
                self.set_twist(0.0, 0.0)
                self.done_ts = None
                self.phase = DONE
            else:
                r = self._navigate_return_to_shelf()
                if r is True:
                    self.get_logger().info("[return] returned to next pick goal")
                    # 不重置 mppi_steer, 避免清理已生成的送货/返程可视化.
                    self.nav_idx = 0
                    self.nav_mode = "turn"
                    self.avoid_dir = 0
                    self.cur_lin = 0.0
                    self.cur_ang = 0.0
                    self.tc[0] = 0.0
                    self.tc[1] = 0.0
                    self.nav_t0 = self.now()
                    self.phase = PLAN_PICK
                elif r is False:
                    self.get_logger().error(
                        "[return] route failed; stop at current pose")
                    self.set_twist(0.0, 0.0)
                    self.done_ts = None
                    self.phase = DONE
        else:  # DONE
            self.set_twist(0.0, 0.0)
            if self.done_ts is not None and self.now() - self.done_ts > 3.0:
                self.get_logger().info("[stage3] done; exiting")
                os._exit(0)

        self.ramp_twist()
        self.smooth_step()
        self.publish()

        if self.now() - self.last_log > 2.0:
            self.get_logger().info(
                f"phase={PHASE_NAME[self.phase]} shelf={self.cur_shelf} "
                f"n_det={len(self.det_buf)} n_aruco={len(self.aruco_buf)} "
                f"mapped={len(self.slot_map.mapped_kinds())} "
                f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f})")
            self.last_log = self.now()


def _run_ros():
    rclpy.init()
    node = Stage35Client()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def selftest() -> int:
    """离线验证: 合成场景建图 + 逐订单 FIND-TEST 语义检查."""
    from common.slot_map_j3 import make_synthetic_scene, selftest as map_selftest
    res = map_selftest(seed=7, noise_xy=0.02, noise_z=0.01)
    slots = make_synthetic_scene(seed=7)
    sm = SlotMap()
    for shelf in "ABCDE":
        shelf_slots = [s for s in slots if s["shelf"] == shelf]
        markers = [(s["marker_id"], s["marker_xyz"]) for s in shelf_slots]
        for s in shelf_slots:
            for _ in range(5):
                sm.observe(s["kind"], s["product_xyz"], markers)
    # FIND-TEST 语义: 每个 kind 的解析结果必须落在该 kind 的真值货位集
    ok = 0
    for kind in sorted({s["kind"] for s in slots}):
        e = sm.kind_to_slot(kind)
        gt = {s["marker_id"] for s in slots if s["kind"] == kind}
        if e.mapped and e.marker_id in gt:
            ok += 1
    total = len({s["kind"] for s in slots})
    print(f"[client_task_5_j35] selftest: find-test semantic {ok}/{total}")
    print(sm.report())
    return 0 if (res["obs_acc"] >= 0.95 and res["kind_acc"] >= 0.95 and ok == total) else 1


def main():
    parser = argparse.ArgumentParser(
        description="Stage-3 建图 + Stage-5 最近货位抓取客户端 (j35)")
    parser.add_argument("--selftest", action="store_true",
                        help="离线验证映射算法(无需 ROS/服务器)")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if not _ROS_AVAILABLE:
        print("rclpy not available; run with --selftest or run inside the client container")
        raise SystemExit(2)
    _run_ros()


if __name__ == "__main__":
    main()
