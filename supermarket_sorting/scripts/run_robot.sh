#!/usr/bin/env bash
# 最终主控运行脚本: 阶段3建图 + 阶段5抓取 + 阶段4 NAV+MPPI 送货/返程。
# 用法: bash scripts/run_robot.sh [--selftest]
set -eo pipefail

baseline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
cd "$baseline_root"

# 物理 GPU 4 only; 容器内映射为 cuda:0。
export CUDA_VISIBLE_DEVICES="${SUPERMARKET_CUDA_VISIBLE_DEVICES:-4}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export NAV_DIAG_VIS="${NAV_DIAG_VIS:-1}"

# ---- 阶段4 NAV+MPPI 送货/放置/返程可调参数(环境变量覆盖, 默认值见 client_task.py) ----
# 配送台(桌子)世界坐标 [x,y] 与朝向 yaw. 默认: 点数 [-1.94,-3.22], 朝向 -pi/2 (-90deg).
#   SUPERMARKET_DELIVERY_POINT="-1.94,-3.22"   SUPERMARKET_DELIVERY_YAW="-1.5708"
# 放置点在车头正前方 PLACE_FWD_DIST(米) 与桌面高度 PLACE_Z(米). 默认 0.42 / 0.62.
#   SUPERMARKET_PLACE_FWD_DIST="0.42"         SUPERMARKET_PLACE_Z="0.62"
# 返程兜底货位(找不到剩余候选时). 默认 A 架前.
#   SUPERMARKET_RETURN_SHELF_POINT="-1.803,2.20"   SUPERMARKET_RETURN_SHELF_YAW="1.4835"
# 抓取失败判定: 末端够到目标前位移小于 CREEP_MOVE_EPS 且持续 CREEP_STUCK_TIMEOUT 秒则判失败重规划.
#   SUPERMARKET_CREEP_MOVE_EPS="0.03"  SUPERMARKET_CREEP_STUCK_TIMEOUT="6.0"
# 感知/模型:
#   SUPERMARKET_BASELINE_WEIGHTS=<权重路径>  SUPERMARKET_DETECTOR_DEVICE="cuda:0"

if [[ "${1:-}" == "--selftest" ]]; then
  exec python3 client_task.py --selftest
fi

# Stage-1 perception: 9-class detect + ArUco (head camera).
mkdir -p "$baseline_root/logs"
detector_log="$baseline_root/logs/product_detect_$(date +%Y%m%d_%H%M%S).log"
aruco_log="$baseline_root/logs/aruco_detect_$(date +%Y%m%d_%H%M%S).log"
python3 -m perception.product_detect \
  --weights "${SUPERMARKET_BASELINE_WEIGHTS:-$baseline_root/weights/product9.pt}" \
  --device "${SUPERMARKET_DETECTOR_DEVICE:-cuda:0}" 2>&1 | tee "$detector_log" &
detector_pid=$!

python3 -m perception.aruco_detect --cameras head --marker-size 0.03 2>&1 | tee "$aruco_log" &
aruco_pid=$!

cleanup() {
  kill "$detector_pid" "$aruco_pid" 2>/dev/null || true
  wait "$detector_pid" "$aruco_pid" 2>/dev/null || true
}
trap cleanup EXIT

logfile="$baseline_root/logs/client_task_$(date +%Y%m%d_%H%M%S).log"
python3 client_task.py "$@" 2>&1 | tee "$logfile"
