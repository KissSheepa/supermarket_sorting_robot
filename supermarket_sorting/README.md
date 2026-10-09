# Supermarket Sorting Task

## 运行环境版本

- CUDA：12.8
- PyTorch：2.7.1+cu128
- ROS2：Humble

本仓库是独立 Baseline。Server 运行仿真，Client 提供固定运行环境，Baseline 源码通过
挂载进入 Client。固定 Baseline 用于验证一次抓取流程。正式任务中，选手需要在规定时间内
尽可能多地完成商品抓取和放置。

## 部署

宿主机需要 Docker、NVIDIA Driver（Linux >= 570.26）、NVIDIA Container Toolkit 和
NVIDIA GPU。下载离线镜像包：

- [Server 镜像 tar]链接: https://pan.baidu.com/s/1w6qeDOqi0hcLstdCetfGvQ 提取码: dhti 
- [Client 镜像 tar]链接: https://pan.baidu.com/s/1oSuz6v0cloXq1Mg74gF0tQ 提取码: d64j 

下载完成后，在 tar 文件所在目录加载镜像：

```bash
docker load -i supermarket_sorting_server.tar
docker load -i supermarket_sorting_client.tar

docker tag \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:server \
  supermarket_sorting:server
docker tag \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:client \
  supermarket_sorting:client
```

```bash
xhost +local:docker
docker volume create supermarket_sorting_cache
```

## 固定 Baseline

启动固定 Server：

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_server \
  -e DISPLAY=${DISPLAY} \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 \
  -e SUPERMARKET_USE_GS=1 \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e SUPERMARKET_FIXED_BASELINE=1 \
  -e SUPERMARKET_RANDOMIZE=0 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=0 \
  -e SUPERMARKET_TASKS=product_032 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  supermarket_sorting:server \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

启动固定 Client，并挂载本仓库：

```bash
docker run --rm -dit \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_client \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v "$(pwd)":/workspace/baseline:ro \
  supermarket_sorting:client
```

在 Client 中启动 Baseline：

```bash
docker exec -it supermarket_sorting_client \
  bash -lc 'cd /workspace/baseline && ./scripts/run_baseline.sh'
```

## 正式运行

正式 Server 使用随机商品和随机障碍物：

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_server \
  -e DISPLAY=${DISPLAY} \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 \
  -e SUPERMARKET_USE_GS=1 \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e SUPERMARKET_RANDOMIZE=1 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=1 \
  -e SUPERMARKET_SEED=11 \
  -e SUPERMARKET_TASKS=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  supermarket_sorting:server \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

正式 Client 挂载选手 Baseline，启动后保持运行：

```bash
docker run -dit \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_client \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /path/to/your_baseline:/workspace/baseline:rw \
  supermarket_sorting:client
```

选手程序通过 `docker exec` 在 Client 容器内启动。

替换权重：

```text
weights/product9.pt
```

也可以设置 `SUPERMARKET_BASELINE_WEIGHTS=/workspace/baseline/weights/custom.pt`。

## 任务指令

Server 每次启动会随机放置 45 个商品和通道内 5 个障碍物。障碍物保持箱体竖直，只随机
改变平面偏航角，并通过路径检查保证货架入口至配送台入口存在通路。

Baseline 固定航点不处理随机障碍物。正式程序应读取二维雷达并自行规划：

```text
/slamware_ros_sdk_server_node/scan
```

任务消息示例：

```json
{"schema_version":1,"run_prefix":"run_a1b2c3d4e5f6","count":2,"targets":[{"id":"item_run_a1b2c3d4e5f6_01","kind":"kele"},{"id":"item_run_a1b2c3d4e5f6_02","kind":"kele"}]}
```

查看当前任务：

```bash
ros2 topic echo --once /supermarket_sorting/task
```

## ROS2 话题

`ROS_DOMAIN_ID` 必须在 Server 和 Client 之间保持一致。

### Server 发布

| Topic | Type | 说明 |
| --- | --- | --- |
| `/slamware_ros_sdk_server_node/odom` | `nav_msgs/msg/Odometry` | 底盘位姿和速度 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 动态 TF |
| `/slamware_ros_sdk_server_node/scan` | `sensor_msgs/msg/LaserScan` | 二维激光雷达，默认 12 Hz |
| `/joint_states` | `sensor_msgs/msg/JointState` | 关节状态 |
| `/head_camera/color/image_raw` | `sensor_msgs/msg/Image` | 头部 RGB |
| `/head_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 头部 RGB 内参 |
| `/head_camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | 头部深度，毫米 |
| `/head_camera/aligned_depth_to_color/camera_info` | `sensor_msgs/msg/CameraInfo` | 深度内参 |
| `/left_camera/color/image_raw` | `sensor_msgs/msg/Image` | 左腕 RGB |
| `/left_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 左腕内参 |
| `/right_camera/color/image_raw` | `sensor_msgs/msg/Image` | 右腕 RGB |
| `/right_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 右腕内参 |
| `/supermarket_sorting/task` | `std_msgs/msg/String` | JSON 任务清单 |

### Server 订阅

| Topic | Type | 控制格式 |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `linear.x`、`angular.z` |
| `/spine_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 升降柱 |
| `/head_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 头部关节 |
| `/left_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 左臂 6 轴和夹爪 |
| `/right_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 右臂 6 轴和夹爪 |

### Baseline 发布

| Topic | Type | 说明 |
| --- | --- | --- |
| `/product/detections` | `vision_msgs/msg/Detection3DArray` | 9 类商品世界坐标检测结果 |
| `/product/result_image` | `sensor_msgs/msg/Image` | 检测可视化图 |

`/joint_states` 顺序：

```text
slide_joint, head_yaw_joint, head_pitch_joint,
left_arm_joint1..left_arm_joint6, left_arm_eef_gripper_joint,
right_arm_joint1..right_arm_joint6, right_arm_eef_gripper_joint
```

## 参数说明

| 参数 | 推荐值 | 含义 |
| --- | --- | --- |
| `ROS_DOMAIN_ID` | `99` | Server 和 Client 通信域 |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | ROS2 RMW 实现 |
| `MUJOCO_GL` | `glfw` | 图形窗口；无头用 `egl` |
| `SUPERMARKET_HEADLESS` | `0` | 是否显示窗口 |
| `SUPERMARKET_ENABLE_RENDER` | `1` | 发布 RGB-D |
| `SUPERMARKET_ENABLE_LIDAR` | `1` | 发布雷达 |
| `SUPERMARKET_USE_GS` | `1` | 启用 3DGS |
| `SUPERMARKET_RANDOMIZE` | `1` | 随机商品位置 |
| `SUPERMARKET_SEED` | `11` | 商品随机种子 |
| `SUPERMARKET_RANDOMIZE_OBSTACLES` | `1` | 随机障碍物 |
| `SUPERMARKET_OBSTACLE_SEED` | 可选 | 障碍物随机种子 |
| `SUPERMARKET_TASKS` | `all` | 任务筛选 |
| `SUPERMARKET_BASELINE_WEIGHTS` | 可选 | 自定义权重路径 |

## 主要文件

```text
client_task_1.py             # 任务一控制程序
perception/product_detect.py  # 9 类商品视觉检测
perception/yolo_backend.py   # YOLO 后端
kinematics/                  # MMK2 FK/IK
models/mmk2_head_fk.xml      # 相机 FK 模型
weights/product9.pt              # 默认权重
scripts/run_baseline.sh      # 启动脚本
```

## 阶段 4：导航避障模块（2026-08-11）

### 代码组成
- `common/nav_map.py`：场地常量 + 占用栅格 + A* 全局规划（纯逻辑，无 ROS）
  - 场地 5m x 7.5m 实测建模：四面墙、货架区、送货区平台、固定走廊挡板（静态）
  - 动态障碍由激光/视觉注入，膨胀 0.35m 与官方生成器 ROBOT_CLEARANCE_RADIUS 对齐
- `common/laser_obstacle.py`：360 束激光 -> 世界系点 / 聚类 / 前侧净空
- `common/navigation.py`：导航状态机（turn/drive/avoid/recover/final_yaw）
  - 停滞检测 + 后退/转向脱困；同一航点连续 2 次 recover 强制重规划；8 次无进展判失败
  - 激光注入"成员点"而非聚类中心（旧实现把墙面巨型聚类压成圆心导致穿箱/贴墙）
- `scripts/plan_and_drive.py`：ROS2 执行节点（订阅 odom/scan，发布 /cmd_vel，`--stow-arm`）
- `common/fusion_obstacle.py`（新增）：RGB-D 深度图反投影 -> 世界点云 -> 过滤/下采样
  - `--fusion` 时订阅头相机深度图，与激光一起注入栅格，补充激光盲区
  - 相机外参：odom + joint_states -> MMK2FK -> head camera 位姿（与 aruco_detect 同链路）

### 验证结果（2026-08-11，随机障碍布局，GPU7）
| 场景 | 目标 | 结果 |
| --- | --- | --- |
| 单激光 | 货架 D 前方 (0.852, 2.475) 朝向 1.379 | 到达 + 对齐，1 次重规划 |
| 单激光 | 送货区 (-1.94, -2.64) | 到达，4 次重规划避障无死锁 |
| 激光+视觉 | 返回货架 D 前方 | 到达，视觉注入全程正常（100~1100+ 点/帧） |

### 仿真速度问题与修复（重要）
- 根因：官方 Server 的 3DGS 渲染每帧约 1s（3 相机顺序渲染 640x480），物理主循环
  （无独立节流）被渲染拖到 3-5% 实时；`cfg.sync` 睡眠只影响 <41ms 的快速帧，非主因
- 修复：`supermarket_sorting_server.py` 增加环境变量
  `SUPERMARKET_RENDER_FPS`（默认 24）、`SUPERMARKET_RENDER_W/H`（默认 640/480）；
  验证时用 `FPS=6 W=320 H=240`，物理恢复到约 19% 实时，挡板北角卡死消失
- 注意：保持 `cfg.sync` 官方默认（`sync=False` 曾导致 GS 渲染崩溃/僵尸）；
  首次启动加 `TORCH_CUDA_ARCH_LIST=8.6` 可大幅缩短 GS 扩展编译时间

### 下一步
- 主控 `client_task_1.py` 仍是旧反应式导航（硬编码航点 + bug 式绕行），
  后续把其导航段替换为 NavigationCore（A* + 动态注入 + 脱困）
- 融合升级：继续评估视觉点对低矮/贴地障碍的补充效果；接入主控后随任务流验证
