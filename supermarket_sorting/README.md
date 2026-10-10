# 智慧零售自主服务机器人客户端部署说明

版本：2026-09-05  
适用对象：需要使用已打包客户端镜像复现项目的开发者

## 1. 文档范围

本说明只覆盖客户端镜像的导入、容器创建、运行和验证。客户端镜像内已经包含：

- ROS 2 Humble、Python 运行环境及项目依赖；
- 客户端代码，路径为 `/workspace/baseline`；
- 当前使用的商品检测模型 `weights/product9.pt`；
- 运动学模型 `models/mmk2_head_fk.xml`；
- 主控、感知、抓取、导航和任务发布脚本。

Server 仿真环境仍需单独准备。客户端容器通过 host 网络与 Server 容器使用 ROS 2 通信。

## 2. 软件和硬件要求

- Linux 主机，已安装 Docker；
- NVIDIA 驱动和 NVIDIA Container Toolkit；
- 可用 NVIDIA GPU；
- Server 与 Client 使用相同的 ROS 2 Domain；
- 推荐使用 `ROS_DOMAIN_ID=99` 和 `rmw_cyclonedds_cpp`；
- 已获取客户端镜像压缩包，例如 `supermarket_sorting_client_20260905.tar.gz`。

## 3. 导入客户端镜像

将镜像压缩包放到目标机器后执行：

```bash
gunzip supermarket_sorting_client_20260905.tar.gz
sudo docker load -i supermarket_sorting_client_20260905.tar
```

确认镜像：

```bash
sudo docker image ls | grep supermarket_sorting
```

镜像标签通常为：

```text
supermarket_sorting:client-self-contained-20260905
```

如果实际标签不同，以 `docker image ls` 显示的标签为准。

## 4. 创建客户端容器

客户端代码已经写入镜像，因此创建容器时不需要挂载宿主机项目目录：

```bash
sudo docker run -dit \
  --name y25_jqr_client \
  --gpus all \
  --network host \
  --ipc host \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  supermarket_sorting:client-self-contained-20260905
```

确认容器：

```bash
sudo docker ps --filter name=y25_jqr_client
```

进入容器：

```bash
sudo docker exec -it y25_jqr_client bash
```

## 5. 验证镜像内项目代码

容器内执行：

```bash
source /opt/ros/humble/setup.bash
cd /workspace/baseline

pwd
test -f client_task.py
test -f scripts/run_robot.sh
test -f scripts/task_publisher.py
test -f weights/product9.pt
test -f models/mmk2_head_fk.xml
python3 client_task.py --selftest
bash -n scripts/run_robot.sh
```

最后两条命令分别用于验证主控离线逻辑和启动脚本语法。

## 6. 启动 Server

Server 需要单独使用官方仿真项目和镜像。进入 Server 容器：

```bash
sudo docker start y25_jqr_server
sudo docker exec -it y25_jqr_server bash
```

容器内执行固定场景测试配置：

```bash
export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL=glfw
export SUPERMARKET_HEADLESS=0
export DISPLAY=:1
export SUPERMARKET_ENABLE_RENDER=1
export SUPERMARKET_ENABLE_LIDAR=1
export SUPERMARKET_USE_GS=1
export SUPERMARKET_GS_SEQUENTIAL=1
export SUPERMARKET_FIXED_BASELINE=0
export SUPERMARKET_RANDOMIZE=1
export SUPERMARKET_RANDOMIZE_OBSTACLES=1
export SUPERMARKET_SEED=11
export SUPERMARKET_TASKS=all
export ROS_DOMAIN_ID=99
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
cd /workspace/supermarket_sorting_task
python3 examples/supermarket_sorting/supermarket_sorting_server.py
```

需要固定单可乐基线时，将任务相关环境变量改为：

```bash
export SUPERMARKET_FIXED_BASELINE=1
export SUPERMARKET_RANDOMIZE=0
export SUPERMARKET_RANDOMIZE_OBSTACLES=0
export SUPERMARKET_TASKS=product_032
```

Server 启动成功的标志是日志出现任务发布信息，并且仿真场景正常显示。

## 7. 启动 Client

另开一个终端，在客户端容器内执行：

```bash
sudo docker start y25_jqr_client
sudo docker exec -it y25_jqr_client bash
source /opt/ros/humble/setup.bash
cd /workspace/baseline
bash scripts/run_robot.sh
```

脚本会自动启动商品检测、头部相机 ArUco 检测和 `client_task.py` 主控。默认检测 GPU 设置为物理 GPU 4；如需更换 GPU，可在启动前设置：

```bash
export SUPERMARKET_CUDA_VISIBLE_DEVICES=5
export SUPERMARKET_DETECTOR_DEVICE=cuda:0
bash scripts/run_robot.sh
```

## 8. 检查 ROS 通信

再开一个终端进入客户端容器：

```bash
sudo docker exec -it y25_jqr_client bash
source /opt/ros/humble/setup.bash
```

查看话题：

```bash
ros2 topic list
```

重点检查：

```bash
ros2 topic echo /supermarket_sorting/task --once
ros2 topic echo /product/detections --once
ros2 topic echo /aruco/detections --once
ros2 topic echo /slamware_ros_sdk_server_node/odom --once
ros2 topic echo /slamware_ros_sdk_server_node/scan --once
```

## 9. 发布任务

如果使用 Server 自动任务，则直接观察任务话题即可：

```bash
ros2 topic echo /supermarket_sorting/task --once
```

如果需要明确发布一组开发任务，在客户端容器内执行：

```bash
cd /workspace/baseline
python3 scripts/task_publisher.py \
  --kinds kele maidong chengzi pingguo shupian \
  --once
```

支持的商品类别包括：

```text
kele maidong chengzi pingguo kouxiangtang zhijin heweidao sanmingzhi shupian
```

同一次测试只选择一种任务来源，避免 Server 自动发布和手动发布同时改变订单。

## 10. 运行日志

日志保存在容器内：

```text
/workspace/baseline/logs/
```

由于项目目录位于镜像内部，日志会写入容器可写层。查看日志：

```bash
tail -f $(ls -t /workspace/baseline/logs/client_task_*.log | head -1)
tail -f $(ls -t /workspace/baseline/logs/product_detect_*.log | head -1)
tail -f $(ls -t /workspace/baseline/logs/aruco_detect_*.log | head -1)
```

主控正常完成时应看到：

```text
phase=done
```

## 11. 停止和恢复

在运行程序的终端按 `Ctrl+C`，`run_robot.sh` 会清理两个感知子进程。停止容器：

```bash
sudo docker stop y25_jqr_client
```

下次继续使用：

```bash
sudo docker start y25_jqr_client
```

## 12. 常见问题

### ROS 话题为空

确认 Server 和 Client 都使用：

```bash
ROS_DOMAIN_ID=99
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

并确认两个容器使用 `--network host`。

### 找不到 `rclpy`

容器内执行：

```bash
source /opt/ros/humble/setup.bash
```

### 找不到模型

确认当前目录为：

```bash
cd /workspace/baseline
```

并检查：

```bash
ls -lh weights/product9.pt models/mmk2_head_fk.xml
```

### 容器内没有项目代码

创建容器时不要使用宿主机空目录覆盖 `/workspace/baseline`，尤其不要添加：

```bash
-v /some/empty/path:/workspace/baseline
```

### GPU 不可用

在宿主机执行：

```bash
nvidia-smi
sudo docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```

## 13. 最简复现流程

```bash
# 导入镜像
gunzip supermarket_sorting_client_20260905.tar.gz
sudo docker load -i supermarket_sorting_client_20260905.tar

# 创建 Client
sudo docker run -dit \
  --name y25_jqr_client \
  --gpus all \
  --network host \
  --ipc host \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  supermarket_sorting:client-self-contained-20260905

# 运行 Client
sudo docker exec -it y25_jqr_client bash
source /opt/ros/humble/setup.bash
cd /workspace/baseline
bash scripts/run_robot.sh
```
