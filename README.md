# supermarket_sorting_robot
# 智慧零售自主移动机器人全流程演示

本项目基于 **ROS 2 Humble**，实现了智慧零售场景下的移动操作机器人端到端闭环：涵盖任务接收、自主导航避障、商品检测定位、机械臂抓取及配送交付。

## 🎬 运行视频演示 (Demo)



https://github.com/user-attachments/assets/7083459d-4e1b-48fa-af82-81b8705fe8ad



### 流程说明：
1. **任务下发**：接收并解析零售订单（支持可乐、脉动、苹果等 9 类常见商品）。
2. **激光导航**：基于激光雷达点云与里程计在货架通道中自主寻路与动态避障。
3. **视觉定位**：头部相机识别 ArUco 标签实现货架精准对齐，并调用深度学习模型实时框选目标商品。
4. **机械臂抓取**：结合 MMK2 头部运动学解算末端位姿，控制机械臂完成无碰撞抓取。
5. **配送闭环**：送达指定位置并由主控状态机判定完成任务（输出 `phase=done`）。

---

## 🛠️ 核心技术栈

- **中间件与通信**：ROS 2 (Humble), CycloneDDS, Docker
- **感知与视觉**：PyTorch (商品目标检测), OpenCV (ArUco 位姿估计)
- **运动与控制**：MMK2 机器人运动学模型, MuJoCo 物理仿真引擎
- **底盘与定位**：2D 激光雷达 (Slamware SDK), 里程计融合导航

---

## 🚀 快速运行客户端

镜像内已集成全部代码、模型权重与运行环境，启动命令如下：

```bash
# 1. 创建并启动容器
sudo docker run -dit \
  --name y25_jqr_client \
  --gpus all \
  --network host \
  --ipc host \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  supermarket_sorting:client-self-contained-20260905

# 2. 容器内一键启动全流程
sudo docker exec -it y25_jqr_client bash -c "source /opt/ros/humble/setup.bash && cd /workspace/baseline && bash scripts/run_robot.sh"
https://github.com/user-attachments/assets/d632b13b-9523-411e-b07e-5fca81dab0b2





