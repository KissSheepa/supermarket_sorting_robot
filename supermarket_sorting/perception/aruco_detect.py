#!/usr/bin/env python3
"""Detect the 45 shelf ArUco markers (DICT_4X4_50, IDs 0..44) from MMK2 cameras.

Baseline adaptation of the official
``examples/supermarket_sorting/perception/aruco_detect.py``:

* keeps the official per-camera topics and TF broadcast
  (/aruco/{cam}/ids|poses|detections|result_image; markers attached to the
  camera frame), so close-up wrist-camera debugging stays identical;
* adds ``/aruco/detections``: a WORLD-frame vision_msgs/Detection3DArray for
  the HEAD camera, built with odom + MMK2FK (the same transform chain as
  ``product_detect.py``).  Marker world coordinates therefore come from the
  robot's own sensors, never from a server-side layout file (规则: 不得读取
  Server 内部布局文件).

The ID -> 货位 (shelf/level/column) mapping is a published constant
(see common/aruco_slots.json) and does NOT encode product kind or position.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
import signal
import tf2_ros
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Int32MultiArray, String
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from kinematics.mmk2_fk import MMK2FK

MARKER_SIZE_M = 0.03
VALID_IDS = set(range(45))
CAMERAS = {
    "head": ("/head_camera/color/image_raw", "/head_camera/color/camera_info", "head_camera"),
    "left": ("/left_camera/color/image_raw", "/left_camera/color/camera_info", "left_camera"),
    "right": ("/right_camera/color/image_raw", "/right_camera/color/camera_info", "right_camera"),
}


def rotation_matrix_to_quaternion(matrix):
    """Return an xyzw quaternion without requiring an extra geometry package."""
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if i == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                         (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                     0.25 * s, (m[1, 0] - m[0, 1]) / s])


def solve_marker_pose(corners, camera_matrix, distortion, marker_size):
    """solvePnP in the camera optical frame; returns (rvec, tvec) or None."""
    half = marker_size * 0.5
    object_points = np.array([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, np.asarray(corners, dtype=np.float32).reshape(4, 2),
        camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


class ArucoDetectNode(Node):
    def __init__(self, camera_name, marker_size=MARKER_SIZE_M, publish_tf=True,
                 publish_result_image=True, enable_world=True):
        super().__init__(f"aruco_detect_{camera_name}")
        image_topic, info_topic, default_frame = CAMERAS[camera_name]
        self.camera_name = camera_name
        self.default_frame = default_frame
        self.marker_size = marker_size
        self.publish_tf = publish_tf
        self.publish_result_image = publish_result_image
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion = np.zeros(5, dtype=float)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            self._detect = lambda gray: detector.detectMarkers(gray)[:2]
        else:
            self._detect = lambda gray: cv2.aruco.detectMarkers(
                gray, dictionary, parameters=parameters)[:2]

        prefix = f"/aruco/{camera_name}"
        self.ids_pub = self.create_publisher(Int32MultiArray, f"{prefix}/ids", 10)
        self.poses_pub = self.create_publisher(PoseArray, f"{prefix}/poses", 10)
        self.detections_pub = self.create_publisher(String, f"{prefix}/detections", 10)
        self.result_pub = self.create_publisher(Image, f"{prefix}/result_image", 5)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if publish_tf else None

        # world-frame marker poses for the head camera only
        self.enable_world = enable_world and camera_name == "head"
        self.fk = MMK2FK()
        self.base_pos = None
        self.base_quat = None
        self.slide = 0.0
        self.head = [0.0, 0.0]
        if self.enable_world:
            self.world_pub = self.create_publisher(Detection3DArray, "/aruco/detections", 10)
            self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
            self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                     self.odom_cb, 10)

        self.create_subscription(CameraInfo, info_topic, self.camera_info_cb, 10)
        self.create_subscription(Image, image_topic, self.image_cb, 10)
        self.get_logger().info(
            f"listening on {image_topic}; dictionary=DICT_4X4_50, "
            f"marker_size={marker_size:.3f}m world={self.enable_world}")

    # ---- live robot state (world-frame head camera only) ----
    def js_cb(self, msg: JointState):
        jp = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        self.slide = jp.get("slide_joint", self.slide)
        self.head = [jp.get("head_yaw_joint", self.head[0]),
                     jp.get("head_pitch_joint", self.head[1])]

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_pos = [p.x, p.y, p.z]
        self.base_quat = [q.w, q.x, q.y, q.z]

    def camera_world_tmat(self):
        """4x4 head-camera(optical)->world built from odom + slide/head (MMK2FK)."""
        if self.base_pos is None or self.base_quat is None:
            return None
        self.fk.set_base_pose(self.base_pos, self.base_quat)
        self.fk.set_slide_joint(float(self.slide))
        self.fk.set_head_joints([float(self.head[0]), float(self.head[1])])
        pos, quat = self.fk.get_head_camera_pose()   # quat wxyz, world
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        return T

    def camera_info_cb(self, msg):
        self.camera_matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
        if msg.d:
            self.distortion = np.asarray(msg.d, dtype=float)

    def image_cb(self, msg):
        if self.camera_matrix is None:
            return
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect(gray)
        valid = []
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                marker_id = int(marker_id)
                if marker_id not in VALID_IDS:
                    continue
                pose = solve_marker_pose(marker_corners, self.camera_matrix,
                                         self.distortion, self.marker_size)
                if pose is not None:
                    valid.append((marker_id, marker_corners, *pose))
        valid.sort(key=lambda item: item[0])
        self.publish_detections(msg, valid)
        if self.publish_result_image:
            self.publish_visualization(msg, image, valid)

    def publish_detections(self, image_msg, detections):
        frame_id = image_msg.header.frame_id or self.default_frame
        ids_msg = Int32MultiArray(data=[item[0] for item in detections])
        poses_msg = PoseArray()
        poses_msg.header = image_msg.header
        poses_msg.header.frame_id = frame_id
        records = []
        transforms = []
        for marker_id, _, rvec, tvec in detections:
            rotation, _ = cv2.Rodrigues(rvec)
            quat = rotation_matrix_to_quaternion(rotation)
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, tvec)
            pose.orientation.x, pose.orientation.y = float(quat[0]), float(quat[1])
            pose.orientation.z, pose.orientation.w = float(quat[2]), float(quat[3])
            poses_msg.poses.append(pose)
            records.append({"id": marker_id, "position": tvec.tolist(),
                            "quaternion_xyzw": quat.tolist()})
            if self.tf_broadcaster is not None:
                transform = TransformStamped()
                transform.header = poses_msg.header
                transform.child_frame_id = f"aruco_{self.camera_name}_{marker_id:02d}"
                transform.transform.translation.x = pose.position.x
                transform.transform.translation.y = pose.position.y
                transform.transform.translation.z = pose.position.z
                transform.transform.rotation = pose.orientation
                transforms.append(transform)
        self.ids_pub.publish(ids_msg)
        self.poses_pub.publish(poses_msg)
        self.detections_pub.publish(String(data=json.dumps(records, separators=(",", ":"))))
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

        # world-frame head-camera markers (robot's own odom+FK)
        if self.enable_world and detections:
            T_cam_world = self.camera_world_tmat()
            if T_cam_world is not None:
                msg = Detection3DArray()
                msg.header = image_msg.header
                msg.header.frame_id = "world"
                for marker_id, _, rvec, tvec in detections:
                    p_world = (T_cam_world @ np.array([tvec[0], tvec[1], tvec[2], 1.0]))[:3]
                    det = Detection3D()
                    det.header = msg.header
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(marker_id)
                    hyp.hypothesis.score = 1.0
                    hyp.pose.pose.position.x = float(p_world[0])
                    hyp.pose.pose.position.y = float(p_world[1])
                    hyp.pose.pose.position.z = float(p_world[2])
                    det.results.append(hyp)
                    msg.detections.append(det)
                self.world_pub.publish(msg)

    def publish_visualization(self, image_msg, image, detections):
        for marker_id, corners, rvec, tvec in detections:
            cv2.aruco.drawDetectedMarkers(image, [corners], np.array([[marker_id]]))
            cv2.drawFrameAxes(image, self.camera_matrix, self.distortion,
                              rvec, tvec, self.marker_size * 0.5)
        result = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        result.header = image_msg.header
        self.result_pub.publish(result)


def main():
    parser = argparse.ArgumentParser(description="Detect supermarket shelf ArUco markers")
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS),
                        help="camera nodes to start (default: head left right)")
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE_M,
                        help="marker side length in metres (default: 0.03)")
    parser.add_argument("--no-tf", action="store_true", help="disable marker TF publishing")
    parser.add_argument("--no-result-image", action="store_true",
                        help="disable annotated image publishing")
    parser.add_argument("--no-world", action="store_true",
                        help="disable world-frame /aruco/detections (head)")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _sigterm)
    rclpy.init()
    nodes = [ArucoDetectNode(camera, args.marker_size, not args.no_tf,
                             not args.no_result_image, not args.no_world)
             for camera in args.cameras]
    executor = MultiThreadedExecutor(num_threads=max(2, len(nodes)))
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # 进程被 kill 时 rclpy 上下文可能已失效, 静默退出
        if "context is not valid" in str(exc) or "ExternalShutdown" in str(exc):
            pass
        else:
            raise
    finally:
        executor.shutdown()
        for node in nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _sigterm(_signum, _frame):
    # 触发 KeyboardInterrupt, 走上面的优雅 shutdown 路径, 避免 RCLError 刷屏.
    raise KeyboardInterrupt


if __name__ == "__main__":
    main()
