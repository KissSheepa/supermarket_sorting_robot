#!/usr/bin/env python3
"""Official-format task publisher for /supermarket_sorting/task.

The official server delivers the whole run's order ONCE per run on
/supermarket_sorting/task (std_msgs/String, JSON):

    {"schema_version":1,"run_prefix":"run_a1b2c3d4e5f6","count":5,
     "targets":[{"id":"item_run_a1b2c3d4e5f6_01","kind":"kele"}, ...]}

Spec facts honoured here (official V2.0):
  * count equals len(targets)
  * id is an anonymous per-run identifier (item_run_<prefix>_NN)
  * kind is the product category
  * the message carries NO location / shelf / aruco / world information
  * every server start regenerates run_prefix and target ids

This publisher mimics that server behaviour for development.  QoS is
RELIABLE + TRANSIENT_LOCAL + depth=1, matching the official topic.  Every
published task is validated through common.task_parser BEFORE it is sent
(malformed tasks are rejected, nothing is published).

Modes
-----
  default   : fresh random 5-order task, official schema
  --kinds   : publish exactly the given kind list (e.g. the official sample
              "kele pingguo chengzi zhijin shupian")
  --task-json: publish a user-supplied official JSON verbatim (a literal
              JSON string or a path to a .json file), validated first
  --once    : publish one task then exit; otherwise a fresh task every
              --interval seconds (each one a NEW run_prefix)

Usage (client container):
  source /opt/ros/humble/setup.bash
  python3 scripts/task_publisher.py --kinds kele pingguo chengzi zhijin shupian --once
  python3 scripts/task_publisher.py --task-json '{"schema_version":1,...}' --once
  python3 scripts/task_publisher.py --count 5 --interval 60
"""

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

from common.task_parser import TaskParseError, parse_task
from common.task_publisher_core import (
    generate_random_task,
    generate_task_from_kinds,
    task_to_json,
)


def _resolve_task(args) -> str:
    """Build the task JSON text to publish, validating it first.

    Returns the exact string that will be sent.  Raises TaskParseError (or
    ValueError) without publishing anything when the task is not valid.
    """
    if args.task_json is not None:
        text = args.task_json
        if os.path.isfile(text):
            with open(text, "r", encoding="utf-8") as f:
                text = f.read().strip()
        parse_task(text)                      # strict official-schema check
        return text

    rng = random.Random(args.seed)
    if args.kinds:
        task = generate_task_from_kinds(args.kinds, rng=rng)
    else:
        task = generate_random_task(args.count, rng=rng)
    text = task_to_json(task)
    parse_task(text)                          # self-check before publishing
    return text


class OfficialTaskPublisher(Node):
    def __init__(self, task_text: str, interval: float, once: bool):
        super().__init__("official_task_publisher")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(String, "/supermarket_sorting/task", qos)
        self.task_text = task_text

        self.publish_now()
        if once:
            # Linger so late-joining TRANSIENT_LOCAL subscribers can latch it.
            self.create_timer(2.0, self._stop)
        else:
            self.create_timer(interval, self.publish_now)

    def publish_now(self):
        msg = String(data=self.task_text)
        self.pub.publish(msg)
        self.get_logger().info(f"published official task: {self.task_text}")

    def _stop(self):
        self.get_logger().info("--once mode: exiting")
        self.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="official-format task publisher (/supermarket_sorting/task)")
    parser.add_argument("--count", type=int, default=5,
                        help="number of random targets (official: 5)")
    parser.add_argument("--kinds", nargs="+", default=None,
                        help="explicit kind list, e.g. kele pingguo chengzi zhijin shupian")
    parser.add_argument("--task-json", default=None,
                        help="official JSON string or path to a .json file (published verbatim)")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="seconds between new tasks (default mode; new run_prefix each)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducibility (default: entropy)")
    parser.add_argument("--once", action="store_true",
                        help="publish one task then exit")
    args = parser.parse_args()

    if args.task_json is not None and (args.kinds or args.count != 5):
        print("[task_publisher] --task-json takes precedence; ignoring --kinds/--count")
    try:
        text = _resolve_task(args)
    except (TaskParseError, ValueError) as exc:
        print(f"[task_publisher] REFUSED to publish invalid task: {exc}")
        sys.exit(2)

    rclpy.init()
    node = OfficialTaskPublisher(text, args.interval, args.once)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
