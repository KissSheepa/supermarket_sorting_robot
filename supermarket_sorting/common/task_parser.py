#!/usr/bin/env python3
"""Stage-2 task parsing for the Supermarket Sorting Task (DG-202606).

Independent, dependency-free parser for the official latched task message
published on /supermarket_sorting/task (std_msgs/String, JSON):

    {
      "schema_version": 1,
      "run_prefix": "run_a1b2c3d4e5f6",
      "count": 5,
      "targets": [
        {"id": "item_run_a1b2c3d4e5f6_01", "kind": "kele"},
        ...
      ]
    }

Design goals
------------
* No ROS2 import here: the parser core can be unit-tested on any machine.
* Strict on structure (malformed JSON / missing fields / count mismatch are
  hard errors), lenient on unknown kinds (forward compatibility with the
  official kind list, but they are reported via invalid_kinds).
* TargetState tracks every target id through the official lifecycle
  unfound -> located -> grasped -> delivered | failed, and resets on a new
  run_prefix (no cross-run cache leakage).

The ROS2 subscription wrapper lives in a separate module
(scripts/task_parser_node.py) so client code can opt in without forcing the
core parser to depend on rclpy.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Official 9 product kinds; order mirrors gen_dataset.CLASSES /
# yolo_backend.CLASS_NAMES (this is only used for validation, not inference).
VALID_KINDS = frozenset({
    "sanmingzhi", "heweidao", "shupian", "zhijin",
    "maidong", "kouxiangtang", "pingguo", "chengzi", "kele",
})


class TaskParseError(ValueError):
    """Raised when the task JSON is structurally invalid."""


@dataclass
class TaskData:
    """Parsed and validated order task."""

    schema_version: int
    run_prefix: str
    count: int
    targets: List[Dict[str, str]] = field(default_factory=list)
    pending_counts: Counter = field(default_factory=Counter)
    invalid_kinds: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Derived convenience fields.
        self.target_ids: List[str] = [t["id"] for t in self.targets]
        self.kinds: List[str] = [t["kind"] for t in self.targets]
        self.pending_counts = Counter(self.kinds)

    @property
    def is_empty(self) -> bool:
        return self.count == 0 or len(self.targets) == 0


def parse_task(text: str) -> TaskData:
    """Parse the official task JSON string into a validated TaskData.

    Raises TaskParseError on any structural problem; unknown kinds are kept
    but reported through TaskData.invalid_kinds.
    """
    if not isinstance(text, str) or not text.strip():
        raise TaskParseError("task message is empty")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TaskParseError(f"invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise TaskParseError(f"task root must be an object, got {type(data).__name__}")

    # schema_version (int; tolerate missing by defaulting to None? -> strict:
    # official message always carries it, so require it but accept 1).
    if "schema_version" not in data:
        raise TaskParseError("missing required field 'schema_version'")
    try:
        schema_version = int(data["schema_version"])
    except (TypeError, ValueError) as exc:
        raise TaskParseError("'schema_version' must be an integer") from exc

    # run_prefix (non-empty string)
    run_prefix = data.get("run_prefix")
    if not isinstance(run_prefix, str) or not run_prefix:
        raise TaskParseError("missing or empty 'run_prefix'")

    # count (int, must equal len(targets))
    if "count" not in data:
        raise TaskParseError("missing required field 'count'")
    try:
        count = int(data["count"])
    except (TypeError, ValueError) as exc:
        raise TaskParseError("'count' must be an integer") from exc

    raw_targets = data.get("targets")
    if not isinstance(raw_targets, list):
        raise TaskParseError("'targets' must be a list")

    targets: List[Dict[str, str]] = []
    invalid_kinds: List[str] = []
    for idx, item in enumerate(raw_targets):
        if not isinstance(item, dict):
            raise TaskParseError(f"targets[{idx}] must be an object")
        tid = item.get("id")
        kind = item.get("kind")
        if not isinstance(tid, str) or not tid:
            raise TaskParseError(f"targets[{idx}] missing non-empty 'id'")
        if not isinstance(kind, str) or not kind:
            raise TaskParseError(f"targets[{idx}] missing non-empty 'kind'")
        if kind not in VALID_KINDS:
            invalid_kinds.append(kind)
        targets.append({"id": tid, "kind": kind})

    if count != len(targets):
        raise TaskParseError(
            f"'count' ({count}) != len(targets) ({len(targets)})")

    return TaskData(
        schema_version=schema_version,
        run_prefix=run_prefix,
        count=count,
        targets=targets,
        invalid_kinds=invalid_kinds,
    )


# ---------------------------------------------------------------------------
# Target lifecycle state machine (Stage-2.2 requirement)
# ---------------------------------------------------------------------------
TARGET_UNFOUND = "unfound"
TARGET_LOCATED = "located"
TARGET_GRASPED = "grasped"
TARGET_DELIVERED = "delivered"
TARGET_FAILED = "failed"

_VALID_TRANSITIONS = {
    TARGET_UNFOUND: {TARGET_LOCATED, TARGET_FAILED},
    TARGET_LOCATED: {TARGET_GRASPED, TARGET_FAILED},
    TARGET_GRASPED: {TARGET_DELIVERED, TARGET_FAILED},
    TARGET_DELIVERED: set(),
    TARGET_FAILED: set(),
}


class TargetState:
    """Per-target lifecycle tracker, scoped to one run_prefix.

    reset() must be called whenever run_prefix changes (cross-run cache guard).
    """

    def __init__(self, task: Optional[TaskData] = None) -> None:
        self.run_prefix: Optional[str] = None
        self._states: Dict[str, str] = {}
        self._kind_of: Dict[str, str] = {}
        if task is not None:
            self.reset(task)

    def reset(self, task: TaskData) -> None:
        """(Re)initialise from a parsed task; clears all previous state."""
        self.run_prefix = task.run_prefix
        self._states = {t["id"]: TARGET_UNFOUND for t in task.targets}
        self._kind_of = {t["id"]: t["kind"] for t in task.targets}

    def _transition(self, target_id: str, new_state: str) -> bool:
        if target_id not in self._states:
            raise KeyError(f"target id not in current task: {target_id}")
        cur = self._states[target_id]
        allowed = _VALID_TRANSITIONS[cur]
        if new_state not in allowed:
            raise ValueError(
                f"illegal transition {cur} -> {new_state} for {target_id}")
        self._states[target_id] = new_state
        return True

    def mark_located(self, target_id: str) -> bool:
        return self._transition(target_id, TARGET_LOCATED)

    def mark_grasped(self, target_id: str) -> bool:
        return self._transition(target_id, TARGET_GRASPED)

    def mark_delivered(self, target_id: str) -> bool:
        return self._transition(target_id, TARGET_DELIVERED)

    def mark_failed(self, target_id: str) -> bool:
        return self._transition(target_id, TARGET_FAILED)

    def state_of(self, target_id: str) -> str:
        return self._states[target_id]

    def kind_of(self, target_id: str) -> str:
        return self._kind_of[target_id]

    def by_state(self, state: str) -> List[str]:
        return [tid for tid, st in self._states.items() if st == state]

    @property
    def remaining(self) -> List[str]:
        """Target ids not yet delivered/failed (still actionable)."""
        return [tid for tid, st in self._states.items()
                if st not in (TARGET_DELIVERED, TARGET_FAILED)]

    @property
    def remaining_kinds(self) -> Counter:
        return Counter(self._kind_of[t] for t in self.remaining)

    @property
    def all_done(self) -> bool:
        return not self.remaining


if __name__ == "__main__":
    import sys

    demo = json.dumps({
        "schema_version": 1,
        "run_prefix": "run_demo123456",
        "count": 5,
        "targets": [
            {"id": "item_run_demo123456_02", "kind": "pingguo"},
            {"id": "item_run_demo123456_01", "kind": "kele"},
            {"id": "item_run_demo123456_03", "kind": "chengzi"},
            {"id": "item_run_demo123456_04", "kind": "zhijin"},
            {"id": "item_run_demo123456_05", "kind": "shupian"},
        ],
    })
    t = parse_task(demo)
    print("parsed:", t.schema_version, t.run_prefix, t.count, dict(t.pending_counts))
    ts = TargetState(t)
    print("state after reset:", ts.by_state(TARGET_UNFOUND))
    for tid in t.target_ids:
        ts.mark_located(tid)
        ts.mark_grasped(tid)
        ts.mark_delivered(tid)
    print("all_done:", ts.all_done)
    sys.exit(0)
