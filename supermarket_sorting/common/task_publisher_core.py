#!/usr/bin/env python3
"""Official-format task generators (dependency-free core).

Used by scripts/task_publisher.py (ROS2) and by the stage-2 verification
suite.  Generates task dicts matching the official /supermarket_sorting/task
JSON schema:

    {"schema_version":1,"run_prefix":"run_a1b2c3d4e5f6","count":5,
     "targets":[{"id":"item_run_a1b2c3d4e5f6_01","kind":"kele"}, ...]}

Rules honoured (official V2.0 spec):
  * count == len(targets)
  * id is an anonymous per-run identifier (item_run_<prefix>_NN)
  * kind is the product category
  * the message carries NO location / shelf / aruco / world information

The core deliberately has no ROS2 import so it can be unit-tested anywhere.
"""

from __future__ import annotations

import json
import random
from typing import List, Optional

from common.task_parser import VALID_KINDS


def _new_run_prefix(rng: random.Random) -> str:
    return "run_" + format(rng.getrandbits(48), "012x")


def generate_task_from_kinds(
    kinds: List[str],
    rng: Optional[random.Random] = None,
    run_prefix: Optional[str] = None,
) -> dict:
    """Build an official-format task from an explicit kind list.

    This is the direct way to mimic the official sample:
        --kinds kele pingguo chengzi zhijin shupian
    The order of ``kinds`` is preserved; ids are anonymous per the spec.
    """
    kinds = list(kinds)
    if not kinds:
        raise ValueError("kinds must not be empty")
    rng = rng or random.Random()
    prefix = run_prefix or _new_run_prefix(rng)
    targets = [
        {"id": f"item_{prefix}_{i + 1:02d}", "kind": kind}
        for i, kind in enumerate(kinds)
    ]
    return {
        "schema_version": 1,
        "run_prefix": prefix,
        "count": len(kinds),
        "targets": targets,
    }


def generate_random_task(
    count: int = 5,
    rng: Optional[random.Random] = None,
    kinds: Optional[List[str]] = None,
) -> dict:
    """Return an official-format task dict with ``count`` random targets.

    - run_prefix: 12 random hex chars, "run_<hex>" (fresh per call)
    - target ids: "item_<run_prefix>_NN" (NN = 1-based index)
    - kinds: drawn from the official 9 kinds (repetition allowed, mirroring
      the server which can issue several items of the same kind)
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    rng = rng or random.Random()
    kinds = list(kinds) if kinds is not None else sorted(VALID_KINDS)
    if not kinds:
        raise ValueError("kinds must not be empty")

    prefix = _new_run_prefix(rng)
    targets = [
        {"id": f"item_{prefix}_{i + 1:02d}", "kind": rng.choice(kinds)}
        for i in range(count)
    ]
    return {
        "schema_version": 1,
        "run_prefix": prefix,
        "count": count,
        "targets": targets,
    }


def task_to_json(task: dict) -> str:
    """Compact official-style JSON serialisation of a task dict."""
    return json.dumps(task, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    import sys
    sample = generate_task_from_kinds(
        [ "pingguo","kele", "chengzi", "zhijin", "shupian"])
    print(task_to_json(sample))
    sys.exit(0)
