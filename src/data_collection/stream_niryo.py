#!/usr/bin/env python3
"""Run repeated randomized Niryo pose moves for data collection.

Sequence per cycle:
1) home
2) random from 1v1..1v5, sleep
3) random from 2v1,2v2,3v3,3v4,3v5, sleep
4) random from 3v1..3v5, sleep
5) random from 4v1..4v5, sleep
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time
from typing import Any

try:
    from pyniryo import NiryoRobot
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyniryo'. Install it with: pip install pyniryo"
    ) from exc

POSE_GROUP_1 = ["1v1", "1v2", "1v3", "1v4", "1v5"]
POSE_GROUP_2 = ["2v1", "2v2", "2v3", "2v4", "2v5"]
POSE_GROUP_3 = ["3v1", "3v2", "3v3", "3v4", "3v5"]
POSE_GROUP_4 = ["4v1", "4v2", "4v3", "4v4", "4v5"]
HOME_POSE_NAME = "Home"


def _call_first_existing(robot: Any, method_names: list[str], *args: Any) -> bool:
    for method_name in method_names:
        method = getattr(robot, method_name, None)
        if callable(method):
            method(*args)
            return True
    return False


def _run_with_collision_recovery(robot: Any, action_name: str, action: Any) -> None:
    try:
        action()
        return
    except Exception as exc:
        msg = str(exc)
        if "clear_collision_detected" not in msg and "collision" not in msg.lower():
            raise

        if not _call_first_existing(robot, ["clear_collision_detected"]):
            raise

        print(f"Collision detected during {action_name}; cleared and retrying once...")
        action()


def get_saved_pose_lookup(robot: Any) -> dict[str, str]:
    list_poses = getattr(robot, "get_saved_pose_list", None)
    if not callable(list_poses):
        return {}

    try:
        poses = list_poses()
    except Exception:
        return {}

    lookup: dict[str, str] = {}
    for pose in poses:
        if not isinstance(pose, str):
            continue
        clean = pose.strip()
        if not clean:
            continue
        lookup[clean.lower()] = clean
    return lookup


def resolve_pose_name(name: str, lookup: dict[str, str]) -> str | None:
    return lookup.get(name.strip().lower())


def filter_existing(group: list[str], lookup: dict[str, str]) -> list[str]:
    resolved: list[str] = []
    for name in group:
        found = resolve_pose_name(name, lookup)
        if found:
            resolved.append(found)
    return resolved


def move_to_pose_name(robot: Any, pose_name: str) -> None:
    """Move the robot to a saved pose name using the first supported API."""
    if pose_name.lower() == "home":
        if _call_first_existing(robot, ["move_to_home_pose"]):
            return

    moved = _call_first_existing(
        robot,
        [
            "move_to_saved_pose",
            "move_pose_saved",
            "move_to_pose_saved",
            "move_saved_pose",
            "move_pose_name",
        ],
        pose_name,
    )

    if moved:
        return

    # Fallback: get named pose object, then move to it.
    get_pose = getattr(robot, "get_pose_saved", None)
    move_pose = getattr(robot, "move_pose", None)
    move = getattr(robot, "move", None)
    if callable(get_pose) and (callable(move) or callable(move_pose)):
        try:
            pose = get_pose(pose_name)
        except Exception as exc:
            list_poses = getattr(robot, "get_saved_pose_list", None)
            available = []
            if callable(list_poses):
                try:
                    available = list_poses()
                except Exception:
                    available = []
            available_msg = (
                f" Available poses: {available}" if available else ""
            )
            raise RuntimeError(
                f"Saved pose '{pose_name}' was not found on the robot.{available_msg}"
            ) from exc

        if callable(move):
            _run_with_collision_recovery(
                robot,
                action_name=f"move('{pose_name}')",
                action=lambda: move(pose),
            )
            return

        if callable(move_pose) and hasattr(pose, "to_list") and callable(pose.to_list):
            _run_with_collision_recovery(
                robot,
                action_name=f"move_pose('{pose_name}')",
                action=lambda: move_pose(*pose.to_list()),
            )
            return

    raise RuntimeError(
        "Could not find a supported 'move to saved pose' API on this pyniryo version. "
        "Try upgrading pyniryo or adjust move_to_pose_name() for your SDK version."
    )


def run_sequence(
    robot: Any,
    repeats: int,
    sleep_s: float,
    dry_run: bool,
    group1: list[str],
    group2: list[str],
    group3: list[str],
    group4: list[str],
) -> None:
    for cycle in range(1, repeats + 1):
        if Path(__file__).parent.joinpath("estop.flag").exists():
            print("Software E-STOP flag detected! Closing out robot actions.")
            break

        print(f"[{cycle}/{repeats}] Moving to {HOME_POSE_NAME}")
        if not dry_run:
            move_to_pose_name(robot, HOME_POSE_NAME)

        p1 = random.choice(group1)
        print(f"[{cycle}/{repeats}] Moving to {p1}")
        if not dry_run:
            move_to_pose_name(robot, p1)
        time.sleep(sleep_s)

        p2 = random.choice(group2)
        print(f"[{cycle}/{repeats}] Moving to {p2}")
        if not dry_run:
            move_to_pose_name(robot, p2)
        time.sleep(sleep_s)

        p3 = random.choice(group3)
        print(f"[{cycle}/{repeats}] Moving to {p3}")
        if not dry_run:
            move_to_pose_name(robot, p3)
        time.sleep(sleep_s)

        p4 = random.choice(group4)
        print(f"[{cycle}/{repeats}] Moving to {p4}")
        if not dry_run:
            move_to_pose_name(robot, p4)
        time.sleep(sleep_s)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Niryo randomized move runner for CSV data collection"
    )
    parser.add_argument("--ip", required=True, help="Niryo robot IP address")
    parser.add_argument(
        "--repeats", type=int, default=100, help="How many cycles to run (default: 100)"
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0, help="Sleep seconds between random moves"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional RNG seed for reproducibility"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without sending commands to the robot",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration test (calibrate_auto) before movement",
    )
    parser.add_argument(
        "--list-poses",
        action="store_true",
        help="Print the robot saved-pose library and exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.dry_run:
        run_sequence(
            robot=None,
            repeats=args.repeats,
            sleep_s=args.sleep,
            dry_run=True,
            group1=POSE_GROUP_1,
            group2=POSE_GROUP_2,
            group3=POSE_GROUP_3,
            group4=POSE_GROUP_4,
        )
        return

    robot = NiryoRobot(args.ip)
    try:
        lookup = get_saved_pose_lookup(robot)

        if args.list_poses:
            if lookup:
                print("Saved poses:")
                for pose_name in sorted(lookup.values(), key=str.lower):
                    print(f"- {pose_name}")
            else:
                print("No saved poses returned by robot.")
            return

        if args.calibrate:
            print("Running calibration test before sequence...")
            _call_first_existing(robot, ["calibrate_auto", "calibrate"])

        g1 = filter_existing(POSE_GROUP_1, lookup)
        g2 = filter_existing(POSE_GROUP_2, lookup)
        g3 = filter_existing(POSE_GROUP_3, lookup)
        g4 = filter_existing(POSE_GROUP_4, lookup)

        if not g1 or not g2 or not g3 or not g4:
            raise RuntimeError(
                "One or more pose groups have no valid saved poses on the robot. "
                "Run with --list-poses and update group names in this script to match exactly."
            )

        # Keep remote control mode active during scripted movement.
        _call_first_existing(robot, ["set_learning_mode"], False)
        run_sequence(
            robot=robot,
            repeats=args.repeats,
            sleep_s=args.sleep,
            dry_run=False,
            group1=g1,
            group2=g2,
            group3=g3,
            group4=g4,
        )
    finally:
        _call_first_existing(robot, ["set_learning_mode"], True)
        _call_first_existing(robot, ["close_connection", "close", "disconnect"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
