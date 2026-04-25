#!/usr/bin/env python3
"""Motion-only adversarial controller with strange (non-predefined) locations.

Purpose:
- Command normal and improper robot actions.
- Add strange Cartesian offset moves that are not saved predefined poses.
- Do NOT log CSV; use existing Arduino + datalogger pipeline.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

try:
    from pyniryo import NiryoRobot
except ImportError as exc:
    raise SystemExit("Missing dependency 'pyniryo'. Install it with: pip install pyniryo") from exc


POSE_GROUP_1 = ["1v1", "1v2", "1v3", "1v4", "1v5"]
POSE_GROUP_2 = ["2v1", "2v2", "2v3", "2v4", "2v5"]
POSE_GROUP_3 = ["3v1", "3v2", "3v3", "3v4", "3v5"]
POSE_GROUP_4 = ["4v1", "4v2", "4v3", "4v4", "4v5"]
HOME_POSE_NAME = "Home"

# Conservative workspace clamps for generated strange Cartesian targets (meters/radians).
MIN_Z_M = 0.06
MAX_Z_M = 0.55
MAX_ABS_XY_M = 0.55
MAX_ABS_RPY_RAD = math.pi


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

    get_pose_saved = getattr(robot, "get_pose_saved", None)
    move_pose = getattr(robot, "move_pose", None)
    move = getattr(robot, "move", None)
    if callable(get_pose_saved) and (callable(move) or callable(move_pose)):
        pose = get_pose_saved(pose_name)

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
        "Could not find a supported saved-pose movement API on this pyniryo version."
    )


def _get_current_pose_components(robot: Any) -> tuple[float, float, float, float, float, float] | None:
    get_pose = getattr(robot, "get_pose", None)
    if not callable(get_pose):
        return None

    try:
        pose = get_pose()
    except Exception:
        return None

    attrs = ["x", "y", "z", "roll", "pitch", "yaw"]
    if all(hasattr(pose, name) for name in attrs):
        return (
            float(getattr(pose, "x")),
            float(getattr(pose, "y")),
            float(getattr(pose, "z")),
            float(getattr(pose, "roll")),
            float(getattr(pose, "pitch")),
            float(getattr(pose, "yaw")),
        )

    if hasattr(pose, "to_list") and callable(pose.to_list):
        raw = pose.to_list()
        if isinstance(raw, list) and len(raw) >= 6:
            return tuple(float(v) for v in raw[:6])

    return None


def _clamp_pose_components(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float, float, float, float]:
    x = max(-MAX_ABS_XY_M, min(MAX_ABS_XY_M, x))
    y = max(-MAX_ABS_XY_M, min(MAX_ABS_XY_M, y))
    z = max(MIN_Z_M, min(MAX_Z_M, z))
    roll = max(-MAX_ABS_RPY_RAD, min(MAX_ABS_RPY_RAD, roll))
    pitch = max(-MAX_ABS_RPY_RAD, min(MAX_ABS_RPY_RAD, pitch))
    yaw = max(-MAX_ABS_RPY_RAD, min(MAX_ABS_RPY_RAD, yaw))
    return x, y, z, roll, pitch, yaw


def move_to_strange_location(
    robot: Any,
    rng: random.Random,
    xy_offset_m: float,
    z_offset_m: float,
    rpy_offset_deg: float,
) -> tuple[bool, str]:
    current = _get_current_pose_components(robot)
    if current is None:
        return False, "get_pose unavailable"

    x, y, z, roll, pitch, yaw = current

    tx = x + rng.uniform(-xy_offset_m, xy_offset_m)
    ty = y + rng.uniform(-xy_offset_m, xy_offset_m)
    tz = z + rng.uniform(-z_offset_m, z_offset_m)

    offset_rad = math.radians(rpy_offset_deg)
    troll = roll + rng.uniform(-offset_rad, offset_rad)
    tpitch = pitch + rng.uniform(-offset_rad, offset_rad)
    tyaw = yaw + rng.uniform(-offset_rad, offset_rad)

    tx, ty, tz, troll, tpitch, tyaw = _clamp_pose_components(tx, ty, tz, troll, tpitch, tyaw)

    moved = _call_first_existing(
        robot,
        ["move_pose"],
        tx,
        ty,
        tz,
        troll,
        tpitch,
        tyaw,
    )
    if not moved:
        return False, "move_pose API unavailable"

    return True, f"x={tx:.3f},y={ty:.3f},z={tz:.3f},rpy=({troll:.2f},{tpitch:.2f},{tyaw:.2f})"


def pick_pose(rng: random.Random, groups: list[list[str]]) -> str:
    return rng.choice(rng.choice(groups))


def run_loop(
    robot: Any,
    dry_run: bool,
    rng: random.Random,
    groups: list[list[str]],
    base_sleep_s: float,
    max_cycles: int,
    improper_rate: float,
    strange_rate: float,
    strange_xy_mm: float,
    strange_z_mm: float,
    strange_rpy_deg: float,
) -> None:
    prev_label = HOME_POSE_NAME
    prev_saved_pose = HOME_POSE_NAME
    cycle = 0

    xy_offset_m = max(0.0, strange_xy_mm) / 1000.0
    z_offset_m = max(0.0, strange_z_mm) / 1000.0

    while True:
        if max_cycles > 0 and cycle >= max_cycles:
            break
        if Path(__file__).parent.joinpath("estop.flag").exists():
            print("Software E-STOP flag detected. Exiting movement loop.")
            break

        cycle += 1
        target_pose = pick_pose(rng, groups)
        event = "normal"
        dwell = max(0.05, rng.normalvariate(base_sleep_s, max(0.05, base_sleep_s * 0.2)))
        stop_hold_s = 0.0

        if rng.random() < improper_rate:
            if rng.random() < strange_rate:
                event = "strange_location"
            else:
                event = rng.choice(["stoppage", "freeze", "burst", "jitter", "home_interrupt"])

            if event == "stoppage":
                stop_hold_s = rng.uniform(2.5, 10.0)

            elif event == "freeze":
                target_pose = prev_saved_pose
                dwell = rng.uniform(1.0, 2.5)

            elif event == "burst":
                dwell = rng.uniform(0.01, 0.16)

            elif event == "jitter":
                dwell = max(0.01, dwell + rng.uniform(-1.0, 1.875))

            elif event == "home_interrupt":
                target_pose = HOME_POSE_NAME
                dwell = rng.uniform(0.05, 0.875)

        moved_label = target_pose

        if event == "strange_location":
            moved_label = "STRANGE"
            if dry_run:
                print(
                    f"cycle={cycle} from={prev_label} to=STRANGE "
                    f"event={event} dwell={dwell:.2f}s stop={stop_hold_s:.2f}s"
                )
            else:
                moved, detail = move_to_strange_location(
                    robot=robot,
                    rng=rng,
                    xy_offset_m=xy_offset_m,
                    z_offset_m=z_offset_m,
                    rpy_offset_deg=strange_rpy_deg,
                )
                if moved:
                    print(
                        f"cycle={cycle} from={prev_label} to=STRANGE {detail} "
                        f"event={event} dwell={dwell:.2f}s stop={stop_hold_s:.2f}s"
                    )
                else:
                    # Fallback to a saved pose move when Cartesian API is unavailable.
                    fallback_pose = pick_pose(rng, groups)
                    move_to_pose_name(robot, fallback_pose)
                    moved_label = fallback_pose
                    prev_saved_pose = fallback_pose
                    print(
                        f"cycle={cycle} strange-move-fallback reason='{detail}' "
                        f"from={prev_label} to={fallback_pose} event={event} "
                        f"dwell={dwell:.2f}s stop={stop_hold_s:.2f}s"
                    )
        else:
            if not dry_run:
                move_to_pose_name(robot, target_pose)
            prev_saved_pose = target_pose
            print(
                f"cycle={cycle} from={prev_label} to={target_pose} "
                f"event={event} dwell={dwell:.2f}s stop={stop_hold_s:.2f}s"
            )

        if stop_hold_s > 0:
            time.sleep(stop_hold_s)
        time.sleep(dwell)
        prev_label = moved_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Motion-only adversarial runner with strange non-predefined locations"
    )
    parser.add_argument("--ip", required=True, help="Niryo robot IP address")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--sleep", type=float, default=1.0, help="Base dwell time between moves")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means infinite")
    parser.add_argument(
        "--improper-rate",
        type=float,
        default=1.0,
        help="Probability per cycle of improper behavior (default 1.0)",
    )
    parser.add_argument(
        "--strange-rate",
        type=float,
        default=0.50,
        help="Probability that an improper cycle uses a strange non-predefined location",
    )
    parser.add_argument(
        "--strange-xy-mm",
        type=float,
        default=120.0,
        help="Max +/- XY offset in mm for strange location moves",
    )
    parser.add_argument(
        "--strange-z-mm",
        type=float,
        default=80.0,
        help="Max +/- Z offset in mm for strange location moves",
    )
    parser.add_argument(
        "--strange-rpy-deg",
        type=float,
        default=35.0,
        help="Max +/- roll/pitch/yaw offset in degrees for strange location moves",
    )
    parser.add_argument(
        "--clear-estop",
        action="store_true",
        help="Delete backend/estop.flag before starting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned behavior without sending robot commands",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration before movement",
    )
    parser.add_argument(
        "--list-poses",
        action="store_true",
        help="Print robot saved poses and exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not (0.0 <= args.improper_rate <= 1.0):
        raise SystemExit("--improper-rate must be in [0, 1]")
    if not (0.0 <= args.strange_rate <= 1.0):
        raise SystemExit("--strange-rate must be in [0, 1]")

    rng = random.Random(args.seed)
    robot = NiryoRobot(args.ip)

    try:
        if args.clear_estop:
            estop_path = Path(__file__).parent.joinpath("estop.flag")
            if estop_path.exists():
                estop_path.unlink()
                print("Cleared software E-STOP flag.")

        lookup = get_saved_pose_lookup(robot)

        if args.list_poses:
            if lookup:
                print("Saved poses:")
                for pose_name in sorted(lookup.values(), key=str.lower):
                    print(f"- {pose_name}")
            else:
                print("No saved poses returned by robot.")
            return

        g1 = filter_existing(POSE_GROUP_1, lookup)
        g2 = filter_existing(POSE_GROUP_2, lookup)
        g3 = filter_existing(POSE_GROUP_3, lookup)
        g4 = filter_existing(POSE_GROUP_4, lookup)
        if not g1 or not g2 or not g3 or not g4:
            raise RuntimeError(
                "One or more pose groups have no valid saved poses on the robot. "
                "Run with --list-poses and update pose names in this script."
            )

        if args.calibrate:
            _call_first_existing(robot, ["calibrate_auto", "calibrate"])

        if not args.dry_run:
            _call_first_existing(robot, ["set_learning_mode"], False)

        print(
            "Starting adversarial loop with strange locations "
            f"(dry_run={args.dry_run}, max_cycles={args.max_cycles or 'infinite'}, "
            f"improper_rate={args.improper_rate}, strange_rate={args.strange_rate})"
        )

        run_loop(
            robot=robot,
            dry_run=args.dry_run,
            rng=rng,
            groups=[g1, g2, g3, g4],
            base_sleep_s=args.sleep,
            max_cycles=args.max_cycles,
            improper_rate=args.improper_rate,
            strange_rate=args.strange_rate,
            strange_xy_mm=args.strange_xy_mm,
            strange_z_mm=args.strange_z_mm,
            strange_rpy_deg=args.strange_rpy_deg,
        )
    finally:
        if not args.dry_run:
            _call_first_existing(robot, ["set_learning_mode"], True)
        _call_first_existing(robot, ["close_connection", "close", "disconnect"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
