#!/usr/bin/env python3
"""Run continuous baseline movement with time-ramping anomaly injections.

Behavior:
- The robot keeps moving through normal random saved poses.
- A running timer increases anomaly injection chance over time.
- When injected, a short random burst sequence is executed, then timer resets.

Default tuning targets an average anomaly interval near 30 seconds.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
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

# Conservative clamps for generated random offset moves.
MIN_Z_M = 0.06
MAX_Z_M = 0.55
MAX_ABS_XY_M = 0.55


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

    get_pose = getattr(robot, "get_pose_saved", None)
    move_pose = getattr(robot, "move_pose", None)
    move = getattr(robot, "move", None)
    if callable(get_pose) and (callable(move) or callable(move_pose)):
        pose = get_pose(pose_name)

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
        "Could not find a supported 'move to saved pose' API on this pyniryo version."
    )


def pick_pose(rng: random.Random, groups: list[list[str]]) -> str:
    return rng.choice(rng.choice(groups))


def log_line(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"ts_utc={ts} {msg}")


def rayleigh_injection_probability(
    since_last_injection_s: float,
    step_s: float,
    target_mean_interval_s: float,
) -> float:
    """Compute anomaly injection probability for this step.

    Hazard increases with elapsed time (Rayleigh process), giving a gradual ramp
    instead of a fixed split phase.
    """
    mean_target = max(1.0, target_mean_interval_s)
    sigma = mean_target / 1.2533141373  # sqrt(pi/2)
    hazard = max(0.0, since_last_injection_s) / (sigma * sigma)
    p = 1.0 - pow(2.718281828459045, -hazard * max(0.01, step_s))
    return max(0.0, min(0.99, p))


def _get_current_pose_xyz(robot: Any) -> tuple[float, float, float] | None:
    get_pose = getattr(robot, "get_pose", None)
    if not callable(get_pose):
        return None

    try:
        pose = get_pose()
    except Exception:
        return None

    if all(hasattr(pose, attr) for attr in ["x", "y", "z"]):
        return (
            float(getattr(pose, "x")),
            float(getattr(pose, "y")),
            float(getattr(pose, "z")),
        )

    if hasattr(pose, "to_list") and callable(pose.to_list):
        raw = pose.to_list()
        if isinstance(raw, list) and len(raw) >= 3:
            return float(raw[0]), float(raw[1]), float(raw[2])

    return None


def try_random_offset_move(robot: Any, rng: random.Random) -> str | None:
    components = _get_current_pose_xyz(robot)
    move_pose = getattr(robot, "move_pose", None)
    if components is None or not callable(move_pose):
        return None

    x, y, z = components
    target_x = max(-MAX_ABS_XY_M, min(MAX_ABS_XY_M, x + rng.uniform(-0.14, 0.14)))
    target_y = max(-MAX_ABS_XY_M, min(MAX_ABS_XY_M, y + rng.uniform(-0.14, 0.14)))
    target_z = max(MIN_Z_M, min(MAX_Z_M, z + rng.uniform(-0.09, 0.09)))

    _run_with_collision_recovery(
        robot,
        action_name="move_pose(random_offset)",
        action=lambda: move_pose(target_x, target_y, target_z, 0.0, 1.57, 0.0),
    )
    return f"RANDOM_OFFSET(x={target_x:.3f},y={target_y:.3f},z={target_z:.3f})"


def run_continuous_loop(
    robot: Any,
    rng: random.Random,
    group1: list[str],
    group2: list[str],
    group3: list[str],
    group4: list[str],
    baseline_sleep: float,
    injection_duration_seconds: float,
    injection_speedup: float,
    target_anomaly_interval: float,
    duration_seconds: float,
    max_steps: int,
    status_every: int,
    dry_run: bool,
) -> None:
    started_at = time.time()
    last_injection_at = started_at
    step = 0
    cycle = 0
    injections = 0

    log_line(
        "STATE mode=NORMAL phase=pick_place_cycle "
        f"target_anomaly_interval_s={target_anomaly_interval:.1f} "
        f"baseline_sleep_s={baseline_sleep:.2f} "
        f"injection_duration_s={injection_duration_seconds:.2f} "
        f"injection_speedup={injection_speedup:.2f}x"
    )

    while True:
        if duration_seconds > 0 and (time.time() - started_at) >= duration_seconds:
            break
        if Path(__file__).parent.joinpath("estop.flag").exists():
            raise RuntimeError("Software E-STOP flag detected")

        cycle += 1
        p1 = rng.choice(group1)
        p2 = rng.choice(group2)
        p3 = rng.choice(group3)
        p4 = rng.choice(group4)
        sequence = [
            ("HOME", HOME_POSE_NAME, False),
            ("G1", p1, True),
            ("G2", p2, True),
            ("G3", p3, True),
            ("G4", p4, True),
        ]

        for stage, target, should_sleep in sequence:
            if max_steps > 0 and step >= max_steps:
                break
            if duration_seconds > 0 and (time.time() - started_at) >= duration_seconds:
                break
            if Path(__file__).parent.joinpath("estop.flag").exists():
                raise RuntimeError("Software E-STOP flag detected")

            step += 1
            now = time.time()
            since_last_injection = now - last_injection_at
            p_inject = rayleigh_injection_probability(
                since_last_injection_s=since_last_injection,
                step_s=baseline_sleep,
                target_mean_interval_s=target_anomaly_interval,
            )

            if not dry_run:
                move_to_pose_name(robot, target)

            log_line(
                "BASELINE_STEP mode=NORMAL "
                f"cycle={cycle} step={step} stage={stage} target={target} "
                f"since_last_injection_s={since_last_injection:.1f} inject_p={p_inject:.3f}"
            )

            inject = rng.random() < p_inject
            if inject:
                injections += 1
                burst_sleep = max(0.05, baseline_sleep / max(1.0, injection_speedup))
                burst_start = time.time()
                burst_step = 0
                log_line(
                    "INJECTION STARTED mode=ANOMALY "
                    f"cycle={cycle} step={step} stage={stage} injection_count={injections} "
                    f"since_last_injection_s={since_last_injection:.1f} "
                    f"inject_p={p_inject:.3f} duration_s={injection_duration_seconds:.2f} "
                    f"speedup={injection_speedup:.2f}x"
                )

                # Fast anomaly burst can use either saved poses or random offsets.
                while (time.time() - burst_start) < max(0.1, injection_duration_seconds):
                    burst_step += 1
                    burst_target = pick_pose(rng, [group1, group2, group3, group4])
                    burst_label = burst_target

                    if not dry_run:
                        do_random_offset = rng.random() < 0.5
                        if do_random_offset:
                            moved_label = try_random_offset_move(robot, rng)
                            if moved_label is not None:
                                burst_label = moved_label
                            else:
                                move_to_pose_name(robot, burst_target)
                        else:
                            move_to_pose_name(robot, burst_target)

                    log_line(
                        "INJECTION BURST mode=ANOMALY "
                        f"cycle={cycle} step={step} burst_step={burst_step} "
                        f"target={burst_label} dwell_s={burst_sleep:.2f}"
                    )
                    time.sleep(burst_sleep)

                burst_elapsed = time.time() - burst_start
                log_line(
                    "INJECTION STOPPED mode=NORMAL "
                    f"cycle={cycle} step={step} injection_count={injections} "
                    f"burst_elapsed_s={burst_elapsed:.2f}"
                )
                last_injection_at = time.time()
                continue

            if status_every > 0 and (step % status_every == 0):
                log_line(
                    "STATE mode=NORMAL "
                    f"cycle={cycle} step={step} stage={stage} "
                    f"since_last_injection_s={since_last_injection:.1f} "
                    f"inject_p={p_inject:.3f} target={target}"
                )

            if should_sleep:
                time.sleep(max(0.05, baseline_sleep))

        if max_steps > 0 and step >= max_steps:
            break

    total = time.time() - started_at
    log_line(
        f"STATE mode=STOP total_s={total:.1f} cycles={cycle} steps={step} injections={injections}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuous baseline with gradually increasing anomaly injection chance"
    )
    parser.add_argument("--ip", required=True, help="Niryo robot IP address")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument(
        "--baseline-sleep",
        type=float,
        default=1.0,
        help="Dwell time between baseline moves",
    )
    parser.add_argument(
        "--injection-duration-seconds",
        type=float,
        default=5.0,
        help="How long each injected anomaly burst runs",
    )
    parser.add_argument(
        "--injection-speedup",
        type=float,
        default=4.0,
        help="Burst speed multiplier relative to baseline dwell",
    )
    parser.add_argument(
        "--target-anomaly-interval",
        type=float,
        default=30.0,
        help="Target average seconds between anomaly injections",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=300.0,
        help="Stop after this many seconds (default: 300 for 5-minute showcase)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum move steps (0 means infinite)",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=5,
        help="Print NORMAL state every N steps",
    )
    parser.add_argument(
        "--clear-estop",
        action="store_true",
        help="Delete backend/estop.flag before starting",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration before movement",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print behavior without sending robot commands",
    )
    parser.add_argument(
        "--list-poses",
        action="store_true",
        help="Print robot saved poses and exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

        run_continuous_loop(
            robot=robot,
            rng=rng,
            group1=g1,
            group2=g2,
            group3=g3,
            group4=g4,
            baseline_sleep=args.baseline_sleep,
            injection_duration_seconds=args.injection_duration_seconds,
            injection_speedup=args.injection_speedup,
            target_anomaly_interval=args.target_anomaly_interval,
            duration_seconds=args.duration_seconds,
            max_steps=args.max_steps,
            status_every=args.status_every,
            dry_run=args.dry_run,
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
