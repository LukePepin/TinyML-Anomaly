import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import serial
from serial.tools import list_ports


def get_csv_header(axes: int) -> list[str]:
    if axes == 6:
        return [
            "NodeID",
            "Accel_X",
            "Accel_Y",
            "Accel_Z",
            "Gyro_X",
            "Gyro_Y",
            "Gyro_Z",
            "Timestamp",
        ]

    return [
        "NodeID",
        "Accel_X",
        "Accel_Y",
        "Accel_Z",
        "Gyro_X",
        "Gyro_Y",
        "Gyro_Z",
        "Mag_X",
        "Mag_Y",
        "Mag_Z",
        "Timestamp",
    ]


def list_available_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def print_available_ports() -> None:
    ports = list_available_ports()
    if not ports:
        print("No serial ports detected.")
        return

    print("Detected serial ports:")
    for port in ports:
        print(f"  - {port}")


def parse_imu_line(line: str, expected_columns: int) -> list[float] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != expected_columns:
        return None

    values: list[float] = []
    try:
        for part in parts:
            values.append(float(part))
    except ValueError:
        return None

    return values


def ensure_csv_header(csv_path: Path, header: list[str]) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def next_incrementing_csv(data_dir: Path, base_name: str = "training_data") -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)

    max_index = 0
    for path in data_dir.glob(f"{base_name}_*.csv"):
        stem = path.stem
        suffix = stem.removeprefix(f"{base_name}_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))

    return data_dir / f"{base_name}_{max_index + 1:04d}.csv"


def resolve_output_csv(out_arg: str | None) -> Path:
    if out_arg and out_arg.lower() != "auto":
        return Path(out_arg)

    script_dir = Path(__file__).resolve().parent
    return next_incrementing_csv(script_dir / "data")


def parse_port_candidates(primary_port: str, fallback_ports_arg: str) -> list[str]:
    candidates: list[str] = [primary_port]
    for raw in fallback_ports_arg.split(","):
        port = raw.strip()
        if not port:
            continue
        if port not in candidates:
            candidates.append(port)
    return candidates


def pick_next_port(current_port: str, candidates: list[str]) -> str:
    if not candidates:
        return current_port

    available = set(list_available_ports())
    available_candidates = [p for p in candidates if p in available]
    if not available_candidates:
        return current_port

    if current_port not in available_candidates:
        return available_candidates[0]

    idx = available_candidates.index(current_port)
    return available_candidates[(idx + 1) % len(available_candidates)]


def run_datalogger(
    port: str,
    port_candidates: list[str],
    baud: int,
    out_csv: Path,
    node_id: str,
    duration_s: float,
    startup_wait_s: float,
    axes: int,
    reconnect_wait_s: float,
    max_reconnects: int,
    no_data_timeout_s: float,
) -> None:
    current_port = port
    print(f"Opening {current_port} at {baud} baud...")

    if os.name != "nt" and current_port.upper().startswith("COM"):
        print(
            "Invalid port for Linux container: COM ports are Windows-only. "
            "Use /dev/ttyUSB0 or /dev/ttyACM0, or run this script on the Windows host."
        )
        print_available_ports()
        raise SystemExit(2)

    ensure_csv_header(out_csv, get_csv_header(axes))

    rows_written = 0
    invalid_rows = 0
    start = time.time()
    reconnect_count = 0
    announced_format = False
    announced_connected = False

    with out_csv.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        while True:
            if duration_s > 0 and (time.time() - start) >= duration_s:
                break

            try:
                with serial.Serial(port=current_port, baudrate=baud, timeout=1) as ser:
                    # Nano 33 BLE may reset on port-open.
                    time.sleep(startup_wait_s)
                    last_data_at = time.time()

                    if not announced_connected:
                        print("Connected. Logging IMU stream...")
                        announced_connected = True
                    else:
                        print(f"Reconnected on {current_port}. Continuing capture...")

                    if not announced_format:
                        if axes == 6:
                            print("Expected format: ax,ay,az,gx,gy,gz")
                        else:
                            print("Expected format: ax,ay,az,gx,gy,gz,mx,my,mz")
                        announced_format = True

                    while True:
                        if duration_s > 0 and (time.time() - start) >= duration_s:
                            break

                        raw = ser.readline().decode("utf-8", errors="replace").strip()
                        if not raw:
                            if no_data_timeout_s > 0 and (time.time() - last_data_at) >= no_data_timeout_s:
                                raise serial.SerialException(
                                    f"No serial data for {no_data_timeout_s:.1f}s on {current_port}"
                                )
                            continue
                            
                        if raw.startswith("ESTOP:"):
                            print(f"\n[!] E-STOP received from Arduino: {raw}")
                            Path(__file__).parent.joinpath("estop.flag").touch()
                            continue

                        parsed = parse_imu_line(raw, expected_columns=axes)
                        if parsed is None:
                            invalid_rows += 1
                            if invalid_rows <= 5:
                                print(f"Skipping non-IMU line: {raw}")
                            continue

                        last_data_at = time.time()
                        timestamp = datetime.now(timezone.utc).isoformat()
                        row = [node_id, *parsed, timestamp]
                        writer.writerow(row)
                        csv_file.flush()
                        rows_written += 1

                        if rows_written % 50 == 0:
                            elapsed = time.time() - start
                            print(f"Logged {rows_written} rows in {elapsed:.1f}s")

            except serial.SerialException as exc:
                reconnect_count += 1
                print(f"Serial open/read failed: {exc}")
                print_available_ports()

                next_port = pick_next_port(current_port, port_candidates)
                if next_port != current_port:
                    print(f"Switching port from {current_port} to {next_port}")
                    current_port = next_port

                if max_reconnects >= 0 and reconnect_count > max_reconnects:
                    print("Max reconnect attempts reached. Stopping capture.")
                    break

                print(
                    f"Waiting {reconnect_wait_s:.1f}s before reconnect attempt #{reconnect_count}..."
                )
                time.sleep(reconnect_wait_s)
                continue

    elapsed = time.time() - start
    rate = rows_written / elapsed if elapsed > 0 else 0.0
    print("Datalogger complete.")
    print(f"Rows written: {rows_written}")
    print(f"Invalid lines skipped: {invalid_rows}")
    print(f"Elapsed: {elapsed:.2f}s, approx rate: {rate:.2f} rows/s")
    print(f"Output: {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2 IMU datalogger for Arduino Nano 33 BLE"
    )
    parser.add_argument("--port", default="COM15", help="Serial port, e.g. COM15 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument(
        "--out",
        default="auto",
        help="Output CSV file path. Use 'auto' (default) for tinyml-anomaly/data/training_data_XXXX.csv",
    )
    parser.add_argument(
        "--node-id",
        default="niryo-wrist-imu",
        help="Node ID written to CSV NodeID column",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=900.0,
        help="Capture duration in seconds. Use 0 for indefinite logging.",
    )
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after opening serial port before reading",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List detected serial ports and exit",
    )
    parser.add_argument(
        "--axes",
        type=int,
        choices=[6, 9],
        default=9,
        help="Number of IMU values expected per serial row (6 for accel+gyro, 9 for accel+gyro+mag)",
    )
    parser.add_argument(
        "--reconnect-wait",
        type=float,
        default=2.0,
        help="Seconds to wait before reconnect after a serial disconnect",
    )
    parser.add_argument(
        "--max-reconnects",
        type=int,
        default=-1,
        help="Maximum reconnect attempts after disconnect (-1 for unlimited)",
    )
    parser.add_argument(
        "--fallback-ports",
        default="COM9,COM14",
        help="Comma-separated fallback ports to try on disconnect (default: COM9,COM14)",
    )
    parser.add_argument(
        "--no-data-timeout",
        type=float,
        default=5.0,
        help="Seconds with no serial data before forcing reconnect and trying fallback ports",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_ports:
        print_available_ports()
        return

    out_csv = resolve_output_csv(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    port_candidates = parse_port_candidates(args.port, args.fallback_ports)
    print(f"Port candidates: {port_candidates}")

    run_datalogger(
        port=args.port,
        port_candidates=port_candidates,
        baud=args.baud,
        out_csv=out_csv,
        node_id=args.node_id,
        duration_s=args.duration,
        startup_wait_s=args.startup_wait,
        axes=args.axes,
        reconnect_wait_s=args.reconnect_wait,
        max_reconnects=args.max_reconnects,
        no_data_timeout_s=args.no_data_timeout,
    )


if __name__ == "__main__":
    main()
