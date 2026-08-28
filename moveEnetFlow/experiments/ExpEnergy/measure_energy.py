#!/usr/bin/env python3
"""
Host-side energy meter for MoveEnet_OFK experiments.

Measures Intel RAPL package energy and NVIDIA GPU cumulative energy around one
host command (including `docker exec ...`). The target application is not
instrumented or modified.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pynvml

DEFAULT_RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
DEFAULT_RAPL_MAX = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
DEFAULT_PSYS = Path("/sys/class/powercap/intel-rapl:1/energy_uj")
DEFAULT_PSYS_MAX = Path("/sys/class/powercap/intel-rapl:1/max_energy_range_uj")

CSV_FIELDS = [
    "timestamp_start", "timestamp_end", "label", "dataset", "sample_id",
    "model", "order_id", "order_position", "net_period_s", "flow_period_s",
    "gpu_index", "gpu_name", "gpu_uuid", "nvidia_driver", "gpu_power_limit_w",
    "gpu_temp_start_c", "gpu_temp_end_c", "gpu_pstate_start", "gpu_pstate_end",
    "wall_seconds", "cpu_package_energy_j", "gpu_energy_j",
    "measured_compute_energy_j", "cpu_avg_power_w", "gpu_avg_power_w",
    "measured_compute_avg_power_w", "psys_energy_j", "psys_avg_power_w",
    "cpu_counter_start_uj", "cpu_counter_end_uj", "gpu_counter_start_mj",
    "gpu_counter_end_mj", "psys_counter_start_uj", "psys_counter_end_uj",
    "exit_code", "status", "command", "log_path",
]


def iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def read_int(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except PermissionError as exc:
        raise RuntimeError(
            f"Permission denied reading {path}. Grant read access to the RAPL "
            "counters before running the benchmark."
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"Energy counter not found: {path}") from exc


def counter_delta(start: int, end: int, maximum: int | None) -> int:
    delta = end - start
    if delta >= 0:
        return delta
    if maximum is None:
        raise RuntimeError(
            f"Energy counter decreased ({start} -> {end}) and no wrap range is available."
        )
    return (maximum - start) + end


def nvml_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def gpu_pstate(handle: Any) -> str:
    try:
        return f"P{pynvml.nvmlDeviceGetPowerState(handle)}"
    except pynvml.NVMLError:
        return ""


def gpu_temp(handle: Any) -> str:
    try:
        return str(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except pynvml.NVMLError:
        return ""


def gpu_power_limit_w(handle: Any) -> str:
    try:
        return f"{pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0:.6f}"
    except pynvml.NVMLError:
        return ""


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CPU-package and NVIDIA-GPU energy around one command."
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--order-id", default="")
    parser.add_argument("--order-position", default="")
    parser.add_argument("--net-period", default="")
    parser.add_argument("--flow-period", default="")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--rapl", type=Path, default=DEFAULT_RAPL)
    parser.add_argument("--rapl-max", type=Path, default=DEFAULT_RAPL_MAX)
    parser.add_argument("--record-psys", action="store_true")
    parser.add_argument("--psys", type=Path, default=DEFAULT_PSYS)
    parser.add_argument("--psys-max", type=Path, default=DEFAULT_PSYS_MAX)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("No command supplied. Use: measure_energy.py [options] -- command args...")
    return args


def main() -> int:
    args = parse_args()

    cpu_max = read_int(args.rapl_max)
    _ = read_int(args.rapl)
    psys_max = None
    if args.record_psys:
        psys_max = read_int(args.psys_max)
        _ = read_int(args.psys)

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
        gpu_name = nvml_str(pynvml.nvmlDeviceGetName(handle))
        gpu_uuid = nvml_str(pynvml.nvmlDeviceGetUUID(handle))
        driver = nvml_str(pynvml.nvmlSystemGetDriverVersion())

        try:
            _ = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        except pynvml.NVMLError as exc:
            raise RuntimeError(
                "NVML total GPU energy counter is not available on this GPU/driver."
            ) from exc

        start_iso = iso_now()
        temp_start = gpu_temp(handle)
        pstate_start = gpu_pstate(handle)
        cpu0 = read_int(args.rapl)
        psys0 = read_int(args.psys) if args.record_psys else None
        gpu0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        t0 = time.perf_counter()

        log_handle = None
        rc = 255
        try:
            if args.log is not None:
                args.log.parent.mkdir(parents=True, exist_ok=True)
                log_handle = args.log.open("w")
                completed = subprocess.run(
                    args.command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            else:
                completed = subprocess.run(args.command, check=False)
            rc = int(completed.returncode)
        finally:
            t1 = time.perf_counter()
            gpu1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            psys1 = read_int(args.psys) if args.record_psys else None
            cpu1 = read_int(args.rapl)
            if log_handle is not None:
                log_handle.flush()
                log_handle.close()

        end_iso = iso_now()
        temp_end = gpu_temp(handle)
        pstate_end = gpu_pstate(handle)
        wall_s = t1 - t0

        cpu_delta_uj = counter_delta(cpu0, cpu1, cpu_max)
        cpu_j = cpu_delta_uj / 1e6
        gpu_delta_mj = gpu1 - gpu0
        if gpu_delta_mj < 0:
            raise RuntimeError(
                f"GPU cumulative energy counter decreased ({gpu0} -> {gpu1}). "
                "The NVIDIA driver may have reset during the run."
            )
        gpu_j = gpu_delta_mj / 1e3
        compute_j = cpu_j + gpu_j

        psys_j = ""
        psys_avg_w = ""
        if args.record_psys and psys0 is not None and psys1 is not None:
            psys_delta_uj = counter_delta(psys0, psys1, psys_max)
            psys_j_value = psys_delta_uj / 1e6
            psys_j = f"{psys_j_value:.9f}"
            psys_avg_w = f"{psys_j_value / wall_s:.9f}" if wall_s > 0 else ""

        row = {
            "timestamp_start": start_iso,
            "timestamp_end": end_iso,
            "label": args.label,
            "dataset": args.dataset,
            "sample_id": args.sample_id,
            "model": args.model,
            "order_id": args.order_id,
            "order_position": args.order_position,
            "net_period_s": args.net_period,
            "flow_period_s": args.flow_period,
            "gpu_index": args.gpu_index,
            "gpu_name": gpu_name,
            "gpu_uuid": gpu_uuid,
            "nvidia_driver": driver,
            "gpu_power_limit_w": gpu_power_limit_w(handle),
            "gpu_temp_start_c": temp_start,
            "gpu_temp_end_c": temp_end,
            "gpu_pstate_start": pstate_start,
            "gpu_pstate_end": pstate_end,
            "wall_seconds": f"{wall_s:.9f}",
            "cpu_package_energy_j": f"{cpu_j:.9f}",
            "gpu_energy_j": f"{gpu_j:.9f}",
            "measured_compute_energy_j": f"{compute_j:.9f}",
            "cpu_avg_power_w": f"{cpu_j / wall_s:.9f}" if wall_s > 0 else "",
            "gpu_avg_power_w": f"{gpu_j / wall_s:.9f}" if wall_s > 0 else "",
            "measured_compute_avg_power_w": f"{compute_j / wall_s:.9f}" if wall_s > 0 else "",
            "psys_energy_j": psys_j,
            "psys_avg_power_w": psys_avg_w,
            "cpu_counter_start_uj": cpu0,
            "cpu_counter_end_uj": cpu1,
            "gpu_counter_start_mj": gpu0,
            "gpu_counter_end_mj": gpu1,
            "psys_counter_start_uj": psys0 if psys0 is not None else "",
            "psys_counter_end_uj": psys1 if psys1 is not None else "",
            "exit_code": rc,
            "status": "OK" if rc == 0 else "FAILED",
            "command": shlex.join(args.command),
            "log_path": str(args.log) if args.log is not None else "",
        }

        append_csv(args.csv, row)

        print(f"Result                 : {row['status']}")
        print(f"Exit code              : {rc}")
        print(f"Wall time              : {wall_s:.3f} s")
        print(f"CPU package energy     : {cpu_j:.3f} J")
        print(f"GPU energy             : {gpu_j:.3f} J")
        print(f"Measured compute energy: {compute_j:.3f} J")
        print(f"CPU average power      : {cpu_j / wall_s:.3f} W")
        print(f"GPU average power      : {gpu_j / wall_s:.3f} W")
        print(f"Combined average power : {compute_j / wall_s:.3f} W")
        if args.record_psys:
            print(f"RAPL psys energy       : {psys_j} J (diagnostic only)")
        print(f"CSV                    : {args.csv}")
        if args.log is not None:
            print(f"Command log            : {args.log}")
        return rc
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
