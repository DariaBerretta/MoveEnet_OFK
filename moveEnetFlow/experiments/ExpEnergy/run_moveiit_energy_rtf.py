#!/usr/bin/env python3
"""Move-IIT-HPE final energy + RTF experiment (HOST runner, corrected protocol).

Exactly three operating points, each evaluated on the SAME 12 recordings:
    movenet_200hz      MoveEnet-only:   network 200 Hz, output 200 Hz
    eventpointpose_200hz EventPointPose: network 200 Hz, output 200 Hz
    moveenetofk_5hz    MoveEnetOFK:    network   5 Hz, flow 200 Hz, output 200 Hz

Reuses the existing ExpEnergy/measure_energy.py without modifying its RAPL/NVML
measurement definitions. The meter's wall_seconds is the numerator of RTF,
whose denominator is the precise first-to-last ATIS packet-timestamp span.

The 5 repeats and 12 recordings balance all three methods over run positions.
A completed run is recorded only after BOTH the energy meter and the specific
algorithm's end-of-recording log have been validated.

Run this on the HOST, with Docker/YARP already running, and leave the GPU idle.
This file replaces run_moveiit_energy_rtf.py; no legacy experiment is included.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MEASURE_SCRIPT = HERE / "measure_energy.py"
DEFAULT_SAMPLES = HERE / "moveiit_energy_samples.csv"
DATA_ROOT = "/data/move-iit-it_testing"
VENV = "/opt/venv"
BASE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
MOVENET_BIN = "/workspace/moveEnetFlow/build/moveEnetOFK_offline"
MOVENET_CKPT = "/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth"
EPP_BIN = "/workspace/moveEnetFlow/build/eventPointPose_offline"
EPP_CKPT = "/workspace/model_mounts/eventpointpose/PointNet/models/model.pth"
EPP_SERVER = "/workspace/moveEnetFlow/utils/eventPointPose_yarp_server.py"
EPP_ADAPTER = "/workspace/moveEnetFlow/utils/epp_moveiit_adapter.py"
PROTOCOL = "moveiit_energy_rtf_200_200_5_v1"
DURATION_SOURCE = "YARP_data_log_first_last_AE_packet_timestamp"
OUTPUT_PERIOD = "0.005"
FLOW_PERIOD = "0.005"

# An ordered three-element tuple rather than a collection of tunable profiles.
# (id, network period, requires optical flow)
CASES = (
    ("movenet_200hz", "0.005", False),
    ("eventpointpose_200hz", "0.005", False),
    ("moveenetofk_5hz", "0.2", True),
)

# Executed in the container; inspect only packet headers, not event payloads.
# Preserve the exact timestamp decimal strings to avoid catastrophic precision
# loss when subtracting two ~1.7e9 Unix timestamps in floating point.
PACKET_SPAN_SCRIPT = r'''
import json, os, re, sys
p = sys.argv[1]
header = re.compile(rb'(?:^|\n)([0-9]+)[ \t]+([0-9]{10}\.[0-9]+)[ \t]+AE[ \t]+([0-9]+)[ \t]+')
with open(p, 'rb') as f:
    first = header.search(f.read(8192))
    if first is None or first.start() != 0:
        raise RuntimeError('Invalid first YARP AE packet header in ' + p)
    first_id = int(first.group(1))
    first_ts = first.group(2).decode('ascii')
    size = os.fstat(f.fileno()).st_size
    window = 262144
    while True:
        f.seek(max(0, size-window))
        candidates = list(header.finditer(f.read()))
        if candidates and int(candidates[-1].group(1)) > first_id:
            last = candidates[-1]
            break
        if window >= size:
            raise RuntimeError('No valid last packet header in ' + p)
        window = min(window*2, size)
    last_id = int(last.group(1))
    last_ts = last.group(2).decode('ascii')
print(json.dumps(dict(first_packet_id=first_id, last_packet_id=last_id,
                      first_timestamp=first_ts, last_timestamp=last_ts)))
'''


def run_capture(cmd, check=True):
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if check and result.returncode != 0:
        raise RuntimeError("Command failed ({}): {}\nSTDERR:\n{}".format(
            result.returncode, shlex.join(map(str, cmd)), result.stderr[-3000:]))
    return result.stdout.strip()


def docker(container, *cmd):
    return run_capture(["docker", "exec", container, *map(str, cmd)])


def sha256_local(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_inside(container, path):
    return docker(container, "sha256sum", path).split()[0]


def load_samples(path):
    if not path.is_file():
        raise RuntimeError("Missing 12-sequence manifest: {}. Use the SAME moveiit_energy_samples.csv as the previous energy experiment.".format(path))
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r.get("sample_id") for r in rows}) != len(rows):
        raise RuntimeError("Sample manifest is empty or contains duplicate sample IDs")
    for r in rows:
        if (not r.get("sample_id") or r.get("subject") not in ("luna", "simon")
                or not re.fullmatch(r"[a-z]+(?:-[a-z]+)?_[fs]\d+", r.get("sequence", ""))):
            raise RuntimeError("Malformed Move-IIT sample entry: {}".format(r))
        subject, seq = r["subject"], r["sequence"]
        r["event_file"] = "{}/{}/{}/atis-s/data.log".format(DATA_ROOT, subject, seq)
        gt_subject = "P1_luna" if subject == "luna" else "P11_simon"
        r["gt_file"] = "{}/GT/{}/{}_joint_projections.csv".format(DATA_ROOT, gt_subject, seq)
    return rows


def input_span(container, sample):
    parsed = json.loads(docker(container, VENV + "/bin/python", "-c",
                               PACKET_SPAN_SCRIPT, sample["event_file"]))
    first = Decimal(parsed["first_timestamp"])
    last = Decimal(parsed["last_timestamp"])
    duration = last - first
    if not (first.is_finite() and last.is_finite() and duration.is_finite() and duration > 0):
        raise RuntimeError("Invalid event duration for {}".format(sample["sample_id"]))
    return dict(duration_s=str(duration), input_start_s=str(first),
                input_end_s=str(last), duration_source=DURATION_SOURCE,
                first_packet_id=parsed["first_packet_id"],
                last_packet_id=parsed["last_packet_id"])


def build_command(container, device, event_file, case):
    name, period, has_flow = case
    env = ["docker", "exec", container, "env", "VIRTUAL_ENV=" + VENV,
           "PATH=" + VENV + "/bin:" + BASE_PATH]
    if name == "eventpointpose_200hz":
        return env + [
            EPP_BIN, "--data_file", event_file, "--moveiit",
            "--moveiit_geometry", "stretch", "--camera", "0",
            "--w", "346", "--h", "260", "--net_period", period,
            "--output_period", OUTPUT_PERIOD, "--input_preprocessing", "already_filtered",
            "--fifo_size", "7500", "--num_points", "2048", "--seed", "1",
            "--model_path", EPP_CKPT, "--EventPointPose_script", EPP_SERVER,
            "--device", device, "--startup_timeout", "30",
            "--response_timeout", "0", "--no_csv",
        ]
    if name not in ("movenet_200hz", "moveenetofk_5hz"):
        raise ValueError("Unknown operating point: " + name)
    cmd = env + [
        MOVENET_BIN, "--data_file", event_file, "--moveiit",
        "--net_period", period, "--flow_period", FLOW_PERIOD,
        "--output_period", OUTPUT_PERIOD, "--roi", "35",
        "--pu", "0.77", "--muD", "0.06", "--muV", "0.97",
        "--use_lc", "true", "--checkpoint_path", MOVENET_CKPT,
        "--device", device, "--no_csv", "--no_video",
    ]
    if not has_flow:
        cmd.append("--moveenet_only")
    return cmd


def is_complete_log(model, content):
    if model == "eventpointpose_200hz":
        valid = re.search(r"Valid PointNet inferences:\s*(\d+)", content)
        requests = re.search(r"PointNet requests:\s*(\d+)", content)
        errors = re.search(r"Server ERROR responses:\s*(\d+)", content)
        return (valid is not None and int(valid.group(1)) > 0
                and requests is not None and int(requests.group(1)) >= int(valid.group(1))
                and errors is not None and int(errors.group(1)) == 0
                and "Packet groups processed:" in content)
    packets = re.search(r"Finished processing event file\. Packets:\s*(\d+)", content)
    return packets is not None and int(packets.group(1)) > 0


def read_csv_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def append_csv(path, fields, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(data)
        f.flush()


def collect_completed(session):
    energy_path = session / "energy_measurements.csv"
    manifest = read_csv_rows(session / "run_manifest.csv")
    manifest_ok = {(r["sample_id"], r["model"], r["order_id"])
                   for r in manifest if r.get("status") == "OK"}
    meter_ok = set()
    for r in read_csv_rows(energy_path):
        key = (r["sample_id"], r["model"], r["order_id"])
        if r["status"] == "OK" and r["exit_code"] in ("0", "0.0"):
            if key in meter_ok:
                raise RuntimeError("Duplicate successful energy measurement: {}".format(key))
            meter_ok.add(key)
            log = Path(r["log_path"])
            if not log.is_file() or not is_complete_log(r["model"], log.read_text(errors="replace")):
                raise RuntimeError("Successful meter entry with incomplete algorithm replay: {}".format(log))
    if meter_ok != manifest_ok:
        raise RuntimeError("Energy/manifest disagreement: meter-only={}, manifest-only={}".format(
            meter_ok - manifest_ok, manifest_ok - meter_ok))
    return meter_ok


def write_rtf_by_run(session, durations):
    """Non-inferential, auditable per-run RTF table from measured wall_seconds."""
    completed = collect_completed(session)
    rows = []
    for r in read_csv_rows(session / "energy_measurements.csv"):
        key = (r["sample_id"], r["model"], r["order_id"])
        if key not in completed or r["status"] != "OK":
            continue
        duration = float(Decimal(durations[r["sample_id"]]["duration_s"]))
        wall = float(r["wall_seconds"])
        energy = float(r["measured_compute_energy_j"])
        if not (wall > 0 and energy > 0 and math.isfinite(wall) and math.isfinite(energy)):
            raise RuntimeError("Invalid measured values: {}".format(key))
        rows.append(dict(dataset="moveiit", sample_id=r["sample_id"], model=r["model"],
                         order_id=r["order_id"], order_position=r["order_position"],
                         input_span_s="{:.9f}".format(duration),
                         wall_seconds=r["wall_seconds"],
                         real_time_factor="{:.9f}".format(wall / duration),
                         cpu_package_energy_j=r["cpu_package_energy_j"],
                         gpu_energy_j=r["gpu_energy_j"],
                         measured_compute_energy_j=r["measured_compute_energy_j"],
                         energy_j_per_input_second="{:.9f}".format(energy / duration)))
    path = session / "rtf_energy_by_run.csv"
    fields = ["dataset", "sample_id", "model", "order_id", "order_position",
              "input_span_s", "wall_seconds", "real_time_factor", "cpu_package_energy_j",
              "gpu_energy_j", "measured_compute_energy_j", "energy_j_per_input_second"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def check_gpu_idle():
    # nvidia-smi is used only outside the timed energy window.
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError("nvidia-smi failed: " + r.stderr.strip())
    if r.stdout.strip():
        raise RuntimeError("Compute processes are active on a GPU; energy is not isolated:\n" + r.stdout)


def check_sidecars(container):
    command = "pgrep -af '[m]ovenet_online.py|[e]ventPointPose_yarp_server.py|[Y]oloPose_yarp_server.py' || true"
    active = docker(container, "bash", "-lc", command)
    if active.strip():
        raise RuntimeError("An HPE Python sidecar is still running. Stop concurrent accuracy jobs first:\n" + active)


def preflight(args, samples):
    if sys.version_info < (3, 10):
        raise RuntimeError("Use Python >= 3.10 on the host (measure_energy.py requirement)")
    if not MEASURE_SCRIPT.is_file():
        raise RuntimeError("Missing established energy meter: " + str(MEASURE_SCRIPT))
    run_capture([sys.executable, "-c", "import pynvml"])
    try:
        int(Path("/sys/class/powercap/intel-rapl:0/energy_uj").read_text().strip())
        int(Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj").read_text().strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Host RAPL package counter unavailable/unreadable") from exc
    if run_capture(["docker", "inspect", "-f", "{{.State.Running}}", args.container]) != "true":
        raise RuntimeError("Container not running: " + args.container)
    for path, kind in ((MOVENET_BIN, "-x"), (EPP_BIN, "-x"),
                       (MOVENET_CKPT, "-s"), (EPP_CKPT, "-s"),
                       (EPP_SERVER, "-s"), (EPP_ADAPTER, "-s"),
                       (VENV + "/bin/python", "-x")):
        docker(args.container, "test", kind, path)
    docker(args.container, "bash", "-lc", "yarp where >/dev/null 2>&1")
    docker(args.container, VENV + "/bin/python", "-c", "import torch, yarp, numpy")
    for sample in samples:
        docker(args.container, "test", "-s", sample["event_file"])
        docker(args.container, "test", "-s", sample["gt_file"])
    check_sidecars(args.container)
    check_gpu_idle()
    spans = {s["sample_id"]: input_span(args.container, s) for s in samples}
    print("Preflight OK: models, sidecar, YARP, sample files, GPU/RAPL and exact ATIS packet spans", flush=True)
    return spans


def session_config(args, samples):
    return dict(protocol=PROTOCOL, dataset="move-iit-hpe", method_order=[c[0] for c in CASES],
                network_period_s={c[0]: float(c[1]) for c in CASES},
                flow_period_s=float(FLOW_PERIOD), output_period_s=float(OUTPUT_PERIOD),
                repeats=args.repeats, cooldown_s=args.cooldown,
                roi=35, gain=1.0, kf=[0.77, 0.06, 0.97], use_latency_compensation=True,
                epp_geometry="stretch", epp_fifo=7500, epp_points=2048, epp_seed=1,
                epp_preprocessing="already_filtered", device=args.device,
                gpu_index=args.gpu_index, container=args.container,
                sample_ids=[s["sample_id"] for s in samples],
                sample_manifest_sha256=sha256_local(args.samples_csv),
                binary_sha256={"movenet": sha256_inside(args.container, MOVENET_BIN),
                               "eventpointpose": sha256_inside(args.container, EPP_BIN)},
                checkpoint_sha256={"movenet": sha256_inside(args.container, MOVENET_CKPT),
                                   "eventpointpose": sha256_inside(args.container, EPP_CKPT)},
                epp_server_sha256=sha256_inside(args.container, EPP_SERVER),
                epp_adapter_sha256=sha256_inside(args.container, EPP_ADAPTER),
                energy_meter_sha256=sha256_local(MEASURE_SCRIPT),
                runner_sha256=sha256_local(Path(__file__).resolve()),
                duration_source=DURATION_SOURCE)


def initialise_session(session, config, samples, durations, resume):
    config_path = session / "config.json"
    if session.exists() and (config_path.exists() or any(session.iterdir())):
        if not resume:
            raise RuntimeError("Session already exists; use --resume or a NEW --session-dir: " + str(session))
        if not config_path.is_file() or json.loads(config_path.read_text()) != config:
            raise RuntimeError("Cannot resume: session configuration/hash differs")
        existing = read_csv_rows(session / "samples_durations.csv")
        old = {r["sample_id"]: r["duration_s"] for r in existing}
        if old != {sid: d["duration_s"] for sid, d in durations.items()}:
            raise RuntimeError("Cannot resume: ATIS recording spans changed")
    else:
        session.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        with (session / "samples_durations.csv").open("w", newline="") as f:
            fields = ["sample_id", "subject", "sequence", "event_file", "gt_file",
                      "duration_s", "input_start_s", "input_end_s", "duration_source",
                      "first_packet_id", "last_packet_id"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for s in samples:
                writer.writerow({**s, **durations[s["sample_id"]]})
    (session / "runs").mkdir(exist_ok=True)


def build_schedule(samples, repeats):
    tasks = []
    for sample_index, sample in enumerate(samples):
        for rep in range(1, repeats + 1):
            shift = (sample_index + rep - 1) % len(CASES)
            order = CASES[shift:] + CASES[:shift]
            for pos, case in enumerate(order, 1):
                tasks.append((sample, rep, pos, case))
    if len(samples) == 12 and repeats == 5:
        counts = Counter((case[0], pos) for _, _, pos, case in tasks)
        assert all(counts[(case[0], pos)] == 20 for case in CASES for pos in (1, 2, 3))
    return tasks


def cutoff_from_time(value):
    if not value:
        return None
    try:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("Bad format")
        hh, mm = int(parts[0]), int(parts[1])
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise ValueError("Bad clock time")
    except ValueError as exc:
        raise ValueError("--stop-at must be HH:MM (local host time)") from exc
    now = datetime.now().astimezone()
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--container", default="moveEnetOFK")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--session-dir", type=Path,
                   default=Path.home() / "data/MoveEnet_OFK_results/Energy/moveiit_200_200_5_v1")
    p.add_argument("--sample", default="", help="Single sample ID (smoke test); default all 12")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--cooldown", type=float, default=20.0)
    p.add_argument("--stop-at", default="", metavar="HH:MM",
                   help="Stop starting new runs at this local time; active run is allowed to finish")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.repeats <= 0 or args.cooldown < 0 or args.gpu_index < 0:
        p.error("repeats must be positive; cooldown and gpu-index must be non-negative")
    cutoff = cutoff_from_time(args.stop_at)
    all_samples = load_samples(args.samples_csv)
    if len(all_samples) != 12 and not args.sample:
        raise RuntimeError("Final benchmark expects exactly 12 selected recordings; found {}".format(len(all_samples)))
    samples = [s for s in all_samples if s["sample_id"] == args.sample] if args.sample else all_samples
    if not samples:
        p.error("Unknown --sample: " + args.sample)
    schedule = build_schedule(samples, args.repeats)
    print("Move-IIT energy + RTF | corrected 200/200/5 Hz protocol", flush=True)
    print("{} recordings x 3 methods x {} repeats = {} runs".format(
        len(samples), args.repeats, len(schedule)), flush=True)
    print("  MoveEnet-only: net=200Hz, output=200Hz", flush=True)
    print("  EventPointPose: net=200Hz, output=200Hz, geometry=stretch", flush=True)
    print("  MoveEnetOFK: net=5Hz, flow=200Hz, output=200Hz, ROI=35", flush=True)
    print("  Measurement: host RAPL CPU package + NVML GPU, complete-command wall clock", flush=True)
    print("  Session: {}".format(args.session_dir), flush=True)
    if cutoff:
        print("  Stop starting new runs at: {}".format(cutoff.isoformat(timespec="minutes")), flush=True)
    if args.dry_run:
        for sample, rep, pos, case in schedule[:min(6, len(schedule))]:
            print("  [{} R{} position {}] {}".format(sample["sample_id"], rep, pos,
                  shlex.join(build_command(args.container, args.device, sample["event_file"], case))))
        print("DRY-RUN: no files, benchmarks or energy measurements created", flush=True)
        return 0
    spans = preflight(args, samples)
    if args.preflight_only:
        for s in samples:
            print("  {}: {} s".format(s["sample_id"], spans[s["sample_id"]]["duration_s"]))
        return 0
    config = session_config(args, samples)
    session = args.session_dir.expanduser().resolve()
    initialise_session(session, config, samples, spans, args.resume)
    done = collect_completed(session)
    remaining = sum((sample["sample_id"], case[0], "R{}".format(rep)) not in done
                    for sample, rep, pos, case in schedule)
    print("Already complete: {} | pending: {}".format(len(done), remaining), flush=True)
    if remaining == 0:
        write_rtf_by_run(session, spans)
        print("All runs complete; RTF data: " + str(session / "rtf_energy_by_run.csv"), flush=True)
        return 0
    if args.cooldown:
        print("Initial cooldown {} seconds".format(args.cooldown), flush=True)
        time.sleep(args.cooldown)
    manifest_cols = ["dataset", "sample_id", "subject", "sequence", "camera", "motion_class",
                     "model", "repetition", "order_position", "net_period", "net_hz",
                     "flow_period", "output_period", "status", "energy_csv", "log_path", "order_id"]
    energy_csv = session / "energy_measurements.csv"
    for sample, rep, pos, case in schedule:
        model, net, has_flow = case
        key = (sample["sample_id"], model, "R{}".format(rep))
        if key in done:
            continue
        if cutoff and datetime.now().astimezone() >= cutoff:
            n = write_rtf_by_run(session, spans)
            print("Nightly cutoff reached. {} measured runs saved; restart with --resume.".format(n), flush=True)
            return 0
        check_sidecars(args.container)
        check_gpu_idle()
        log = session / "runs" / "{}_{}_r{:02d}.log".format(sample["sample_id"], model, rep)
        command = build_command(args.container, args.device, sample["event_file"], case)
        print("RUN {}/{}: {} method={} rep={} position={} network={} s".format(
            len(done)+1, len(schedule), sample["sample_id"], model, rep, pos, net), flush=True)
        measure = [sys.executable, str(MEASURE_SCRIPT), "--csv", str(energy_csv),
                   "--log", str(log), "--label", "{}_{}_{}".format(*key),
                   "--dataset", "moveiit", "--sample-id", sample["sample_id"],
                   "--model", model, "--order-id", key[2], "--order-position", str(pos),
                   "--net-period", net, "--flow-period", FLOW_PERIOD if has_flow else "",
                   "--gpu-index", str(args.gpu_index), "--", *command]
        result = subprocess.run(measure, check=False)
        content = log.read_text(errors="replace") if log.exists() else ""
        log_complete = is_complete_log(model, content)
        status = "OK" if result.returncode == 0 and log_complete else "FAILED"
        append_csv(session / "run_manifest.csv", manifest_cols,
                   dict(dataset="moveiit", sample_id=sample["sample_id"],
                        subject=sample["subject"], sequence=sample["sequence"], camera="atis-s",
                        motion_class=sample["sequence"].rsplit("_", 1)[-1][0],
                        model=model, repetition=rep, order_position=pos,
                        net_period=net, net_hz="{:.6f}".format(1.0/float(net)),
                        flow_period=FLOW_PERIOD if has_flow else "", output_period=OUTPUT_PERIOD,
                        status=status, energy_csv=str(energy_csv), log_path=str(log),
                        order_id=key[2]))
        if status != "OK":
            print("FAILED: {}. Last 30 log lines:\n{}".format(key,
                  "\n".join(content.splitlines()[-30:])), file=sys.stderr)
            raise RuntimeError("Benchmark stopped. Inspect {} and the meter CSV before resuming".format(log))
        done.add(key)
        n = write_rtf_by_run(session, spans)
        print("DONE: {} | rtf_energy_by_run.csv rows={}".format(key, n), flush=True)
        if args.cooldown and len(done) < len(schedule):
            time.sleep(args.cooldown)
    print("COMPLETE: {} runs; {}".format(len(done), session), flush=True)
    print("Energy measurements: " + str(energy_csv), flush=True)
    print("RTF + per-run energy: " + str(session / "rtf_energy_by_run.csv"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
