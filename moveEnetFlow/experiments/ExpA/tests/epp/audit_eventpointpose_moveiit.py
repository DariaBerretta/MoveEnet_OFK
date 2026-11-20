#!/usr/bin/env python3
"""Read-only audit of the move-iit-hpe EventPointPose ExpA output grid.

Python 3.8+; standard library only. Never edits inference CSVs or logs.
"""
import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

PERIODS = ("0.01", "0.02", "0.05", "0.1", "0.2", "0.5")
HEADER = ["timestamp", "latency"] + [
    "joint{}_{}".format(j, axis) for j in range(13) for axis in ("x", "y")
]


def read_metric(log, label):
    matches = re.findall(r"^\[INFO\]\s*" + re.escape(label) + r":\s*(\d+)\s*$", log, re.M)
    return int(matches[-1]) if matches else None


def audit_csv(path, period):
    info = {"csv_rows": 0, "first_ts_s": "", "last_ts_s": "", "csv_issue": ""}
    if not path.is_file() or path.stat().st_size == 0:
        info["csv_issue"] = "CSV missing or empty"
        return info
    problems = []
    previous = None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if header[:28] != HEADER:
                problems.append("unexpected CSV header")
            for line_no, fields in enumerate(reader, 2):
                if len(fields) != 28:
                    problems.append("line {} has {} columns (expected 28)".format(line_no, len(fields)))
                    break
                try:
                    values = [float(s) for s in fields]
                except ValueError:
                    problems.append("non-numeric values at line {}".format(line_no))
                    break
                if not all(math.isfinite(x) for x in values):
                    problems.append("NaN/Inf at line {}".format(line_no))
                    break
                t = values[0]
                if previous is not None:
                    delta = t - previous
                    if delta <= 0:
                        problems.append("non-increasing timestamps at line {}".format(line_no))
                        break
                    if abs(delta - period) > 0.00005:
                        problems.append("unexpected timestamp gap {:.6f}s at line {}".format(delta, line_no))
                        break
                else:
                    info["first_ts_s"] = "{:.6f}".format(t)
                previous = t
                info["csv_rows"] += 1
        if previous is not None:
            info["last_ts_s"] = "{:.6f}".format(previous)
        if info["csv_rows"] == 0:
            problems.append("no pose rows")
    except (OSError, UnicodeError, csv.Error) as exc:
        problems.append("CSV read error: {}".format(exc))
    info["csv_issue"] = "; ".join(problems)
    return info


def audit_latency(path):
    if not path.is_file():
        return None, "latency CSV missing", None
    n = 0
    last_ts = None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"request_id", "dataset_timestamp", "status"}.issubset(reader.fieldnames):
                return None, "latency header invalid", None
            for row in reader:
                n += 1
                last_ts = float(row["dataset_timestamp"])
                if not math.isfinite(last_ts):
                    return n, "nonfinite latency timestamp", None
    except (OSError, UnicodeError, csv.Error, ValueError, TypeError) as exc:
        return n, "latency read error: {}".format(exc), None
    return n, "", last_ts


def expected_recordings(root):
    recordings = []
    for subject, gt_name in (("luna", "P1_luna"), ("simon", "P11_simon")):
        for path in sorted((root / subject).glob("*/atis-s/data.log")):
            sequence = path.parent.parent.name
            if (root / "GT" / gt_name / (sequence + "_joint_projections.csv")).is_file():
                recordings.append((subject, sequence, path))
    return recordings


def check_run(results, geometry, subject, sequence, period, period_output):
    stem = "{}__{}__atis-s__epp_{}_np_{}".format(subject, sequence, geometry, period.replace(".", "p"))
    base = results / geometry
    pred = base / "raw" / (stem + ".csv")
    log_path = base / "logs" / (stem + ".log")
    latency_path = base / "latency" / (stem + ".csv")
    row = {"subject": subject, "sequence": sequence, "net_period_s": period,
           "stem": stem, "status": "", "reason": "", "csv_path": str(pred),
           "log_path": str(log_path), "latency_path": str(latency_path),
           "events_loaded": "", "events_read": "", "packets_loaded": "",
           "packets_processed": "", "pointnet_requests": "", "valid_inferences": "",
           "csv_rows_logged": "", "csv_rows": "", "latency_rows": "",
           "first_ts_s": "", "last_ts_s": ""}
    if not log_path.is_file():
        row.update(status="MISSING", reason="log missing")
        return row
    log = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(\d+)\s+packets loaded with\s+(\d+)\s+total events", log)
    loaded_packets = int(m.group(1)) if m else None
    loaded_events = int(m.group(2)) if m else None
    processed = read_metric(log, "Packet groups processed")
    events = read_metric(log, "Events read")
    requests = read_metric(log, "PointNet requests")
    valid = read_metric(log, "Valid PointNet inferences")
    csv_logged = read_metric(log, "CSV rows written")
    server_errors = read_metric(log, "Server ERROR responses")
    row.update(events_loaded=loaded_events if loaded_events is not None else "",
               events_read=events if events is not None else "",
               packets_loaded=loaded_packets if loaded_packets is not None else "",
               packets_processed=processed if processed is not None else "",
               pointnet_requests=requests if requests is not None else "",
               valid_inferences=valid if valid is not None else "",
               csv_rows_logged=csv_logged if csv_logged is not None else "")
    issues = []
    warnings = []
    explicit_interruption = bool(re.search(r"Termination signal received|EventPointPose request failed:|Failed to compute|Could not expose events", log, re.I))
    if explicit_interruption:
        issues.append("interruption/error reported in C++ log")
    if any(x is None for x in (processed, events, requests, valid, csv_logged)):
        issues.append("missing final C++ summary (possibly still running)")
    if loaded_events is None:
        issues.append("missing input event count in log")
    elif events is not None and loaded_events != events:
        issues.append("only {} of {} input events processed".format(events, loaded_events))
    if loaded_packets is not None and processed is not None and loaded_packets != processed:
        warnings.append("processed packet groups {} != input packets {}".format(processed, loaded_packets))
    if valid is not None and valid <= 0:
        issues.append("no valid PointNet inferences")
    if requests is not None and valid is not None and valid > requests:
        issues.append("more valid inferences than requests")
    if server_errors is not None and server_errors > 0:
        warnings.append("{} server ERROR responses".format(server_errors))
    csv_info = audit_csv(pred, period_output)
    row.update({key: csv_info[key] for key in ("csv_rows", "first_ts_s", "last_ts_s")})
    if csv_info["csv_issue"]:
        issues.append(csv_info["csv_issue"])
    if csv_logged is not None and csv_logged != csv_info["csv_rows"]:
        issues.append("CSV rows {} != logged {}".format(csv_info["csv_rows"], csv_logged))
    latency_count, latency_error, last_request_time = audit_latency(latency_path)
    row["latency_rows"] = latency_count if latency_count is not None else ""
    if latency_error:
        issues.append(latency_error)
    if requests is not None and latency_count is not None and requests != latency_count:
        issues.append("latency requests {} != logged {}".format(latency_count, requests))
    # This cross-check catches a truncated pose stream even when timestamps are regular.
    if last_request_time is not None and row["last_ts_s"]:
        last_pose_time = float(row["last_ts_s"])
        if last_pose_time < last_request_time - max(0.10, 2 * float(period)):
            issues.append("last pose timestamp predates last request by {:.3f}s".format(last_request_time - last_pose_time))
    if issues:
        row.update(status="INCOMPLETE", reason="; ".join(issues + warnings))
    elif warnings:
        row.update(status="REVIEW", reason="; ".join(warnings))
    else:
        row.update(status="OK", reason="full event count, summary, valid CSV and latency")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("/data/move-iit-it_testing"))
    ap.add_argument("--results", type=Path, default=Path("/data/MoveEnet_OFK_results/Accuracy/moveiit/EventPointPose"))
    ap.add_argument("--geometry", choices=("stretch", "letterbox"), default="stretch")
    ap.add_argument("--periods", nargs="+", default=list(PERIODS))
    ap.add_argument("--output-period", type=float, default=0.005)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    recordings = expected_recordings(args.data_root)
    if not recordings:
        ap.error("no paired event+GT recordings found under {}".format(args.data_root))
    report = args.report or args.results / args.geometry / "audit_epp_moveiit.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for subject, sequence, _ in recordings:
        for period in args.periods:
            entries.append(check_run(args.results, args.geometry, subject, sequence,
                                     period, args.output_period))
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    summary = Counter(row["status"] for row in entries)
    print("Expected: {} recordings x {} periods = {} runs".format(len(recordings), len(args.periods), len(entries)))
    print(" ".join("{}={}".format(key, summary.get(key, 0)) for key in ("OK", "REVIEW", "INCOMPLETE", "MISSING")))
    print("\nRuns requiring attention:")
    for row in entries:
        if row["status"] != "OK":
            print("  [{}] {}: {}".format(row["status"], row["stem"], row["reason"]))
    print("\nFull report: {}".format(report))
    if any(row["status"] != "OK" for row in entries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
