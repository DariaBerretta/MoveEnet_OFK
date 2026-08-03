#!/usr/bin/env python3
"""Compare coordinate-convention candidates against GT with MPJPE/PCK.

This script is intentionally lightweight (stdlib + numpy only) so it can run in
minimal environments.

Expected prediction CSV format (from eventPointPose_offline.cpp):
    timestamp,latency,head_x,head_y,...,footL_x,footL_y[,window_start,window_end,window_mid]

GT CSV can use either:
  1) the same joint-name columns as prediction CSV, or
  2) x0,y0,...,x12,y12 columns.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


JOINT_NAMES: Tuple[str, ...] = (
    "head",
    "shoulderR",
    "shoulderL",
    "elbowR",
    "elbowL",
    "hipL",
    "hipR",
    "handR",
    "handL",
    "kneeR",
    "kneeL",
    "footR",
    "footL",
)


@dataclass
class TimedPose:
    timestamp: float
    xy: np.ndarray  # [13, 2]


def _required_joint_columns_named() -> List[str]:
    columns: List[str] = []
    for name in JOINT_NAMES:
        columns.append(f"{name}_x")
        columns.append(f"{name}_y")
    return columns


def _required_joint_columns_indexed() -> List[str]:
    columns: List[str] = []
    for index in range(len(JOINT_NAMES)):
        columns.append(f"x{index}")
        columns.append(f"y{index}")
    return columns


def _parse_pose_row_named(row: Dict[str, str]) -> np.ndarray:
    xy = np.zeros((len(JOINT_NAMES), 2), dtype=np.float64)
    for joint_index, name in enumerate(JOINT_NAMES):
        xy[joint_index, 0] = float(row[f"{name}_x"])
        xy[joint_index, 1] = float(row[f"{name}_y"])
    return xy


def _parse_pose_row_indexed(row: Dict[str, str]) -> np.ndarray:
    xy = np.zeros((len(JOINT_NAMES), 2), dtype=np.float64)
    for joint_index in range(len(JOINT_NAMES)):
        xy[joint_index, 0] = float(row[f"x{joint_index}"])
        xy[joint_index, 1] = float(row[f"y{joint_index}"])
    return xy


def _read_timed_poses(path: str, source_name: str) -> List[TimedPose]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        fields = set(fieldnames)

        if "timestamp" not in fields:
            raise RuntimeError(f"{source_name}: missing required column 'timestamp'.")

        named_columns = set(_required_joint_columns_named())
        indexed_columns = set(_required_joint_columns_indexed())

        if named_columns.issubset(fields):
            parser = _parse_pose_row_named
        elif indexed_columns.issubset(fields):
            parser = _parse_pose_row_indexed
        else:
            raise RuntimeError(
                f"{source_name}: could not find joint columns. "
                "Expected either named columns (head_x,...,footL_y) or indexed columns (x0,y0,...,x12,y12)."
            )

        poses: List[TimedPose] = []
        for row in reader:
            timestamp = float(row["timestamp"])
            xy = parser(row)
            poses.append(TimedPose(timestamp=timestamp, xy=xy))

    poses.sort(key=lambda item: item.timestamp)
    return poses


def _transform(xy: np.ndarray, mode: str, sensor_w: int, sensor_h: int) -> np.ndarray:
    result = np.asarray(xy, dtype=np.float64).copy()

    if mode == "identity":
        return result
    if mode == "raw_flip_x":
        result[:, 0] = float(sensor_w - 1) - result[:, 0]
        return result
    if mode == "raw_flip_y":
        result[:, 1] = float(sensor_h - 1) - result[:, 1]
        return result
    if mode == "raw_dhp19":
        result[:, 0] = float(sensor_w - 1) - result[:, 0]
        result[:, 1] = np.mod(float(sensor_h) - result[:, 1], float(sensor_h))
        return result

    raise ValueError(f"Unsupported convention mode: {mode}")


def _match_by_timestamp(
    pred: Sequence[TimedPose],
    gt: Sequence[TimedPose],
    tolerance_s: float,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    matches: List[Tuple[np.ndarray, np.ndarray, float]] = []
    if not pred or not gt:
        return matches

    gt_timestamps = np.asarray([item.timestamp for item in gt], dtype=np.float64)

    for pred_item in pred:
        position = int(np.searchsorted(gt_timestamps, pred_item.timestamp))

        candidates: List[int] = []
        if position < len(gt):
            candidates.append(position)
        if position - 1 >= 0:
            candidates.append(position - 1)

        if not candidates:
            continue

        best_index = min(
            candidates,
            key=lambda index: abs(gt[index].timestamp - pred_item.timestamp),
        )
        dt = abs(gt[best_index].timestamp - pred_item.timestamp)
        if dt <= tolerance_s:
            matches.append((pred_item.xy, gt[best_index].xy, dt))

    return matches


def _compute_metrics(
    transformed_pred: Sequence[np.ndarray],
    gt: Sequence[np.ndarray],
    pck_thresholds: Sequence[float],
) -> Tuple[float, Dict[float, float]]:
    if len(transformed_pred) == 0:
        return float("nan"), {float(th): float("nan") for th in pck_thresholds}

    pred_array = np.stack(transformed_pred, axis=0)
    gt_array = np.stack(gt, axis=0)
    distances = np.linalg.norm(pred_array - gt_array, axis=2)  # [N, 13]

    mpjpe = float(np.mean(distances))
    pck = {
        float(th): float(np.mean(distances <= float(th))) for th in pck_thresholds
    }
    return mpjpe, pck


def _parse_thresholds(value: str) -> List[float]:
    thresholds: List[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        thresholds.append(float(token))
    if not thresholds:
        raise ValueError("At least one PCK threshold is required.")
    return thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep EventPointPose coordinate conventions against GT."
    )
    parser.add_argument("--pred_csv", required=True, help="Prediction CSV path.")
    parser.add_argument("--gt_csv", required=True, help="GT CSV path.")
    parser.add_argument("--sensor_w", type=int, default=346)
    parser.add_argument("--sensor_h", type=int, default=260)
    parser.add_argument(
        "--timestamp_tolerance_s",
        type=float,
        default=0.0025,
        help="Maximum allowed |pred_ts - gt_ts| for a match.",
    )
    parser.add_argument(
        "--modes",
        default="identity,raw_flip_x,raw_flip_y,raw_dhp19",
        help="Comma-separated list of convention candidates to score.",
    )
    parser.add_argument(
        "--pck_thresholds_px",
        default="5,10,20",
        help="Comma-separated PCK distance thresholds in pixels.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    modes = [token.strip() for token in args.modes.split(",") if token.strip()]
    if not modes:
        raise RuntimeError("At least one mode is required.")

    pck_thresholds = _parse_thresholds(args.pck_thresholds_px)
    pred_poses = _read_timed_poses(args.pred_csv, "Prediction CSV")
    gt_poses = _read_timed_poses(args.gt_csv, "GT CSV")

    matches = _match_by_timestamp(pred_poses, gt_poses, args.timestamp_tolerance_s)
    if not matches:
        raise RuntimeError(
            "No timestamp matches found. Increase --timestamp_tolerance_s or verify CSV timestamps."
        )

    mean_dt = float(np.mean([item[2] for item in matches]))
    max_dt = float(np.max([item[2] for item in matches]))

    print(
        "MATCHES count={} pred_total={} gt_total={} mean_dt_s={:.6f} max_dt_s={:.6f}".format(
            len(matches),
            len(pred_poses),
            len(gt_poses),
            mean_dt,
            max_dt,
        )
    )

    rows: List[Tuple[str, float, Dict[float, float]]] = []
    gt_xy = [item[1] for item in matches]

    for mode in modes:
        transformed = [
            _transform(item[0], mode, args.sensor_w, args.sensor_h) for item in matches
        ]
        mpjpe, pck = _compute_metrics(transformed, gt_xy, pck_thresholds)
        rows.append((mode, mpjpe, pck))

    rows.sort(key=lambda row: row[1])

    header = ["mode", "mpjpe_px"] + [f"pck@{th:g}px" for th in pck_thresholds]
    print(",".join(header))
    for mode, mpjpe, pck in rows:
        fields = [mode, f"{mpjpe:.6f}"]
        for threshold in pck_thresholds:
            fields.append(f"{pck[threshold]:.6f}")
        print(",".join(fields))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
