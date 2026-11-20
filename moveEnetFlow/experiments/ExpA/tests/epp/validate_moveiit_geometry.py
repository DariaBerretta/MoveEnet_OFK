#!/usr/bin/env python3
"""Non-invasive real-data geometry audit for EventPointPose on move-iit-hpe.

Reads existing --debug NumPy dumps and the already-projected 2D GT. Never
runs PointNet, writes source files, or changes model/preprocessing logic.

Run from any working directory with --dump-dir, --gt and --output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "utils"))
from epp_moveiit_adapter import Geometry  # noqa: E402

# ExpA MoveIIT_ExpA_2DGT_Analysis.ipynb: PROVISIONAL joint mapping.
EXPA_GT_TO_HPE = np.array([0, 1, 2, 3, 4, 8, 7, 5, 6, 9, 10, 11, 12])
HPE_FROM_EPP = np.array([0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12])
EDGES = ((0, 1), (0, 2), (1, 3), (3, 7), (2, 4), (4, 8),
         (1, 6), (2, 5), (5, 6), (6, 9), (9, 11), (5, 10), (10, 12))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_gt(path, order):
    import pandas as pd
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]
    t_col = "event_timestamp" if "event_timestamp" in df else (
        "timestamp" if "timestamp" in df else df.columns[0]
    )
    coords_cols = [c for c in df.columns if c != t_col]
    require(len(coords_cols) == 26,
            f"GT expected 13 XY pairs, got {len(coords_cols)} columns: {coords_cols}")
    times = df[t_col].to_numpy(dtype=np.float64)
    xy = df[coords_cols].to_numpy(dtype=np.float64).reshape(-1, 13, 2)
    require(len(times) >= 2 and np.isfinite(times).all()
            and np.all(np.diff(times) > 0), "GT timestamps must be finite, increasing")
    named_native = coords_cols == [f"joint{j}_{axis}" for j in range(13) for axis in "xy"]
    if named_native:
        applied_order = "identity (native headers)"
    elif order == "expa":
        xy = xy[:, EXPA_GT_TO_HPE]
        applied_order = "ExpA provisional GT mapping"
    else:
        applied_order = "identity (requested)"
    return times, xy, t_col, coords_cols, applied_order


def plot_skeleton(ax, xy, color, label, alpha=1.0, annotate=False, linewidth=1.4):
    valid = np.isfinite(xy).all(axis=1)
    first = True
    for a, b in EDGES:
        if valid[a] and valid[b]:
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=color,
                    linewidth=linewidth, alpha=alpha,
                    label=label if first else None)
            first = False
    ax.scatter(xy[valid, 0], xy[valid, 1], s=16, color=color,
               alpha=alpha, zorder=5, label=label if first else None)
    if annotate:
        for j, (x, y) in enumerate(xy):
            if np.isfinite(x) and np.isfinite(y):
                ax.annotate(str(j), (x, y), fontsize=6, xytext=(3, 3),
                            textcoords="offset points", color=color)


def config_ax(ax, width, height, title, xlabel, ylabel):
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True, help="*_joint_projections.csv")
    ap.add_argument("--output", type=Path, required=True, help="Output PNG path")
    ap.add_argument("--geometry", choices=("stretch", "letterbox"), default="stretch")
    ap.add_argument("--request-id", type=int, default=None)
    ap.add_argument("--gt-time-offset", type=float, default=0.0,
                    help="GT queried at request_time + offset, in seconds (ExpA default 0)")
    ap.add_argument("--gt-order", choices=("expa", "identity"), default="expa",
                    help="ExpA PROVISIONAL mapping or literal CSV column order")
    args = ap.parse_args()
    require(args.gt.is_file(), f"GT not found: {args.gt}")
    require(args.dump_dir.is_dir(), f"Debug directory not found: {args.dump_dir}")

    dumps = sorted(args.dump_dir.glob("request_*_fifo_epp.npy"))
    require(dumps, "No *_fifo_epp.npy found: run the existing experiment with --debug")
    stem = (dumps[-1].name[:-len("_fifo_epp.npy")] if args.request_id is None
            else f"request_{args.request_id:08d}")

    def arr(name):
        p = args.dump_dir / f"{stem}_{name}.npy"
        require(p.is_file(), f"Missing saved debug input: {p}")
        return np.load(p, allow_pickle=False)

    raw = arr("request_log")
    accepted = arr("accepted_batch_epp")
    fifo = arr("fifo_epp")
    rasepc_full = arr("rasepc_full")
    native_pred = arr("native_prediction")
    hpe_pred = arr("hpecore_prediction")
    require(raw.ndim == 2 and raw.shape[1] == 4, f"Bad request_log shape {raw.shape}")
    require(fifo.shape == (7500, 4), f"Bad FIFO shape {fifo.shape}")
    require(accepted.ndim == 2 and accepted.shape[1] == 4, "Bad accepted EPP events")
    require(rasepc_full.ndim == 2 and rasepc_full.shape[1] == 6, "Bad RasEPC cloud")
    require(native_pred.shape == (13, 2) and hpe_pred.shape == (13, 2), "Bad pose arrays")

    geometry = Geometry(mode=args.geometry)
    independently_adapted, invalid = geometry.preprocess(raw)
    require(independently_adapted.shape == accepted.shape,
            f"Input event-count mismatch: replay {independently_adapted.shape}, server {accepted.shape}")
    require(np.array_equal(independently_adapted, accepted),
            "FAIL: saved server accepted batch differs from replay of geometry adapter")

    expected_pose = geometry.to_native_pose(native_pred[HPE_FROM_EPP]).astype(np.float32)
    pose_max_delta = float(np.max(np.abs(expected_pose - hpe_pred)))
    require(np.allclose(expected_pose, hpe_pred, atol=1e-4, rtol=0),
            f"FAIL: joint inverse mapping differs from server by {pose_max_delta:.5f}px")

    x, y, t, p = raw.T
    valid = (np.isfinite(raw).all(axis=1) & (x >= 0) & (x < 1280)
             & (y >= 0) & (y < 720) & (x == np.floor(x))
             & (y == np.floor(y)) & ((p == 0) | (p == 1)))
    original_xy = raw[valid, :2]
    accepted_virtual = np.column_stack((accepted[:, 0] - 1,
                                         geometry.target_height - 1 - accepted[:, 1]))
    roundtrip_xy = geometry.to_native_pose(accepted_virtual)
    error = np.abs(roundtrip_xy - original_xy)
    roundtrip_x = float(np.max(error[:, 0])) if len(error) else 0.0
    roundtrip_y = float(np.max(error[:, 1])) if len(error) else 0.0
    sx, sy, ox, oy = geometry.coefficients
    require(roundtrip_x <= .5 / sx + 1e-8 and roundtrip_y <= .5 / sy + 1e-8,
            "FAIL: event-coordinate roundtrip exceeds quantization bound")
    require(np.array_equal(accepted[:, 2:], raw[valid, 2:]),
            "FAIL: the geometry adapter changed event times or polarities")

    gt_times, gt_all, gt_tcol, gt_columns, gt_order = load_gt(args.gt, args.gt_order)
    request_t = float(np.max(raw[:, 2]))
    gt_query_t = request_t + args.gt_time_offset
    print(f"Debug request: {stem}; request time={request_t:.6f} s")
    print(f"GT: {args.gt}; '{gt_tcol}' range = [{gt_times[0]:.6f}, {gt_times[-1]:.6f}] s")
    print(f"GT joint order: {gt_order} (ExpA mapping NOT anatomically verified)")
    require(gt_times[0] <= gt_query_t <= gt_times[-1],
            f"GT time {gt_query_t:.6f}s OUTSIDE GT range. Verify time origin / --gt-time-offset; "
            "do not use endpoint-held GT for geometry validation.")
    gt_at_t = np.column_stack([
        np.interp(gt_query_t, gt_times, gt_all[:, j, a])
        for j in range(13) for a in range(2)
    ]).reshape(13, 2)

    # The 7,500 EPP FIFO events are what RasEPC actually consumed.
    fifo_virtual = np.column_stack((fifo[:, 0] - 1,
                                    geometry.target_height - 1 - fifo[:, 1]))
    fifo_native = geometry.to_native_pose(fifo_virtual)
    virtual_gt = np.column_stack((gt_at_t[:, 0] * sx + ox,
                                   gt_at_t[:, 1] * sy + oy))
    virtual_pred = native_pred[HPE_FROM_EPP]

    # Exact coordinates used to rasterize inputs: original rasterizer indexes int()-1.
    rasepc_gt = np.column_stack((virtual_gt[:, 0],
                                 (geometry.target_height - 2) - virtual_gt[:, 1]))
    edge_y_zero = int(np.count_nonzero(fifo[:, 1] == 0))
    print(f"PASS: full request batch ({len(accepted)} events) is bitwise equal to adapter replay")
    print(f"PASS: order, packet timestamps and polarities preserved; rejected invalid={invalid}")
    print(f"PASS: inverse joint map exactly reproduced; max |delta|={pose_max_delta:.6g}px")
    print(f"PASS: max event-coordinate roundtrip error x={roundtrip_x:.3f}px, y={roundtrip_y:.3f}px")
    print(f"RasEPC input: FIFO={len(fifo)} events, cloud={len(rasepc_full)} points")
    print(f"NOTE: {edge_y_zero} FIFO events have EPP y=0 and exercise original RasEPC y=-1 flat indexing")
    print("NOTE: GT interpolation uses zero offset by default; GT->HPE mapping is provisional.")
    print("NOTE: a correct overlay is NOT proof that the model generalizes well.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), dpi=170)
    ax = axes[0]
    ax.scatter(fifo_native[:, 0], fifo_native[:, 1], s=.7, color="#458dc0",
               alpha=.33, linewidths=0, rasterized=True)
    plot_skeleton(ax, gt_at_t, "#20813a", "2D GT (provisional mapping)", annotate=True)
    plot_skeleton(ax, hpe_pred, "#dd6d22", "EventPointPose prediction", alpha=.8)
    config_ax(ax, 1280, 720, "A. ATIS native + GT + prediction", "ATIS x [px]", "ATIS y [px]")
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[1]
    ax.scatter(fifo_virtual[:, 0], fifo_virtual[:, 1], s=1, color="#458dc0",
               alpha=.3, linewidths=0, rasterized=True)
    plot_skeleton(ax, virtual_gt, "#20813a", "GT scaled to DHP19", annotate=True)
    plot_skeleton(ax, virtual_pred, "#dd6d22", "PointNet output indices", alpha=.8)
    config_ax(ax, 346, 260, "B. Virtual DHP19 (before EPP Y conversion)",
              "virtual x [px]", "virtual y [px]")

    ax = axes[2]
    ax.scatter(rasepc_full[:, 1], rasepc_full[:, 2], s=1, color="#458dc0",
               alpha=.38, linewidths=0, rasterized=True)
    plot_skeleton(ax, rasepc_gt, "#20813a", "GT in RasEPC pixel coordinates", annotate=True)
    config_ax(ax, 346, 260, "C. Actual RasEPC point cloud (Y inverted)",
              "RasEPC x [px]", "RasEPC y [px]")

    fig.suptitle(f"{stem} | {args.geometry} | request {request_t:.4f}s | GT {gt_query_t:.4f}s\n"
                 "Green: 2D GT, orange: PointNet (not an accuracy test); blue: event evidence",
                 fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    summary = dict(request=stem, geometry=args.geometry, request_time_s=request_t,
                   gt_time_s=gt_query_t, gt_time_offset_s=args.gt_time_offset,
                   gt_time_min_s=float(gt_times[0]), gt_time_max_s=float(gt_times[-1]),
                   gt_mapping=gt_order, gt_headers=gt_columns,
                   accepted_events=len(accepted), fifo_events=len(fifo),
                   rasepc_points=len(rasepc_full), invalid_events=invalid,
                   max_event_roundtrip_error_x_px=roundtrip_x,
                   max_event_roundtrip_error_y_px=roundtrip_y,
                   max_prediction_transform_error_px=pose_max_delta,
                   epp_y_zero_events=edge_y_zero,
                   note="Verified transformations only, not GT anatomical ordering or model accuracy")
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved image: {args.output}")
    print(f"Saved diagnostic summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"GEOMETRY VALIDATION ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
