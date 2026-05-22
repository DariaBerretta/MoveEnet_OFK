#!/usr/bin/env python3
"""Offline EventPointPose runner.

This mirrors the C++/OpenPose runners by exposing separate net/output periods:
- `--net_period` controls model inference rate (seconds)
- `--output_period` controls CSV/video write rate (seconds)

If not provided the script falls back to `--frequency` for both behaviours.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from eventpointpose_estimator import (
    EventPointPoseConfig,
    EventPointPoseEstimator,
    KEYPOINT_ORDER,
    SKELETON_PARENT_IDS,
    discover_event_inputs,
    iter_official_h5_frames,
    iter_official_npy_frames,
    iter_last_n_at_fixed_steps,
    parse_skeleton_log,
    read_yarp_ae_log,
    rescale_events_to_sensor,
    rescale_keypoints,
    write_pose_csv_header,
    write_pose_csv_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EventPointPose offline inference")
    parser.add_argument("--dataset_path", required=True, help="DHP19EPC folder/file or raw ch*dvs/data.log")
    parser.add_argument(
        "--model_type",
        default="pointnet",
        choices=["pointnet", "dgcnn", "pointtransformer"],
        help="EventPointPose backend to use",
    )
    parser.add_argument("--checkpoint", default="/home/model_mounts/eventpointpose/PointNet/models/model.pth", help="Path to model.pth checkpoint")
    parser.add_argument("--eventpointpose_repo", default="/home/EventPointPose", help="Official repo path")
    parser.add_argument("--frequency", type=float, default=30.0, help="Fallback inference/output frequency in Hz")
    parser.add_argument(
        "--net_period",
        type=float,
        default=None,
        help="Model inference period in seconds. Overrides `frequency` when set.",
    )
    parser.add_argument(
        "--output_period",
        type=float,
        default=None,
        help="CSV/visualization write period in seconds. Overrides `frequency` when set.",
    )
    parser.add_argument(
        "--window_seconds",
        type=float,
        default=None,
        help="Raw-event time window width. Default is 1/frequency.",
    )
    parser.add_argument("--output_csv", default=None, help="Output CSV path")
    parser.add_argument("--no_csv", action="store_true", help="Skip CSV logging")
    parser.add_argument("--vis", action="store_true", help="Show visualization while running")
    parser.add_argument("--output_video", default="", help="Optional visualization video path")
    parser.add_argument("--no_video", action="store_true", help="Disable video output")
    parser.add_argument("--sequence", default=None, help="Optional filename/path filter for a sequence")
    parser.add_argument("--camera", type=int, default=None, help="Optional DHP19 camera index, 0-3")
    parser.add_argument("--label", default="mean", choices=["mean", "last"], help="Official label/data setting")
    parser.add_argument("--num_points", type=int, default=2048, help="Point cloud size expected by checkpoint")
    parser.add_argument("--sensor_width", type=int, default=346, help="EventPointPose model sensor width")
    parser.add_argument("--sensor_height", type=int, default=260, help="EventPointPose model sensor height")
    parser.add_argument("--input_width", type=int, default=346, help="Raw AE log input width before scaling")
    parser.add_argument("--input_height", type=int, default=260, help="Raw AE log input height before scaling")
    parser.add_argument(
        "--input_coord_base",
        default="zero",
        choices=["zero", "one"],
        help="Coordinate base for raw AE logs before scaling",
    )
    parser.add_argument(
        "--no_raw_y_flip",
        action="store_true",
        help="Do not flip raw AE y coordinates into the official DHP19EPC convention",
    )
    parser.add_argument("--output_width", type=int, default=None, help="Optional CSV/video output width")
    parser.add_argument("--output_height", type=int, default=None, help="Optional CSV/video output height")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--cuda_num", type=int, default=0, help="CUDA device number used by official args")
    parser.add_argument("--seed", type=int, default=1, help="Sampling seed")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional limit for quick checks")
    parser.add_argument(
        "--skeleton_passthrough",
        action="store_true",
        help="Write SKLT labels as CSV without model inference. For parser checks only.",
    )
    return parser


def default_checkpoint(model_type: str) -> Path:
    return Path(__file__).resolve().parent / "EventPointPoseModel" / model_type / "model.pth"


def default_output_csv(frequency: float) -> Path:
    suffix = f"{frequency:g}".replace(".", "p")
    return Path(__file__).resolve().parent / "outputs" / f"eventpointpose_{suffix}hz.csv"


def ensure_parent(path: Path) -> None:
    parent = path.expanduser().parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def accumulate_event_image(events_one_based: np.ndarray, width: int, height: int) -> np.ndarray:
    img = np.zeros((height, width), dtype=np.float32)
    if events_one_based.size:
        x = np.clip(events_one_based[:, 0].astype(np.int32) - 1, 0, width - 1)
        y_official = np.clip(events_one_based[:, 1].astype(np.int32), 1, height)
        y = np.clip(height - y_official, 0, height - 1)
        np.add.at(img, (y, x), 1.0)
    if img.max() > 0:
        img = img / img.max() * 255.0
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def draw_skeleton(
    canvas: np.ndarray,
    keypoints_xy: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
    thickness: int = 2,
) -> None:
    h, w = canvas.shape[:2]
    pts = keypoints_xy.astype(np.int32)
    for joint_idx, parent_idx in enumerate(SKELETON_PARENT_IDS):
        x, y = pts[joint_idx]
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), radius, color, -1)
        if joint_idx == parent_idx:
            continue
        px, py = pts[parent_idx]
        if 0 <= x < w and 0 <= y < h and 0 <= px < w and 0 <= py < h:
            cv2.line(canvas, (px, py), (x, y), color, thickness, cv2.LINE_AA)


def render_frame(
    events_sensor: np.ndarray,
    keypoints_xy: np.ndarray,
    sensor_width: int,
    sensor_height: int,
    output_width: Optional[int],
    output_height: Optional[int],
) -> np.ndarray:
    canvas = accumulate_event_image(events_sensor, sensor_width, sensor_height)
    if output_width is not None and output_height is not None:
        canvas = cv2.resize(canvas, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    draw_skeleton(canvas, keypoints_xy, color=(255, 0, 0), radius=3, thickness=2)
    return canvas


def maybe_init_video(args, frame_size: Tuple[int, int]):
    if args.no_video or not args.output_video:
        return None
    output_video = Path(args.output_video).expanduser()
    ensure_parent(output_video)
    fps = max(1, int(round(1.0 / (args.output_period if args.output_period is not None else 1.0 / args.frequency))))
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video}")
    print(f"Video output enabled -> {output_video} at {fps} FPS")
    return writer


def show_or_write_frame(writer, args, frame: np.ndarray) -> bool:
    if writer is not None:
        writer.write(frame)
    if not args.vis:
        return False
    cv2.imshow("EventPointPose_offline", frame)
    key = cv2.waitKey(1)
    return key == 27 or key == ord("q")


def run_skeleton_passthrough(paths, csv_writer, args) -> int:
    count = 0
    for path in paths:
        for timestamp, keypoints in parse_skeleton_log(path):
            output = rescale_keypoints(
                keypoints,
                args.sensor_width,
                args.sensor_height,
                args.output_width,
                args.output_height,
            )
            if csv_writer is not None:
                write_pose_csv_row(csv_writer, timestamp, 0.0, output)
            count += 1
            if args.max_frames is not None and count >= args.max_frames:
                return count
    return count


def run_raw_logs(paths, estimator, csv_writer, video_writer, args, net_period, output_period) -> int:
    count = 0
    input_zero_based = args.input_coord_base == "zero"
    flip_raw_y = not args.no_raw_y_flip

    base_freq = max(args.frequency, 1.0 / net_period, 1.0 / output_period)
    for path in paths:
        print(f"Loading raw AE log: {path}")
        raw_events = read_yarp_ae_log(path)
        print(f"Decoded {len(raw_events)} events from {path}")

        next_net_upd = net_period
        next_out_upd = output_period
        last_keypoints = None
        last_latency = 0.0

        # Use fixed-step, last-N-event windowing: at each inference timestamp
        # take the most recent `args.num_points` events with event time <= t.
        for timestamp, window in iter_last_n_at_fixed_steps(raw_events, net_period, args.num_points):
            events_sensor = rescale_events_to_sensor(
                window,
                input_width=args.input_width,
                input_height=args.input_height,
                sensor_width=args.sensor_width,
                sensor_height=args.sensor_height,
                input_zero_based=input_zero_based,
                flip_y=flip_raw_y,
            )

            did_stop = False

            # Run model at net_period
            if timestamp >= next_net_upd or last_keypoints is None:
                last_keypoints, last_latency = estimator.predict_events(events_sensor)
                next_net_upd += net_period

            # Output (CSV/video/vis) at output_period
            if timestamp >= next_out_upd:
                kp_out = rescale_keypoints(
                    last_keypoints,
                    args.sensor_width,
                    args.sensor_height,
                    args.output_width,
                    args.output_height,
                )
                if csv_writer is not None:
                    write_pose_csv_row(csv_writer, timestamp, last_latency, kp_out)
                count += 1
                if args.vis or (video_writer is not None):
                    frame = render_frame(
                        events_sensor,
                        kp_out,
                        args.sensor_width,
                        args.sensor_height,
                        args.output_width,
                        args.output_height,
                    )
                    did_stop = show_or_write_frame(video_writer, args, frame)
                next_out_upd += output_period

            if count % 100 == 0 and count > 0:
                print(f"Processed {count} EventPointPose windows")
            if did_stop or (args.max_frames is not None and count >= args.max_frames):
                return count
    return count


def run_official_frames(kind, paths, estimator, csv_writer, video_writer, args, net_period, output_period) -> int:
    count = 0
    base_freq = max(args.frequency, 1.0 / net_period, 1.0 / output_period)
    if kind == "npy":
        frame_iter = iter_official_npy_frames(paths, args.label, base_freq)
    elif kind == "h5":
        frame_iter = iter_official_h5_frames(paths, args.label, args.camera, base_freq)
    else:
        raise ValueError(f"Unsupported official frame kind: {kind}")

    next_net_upd = net_period
    next_out_upd = output_period
    last_keypoints = None
    last_latency = 0.0

    for timestamp, events_sensor, _source_path in frame_iter:
        did_stop = False
        if timestamp >= next_net_upd or last_keypoints is None:
            last_keypoints, last_latency = estimator.predict_events(events_sensor)
            next_net_upd += net_period

        if timestamp >= next_out_upd:
            kp_out = rescale_keypoints(
                last_keypoints,
                args.sensor_width,
                args.sensor_height,
                args.output_width,
                args.output_height,
            )
            if csv_writer is not None:
                write_pose_csv_row(csv_writer, timestamp, last_latency, kp_out)
            count += 1
            if args.vis or (video_writer is not None):
                frame = render_frame(
                    events_sensor,
                    kp_out,
                    args.sensor_width,
                    args.sensor_height,
                    args.output_width,
                    args.output_height,
                )
                did_stop = show_or_write_frame(video_writer, args, frame)
            next_out_upd += output_period

        if count % 100 == 0 and count > 0:
            print(f"Processed {count} EventPointPose frames")
        if did_stop or (args.max_frames is not None and count >= args.max_frames):
            break
    return count


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if len(KEYPOINT_ORDER) != 13:
        raise RuntimeError("Internal keypoint order must contain exactly 13 joints.")
    if args.output_width is None and args.output_height is not None:
        parser.error("--output_width and --output_height must be provided together")
    if args.output_width is not None and args.output_height is None:
        parser.error("--output_width and --output_height must be provided together")

    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.model_type)
    output_csv = Path(args.output_csv) if args.output_csv else default_output_csv(args.frequency)
    if not args.no_csv:
        ensure_parent(output_csv)

    # Determine net/output periods (seconds). Fall back to frequency when not provided.
    net_period = args.net_period if args.net_period is not None else 1.0 / max(args.frequency, 1e-9)
    output_period = args.output_period if args.output_period is not None else 1.0 / max(args.frequency, 1e-9)

    # If the user did not set a window width explicitly, align the event
    # window duration to the model inference period so the sampled point
    # cloud reflects the same temporal interval used by the network.
    if args.window_seconds is None:
        args.window_seconds = net_period

    kind, paths = discover_event_inputs(Path(args.dataset_path), args.sequence, args.camera)
    print(f"Discovered {len(paths)} {kind} input(s)")

    csv_file = None
    csv_writer = None
    try:
        if not args.no_csv:
            csv_file = output_csv.expanduser().open("w", newline="")
            csv_writer = csv.writer(csv_file)
            write_pose_csv_header(csv_writer)
            print(f"CSV logging enabled -> {output_csv}")

        if kind == "skeleton_log":
            if not args.skeleton_passthrough:
                print(
                    "The discovered input is an SKLT skeleton/label log, not an event stream. "
                    "Pass a DHP19EPC data folder or ch*dvs/data.log for EventPointPose inference. "
                    "Use --skeleton_passthrough only to test the SKLT parser."
                )
                return 2
            count = run_skeleton_passthrough(paths, csv_writer, args)
            print(f"SKLT rows written: {count}")
            return 0

        cfg = EventPointPoseConfig(
            model_type=args.model_type,
            checkpoint=checkpoint,
            eventpointpose_repo=Path(args.eventpointpose_repo),
            num_points=args.num_points,
            sensor_width=args.sensor_width,
            sensor_height=args.sensor_height,
            label=args.label,
            cuda_num=args.cuda_num,
            device=args.device,
            seed=args.seed,
        )
        estimator = EventPointPoseEstimator(cfg)
        print(
            f"Loaded EventPointPose {estimator.model_name} on {estimator.device}. "
            f"Checkpoint: {checkpoint}"
        )

        frame_width = args.output_width or args.sensor_width
        frame_height = args.output_height or args.sensor_height
        video_writer = maybe_init_video(args, (frame_width, frame_height))
        try:
            if args.vis:
                cv2.namedWindow("EventPointPose_offline", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("EventPointPose_offline", frame_width, frame_height)
            if kind == "raw_log":
                count = run_raw_logs(paths, estimator, csv_writer, video_writer, args, net_period, output_period)
            else:
                count = run_official_frames(kind, paths, estimator, csv_writer, video_writer, args, net_period, output_period)
        finally:
            if video_writer is not None:
                video_writer.release()
                print(f"Video saved to: {args.output_video}")
            if args.vis:
                cv2.destroyAllWindows()
        print(f"EventPointPose rows written: {count}")
        return 0
    finally:
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
