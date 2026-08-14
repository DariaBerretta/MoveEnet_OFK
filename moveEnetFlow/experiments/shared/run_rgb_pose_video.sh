#!/usr/bin/env bash
set -euo pipefail

# Save a synchronized RGB-pose video for one eH36M RGB sequence.
#
# Final video contents:
#   - eH36M RGB frames
#   - eH36M ground-truth skeleton in green
#   - OpenPose prediction in magenta
#
# OpenPose inference and video/CSV output use independent rates:
#   - NET_PERIOD controls how often OpenPose is executed.
#   - OUTPUT_PERIOD controls the held-pose CSV/video rate.
#
# With the defaults:
#   - OpenPose runs at 50 Hz       (NET_PERIOD=0.02 s)
#   - output is produced at 200 Hz (OUTPUT_PERIOD=0.005 s)
#
# Between two OpenPose inferences, OpenPose_offline keeps the latest valid pose.
# The script is headless: it writes MP4 files without opening an OpenCV window.

POSE_METHOD="${POSE_METHOD:-openpose}"
RESULTS_ROOT="${MOVENET_RESULTS_ROOT:-/data/MovEnet_OFK_results}"
case "$POSE_METHOD" in
  openpose)
    POSE_LABEL="OpenPose"
    BINARY="/workspace/moveEnetFlow/build2/OpenPose_offline"
    MODEL_PATH="/usr/local/src/openpose/models/"
    OUTPUT_DIR="$RESULTS_ROOT/OP_h36m_full_test/videos"
    ;;
  yolo)
    POSE_LABEL="YOLO Pose"
    BINARY="/workspace/moveEnetFlow/build2/YoloPose_offline"
    MODEL_PATH="/workspace/model_mounts/YoloPose/yolo26n-pose.pt"
    YOLO_SCRIPT="/workspace/model_mounts/YoloPose/YoloPose_yarp_server.py"
    OUTPUT_DIR="$RESULTS_ROOT/YOLO_h36m_full_test/videos"
    ;;
  *) echo "Unknown POSE_METHOD: $POSE_METHOD" >&2; exit 2 ;;
esac

DATA_ROOT="/data/eh36m_testing_set_S9S11/rgb"
GT_ROOT="/data/eh36m_testing_set_S9S11/events"
DATA_FILE=""
DATA_GLOB="*.mp4"

NET_PERIOD="0.02"
OUTPUT_PERIOD="0.005"
DEVICE="cuda:0"
IMG_W="640"
IMG_H="480"
GPU_PERIOD_MS="5"
KEEP_BASE_VIDEO="false"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Input selection:
  --data_root <path>          eH36M RGB directory containing .mp4 files
  --gt_root <path>            eH36M events directory containing GT skeletons
  --data_file <path>          Use this .mp4 instead of auto-selecting a sequence
  --data_glob <pattern>       Filename pattern used during automatic selection

Pose model:
  --binary <path>             Pose executable
  --op_model_path <path>      OpenPose models directory
  --yolo_model_path <path>    YOLO model (.pt)
  --yolo_script <path>        YOLO YARP sidecar
  --net_period <seconds>      Interval between inferences. Default: 0.02
  --output_period <seconds>   Held-pose CSV/video period. Default: 0.005
  --device <device>           OpenPose device. Default: cuda:0
  --w <pixels>                Input width. Default: 640
  --h <pixels>                Input height. Default: 480

Output:
  --output_dir <path>         Output directory
  --keep_base_video           Keep the intermediate C++ video
  --help                      Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --gt_root) GT_ROOT="$2"; shift 2 ;;
    --data_file) DATA_FILE="$2"; shift 2 ;;
    --data_glob) DATA_GLOB="$2"; shift 2 ;;
    --op_model_path|--yolo_model_path) MODEL_PATH="$2"; shift 2 ;;
    --yolo_script) YOLO_SCRIPT="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --net_period) NET_PERIOD="$2"; shift 2 ;;
    --output_period) OUTPUT_PERIOD="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --w) IMG_W="$2"; shift 2 ;;
    --h) IMG_H="$2"; shift 2 ;;
    --keep_base_video) KEEP_BASE_VIDEO="true"; shift ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Validate dependencies and parameters
# -----------------------------------------------------------------------------
if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi

if [[ "$POSE_METHOD" == "openpose" && ! -d "$MODEL_PATH" ]]; then
  echo "OpenPose models directory not found: $MODEL_PATH" >&2; exit 1
fi
if [[ "$POSE_METHOD" == "yolo" && ! -f "$MODEL_PATH" ]]; then
  echo "YOLO model not found: $MODEL_PATH" >&2; exit 1
fi
if [[ "$POSE_METHOD" == "yolo" && ! -f "$YOLO_SCRIPT" ]]; then
  echo "YOLO sidecar not found: $YOLO_SCRIPT" >&2; exit 1
fi

if [[ ! -d "$GT_ROOT" ]]; then
  echo "Ground-truth root not found: $GT_ROOT" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found in PATH." >&2
  exit 1
fi

if ! awk "BEGIN{exit !($NET_PERIOD > 0.0)}"; then
  echo "net_period must be greater than zero: $NET_PERIOD" >&2
  exit 1
fi

if ! awk "BEGIN{exit !($OUTPUT_PERIOD > 0.0)}"; then
  echo "output_period must be greater than zero: $OUTPUT_PERIOD" >&2
  exit 1
fi

# Select the first RGB sequence for which the corresponding 200 Hz GT exists.
if [[ -z "$DATA_FILE" ]]; then
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "eH36M RGB root not found: $DATA_ROOT" >&2
    exit 1
  fi

  while IFS= read -r candidate; do
    candidate_sequence="$(basename "$candidate" .mp4)"
    candidate_gt="$GT_ROOT/$candidate_sequence/ch0GT200Hzskeleton/data.log"

    if [[ -f "$candidate_gt" ]]; then
      DATA_FILE="$candidate"
      break
    fi
  done < <(find "$DATA_ROOT" -type f -name "$DATA_GLOB" | sort)
fi

if [[ -z "$DATA_FILE" || ! -f "$DATA_FILE" ]]; then
  echo "No eH36M RGB video with matching GT was found." >&2
  echo "RGB root: $DATA_ROOT" >&2
  echo "GT root : $GT_ROOT" >&2
  echo "Pattern : $DATA_GLOB" >&2
  exit 1
fi

SEQUENCE="$(basename "$DATA_FILE" .mp4)"
GT_FILE="$GT_ROOT/$SEQUENCE/ch0GT200Hzskeleton/data.log"

if [[ ! -f "$GT_FILE" ]]; then
  echo "Ground-truth skeleton file not found: $GT_FILE" >&2
  echo "Expected sequence folder derived from: $DATA_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

SAFE_NET_PERIOD="${NET_PERIOD//./p}"
SAFE_OUTPUT_PERIOD="${OUTPUT_PERIOD//./p}"
STEM="${SEQUENCE}_${POSE_METHOD}_np_${SAFE_NET_PERIOD}_op_${SAFE_OUTPUT_PERIOD}"

PRED_CSV="$OUTPUT_DIR/${STEM}_prediction.csv"
BASE_VIDEO="$OUTPUT_DIR/${STEM}_base_rgb_prediction.mp4"
FINAL_VIDEO="$OUTPUT_DIR/${STEM}_rgb_gt_prediction.mp4"
RUN_LOG="$OUTPUT_DIR/${STEM}.log"
GPU_CSV="$OUTPUT_DIR/${STEM}_gpu.csv"

COMMON_ARGS=(
  --data_file "$DATA_FILE"
  --net_period "$NET_PERIOD"
  --output_period "$OUTPUT_PERIOD"
  --w "$IMG_W"
  --h "$IMG_H"
  --device "$DEVICE"
  --output_csv "$PRED_CSV"
  --output_video "$BASE_VIDEO"
  --gpu_file "$GPU_CSV"
  --gpu_period_ms "$GPU_PERIOD_MS"
)
if [[ "$POSE_METHOD" == "openpose" ]]; then
  COMMON_ARGS+=(--op_model_path "$MODEL_PATH")
else
  COMMON_ARGS+=(--yolo_model_path "$MODEL_PATH" --YoloPose_script "$YOLO_SCRIPT")
fi

echo "$POSE_LABEL synchronized video"
echo "RGB video      : $DATA_FILE"
echo "GT file        : $GT_FILE"
echo "Model          : $MODEL_PATH"
echo "Net period     : $NET_PERIOD s ($(awk "BEGIN{printf \"%.3f\", 1/$NET_PERIOD}") Hz)"
echo "Output period  : $OUTPUT_PERIOD s ($(awk "BEGIN{printf \"%.3f\", 1/$OUTPUT_PERIOD}") Hz)"
echo "Base video     : $BASE_VIDEO"
echo "Final video    : $FINAL_VIDEO"
echo "Prediction CSV : $PRED_CSV"
echo "Run log        : $RUN_LOG"

echo
echo "Running $POSE_LABEL..."
"$BINARY" "${COMMON_ARGS[@]}" >"$RUN_LOG" 2>&1

if [[ ! -s "$PRED_CSV" ]]; then
  echo "Prediction CSV was not created or is empty: $PRED_CSV" >&2
  echo "Inspect: $RUN_LOG" >&2
  exit 1
fi

if [[ ! -s "$BASE_VIDEO" ]]; then
  echo "Base video was not created or is empty: $BASE_VIDEO" >&2
  echo "Inspect: $RUN_LOG" >&2
  exit 1
fi

echo "Overlaying eH36M GT and redrawing $POSE_LABEL prediction..."

python3 - "$BASE_VIDEO" "$PRED_CSV" "$GT_FILE" "$FINAL_VIDEO" "$POSE_LABEL" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from datasets.utils import constants as ds_constants
    from datasets.utils import parsing as ds_parsing
except Exception as exc:
    raise SystemExit(
        "Could not import datasets.utils from hpe-core. "
        "Run this script in the same environment used by the evaluation notebook. "
        f"Import error: {exc!r}"
    )

base_video = Path(sys.argv[1])
pred_csv = Path(sys.argv[2])
gt_file = Path(sys.argv[3])
final_video = Path(sys.argv[4])
pose_label = sys.argv[5]

# HPE-core skeleton13 order:
# 0 head, 1 shoulder_right, 2 shoulder_left, 3 elbow_right, 4 elbow_left,
# 5 hip_left, 6 hip_right, 7 wrist_right, 8 wrist_left,
# 9 knee_right, 10 knee_left, 11 ankle_right, 12 ankle_left.
EDGES = [
    (0, 1), (0, 2), (1, 2),
    (1, 3), (3, 7),
    (2, 4), (4, 8),
    (1, 6), (2, 5), (5, 6),
    (6, 9), (9, 11),
    (5, 10), (10, 12),
]

GT_COLOR = (0, 255, 0)          # green, BGR
PRED_COLOR = (255, 0, 255)      # magenta, BGR
OUTLINE_COLOR = (0, 0, 0)

pred_array = np.loadtxt(pred_csv, delimiter=",", skiprows=1)
if pred_array.size == 0:
    raise SystemExit(f"Prediction CSV contains no data rows: {pred_csv}")
if pred_array.ndim == 1:
    pred_array = pred_array[None, :]
if pred_array.shape[1] < 28:
    raise SystemExit(
        f"Prediction CSV has {pred_array.shape[1]} columns; expected at least 28."
    )

timestamps = pred_array[:, 0].astype(np.float64)
pred_joints = pred_array[:, 2:28].reshape(-1, 13, 2).astype(np.float64)

joint_names = list(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys())
if len(joint_names) != 13:
    raise SystemExit(f"Expected 13 HPE-core joints, found {len(joint_names)}")

gt_data = ds_parsing.import_yarp_skeleton_data(
    gt_file,
    multi_channel=False,
)

ts_gt_raw = np.asarray(gt_data["ts"], dtype=np.float64)
if ts_gt_raw.size == 0:
    raise SystemExit(f"GT file contains no timestamps: {gt_file}")

# Match the interpolation convention used by the evaluation notebook.
ts_gt = np.concatenate(([0.0], ts_gt_raw, [ts_gt_raw[-1] + 1.0]))
gt_joints = np.zeros((len(timestamps), 13, 2), dtype=np.float64)

for joint_index, joint_name in enumerate(joint_names):
    xy = np.asarray(gt_data[joint_name], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise SystemExit(f"Unexpected GT shape for {joint_name}: {xy.shape}")

    x = np.concatenate(([xy[0, 0]], xy[:, 0], [xy[-1, 0]]))
    y = np.concatenate(([xy[0, 1]], xy[:, 1], [xy[-1, 1]]))
    gt_joints[:, joint_index, 0] = np.interp(timestamps, ts_gt, x)
    gt_joints[:, joint_index, 1] = np.interp(timestamps, ts_gt, y)

capture = cv2.VideoCapture(str(base_video))
if not capture.isOpened():
    raise SystemExit(f"Could not open base video: {base_video}")

fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

if not np.isfinite(fps) or fps <= 0:
    if len(timestamps) > 1:
        median_dt = float(np.median(np.diff(timestamps)))
        fps = 1.0 / median_dt if median_dt > 0 else 200.0
    else:
        fps = 200.0

writer = cv2.VideoWriter(
    str(final_video),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
if not writer.isOpened():
    capture.release()
    raise SystemExit(f"Could not open final video writer: {final_video}")


def valid_point(point: np.ndarray) -> bool:
    if point.shape[0] < 2 or not np.isfinite(point[:2]).all():
        return False
    x, y = float(point[0]), float(point[1])
    return 0.0 <= x < width and 0.0 <= y < height


def as_int_point(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_skeleton(
    image: np.ndarray,
    joints: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    # Black outline improves visibility and covers most of the prediction
    # already drawn in the intermediate C++ video.
    for a, b in EDGES:
        if valid_point(joints[a]) and valid_point(joints[b]):
            pa = as_int_point(joints[a])
            pb = as_int_point(joints[b])
            cv2.line(image, pa, pb, OUTLINE_COLOR, 3, cv2.LINE_AA)
            cv2.line(image, pa, pb, color, 1, cv2.LINE_AA)

    for point in joints:
        if valid_point(point):
            p = as_int_point(point)
            cv2.circle(image, p, 3, OUTLINE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(image, p, 1, color, -1, cv2.LINE_AA)


def draw_legend(image: np.ndarray, timestamp: float) -> None:
    cv2.rectangle(image, (4, 4), (132, 43), (0, 0, 0), -1)

    cv2.line(image, (10, 16), (25, 16), GT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        image,
        "GT",
        (31, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.line(image, (10, 32), (25, 32), PRED_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        image,
        pose_label,
        (31, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"t={timestamp:.3f}s",
        (max(4, width - 86), height - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


frame_index = 0
while frame_index < len(timestamps):
    ok, frame = capture.read()
    if not ok:
        break

    # GT is drawn first, prediction second, so the pose remains visible where the
    # two skeletons overlap.
    draw_skeleton(frame, gt_joints[frame_index], GT_COLOR)
    draw_skeleton(frame, pred_joints[frame_index], PRED_COLOR)
    draw_legend(frame, float(timestamps[frame_index]))

    writer.write(frame)
    frame_index += 1

capture.release()
writer.release()

if frame_index == 0:
    raise SystemExit("No video frames were written.")

if reported_frames > 0 and reported_frames != len(timestamps):
    print(
        "Warning: base video reports "
        f"{reported_frames} frames while CSV contains {len(timestamps)} rows. "
        f"Wrote {frame_index} synchronized frames."
    )
elif frame_index != len(timestamps):
    print(f"Warning: wrote {frame_index} frames for {len(timestamps)} CSV rows.")

print(f"Final synchronized frames: {frame_index}")
print(f"Final video FPS: {fps:g}")
print(f"Final video: {final_video}")
PY

if [[ ! -s "$FINAL_VIDEO" ]]; then
  echo "Final overlaid video was not created: $FINAL_VIDEO" >&2
  exit 1
fi

if [[ "$KEEP_BASE_VIDEO" != "true" ]]; then
  rm -f "$BASE_VIDEO"
fi

echo
echo "Completed."
echo "Final video : $FINAL_VIDEO"
echo "Prediction  : $PRED_CSV"
echo "Run log     : $RUN_LOG"
echo "GPU samples : $GPU_CSV"
