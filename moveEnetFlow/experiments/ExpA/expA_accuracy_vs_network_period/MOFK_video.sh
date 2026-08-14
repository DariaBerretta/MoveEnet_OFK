#!/usr/bin/env bash
set -euo pipefail

# Generate first-sequence videos for MoveEnetOFK and/or MoveEnetOnly.
# Supported datasets:
#   --dataset dhp19
#   --dataset eh36m

DATASET="dhp19"
BINARY="/workspace/moveEnetFlow/build2/moveEnetOFK_offline"
RESULTS_ROOT="${MOVENET_RESULTS_ROOT:-/data/MovEnet_OFK_results}"

# Empty values are filled by the selected dataset preset after argument parsing.
DATA_ROOT=""
DATA_FILE=""
DATA_GLOB=""
GT_ROOT=""
GT_FILE=""
CHECKPOINT_PATH=""
OUTPUT_DIR=""
IMG_W=""
IMG_H=""

MODE="both"                         # both | ofk | moveenet_only
NET_PERIOD="0.02"
FLOW_PERIOD="0.005"
OUTPUT_PERIOD="0.005"
DEVICE="cuda:0"
GPU_PERIOD_MS="5"
GPU_INDEX=""

# MoveEnetOFK/KF parameters.
PROC_U="0.77"
MEAS_U_D="0.06"
MEAS_U_V="0.97"
ROI_SIZE="20"
USE_LC="true"

KEEP_BASE_VIDEO="false"
FORCE_RESTART_MOVENET="false"
PRINT_CONFIG="false"

usage() {
  cat <<USAGE
Usage: $(basename "$0") --dataset <dhp19|eh36m> [options]

Dataset:
  --dataset <dhp19|eh36m>       Dataset preset. Default: $DATASET

Input overrides:
  --data_root <path>            Override dataset event root
  --data_file <path>            Use a specific chXdvs/data.log
  --data_glob <pattern>         Override event discovery pattern
  --gt_root <path>              Optional root containing GT sequence folders
  --gt_file <path>              Use a specific GT data.log

Execution:
  --binary <path>               moveEnetOFK_offline executable
  --checkpoint_path <path>      Override dataset MoveNet checkpoint
  --mode <mode>                 both, ofk, or moveenet_only. Default: $MODE
  --net_period <seconds>        MoveNet inference period. Default: $NET_PERIOD
  --flow_period <seconds>       OF/KF update period. Default: $FLOW_PERIOD
  --output_period <seconds>     CSV/video period. Default: $OUTPUT_PERIOD
  --device <device>             cpu, cuda:0, cuda:1, ... Default: $DEVICE
  --w <pixels>                  Override dataset image width
  --h <pixels>                  Override dataset image height

OF/KF parameters:
  --pu <value>                  Process uncertainty. Default: $PROC_U
  --muD <value>                 Position uncertainty. Default: $MEAS_U_D
  --muV <value>                 Velocity uncertainty. Default: $MEAS_U_V
  --roi <pixels>                Optical-flow ROI size. Default: $ROI_SIZE
  --use_lc <true|false>         Latency compensation. Default: $USE_LC

Output / safety:
  --output_dir <path>           Destination directory
  --gpu_period_ms <int>         GPU sampling period. Default: $GPU_PERIOD_MS
  --gpu_index <int>             GPU monitor index; inferred if omitted
  --keep_base_video             Keep intermediate C++ videos
  --force_restart_movenet       Allow killing an existing movenet_online.py
  --print_config                Print resolved paths and exit
  --help                        Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="${2,,}"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --data_file) DATA_FILE="$2"; shift 2 ;;
    --data_glob) DATA_GLOB="$2"; shift 2 ;;
    --gt_root) GT_ROOT="$2"; shift 2 ;;
    --gt_file) GT_FILE="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --net_period) NET_PERIOD="$2"; shift 2 ;;
    --flow_period) FLOW_PERIOD="$2"; shift 2 ;;
    --output_period) OUTPUT_PERIOD="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --w) IMG_W="$2"; shift 2 ;;
    --h) IMG_H="$2"; shift 2 ;;
    --pu) PROC_U="$2"; shift 2 ;;
    --muD) MEAS_U_D="$2"; shift 2 ;;
    --muV) MEAS_U_V="$2"; shift 2 ;;
    --roi) ROI_SIZE="$2"; shift 2 ;;
    --use_lc) USE_LC="$2"; shift 2 ;;
    --gpu_period_ms) GPU_PERIOD_MS="$2"; shift 2 ;;
    --gpu_index) GPU_INDEX="$2"; shift 2 ;;
    --keep_base_video) KEEP_BASE_VIDEO="true"; shift ;;
    --force_restart_movenet) FORCE_RESTART_MOVENET="true"; shift ;;
    --print_config) PRINT_CONFIG="true"; shift ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Dataset presets are applied only after parsing, so explicit command-line
# overrides work regardless of the position of --dataset.
DATASET_ARGS=()
case "$DATASET" in
  dhp19)
    DATASET_LABEL="DHP19"

    if [[ -z "$DATA_ROOT" ]]; then
      if [[ -d /data/dhp19_testing_set_S13toS17 ]]; then
        DATA_ROOT="/data/dhp19_testing_set_S13toS17"
      else
        DATA_ROOT="/data/DHP19_subset/raw"
      fi
    fi

    DATA_GLOB="${DATA_GLOB:-*/ch*dvs/data.log}"
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-/usr/local/src/hpe-core/example/movenet/models/dhp19_allcams_e33_valacc0.87996.pth}"
    IMG_W="${IMG_W:-346}"
    IMG_H="${IMG_H:-260}"
    OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_ROOT/MOFK_dhp19_full_test/videos}"

    # The binary uses a 346x260 canvas and a padded 352x260 MoveNet transport.
    DATASET_ARGS=(--dhp19)
    ;;

  eh36m)
    DATASET_LABEL="eH36M"
    DATA_ROOT="${DATA_ROOT:-/data/eh36m_testing_set_S9S11/events}"
    DATA_GLOB="${DATA_GLOB:-*/ch0dvs/data.log}"
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth}"
    IMG_W="${IMG_W:-640}"
    IMG_H="${IMG_H:-480}"
    OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_ROOT/MOFK_eh36m_full_test/videos}"

    # Important: eH36M must NOT receive --dhp19.
    DATASET_ARGS=(--w "$IMG_W" --h "$IMG_H")
    ;;

  *)
    echo "Invalid --dataset: $DATASET. Use dhp19 or eh36m." >&2
    exit 1
    ;;
esac

case "$MODE" in
  both|ofk|moveenet_only) ;;
  *)
    echo "Invalid --mode: $MODE. Use both, ofk, or moveenet_only." >&2
    exit 1
    ;;
esac

case "$USE_LC" in
  true|false) ;;
  *)
    echo "Invalid --use_lc: $USE_LC. Use true or false." >&2
    exit 1
    ;;
esac

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found in PATH." >&2
  exit 1
fi

if command -v yarp >/dev/null 2>&1; then
  if ! yarp where >/dev/null 2>&1; then
    echo "YARP name server is unavailable. Start yarpserver, then rerun." >&2
    exit 1
  fi
fi

# Protect any MoveNet experiment already using the globally named sidecar.
if pgrep -f '[m]ovenet_online.py' >/dev/null 2>&1; then
  if [[ "$FORCE_RESTART_MOVENET" != "true" ]]; then
    cat >&2 <<'WARNING'
A movenet_online.py process is already running.

moveEnetOFK_offline will kill that process and unregister the global
/movenet/img:i and /movenet/sklt:o ports. Stop the other MoveNet experiment
first, or rerun with --force_restart_movenet only if interrupting it is safe.
WARNING
    exit 1
  fi
fi

if [[ -z "$DATA_FILE" ]]; then
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "$DATASET_LABEL event root not found: $DATA_ROOT" >&2
    exit 1
  fi
  DATA_FILE="$(find "$DATA_ROOT" -type f -path "$DATA_GLOB" | sort | head -n 1 || true)"
fi

if [[ -z "$DATA_FILE" || ! -f "$DATA_FILE" ]]; then
  echo "No $DATASET_LABEL event data.log found." >&2
  echo "Data root: $DATA_ROOT" >&2
  echo "Pattern:   $DATA_GLOB" >&2
  exit 1
fi

CHANNEL_DIR="$(basename "$(dirname "$DATA_FILE")")"
SEQUENCE_PATH="$(dirname "$(dirname "$DATA_FILE")")"
SEQUENCE_DIR="$(basename "$SEQUENCE_PATH")"

if [[ ! "$CHANNEL_DIR" =~ ^ch([0-9]+)dvs$ ]]; then
  echo "Unexpected event folder: $CHANNEL_DIR" >&2
  echo "Expected a folder named chXdvs, for example ch0dvs or ch3dvs." >&2
  exit 1
fi
CHANNEL_NUMBER="${BASH_REMATCH[1]}"

find_ground_truth() {
  local event_sequence="$1"
  local channel_number="$2"
  local rel_sequence=""
  local dataset_parent=""
  local candidate=""
  local search_root=""
  local -a candidates=()
  local -a roots=()

  # Most converted datasets keep events and GT beside each other.
  candidates+=(
    "$event_sequence/ch${channel_number}GT200Hzskeleton/data.log"
    "$event_sequence/ch${channel_number}GT50Hzskeleton/data.log"
  )

  if [[ -n "$GT_ROOT" ]]; then
    roots+=("$GT_ROOT")
  fi

  # eH36M installations often separate events and labels under sibling roots.
  if [[ "$DATASET" == "eh36m" ]]; then
    dataset_parent="$(dirname "$DATA_ROOT")"
    rel_sequence="$(realpath --relative-to="$DATA_ROOT" "$event_sequence" 2>/dev/null || true)"

    roots+=(
      "$dataset_parent"
      "$dataset_parent/labels"
      "$dataset_parent/ground_truth"
      "$dataset_parent/gt"
      "$dataset_parent/skeletons"
      "$dataset_parent/poses"
      "$dataset_parent/annotations"
    )

    if [[ -n "$rel_sequence" && "$rel_sequence" != ..* ]]; then
      for search_root in "${roots[@]}"; do
        candidates+=(
          "$search_root/$rel_sequence/ch${channel_number}GT200Hzskeleton/data.log"
          "$search_root/$rel_sequence/ch${channel_number}GT50Hzskeleton/data.log"
        )
      done
    fi
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  # Last fallback: search likely roots for the same sequence and channel.
  if [[ ${#roots[@]} -eq 0 ]]; then
    roots+=("$(dirname "$DATA_ROOT")")
  fi

  for search_root in "${roots[@]}"; do
    [[ -d "$search_root" ]] || continue
    candidate="$(
      find "$search_root" -type f \
        \( -path "*/${SEQUENCE_DIR}/ch${channel_number}GT200Hzskeleton/data.log" \
           -o -path "*/${SEQUENCE_DIR}/ch${channel_number}GT50Hzskeleton/data.log" \) \
        2>/dev/null | sort | head -n 1 || true
    )"
    if [[ -n "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

if [[ -n "$GT_FILE" ]]; then
  if [[ ! -f "$GT_FILE" ]]; then
    echo "Ground-truth file not found: $GT_FILE" >&2
    exit 1
  fi
else
  GT_FILE="$(find_ground_truth "$SEQUENCE_PATH" "$CHANNEL_NUMBER" || true)"
fi

if [[ -z "$GT_FILE" || ! -f "$GT_FILE" ]]; then
  echo "$DATASET_LABEL ground-truth data.log was not found." >&2
  echo "Event file: $DATA_FILE" >&2
  echo "Looked for ch${CHANNEL_NUMBER}GT200Hzskeleton/data.log and ch${CHANNEL_NUMBER}GT50Hzskeleton/data.log." >&2
  echo "For a custom layout, pass --gt_file /path/to/data.log or --gt_root /path/to/gt/root." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

SAFE_NET="${NET_PERIOD//./p}"
SAFE_FLOW="${FLOW_PERIOD//./p}"
SAFE_OUT="${OUTPUT_PERIOD//./p}"
STEM="${DATASET}_${SEQUENCE_DIR}__${CHANNEL_DIR}_np_${SAFE_NET}_fp_${SAFE_FLOW}_op_${SAFE_OUT}"

COMMON_ARGS=(
  --data_file "$DATA_FILE"
  --checkpoint_path "$CHECKPOINT_PATH"
  "${DATASET_ARGS[@]}"
  --net_period "$NET_PERIOD"
  --flow_period "$FLOW_PERIOD"
  --output_period "$OUTPUT_PERIOD"
  --device "$DEVICE"
  --pu "$PROC_U"
  --muD "$MEAS_U_D"
  --muV "$MEAS_U_V"
  --roi "$ROI_SIZE"
  --use_lc "$USE_LC"
  --gpu_period_ms "$GPU_PERIOD_MS"
)

if [[ -n "$GPU_INDEX" ]]; then
  COMMON_ARGS+=(--gpu_index "$GPU_INDEX")
fi

if [[ "$PRINT_CONFIG" == "true" ]]; then
  echo "Resolved configuration"
  echo "Dataset       : $DATASET_LABEL ($DATASET)"
  echo "Data root     : $DATA_ROOT"
  echo "Data glob     : $DATA_GLOB"
  echo "Event file    : $DATA_FILE"
  echo "GT file       : $GT_FILE"
  echo "Checkpoint    : $CHECKPOINT_PATH"
  echo "Image size    : ${IMG_W}x${IMG_H}"
  echo "Dataset args  : ${DATASET_ARGS[*]:-(none)}"
  echo "Output dir    : $OUTPUT_DIR"
  echo "Mode          : $MODE"
  exit 0
fi

run_mode() {
  local mode="$1"
  local label
  local -a flag

  case "$mode" in
    ofk)
      label="MoveEnetOFK"
      flag=()
      ;;
    moveenet_only)
      label="MoveEnetOnly"
      flag=(--moveenet_only)
      ;;
    *)
      echo "Internal error: invalid mode $mode" >&2
      return 1
      ;;
  esac

  local pred_csv="$OUTPUT_DIR/${STEM}_${mode}_prediction.csv"
  local base_video="$OUTPUT_DIR/${STEM}_${mode}_base.mp4"
  local final_video="$OUTPUT_DIR/${STEM}_${mode}_events_gt_prediction.mp4"
  local run_log="$OUTPUT_DIR/${STEM}_${mode}.log"
  local gpu_csv="$OUTPUT_DIR/${STEM}_${mode}_gpu.csv"

  echo
  echo "Running $label"
  echo "  Event file : $DATA_FILE"
  echo "  GT file    : $GT_FILE"
  echo "  Prediction : $pred_csv"
  echo "  Final video: $final_video"
  echo "  Log        : $run_log"

  "$BINARY" \
    "${COMMON_ARGS[@]}" \
    "${flag[@]}" \
    --output_csv_f "$pred_csv" \
    --output_video "$base_video" \
    --gpu_file "$gpu_csv" \
    >"$run_log" 2>&1

  if [[ ! -s "$pred_csv" ]]; then
    echo "$label prediction CSV is missing or empty: $pred_csv" >&2
    echo "Inspect: $run_log" >&2
    return 1
  fi

  if [[ ! -s "$base_video" ]]; then
    echo "$label base video is missing or empty: $base_video" >&2
    echo "Inspect: $run_log" >&2
    return 1
  fi

  echo "  Removing the C++ skeleton overlay and adding compact GT/prediction..."

  python3 - "$base_video" "$pred_csv" "$GT_FILE" "$final_video" "$label" "$DATASET_LABEL" <<'PY'
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
        "Could not import datasets.utils from hpe-core. Run this script inside "
        f"the MoveEnet experiment environment. Import error: {exc!r}"
    )

base_video = Path(sys.argv[1])
pred_csv = Path(sys.argv[2])
gt_file = Path(sys.argv[3])
final_video = Path(sys.argv[4])
method_label = sys.argv[5]
dataset_label = sys.argv[6]

# HPE-core skeleton13 order.
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

# Compact rendering parameters. Change these values to resize the overlay.
STICK_OUTLINE_THICKNESS = 3
STICK_COLOR_THICKNESS = 1
JOINT_OUTLINE_RADIUS = 3
JOINT_COLOR_RADIUS = 2
LEGEND_FONT_SCALE = 0.31
TIMESTAMP_FONT_SCALE = 0.30

pred_array = np.loadtxt(pred_csv, delimiter=",", skiprows=1)
if pred_array.size == 0:
    raise SystemExit(f"Prediction CSV contains no rows: {pred_csv}")
if pred_array.ndim == 1:
    pred_array = pred_array[None, :]
if pred_array.shape[1] < 28:
    raise SystemExit(
        f"Prediction CSV has {pred_array.shape[1]} columns; expected at least 28."
    )

pred_timestamps = pred_array[:, 0].astype(np.float64)
pred_joints = pred_array[:, 2:28].reshape(-1, 13, 2).astype(np.float64)

joint_names = list(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys())
if len(joint_names) != 13:
    raise SystemExit(f"Expected 13 GT joints, found {len(joint_names)}")

gt_data = ds_parsing.import_yarp_skeleton_data(gt_file, multi_channel=False)
ts_gt_raw = np.asarray(gt_data["ts"], dtype=np.float64)
if ts_gt_raw.size == 0:
    raise SystemExit(f"GT file contains no timestamps: {gt_file}")

# Extend endpoints so np.interp also covers prediction samples just outside the
# original GT timestamp interval.
ts_gt = np.concatenate(([0.0], ts_gt_raw, [ts_gt_raw[-1] + 1.0]))
gt_joints = np.zeros((len(pred_timestamps), 13, 2), dtype=np.float64)

for joint_index, joint_name in enumerate(joint_names):
    xy = np.asarray(gt_data[joint_name], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise SystemExit(f"Unexpected GT shape for {joint_name}: {xy.shape}")

    x = np.concatenate(([xy[0, 0]], xy[:, 0], [xy[-1, 0]]))
    y = np.concatenate(([xy[0, 1]], xy[:, 1], [xy[-1, 1]]))
    gt_joints[:, joint_index, 0] = np.interp(pred_timestamps, ts_gt, x)
    gt_joints[:, joint_index, 1] = np.interp(pred_timestamps, ts_gt, y)

capture = cv2.VideoCapture(str(base_video))
if not capture.isOpened():
    raise SystemExit(f"Could not open base video: {base_video}")

fps = float(capture.get(cv2.CAP_PROP_FPS))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

if not np.isfinite(fps) or fps <= 0:
    if len(pred_timestamps) > 1:
        median_dt = float(np.median(np.diff(pred_timestamps)))
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
    raise SystemExit(f"Could not create final video: {final_video}")


def valid_point(point: np.ndarray) -> bool:
    if point.shape[0] < 2 or not np.isfinite(point[:2]).all():
        return False
    x, y = float(point[0]), float(point[1])
    return 0.0 <= x < width and 0.0 <= y < height


def as_int_point(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def remove_cpp_skeletons(frame: np.ndarray) -> np.ndarray:
    """Remove blue filtered and red raw skeletons drawn by C++.

    The EROS background is grayscale, while the C++ skeletons are saturated
    blue/red. HSV masking therefore removes them without erasing normal events.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blue = cv2.inRange(hsv, np.array([90, 65, 35]), np.array([145, 255, 255]))
    red_low = cv2.inRange(hsv, np.array([0, 65, 35]), np.array([15, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([165, 65, 35]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(blue, cv2.bitwise_or(red_low, red_high))

    # Cover anti-aliased borders and joint circles before inpainting.
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if cv2.countNonZero(mask) == 0:
        return frame
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)


def draw_skeleton(
    image: np.ndarray,
    joints: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    for a, b in EDGES:
        if valid_point(joints[a]) and valid_point(joints[b]):
            pa = as_int_point(joints[a])
            pb = as_int_point(joints[b])
            cv2.line(
                image, pa, pb, OUTLINE_COLOR,
                STICK_OUTLINE_THICKNESS, cv2.LINE_AA,
            )
            cv2.line(
                image, pa, pb, color,
                STICK_COLOR_THICKNESS, cv2.LINE_AA,
            )

    for point in joints:
        if valid_point(point):
            p = as_int_point(point)
            cv2.circle(
                image, p, JOINT_OUTLINE_RADIUS,
                OUTLINE_COLOR, -1, cv2.LINE_AA,
            )
            cv2.circle(
                image, p, JOINT_COLOR_RADIUS,
                color, -1, cv2.LINE_AA,
            )


def draw_legend(image: np.ndarray, timestamp: float) -> None:
    # Compact two-line legend. Width is adjusted for the method name.
    box_width = 164 if method_label == "MoveEnetOFK" else 174
    cv2.rectangle(image, (4, 4), (box_width, 43), (0, 0, 0), -1)

    cv2.line(image, (10, 16), (25, 16), GT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        image, f"{dataset_label} GT", (31, 19), cv2.FONT_HERSHEY_SIMPLEX,
        LEGEND_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA,
    )

    cv2.line(image, (10, 32), (25, 32), PRED_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        image, method_label, (31, 35), cv2.FONT_HERSHEY_SIMPLEX,
        LEGEND_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA,
    )

    cv2.putText(
        image, f"t={timestamp:.3f}s", (max(4, width - 82), height - 7),
        cv2.FONT_HERSHEY_SIMPLEX, TIMESTAMP_FONT_SCALE,
        (255, 255, 255), 1, cv2.LINE_AA,
    )


frame_index = 0
while frame_index < len(pred_timestamps):
    ok, frame = capture.read()
    if not ok:
        break

    frame = remove_cpp_skeletons(frame)

    # Draw GT first, prediction second, so the selected model remains visible
    # when both poses overlap.
    draw_skeleton(frame, gt_joints[frame_index], GT_COLOR)
    draw_skeleton(frame, pred_joints[frame_index], PRED_COLOR)
    draw_legend(frame, float(pred_timestamps[frame_index]))

    writer.write(frame)
    frame_index += 1

capture.release()
writer.release()

if frame_index == 0:
    raise SystemExit("No synchronized frames were written.")

if reported_frames > 0 and reported_frames != len(pred_timestamps):
    print(
        "Warning: base video reports "
        f"{reported_frames} frames, prediction CSV has {len(pred_timestamps)} rows; "
        f"wrote {frame_index} synchronized frames."
    )
elif frame_index != len(pred_timestamps):
    print(
        f"Warning: wrote {frame_index} frames for "
        f"{len(pred_timestamps)} prediction rows."
    )

print(f"Method: {method_label}")
print(f"Frames: {frame_index}")
print(f"FPS: {fps:g}")
print(f"Final video: {final_video}")
PY

  if [[ ! -s "$final_video" ]]; then
    echo "$label final video was not created: $final_video" >&2
    return 1
  fi

  if [[ "$KEEP_BASE_VIDEO" != "true" ]]; then
    rm -f "$base_video"
  fi

  echo "  Completed: $final_video"

  # The binary normally terminates its sidecar. Wait briefly before starting
  # the second mode, so fixed YARP ports are fully unregistered.
  for _ in $(seq 1 20); do
    if ! pgrep -f '[m]ovenet_online.py' >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
}

echo "MoveEnet first-sequence video generation"
echo "Dataset       : $DATASET_LABEL ($DATASET)"
echo "Data root     : $DATA_ROOT"
echo "Event file    : $DATA_FILE"
echo "GT file       : $GT_FILE"
echo "Mode          : $MODE"
echo "Net period    : $NET_PERIOD s"
echo "Flow period   : $FLOW_PERIOD s"
echo "Output period : $OUTPUT_PERIOD s"
echo "Device        : $DEVICE"
echo "Output dir    : $OUTPUT_DIR"

case "$MODE" in
  both)
    run_mode ofk
    run_mode moveenet_only
    ;;
  ofk)
    run_mode ofk
    ;;
  moveenet_only)
    run_mode moveenet_only
    ;;
esac

echo
echo "Completed."
if [[ "$MODE" == "both" || "$MODE" == "ofk" ]]; then
  echo "MoveEnetOFK video : $OUTPUT_DIR/${STEM}_ofk_events_gt_prediction.mp4"
fi
if [[ "$MODE" == "both" || "$MODE" == "moveenet_only" ]]; then
  echo "MoveEnetOnly video: $OUTPUT_DIR/${STEM}_moveenet_only_events_gt_prediction.mp4"
fi