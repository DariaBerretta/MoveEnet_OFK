#!/usr/bin/env bash
set -euo pipefail

# Run EventPointPose on one DHP19 entry and save:
#   1. a direct C++ video containing EventPointPose-view event frames and the
#      prediction with the exact same network-trigger timestamp;
#   2. an optional post-processed video with DHP19 GT in green and the
#      EventPointPose prediction in magenta.
#
# This version is compatible with the current EventPointPose files:
#   - one precomputed hot-pixel mask (.npy) per entry;
#   - static hot-pixel mask enabled;
#   - adaptive hot-pixel filtering disabled;
#   - background and IR filters enabled in the Python sidecar;
#   - 7500-event rolling FIFO and 2048 RasEPC points;
#   - an independent video clock controlled by --output_period;
#   - no second image flip: DHP19 data.log coordinates are already transformed;
#   - GT coordinates are already stored as OpenCV/hpe-core (x,y);
#   - no GT axis swap or spatial flip is applied.

BINARY="/workspace/moveEnetFlow/build2/eventPointPose_offline"
DATA_ROOT="/data/dhp19_testing_set_S13toS17"
DATA_FILE=""
DATA_GLOB="*/ch[23]dvs/data.log"

HOTPIXEL_ROOT="/workspace/moveEnetFlow/experiments/ExpA/EPP_expA_accuracy_vs_network_period/dhp19_full_test/hotpixel_masks"
HOTPIXEL_MASK=""

CHECKPOINT_PATH="/workspace/model_mounts/eventpointpose/PointNet/models/model.pth"
EPP_SCRIPT="/workspace/model_mounts/eventpointpose/PointNet/models/eventPointPose_yarp_server.py"

OUTPUT_DIR="$(pwd)/epp_single_entry_video"
NET_PERIOD="0.1"
OUTPUT_PERIOD="0.005"
DEVICE="cuda:0"
IMG_W="346"
IMG_H="260"
NUM_POINTS="2048"
EVENTS_PER_WINDOW="7500"
MAX_PACKETS="-1"
GPU_PERIOD_MS="5"
SWAP_LR="false"
KEEP_BASE_VIDEO="true"
WITH_GT_OVERLAY="true"
RUN_SELFTEST="true"
DETERMINISTIC_DEBUG="false"
INPUT_COORDINATES="raw_zero_based"
OUTPUT_COORDINATES="raw_flip_x"
SAMPLING_MODE="random"
DETERMINISTIC_SAMPLING_SEED="0"
DEBUG_HASH_EVERY_WINDOWS="0"
PRINT_CHECKPOINT_SUMMARY="false"
COMPUTE_METRICS="true"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Input selection:
  --data_root <path>              DHP19 root. Default: $DATA_ROOT
  --data_file <path>              Run this single ch2dvs/ch3dvs data.log
  --data_glob <pattern>           Auto-selection pattern. Default: $DATA_GLOB

Hot-pixel mask:
  --hotpixel_root <path>          Root containing <sequence>/<channel>/hotpixel_mask.npy
  --hotpixel_mask <path>          Explicit .npy mask; overrides automatic lookup

EventPointPose:
  --binary <path>                 eventPointPose_offline executable
  --checkpoint_path <path>        EventPointPose checkpoint
  --epp_script <path>             eventPointPose_yarp_server.py
  --net_period <seconds>          Event-reading period. Default: $NET_PERIOD
  --output_period <seconds>       Base-video frame period. Default: $OUTPUT_PERIOD
  --device <device>               PyTorch device. Default: $DEVICE
  --num_points <int>              RasEPC points. Default: $NUM_POINTS
  --events_per_window <int>       Rolling FIFO capacity. Default: $EVENTS_PER_WINDOW
  --events_per_block <int>        Legacy alias for --events_per_window
  --swap_lr                       Pass --swap_lr to the sidecar
  --input_coordinates <mode>      raw_zero_based or raw_minus_one_legacy
  --output_coordinates <mode>     epp | raw_flip_x | raw_dhp19
  --sampling_mode <mode>          random | deterministic_window
  --det_sampling_seed <int>       Seed for deterministic_window sampling
  --debug_hash_every <int>        Print sidecar hash diagnostics every N windows
  --print_checkpoint_summary      Print sidecar checkpoint summary/fingerprint

Output / debug:
  --output_dir <path>             Output directory
  --max_packets <int>             Debug limit; -1 processes the full entry
  --skip_selftest                 Skip RasEPC numeric self-check before run
  --deterministic_debug           Force deterministic sampling + hash logs
  --skip_metrics                  Skip GT-vs-pred MPJPE/PCK report
  --prediction_only               Do not create the GT-overlay video
  --remove_base_video             Delete the direct C++ video after GT overlay
  --help                          Show this help

Examples:
  $(basename "$0") --data_file /data/.../S13_1_1/ch2dvs/data.log --max_packets 500
  $(basename "$0") --data_file /data/.../S13_1_1/ch3dvs/data.log
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --data_file) DATA_FILE="$2"; shift 2 ;;
    --data_glob) DATA_GLOB="$2"; shift 2 ;;
    --hotpixel_root) HOTPIXEL_ROOT="$2"; shift 2 ;;
    --hotpixel_mask) HOTPIXEL_MASK="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --epp_script) EPP_SCRIPT="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --net_period) NET_PERIOD="$2"; shift 2 ;;
    --output_period) OUTPUT_PERIOD="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --num_points) NUM_POINTS="$2"; shift 2 ;;
    --events_per_window) EVENTS_PER_WINDOW="$2"; shift 2 ;;
    --events_per_block) EVENTS_PER_WINDOW="$2"; shift 2 ;;
    --max_packets) MAX_PACKETS="$2"; shift 2 ;;
    --swap_lr) SWAP_LR="true"; shift ;;
    --input_coordinates) INPUT_COORDINATES="$2"; shift 2 ;;
    --output_coordinates) OUTPUT_COORDINATES="$2"; shift 2 ;;
    --sampling_mode) SAMPLING_MODE="$2"; shift 2 ;;
    --det_sampling_seed) DETERMINISTIC_SAMPLING_SEED="$2"; shift 2 ;;
    --debug_hash_every) DEBUG_HASH_EVERY_WINDOWS="$2"; shift 2 ;;
    --print_checkpoint_summary) PRINT_CHECKPOINT_SUMMARY="true"; shift ;;
    --skip_selftest) RUN_SELFTEST="false"; shift ;;
    --deterministic_debug) DETERMINISTIC_DEBUG="true"; shift ;;
    --skip_metrics) COMPUTE_METRICS="false"; shift ;;
    --prediction_only) WITH_GT_OVERLAY="false"; shift ;;
    --remove_base_video) KEEP_BASE_VIDEO="false"; shift ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Deterministic mode is for debugging: unchanged FIFO window should produce
# unchanged sampled tensor and therefore unchanged prediction.
if [[ "$DETERMINISTIC_DEBUG" == "true" ]]; then
  SAMPLING_MODE="deterministic_window"
  DEBUG_HASH_EVERY_WINDOWS="1"
  PRINT_CHECKPOINT_SUMMARY="true"
fi

case "$INPUT_COORDINATES" in
  raw_zero_based|raw_minus_one_legacy) ;;
  *)
    echo "Invalid --input_coordinates: $INPUT_COORDINATES" >&2
    exit 1
    ;;
esac

case "$OUTPUT_COORDINATES" in
  epp|raw_flip_x|raw_dhp19) ;;
  *)
    echo "Invalid --output_coordinates: $OUTPUT_COORDINATES" >&2
    exit 1
    ;;
esac

case "$SAMPLING_MODE" in
  random|deterministic_window) ;;
  *)
    echo "Invalid --sampling_mode: $SAMPLING_MODE" >&2
    exit 1
    ;;
esac

# -----------------------------------------------------------------------------
# Validate programs and input selection
# -----------------------------------------------------------------------------
if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ ! -f "$EPP_SCRIPT" ]]; then
  echo "EventPointPose sidecar not found: $EPP_SCRIPT" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found in PATH." >&2
  exit 1
fi

if command -v yarp >/dev/null 2>&1; then
  if ! yarp where >/dev/null 2>&1; then
    echo "YARP name server is unavailable. Start yarpserver and retry." >&2
    exit 1
  fi
fi

if [[ -z "$DATA_FILE" ]]; then
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "DHP19 data root not found: $DATA_ROOT" >&2
    exit 1
  fi

  DATA_FILE="$(find "$DATA_ROOT" -type f -path "$DATA_GLOB" | sort | head -n 1 || true)"
fi

if [[ -z "$DATA_FILE" || ! -f "$DATA_FILE" ]]; then
  echo "No DHP19 data.log found." >&2
  echo "Data root: $DATA_ROOT" >&2
  echo "Pattern:   $DATA_GLOB" >&2
  exit 1
fi

DATA_FILE="$(realpath "$DATA_FILE")"
CHANNEL_DIR="$(basename "$(dirname "$DATA_FILE")")"
SEQUENCE_DIR="$(basename "$(dirname "$(dirname "$DATA_FILE")")")"

case "$CHANNEL_DIR" in
  ch2dvs) CAMERA_INDEX="2" ;;
  ch3dvs) CAMERA_INDEX="3" ;;
  *)
    echo "Unsupported EventPointPose entry: $CHANNEL_DIR" >&2
    echo "Use a ch2dvs or ch3dvs data.log." >&2
    exit 1
    ;;
esac

# -----------------------------------------------------------------------------
# Resolve the mask corresponding to this exact dataset entry
# -----------------------------------------------------------------------------
if [[ -z "$HOTPIXEL_MASK" ]]; then
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Cannot infer the mask because data_root does not exist: $DATA_ROOT" >&2
    echo "Pass --hotpixel_mask explicitly." >&2
    exit 1
  fi

  DATA_ROOT_REAL="$(realpath "$DATA_ROOT")"
  REL_DATA="$(realpath --relative-to="$DATA_ROOT_REAL" "$DATA_FILE")"
  REL_ENTRY="${REL_DATA%/data.log}"
  HOTPIXEL_MASK="$HOTPIXEL_ROOT/$REL_ENTRY/hotpixel_mask.npy"
fi

if [[ ! -f "$HOTPIXEL_MASK" ]]; then
  echo "Hot-pixel mask not found: $HOTPIXEL_MASK" >&2
  echo "Expected layout: <hotpixel_root>/<sequence>/<channel>/hotpixel_mask.npy" >&2
  exit 1
fi

HOTPIXEL_MASK="$(realpath "$HOTPIXEL_MASK")"

# Validate the NPY before starting the expensive run.
python3 - "$HOTPIXEL_MASK" "$IMG_W" "$IMG_H" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
mask = np.load(path, allow_pickle=False)

if mask.shape != (height, width):
    raise SystemExit(
        f"Invalid hot-pixel mask shape {mask.shape}; expected {(height, width)}"
    )

print(f"Hot-pixel mask: {path}")
print(f"Hot-pixel count: {int(np.count_nonzero(mask))}")
PY

# -----------------------------------------------------------------------------
# Resolve GT and output paths
# -----------------------------------------------------------------------------
GT_CHANNEL_DIR="${CHANNEL_DIR/dvs/GT200Hzskeleton}"
GT_FILE="$(dirname "$(dirname "$DATA_FILE")")/$GT_CHANNEL_DIR/data.log"

if [[ "$WITH_GT_OVERLAY" == "true" && ! -f "$GT_FILE" ]]; then
  echo "GT skeleton file not found: $GT_FILE" >&2
  echo "Use --prediction_only to generate only the direct C++ video." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

SAFE_PERIOD="${NET_PERIOD//./p}"
STEM="${SEQUENCE_DIR}__${CHANNEL_DIR}_np_${SAFE_PERIOD}"
PRED_CSV="$OUTPUT_DIR/${STEM}_prediction.csv"
BASE_VIDEO="$OUTPUT_DIR/${STEM}_events_prediction.mp4"
FINAL_VIDEO="$OUTPUT_DIR/${STEM}_events_gt_prediction.mp4"
RUN_LOG="$OUTPUT_DIR/${STEM}.log"
GPU_CSV="$OUTPUT_DIR/${STEM}_gpu.csv"

COMMON_ARGS=(
  --data_file "$DATA_FILE"
  --camera_index "$CAMERA_INDEX"
  --hotpixel_mask "$HOTPIXEL_MASK"
  --adaptive_hotpixel_thresh 0
  --checkpoint_path "$CHECKPOINT_PATH"
  --epp_script "$EPP_SCRIPT"
  --net_period "$NET_PERIOD"
  --output_period "$OUTPUT_PERIOD"
  --w "$IMG_W"
  --h "$IMG_H"
  --device "$DEVICE"
  --num_points "$NUM_POINTS"
  --events_per_window "$EVENTS_PER_WINDOW"
  --output_coordinates "$OUTPUT_COORDINATES"
  --input_coordinates "$INPUT_COORDINATES"
  --sampling_mode "$SAMPLING_MODE"
  --deterministic_sampling_seed "$DETERMINISTIC_SAMPLING_SEED"
  --debug_hash_every_windows "$DEBUG_HASH_EVERY_WINDOWS"
  --output_csv "$PRED_CSV"
  --output_video "$BASE_VIDEO"
  --gpu_file "$GPU_CSV"
  --gpu_period_ms "$GPU_PERIOD_MS"
  --max_packets "$MAX_PACKETS"
)

if [[ "$SWAP_LR" == "true" ]]; then
  COMMON_ARGS+=(--swap_lr)
fi

if [[ "$PRINT_CHECKPOINT_SUMMARY" == "true" ]]; then
  COMMON_ARGS+=(--print_checkpoint_summary)
fi

echo "EventPointPose single-entry video"
echo "Event file       : $DATA_FILE"
echo "Camera           : $CAMERA_INDEX"
echo "Hot-pixel mask   : $HOTPIXEL_MASK"
echo "GT file          : ${GT_FILE:-disabled}"
echo "Net period       : $NET_PERIOD s"
echo "Output period    : $OUTPUT_PERIOD s"
echo "Events/window    : $EVENTS_PER_WINDOW"
echo "Input coords     : $INPUT_COORDINATES"
echo "Output coords    : $OUTPUT_COORDINATES"
echo "Sampling mode    : $SAMPLING_MODE"
echo "Det. seed        : $DETERMINISTIC_SAMPLING_SEED"
echo "Hash debug every : $DEBUG_HASH_EVERY_WINDOWS"
echo "Base video       : $BASE_VIDEO"
echo "Prediction CSV   : $PRED_CSV"
echo "Run log          : $RUN_LOG"
if [[ "$WITH_GT_OVERLAY" == "true" ]]; then
  echo "GT overlay video : $FINAL_VIDEO"
fi

if [[ "$RUN_SELFTEST" == "true" ]]; then
  echo
  echo "Running RasEPC self-check..."
  python3 "$EPP_SCRIPT" \
    --checkpoint "$CHECKPOINT_PATH" \
    --sensor_w "$IMG_W" \
    --sensor_h "$IMG_H" \
    --seed 1 \
    --ras_epc_selfcheck \
    --ras_epc_selfcheck_events 4096
fi

echo
echo "Running EventPointPose..."
"$BINARY" "${COMMON_ARGS[@]}" >"$RUN_LOG" 2>&1

if ! grep -q "FINAL_STATS" "$RUN_LOG"; then
  echo "Missing FINAL_STATS in run log. Inspect: $RUN_LOG" >&2
  exit 1
fi

if grep -q "epp_out_of_range=[1-9]" "$RUN_LOG"; then
  echo "Warning: found epp_out_of_range > 0 in STATS/FINAL_STATS."
  grep "STATS\|FINAL_STATS" "$RUN_LOG" | tail -n 5 || true
fi

if [[ "$PRINT_CHECKPOINT_SUMMARY" == "true" ]]; then
  if ! grep -q "CHECKPOINT keys=" "$RUN_LOG"; then
    echo "Warning: checkpoint summary not found in run log."
  fi
fi

if [[ "$SAMPLING_MODE" == "deterministic_window" ]]; then
  HASH_LINES="$(grep "HASHES window=" "$RUN_LOG" || true)"
  if [[ -z "$HASH_LINES" ]]; then
    echo "Warning: no HASHES diagnostics found in run log."
  else
    BAD_HASH_COUNT="$(printf "%s\n" "$HASH_LINES" | grep -c "same_window_as_prev=1 same_tensor_as_prev=0" || true)"
    if [[ "${BAD_HASH_COUNT:-0}" -gt 0 ]]; then
      echo "Warning: unchanged windows produced different tensors in ${BAD_HASH_COUNT} cases."
    fi
  fi
fi

if [[ ! -s "$PRED_CSV" ]]; then
  echo "Prediction CSV was not created or contains no bytes: $PRED_CSV" >&2
  echo "Inspect: $RUN_LOG" >&2
  exit 1
fi

PRED_ROWS="$(awk 'END {print NR-1}' "$PRED_CSV")"
if [[ "$PRED_ROWS" -le 0 ]]; then
  echo "Prediction CSV contains no prediction rows: $PRED_CSV" >&2
  echo "The test may be too short to fill the rolling FIFO with $EVENTS_PER_WINDOW valid events." >&2
  echo "Inspect: $RUN_LOG" >&2
  exit 1
fi

if [[ ! -s "$BASE_VIDEO" ]]; then
  echo "Direct C++ video was not created or is empty: $BASE_VIDEO" >&2
  echo "Inspect: $RUN_LOG" >&2
  exit 1
fi

echo "Predictions produced: $PRED_ROWS"
echo "Direct prediction video: $BASE_VIDEO"

if [[ "$WITH_GT_OVERLAY" != "true" ]]; then
  echo
echo "Completed."
  echo "Video      : $BASE_VIDEO"
  echo "Prediction : $PRED_CSV"
  echo "Run log    : $RUN_LOG"
  echo "GPU samples: $GPU_CSV"
  exit 0
fi

echo "Creating GT/prediction overlay video..."

python3 - "$BASE_VIDEO" "$PRED_CSV" "$GT_FILE" "$FINAL_VIDEO" <<'PY'
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
        "Run the script in the environment used by the evaluation notebook. "
        f"Import error: {exc!r}"
    )

base_video = Path(sys.argv[1])
pred_csv = Path(sys.argv[2])
gt_file = Path(sys.argv[3])
final_video = Path(sys.argv[4])

EDGES = [
    (0, 1), (0, 2), (1, 2),
    (1, 3), (3, 7),
    (2, 4), (4, 8),
    (1, 6), (2, 5), (5, 6),
    (6, 9), (9, 11),
    (5, 10), (10, 12),
]

GT_COLOR = (0, 255, 0)
PRED_COLOR = (255, 0, 255)
OUTLINE_COLOR = (0, 0, 0)

with pred_csv.open("r", encoding="utf-8") as stream:
    csv_header = [field.strip() for field in stream.readline().strip().split(",")]

pred_array = np.loadtxt(pred_csv, delimiter=",", skiprows=1)
if pred_array.size == 0:
    raise SystemExit(f"Prediction CSV contains no rows: {pred_csv}")
if pred_array.ndim == 1:
    pred_array = pred_array[None, :]
if pred_array.shape[1] < 28:
    raise SystemExit(
        f"Prediction CSV has {pred_array.shape[1]} columns; expected at least 28."
    )

pred_ts = pred_array[:, 0].astype(np.float64)
pred_joints = pred_array[:, 2:28].reshape(-1, 13, 2).astype(np.float64)

if "video_frame" in csv_header:
    video_frame_column = csv_header.index("video_frame")
    video_mask = pred_array[:, video_frame_column] > 0.5
else:
    # Compatibility with an older binary. This is less robust because the CSV
    # does not identify which prediction rows were selected by output_period.
    video_mask = np.zeros(pred_ts.shape, dtype=bool)

if np.any(np.diff(pred_ts) < 0):
    order = np.argsort(pred_ts)
    pred_ts = pred_ts[order]
    pred_joints = pred_joints[order]
    video_mask = video_mask[order]

video_ts = pred_ts[video_mask]

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

capture = cv2.VideoCapture(str(base_video))
if not capture.isOpened():
    raise SystemExit(f"Could not open base video: {base_video}")

fps = float(capture.get(cv2.CAP_PROP_FPS))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

if not np.isfinite(fps) or fps <= 0.0:
    fps = 50.0

writer = cv2.VideoWriter(
    str(final_video),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
if not writer.isOpened():
    capture.release()
    raise SystemExit(f"Could not open output video: {final_video}")

if video_ts.size > 0:
    if reported_frames > 0 and reported_frames != video_ts.size:
        capture.release()
        writer.release()
        raise SystemExit(
            f"Video/CSV mismatch: {reported_frames} MP4 frames but "
            f"{video_ts.size} rows marked video_frame=1"
        )
else:
    video_start_ts = float(pred_ts[0])


def valid_point(point: np.ndarray) -> bool:
    return (
        point.shape[0] >= 2
        and np.isfinite(point[:2]).all()
        and 0.0 <= float(point[0]) < width
        and 0.0 <= float(point[1]) < height
    )


def as_int_point(point: np.ndarray):
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_skeleton(image: np.ndarray, joints: np.ndarray, color) -> None:
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


def interpolate_gt(timestamp: float) -> np.ndarray:
    result = np.full((13, 2), np.nan, dtype=np.float64)

    for joint_index, joint_name in enumerate(joint_names):
        xy = np.asarray(gt_data[joint_name], dtype=np.float64)

        if (
            xy.ndim != 2
            or xy.shape[0] != ts_gt_raw.shape[0]
            or xy.shape[1] < 2
        ):
            raise SystemExit(
                f"Unexpected GT shape for {joint_name}: {xy.shape}; "
                f"timestamps: {ts_gt_raw.shape}"
            )

        # Il GT YARP utilizzato da questa pipeline è già nell'ordine
        # OpenCV/hpe-core: prima coordinata x, seconda coordinata y.
        # Non scambiare gli assi e non applicare flip.
        result[joint_index, 0] = np.interp(
            timestamp,
            ts_gt_raw,
            xy[:, 0],
            left=xy[0, 0],
            right=xy[-1, 0],
        )

        result[joint_index, 1] = np.interp(
            timestamp,
            ts_gt_raw,
            xy[:, 1],
            left=xy[0, 1],
            right=xy[-1, 1],
        )

    return result


def draw_legend(image: np.ndarray, timestamp: float) -> None:
    cv2.rectangle(image, (4, 4), (145, 43), (0, 0, 0), -1)
    cv2.line(image, (10, 16), (25, 16), GT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        image, "GT", (31, 19), cv2.FONT_HERSHEY_SIMPLEX,
        0.32, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.line(image, (10, 32), (25, 32), PRED_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        image, "EventPointPose", (31, 35), cv2.FONT_HERSHEY_SIMPLEX,
        0.32, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image, f"t={timestamp:.3f}s", (max(4, width - 82), height - 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.30,
        (255, 255, 255), 1, cv2.LINE_AA,
    )

frame_index = 0
last_prediction_index = 0

while True:
    ok, frame = capture.read()
    if not ok:
        break

    if video_ts.size > 0:
        if frame_index >= video_ts.size:
            raise SystemExit("More MP4 frames than video_frame timestamps.")
        frame_ts = float(video_ts[frame_index])
    else:
        frame_ts = video_start_ts + frame_index / fps

    prediction_index = int(np.searchsorted(pred_ts, frame_ts, side="left"))
    prediction_index = max(0, min(prediction_index, len(pred_ts) - 1))
    last_prediction_index = prediction_index

    gt_joints = interpolate_gt(frame_ts)
    draw_skeleton(frame, gt_joints, GT_COLOR)
    # draw_skeleton(frame, pred_joints[prediction_index], PRED_COLOR)
    draw_legend(frame, frame_ts)

    writer.write(frame)
    frame_index += 1

capture.release()
writer.release()

if frame_index == 0:
    raise SystemExit("No video frames were written.")

print(f"Base video frames reported: {reported_frames}")
print(f"Final video frames written: {frame_index}")
print(f"Video FPS: {fps:g}")
print(f"Last prediction index used: {last_prediction_index}")
print(f"Final video: {final_video}")
if video_ts.size > 0:
    print("GT overlay used exact per-frame timestamps from video_frame CSV markers.")
else:
    print(
        "Warning: legacy CSV without video_frame markers; overlay timestamps "
        "were estimated from MP4 FPS."
    )
PY

if [[ ! -s "$FINAL_VIDEO" ]]; then
  echo "GT-overlay video was not created: $FINAL_VIDEO" >&2
  exit 1
fi

if [[ "$KEEP_BASE_VIDEO" != "true" ]]; then
  rm -f "$BASE_VIDEO"
fi

if [[ "$COMPUTE_METRICS" == "true" ]]; then
  echo
  echo "Computing GT-vs-pred metrics (timestamp-matched MPJPE/PCK)..."
  python3 - "$PRED_CSV" "$GT_FILE" <<'PY'
from __future__ import annotations

import csv
import sys
import numpy as np

try:
  from datasets.utils import constants as ds_constants
  from datasets.utils import parsing as ds_parsing
except Exception as exc:
  raise SystemExit(
    "Could not import datasets.utils from hpe-core environment. "
    f"Import error: {exc!r}"
  )

pred_csv = sys.argv[1]
gt_file = sys.argv[2]

joint_names = list(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys())
if len(joint_names) != 13:
  raise SystemExit(f"Expected 13 joints, found {len(joint_names)}")

with open(pred_csv, "r", encoding="utf-8") as stream:
  reader = csv.DictReader(stream)
  rows = list(reader)

if not rows:
  raise SystemExit("Prediction CSV has no rows.")

pred_ts = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)
pred_xy = np.zeros((len(rows), 13, 2), dtype=np.float64)
for i, row in enumerate(rows):
  for j, joint in enumerate(joint_names):
    pred_xy[i, j, 0] = float(row[f"{joint}_x"])
    pred_xy[i, j, 1] = float(row[f"{joint}_y"])

gt_data = ds_parsing.import_yarp_skeleton_data(gt_file, multi_channel=False)
gt_ts = np.asarray(gt_data["ts"], dtype=np.float64)
if gt_ts.size == 0:
  raise SystemExit("GT file has no timestamps.")

gt_xy = np.zeros((gt_ts.size, 13, 2), dtype=np.float64)
for j, joint in enumerate(joint_names):
  xy = np.asarray(gt_data[joint], dtype=np.float64)
  if xy.shape[0] != gt_ts.size or xy.shape[1] < 2:
    raise SystemExit(f"Unexpected GT shape for {joint}: {xy.shape}")
  gt_xy[:, j, 0] = xy[:, 0]
  gt_xy[:, j, 1] = xy[:, 1]

indices = np.searchsorted(gt_ts, pred_ts, side="left")
indices = np.clip(indices, 0, gt_ts.size - 1)
left = np.clip(indices - 1, 0, gt_ts.size - 1)
use_left = np.abs(gt_ts[left] - pred_ts) < np.abs(gt_ts[indices] - pred_ts)
indices[use_left] = left[use_left]

matched_gt = gt_xy[indices]
dt = np.abs(gt_ts[indices] - pred_ts)
dist = np.linalg.norm(pred_xy - matched_gt, axis=2)

mpjpe = float(np.mean(dist))
pck5 = float(np.mean(dist <= 5.0))
pck10 = float(np.mean(dist <= 10.0))
pck20 = float(np.mean(dist <= 20.0))

print(
  "METRICS matches={} mean_dt_s={:.6f} mpjpe_px={:.3f} pck@5={:.4f} pck@10={:.4f} pck@20={:.4f}".format(
    dist.shape[0],
    float(np.mean(dt)),
    mpjpe,
    pck5,
    pck10,
    pck20,
  )
)
PY
fi

echo
echo "Completed."
echo "Direct video : $BASE_VIDEO"
echo "GT overlay   : $FINAL_VIDEO"
echo "Prediction   : $PRED_CSV"
echo "Run log      : $RUN_LOG"
echo "GPU samples  : $GPU_CSV"