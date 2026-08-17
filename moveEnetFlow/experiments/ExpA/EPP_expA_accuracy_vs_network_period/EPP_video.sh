#!/usr/bin/env bash
set -euo pipefail

# Run the current EventPointPose offline YARP pipeline on one DHP19 camera log.
#
# The selected data.log is assumed to come from the standard hpe-core DHP19
# preprocessing/export path. The Python sidecar is therefore started with
# --input_preprocessing already_filtered: hot-pixel, background, and IR filters
# are NOT applied a second time.
#
# Outputs:
#   1. the prediction CSV produced by EventPointPose_offline;
#   2. a prediction-only MP4 reconstructed from the same persistent 7500-event
#      FIFO windows used at PointNet request times;
#   3. optionally, a second MP4 with DHP19 GT overlaid;
#   4. optionally, timestamp-matched MPJPE/PCK metrics.
#
# Important: the current EventPointPose_offline.cpp writes CSV only. Video is
# reconstructed here after inference from data.log + CSV. The event background
# is rendered in EventPointPose coordinates:
#     x_view = x_log
#     y_view = 259 - y_log
# so it is aligned with the PointNet coordinate output.

BINARY="/workspace/moveEnetFlow/build2/eventPointPose_offline"
DATA_ROOT="/data/dhp19_testing_set_S13toS17"
DATA_FILE=""
DATA_GLOB="*/ch[23]dvs/data.log"

MODEL_PATH="/workspace/model_mounts/eventpointpose/PointNet/models/model.pth"
EPP_SCRIPT="/workspace/model_mounts/eventpointpose/PointNet/models/eventPointPose_yarp_server.py"
UPSTREAM_REPO=""

RESULTS_ROOT="${MOVENET_RESULTS_ROOT:-/data/MoveEnet_OFK_results}"
OUTPUT_DIR="$RESULTS_ROOT/EPP_dhp19_full_test/videos/rolling"
NET_PERIOD="0.02"
OUTPUT_PERIOD="0.005"
DEVICE="cuda:0"
IMG_W="346"
IMG_H="260"
NUM_POINTS="2048"
FIFO_SIZE="7500"
SEED="1"
MAX_PACKETS="-1"
STARTUP_TIMEOUT="30"
RESPONSE_TIMEOUT="0"
SERVER_VERBOSE="false"
DUMP_DIR=""
DUMP_FIRST_N="1"
RUN_SELFTEST="true"
GENERATE_VIDEO="true"
WITH_GT_OVERLAY="true"
KEEP_BASE_VIDEO="true"
COMPUTE_METRICS="true"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Input selection:
  --data_root <path>              DHP19 root. Default: $DATA_ROOT
  --data_file <path>              Run one ch2dvs/ch3dvs data.log
  --data_glob <pattern>           Auto-selection pattern. Default: $DATA_GLOB
  --max_packets <int>             Use only the first N packet lines; -1 = full log

EventPointPose:
  --binary <path>                 eventPointPose_offline executable
  --model_path <path>             PointNet MeanLabel checkpoint
  --checkpoint_path <path>        Alias for --model_path
  --epp_script <path>             eventPointPose_yarp_server.py
  --net_period <seconds>          PointNet request period. Default: $NET_PERIOD
  --output_period <seconds>       Held-pose CSV/video period. Default: $OUTPUT_PERIOD
  --device <device>               PyTorch device. Default: $DEVICE
  --seed <int>                    Deterministic sampling seed. Default: $SEED
  --num_points <int>              Must remain 2048 for this checkpoint
  --events_per_window <int>       Must remain 7500 for this deployment
  --events_per_block <int>        Legacy alias for --events_per_window
  --startup_timeout <seconds>     Sidecar startup timeout. Default: $STARTUP_TIMEOUT
  --response_timeout <seconds>    0 blocks indefinitely. Default: $RESPONSE_TIMEOUT
  --server_verbose                Print every sidecar request in its log
  --dump_dir <path>               Save PointNet/RasEPC NumPy validation dumps
  --dump_first_n <int>            Successful requests to dump. Default: $DUMP_FIRST_N

Validation:
  --upstream_repo <path>          EventPointPose repo root for bitwise RasEPC test
  --skip_selftest                 Skip coordinate/RasEPC self-tests

Output:
  --output_dir <path>             Output directory. Default: $OUTPUT_DIR
  --no_video                      Produce CSV/logs only
  --prediction_only               Do not create the GT-overlay video
  --remove_base_video             Delete prediction-only MP4 after GT overlay
  --skip_metrics                  Skip timestamp-matched GT metrics
  --help                          Show this help

Examples:
  $(basename "$0") --data_file /data/dhp19_testing_set_S13toS17/S13_1_1/ch2dvs/data.log
  $(basename "$0") --data_file /data/dhp19_testing_set_S13toS17/S13_1_1/ch3dvs/data.log --net_period 0.02
  $(basename "$0") --data_file /data/.../S13_1_1/ch2dvs/data.log --max_packets 500 --prediction_only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --data_file) DATA_FILE="$2"; shift 2 ;;
    --data_glob) DATA_GLOB="$2"; shift 2 ;;
    --model_path|--checkpoint_path) MODEL_PATH="$2"; shift 2 ;;
    --epp_script) EPP_SCRIPT="$2"; shift 2 ;;
    --upstream_repo) UPSTREAM_REPO="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --net_period) NET_PERIOD="$2"; shift 2 ;;
    --output_period) OUTPUT_PERIOD="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --num_points) NUM_POINTS="$2"; shift 2 ;;
    --events_per_window|--events_per_block) FIFO_SIZE="$2"; shift 2 ;;
    --max_packets) MAX_PACKETS="$2"; shift 2 ;;
    --startup_timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
    --response_timeout) RESPONSE_TIMEOUT="$2"; shift 2 ;;
    --server_verbose) SERVER_VERBOSE="true"; shift ;;
    --dump_dir) DUMP_DIR="$2"; shift 2 ;;
    --dump_first_n) DUMP_FIRST_N="$2"; shift 2 ;;
    --skip_selftest) RUN_SELFTEST="false"; shift ;;
    --no_video) GENERATE_VIDEO="false"; shift ;;
    --prediction_only) WITH_GT_OVERLAY="false"; shift ;;
    --remove_base_video) KEEP_BASE_VIDEO="false"; shift ;;
    --skip_metrics) COMPUTE_METRICS="false"; shift ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# The checkpoint and the agreed deployment have fixed dimensions.
if [[ "$IMG_W" != "346" || "$IMG_H" != "260" ]]; then
  echo "Internal error: this script must use the DHP19 resolution 346x260." >&2
  exit 1
fi
if [[ "$NUM_POINTS" != "2048" ]]; then
  echo "--num_points must be 2048 for the PointNet MeanLabel checkpoint." >&2
  exit 1
fi
if [[ "$FIFO_SIZE" != "7500" ]]; then
  echo "--events_per_window must be 7500 for the agreed online-style FIFO." >&2
  exit 1
fi
if ! [[ "$MAX_PACKETS" =~ ^-?[0-9]+$ ]] || [[ "$MAX_PACKETS" -eq 0 ]] || [[ "$MAX_PACKETS" -lt -1 ]]; then
  echo "--max_packets must be -1 or a positive integer." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Validate programs and select the input entry
# -----------------------------------------------------------------------------
if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "PointNet checkpoint not found: $MODEL_PATH" >&2
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
ENTRY_DIR="$(dirname "$(dirname "$DATA_FILE")")"

case "$CHANNEL_DIR" in
  ch2dvs) CAMERA="2" ;;
  ch3dvs) CAMERA="3" ;;
  *)
    echo "Unsupported EventPointPose entry: $CHANNEL_DIR" >&2
    echo "Use a ch2dvs or ch3dvs data.log." >&2
    exit 1
    ;;
esac

GT_CHANNEL_DIR="${CHANNEL_DIR/dvs/GT200Hzskeleton}"
GT_FILE="$ENTRY_DIR/$GT_CHANNEL_DIR/data.log"

if [[ "$WITH_GT_OVERLAY" == "true" || "$COMPUTE_METRICS" == "true" ]]; then
  if [[ ! -f "$GT_FILE" ]]; then
    echo "GT skeleton file not found: $GT_FILE" >&2
    echo "Use --prediction_only --skip_metrics to run without GT." >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

SAFE_NET_PERIOD="${NET_PERIOD//./p}"
SAFE_OUTPUT_PERIOD="${OUTPUT_PERIOD//./p}"
STEM="${SEQUENCE_DIR}__${CHANNEL_DIR}_np_${SAFE_NET_PERIOD}_op_${SAFE_OUTPUT_PERIOD}"
PRED_CSV="$OUTPUT_DIR/${STEM}_prediction.csv"
BASE_VIDEO="$OUTPUT_DIR/${STEM}_events_prediction.mp4"
FINAL_VIDEO="$OUTPUT_DIR/${STEM}_events_gt_prediction.mp4"
RUN_LOG="$OUTPUT_DIR/${STEM}.log"
SERVER_LOG="$OUTPUT_DIR/${STEM}_server.log"

if [[ -n "$DUMP_DIR" ]]; then
  mkdir -p "$DUMP_DIR"
  DUMP_DIR="$(realpath "$DUMP_DIR")"
fi

# --max_packets is implemented by creating a temporary data.log containing the
# first N packet lines. EventPointPose_offline itself intentionally has no
# packet-limit option.
TMP_DIR=""
WORK_DATA_FILE="$DATA_FILE"
cleanup() {
  if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT INT TERM

if [[ "$MAX_PACKETS" -gt 0 ]]; then
  TMP_DIR="$(mktemp -d /tmp/epp_video_packets.XXXXXX)"
  WORK_DATA_FILE="$TMP_DIR/data.log"
  head -n "$MAX_PACKETS" "$DATA_FILE" > "$WORK_DATA_FILE"
  if [[ ! -s "$WORK_DATA_FILE" ]]; then
    echo "The truncated data.log is empty: $WORK_DATA_FILE" >&2
    exit 1
  fi
fi

# Resolve an upstream repository only for the optional bitwise RasEPC test.
if [[ -z "$UPSTREAM_REPO" ]]; then
  for candidate in \
    /workspace/model_mounts/eventpointpose/PointNet \
    /workspace/model_mounts/eventpointpose/EventPointPose \
    /workspace/EventPointPose; do
    if [[ -f "$candidate/dataset/rasterized.py" ]]; then
      UPSTREAM_REPO="$candidate"
      break
    fi
  done
fi

COMMON_ARGS=(
  --data_file "$WORK_DATA_FILE"
  --camera "$CAMERA"
  --model_path "$MODEL_PATH"
  --EventPointPose_script "$EPP_SCRIPT"
  --net_period "$NET_PERIOD"
  --output_period "$OUTPUT_PERIOD"
  --output_csv "$PRED_CSV"
  --w "$IMG_W"
  --h "$IMG_H"
  --device "$DEVICE"
  --fifo_size "$FIFO_SIZE"
  --num_points "$NUM_POINTS"
  --seed "$SEED"
  --input_preprocessing already_filtered
  --startup_timeout "$STARTUP_TIMEOUT"
  --response_timeout "$RESPONSE_TIMEOUT"
  --server_log "$SERVER_LOG"
)

if [[ "$SERVER_VERBOSE" == "true" ]]; then
  COMMON_ARGS+=(--server_verbose)
fi
if [[ -n "$DUMP_DIR" ]]; then
  COMMON_ARGS+=(--dump_dir "$DUMP_DIR" --dump_first_n "$DUMP_FIRST_N")
fi

echo "EventPointPose single-entry run"
echo "Event file        : $DATA_FILE"
if [[ "$WORK_DATA_FILE" != "$DATA_FILE" ]]; then
  echo "Replay file       : $WORK_DATA_FILE (first $MAX_PACKETS packets)"
fi
echo "Sequence          : $SEQUENCE_DIR"
echo "Camera            : $CAMERA ($CHANNEL_DIR)"
echo "Preprocessing     : already_filtered (hot/background/IR skipped)"
echo "Checkpoint        : $MODEL_PATH"
echo "Net period        : $NET_PERIOD s"
echo "Output period     : $OUTPUT_PERIOD s"
echo "FIFO / RasEPC     : $FIFO_SIZE events / $NUM_POINTS points"
echo "Seed              : $SEED"
echo "Prediction CSV    : $PRED_CSV"
echo "C++ run log       : $RUN_LOG"
echo "Python server log : $SERVER_LOG"
if [[ "$GENERATE_VIDEO" == "true" ]]; then
  echo "Prediction video  : $BASE_VIDEO"
  if [[ "$WITH_GT_OVERLAY" == "true" ]]; then
    echo "GT overlay video  : $FINAL_VIDEO"
  fi
fi

if [[ "$RUN_SELFTEST" == "true" ]]; then
  echo
  echo "Running coordinate/RasEPC self-tests..."
  SELFTEST_ARGS=(--self_test_all)
  if [[ -n "$UPSTREAM_REPO" && -f "$UPSTREAM_REPO/dataset/rasterized.py" ]]; then
    SELFTEST_ARGS+=(--upstream_repo "$UPSTREAM_REPO")
  fi
  python3 "$EPP_SCRIPT" "${SELFTEST_ARGS[@]}"
fi

echo
echo "Running EventPointPose..."
if "$BINARY" "${COMMON_ARGS[@]}" >"$RUN_LOG" 2>&1; then
  :
else
  status=$?
  echo "EventPointPose_offline failed with status $status." >&2
  echo "Last C++ log lines:" >&2
  tail -n 40 "$RUN_LOG" >&2 || true
  echo "Last Python sidecar log lines:" >&2
  tail -n 40 "$SERVER_LOG" >&2 || true
  exit "$status"
fi

if [[ ! -s "$PRED_CSV" ]]; then
  echo "Prediction CSV was not created or is empty: $PRED_CSV" >&2
  tail -n 40 "$RUN_LOG" >&2 || true
  exit 1
fi

PRED_ROWS="$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' "$PRED_CSV")"
if [[ "$PRED_ROWS" -le 0 ]]; then
  echo "Prediction CSV contains no pose rows: $PRED_CSV" >&2
  echo "The selected replay may be too short to fill the $FIFO_SIZE-event FIFO." >&2
  tail -n 40 "$RUN_LOG" >&2 || true
  exit 1
fi

VALID_INFERENCES="$(grep -oE 'Valid PointNet inferences:[[:space:]]*[0-9]+' "$RUN_LOG" | tail -n 1 | grep -oE '[0-9]+' || true)"
SERVER_ERRORS="$(grep -oE 'Server ERROR responses:[[:space:]]*[0-9]+' "$RUN_LOG" | tail -n 1 | grep -oE '[0-9]+' || true)"
VALID_INFERENCES="${VALID_INFERENCES:-unknown}"
SERVER_ERRORS="${SERVER_ERRORS:-unknown}"

echo "Prediction rows    : $PRED_ROWS"
echo "Valid inferences  : $VALID_INFERENCES"
echo "Server errors     : $SERVER_ERRORS"

if [[ "$SERVER_ERRORS" != "unknown" && "$SERVER_ERRORS" != "0" ]]; then
  echo "The server reported $SERVER_ERRORS ERROR responses. Inspect $SERVER_LOG." >&2
  exit 1
fi

if [[ "$GENERATE_VIDEO" == "true" ]]; then
  echo
  echo "Reconstructing EventPointPose FIFO video..."

  FINAL_VIDEO_ARG=""
  GT_FILE_ARG=""
  if [[ "$WITH_GT_OVERLAY" == "true" ]]; then
    FINAL_VIDEO_ARG="$FINAL_VIDEO"
    GT_FILE_ARG="$GT_FILE"
  fi

  python3 - \
    "$WORK_DATA_FILE" \
    "$PRED_CSV" \
    "$BASE_VIDEO" \
    "$FINAL_VIDEO_ARG" \
    "$GT_FILE_ARG" \
    "$NET_PERIOD" \
    "$OUTPUT_PERIOD" \
    "$FIFO_SIZE" \
    "$IMG_W" \
    "$IMG_H" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

data_file = Path(sys.argv[1])
pred_csv = Path(sys.argv[2])
base_video = Path(sys.argv[3])
final_video = Path(sys.argv[4]) if sys.argv[4] else None
gt_file = Path(sys.argv[5]) if sys.argv[5] else None
net_period = float(sys.argv[6])
output_period = float(sys.argv[7])
fifo_size = int(sys.argv[8])
width = int(sys.argv[9])
height = int(sys.argv[10])

EPS = 1.0e-9
EDGES = [
    (0, 1), (0, 2), (1, 2),
    (1, 3), (3, 7),
    (2, 4), (4, 8),
    (1, 6), (2, 5), (5, 6),
    (6, 9), (9, 11),
    (5, 10), (10, 12),
]
PRED_COLOR = (255, 0, 255)  # BGR magenta
GT_COLOR = (0, 255, 0)      # BGR green
OUTLINE_COLOR = (0, 0, 0)


class EventFIFO:
    """Ring FIFO preserving the latest accepted events in chronological order."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data = np.empty((self.capacity, 3), dtype=np.int64)  # x, y, polarity
        self.count = 0
        self.write = 0

    def append(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return
        batch = np.asarray(batch, dtype=np.int64)
        if batch.ndim != 2 or batch.shape[1] != 3:
            raise ValueError(f"Expected an [N,3] event batch, received {batch.shape}")

        if batch.shape[0] >= self.capacity:
            self.data[:, :] = batch[-self.capacity :, :]
            self.count = self.capacity
            self.write = 0
            return

        first = min(batch.shape[0], self.capacity - self.write)
        self.data[self.write : self.write + first, :] = batch[:first, :]
        remaining = batch.shape[0] - first
        if remaining:
            self.data[:remaining, :] = batch[first:, :]
        self.write = (self.write + batch.shape[0]) % self.capacity
        self.count = min(self.capacity, self.count + batch.shape[0])

    def latest(self) -> Optional[np.ndarray]:
        if self.count < self.capacity:
            return None
        if self.write == 0:
            return self.data.copy()
        return np.concatenate(
            (self.data[self.write :, :], self.data[: self.write, :]), axis=0
        )


def decode_yarp_string(payload: bytes, line_number: int) -> bytes:
    """Decode the escaping used by YARP Bottle strings in data.log."""
    decoded = bytearray()
    index = 0
    escapes = {
        ord("0"): 0,
        ord("n"): 10,
        ord("r"): 13,
        ord("t"): 9,
        ord('"'): 34,
        ord("\\"): 92,
    }

    while index < len(payload):
        value = payload[index]
        if value != 92:  # backslash
            decoded.append(value)
            index += 1
            continue

        if index + 1 >= len(payload):
            raise SystemExit(f"Trailing escape in event log line {line_number}")
        escaped = payload[index + 1]
        if escaped not in escapes:
            raise SystemExit(
                f"Unsupported YARP escape \\{chr(escaped)!s} in event log line {line_number}"
            )
        decoded.append(escapes[escaped])
        index += 2

    return bytes(decoded)


def read_event_log(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Read ev::AE packets without requiring Bimvee.

    Each packet line stores one envelope timestamp and a binary string of 32-bit
    address events. The event codec is the robotology/event-driven AE layout:
    polarity bit 0, x bits 1..11, y bits 12..21.
    """
    ts_chunks = []
    x_chunks = []
    y_chunks = []
    p_chunks = []
    packet_count = 0

    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip(b"\r\n")
            if not line:
                continue

            first_quote = line.find(b'"')
            last_quote = line.rfind(b'"')
            if first_quote < 0 or last_quote <= first_quote:
                raise SystemExit(f"Malformed event log line {line_number}: missing quoted payload")

            header = line[:first_quote].strip().split()
            if len(header) < 4 or header[2] != b"AE":
                raise SystemExit(
                    f"Malformed event log line {line_number}: expected '<id> <ts> AE <duration> ...'"
                )
            try:
                packet_timestamp = float(header[1])
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid packet timestamp in event log line {line_number}: {header[1]!r}"
                ) from exc
            if not np.isfinite(packet_timestamp):
                raise SystemExit(f"Non-finite packet timestamp in event log line {line_number}")

            raw = decode_yarp_string(line[first_quote + 1 : last_quote], line_number)
            if len(raw) % 4 != 0:
                raise SystemExit(
                    f"Invalid AE payload length {len(raw)} in line {line_number}; expected a multiple of 4"
                )

            words = np.frombuffer(raw, dtype="<u4")
            if words.size:
                x = ((words >> 1) & 0x7FF).astype(np.int64, copy=False)
                y = ((words >> 12) & 0x3FF).astype(np.int64, copy=False)
                p = (words & 0x1).astype(np.int64, copy=False)
                ts = np.full(words.size, packet_timestamp, dtype=np.float64)

                ts_chunks.append(ts)
                x_chunks.append(x)
                y_chunks.append(y)
                p_chunks.append(p)

            packet_count += 1

    if packet_count == 0 or not ts_chunks:
        raise SystemExit(f"No AE events found in {path}")

    event_ts = np.concatenate(ts_chunks)
    event_x = np.concatenate(x_chunks)
    event_y = np.concatenate(y_chunks)
    event_p = np.concatenate(p_chunks)

    if np.any(np.diff(event_ts) < -EPS):
        raise SystemExit("Event packet timestamps are not monotonic")

    return event_ts, event_x, event_y, event_p, packet_count


def read_prediction_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as stream:
        header = [item.strip() for item in stream.readline().strip().split(",")]

    required = ["timestamp", "latency"]
    for joint in range(13):
        required.extend((f"joint{joint}_x", f"joint{joint}_y"))
    if header != required:
        raise SystemExit(
            "Unexpected prediction CSV header.\n"
            f"Expected: {','.join(required)}\n"
            f"Got:      {','.join(header)}"
        )

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.size == 0:
        raise SystemExit(f"Prediction CSV contains no rows: {path}")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 28:
        raise SystemExit(f"Prediction CSV has {data.shape[1]} columns; expected 28")

    timestamps = data[:, 0].astype(np.float64)
    joints = data[:, 2:28].reshape(-1, 13, 2).astype(np.float64)
    if np.any(np.diff(timestamps) < -EPS):
        raise SystemExit("Prediction CSV timestamps are not monotonic")
    return timestamps, joints


def read_gt_skeletons(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse hpe-core YARP SKLT rows without importing hpe-core Python modules."""
    timestamps = []
    skeletons = []

    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            marker = " SKLT ("
            if marker not in line:
                raise SystemExit(f"Malformed GT line {line_number}: missing 'SKLT ('")
            prefix, remainder = line.split(marker, 1)
            if ")" not in remainder:
                raise SystemExit(f"Malformed GT line {line_number}: missing closing ')' ")
            coordinates_text, _suffix = remainder.split(")", 1)
            prefix_fields = prefix.split()
            if len(prefix_fields) < 2:
                raise SystemExit(f"Malformed GT line {line_number}: missing timestamp")

            try:
                timestamp = float(prefix_fields[1])
            except ValueError as exc:
                raise SystemExit(f"Invalid GT timestamp in line {line_number}") from exc
            coordinates = np.fromstring(coordinates_text, sep=" ", dtype=np.float64)
            if coordinates.size != 26:
                raise SystemExit(
                    f"GT line {line_number} has {coordinates.size} coordinates; expected 26"
                )
            timestamps.append(timestamp)
            skeletons.append(coordinates.reshape(13, 2))

    if not timestamps:
        raise SystemExit(f"GT file contains no skeletons: {path}")

    ts = np.asarray(timestamps, dtype=np.float64)
    xy = np.asarray(skeletons, dtype=np.float64)
    if np.any(np.diff(ts) < -EPS):
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        xy = xy[order]
    return ts, xy


def valid_point(point: np.ndarray) -> bool:
    return (
        point.shape[0] >= 2
        and np.isfinite(point[:2]).all()
        and 0.0 <= float(point[0]) < width
        and 0.0 <= float(point[1]) < height
    )


def int_point(point: np.ndarray) -> Tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_skeleton(image: np.ndarray, joints: np.ndarray, color) -> None:
    for first, second in EDGES:
        if valid_point(joints[first]) and valid_point(joints[second]):
            p1 = int_point(joints[first])
            p2 = int_point(joints[second])
            cv2.line(image, p1, p2, OUTLINE_COLOR, 3, cv2.LINE_AA)
            cv2.line(image, p1, p2, color, 1, cv2.LINE_AA)
    for point in joints:
        if valid_point(point):
            p = int_point(point)
            cv2.circle(image, p, 3, OUTLINE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(image, p, 1, color, -1, cv2.LINE_AA)


def draw_legend(image: np.ndarray, timestamp: float, include_gt: bool) -> None:
    bottom = 45 if include_gt else 29
    cv2.rectangle(image, (4, 4), (152, bottom), (0, 0, 0), -1)
    cv2.line(image, (10, 16), (25, 16), PRED_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        image, "EventPointPose", (31, 19), cv2.FONT_HERSHEY_SIMPLEX,
        0.32, (255, 255, 255), 1, cv2.LINE_AA,
    )
    if include_gt:
        cv2.line(image, (10, 32), (25, 32), GT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(
            image, "GT", (31, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.32, (255, 255, 255), 1, cv2.LINE_AA,
        )
    cv2.putText(
        image, f"t={timestamp:.3f}s", (max(4, width - 82), height - 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA,
    )


def render_fifo(window: np.ndarray) -> np.ndarray:
    # EventPointPose reconstructs MeanLabel coordinates as:
    #   x_pointnet = x_log
    #   y_pointnet = 259 - y_log
    x = window[:, 0]
    # y = (height - 1) - window[:, 1]
    y = window[:, 1]
    polarity = window[:, 2]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    polarity = polarity[valid]

    gray = np.zeros((height, width), dtype=np.uint8)
    if x.size:
        recency = np.linspace(0.0, 1.0, x.size, endpoint=True)
        intensity = 35.0 + recency * 220.0
        intensity = np.where(polarity > 0, intensity, intensity * 0.55)
        intensity = np.clip(intensity, 0, 255).astype(np.uint8)
        np.maximum.at(gray, (y.astype(np.int64), x.astype(np.int64)), intensity)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


pred_ts, pred_xy = read_prediction_csv(pred_csv)
event_ts, event_x, event_y, event_p, source_packet_count = read_event_log(data_file)

# Consecutive equal timestamps form one packet group, matching the C++ replay.
changes = np.flatnonzero(event_ts[1:] != event_ts[:-1]) + 1
starts = np.concatenate(([0], changes))
ends = np.concatenate((changes, [event_ts.size]))
packet_ts = event_ts[starts]

if pred_ts[-1] > packet_ts[-1] + output_period + EPS:
    raise SystemExit(
        f"Prediction timestamps extend beyond the event log: "
        f"{pred_ts[-1]:.6f} > {packet_ts[-1]:.6f}"
    )

# GT is already stored in hpe-core order and OpenCV coordinates (x,y).
gt_ts = None
gt_xy = None
if final_video is not None:
    if gt_file is None:
        raise SystemExit("GT-overlay output requested without a GT file")
    gt_ts, gt_xy = read_gt_skeletons(gt_file)


def interpolate_gt(timestamp: float) -> np.ndarray:
    if gt_ts is None or gt_xy is None:
        raise RuntimeError("GT was not loaded")
    result = np.empty((13, 2), dtype=np.float64)
    for joint in range(13):
        result[joint, 0] = np.interp(
            timestamp, gt_ts, gt_xy[:, joint, 0],
            left=gt_xy[0, joint, 0], right=gt_xy[-1, joint, 0],
        )
        result[joint, 1] = np.interp(
            timestamp, gt_ts, gt_xy[:, joint, 1],
            left=gt_xy[0, joint, 1], right=gt_xy[-1, joint, 1],
        )
    return result


fps = 1.0 / output_period
base_writer = cv2.VideoWriter(
    str(base_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)
if not base_writer.isOpened():
    raise SystemExit(f"Could not open prediction video: {base_video}")

final_writer = None
if final_video is not None:
    final_writer = cv2.VideoWriter(
        str(final_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not final_writer.isOpened():
        base_writer.release()
        raise SystemExit(f"Could not open GT-overlay video: {final_video}")

# Reproduce EventPointPose_offline's absolute schedules. The C++ file writes no
# CSV rows before the first valid prediction, so every prediction row below must
# have a corresponding full FIFO state.
fifo = EventFIFO(fifo_size)
next_inference_ts = float(packet_ts[0])
pending_start = 0
packet_index = 0
pose_available = False
successful_windows = 0

try:
    for row_index, output_ts in enumerate(pred_ts):
        while packet_index < packet_ts.size and packet_ts[packet_index] <= output_ts + EPS:
            current_ts = float(packet_ts[packet_index])
            current_end = int(ends[packet_index])

            if current_ts + EPS >= next_inference_ts:
                batch_x = event_x[pending_start:current_end]
                batch_y = event_y[pending_start:current_end]
                batch_p = event_p[pending_start:current_end]
                batch_t = event_ts[pending_start:current_end]

                valid = (
                    (batch_x >= 0) & (batch_x < width)
                    & (batch_y >= 0) & (batch_y < height)
                    & ((batch_p == 0) | (batch_p == 1))
                    & np.isfinite(batch_t)
                )
                accepted = np.column_stack(
                    (batch_x[valid], batch_y[valid], batch_p[valid])
                )
                if accepted.shape[0] > 0:
                    fifo.append(accepted)
                    if fifo.count == fifo_size:
                        pose_available = True
                        successful_windows += 1

                # The server consumes the request batch for OK, WARMUP,
                # NO_UPDATE, and protocol-level ERROR responses.
                pending_start = current_end
                while next_inference_ts <= current_ts + EPS:
                    next_inference_ts += net_period

            packet_index += 1

        window = fifo.latest()
        if not pose_available or window is None:
            raise SystemExit(
                f"Could not reconstruct a full FIFO for prediction row {row_index} "
                f"at t={output_ts:.6f}"
            )

        event_frame = render_fifo(window)

        prediction_frame = event_frame.copy()
        draw_skeleton(prediction_frame, pred_xy[row_index], PRED_COLOR)
        draw_legend(prediction_frame, float(output_ts), include_gt=False)
        base_writer.write(prediction_frame)

        if final_writer is not None:
            overlay = event_frame.copy()
            draw_skeleton(overlay, interpolate_gt(float(output_ts)), GT_COLOR)
            draw_skeleton(overlay, pred_xy[row_index], PRED_COLOR)
            draw_legend(overlay, float(output_ts), include_gt=True)
            final_writer.write(overlay)
finally:
    base_writer.release()
    if final_writer is not None:
        final_writer.release()

print(f"Source packet lines: {source_packet_count}")
print(f"Decoded events: {event_ts.size}")
print(f"Packet timestamp groups: {packet_ts.size}")
print(f"Reconstructed full-FIFO inference windows: {successful_windows}")
print(f"Video frames written: {pred_ts.size}")
print(f"Video FPS: {fps:g}")
print(f"Prediction video: {base_video}")
if final_video is not None:
    print(f"GT-overlay video: {final_video}")
PY

  if [[ ! -s "$BASE_VIDEO" ]]; then
    echo "Prediction video was not created: $BASE_VIDEO" >&2
    exit 1
  fi
  if [[ "$WITH_GT_OVERLAY" == "true" && ! -s "$FINAL_VIDEO" ]]; then
    echo "GT-overlay video was not created: $FINAL_VIDEO" >&2
    exit 1
  fi

  if [[ "$WITH_GT_OVERLAY" == "true" && "$KEEP_BASE_VIDEO" != "true" ]]; then
    rm -f "$BASE_VIDEO"
  fi
fi

if [[ "$COMPUTE_METRICS" == "true" ]]; then
  echo
  echo "Computing timestamp-matched GT metrics..."
  python3 - "$PRED_CSV" "$GT_FILE" "$IMG_W" "$IMG_H" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

pred_csv = Path(sys.argv[1])
gt_file = Path(sys.argv[2])
width = int(sys.argv[3])
height = int(sys.argv[4])


def read_gt_skeletons(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    timestamps = []
    skeletons = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            marker = " SKLT ("
            if marker not in line:
                raise SystemExit(f"Malformed GT line {line_number}: missing 'SKLT ('")
            prefix, remainder = line.split(marker, 1)
            if ")" not in remainder:
                raise SystemExit(f"Malformed GT line {line_number}: missing closing ')' ")
            coordinates_text, _suffix = remainder.split(")", 1)
            fields = prefix.split()
            if len(fields) < 2:
                raise SystemExit(f"Malformed GT line {line_number}: missing timestamp")
            try:
                timestamp = float(fields[1])
            except ValueError as exc:
                raise SystemExit(f"Invalid GT timestamp in line {line_number}") from exc
            coordinates = np.fromstring(coordinates_text, sep=" ", dtype=np.float64)
            if coordinates.size != 26:
                raise SystemExit(
                    f"GT line {line_number} has {coordinates.size} coordinates; expected 26"
                )
            timestamps.append(timestamp)
            skeletons.append(coordinates.reshape(13, 2))

    if not timestamps:
        raise SystemExit(f"GT file contains no skeletons: {path}")
    ts = np.asarray(timestamps, dtype=np.float64)
    xy = np.asarray(skeletons, dtype=np.float64)
    if np.any(np.diff(ts) < 0.0):
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        xy = xy[order]
    return ts, xy


with pred_csv.open("r", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit("Prediction CSV has no rows")

pred_ts = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
pred_xy = np.zeros((len(rows), 13, 2), dtype=np.float64)
for row_index, row in enumerate(rows):
    for joint in range(13):
        pred_xy[row_index, joint, 0] = float(row[f"joint{joint}_x"])
        pred_xy[row_index, joint, 1] = float(row[f"joint{joint}_y"])

gt_ts, gt_xy = read_gt_skeletons(gt_file)

right = np.searchsorted(gt_ts, pred_ts, side="left")
right = np.clip(right, 0, gt_ts.size - 1)
left = np.clip(right - 1, 0, gt_ts.size - 1)
use_left = np.abs(gt_ts[left] - pred_ts) < np.abs(gt_ts[right] - pred_ts)
indices = right.copy()
indices[use_left] = left[use_left]

matched_gt = gt_xy[indices]
time_error = np.abs(gt_ts[indices] - pred_ts)
valid = (
    np.isfinite(pred_xy).all(axis=2)
    & np.isfinite(matched_gt).all(axis=2)
    & (matched_gt[:, :, 0] >= 0.0)
    & (matched_gt[:, :, 0] < width)
    & (matched_gt[:, :, 1] >= 0.0)
    & (matched_gt[:, :, 1] < height)
)

if not np.any(valid):
    raise SystemExit("No valid timestamp-matched GT joints were found")

distance = np.linalg.norm(pred_xy - matched_gt, axis=2)
valid_distance = distance[valid]

print(
    "METRICS rows={} valid_joints={} mean_dt_s={:.6f} "
    "mpjpe_px={:.3f} pck@5={:.4f} pck@10={:.4f} pck@20={:.4f}".format(
        pred_xy.shape[0],
        valid_distance.size,
        float(np.mean(time_error)),
        float(np.mean(valid_distance)),
        float(np.mean(valid_distance <= 5.0)),
        float(np.mean(valid_distance <= 10.0)),
        float(np.mean(valid_distance <= 20.0)),
    )
)
PY
fi

echo
echo "Completed."
echo "Prediction CSV    : $PRED_CSV"
echo "C++ run log      : $RUN_LOG"
echo "Python server log : $SERVER_LOG"
if [[ "$GENERATE_VIDEO" == "true" ]]; then
  if [[ "$KEEP_BASE_VIDEO" == "true" || "$WITH_GT_OVERLAY" != "true" ]]; then
    echo "Prediction video  : $BASE_VIDEO"
  fi
  if [[ "$WITH_GT_OVERLAY" == "true" ]]; then
    echo "GT overlay video  : $FINAL_VIDEO"
  fi
fi