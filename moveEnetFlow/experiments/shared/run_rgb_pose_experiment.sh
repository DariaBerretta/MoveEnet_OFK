#!/usr/bin/env bash
set -euo pipefail

# Shared RGB-pose batch runner. POSE_METHOD=openpose|yolo selects the binary.
EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${MOVENET_RESULTS_ROOT:-/data/MovEnet_OFK_results}"
POSE_METHOD="${POSE_METHOD:-openpose}"
case "$POSE_METHOD" in
  openpose) EXP_DIR="$EXPERIMENTS_DIR/ExpA/OP_expA_accuracy_vs_network_period"; RESULT_DIR="$RESULTS_ROOT/OP_h36m_full_test"; BINARY="/workspace/moveEnetFlow/build/OpenPose_offline" ;;
  yolo) EXP_DIR="$EXPERIMENTS_DIR/ExpA/YOLO_expA_accuracy_vs_network_period"; RESULT_DIR="$RESULTS_ROOT/YOLO_h36m_full_test"; BINARY="/workspace/moveEnetFlow/build/YoloPose_offline" ;;
  *) echo "Unknown POSE_METHOD: $POSE_METHOD" >&2; exit 2 ;;
esac
RAW_DIR="$RESULT_DIR/results/raw"                                    # Output directory for raw CSV results
MP4_DIR="$RESULT_DIR/results/logs"                                   # Output directory for execution logs

# Network periods to test (seconds) — corresponds to detection rates: 100Hz, 50Hz, 20Hz, 10Hz, 5Hz, 2Hz
# Network periods in seconds: 100 Hz, 50 Hz, 20 Hz, 10 Hz, 5 Hz, 2 Hz
PERIODS=("0.02" "0.05" "0.1" "0.2" "0.5")

# Core executable and data paths
# Use h36m RGB mp4 dataset by default
DATA_ROOT="/data/eh36m_testing_set_S9S11/rgb"                       # Root directory containing MP4 datasets (h36m)
DATA_FILE=""                                                        # Optional single video override (empty = use all in DATA_ROOT)
DEVICE="cuda:0"                                                     # OpenPose device (GPU required). Use e.g. cuda:0

# Timing and processing parameters
OUTPUT_PERIOD="0.02"                                              # CSV output sampling period in seconds (minimum 0.02)
IMG_W="640"                                                        # Event camera image width in pixels
IMG_H="480"                                                        # Event camera image height in pixels


usage() {
  cat << USAGE
Usage: $(basename "$0") [options]

Options:
  --binary <path>            Pose executable
  --periods <list>           Comma-separated network periods
  --net_period <float>       Run a single network period
  --data_root <path>         Dataset root containing .mp4 sequences
  --data_file <path>         Optional single .mp4 file override
  --raw_dir <path>           Output directory for raw CSV results
  --log_dir <path>           Output directory for execution logs
  --output_period <float>    CSV output period (default: 0.02)
  --w <int>                  Image width (default: 640)
  --h <int>                  Image height (default: 480)
  --device <string>          Device for OpenPose (required, e.g. cuda:0)
  --help                     Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --data_file) DATA_FILE="$2"; shift 2 ;;
    --raw_dir) RAW_DIR="$2"; shift 2 ;;
    --log_dir) MP4_DIR="$2"; shift 2 ;;
    --periods) IFS=',' read -r -a PERIODS <<< "$2"; shift 2 ;;
    --net_period) PERIODS=("$2"); shift 2 ;;
    --output_period) OUTPUT_PERIOD="$2"; shift 2 ;;
    --w) IMG_W="$2"; shift 2 ;;
    --h) IMG_H="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if awk "BEGIN{exit !($OUTPUT_PERIOD >= 0.02)}"; then
  : # ok
else
  echo "Pose runner cannot log faster than 0.02s per row; clamping output_period to 0.02" >&2
  OUTPUT_PERIOD="0.02"
fi

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found/executable: $BINARY" >&2
  echo "Build first, e.g. in /workspace/moveEnetFlow/build: cmake .. && make -j" >&2
  exit 1
fi
if [[ -n "$DATA_FILE" && ! -f "$DATA_FILE" ]]; then
  echo "Data file not found: $DATA_FILE" >&2
  exit 1
fi
if [[ -z "$DATA_FILE" && ! -d "$DATA_ROOT" ]]; then
  echo "Data root not found: $DATA_ROOT" >&2
  exit 1
fi

# Require GPU device (don't allow CPU for OpenPose runs in this experiment)
if [[ -z "$DEVICE" ]]; then
  echo "Device not set. Please specify a GPU device with --device, e.g. cuda:0" >&2
  exit 1
fi
dev_l="$DEVICE"
dev_l="${dev_l,,}"
if [[ "$dev_l" == cpu* || "$dev_l" == "cpu" ]]; then
  echo "GPU required for $POSE_METHOD. Device set to CPU: $DEVICE" >&2
  exit 1
fi

mkdir -p "$RAW_DIR" "$MP4_DIR"

echo "Experiment A: $POSE_METHOD accuracy vs network_period"
echo "Binary        : $BINARY"
if [[ -n "$DATA_FILE" ]]; then
  echo "Data mode     : single file"
  echo "Data file     : $DATA_FILE"
else
  echo "Data mode     : dataset"
  echo "Data root     : $DATA_ROOT"
fi
echo "Output period : $OUTPUT_PERIOD"
echo "Results dir   : $RAW_DIR"

declare -a MP4_FILES=()
if [[ -n "$DATA_FILE" ]]; then
  MP4_FILES+=("$DATA_FILE")
else
  mapfile -t MP4_FILES < <(find "$DATA_ROOT" -type f -name "*.mp4" | sort)
fi

if [[ ${#MP4_FILES[@]} -eq 0 ]]; then
  echo "No input files found." >&2
  exit 1
fi

echo "Sequences found: ${#MP4_FILES[@]}"

for NP in "${PERIODS[@]}"; do
  SAFE_NP="${NP//./p}"
  NP_DIR="$RAW_DIR/np_${SAFE_NP}"
  mkdir -p "$NP_DIR"

  echo ""
  echo "[network_period=${NP}s, detection_rate=$(awk "BEGIN{printf \"%.3f\", 1/$NP}") Hz]"

  for MP4_FILE in "${MP4_FILES[@]}"; do
    if [[ -n "$DATA_FILE" ]]; then
      REL_STEM="$(basename "$MP4_FILE" .mp4)"
    else
      REL_PATH="$(realpath --relative-to="$DATA_ROOT" "$MP4_FILE")"
      REL_STEM="${REL_PATH%.mp4}"
      REL_STEM="${REL_STEM//\//__}"
    fi

    OUT_CSV="$NP_DIR/${REL_STEM}_${POSE_METHOD}_np_${SAFE_NP}.csv"
    LOG_FILE="$MP4_DIR/${REL_STEM}_${POSE_METHOD}_np_${SAFE_NP}.log"

    COMMON_ARGS=(
      --data_file "$MP4_FILE"
      --output_period "$OUTPUT_PERIOD"
      --net_period "$NP"
      --w "$IMG_W"
      --h "$IMG_H"
      --no_video
      --device "$DEVICE"
    )

    echo "  -> [$REL_STEM] $POSE_METHOD"
    "$BINARY" "${COMMON_ARGS[@]}" --output_csv "$OUT_CSV" \
      > "$LOG_FILE" 2>&1
  done

done

echo ""
echo "Experiment completed. CSV files are in: $RAW_DIR"
echo "Next: open the retained combined analysis notebook."
