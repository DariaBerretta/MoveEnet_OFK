#!/usr/bin/env bash
set -euo pipefail

# Experiment directory structure
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_BASE_DIR="$EXP_DIR/results/raw"
LOG_BASE_DIR="$EXP_DIR/results/logs"

# Flow periods to test (in seconds)
FLOW_PERIODS=("0.001" "0.002" "0.005" "0.01" "0.02" "0.05" "0.1")

# Core executable and data paths
BINARY="/home/moveEnetFlow/build/moveEnetOFK_offline"
DATA_ROOT="/data/moveEnet_test/raw/"
DATA_FILE=""  # Optional single DVS file override (empty = use all in DATA_ROOT)
CHECKPOINT_PATH="/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth"

# Fixed timing for Experiment B (as requested)
NETWORK_PERIOD="0.1"   # 50 Hz detections
OUTPUT_PERIOD="0.005"   # 200 Hz output logging

# Run tag to keep outputs from different configurations separate
RUN_TAG="net_${NETWORK_PERIOD//./p}"

# Image and KF parameters
IMG_W="640"
IMG_H="480"
PROC_U="1e-1"
MEAS_UD="1e-4"
MEAS_UV="0.0"
ROI="20"

# Feature flags
USE_LC="false"

usage() {
  cat << USAGE
Usage: $(basename "$0") [options]

Options:
  --binary <path>            Path to moveEnetOFK_offline binary
  --data_root <path>         Dataset root containing */ch0dvs/data.log (default: /data/moveEnet_test/raw/)
  --data_file <path>         Optional single DVS file override (ch0dvs/data.log)
  --checkpoint_path <path>   MoveNet checkpoint
  --run_tag <string>         Output tag folder under results/raw and results/logs
  --output_period <float>    CSV output period (default: 0.005)
  --w <int>                  Image width (default: 640)
  --h <int>                  Image height (default: 480)
  --pu <float>               KF process uncertainty (default: 1e-1)
  --muD <float>              KF position measurement uncertainty (default: 1e-4)
  --muV <float>              KF velocity measurement uncertainty (default: 0.0)
  --roi <int>                Velocity ROI size (default: 20)
  --use_lc                   Enable latency compensation
  --help                     Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --data_file) DATA_FILE="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --run_tag) RUN_TAG="$2"; shift 2 ;;
    --output_period) OUTPUT_PERIOD="$2"; shift 2 ;;
    --w) IMG_W="$2"; shift 2 ;;
    --h) IMG_H="$2"; shift 2 ;;
    --pu) PROC_U="$2"; shift 2 ;;
    --muD) MEAS_UD="$2"; shift 2 ;;
    --muV) MEAS_UV="$2"; shift 2 ;;
    --roi) ROI="$2"; shift 2 ;;
    --use_lc) USE_LC="true"; shift 1 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

RUN_TAG_SAFE="${RUN_TAG//\//_}"
RUN_TAG_SAFE="${RUN_TAG_SAFE// /_}"
RAW_DIR="$RAW_BASE_DIR/$RUN_TAG_SAFE"
LOG_DIR="$LOG_BASE_DIR/$RUN_TAG_SAFE"

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found/executable: $BINARY" >&2
  echo "Build first in /home/moveEnetFlow/build" >&2
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

mkdir -p "$RAW_DIR" "$LOG_DIR"

echo "Experiment B: Accuracy vs flow_period"
echo "Binary          : $BINARY"
if [[ -n "$DATA_FILE" ]]; then
  echo "Data mode       : single file"
  echo "Data file       : $DATA_FILE"
else
  echo "Data mode       : dataset"
  echo "Data root       : $DATA_ROOT"
fi
echo "Network period  : $NETWORK_PERIOD (fixed, 20 Hz)"
echo "Run tag         : $RUN_TAG_SAFE"
echo "Output period   : $OUTPUT_PERIOD"
echo "Results dir     : $RAW_DIR"

declare -a LOG_FILES=()
if [[ -n "$DATA_FILE" ]]; then
  LOG_FILES+=("$DATA_FILE")
else
  mapfile -t LOG_FILES < <(find "$DATA_ROOT" -type f -path "*/ch0dvs/data.log" | sort)
fi

if [[ ${#LOG_FILES[@]} -eq 0 ]]; then
  echo "No input files found." >&2
  exit 1
fi

echo "Sequences found: ${#LOG_FILES[@]}"

for FP in "${FLOW_PERIODS[@]}"; do
  SAFE_FP="${FP//./p}"
  FP_DIR="$RAW_DIR/fp_${SAFE_FP}"
  mkdir -p "$FP_DIR"

  echo ""
  echo "[flow_period=${FP}s, flow_rate=$(awk "BEGIN{printf \"%.3f\", 1/$FP}") Hz]"

  for LOG_FILE in "${LOG_FILES[@]}"; do
    if [[ -n "$DATA_FILE" ]]; then
      REL_STEM="$(basename "$(dirname "$(dirname "$LOG_FILE")")")"
    else
      REL_PATH="$(realpath --relative-to="$DATA_ROOT" "$LOG_FILE")"
      REL_STEM="${REL_PATH%/ch0dvs/data.log}"
      REL_STEM="${REL_STEM//\//__}"
    fi

    OUT_OFK="$FP_DIR/${REL_STEM}__moveenet_ofk_fp_${SAFE_FP}.csv"

    COMMON_ARGS=(
      --data_file "$LOG_FILE"
      --output_period "$OUTPUT_PERIOD"
      --net_period "$NETWORK_PERIOD"
      --flow_period "$FP"
      --w "$IMG_W"
      --h "$IMG_H"
      --pu "$PROC_U"
      --muD "$MEAS_UD"
      --muV "$MEAS_UV"
      --roi "$ROI"
      --checkpoint_path "$CHECKPOINT_PATH"
      --no_video
    )
    if [[ "$USE_LC" == "true" ]]; then
      COMMON_ARGS+=(--use_lc)
    fi

    echo "  -> [$REL_STEM] MoveEnet + OFK"
    "$BINARY" "${COMMON_ARGS[@]}" --output_csv "$OUT_OFK" \
      > "$LOG_DIR/${REL_STEM}__ofk_fp_${SAFE_FP}.log" 2>&1
  done
done

echo ""
echo "Experiment completed. CSV files are in: $RAW_DIR"
echo "Next: open notebooks/ExperimentB_MPJPE_vs_flow_period.ipynb"
