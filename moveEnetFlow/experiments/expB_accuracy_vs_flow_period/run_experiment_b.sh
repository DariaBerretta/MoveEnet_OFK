#!/usr/bin/env bash
set -euo pipefail

# Experiment B: accuracy vs flow_period, for eH36M or DHP19.
# Runs MoveEnet+OFK for each:
#   dataset sequence x network_period x flow_period

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Defaults that can be overridden from the command line
# -----------------------------------------------------------------------------
DATASET="dhp19"                         # allowed: eh36m, dhp19
BINARY="/home/moveEnetFlow/build/moveEnetOFK_offline"

DATA_FILE=""                            # optional single */chXdvs/data.log file
DATA_GLOB=""                            # pattern used by find under DATA_ROOT
DATA_ROOT=""
CHECKPOINT_PATH=""
RAW_DIR=""
LOG_DIR=""
IMG_W=""
IMG_H=""

# Network periods in seconds.
# Defaults match the historical Experiment B runs: 50 Hz, 20 Hz, 10 Hz.
NETWORK_PERIODS=("0.02" "0.05" "0.1")

# Optical-flow periods in seconds.
FLOW_PERIODS=("0.005" "0.01" "0.02" "0.05" "0.1")

OUTPUT_PERIOD="0.005"                   # CSV output sampling period, seconds
DEVICE="cuda:0"

# Kalman filter / OFK parameters
PROC_U="0.77"
MEAS_UD="0.06"
MEAS_UV="0.97"
ROI="20"

# Feature flags
USE_LC="false"
INCLUDE_VELOCITIES="false"
GPU_PERIOD_MS="5"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Dataset presets:
  --dataset <eh36m|dhp19>       Dataset preset. Default: dhp19

Paths:
  --binary <path>               Path to moveEnetOFK_offline binary
  --data_root <path>            Dataset root containing event files
  --data_file <path>            Optional single DVS file override, e.g. .../ch0dvs/data.log
  --data_glob <pattern>         find -path pattern under data_root
  --checkpoint_path <path>      MoveNet checkpoint
  --raw_dir <path>              Output directory for raw CSV results
  --log_dir <path>              Output directory for logs

Timing:
  --network_periods <list>      Comma-separated network periods. Default: 0.02,0.05,0.1
  --periods <list>              Alias for --network_periods
  --net_period <float>          Run a single network period
  --network_period <float>      Alias for --net_period
  --flow_periods <list>         Comma-separated flow periods. Default: 0.005,0.01,0.02,0.05,0.1
  --flow_period <float>         Run a single flow period
  --output_period <float>       CSV output period. Default: 0.005

Image / device:
  --w <int>                     Image width. Dataset preset default if omitted
  --h <int>                     Image height. Dataset preset default if omitted
  --device <string>             MoveNet device, e.g. cpu, cuda:0, cuda:1. Default: cuda:0

Kalman / OFK:
  --pu <float>                  KF process uncertainty. Default: 0.77
  --muD <float>                 KF position measurement uncertainty. Default: 0.06
  --muV <float>                 KF velocity measurement uncertainty. Default: 0.97
  --roi <int>                   Velocity ROI size. Default: 20
  --use_lc                      Enable latency compensation
  --include_velocities          Add velocity columns to CSV

Monitoring:
  --gpu_period_ms <int>         GPU monitor sampling period. Default: 5

Other:
  --help                        Show this help
USAGE
}

split_csv_into_array() {
  local csv="$1"
  local -n out_array="$2"
  IFS=',' read -r -a out_array <<< "$csv"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --data_file) DATA_FILE="$2"; shift 2 ;;
    --data_glob) DATA_GLOB="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --raw_dir) RAW_DIR="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;

    --network_periods|--periods) split_csv_into_array "$2" NETWORK_PERIODS; shift 2 ;;
    --net_period|--network_period) NETWORK_PERIODS=("$2"); shift 2 ;;
    --flow_periods) split_csv_into_array "$2" FLOW_PERIODS; shift 2 ;;
    --flow_period) FLOW_PERIODS=("$2"); shift 2 ;;
    --output_period) OUTPUT_PERIOD="$2"; shift 2 ;;

    --w) IMG_W="$2"; shift 2 ;;
    --h) IMG_H="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;

    --pu) PROC_U="$2"; shift 2 ;;
    --muD) MEAS_UD="$2"; shift 2 ;;
    --muV) MEAS_UV="$2"; shift 2 ;;
    --roi) ROI="$2"; shift 2 ;;
    --use_lc) USE_LC="true"; shift 1 ;;
    --include_velocities) INCLUDE_VELOCITIES="true"; shift 1 ;;
    --gpu_period_ms) GPU_PERIOD_MS="$2"; shift 2 ;;

    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------------
# Dataset presets
# -----------------------------------------------------------------------------
DATASET_ARGS=()
case "$DATASET" in
  eh36m)
    DATA_ROOT="${DATA_ROOT:-/data/eh36m_testing_set_S9S11/events}"
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth}"
    RAW_DIR="${RAW_DIR:-$EXP_DIR/eh36m_full_test/results/raw}"
    LOG_DIR="${LOG_DIR:-$EXP_DIR/eh36m_full_test/results/logs}"
    IMG_W="${IMG_W:-640}"
    IMG_H="${IMG_H:-480}"
    DATA_GLOB="${DATA_GLOB:-*/ch0dvs/data.log}"
    DATASET_ARGS=()
    ;;

  dhp19)
    DATA_ROOT="${DATA_ROOT:-/data/DHP19_subset/raw}"
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-/usr/local/src/hpe-core/example/movenet/models/dhp19_allcams_e33_valacc0.87996.pth}"
    RAW_DIR="${RAW_DIR:-$EXP_DIR/dhp19_full_test/results/raw}"
    LOG_DIR="${LOG_DIR:-$EXP_DIR/dhp19_full_test/results/logs}"
    IMG_W="${IMG_W:-346}"
    IMG_H="${IMG_H:-260}"
    DATA_GLOB="${DATA_GLOB:-*/ch*dvs/data.log}"
    DATASET_ARGS=(--dhp19)
    ;;

  *)
    echo "Unknown dataset: $DATASET. Use eh36m or dhp19." >&2
    exit 1
    ;;
esac

# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  echo "Build first, e.g. in /home/moveEnetFlow/build: cmake .. && make -j" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
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

if [[ ${#NETWORK_PERIODS[@]} -eq 0 ]]; then
  echo "No network periods specified." >&2
  exit 1
fi

if [[ ${#FLOW_PERIODS[@]} -eq 0 ]]; then
  echo "No flow periods specified." >&2
  exit 1
fi

mkdir -p "$RAW_DIR" "$LOG_DIR"

# -----------------------------------------------------------------------------
# Input discovery
# -----------------------------------------------------------------------------
declare -a LOG_FILES=()
if [[ -n "$DATA_FILE" ]]; then
  LOG_FILES+=("$DATA_FILE")
else
  mapfile -t LOG_FILES < <(find "$DATA_ROOT" -type f -path "$DATA_GLOB" | sort)
fi

if [[ ${#LOG_FILES[@]} -eq 0 ]]; then
  echo "No input files found with pattern: $DATA_GLOB" >&2
  exit 1
fi

make_rel_stem() {
  local log_file="$1"
  local rel_path

  if [[ -n "$DATA_FILE" ]]; then
    rel_path="$(basename "$(dirname "$(dirname "$log_file")")")/$(basename "$(dirname "$log_file")")"
  else
    rel_path="$(realpath --relative-to="$DATA_ROOT" "$log_file")"
    rel_path="${rel_path%/data.log}"
  fi

  rel_path="${rel_path//\//__}"
  rel_path="${rel_path// /_}"
  echo "$rel_path"
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "Experiment B: accuracy vs flow_period and network_period"
echo "Dataset        : $DATASET"
echo "Binary         : $BINARY"
echo "Checkpoint     : $CHECKPOINT_PATH"
echo "Device         : $DEVICE"
echo "Image size     : ${IMG_W}x${IMG_H}"
if [[ -n "$DATA_FILE" ]]; then
  echo "Data mode      : single file"
  echo "Data file      : $DATA_FILE"
else
  echo "Data mode      : dataset"
  echo "Data root      : $DATA_ROOT"
  echo "Data glob      : $DATA_GLOB"
fi
echo "Network periods: ${NETWORK_PERIODS[*]}"
echo "Flow periods   : ${FLOW_PERIODS[*]}"
echo "Output period  : $OUTPUT_PERIOD"
echo "Raw dir        : $RAW_DIR"
echo "Log dir        : $LOG_DIR"
echo "Sequences found: ${#LOG_FILES[@]}"

# -----------------------------------------------------------------------------
# Main experiment grid
# -----------------------------------------------------------------------------
for NP in "${NETWORK_PERIODS[@]}"; do
  SAFE_NP="${NP//./p}"
  DET_RATE="$(awk "BEGIN{printf \"%.3f\", 1/$NP}")"

  for FP in "${FLOW_PERIODS[@]}"; do
    SAFE_FP="${FP//./p}"
    RUN_DIR="$RAW_DIR/net_${SAFE_NP}/fp_${SAFE_FP}"
    RUN_LOG_DIR="$LOG_DIR/net_${SAFE_NP}/fp_${SAFE_FP}"
    mkdir -p "$RUN_DIR" "$RUN_LOG_DIR"

    FLOW_RATE="$(awk "BEGIN{printf \"%.3f\", 1/$FP}")"

    echo ""
    echo "[network_period=${NP}s (${DET_RATE} Hz), flow_period=${FP}s (${FLOW_RATE} Hz)]"

    for LOG_FILE in "${LOG_FILES[@]}"; do
      REL_STEM="$(make_rel_stem "$LOG_FILE")"

      OUT_OFK="$RUN_DIR/${REL_STEM}__moveenet_ofk_fp_${SAFE_FP}.csv"
      LOG_OFK="$RUN_LOG_DIR/${REL_STEM}__moveenet_ofk_np_${SAFE_NP}_fp_${SAFE_FP}.log"
      GPU_OFK="$RUN_LOG_DIR/${REL_STEM}__moveenet_ofk_np_${SAFE_NP}_fp_${SAFE_FP}_gpu.csv"

      COMMON_ARGS=(
        --data_file "$LOG_FILE"
        --output_period "$OUTPUT_PERIOD"
        --net_period "$NP"
        --flow_period "$FP"
        --w "$IMG_W"
        --h "$IMG_H"
        --pu "$PROC_U"
        --muD "$MEAS_UD"
        --muV "$MEAS_UV"
        --roi "$ROI"
        --checkpoint_path "$CHECKPOINT_PATH"
        --device "$DEVICE"
        --gpu_period_ms "$GPU_PERIOD_MS"
        --gpu_file "$GPU_OFK"
        --no_video
        "${DATASET_ARGS[@]}"
      )

      if [[ "$USE_LC" == "true" ]]; then
        COMMON_ARGS+=(--use_lc true)
      fi

      if [[ "$INCLUDE_VELOCITIES" == "true" ]]; then
        COMMON_ARGS+=(--include_velocities)
      fi

      echo "  -> [$REL_STEM] MoveEnet + OFK"
      "$BINARY" "${COMMON_ARGS[@]}" \
        --output_csv "$OUT_OFK" \
        > "$LOG_OFK" 2>&1
    done
  done
done

echo ""
echo "Experiment completed."
echo "CSV files: $RAW_DIR"
echo "Logs     : $LOG_DIR"
