#!/usr/bin/env bash
set -euo pipefail

# Experiment directory structure
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"           # Current experiment directory --> {BASH_SOURCE[0]} is the path to the script itself.
RAW_DIR="$EXP_DIR/results/raw"                                    # Output directory for raw CSV results
LOG_DIR="$EXP_DIR/results/logs"                                   # Output directory for execution logs

# Network periods (MoveEnet) to test (in seconds) - corresponds to detection rates: 100Hz, 50Hz, 20Hz, 10Hz, 5Hz, 2Hz
PERIODS=("0.01" "0.02" "0.05" "0.1" "0.2" "0.5")

# Core executable and data paths
BINARY="/home/moveEnetFlow/build/moveEnetOFK_offline"              # MoveEnet + Optical Flow Kalman binary
DATA_ROOT="/data/moveEnet_test/raw/"                               # Root directory containing event-based datasets
DATA_FILE=""                                                       # Optional single DVS file override (empty = use all in DATA_ROOT)
CHECKPOINT_PATH="/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth"  # Pre-trained MoveNet model weights

# Timing and processing parameters
FLOW_PERIOD="0.001"                                                # Optical flow update period in seconds (1ms = 1kHz)
OUTPUT_PERIOD="0.005"                                              # CSV output sampling period in seconds (5ms = 200Hz)
IMG_W="640"                                                        # Event camera image width in pixels
IMG_H="480"                                                        # Event camera image height in pixels

# Kalman filter parameters for pose tracking
# PROC_U="1e-1"                                                      # Process noise uncertainty (motion model uncertainty)
PROC_U="0.77"                                                        
# MEAS_UD="1e-4"                                                     # Position measurement uncertainty (detection accuracy)
MEAS_UD="0.06"                                                    
# MEAS_UV="0.0"                                                      # Velocity measurement uncertainty (set to 0 = no direct velocity measurements)
MEAS_UV="0.97"                                                      
ROI="20"                                                             # Region of interest size for velocity estimation (pixels)

# Feature flags
USE_LC="false"                                                     # Enable/disable latency compensation

usage() {
  cat << USAGE
Usage: $(basename "$0") [options]

Options:
  --binary <path>            Path to moveEnetOFK_offline binary
  --data_root <path>         Dataset root containing */ch0dvs/data.log (default: /data/moveEnet_test/raw/)
  --data_file <path>         Optional single DVS file override (ch0dvs/data.log)
  --checkpoint_path <path>   MoveNet checkpoint
  --flow_period <float>      Optical-flow update period (default: 0.001)
  --output_period <float>    CSV output period (default: 0.005)
  --w <int>                  Image width (default: 640)
  --h <int>                  Image height (default: 480)
  --pu <float>               KF process uncertainty (default: 0.77)
  --muD <float>              KF position measurement uncertainty (default: 0.06)
  --muV <float>              KF velocity measurement uncertainty (default: 0.97)
  --roi <int>                Velocity ROI size (default: 20)
  --use_lc                   Enable latency compensation
  --help                     Show this help
USAGE
}

while [[ $# -gt 0 ]]; do                                    # rocesses command-line arguments passed to the script
  case "$1" in
    --binary) BINARY="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --data_file) DATA_FILE="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --flow_period) FLOW_PERIOD="$2"; shift 2 ;;
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

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found/executable: $BINARY" >&2
  echo "Build first, e.g. in /home/moveEnetFlow/build: cmake .. && make -j" >&2
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

echo "Experiment A: Accuracy vs network_period"
echo "Binary        : $BINARY"
if [[ -n "$DATA_FILE" ]]; then
  echo "Data mode     : single file"
  echo "Data file     : $DATA_FILE"
else
  echo "Data mode     : dataset"
  echo "Data root     : $DATA_ROOT"
fi
echo "Flow period   : $FLOW_PERIOD"
echo "Output period : $OUTPUT_PERIOD"
echo "Results dir   : $RAW_DIR"

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

for NP in "${PERIODS[@]}"; do
  SAFE_NP="${NP//./p}"
  NP_DIR="$RAW_DIR/np_${SAFE_NP}"
  mkdir -p "$NP_DIR"

  echo ""
  echo "[network_period=${NP}s, detection_rate=$(awk "BEGIN{printf \"%.3f\", 1/$NP}") Hz]"

  for LOG_FILE in "${LOG_FILES[@]}"; do
    if [[ -n "$DATA_FILE" ]]; then
      REL_STEM="$(basename "$(dirname "$(dirname "$LOG_FILE")")")"
    else
      REL_PATH="$(realpath --relative-to="$DATA_ROOT" "$LOG_FILE")"
      REL_STEM="${REL_PATH%/ch0dvs/data.log}"
      REL_STEM="${REL_STEM//\//__}"
    fi

    OUT_OFK="$NP_DIR/${REL_STEM}_moveenet_ofk_np_${SAFE_NP}.csv"
    OUT_MN="$NP_DIR/${REL_STEM}_moveenet_np_${SAFE_NP}.csv"

    COMMON_ARGS=(
      --data_file "$LOG_FILE"
      --output_period "$OUTPUT_PERIOD"
      --net_period "$NP"
      --flow_period "$FLOW_PERIOD"
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
      > "$LOG_DIR/${REL_STEM}_ofk_np_${SAFE_NP}.log" 2>&1

    echo "  -> [$REL_STEM] MoveEnet only"
    "$BINARY" "${COMMON_ARGS[@]}" --moveenet_only --output_csv "$OUT_MN" \
      > "$LOG_DIR/${REL_STEM}_moveenet_only_np_${SAFE_NP}.log" 2>&1
  done

done

echo ""
echo "Experiment completed. CSV files are in: $RAW_DIR"
echo "Next: open notebooks/ExperimentA_MPJPE_vs_network_period.ipynb"
