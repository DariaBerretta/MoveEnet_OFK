#!/bin/bash
# Batch processing script for moveEnetOFK_offline
# Processes all data.log files in subdirectories and creates corresponding CSV files

# Usage: ./run_batch_processing.sh [options]

# Default parameters
DATA_ROOT="/data/moveEnet_test/raw"
OUTPUT_DIR="/home/moveEnetFlow/csv_file"

# moveEnetOFK_offline parameters
OUT_PERIOD=0.005
NET_PERIOD=0.1
FLOW_PERIOD=0.001
IMG_W=640
IMG_H=480
PROC_U=1e-1
MEAS_UD=1e-4
MEAS_UV=0.0
ROI=20
USE_LC=false
INCLUDE_VELOCITIES=false
NO_CSV=false
NO_VIDEO=true
CHECKPOINT_PATH="/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth"
USE_POWERJOULAR=false
POWERJOULAR_DIR=""
USE_GPU_MONITOR=false
GPU_MONITOR_PERIOD_MS=200
GPU_MONITOR_DIR=""
GPU_INDEX=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --out_period)
      OUT_PERIOD="$2"
      shift 2
      ;;
    --net_period)
      NET_PERIOD="$2"
      shift 2
      ;;
    --flow_period)
      FLOW_PERIOD="$2"
      shift 2
      ;;
    --w)
      IMG_W="$2"
      shift 2
      ;;
    --h)
      IMG_H="$2"
      shift 2
      ;;
    --pu)
      PROC_U="$2"
      shift 2
      ;;
    --muD)
      MEAS_UD="$2"
      shift 2
      ;;
    --muV)
      MEAS_UV="$2"
      shift 2
      ;;
    --roi)
      ROI="$2"
      shift 2
      ;;
    --use_lc)
      USE_LC=true
      shift 1
      ;;
    --include_velocities)
      INCLUDE_VELOCITIES=true
      shift 1
      ;;
    --no_csv)
      NO_CSV=true
      shift 1
      ;;
    --no_video)
      NO_VIDEO=true
      shift 1
      ;;
    --checkpoint_path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --use_powerjoular)
      USE_POWERJOULAR=true
      shift 1
      ;;
    --powerjoular_dir)
      POWERJOULAR_DIR="$2"
      shift 2
      ;;
    --powerjoular_period)
      echo "Warning: --powerjoular_period is deprecated and ignored (sampling is fixed by PowerJoular)."
      shift 2
      ;;
    --use_gpu_monitor)
      USE_GPU_MONITOR=true
      shift 1
      ;;
    --gpu_monitor_period_ms)
      GPU_MONITOR_PERIOD_MS="$2"
      shift 2
      ;;
    --gpu_monitor_dir)
      GPU_MONITOR_DIR="$2"
      shift 2
      ;;
    --gpu_index)
      GPU_INDEX="$2"
      shift 2
      ;;
    --help)
      echo "Batch processing script for moveEnetOFK_offline"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --data_root <path>    Root directory containing subdirectories with data.log files (default: /data/new_scarfGNN_full/raw)"
      echo "  --output_dir <path>   Directory to save output CSV files (default: /home/moveEnetFlow/csv_file)"
      echo "  --out_period <float>  Output period in seconds (default: 0.005)"
      echo "  --net_period <float>  Network update period in seconds (default: 0.005)"
      echo "  --flow_period <float> Optical flow update period in seconds (default: 0.005)"
      echo "  --w <int>             Image width (default: 640)"
      echo "  --h <int>             Image height (default: 480)"
      echo "  --pu <float>          KF process uncertainty (default: 1e-1)"
      echo "  --muD <float>         KF measurement uncertainty (position) (default: 1e-4)"
      echo "  --muV <float>         KF measurement uncertainty (velocity) (default: 0.0)"
      echo "  --roi <int>           ROI size for velocity estimation (default: 20)"
      echo "  --use_lc              Enable latency compensation"
      echo "  --include_velocities  Include velocities in CSV output"
      echo "  --no_csv              Skip CSV logging (for debugging)"
      echo "  --no_video            Disable video output"
      echo "  --checkpoint_path <path>  MoveNet checkpoint path"
      echo "  --use_powerjoular      Enable PowerJoular profiling per run"
      echo "  --powerjoular_dir <path>  Directory for PowerJoular output files (default: <out_subdir>/powerjoular)"
      echo "  --use_gpu_monitor      Enable NVIDIA GPU telemetry logging per run"
      echo "  --gpu_monitor_period_ms <int>  GPU telemetry sampling period in ms (default: 200)"
      echo "  --gpu_monitor_dir <path>  Directory for GPU CSV output files (default: <out_subdir>/gpu_monitor)"
      echo "  --gpu_index <int>      GPU index to monitor with nvidia-smi (default: 0)"
      echo "  --help                Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --data_root /data/my_dataset/raw --output_dir /home/results --out_period 0.01 --net_period 0.05 --flow_period 0.02"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "Starting batch processing..."
echo "Data root: $DATA_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo "Output period: ${OUT_PERIOD} s"
echo "Net period: ${NET_PERIOD} s"
echo "Flow period: ${FLOW_PERIOD} s"
echo "Image: ${IMG_W}x${IMG_H}"
echo "KF params: pu=${PROC_U}, muD=${MEAS_UD}, muV=${MEAS_UV}"
echo "ROI: ${ROI}"
echo "Latency compensation: ${USE_LC}"
echo "Include velocities: ${INCLUDE_VELOCITIES}"
echo "No CSV: ${NO_CSV}"
echo "No video: ${NO_VIDEO}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "PowerJoular enabled: ${USE_POWERJOULAR}"
echo "GPU monitor enabled: ${USE_GPU_MONITOR}"
echo "GPU monitor period: ${GPU_MONITOR_PERIOD_MS} ms"
echo "GPU index: ${GPU_INDEX}"
echo ""

# Ensure output directory exists and create param-based subdir
OUT_SUBDIR="${OUTPUT_DIR}/moveEnetOFK_test/op${OUT_PERIOD}_np${NET_PERIOD}_fp${FLOW_PERIOD}"
mkdir -p "$OUT_SUBDIR"

if [[ "$USE_POWERJOULAR" == "true" ]]; then
  if ! command -v powerjoular >/dev/null 2>&1; then
    echo "PowerJoular not found in PATH. Install it or run without --use_powerjoular."
    exit 1
  fi
  if [[ -z "$POWERJOULAR_DIR" ]]; then
    POWERJOULAR_DIR="${OUT_SUBDIR}/powerjoular"
  fi
  mkdir -p "$POWERJOULAR_DIR"
fi

if [[ "$USE_GPU_MONITOR" == "true" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found in PATH. Install NVIDIA tools or run without --use_gpu_monitor."
    exit 1
  fi
  if [[ -z "$GPU_MONITOR_DIR" ]]; then
    GPU_MONITOR_DIR="${OUT_SUBDIR}/gpu_monitor"
  fi
  mkdir -p "$GPU_MONITOR_DIR"
fi

# Run the batch processing (find all data.log files)
mapfile -t LOG_FILES < <(find "$DATA_ROOT" -type f -path "*/ch0dvs/data.log" | sort)

if [[ ${#LOG_FILES[@]} -eq 0 ]]; then
  echo "No data.log files found under: $DATA_ROOT"
  exit 1
fi

for LOG_FILE in "${LOG_FILES[@]}"; do
  REL_PATH=$(realpath --relative-to="$DATA_ROOT" "$LOG_FILE")
  REL_STEM="${REL_PATH%/data.log}"
  REL_SAFE="${REL_STEM//\//_}"
  REL_SAFE="${REL_SAFE%_ch0dvs}"  # Remove _ch0dvs suffix
  OUT_CSV="${OUT_SUBDIR}/${REL_SAFE}.csv"

  echo "Processing: $LOG_FILE"
  echo "Output CSV: $OUT_CSV"

  if [[ "$USE_POWERJOULAR" == "true" ]]; then
    PJ_BASE="${POWERJOULAR_DIR}/${REL_SAFE}"
    echo "Power log base: ${PJ_BASE}"
  else
    PJ_BASE=""
  fi

  if [[ "$USE_GPU_MONITOR" == "true" ]]; then
    GPU_LOG="${GPU_MONITOR_DIR}/${REL_SAFE}.csv"
    echo "GPU log file: $GPU_LOG"
  else
    GPU_LOG=""
  fi

  CMD=("/home/moveEnetFlow/build/moveEnetOFK_offline"
    --data_file "$LOG_FILE"
    --output_csv "$OUT_CSV"
    --output_period "$OUT_PERIOD"
    --net_period "$NET_PERIOD"
    --flow_period "$FLOW_PERIOD"
    --w "$IMG_W"
    --h "$IMG_H"
    --pu "$PROC_U"
    --muD "$MEAS_UD"
    --muV "$MEAS_UV"
    --roi "$ROI"
    --checkpoint_path "$CHECKPOINT_PATH"
    --pwrjlr_file "$PJ_BASE"
    --gpu_file "$GPU_LOG"
    --gpu_period_ms "$GPU_MONITOR_PERIOD_MS"
    --gpu_index "$GPU_INDEX"
  )

  if [[ "$USE_LC" == "true" ]]; then
    CMD+=(--use_lc)
  fi
  if [[ "$INCLUDE_VELOCITIES" == "true" ]]; then
    CMD+=(--include_velocities)
  fi
  if [[ "$NO_CSV" == "true" ]]; then
    CMD+=(--no_csv)
  fi
  if [[ "$NO_VIDEO" == "true" ]]; then
    CMD+=(--no_video)
  fi
  "${CMD[@]}"
  RUN_STATUS=$?
  if [[ $RUN_STATUS -ne 0 ]]; then
    echo "Processing failed (exit ${RUN_STATUS}): $LOG_FILE"
    exit "$RUN_STATUS"
  fi
done

echo ""
echo "Batch processing completed!"
echo "CSV files saved in: $OUT_SUBDIR"
if [[ "$USE_POWERJOULAR" == "true" ]]; then
  echo "Power logs saved in: $POWERJOULAR_DIR"
fi
if [[ "$USE_GPU_MONITOR" == "true" ]]; then
  echo "GPU logs saved in: $GPU_MONITOR_DIR"
fi
