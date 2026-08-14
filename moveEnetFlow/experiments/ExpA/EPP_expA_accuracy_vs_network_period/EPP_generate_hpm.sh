#!/usr/bin/env bash
set -euo pipefail

# Generate one recording-specific hot-pixel mask for every DHP19 cam2/cam3
# data.log entry.
#
# Default input layout:
#   DATA_ROOT/S13_1_1/ch2dvs/data.log
#   DATA_ROOT/S13_1_1/ch3dvs/data.log
#
# Default output layout:
#   MASK_ROOT/S13_1_1/ch2dvs/hotpixel_mask.png
#   MASK_ROOT/S13_1_1/ch2dvs/hotpixel_pixels.csv
#   MASK_ROOT/S13_1_1/ch3dvs/hotpixel_mask.png
#   MASK_ROOT/S13_1_1/ch3dvs/hotpixel_pixels.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${MOVENET_RESULTS_ROOT:-/data/MovEnet_OFK_results}"
RESULT_DIR="$RESULTS_ROOT/EPP_dhp19_full_test"

# -----------------------------------------------------------------------------
# Defaults; every value can be overridden from the command line.
# -----------------------------------------------------------------------------
BINARY="/workspace/moveEnetFlow/build2/eventPointPose_hotpixel_mask"

DATA_FILE=""                                      # optional single data.log
DATA_ROOT="/data/dhp19_testing_set_S13toS17"
DATA_GLOB="*/ch[23]dvs/data.log"                  # cam2 and cam3 only

MASK_ROOT="$RESULT_DIR/hotpixel_masks"
LOG_DIR="$RESULT_DIR/hotpixel_mask_logs"

THRESHOLD="10000"
IMG_W="346"
IMG_H="260"
WRITE_CSV="true"
FORCE="false"
DRY_RUN="false"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Input:
  --binary <path>          Path to eventPointPose_hotpixel_mask binary
  --data_root <path>       DHP19 root containing sequence/chXdvs/data.log
  --data_file <path>       Process only one data.log entry
  --data_glob <pattern>    find -path pattern under data_root
                           Default: */ch[23]dvs/data.log

Output:
  --mask_root <path>       Root directory for generated masks and CSV files
  --log_dir <path>         Directory for one execution log per entry
  --no_csv                 Do not generate hotpixel_pixels.csv files
  --force                  Regenerate masks that already exist

Hot-pixel parameters:
  --threshold <int>        Event-count threshold. Default: 10000
  --w <int>                Sensor width. Default: 346
  --h <int>                Sensor height. Default: 260

Other:
  --dry_run                Print commands without executing them
  --help                   Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --data_file) DATA_FILE="$2"; shift 2 ;;
    --data_glob) DATA_GLOB="$2"; shift 2 ;;

    --mask_root) MASK_ROOT="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --no_csv) WRITE_CSV="false"; shift ;;
    --force) FORCE="true"; shift ;;

    --threshold) THRESHOLD="$2"; shift 2 ;;
    --w) IMG_W="$2"; shift 2 ;;
    --h) IMG_H="$2"; shift 2 ;;

    --dry_run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Validation.
# -----------------------------------------------------------------------------
if [[ ! "$THRESHOLD" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --threshold: $THRESHOLD" >&2
  exit 1
fi
if [[ ! "$IMG_W" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --w: $IMG_W" >&2
  exit 1
fi
if [[ ! "$IMG_H" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --h: $IMG_H" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "true" && ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  echo "Build eventPointPose_hotpixel_mask or pass --binary /path/to/binary." >&2
  exit 1
fi

if [[ -n "$DATA_FILE" ]]; then
  if [[ ! -f "$DATA_FILE" ]]; then
    echo "Data file not found: $DATA_FILE" >&2
    exit 1
  fi
else
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Data root not found: $DATA_ROOT" >&2
    exit 1
  fi
fi

mkdir -p "$MASK_ROOT" "$LOG_DIR"

# -----------------------------------------------------------------------------
# Discover input entries.
# -----------------------------------------------------------------------------
declare -a LOG_FILES=()
if [[ -n "$DATA_FILE" ]]; then
  LOG_FILES+=("$(realpath "$DATA_FILE")")
else
  mapfile -t LOG_FILES < <(
    find "$DATA_ROOT" -type f -path "$DATA_GLOB" -print | sort
  )
fi

if [[ ${#LOG_FILES[@]} -eq 0 ]]; then
  echo "No input entries found." >&2
  echo "DATA_ROOT: $DATA_ROOT" >&2
  echo "DATA_GLOB: $DATA_GLOB" >&2
  exit 1
fi

# Return a relative output directory such as S13_1_1/ch2dvs.
entry_relative_dir() {
  local log_file="$1"
  local rel

  if [[ -d "$DATA_ROOT" ]]; then
    rel="$(realpath --relative-to="$DATA_ROOT" "$log_file")"
    if [[ "$rel" != ../* && "$rel" != ".." ]]; then
      printf '%s\n' "${rel%/data.log}"
      return
    fi
  fi

  # Fallback for a single file outside DATA_ROOT.
  local camera_dir sequence_dir
  camera_dir="$(basename "$(dirname "$log_file")")"
  sequence_dir="$(basename "$(dirname "$(dirname "$log_file")")")"
  printf '%s/%s\n' "$sequence_dir" "$camera_dir"
}

entry_log_stem() {
  local rel_dir="$1"
  rel_dir="${rel_dir//\//__}"
  rel_dir="${rel_dir// /_}"
  printf '%s\n' "$rel_dir"
}

# -----------------------------------------------------------------------------
# Summary.
# -----------------------------------------------------------------------------
echo "DHP19 EventPointPose hot-pixel mask generation"
echo "Binary       : $BINARY"
if [[ -n "$DATA_FILE" ]]; then
  echo "Input mode   : single entry"
  echo "Data file    : $DATA_FILE"
else
  echo "Input mode   : dataset"
  echo "Data root    : $DATA_ROOT"
  echo "Data glob    : $DATA_GLOB"
fi
echo "Mask root    : $MASK_ROOT"
echo "Log dir      : $LOG_DIR"
echo "Resolution   : ${IMG_W}x${IMG_H}"
echo "Threshold    : $THRESHOLD"
echo "Write CSV    : $WRITE_CSV"
echo "Force        : $FORCE"
echo "Dry run      : $DRY_RUN"
echo "Entries found: ${#LOG_FILES[@]}"

# -----------------------------------------------------------------------------
# Generate one mask per entry.
# -----------------------------------------------------------------------------
processed=0
skipped=0
failed=0

for LOG_FILE in "${LOG_FILES[@]}"; do
  REL_DIR="$(entry_relative_dir "$LOG_FILE")"
  OUT_DIR="$MASK_ROOT/$REL_DIR"
  OUT_MASK="$OUT_DIR/hotpixel_mask.png"
  OUT_CSV="$OUT_DIR/hotpixel_pixels.csv"
  OUT_NPY="$OUT_DIR/hotpixel_mask.npy"
  RUN_LOG="$LOG_DIR/$(entry_log_stem "$REL_DIR").log"

  mkdir -p "$OUT_DIR"

  if [[ -f "$OUT_MASK" && "$FORCE" != "true" ]]; then
    echo "[SKIP] $REL_DIR -> mask already exists"
    ((skipped += 1))
    continue
  fi

  ARGS=(
    --data_file "$LOG_FILE"
    --output_mask "$OUT_MASK"
    --threshold "$THRESHOLD"
    --w "$IMG_W"
    --h "$IMG_H"
  )

  if [[ "$WRITE_CSV" == "true" ]]; then
    ARGS+=(--output_csv "$OUT_CSV")
  fi

  echo "[RUN ] $REL_DIR"
  echo "       input : $LOG_FILE"
  echo "       mask  : $OUT_MASK"
  if [[ "$WRITE_CSV" == "true" ]]; then
    echo "       csv   : $OUT_CSV"
  fi
  echo "       log   : $RUN_LOG"

  if [[ "$DRY_RUN" == "true" ]]; then
    printf '       command:'
    printf ' %q' "$BINARY" "${ARGS[@]}"
    printf '\n'
    ((processed += 1))
    continue
  fi

  if "$BINARY" "${ARGS[@]}" >"$RUN_LOG" 2>&1; then

    python3 - "$OUT_MASK" "$OUT_NPY" "$IMG_W" "$IMG_H" \
      >>"$RUN_LOG" 2>&1 <<'PY'

import sys
import cv2
import numpy as np

png_path = sys.argv[1]
npy_path = sys.argv[2]

expected_width = int(sys.argv[3])
expected_height = int(sys.argv[4])

mask = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise RuntimeError(f"Cannot read generated mask: {png_path}")

expected_shape = (expected_height, expected_width)

if mask.shape != expected_shape:
    raise RuntimeError(
        f"Invalid mask shape {mask.shape}; expected {expected_shape}"
    )

mask = mask > 0

np.save(npy_path, mask)

print(f"[INFO] NPY mask saved: {npy_path}")
print(f"[INFO] NPY shape: {mask.shape}")
print(f"[INFO] NPY hot pixels: {int(mask.sum())}")

PY

    ((processed += 1))

  else
    echo "[FAIL] $REL_DIR; see $RUN_LOG" >&2
    ((failed += 1))
  fi
done

echo ""
echo "Hot-pixel mask generation completed."
echo "Processed: $processed"
echo "Skipped  : $skipped"
echo "Failed   : $failed"
echo "Masks    : $MASK_ROOT"
echo "Logs     : $LOG_DIR"

if [[ $failed -ne 0 ]]; then
  exit 1
fi