#!/usr/bin/env bash
set -Eeuo pipefail

# ATIS move-iit-hpe baseline using the one shared EventPointPose C++/Python pipeline.
# No model or original DHP19 algorithm changes. Reuses the six ExpA periods.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/move-iit-it_testing}"
BINARY="${BINARY:-$REPO/build/eventPointPose_offline}"
SERVER_SCRIPT="${SERVER_SCRIPT:-$REPO/utils/eventPointPose_yarp_server.py}"
MODEL_PATH="${MODEL_PATH:-/workspace/model_mounts/eventpointpose/PointNet/models/model.pth}"
RESULTS="${RESULTS:-/data/MoveEnet_OFK_results/Accuracy/moveiit/EventPointPose}"
DEVICE="${DEVICE:-cuda:0}"
DATA_FILE="${DATA_FILE:-$DATA_ROOT/luna/box_f1/atis-s/data.log}"
NET_PERIOD="0.02"
OUTPUT_PERIOD="0.005"
GEOMETRY="stretch"
ALL=false
GRID=false
DEBUG=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: run_moveiit_epp.sh [options]
  --data_file PATH      One ATIS data.log, default luna/box_f1/atis-s/data.log
  --data_root PATH      Dataset root (default /data/move-iit-it_testing)
  --output_dir PATH     Output root, outside repository
  --model_path PATH     Original PointNet MeanLabel .pth (unchanged)
  --binary PATH         eventPointPose_offline executable
  --server_script PATH  Unified utils/eventPointPose_yarp_server.py
  --device DEVICE       cpu or cuda:0, default cuda:0
  --net_period SEC      Default 0.02 (nominal 50 Hz)
  --output_period SEC   Default 0.005 (200 Hz held-pose CSV)
  --geometry MODE       stretch or letterbox; default stretch
  --all                 All recordings with GT projections, same selection as run_moveiit_expa.sh
  --grid                Network periods: 0.01 0.02 0.05 0.1 0.2 0.5
  --debug               Save first two PointNet input/output diagnostic dumps
  --dry-run             Show commands without executing or writing results
  --help
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_file) DATA_FILE="$2"; shift 2;;
    --data_root) DATA_ROOT="$2"; shift 2;;
    --output_dir) RESULTS="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    --binary) BINARY="$2"; shift 2;;
    --server_script) SERVER_SCRIPT="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --net_period) NET_PERIOD="$2"; shift 2;;
    --output_period) OUTPUT_PERIOD="$2"; shift 2;;
    --geometry) GEOMETRY="$2"; shift 2;;
    --all) ALL=true; shift;;
    --grid) GRID=true; shift;;
    --debug) DEBUG=true; shift;;
    --dry-run) DRY_RUN=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Invalid option: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$GEOMETRY" == stretch || "$GEOMETRY" == letterbox ]] || { echo "Bad --geometry" >&2; exit 2; }
[[ -f "$SERVER_SCRIPT" ]] || { echo "Missing server: $SERVER_SCRIPT" >&2; exit 1; }
if [[ "$DRY_RUN" == false ]]; then
  [[ -x "$BINARY" ]] || { echo "Missing executable: $BINARY" >&2; exit 1; }
  [[ -f "$MODEL_PATH" ]] || { echo "Missing checkpoint: $MODEL_PATH" >&2; exit 1; }
fi
periods=("$NET_PERIOD")
if [[ "$GRID" == true ]]; then periods=(0.01 0.02 0.05 0.1 0.2 0.5); fi
recordings=()
if [[ "$ALL" == true ]]; then
  for subject in luna simon; do
    [[ -d "$DATA_ROOT/$subject" ]] || { echo "Missing subject directory: $DATA_ROOT/$subject" >&2; exit 1; }
    while IFS= read -r -d '' file; do
      seq="$(basename "$(dirname "$(dirname "$file")")")"
      gt_subject=P1_luna; [[ "$subject" == simon ]] && gt_subject=P11_simon
      if [[ -f "$DATA_ROOT/GT/$gt_subject/${seq}_joint_projections.csv" ]]; then
        recordings+=("$file")
      fi
    done < <(find "$DATA_ROOT/$subject" -type f -path '*/atis-s/data.log' -print0 | sort -z)
  done
  [[ "${#recordings[@]}" -gt 0 ]] || { echo 'No ATIS/GT paired recordings found' >&2; exit 1; }
else
  recordings=("$DATA_FILE")
fi
printf 'Selected: %s recordings x %s periods | %s\n' "${#recordings[@]}" "${#periods[@]}" "$GEOMETRY"
if [[ "$DRY_RUN" == false ]]; then
  mkdir -p "$RESULTS/$GEOMETRY/raw" "$RESULTS/$GEOMETRY/logs" "$RESULTS/$GEOMETRY/latency"
  manifest="$RESULTS/$GEOMETRY/manifest.csv"
  printf 'subject,sequence,geometry,net_period,output_period,event_file,pred_csv,run_log\n' > "$manifest"
fi
for file in "${recordings[@]}"; do
  [[ -f "$file" ]] || { echo "Missing recording: $file" >&2; exit 1; }
  seq="$(basename "$(dirname "$(dirname "$file")")")"
  subject="$(basename "$(dirname "$(dirname "$(dirname "$file")")")")"
  for np in "${periods[@]}"; do
    stem="${subject}__${seq}__atis-s__epp_${GEOMETRY}_np_${np//./p}"
    pred="$RESULTS/$GEOMETRY/raw/$stem.csv"
    log="$RESULTS/$GEOMETRY/logs/$stem.log"
    serverlog="$RESULTS/$GEOMETRY/logs/$stem.server.log"
    latency="$RESULTS/$GEOMETRY/latency/$stem.csv"
    cmd=("$BINARY" --data_file "$file" --moveiit --moveiit_geometry "$GEOMETRY" \
      --camera 0 --w 346 --h 260 --net_period "$np" --output_period "$OUTPUT_PERIOD" \
      --input_preprocessing already_filtered --fifo_size 7500 --num_points 2048 --seed 1 \
      --model_path "$MODEL_PATH" --EventPointPose_script "$SERVER_SCRIPT" --device "$DEVICE" \
      --output_csv "$pred" --latency_csv "$latency" --server_log "$serverlog")
    if [[ "$DEBUG" == true ]]; then
      dump="$RESULTS/$GEOMETRY/debug/$stem"
      cmd+=(--dump_dir "$dump" --dump_first_n 2)
    fi
    if [[ "$DRY_RUN" == true ]]; then
      printf 'DRY RUN: '; printf '%q ' "${cmd[@]}"; printf '\n'
      continue
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$subject" "$seq" "$GEOMETRY" "$np" "$OUTPUT_PERIOD" "$file" "$pred" "$log" >> "$manifest"
    if [[ -s "$pred" && -f "$log" ]] && grep -q 'Valid PointNet inferences:' "$log"; then
      if [[ "$DEBUG" == false ]] || [[ -n "$(find "$dump" -maxdepth 1 -name '*_accepted_batch_epp.npy' -print -quit 2>/dev/null)" ]]; then
        echo "SKIP $stem"
        continue
      fi
    fi
    echo "RUN $stem"
    if [[ "$DEBUG" == true ]]; then mkdir -p "$dump"; fi
    if ! "${cmd[@]}" > "$log" 2>&1; then
      tail -n 50 "$log" >&2
      [[ -f "$serverlog" ]] && tail -n 30 "$serverlog" >&2
      echo "FAILED $stem" >&2
      exit 1
    fi
    [[ -s "$pred" && $(wc -l < "$pred") -gt 1 ]] || {
      echo "No valid pose rows: $pred; see $log and $serverlog" >&2
      exit 1
    }
  done
done
echo "Results: $RESULTS/$GEOMETRY"
