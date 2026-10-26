#!/usr/bin/env bash
set -Eeuo pipefail

# ExpA: MoveEnet-only vs MoveEnetOFK on paired move-iit-hpe recordings.
# Both methods use the same network and output periods; OFK additionally runs
# optical flow at 200 Hz. Requires the gain=1, native-1280x720 C++ binary.

DATA_ROOT="${DATA_ROOT:-/data/move-iit-it_testing}"
BINARY="${BINARY:-/workspace/moveEnetFlow/build/moveEnetOFK_offline}"
CHECKPOINT="${CHECKPOINT:-/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth}"
RESULTS="${RESULTS:-/data/MoveEnet_OFK_results/Final_ROI_Gain1/Accuracy/moveiit/ExpA}"
DEVICE="${DEVICE:-cuda:0}"
NETWORK_PERIODS=(0.01 0.02 0.05 0.1 0.2 0.5)
FLOW_PERIOD=0.005
OUTPUT_PERIOD=0.005
ROI=35

[[ -x "$BINARY" ]] || { echo "Binary missing: $BINARY" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint missing: $CHECKPOINT" >&2; exit 1; }
[[ -d "$DATA_ROOT/GT" ]] || { echo "GT directory missing: $DATA_ROOT/GT" >&2; exit 1; }

# Only recordings for which 2D Vicon ground truth is available.
declare -a EVENTS=() SUBJECTS=() SEQUENCES=() GTS=()
while IFS= read -r -d '' event_file; do
    seq="$(basename "$(dirname "$(dirname "$event_file")")")"
    subject="$(basename "$(dirname "$(dirname "$(dirname "$event_file")")")")"
    case "$subject" in
        luna) gt_subject="P1_luna" ;;
        simon) gt_subject="P11_simon" ;;
        *) continue ;;
    esac
    gt="$DATA_ROOT/GT/$gt_subject/${seq}_joint_projections.csv"
    if [[ -f "$gt" ]]; then
        EVENTS+=("$event_file")
        SUBJECTS+=("$subject")
        SEQUENCES+=("$seq")
        GTS+=("$gt")
    fi
done < <(find "$DATA_ROOT/luna" "$DATA_ROOT/simon" -type f -path '*/atis-s/data.log' -print0 | sort -z)

if [[ ${#EVENTS[@]} -ne 59 ]]; then
    echo "Expected 59 paired recordings; found ${#EVENTS[@]}. Check the dataset before running." >&2
    exit 1
fi

mkdir -p "$RESULTS/raw" "$RESULTS/logs"

# Prevent accidental mixing of outputs made with different binaries/parameters.
config="$(printf 'binary_sha256=%s\ncheckpoint_sha256=%s\ndata_root=%s\ndevice=%s\nnetwork_periods=%s\nflow_period=%s\noutput_period=%s\nroi=%s\npu=0.77\nmuD=0.06\nmuV=0.97\nmoveiit=true\n' \
    "$(sha256sum "$BINARY" | cut -d' ' -f1)" \
    "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" \
    "$DATA_ROOT" "$DEVICE" "${NETWORK_PERIODS[*]}" "$FLOW_PERIOD" "$OUTPUT_PERIOD" "$ROI")"
if [[ -f "$RESULTS/config.txt" ]]; then
    if [[ "$(cat "$RESULTS/config.txt")" != "$config" ]]; then
        echo "Configuration mismatch in $RESULTS/config.txt. Use a different RESULTS directory." >&2
        exit 1
    fi
else
    printf '%s\n' "$config" > "$RESULTS/config.txt"
fi

printf 'subject,sequence,events,ground_truth\n' > "$RESULTS/manifest.csv"
for ((i=0; i<${#EVENTS[@]}; i++)); do
    printf '%s,%s,%s,%s\n' "${SUBJECTS[$i]}" "${SEQUENCES[$i]}" "${EVENTS[$i]}" "${GTS[$i]}" \
        >> "$RESULTS/manifest.csv"
done

echo "Dataset: move-iit-hpe (native 1280x720)"
echo "Paired sequences: ${#EVENTS[@]}"
echo "Network periods: ${NETWORK_PERIODS[*]}"
echo "OF flow: $FLOW_PERIOD s | output: $OUTPUT_PERIOD s | ROI: $ROI | gain: 1"
echo "Result directory: $RESULTS"
echo "Runs: $((${#EVENTS[@]} * ${#NETWORK_PERIODS[@]} * 2)); completed runs will be skipped on restart."

run_one() {
    local method="$1" label="$2" file="$3" np="$4" out="$5" log="$6"
    if [[ -s "$out" && -f "$log" ]] && grep -q 'Finished processing event file.' "$log"; then
        echo "  SKIP $label"
        return 0
    fi
    echo "  RUN  $label"
    local -a args=(
        --data_file "$file"
        --moveiit
        --net_period "$np"
        --flow_period "$FLOW_PERIOD"
        --output_period "$OUTPUT_PERIOD"
        --roi "$ROI"
        --pu 0.77 --muD 0.06 --muV 0.97
        --use_lc true
        --checkpoint_path "$CHECKPOINT"
        --device "$DEVICE"
        --output_csv_f "$out"
        --no_video
    )
    if [[ "$method" == "net" ]]; then args+=(--moveenet_only); fi
    if ! "$BINARY" "${args[@]}" > "$log" 2>&1; then
        echo "FAILED: $label (log: $log)" >&2
        tail -n 25 "$log" >&2
        exit 1
    fi
    if [[ ! -s "$out" || "$(wc -l < "$out")" -le 1 ]] || \
       ! grep -q 'Finished processing event file.' "$log"; then
        echo "Incomplete output: $out (log: $log)" >&2
        exit 1
    fi
}

for np in "${NETWORK_PERIODS[@]}"; do
    safe_np="${np//./p}"
    run_dir="$RESULTS/raw/fp_0p005/np_${safe_np}"
    mkdir -p "$run_dir"
    for ((i=0; i<${#EVENTS[@]}; i++)); do
        stem="${SUBJECTS[$i]}__${SEQUENCES[$i]}__atis-s"
        echo "[$((i+1))/${#EVENTS[@]}] network=$np s | ${SUBJECTS[$i]}/${SEQUENCES[$i]}"
        run_one ofk "$stem OFK" "${EVENTS[$i]}" "$np" \
            "$run_dir/${stem}_moveenet_ofk_np_${safe_np}_fp_0p005.csv" \
            "$RESULTS/logs/${stem}_moveenet_ofk_np_${safe_np}_fp_0p005.log"
        run_one net "$stem MoveEnet-only" "${EVENTS[$i]}" "$np" \
            "$run_dir/${stem}_moveenet_only_np_${safe_np}_fp_0p005.csv" \
            "$RESULTS/logs/${stem}_moveenet_only_np_${safe_np}_fp_0p005.log"
    done
done

echo "ExpA completed. CSV: $RESULTS/raw | manifest: $RESULTS/manifest.csv"
