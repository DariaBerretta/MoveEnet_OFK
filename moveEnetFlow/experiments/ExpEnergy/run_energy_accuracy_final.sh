#!/usr/bin/env bash
set -Eeuo pipefail

# Final Energy-Accuracy benchmark protocol.
#
# Fixed operating points agreed for the paper:
#   eH36M/H36M:
#     MoveNet-only   net=0.020 s (50 Hz),  output=0.005 s (200 Hz)
#     MoveEnetOFK    net=0.200 s (5 Hz),   flow=0.005 s (200 Hz), output=0.005 s
#     OpenPose       net=0.020 s (50 Hz),  output=0.005 s (200 Hz, ZOH)
#     YOLOPose       net=0.020 s (50 Hz),  output=0.005 s (200 Hz, ZOH)
#   DHP19:
#     MoveNet-only   net=0.005 s (200 Hz), output=0.005 s (200 Hz)
#     MoveEnetOFK    net=0.200 s (5 Hz),   flow=0.005 s (200 Hz), output=0.005 s
#     EventPointPose net=0.005 s (200 Hz), output=0.005 s (200 Hz)
#
# Each sample is repeated REPEATS times (default 5). Model order is rotated
# across repetitions to avoid systematically coupling a method to thermal/order
# effects. The script uses the same 12+12 samples as ExpLatency.
#
# Run on the HOST, with the moveEnetOFK container already running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." 2>/dev/null && pwd || true)"

# When installed in moveEnetFlow/experiments/ExpEnergy, these defaults resolve
# to the existing experiment files. They can all be overridden from the shell.
EXP_DIR="${EXP_DIR:-$SCRIPT_DIR}"
LATENCY_DIR="${LATENCY_DIR:-$EXP_DIR/../ExpLatency}"
H36M_MANIFEST="${H36M_MANIFEST:-$LATENCY_DIR/manifests/h36m_samples.sh}"
DHP19_MANIFEST="${DHP19_MANIFEST:-$LATENCY_DIR/manifests/dhp19_samples.sh}"
MEASURE_SCRIPT="${MEASURE_SCRIPT:-$EXP_DIR/measure_energy.py}"
ENERGY_PYTHON="${ENERGY_PYTHON:-python3}"

CONTAINER="${CONTAINER:-moveEnetOFK}"
VENV_DIR="${VENV_DIR:-/opt/venv}"
CONTAINER_BASE_PATH="${CONTAINER_BASE_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
BUILD_DIR="${BUILD_DIR:-/workspace/moveEnetFlow/build}"
MOVENET_BIN="${MOVENET_BIN:-$BUILD_DIR/moveEnetOFK_offline}"
YOLO_BIN="${YOLO_BIN:-$BUILD_DIR/YoloPose_offline}"
OPENPOSE_BIN="${OPENPOSE_BIN:-$BUILD_DIR/OpenPose_offline}"
EPP_BIN="${EPP_BIN:-$BUILD_DIR/eventPointPose_offline}"

H36M_MOVENET_CKPT="${H36M_MOVENET_CKPT:-/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth}"
DHP19_MOVENET_CKPT="${DHP19_MOVENET_CKPT:-/usr/local/src/hpe-core/example/movenet/models/dhp19_allcams_e33_valacc0.87996.pth}"
YOLO_MODEL="${YOLO_MODEL:-/workspace/model_mounts/YoloPose/yolo26n-pose.pt}"
YOLO_SCRIPT="${YOLO_SCRIPT:-/workspace/model_mounts/YoloPose/YoloPose_yarp_server.py}"
OPENPOSE_MODEL_DIR="${OPENPOSE_MODEL_DIR:-/usr/local/src/openpose/models/}"
EPP_MODEL="${EPP_MODEL:-/workspace/model_mounts/eventpointpose/PointNet/models/model.pth}"
EPP_SCRIPT="${EPP_SCRIPT:-/workspace/model_mounts/eventpointpose/PointNet/models/eventPointPose_yarp_server.py}"

DEVICE="${DEVICE:-cuda:0}"
GPU_INDEX="${GPU_INDEX:-0}"
OUTPUT_PERIOD="0.005"
FLOW_PERIOD="0.005"
H36M_NET_PERIOD="0.02"
H36M_MOVENET_NET_PERIOD="0.02"
H36M_OFK_NET_PERIOD="0.05"
DHP19_MOVENET_NET_PERIOD="0.005"
DHP19_OFK_NET_PERIOD="0.2"
DHP19_EPP_NET_PERIOD="0.005"

REPEATS="${REPEATS:-5}"
COOLDOWN_S="${COOLDOWN_S:-20}"
DATASET="both"
ONLY_SAMPLE=""
SESSION_ID="${SESSION_ID:-energy_accuracy_final_$(date +%Y%m%d_%H%M%S)}"
RESULTS_BASE="${RESULTS_BASE:-$HOME/data/MoveEnet_OFK_results/Energy}"
RESUME="false"
DRY_RUN="false"
PREFLIGHT_ONLY="false"
CONTINUE_ON_ERROR="false"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --dataset <both|h36m|dhp19>  Dataset selection (default: both)
  --sample <H01|D01|...>       Run one selected sample only
  --repeats <N>                Repetitions per sample (default: 5)
  --cooldown <seconds>         Cooldown after each measured run (default: 20)
  --session <name>             Output session name
  --results_base <path>        Host result root
  --device <device>            Inference device (default: cuda:0)
  --gpu_index <N>              GPU energy counter index (default: 0)
  --container <name>           Docker container (default: moveEnetOFK)
  --resume                     Skip runs already marked OK
  --continue_on_error          Continue after a failed run
  --preflight-only             Validate configuration and exit
  --dry-run                    Print commands without executing them
  --help

Examples:
  ./run_energy_accuracy_final.sh --preflight-only
  ./run_energy_accuracy_final.sh --dataset h36m --sample H01 --repeats 2 --dry-run
  ./run_energy_accuracy_final.sh --dataset both --repeats 5 --session energy_accuracy_main
USAGE
}

is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_non_negative_number() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --sample) ONLY_SAMPLE="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --cooldown) COOLDOWN_S="$2"; shift 2 ;;
        --session) SESSION_ID="$2"; shift 2 ;;
        --results_base) RESULTS_BASE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --gpu_index) GPU_INDEX="$2"; shift 2 ;;
        --container) CONTAINER="$2"; shift 2 ;;
        --resume) RESUME="true"; shift ;;
        --continue_on_error) CONTINUE_ON_ERROR="true"; shift ;;
        --preflight-only) PREFLIGHT_ONLY="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$DATASET" in both|h36m|dhp19) ;; *) echo "Invalid dataset: $DATASET" >&2; exit 2 ;; esac
is_positive_integer "$REPEATS" || { echo "--repeats must be a positive integer" >&2; exit 2; }
is_non_negative_number "$COOLDOWN_S" || { echo "Invalid cooldown: $COOLDOWN_S" >&2; exit 2; }

[[ -f "$MEASURE_SCRIPT" ]] || { echo "Missing measure script: $MEASURE_SCRIPT" >&2; exit 1; }
[[ -f "$H36M_MANIFEST" ]] || { echo "Missing H36M manifest: $H36M_MANIFEST" >&2; exit 1; }
[[ -f "$DHP19_MANIFEST" ]] || { echo "Missing DHP19 manifest: $DHP19_MANIFEST" >&2; exit 1; }

# shellcheck disable=SC1090
source "$H36M_MANIFEST"
# shellcheck disable=SC1090
source "$DHP19_MANIFEST"

SESSION_DIR="$RESULTS_BASE/$SESSION_ID"
RUNS_DIR="$SESSION_DIR/runs"
ENERGY_CSV="$SESSION_DIR/energy_measurements.csv"
MANIFEST_CSV="$SESSION_DIR/run_manifest.csv"
mkdir -p "$RUNS_DIR"

if [[ ! -f "$MANIFEST_CSV" ]]; then
    echo "dataset,sample_id,subject,sequence,camera,motion_class,model,repetition,order_position,net_period,net_hz,flow_period,output_period,status,energy_csv,log_path" > "$MANIFEST_CSV"
fi

METADATA="$SESSION_DIR/metadata.txt"
{
    echo "session_id=$SESSION_ID"
    echo "created=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "dataset=$DATASET"
    echo "only_sample=$ONLY_SAMPLE"
    echo "repeats=$REPEATS"
    echo "cooldown_s=$COOLDOWN_S"
    echo "output_period_s=$OUTPUT_PERIOD"
    echo "flow_period_s=$FLOW_PERIOD"
    echo "h36m_net_period_s=$H36M_NET_PERIOD"
    echo "dhp19_movenet_net_period_s=$DHP19_MOVENET_NET_PERIOD"
    echo "dhp19_ofk_net_period_s=$DHP19_OFK_NET_PERIOD"
    echo "dhp19_epp_net_period_s=$DHP19_EPP_NET_PERIOD"
    echo "container=$CONTAINER"
    echo "venv_dir=$VENV_DIR"
    echo "device=$DEVICE"
    echo "gpu_index=$GPU_INDEX"
    echo
    echo "[git]"
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true
    git -C "$REPO_ROOT" status --short 2>/dev/null || true
    echo
    echo "[gpu]"
    nvidia-smi 2>/dev/null || true
} > "$METADATA"

models_h36m=(movenet moveenetofk openpose yolo)
models_dhp19=(movenet moveenetofk eventpointpose)

# Rotate a base order by repetition index. Because every sample uses all
# repetitions, each method appears across different positions rather than being
# permanently tied to one position/temperature regime.
rotate_models() {
    local sample_id="$1"
    local repetition="$2"
    shift 2
    local models=("$@")
    local n="${#models[@]}"
    local sample_number=$((10#${sample_id:1}))
    # Sample-dependent offset distributes the extra position caused by 5 repeats
    # across methods instead of always favouring the same method.
    local offset=$(((sample_number + repetition - 2) % n))
    local i idx
    for ((i=0; i<n; i++)); do
        idx=$(((offset + i) % n))
        printf '%s%s' "${models[$idx]}" "$([[ $i -lt $((n-1)) ]] && echo ' ')"
    done
}

check_sidecars() {
    local p
    p="$(docker exec "$CONTAINER" bash -lc "pgrep -af '[m]ovenet_online.py|[Y]oloPose_yarp_server.py|[e]ventPointPose_yarp_server.py' || true")"
    if [[ -n "$p" ]]; then
        echo "ERROR: HPE sidecar already active inside $CONTAINER:" >&2
        echo "$p" >&2
        return 1
    fi
}

warn_gpu_processes() {
    local apps
    apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "$apps" ]]; then
        echo "WARNING: GPU compute processes are already active:" >&2
        echo "$apps" >&2
        echo "For the final benchmark, keep the target GPU otherwise idle." >&2
    fi
}

preflight() {
    command -v docker >/dev/null || { echo "docker not found" >&2; return 1; }
    "$ENERGY_PYTHON" -c 'import pynvml' >/dev/null 2>&1 || { echo "Python pynvml unavailable" >&2; return 1; }
    [[ -r /sys/class/powercap/intel-rapl:0/energy_uj ]] || { echo "RAPL counter unreadable" >&2; return 1; }
    docker exec "$CONTAINER" test -d "$VENV_DIR"
        docker exec "$CONTAINER" \
        "$VENV_DIR/bin/python" \
        -c 'import torch, yarp; print("MoveNet Python OK")'

    docker exec "$CONTAINER" \
        "$VENV_DIR/bin/python" \
        -c 'import ultralytics; print("YOLO Python OK")'
    docker exec "$CONTAINER" test -x "$VENV_DIR/bin/python3"
    docker exec "$CONTAINER" test -x "$MOVENET_BIN"
    docker exec "$CONTAINER" test -x "$YOLO_BIN"
    docker exec "$CONTAINER" test -x "$OPENPOSE_BIN"
    docker exec "$CONTAINER" test -x "$EPP_BIN"
    docker exec "$CONTAINER" test -f "$H36M_MOVENET_CKPT"
    docker exec "$CONTAINER" test -f "$DHP19_MOVENET_CKPT"
    docker exec "$CONTAINER" test -f "$YOLO_MODEL"
    docker exec "$CONTAINER" test -f "$YOLO_SCRIPT"
    docker exec "$CONTAINER" test -d "$OPENPOSE_MODEL_DIR"
    docker exec "$CONTAINER" test -f "$EPP_MODEL"
    docker exec "$CONTAINER" test -f "$EPP_SCRIPT"
    docker exec "$CONTAINER" bash -lc 'yarp where >/dev/null 2>&1' || { echo "YARP server is not reachable inside $CONTAINER" >&2; return 1; }
    check_sidecars
    warn_gpu_processes
    echo "Preflight OK"
    echo "Final protocol: output=200 Hz for every method; eH36M net=50 Hz; DHP19 MoveNet/EPP=200 Hz; DHP19 OFK net=5 Hz + flow=200 Hz"
    echo "Repetitions per sample: $REPEATS"
}

if [[ "$PREFLIGHT_ONLY" == "true" ]]; then preflight; exit 0; fi
preflight

if [[ "$DRY_RUN" != "true" && "$COOLDOWN_S" != "0" ]]; then
    echo "Initial cooldown: ${COOLDOWN_S}s"
    sleep "$COOLDOWN_S"
fi

already_done() {
    local dataset="$1" sample="$2" model="$3" rep="$4"
    [[ "$RESUME" == "true" ]] || return 1
    grep -q "^${dataset},${sample},[^,]*,[^,]*,[^,]*,[^,]*,${model},${rep},.*OK," "$MANIFEST_CSV" 2>/dev/null
}

run_measured() {
    local dataset="$1" sample="$2" subject="$3" sequence="$4" camera="$5" motion_class="$6"
    local event_path="$7" rgb_path="$8" model="$9" rep="${10}" order_pos="${11}"

    local net_period flow_period="" net_hz command_label
    local -a cmd

    if [[ "$dataset" == "h36m" ]]; then
        net_period="$H36M_NET_PERIOD"
        case "$model" in
            movenet)
                net_period="$H36M_NET_PERIOD"
                cmd=(docker exec "$CONTAINER" env VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$CONTAINER_BASE_PATH" "$MOVENET_BIN" --data_file "$event_path" --net_period "$net_period" --flow_period "$FLOW_PERIOD" --output_period "$OUTPUT_PERIOD" --w 640 --h 480 --checkpoint_path "$H36M_MOVENET_CKPT" --device "$DEVICE" --moveenet_only --no_csv --no_video)
                ;;
            moveenetofk)
                net_period="$H36M_OFK_NET_PERIOD"
                flow_period="$FLOW_PERIOD"
                cmd=(docker exec "$CONTAINER" env VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$CONTAINER_BASE_PATH" "$MOVENET_BIN" --data_file "$event_path" --net_period "$net_period" --flow_period "$FLOW_PERIOD" --output_period "$OUTPUT_PERIOD" --w 640 --h 480 --checkpoint_path "$H36M_MOVENET_CKPT" --device "$DEVICE" --no_csv --no_video)
                ;;
            openpose)
                cmd=(docker exec "$CONTAINER" env -u VIRTUAL_ENV PATH="$CONTAINER_BASE_PATH" "$OPENPOSE_BIN" --data_file "$rgb_path" --net_period "$net_period" --output_period "$OUTPUT_PERIOD" --w 640 --h 480 --op_model_path "$OPENPOSE_MODEL_DIR" --device "$DEVICE" --no_csv --no_video)
                ;;
            yolo)
                cmd=(docker exec "$CONTAINER" env VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$CONTAINER_BASE_PATH" "$YOLO_BIN" --data_file "$rgb_path" --net_period "$net_period" --output_period "$OUTPUT_PERIOD" --w 640 --h 480 --yolo_model_path "$YOLO_MODEL" --YoloPose_script "$YOLO_SCRIPT" --device "$DEVICE" --no_csv --no_video)
                ;;
            *) echo "Unknown H36M model: $model" >&2; return 2 ;;
        esac
    else
        case "$model" in
            movenet)
                net_period="$DHP19_MOVENET_NET_PERIOD"
                cmd=(docker exec "$CONTAINER" env VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$CONTAINER_BASE_PATH" "$MOVENET_BIN" --data_file "$event_path" --net_period "$net_period" --flow_period "$FLOW_PERIOD" --output_period "$OUTPUT_PERIOD" --dhp19 --checkpoint_path "$DHP19_MOVENET_CKPT" --device "$DEVICE" --moveenet_only --no_csv --no_video)
                ;;
            moveenetofk)
                net_period="$DHP19_OFK_NET_PERIOD"; flow_period="$FLOW_PERIOD"
                cmd=(docker exec "$CONTAINER" env VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$CONTAINER_BASE_PATH" "$MOVENET_BIN" --data_file "$event_path" --net_period "$net_period" --flow_period "$FLOW_PERIOD" --output_period "$OUTPUT_PERIOD" --dhp19 --checkpoint_path "$DHP19_MOVENET_CKPT" --device "$DEVICE" --no_csv --no_video)
                ;;
            eventpointpose)
                net_period="$DHP19_EPP_NET_PERIOD"
                local camera_id
                case "$camera" in
                    ch2dvs) camera_id="2" ;;
                    ch3dvs) camera_id="3" ;;
                    *) echo "Unsupported EPP camera: $camera" >&2; return 2 ;;
                esac
                cmd=(docker exec "$CONTAINER" env VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$CONTAINER_BASE_PATH" "$EPP_BIN" --data_file "$event_path" --camera "$camera_id" --net_period "$net_period" --output_period "$OUTPUT_PERIOD" --w 346 --h 260 --model_path "$EPP_MODEL" --EventPointPose_script "$EPP_SCRIPT" --device "$DEVICE" --startup_timeout 30 --response_timeout 0 --no_csv)
                ;;
            *) echo "Unknown DHP19 model: $model" >&2; return 2 ;;
        esac
    fi

    net_hz="$(awk -v p="$net_period" 'BEGIN{printf "%.6f",1/p}')"
    local tag="${dataset}_${sample}_${model}_r$(printf '%02d' "$rep")"
    local log="$RUNS_DIR/${tag}.log"
    command_label="${sample}_${model}_r${rep}"

    if already_done "$dataset" "$sample" "$model" "$rep"; then
        echo "SKIP $tag (already OK)"
        return 0
    fi

    echo "[$dataset][$sample][rep $rep][$order_pos] $model: net=${net_hz}Hz output=200Hz${flow_period:+ flow=200Hz}"
    if [[ "$DRY_RUN" == "true" ]]; then
        printf '  '; printf '%q ' "${cmd[@]}"; echo
        return 0
    fi

    check_sidecars
    local status="OK"
    set +e
    "$ENERGY_PYTHON" "$MEASURE_SCRIPT" \
        --csv "$ENERGY_CSV" --log "$log" --label "$command_label" \
        --dataset "$dataset" --sample-id "$sample" --model "$model" \
        --order-id "R${rep}" --order-position "$order_pos" \
        --net-period "$net_period" --flow-period "$flow_period" --gpu-index "$GPU_INDEX" \
        -- "${cmd[@]}"
    local rc=$?
    set -e
    [[ $rc -eq 0 ]] || status="FAILED"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$dataset" "$sample" "$subject" "$sequence" "$camera" "$motion_class" "$model" "$rep" "$order_pos" \
        "$net_period" "$net_hz" "$flow_period" "$OUTPUT_PERIOD" "$status" "$ENERGY_CSV" "$log" >> "$MANIFEST_CSV"

    if [[ $rc -ne 0 && "$CONTINUE_ON_ERROR" != "true" ]]; then return "$rc"; fi
    if [[ "$COOLDOWN_S" != "0" ]]; then sleep "$COOLDOWN_S"; fi
    return 0
}

run_h36m() {
    local entry sample subject sequence camera order event_path rgb_path rep order_pos model
    for entry in "${H36M_SAMPLES[@]}"; do
        IFS='|' read -r sample subject sequence camera order event_path rgb_path <<< "$entry"
        [[ -z "$ONLY_SAMPLE" || "$sample" == "$ONLY_SAMPLE" ]] || continue
        docker exec "$CONTAINER" test -f "$event_path" || { echo "Missing $event_path" >&2; exit 1; }
        docker exec "$CONTAINER" test -f "$rgb_path" || { echo "Missing $rgb_path" >&2; exit 1; }
        for ((rep=1; rep<=REPEATS; rep++)); do
            read -r -a ordered <<< "$(rotate_models "$sample" "$rep" "${models_h36m[@]}")"
            order_pos=0
            for model in "${ordered[@]}"; do
                ((order_pos+=1))
                run_measured h36m "$sample" "$subject" "$sequence" "$camera" "" "$event_path" "$rgb_path" "$model" "$rep" "$order_pos"
            done
        done
    done
}

run_dhp19() {
    local entry sample subject sequence camera motion_class motion_name order event_path rep order_pos model
    for entry in "${DHP19_SAMPLES[@]}"; do
        IFS='|' read -r sample subject sequence camera motion_class motion_name order event_path <<< "$entry"
        [[ -z "$ONLY_SAMPLE" || "$sample" == "$ONLY_SAMPLE" ]] || continue
        docker exec "$CONTAINER" test -f "$event_path" || { echo "Missing $event_path" >&2; exit 1; }
        for ((rep=1; rep<=REPEATS; rep++)); do
            read -r -a ordered <<< "$(rotate_models "$sample" "$rep" "${models_dhp19[@]}")"
            order_pos=0
            for model in "${ordered[@]}"; do
                ((order_pos+=1))
                run_measured dhp19 "$sample" "$subject" "$sequence" "$camera" "$motion_class" "$event_path" "" "$model" "$rep" "$order_pos"
            done
        done
    done
}

case "$DATASET" in
    both) run_h36m; run_dhp19 ;;
    h36m) run_h36m ;;
    dhp19) run_dhp19 ;;
esac

echo "Experiment complete."
echo "Energy CSV : $ENERGY_CSV"
echo "Manifest   : $MANIFEST_CSV"
echo "Logs       : $RUNS_DIR"

