#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MoveEnet_OFK latency benchmark runner
#
# H36M/eH36M:
#   MoveNet, MoveEnetOFK, OpenPose, YOLOPose
#
# DHP19:
#   MoveNet, MoveEnetOFK, EventPointPose
#
# Primary timing configuration:
#   net_period  = 0.02 s (50 Hz nominal dataset-time request rate)
#   flow_period = 0.005 s for MoveEnetOFK / MoveNet runner
#
# Environment policy:
#   - MoveNet, MoveEnetOFK, OpenPose run OUTSIDE /workspace/.nbvenv
#   - YOLOPose and EventPointPose run INSIDE /workspace/.nbvenv
#
# The script does not activate/deactivate the caller shell. It constructs the
# required PATH/VIRTUAL_ENV independently for each launched process.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H36M_MANIFEST="$SCRIPT_DIR/manifests/h36m_samples.sh"
DHP19_MANIFEST="$SCRIPT_DIR/manifests/dhp19_samples.sh"

VENV_DIR="${VENV_DIR:-/workspace/.nbvenv}"
BUILD_DIR="${BUILD_DIR:-/workspace/moveEnetFlow/build}"
REPO_ROOT="${REPO_ROOT:-/workspace/moveEnetFlow}"
RESULTS_BASE="${RESULTS_BASE:-/data/MoveEnet_OFK_results/Latency}"

MOVENET_BIN="$BUILD_DIR/moveEnetOFK_offline"
YOLO_BIN="$BUILD_DIR/YoloPose_offline"
OPENPOSE_BIN="$BUILD_DIR/OpenPose_offline"
EPP_BIN="$BUILD_DIR/eventPointPose_offline"

H36M_MOVENET_CKPT="${H36M_MOVENET_CKPT:-/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth}"
DHP19_MOVENET_CKPT="${DHP19_MOVENET_CKPT:-/usr/local/src/hpe-core/example/movenet/models/dhp19_allcams_e33_valacc0.87996.pth}"
YOLO_MODEL="${YOLO_MODEL:-/workspace/model_mounts/YoloPose/yolo26n-pose.pt}"
YOLO_SCRIPT="${YOLO_SCRIPT:-/workspace/model_mounts/YoloPose/YoloPose_yarp_server.py}"
OPENPOSE_MODEL_DIR="${OPENPOSE_MODEL_DIR:-/usr/local/src/openpose/models/}"
EPP_MODEL="${EPP_MODEL:-/workspace/model_mounts/eventpointpose/PointNet/models/model.pth}"
EPP_SCRIPT="${EPP_SCRIPT:-/workspace/model_mounts/eventpointpose/PointNet/models/eventPointPose_yarp_server.py}"

DEVICE="${DEVICE:-cuda:0}"
NET_PERIOD="0.02"
FLOW_PERIOD="0.005"
MOVENET_OUTPUT_PERIOD="0.005"
YOLO_OUTPUT_PERIOD="0.02"
OPENPOSE_OUTPUT_PERIOD="0.02"
EPP_OUTPUT_PERIOD="0.005"

COOLDOWN_S="20"
DATASET="both"
ONLY_SAMPLE=""
SESSION_ID="$(date +%Y%m%d_%H%M%S)"
RESUME="false"
DRY_RUN="false"
PREFLIGHT_ONLY="false"
CONTINUE_ON_ERROR="false"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --dataset <both|h36m|dhp19>  Dataset group to run. Default: both
  --sample <ID>                Run only one sample, e.g. H01 or D03
  --session <name>             Results session name. Default: timestamp
  --results_base <path>        Results root. Default: $RESULTS_BASE
  --cooldown <seconds>         Fixed delay between model runs. Default: 20
  --device <device>            CUDA/CPU device string. Default: cuda:0
  --resume                     Skip runs whose latency.csv already has data
  --continue_on_error          Continue after a failed model run
  --preflight-only             Validate files/binaries/manifests, then exit
  --dry-run                    Print commands without executing them
  --help                       Show this help

Examples:
  ./run_latency_experiment.sh --preflight-only
  ./run_latency_experiment.sh --dataset h36m --sample H01 --session smoke_H01
  ./run_latency_experiment.sh --dataset both --session latency_50hz_main
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --sample) ONLY_SAMPLE="$2"; shift 2 ;;
        --session) SESSION_ID="$2"; shift 2 ;;
        --results_base) RESULTS_BASE="$2"; shift 2 ;;
        --cooldown) COOLDOWN_S="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --resume) RESUME="true"; shift ;;
        --continue_on_error) CONTINUE_ON_ERROR="true"; shift ;;
        --preflight-only) PREFLIGHT_ONLY="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

case "$DATASET" in
    both|h36m|dhp19) ;;
    *) echo "Invalid --dataset: $DATASET" >&2; exit 2 ;;
esac

if ! [[ "$COOLDOWN_S" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--cooldown must be a non-negative number." >&2
    exit 2
fi

# ------------------------------------------------------------------------------
# Environment isolation
# ------------------------------------------------------------------------------
# If this runner itself is launched while .nbvenv is active, strip that path
# from the system environment. YOLO/EPP explicitly add it back.
SYSTEM_PATH="$PATH"
SYSTEM_PATH="${SYSTEM_PATH//${VENV_DIR}\/bin:/}"
SYSTEM_PATH="${SYSTEM_PATH//:${VENV_DIR}\/bin/}"
SYSTEM_PATH="${SYSTEM_PATH//${VENV_DIR}\/bin/}"

run_system() {
    env -u VIRTUAL_ENV PATH="$SYSTEM_PATH" "$@"
}

run_venv() {
    env VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$SYSTEM_PATH" "$@"
}

# ------------------------------------------------------------------------------
# Balanced model orders
# ------------------------------------------------------------------------------
models_for_order() {
    case "$1" in
        H0) echo "movenet moveenetofk openpose yolo" ;;
        H1) echo "moveenetofk yolo movenet openpose" ;;
        H2) echo "yolo openpose moveenetofk movenet" ;;
        H3) echo "openpose movenet yolo moveenetofk" ;;

        D0) echo "movenet moveenetofk eventpointpose" ;;
        D1) echo "movenet eventpointpose moveenetofk" ;;
        D2) echo "moveenetofk movenet eventpointpose" ;;
        D3) echo "moveenetofk eventpointpose movenet" ;;
        D4) echo "eventpointpose movenet moveenetofk" ;;
        D5) echo "eventpointpose moveenetofk movenet" ;;

        *) echo "Unknown order id: $1" >&2; return 1 ;;
    esac
}

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
require_file() {
    local p="$1"
    [[ -f "$p" ]] || { echo "Missing file: $p" >&2; return 1; }
}

require_dir() {
    local p="$1"
    [[ -d "$p" ]] || { echo "Missing directory: $p" >&2; return 1; }
}

require_executable() {
    local p="$1"
    [[ -x "$p" ]] || { echo "Missing/non-executable binary: $p" >&2; return 1; }
}

gpu_temp() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 || true
    fi
}

csv_rows() {
    local f="$1"
    if [[ -f "$f" ]]; then
        local n
        n="$(wc -l < "$f")"
        if [[ "$n" -gt 0 ]]; then
            echo $((n - 1))
        else
            echo 0
        fi
    else
        echo 0
    fi
}

quote_command() {
    local out=""
    printf -v out '%q ' "$@"
    printf '%s' "$out"
}

# ------------------------------------------------------------------------------
# Load manifests
# ------------------------------------------------------------------------------
require_file "$H36M_MANIFEST"
require_file "$DHP19_MANIFEST"
# shellcheck disable=SC1090
source "$H36M_MANIFEST"
# shellcheck disable=SC1090
source "$DHP19_MANIFEST"

# ------------------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------------------
preflight() {
    echo "=== PREFLIGHT ==="

    require_dir "$VENV_DIR"
    require_executable "$VENV_DIR/bin/python3"

    require_executable "$MOVENET_BIN"
    require_executable "$YOLO_BIN"
    require_executable "$OPENPOSE_BIN"
    require_executable "$EPP_BIN"

    require_file "$H36M_MOVENET_CKPT"
    require_file "$DHP19_MOVENET_CKPT"
    require_file "$YOLO_MODEL"
    require_file "$YOLO_SCRIPT"
    require_dir "$OPENPOSE_MODEL_DIR"
    require_file "$EPP_MODEL"
    require_file "$EPP_SCRIPT"

    local checked=0

    if [[ "$DATASET" == "both" || "$DATASET" == "h36m" ]]; then
        for row in "${H36M_SAMPLES[@]}"; do
            IFS='|' read -r id subject sequence camera order_id event_path rgb_path <<< "$row"
            if [[ -n "$ONLY_SAMPLE" && "$id" != "$ONLY_SAMPLE" ]]; then
                continue
            fi
            require_file "$event_path"
            require_file "$rgb_path"
            models_for_order "$order_id" >/dev/null
            checked=$((checked + 1))
        done
    fi

    if [[ "$DATASET" == "both" || "$DATASET" == "dhp19" ]]; then
        for row in "${DHP19_SAMPLES[@]}"; do
            IFS='|' read -r id subject sequence camera motion_class motion_name order_id event_path <<< "$row"
            if [[ -n "$ONLY_SAMPLE" && "$id" != "$ONLY_SAMPLE" ]]; then
                continue
            fi
            require_file "$event_path"
            models_for_order "$order_id" >/dev/null
            case "$camera" in ch2dvs|ch3dvs) ;; *) echo "Invalid DHP19 camera in $id: $camera" >&2; return 1 ;; esac
            checked=$((checked + 1))
        done
    fi

    if [[ -n "$ONLY_SAMPLE" && "$checked" -eq 0 ]]; then
        echo "Requested sample not found for selected dataset: $ONLY_SAMPLE" >&2
        return 1
    fi

    echo "Samples selected : $checked"
    local system_python
    system_python="$(PATH="$SYSTEM_PATH" command -v python3 || true)"
    echo "System python    : $system_python"
    echo "Venv python      : $VENV_DIR/bin/python3"
    echo "Device           : $DEVICE"
    echo "net_period       : $NET_PERIOD s"
    echo "flow_period      : $FLOW_PERIOD s"
    echo "cooldown         : $COOLDOWN_S s"

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo
        echo "GPU:"
        nvidia-smi --query-gpu=name,driver_version,temperature.gpu,pstate --format=csv,noheader 2>/dev/null || true

        local compute_apps
        compute_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
        if [[ -n "$compute_apps" ]]; then
            echo
            echo "WARNING: GPU compute processes are already active:"
            echo "$compute_apps"
            echo "For the final benchmark, run when the target GPU is otherwise idle."
        fi
    fi

    echo "=== PREFLIGHT OK ==="
}

preflight

if [[ "$PREFLIGHT_ONLY" == "true" ]]; then
    exit 0
fi

# ------------------------------------------------------------------------------
# Session output
# ------------------------------------------------------------------------------
RUN_ROOT="$RESULTS_BASE/$SESSION_ID"
mkdir -p "$RUN_ROOT"
RUN_MANIFEST="$RUN_ROOT/run_manifest.csv"
METADATA="$RUN_ROOT/metadata.txt"

if [[ ! -f "$RUN_MANIFEST" ]]; then
    cat > "$RUN_MANIFEST" <<'CSV'
dataset,sample_id,subject,sequence,camera,motion_class,model,order_id,order_position,env_mode,start_time,end_time,wall_seconds,gpu_temp_start_c,gpu_temp_end_c,exit_code,status,latency_rows,latency_csv,run_log
CSV
fi

{
    echo "session_id=$SESSION_ID"
    echo "created=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "dataset_selection=$DATASET"
    echo "only_sample=$ONLY_SAMPLE"
    echo "device=$DEVICE"
    echo "net_period=$NET_PERIOD"
    echo "flow_period=$FLOW_PERIOD"
    echo "movenet_output_period=$MOVENET_OUTPUT_PERIOD"
    echo "yolo_output_period=$YOLO_OUTPUT_PERIOD"
    echo "openpose_output_period=$OPENPOSE_OUTPUT_PERIOD"
    echo "epp_output_period=$EPP_OUTPUT_PERIOD"
    echo "cooldown_s=$COOLDOWN_S"
    echo "venv_dir=$VENV_DIR"
    echo "system_path=$SYSTEM_PATH"
    echo
    echo "[git]"
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true
    git -C "$REPO_ROOT" status --short 2>/dev/null || true
    echo
    echo "[gpu]"
    nvidia-smi 2>/dev/null || true
    echo
    echo "[binaries_sha256]"
    sha256sum "$MOVENET_BIN" "$YOLO_BIN" "$OPENPOSE_BIN" "$EPP_BIN" 2>/dev/null || true
} > "$METADATA"

# Give the machine one idle interval before the first measured run.
if [[ "$DRY_RUN" != "true" && "$COOLDOWN_S" != "0" ]]; then
    echo "Initial cooldown: ${COOLDOWN_S}s"
    sleep "$COOLDOWN_S"
fi

# ------------------------------------------------------------------------------
# One model run
# ------------------------------------------------------------------------------
run_model() {
    local dataset="$1"
    local sample_id="$2"
    local subject="$3"
    local sequence="$4"
    local camera="$5"
    local motion_class="$6"
    local order_id="$7"
    local order_position="$8"
    local model="$9"
    local event_path="${10}"
    local rgb_path="${11}"

    local run_dir="$RUN_ROOT/$dataset/$sample_id/$model"
    local latency_csv="$run_dir/latency.csv"
    local run_log="$run_dir/run.log"
    local epp_server_log="$run_dir/epp_server.log"

    mkdir -p "$run_dir"

    if [[ "$RESUME" == "true" && -f "$latency_csv" && "$(csv_rows "$latency_csv")" -gt 0 ]]; then
        echo "[SKIP] $dataset $sample_id $model -> existing latency.csv"
        return 0
    fi

    rm -f "$latency_csv" "$run_log" "$epp_server_log"

    local env_mode="system"
    local -a cmd=()

    case "$model" in
        movenet)
            cmd=(
                "$MOVENET_BIN"
                --data_file "$event_path"
                --checkpoint_path "$([[ "$dataset" == "h36m" ]] && echo "$H36M_MOVENET_CKPT" || echo "$DHP19_MOVENET_CKPT")"
                --net_period "$NET_PERIOD"
                --flow_period "$FLOW_PERIOD"
                --output_period "$MOVENET_OUTPUT_PERIOD"
                --device "$DEVICE"
                --moveenet_only
                --latency_csv "$latency_csv"
                --no_csv
                --no_video
            )
            if [[ "$dataset" == "h36m" ]]; then
                cmd+=(--w 640 --h 480)
            else
                cmd+=(--dhp19 --w 346 --h 260)
            fi
            ;;

        moveenetofk)
            cmd=(
                "$MOVENET_BIN"
                --data_file "$event_path"
                --checkpoint_path "$([[ "$dataset" == "h36m" ]] && echo "$H36M_MOVENET_CKPT" || echo "$DHP19_MOVENET_CKPT")"
                --net_period "$NET_PERIOD"
                --flow_period "$FLOW_PERIOD"
                --output_period "$MOVENET_OUTPUT_PERIOD"
                --device "$DEVICE"
                --use_lc true
                --latency_csv "$latency_csv"
                --no_csv
                --no_video
            )
            if [[ "$dataset" == "h36m" ]]; then
                cmd+=(--w 640 --h 480)
            else
                cmd+=(--dhp19 --w 346 --h 260)
            fi
            ;;

        yolo)
            [[ "$dataset" == "h36m" ]] || { echo "YOLO is not part of DHP19 latency benchmark." >&2; return 2; }
            env_mode="venv"
            cmd=(
                "$YOLO_BIN"
                --data_file "$rgb_path"
                --net_period "$NET_PERIOD"
                --output_period "$YOLO_OUTPUT_PERIOD"
                --w 640
                --h 480
                --device "$DEVICE"
                --yolo_model_path "$YOLO_MODEL"
                --YoloPose_script "$YOLO_SCRIPT"
                --latency_csv "$latency_csv"
                --no_csv
                --no_video
            )
            ;;

        openpose)
            [[ "$dataset" == "h36m" ]] || { echo "OpenPose is not part of DHP19 latency benchmark." >&2; return 2; }
            cmd=(
                "$OPENPOSE_BIN"
                --data_file "$rgb_path"
                --net_period "$NET_PERIOD"
                --output_period "$OPENPOSE_OUTPUT_PERIOD"
                --w 640
                --h 480
                --device "$DEVICE"
                --op_model_path "$OPENPOSE_MODEL_DIR"
                --latency_csv "$latency_csv"
                --no_csv
                --no_video
            )
            ;;

        eventpointpose)
            [[ "$dataset" == "dhp19" ]] || { echo "EventPointPose is not part of H36M latency benchmark." >&2; return 2; }
            env_mode="venv"
            local camera_id
            case "$camera" in
                ch2dvs) camera_id=2 ;;
                ch3dvs) camera_id=3 ;;
                *) echo "Unsupported EPP camera: $camera" >&2; return 2 ;;
            esac
            cmd=(
                "$EPP_BIN"
                --data_file "$event_path"
                --camera "$camera_id"
                --net_period "$NET_PERIOD"
                --output_period "$EPP_OUTPUT_PERIOD"
                --w 346
                --h 260
                --device "$DEVICE"
                --model_path "$EPP_MODEL"
                --EventPointPose_script "$EPP_SCRIPT"
                --input_preprocessing already_filtered
                --fifo_size 7500
                --num_points 2048
                --seed 1
                --latency_csv "$latency_csv"
                --server_log "$epp_server_log"
                --startup_timeout 30
                --response_timeout 0
                --no_csv
            )
            ;;

        *)
            echo "Unknown model: $model" >&2
            return 2
            ;;
    esac

    echo
    echo "======================================================================"
    echo "Dataset       : $dataset"
    echo "Sample        : $sample_id ($sequence)"
    echo "Model         : $model"
    echo "Order         : $order_id position $order_position"
    echo "Environment   : $env_mode"
    echo "Latency CSV   : $latency_csv"
    echo "Command       : $(quote_command "${cmd[@]}")"
    echo "======================================================================"

    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    local start_time end_time start_epoch end_epoch wall_s temp_start temp_end rc status rows
    start_time="$(date --iso-8601=seconds)"
    start_epoch="$(date +%s)"
    temp_start="$(gpu_temp)"

    set +e
    if [[ "$env_mode" == "venv" ]]; then
        run_venv "${cmd[@]}" > "$run_log" 2>&1
        rc=$?
    else
        run_system "${cmd[@]}" > "$run_log" 2>&1
        rc=$?
    fi
    set -e

    end_time="$(date --iso-8601=seconds)"
    end_epoch="$(date +%s)"
    wall_s=$((end_epoch - start_epoch))
    temp_end="$(gpu_temp)"
    rows="$(csv_rows "$latency_csv")"

    if [[ "$rc" -ne 0 ]]; then
        status="FAILED"
    elif [[ "$rows" -le 0 ]]; then
        status="NO_LATENCY_ROWS"
        rc=90
    else
        status="OK"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$dataset" "$sample_id" "$subject" "$sequence" "$camera" "$motion_class" \
        "$model" "$order_id" "$order_position" "$env_mode" \
        "$start_time" "$end_time" "$wall_s" "${temp_start:-}" "${temp_end:-}" \
        "$rc" "$status" "$rows" "$latency_csv" "$run_log" \
        >> "$RUN_MANIFEST"

    echo "Result        : $status"
    echo "Exit code     : $rc"
    echo "Latency rows  : $rows"
    echo "Wall time     : ${wall_s}s"
    echo "GPU temp      : ${temp_start:-NA} -> ${temp_end:-NA} C"
    echo "Run log       : $run_log"

    if [[ "$rc" -ne 0 ]]; then
        echo "Last log lines:"
        tail -n 30 "$run_log" || true
        if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
            echo "Stopping after failure. Use --continue_on_error to continue." >&2
            return "$rc"
        fi
    fi

    if [[ "$COOLDOWN_S" != "0" ]]; then
        echo "Cooldown      : ${COOLDOWN_S}s"
        sleep "$COOLDOWN_S"
    fi

    return 0
}

run_h36m() {
    local row id subject sequence camera order_id event_path rgb_path
    local order_string model pos

    for row in "${H36M_SAMPLES[@]}"; do
        IFS='|' read -r id subject sequence camera order_id event_path rgb_path <<< "$row"

        if [[ -n "$ONLY_SAMPLE" && "$id" != "$ONLY_SAMPLE" ]]; then
            continue
        fi

        order_string="$(models_for_order "$order_id")"
        pos=0
        for model in $order_string; do
            pos=$((pos + 1))
            run_model \
                "h36m" "$id" "$subject" "$sequence" "$camera" "mixed" \
                "$order_id" "$pos" "$model" "$event_path" "$rgb_path"
        done
    done
}

run_dhp19() {
    local row id subject sequence camera motion_class motion_name order_id event_path
    local order_string model pos

    for row in "${DHP19_SAMPLES[@]}"; do
        IFS='|' read -r id subject sequence camera motion_class motion_name order_id event_path <<< "$row"

        if [[ -n "$ONLY_SAMPLE" && "$id" != "$ONLY_SAMPLE" ]]; then
            continue
        fi

        order_string="$(models_for_order "$order_id")"
        pos=0
        for model in $order_string; do
            pos=$((pos + 1))
            run_model \
                "dhp19" "$id" "$subject" "$sequence" "$camera" "$motion_class" \
                "$order_id" "$pos" "$model" "$event_path" ""
        done
    done
}

echo
echo "=== LATENCY EXPERIMENT ==="
echo "Session       : $SESSION_ID"
echo "Results       : $RUN_ROOT"
echo "Dataset       : $DATASET"
echo "Sample filter : ${ONLY_SAMPLE:-ALL}"
echo "Resume        : $RESUME"
echo "Dry run       : $DRY_RUN"
echo

case "$DATASET" in
    h36m) run_h36m ;;
    dhp19) run_dhp19 ;;
    both)
        run_h36m
        run_dhp19
        ;;
esac

echo
echo "Experiment completed."
echo "Results      : $RUN_ROOT"
echo "Run manifest : $RUN_MANIFEST"
echo "Metadata     : $METADATA"