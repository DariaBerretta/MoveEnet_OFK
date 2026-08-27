#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# MoveEnet_OFK - Energy vs Network Inference Rate experiment
#
# HOST-side measurement:
#   CPU package energy : Intel RAPL
#   GPU energy         : NVIDIA NVML cumulative energy counter
#
# H36M:
#   MoveNet-only
#   MoveEnetOFK
#   OpenPose
#   YOLOPose
#
# DHP19:
#   MoveNet-only
#   MoveEnetOFK
#   EventPointPose
#
# Experimental factors:
#   network_period = 0.02, 0.05, 0.1, 0.2, 0.5 s
#                  = 50,   20,   10,  5,   2 Hz
#
# MoveEnetOFK:
#   flow_period   fixed at 0.005 s (200 Hz)
#   output_period fixed at 0.005 s (200 Hz)
#
# Experimental policy:
#   - same samples as ExpLatency
#   - same balanced model orders as ExpLatency
#   - rotated network-period order across samples
#   - same offline replay behaviour
#   - energy instrumentation external to the HPE pipelines
#
# Run this script on the HOST.
# ==============================================================================


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_HOST="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LATENCY_DIR="$SCRIPT_DIR/../ExpLatency"

H36M_MANIFEST="$LATENCY_DIR/manifests/h36m_samples.sh"
DHP19_MANIFEST="$LATENCY_DIR/manifests/dhp19_samples.sh"

MEASURE_SCRIPT="${MEASURE_SCRIPT:-$SCRIPT_DIR/measure_energy.py}"
ENERGY_PYTHON="${ENERGY_PYTHON:-python3}"

CONTAINER="${CONTAINER:-moveEnetOFK}"

CONTAINER_REPO="/workspace/moveEnetFlow"
BUILD_DIR="$CONTAINER_REPO/build"

MOVENET_BIN="$BUILD_DIR/moveEnetOFK_offline"
YOLO_BIN="$BUILD_DIR/YoloPose_offline"
OPENPOSE_BIN="$BUILD_DIR/OpenPose_offline"
EPP_BIN="$BUILD_DIR/eventPointPose_offline"

H36M_MOVENET_CKPT="${H36M_MOVENET_CKPT:-/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth}"
DHP19_MOVENET_CKPT="${DHP19_MOVENET_CKPT:-/usr/local/src/hpe-core/example/movenet/models/dhp19_allcams_e33_valacc0.87996.pth}"

YOLO_MODEL="${YOLO_MODEL:-/workspace/model_mounts/YoloPose/yolo26n-pose.pt}"
YOLO_SCRIPT="${YOLO_SCRIPT:-/workspace/model_mounts/YoloPose/YoloPose_yarp_server.py}"

OPENPOSE_MODEL_DIR="${OPENPOSE_MODEL_DIR:-/usr/local/src/openpose/models/}"
OPENPOSE_BODY25_MODEL="$OPENPOSE_MODEL_DIR/pose/body_25/pose_iter_584000.caffemodel"

EPP_MODEL="${EPP_MODEL:-/workspace/model_mounts/eventpointpose/PointNet/models/model.pth}"
EPP_SCRIPT="${EPP_SCRIPT:-/workspace/model_mounts/eventpointpose/PointNet/models/eventPointPose_yarp_server.py}"


# ------------------------------------------------------------------------------
# Experimental configuration
# ------------------------------------------------------------------------------

DEVICE="${DEVICE:-cuda:0}"

# Common inference-rate grid for ALL methods.
#
# 0.02 s -> 50 Hz
# 0.05 s -> 20 Hz
# 0.10 s -> 10 Hz
# 0.20 s ->  5 Hz
# 0.50 s ->  2 Hz
NETWORK_PERIODS=("0.02" "0.05" "0.1" "0.2" "0.5")

# OF remains high-rate while the DNN rate changes.
FLOW_PERIOD="0.005"

MOVENET_OUTPUT_PERIOD="0.005"
YOLO_OUTPUT_PERIOD="0.02"
OPENPOSE_OUTPUT_PERIOD="0.02"
EPP_OUTPUT_PERIOD="0.005"

COOLDOWN_S="20"

DATASET="both"
ONLY_SAMPLE=""

SESSION_ID="$(date +%Y%m%d_%H%M%S)"

RESULTS_BASE="${RESULTS_BASE:-$HOME/data/MoveEnet_OFK_results/Energy}"

RESUME="false"
DRY_RUN="false"
PREFLIGHT_ONLY="false"
CONTINUE_ON_ERROR="false"

IDLE_BASELINE_S="0"


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

split_csv_into_array() {
    local csv="$1"
    local -n result="$2"
    IFS=',' read -r -a result <<< "$csv"
}

is_non_negative_number() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

is_positive_number() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
        awk -v x="$1" 'BEGIN { exit !(x > 0) }'
}

safe_period() {
    local p="$1"
    echo "${p//./p}"
}

period_to_hz() {
    local p="$1"
    awk -v p="$p" 'BEGIN { printf "%.6f", 1.0/p }'
}

require_host_file() {
    [[ -f "$1" ]] || {
        echo "Missing host file: $1" >&2
        return 1
    }
}

require_host_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing host command: $1" >&2
        return 1
    }
}

require_container_file() {
    docker exec "$CONTAINER" test -f "$1" || {
        echo "Missing container file: $1" >&2
        return 1
    }
}

require_container_dir() {
    docker exec "$CONTAINER" test -d "$1" || {
        echo "Missing container directory: $1" >&2
        return 1
    }
}

require_container_executable() {
    docker exec "$CONTAINER" test -x "$1" || {
        echo "Missing/non-executable container binary: $1" >&2
        return 1
    }
}

quote_command() {
    local out=""
    printf -v out '%q ' "$@"
    printf '%s' "$out"
}

csv_rows() {
    local f="$1"

    if [[ ! -f "$f" ]]; then
        echo 0
        return
    fi

    local lines
    lines="$(wc -l < "$f")"

    if [[ "$lines" -le 1 ]]; then
        echo 0
    else
        echo $((lines - 1))
    fi
}

cooldown() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return
    fi

    if [[ "$COOLDOWN_S" != "0" ]]; then
        echo "Cooldown: ${COOLDOWN_S}s"
        sleep "$COOLDOWN_S"
    fi
}

check_no_sidecars() {
    local processes

    processes="$(
        docker exec "$CONTAINER" bash -lc \
            "pgrep -af '[m]ovenet_online.py|[Y]oloPose_yarp_server.py|[e]ventPointPose_yarp_server.py' || true"
    )"

    if [[ -n "$processes" ]]; then
        echo
        echo "ERROR: HPE sidecar process already running inside $CONTAINER:"
        echo "$processes"
        echo
        echo "Do not start an energy measurement with another HPE process active."
        return 1
    fi
}

warn_gpu_processes() {
    local apps

    apps="$(
        nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader \
            2>/dev/null || true
    )"

    if [[ -n "$apps" ]]; then
        echo
        echo "WARNING: GPU compute processes currently active:"
        echo "$apps"
        echo
        echo "For the final benchmark the GPU should otherwise be idle."
    fi
}


# ------------------------------------------------------------------------------
# Usage
# ------------------------------------------------------------------------------

usage() {
    cat <<USAGE

Usage:
  $(basename "$0") [options]

Options:

  --dataset <both|h36m|dhp19>
      Dataset selection.
      Default: both

  --sample <ID>
      Run one sample only, e.g. H01 or D01.

  --session <name>
      Results session name.

  --results_base <path>
      Host-side results root.

  --network_periods <list>
      Comma-separated network periods used by ALL methods.
      Default: 0.02,0.05,0.1,0.2,0.5

  --net_period <seconds>
      Run only one network period.

  --flow_period <seconds>
      Optical-flow period for MoveEnetOFK.
      Default: 0.005

  --cooldown <seconds>
      Delay after every measured run.
      Default: 20

  --idle_baseline <seconds>
      Optional idle baseline before the experiment.
      Example: 300
      Default: 0

  --device <device>
      Default: cuda:0

  --container <name>
      Default: moveEnetOFK

  --resume
      Skip successful model-period runs.

  --continue_on_error
      Continue after a failed run.

  --preflight-only
      Validate setup and exit.

  --dry-run
      Print experiment commands but do not execute.

  --help


Examples:

  ./run_energy_experiment.sh --preflight-only

  ./run_energy_experiment.sh \
      --dataset h36m \
      --sample H01 \
      --session energy_grid_H01 \
      --network_periods 0.02,0.05,0.1,0.2,0.5

  ./run_energy_experiment.sh \
      --dataset both \
      --session energy_accuracy_main \
      --network_periods 0.02,0.05,0.1,0.2,0.5 \
      --flow_period 0.005 \
      --cooldown 20 \
      --idle_baseline 300

USAGE
}


# ------------------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in

        --dataset)
            DATASET="$2"
            shift 2
            ;;

        --sample)
            ONLY_SAMPLE="$2"
            shift 2
            ;;

        --session)
            SESSION_ID="$2"
            shift 2
            ;;

        --results_base)
            RESULTS_BASE="$2"
            shift 2
            ;;

        --network_periods|--periods)
            split_csv_into_array "$2" NETWORK_PERIODS
            shift 2
            ;;

        --net_period|--network_period)
            NETWORK_PERIODS=("$2")
            shift 2
            ;;

        --flow_period)
            FLOW_PERIOD="$2"
            shift 2
            ;;

        --cooldown)
            COOLDOWN_S="$2"
            shift 2
            ;;

        --idle_baseline)
            IDLE_BASELINE_S="$2"
            shift 2
            ;;

        --device)
            DEVICE="$2"
            shift 2
            ;;

        --container)
            CONTAINER="$2"
            shift 2
            ;;

        --resume)
            RESUME="true"
            shift
            ;;

        --continue_on_error)
            CONTINUE_ON_ERROR="true"
            shift
            ;;

        --preflight-only)
            PREFLIGHT_ONLY="true"
            shift
            ;;

        --dry-run)
            DRY_RUN="true"
            shift
            ;;

        --help)
            usage
            exit 0
            ;;

        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done


case "$DATASET" in
    both|h36m|dhp19)
        ;;
    *)
        echo "Invalid dataset: $DATASET" >&2
        exit 2
        ;;
esac

if [[ ${#NETWORK_PERIODS[@]} -eq 0 ]]; then
    echo "No network periods specified." >&2
    exit 2
fi

for NP in "${NETWORK_PERIODS[@]}"; do
    if ! is_positive_number "$NP"; then
        echo "Invalid network period: $NP" >&2
        exit 2
    fi
done

if ! is_positive_number "$FLOW_PERIOD"; then
    echo "Invalid flow period: $FLOW_PERIOD" >&2
    exit 2
fi

if ! is_non_negative_number "$COOLDOWN_S"; then
    echo "Invalid cooldown: $COOLDOWN_S" >&2
    exit 2
fi

if ! is_non_negative_number "$IDLE_BASELINE_S"; then
    echo "Invalid idle baseline: $IDLE_BASELINE_S" >&2
    exit 2
fi


# ------------------------------------------------------------------------------
# Model order
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

        *)
            echo "Unknown model order: $1" >&2
            return 1
            ;;
    esac
}


# ------------------------------------------------------------------------------
# Network-period order
#
# Rotate the period grid according to sample ID.
#
# With five periods:
#
# P0 = 50,20,10,5,2 Hz
# P1 = 20,10,5,2,50 Hz
# P2 = 10,5,2,50,20 Hz
# P3 = 5,2,50,20,10 Hz
# P4 = 2,50,20,10,5 Hz
#
# H01/D01 -> P0
# H02/D02 -> P1
# ...
#
# With 12 samples, order frequencies are distributed 3,3,2,2,2.
# ------------------------------------------------------------------------------

period_offset_for_sample() {
    local sample_id="$1"
    local suffix="${sample_id:1}"
    local n="${#NETWORK_PERIODS[@]}"

    local sample_number=$((10#$suffix))
    echo $(((sample_number - 1) % n))
}

period_order_id_for_sample() {
    local offset
    offset="$(period_offset_for_sample "$1")"
    echo "P${offset}"
}

periods_for_sample() {
    local sample_id="$1"
    local offset
    offset="$(period_offset_for_sample "$sample_id")"

    local n="${#NETWORK_PERIODS[@]}"
    local i
    local idx
    local values=()

    for ((i = 0; i < n; i++)); do
        idx=$(((offset + i) % n))
        values+=("${NETWORK_PERIODS[$idx]}")
    done

    echo "${values[*]}"
}


# ------------------------------------------------------------------------------
# Manifests
# ------------------------------------------------------------------------------

require_host_file "$H36M_MANIFEST"
require_host_file "$DHP19_MANIFEST"

# shellcheck disable=SC1090
source "$H36M_MANIFEST"

# shellcheck disable=SC1090
source "$DHP19_MANIFEST"


# ------------------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------------------

preflight() {

    echo
    echo "======================================================================"
    echo "ENERGY EXPERIMENT PREFLIGHT"
    echo "======================================================================"

    require_host_command docker
    require_host_command nvidia-smi
    require_host_command "$ENERGY_PYTHON"

    require_host_file "$MEASURE_SCRIPT"

    echo
    echo "Host energy Python:"
    "$ENERGY_PYTHON" -c '
import sys
import pynvml
print(sys.executable)
print("pynvml: OK")
'

    echo
    echo "RAPL:"

    [[ -r /sys/class/powercap/intel-rapl:0/energy_uj ]] || {
        echo "CPU package RAPL counter is not readable." >&2
        return 1
    }

    [[ -r /sys/class/powercap/intel-rapl:1/energy_uj ]] || {
        echo "RAPL psys counter is not readable." >&2
        return 1
    }

    echo "CPU package RAPL readable: OK"
    echo "RAPL psys readable       : OK"

    echo
    echo "Docker container:"

    local running
    running="$(
        docker inspect \
            -f '{{.State.Running}}' \
            "$CONTAINER" \
            2>/dev/null || true
    )"

    [[ "$running" == "true" ]] || {
        echo "Container not running: $CONTAINER" >&2
        return 1
    }

    echo "$CONTAINER running: OK"

    echo
    echo "YARP:"

    local yarp_status
    yarp_status="$(docker exec "$CONTAINER" yarp detect 2>&1 || true)"
    echo "$yarp_status"

    grep -q "FOUND" <<< "$yarp_status" || {
        echo "YARP server not detected." >&2
        return 1
    }

    echo
    echo "Container Python:"

    docker exec "$CONTAINER" bash -lc '
echo "python3 = $(command -v python3)"
python3 - <<PY
import sys
import torch
import ultralytics
print("executable   :", sys.executable)
print("torch        :", torch.__version__)
print("ultralytics  :", ultralytics.__version__)
PY
'

    echo
    echo "Binaries:"

    require_container_executable "$MOVENET_BIN"
    require_container_executable "$YOLO_BIN"
    require_container_executable "$OPENPOSE_BIN"
    require_container_executable "$EPP_BIN"

    echo "All HPE binaries: OK"

    echo
    echo "Models:"

    require_container_file "$H36M_MOVENET_CKPT"
    require_container_file "$DHP19_MOVENET_CKPT"

    require_container_file "$YOLO_MODEL"
    require_container_file "$YOLO_SCRIPT"

    require_container_dir "$OPENPOSE_MODEL_DIR"
    require_container_file "$OPENPOSE_BODY25_MODEL"

    require_container_file "$EPP_MODEL"
    require_container_file "$EPP_SCRIPT"

    echo "All model files: OK"

    local h_samples=0
    local d_samples=0

    if [[ "$DATASET" == "both" || "$DATASET" == "h36m" ]]; then

        local row id subject sequence camera order_id event_path rgb_path

        for row in "${H36M_SAMPLES[@]}"; do

            IFS='|' read -r \
                id subject sequence camera order_id event_path rgb_path \
                <<< "$row"

            if [[ -n "$ONLY_SAMPLE" && "$id" != "$ONLY_SAMPLE" ]]; then
                continue
            fi

            require_container_file "$event_path"
            require_container_file "$rgb_path"

            models_for_order "$order_id" >/dev/null

            h_samples=$((h_samples + 1))
        done
    fi

    if [[ "$DATASET" == "both" || "$DATASET" == "dhp19" ]]; then

        local row id subject sequence camera motion_class motion_name
        local order_id event_path

        for row in "${DHP19_SAMPLES[@]}"; do

            IFS='|' read -r \
                id subject sequence camera motion_class motion_name \
                order_id event_path \
                <<< "$row"

            if [[ -n "$ONLY_SAMPLE" && "$id" != "$ONLY_SAMPLE" ]]; then
                continue
            fi

            require_container_file "$event_path"
            models_for_order "$order_id" >/dev/null

            case "$camera" in
                ch2dvs|ch3dvs) ;;
                *)
                    echo "Invalid DHP19 camera: $camera" >&2
                    return 1
                    ;;
            esac

            d_samples=$((d_samples + 1))
        done
    fi

    local total_samples=$((h_samples + d_samples))

    if [[ "$total_samples" -eq 0 ]]; then
        echo "No samples selected." >&2
        return 1
    fi

    local n_periods="${#NETWORK_PERIODS[@]}"
    local total_runs=$(
        echo $((h_samples * 4 * n_periods + d_samples * 3 * n_periods))
    )

    echo
    echo "Experiment configuration:"
    echo "Dataset            : $DATASET"
    echo "Sample             : ${ONLY_SAMPLE:-ALL}"
    echo "H36M samples       : $h_samples"
    echo "DHP19 samples      : $d_samples"
    echo "Network periods    : ${NETWORK_PERIODS[*]}"
    echo "Network rates [Hz] : $(for p in "${NETWORK_PERIODS[@]}"; do period_to_hz "$p"; printf ' '; done)"
    echo "Flow period        : $FLOW_PERIOD s"
    echo "Cooldown           : $COOLDOWN_S s"
    echo "Idle baseline      : $IDLE_BASELINE_S s"
    echo "Total model runs   : $total_runs"
    echo "Results base       : $RESULTS_BASE"

    echo
    echo "GPU:"

    nvidia-smi \
        --query-gpu=name,uuid,driver_version,power.limit,temperature.gpu,pstate \
        --format=csv,noheader

    warn_gpu_processes
    check_no_sidecars

    echo
    echo "=== PREFLIGHT OK ==="
}


preflight

if [[ "$PREFLIGHT_ONLY" == "true" ]]; then
    exit 0
fi


# ------------------------------------------------------------------------------
# Session
# ------------------------------------------------------------------------------

RUN_ROOT="$RESULTS_BASE/$SESSION_ID"

if [[ -d "$RUN_ROOT" &&
      "$RESUME" != "true" &&
      "$DRY_RUN" != "true" ]]; then

    if find "$RUN_ROOT" -mindepth 1 -print -quit | grep -q .; then
        echo "Session already exists: $RUN_ROOT" >&2
        echo "Use another --session or --resume." >&2
        exit 2
    fi
fi

mkdir -p "$RUN_ROOT"

RUN_MANIFEST="$RUN_ROOT/run_manifest.csv"
METADATA="$RUN_ROOT/metadata.txt"

if [[ ! -f "$RUN_MANIFEST" ]]; then
    cat > "$RUN_MANIFEST" <<'CSV'
dataset,sample_id,subject,sequence,camera,motion_class,model,model_order_id,model_order_position,period_order_id,period_order_position,net_period,net_hz,flow_period,output_period,start_time,end_time,exit_code,status,energy_csv,run_log,epp_server_log
CSV
fi


# ------------------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------------------

{
    echo "session_id=$SESSION_ID"
    echo "created=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "container=$CONTAINER"

    echo "dataset_selection=$DATASET"
    echo "only_sample=$ONLY_SAMPLE"

    echo "device=$DEVICE"

    echo "network_periods=${NETWORK_PERIODS[*]}"
    echo "flow_period=$FLOW_PERIOD"

    echo "movenet_output_period=$MOVENET_OUTPUT_PERIOD"
    echo "yolo_output_period=$YOLO_OUTPUT_PERIOD"
    echo "openpose_output_period=$OPENPOSE_OUTPUT_PERIOD"
    echo "epp_output_period=$EPP_OUTPUT_PERIOD"

    echo "cooldown_s=$COOLDOWN_S"
    echo "idle_baseline_s=$IDLE_BASELINE_S"

    echo
    echo "[period_orders]"

    for ((offset = 0; offset < ${#NETWORK_PERIODS[@]}; offset++)); do

        vals=()

        for ((i = 0; i < ${#NETWORK_PERIODS[@]}; i++)); do
            idx=$(((offset + i) % ${#NETWORK_PERIODS[@]}))
            vals+=("${NETWORK_PERIODS[$idx]}")
        done

        echo "P${offset}=${vals[*]}"
    done

    echo
    echo "[git]"
    git -C "$REPO_ROOT_HOST" rev-parse HEAD 2>/dev/null || true
    git -C "$REPO_ROOT_HOST" status --short 2>/dev/null || true

    echo
    echo "[host]"
    uname -a || true

    echo
    echo "[cpu]"
    lscpu || true

    echo
    echo "[gpu]"
    nvidia-smi || true

    echo
    echo "[docker]"
    docker inspect \
        -f 'container={{.Name}} image={{.Config.Image}} image_id={{.Image}}' \
        "$CONTAINER" \
        2>/dev/null || true

    echo
    echo "[measure_energy_sha256]"
    sha256sum "$MEASURE_SCRIPT" 2>/dev/null || true

    echo
    echo "[binaries_sha256]"
    docker exec "$CONTAINER" sha256sum \
        "$MOVENET_BIN" \
        "$YOLO_BIN" \
        "$OPENPOSE_BIN" \
        "$EPP_BIN" \
        2>/dev/null || true

} > "$METADATA"


# ------------------------------------------------------------------------------
# Idle baseline
# ------------------------------------------------------------------------------

if [[ "$DRY_RUN" != "true" && "$IDLE_BASELINE_S" != "0" ]]; then

    echo
    echo "======================================================================"
    echo "IDLE BASELINE - ${IDLE_BASELINE_S}s"
    echo "======================================================================"

    cooldown
    check_no_sidecars

    "$ENERGY_PYTHON" "$MEASURE_SCRIPT" \
        --csv "$RUN_ROOT/idle_baseline.csv" \
        --log "$RUN_ROOT/idle_baseline.log" \
        --label "idle_${IDLE_BASELINE_S}s" \
        --record-psys \
        -- \
        sleep "$IDLE_BASELINE_S"

    cooldown

else

    cooldown
fi


# ------------------------------------------------------------------------------
# One model / one network period
# ------------------------------------------------------------------------------

run_model() {

    local dataset="$1"
    local sample_id="$2"
    local subject="$3"
    local sequence="$4"
    local camera="$5"
    local motion_class="$6"

    local model_order_id="$7"
    local model_order_position="$8"

    local period_order_id="$9"
    local period_order_position="${10}"

    local net_period="${11}"
    local model="${12}"

    local event_path="${13}"
    local rgb_path="${14}"

    local safe_np
    safe_np="$(safe_period "$net_period")"

    local net_hz
    net_hz="$(period_to_hz "$net_period")"

    local run_dir="$RUN_ROOT/$dataset/$sample_id/np_${safe_np}/$model"

    local energy_csv="$run_dir/energy.csv"
    local run_log="$run_dir/run.log"
    local success_marker="$run_dir/OK"

    local epp_server_log=""
    local epp_tmp_log=""

    mkdir -p "$run_dir"

    if [[ "$RESUME" == "true" && -f "$success_marker" ]]; then

        echo
        echo "[SKIP] $dataset $sample_id np=$net_period $model"

        return 0
    fi

    rm -f \
        "$energy_csv" \
        "$run_log" \
        "$success_marker" \
        "$run_dir/epp_server.log"

    check_no_sidecars
    warn_gpu_processes

    local output_period=""
    local checkpoint=""

    local -a target_cmd=()
    local -a measure_cmd=()


    case "$model" in

        # ----------------------------------------------------------------------
        # MoveNet-only
        # ----------------------------------------------------------------------

        movenet)

            output_period="$MOVENET_OUTPUT_PERIOD"

            if [[ "$dataset" == "h36m" ]]; then
                checkpoint="$H36M_MOVENET_CKPT"
            else
                checkpoint="$DHP19_MOVENET_CKPT"
            fi

            target_cmd=(
                docker exec "$CONTAINER"

                "$MOVENET_BIN"

                --data_file "$event_path"
                --checkpoint_path "$checkpoint"

                --net_period "$net_period"
                --flow_period "$FLOW_PERIOD"
                --output_period "$MOVENET_OUTPUT_PERIOD"

                --device "$DEVICE"

                --moveenet_only

                --no_csv
                --no_video
            )

            if [[ "$dataset" == "h36m" ]]; then
                target_cmd+=(--w 640 --h 480)
            else
                target_cmd+=(--dhp19 --w 346 --h 260)
            fi
            ;;


        # ----------------------------------------------------------------------
        # MoveEnetOFK
        # ----------------------------------------------------------------------

        moveenetofk)

            output_period="$MOVENET_OUTPUT_PERIOD"

            if [[ "$dataset" == "h36m" ]]; then
                checkpoint="$H36M_MOVENET_CKPT"
            else
                checkpoint="$DHP19_MOVENET_CKPT"
            fi

            target_cmd=(
                docker exec "$CONTAINER"

                "$MOVENET_BIN"

                --data_file "$event_path"
                --checkpoint_path "$checkpoint"

                --net_period "$net_period"
                --flow_period "$FLOW_PERIOD"
                --output_period "$MOVENET_OUTPUT_PERIOD"

                --device "$DEVICE"

                --use_lc true

                --no_csv
                --no_video
            )

            if [[ "$dataset" == "h36m" ]]; then
                target_cmd+=(--w 640 --h 480)
            else
                target_cmd+=(--dhp19 --w 346 --h 260)
            fi
            ;;


        # ----------------------------------------------------------------------
        # YOLOPose
        # ----------------------------------------------------------------------

        yolo)

            [[ "$dataset" == "h36m" ]] || {
                echo "YOLO is H36M-only." >&2
                return 2
            }

            output_period="$YOLO_OUTPUT_PERIOD"

            target_cmd=(
                docker exec "$CONTAINER"

                "$YOLO_BIN"

                --data_file "$rgb_path"

                --net_period "$net_period"
                --output_period "$YOLO_OUTPUT_PERIOD"

                --w 640
                --h 480

                --device "$DEVICE"

                --yolo_model_path "$YOLO_MODEL"
                --YoloPose_script "$YOLO_SCRIPT"

                --no_csv
                --no_video
            )
            ;;


        # ----------------------------------------------------------------------
        # OpenPose
        # ----------------------------------------------------------------------

        openpose)

            [[ "$dataset" == "h36m" ]] || {
                echo "OpenPose is H36M-only." >&2
                return 2
            }

            output_period="$OPENPOSE_OUTPUT_PERIOD"

            target_cmd=(
                docker exec "$CONTAINER"

                "$OPENPOSE_BIN"

                --data_file "$rgb_path"

                --net_period "$net_period"
                --output_period "$OPENPOSE_OUTPUT_PERIOD"

                --w 640
                --h 480

                --device "$DEVICE"

                --op_model_path "$OPENPOSE_MODEL_DIR"

                --no_csv
                --no_video
            )
            ;;


        # ----------------------------------------------------------------------
        # EventPointPose
        # ----------------------------------------------------------------------

        eventpointpose)

            [[ "$dataset" == "dhp19" ]] || {
                echo "EventPointPose is DHP19-only." >&2
                return 2
            }

            output_period="$EPP_OUTPUT_PERIOD"

            local camera_id

            case "$camera" in
                ch2dvs) camera_id=2 ;;
                ch3dvs) camera_id=3 ;;
                *)
                    echo "Unsupported EPP camera: $camera" >&2
                    return 2
                    ;;
            esac

            epp_server_log="$run_dir/epp_server.log"

            epp_tmp_log="/tmp/energy_${sample_id}_np_${safe_np}_${$}_epp.log"

            docker exec "$CONTAINER" rm -f "$epp_tmp_log" \
                >/dev/null 2>&1 || true

            target_cmd=(
                docker exec "$CONTAINER"

                "$EPP_BIN"

                --data_file "$event_path"
                --camera "$camera_id"

                --net_period "$net_period"
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

                --server_log "$epp_tmp_log"

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


    # --------------------------------------------------------------------------
    # Energy wrapper
    # --------------------------------------------------------------------------

    measure_cmd=(
        "$ENERGY_PYTHON"
        "$MEASURE_SCRIPT"

        --csv "$energy_csv"
        --log "$run_log"

        --label "${sample_id}_${model}_np_${safe_np}"

        --dataset "$dataset"
        --sample-id "$sample_id"
        --model "$model"

        --order-id "$model_order_id"
        --order-position "$model_order_position"

        --net-period "$net_period"

        --record-psys
    )

    if [[ "$model" == "movenet" ||
          "$model" == "moveenetofk" ]]; then

        measure_cmd+=(
            --flow-period "$FLOW_PERIOD"
        )
    fi

    measure_cmd+=(-- "${target_cmd[@]}")


    echo
    echo "======================================================================"
    echo "ENERGY RUN"
    echo "======================================================================"
    echo "Dataset          : $dataset"
    echo "Sample           : $sample_id"
    echo "Sequence         : $sequence"
    echo "Model            : $model"
    echo "Model order      : $model_order_id / $model_order_position"
    echo "Period order     : $period_order_id / $period_order_position"
    echo "net_period       : $net_period s"
    echo "network rate     : $net_hz Hz"
    echo "flow_period      : $FLOW_PERIOD s"
    echo "output_period    : $output_period s"
    echo "Energy CSV       : $energy_csv"
    echo "Run log          : $run_log"
    echo "Command          : $(quote_command "${measure_cmd[@]}")"
    echo "======================================================================"


    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi


    local start_time
    local end_time
    local rc
    local status
    local rows

    start_time="$(date --iso-8601=seconds)"

    set +e
    "${measure_cmd[@]}"
    rc=$?
    set -e

    end_time="$(date --iso-8601=seconds)"


    # --------------------------------------------------------------------------
    # EventPointPose sidecar log -> host
    # --------------------------------------------------------------------------

    if [[ "$model" == "eventpointpose" &&
          -n "$epp_tmp_log" ]]; then

        if docker exec "$CONTAINER" test -f "$epp_tmp_log"; then

            docker cp \
                "$CONTAINER:$epp_tmp_log" \
                "$epp_server_log" \
                >/dev/null

            docker exec "$CONTAINER" rm -f "$epp_tmp_log" \
                >/dev/null 2>&1 || true
        fi
    fi


    rows="$(csv_rows "$energy_csv")"

    if [[ "$rc" -ne 0 ]]; then

        status="FAILED"

    elif [[ "$rows" -le 0 ]]; then

        status="NO_ENERGY_ROW"
        rc=91

    else

        status="OK"
        date --iso-8601=seconds > "$success_marker"
    fi


    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$dataset" \
        "$sample_id" \
        "$subject" \
        "$sequence" \
        "$camera" \
        "$motion_class" \
        "$model" \
        "$model_order_id" \
        "$model_order_position" \
        "$period_order_id" \
        "$period_order_position" \
        "$net_period" \
        "$net_hz" \
        "$FLOW_PERIOD" \
        "$output_period" \
        "$start_time" \
        "$end_time" \
        "$rc" \
        "$status" \
        "$energy_csv" \
        "$run_log" \
        "$epp_server_log" \
        >> "$RUN_MANIFEST"


    echo
    echo "Result           : $status"
    echo "Exit code        : $rc"
    echo "Energy rows      : $rows"

    if [[ "$rc" -ne 0 ]]; then

        echo
        echo "Last log lines:"
        tail -n 40 "$run_log" 2>/dev/null || true

        if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
            echo "Stopping after failure."
            return "$rc"
        fi
    fi


    check_no_sidecars

    cooldown
}


# ------------------------------------------------------------------------------
# H36M
# ------------------------------------------------------------------------------

run_h36m() {

    local row
    local id
    local subject
    local sequence
    local camera
    local model_order_id
    local event_path
    local rgb_path

    local model_order
    local period_order
    local period_order_id

    local model
    local net_period

    local model_pos
    local period_pos

    for row in "${H36M_SAMPLES[@]}"; do

        IFS='|' read -r \
            id \
            subject \
            sequence \
            camera \
            model_order_id \
            event_path \
            rgb_path \
            <<< "$row"

        if [[ -n "$ONLY_SAMPLE" &&
              "$id" != "$ONLY_SAMPLE" ]]; then
            continue
        fi


        model_order="$(models_for_order "$model_order_id")"

        period_order="$(periods_for_sample "$id")"
        period_order_id="$(period_order_id_for_sample "$id")"


        echo
        echo "######################################################################"
        echo "H36M SAMPLE: $id"
        echo "Sequence     : $sequence"
        echo "Model order  : $model_order_id -> $model_order"
        echo "Period order : $period_order_id -> $period_order"
        echo "######################################################################"


        period_pos=0

        for net_period in $period_order; do

            period_pos=$((period_pos + 1))
            model_pos=0

            for model in $model_order; do

                model_pos=$((model_pos + 1))

                run_model \
                    "h36m" \
                    "$id" \
                    "$subject" \
                    "$sequence" \
                    "$camera" \
                    "mixed" \
                    "$model_order_id" \
                    "$model_pos" \
                    "$period_order_id" \
                    "$period_pos" \
                    "$net_period" \
                    "$model" \
                    "$event_path" \
                    "$rgb_path"
            done
        done
    done
}


# ------------------------------------------------------------------------------
# DHP19
# ------------------------------------------------------------------------------

run_dhp19() {

    local row
    local id
    local subject
    local sequence
    local camera
    local motion_class
    local motion_name
    local model_order_id
    local event_path

    local model_order
    local period_order
    local period_order_id

    local model
    local net_period

    local model_pos
    local period_pos


    for row in "${DHP19_SAMPLES[@]}"; do

        IFS='|' read -r \
            id \
            subject \
            sequence \
            camera \
            motion_class \
            motion_name \
            model_order_id \
            event_path \
            <<< "$row"

        if [[ -n "$ONLY_SAMPLE" &&
              "$id" != "$ONLY_SAMPLE" ]]; then
            continue
        fi


        model_order="$(models_for_order "$model_order_id")"

        period_order="$(periods_for_sample "$id")"
        period_order_id="$(period_order_id_for_sample "$id")"


        echo
        echo "######################################################################"
        echo "DHP19 SAMPLE: $id"
        echo "Sequence      : $sequence"
        echo "Motion        : $motion_name"
        echo "Model order   : $model_order_id -> $model_order"
        echo "Period order  : $period_order_id -> $period_order"
        echo "######################################################################"


        period_pos=0

        for net_period in $period_order; do

            period_pos=$((period_pos + 1))
            model_pos=0

            for model in $model_order; do

                model_pos=$((model_pos + 1))

                run_model \
                    "dhp19" \
                    "$id" \
                    "$subject" \
                    "$sequence" \
                    "$camera" \
                    "$motion_class" \
                    "$model_order_id" \
                    "$model_pos" \
                    "$period_order_id" \
                    "$period_pos" \
                    "$net_period" \
                    "$model" \
                    "$event_path" \
                    ""
            done
        done
    done
}


# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------

echo
echo "======================================================================"
echo "ENERGY vs NETWORK RATE EXPERIMENT"
echo "======================================================================"
echo "Session          : $SESSION_ID"
echo "Results          : $RUN_ROOT"
echo "Dataset          : $DATASET"
echo "Sample           : ${ONLY_SAMPLE:-ALL}"
echo "Network periods  : ${NETWORK_PERIODS[*]}"
echo "Flow period      : $FLOW_PERIOD s"
echo "Cooldown         : $COOLDOWN_S s"
echo "Container        : $CONTAINER"
echo "======================================================================"


if [[ "$DATASET" == "both" ||
      "$DATASET" == "h36m" ]]; then

    run_h36m
fi


if [[ "$DATASET" == "both" ||
      "$DATASET" == "dhp19" ]]; then

    run_dhp19
fi


echo
echo "======================================================================"
echo "EXPERIMENT COMPLETE"
echo "======================================================================"
echo "Session   : $SESSION_ID"
echo "Results   : $RUN_ROOT"
echo "Manifest  : $RUN_MANIFEST"
echo "Metadata  : $METADATA"
echo "======================================================================"