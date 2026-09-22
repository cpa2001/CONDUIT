#!/bin/bash
# M1: the paper's single-image main table.
#
#   bash scripts/run_single_image_table5.sh
#
# What the three settings mean (defined in configure_setting below):
#   full    full prefill -- the paper's "full prefill" anchor
#   reuse0  image KV reuse with no token recomputed -- the "cache reuse" anchor
#   reuse5  reuse plus selective recomputation at r=5% -- CONDUIT
#
# By default this runs three backbones x four benchmarks x three settings.
# To narrow it down:
#   MODELS="NanoVLLM-Qwen2.5-VL-3B-Instruct" DATAS="POPE" \
#     bash scripts/run_single_image_table5.sh
#
# Output: outputs/paper_table5_single_image/<model>/<data>/<setting>/
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VLMEVAL_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${VLMEVAL_ROOT}"

NUM_PROCS=${NUM_PROCS:-1}
MASTER_PORT=${MASTER_PORT:-29500}
MODELS=${MODELS:-"NanoVLLM-Qwen2.5-VL-3B-Instruct NanoVLLM-Qwen2.5-VL-7B-Instruct NanoVLLM-InternVL3-9B"}
DATAS=${DATAS:-"MMBench_DEV_EN_V11 OCRBench POPE MMStar"}
SETTINGS=${SETTINGS:-"full reuse0 reuse5"}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs/paper_table5_single_image}
REUSE=${REUSE:-1}
VERBOSE=${VERBOSE:-1}

slugify() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:]]\+/-/g; s/^-//; s/-$//'
}

score_layers_for_model() {
    case "$1" in
        NanoVLLM-Qwen2.5-VL-3B-Instruct)
            printf '14,16,18,20,22'
            ;;
        NanoVLLM-Qwen2.5-VL-7B-Instruct)
            printf '12,14,16,18,20'
            ;;
        NanoVLLM-InternVL3-9B)
            printf '18,21,24,27,30'
            ;;
        *)
            printf ''
            ;;
    esac
}

configure_setting() {
    local setting="$1"
    local model_name="$2"
    local score_layers
    score_layers=$(score_layers_for_model "${model_name}")

    # With a single image the coefficient c_j is always 1 -- the paper's
    # single-image table states: "Since c_1=1, CONDUIT reduces to intra-image
    # throughput selection". Turning it off here is therefore correct; do not
    # change it to 1.0.
    export CONDUIT_SCORE_IMAGE_BIAS_STRENGTH="0.0"
    unset CONDUIT_SCORE_LAYER_IDX
    unset CONDUIT_SCORE_LAYER_FROM_LAST
    unset CONDUIT_SCORE_LAYER_SPLIT_PARTS
    unset CONDUIT_SCORE_LAYER_SPLIT_PART

    case "${setting}" in
        full)
            export NANOVLLM_PREFILL_MODE="full"
            export NANOVLLM_RECOMPUTE_STRATEGY="none"
            export CONDUIT_SCORE_ENABLED="false"
            export CONDUIT_SCORE_USE_V_NORM="false"
            export CONDUIT_SCORE_LAYER_INDICES=""
            export NANOVLLM_IMAGE_PRIORI_MODE="none"
            ;;
        reuse0)
            export NANOVLLM_PREFILL_MODE="image_segment"
            export NANOVLLM_RECOMPUTE_STRATEGY="none"
            export CONDUIT_SCORE_ENABLED="false"
            export CONDUIT_SCORE_USE_V_NORM="false"
            export CONDUIT_SCORE_LAYER_INDICES=""
            export NANOVLLM_IMAGE_PRIORI_MODE="chat_template"
            ;;
        reuse5)
            export NANOVLLM_PREFILL_MODE="image_segment"
            export NANOVLLM_RECOMPUTE_STRATEGY="each=kv_score:5%"
            export CONDUIT_SCORE_ENABLED="true"
            export CONDUIT_SCORE_USE_V_NORM="true"
            export CONDUIT_SCORE_LAYER_INDICES="${score_layers}"
            export NANOVLLM_IMAGE_PRIORI_MODE="chat_template"
            ;;
        *)
            echo "Unknown setting: ${setting}" >&2
            exit 1
            ;;
    esac
}

run_eval() {
    local model_name="$1"
    local dataset_name="$2"
    local setting="$3"
    local model_tag
    local data_tag
    local work_dir
    local cmd

    configure_setting "${setting}" "${model_name}"
    model_tag=$(slugify "${model_name}")
    data_tag=$(slugify "${dataset_name}")
    work_dir="${OUTPUT_ROOT}/${model_tag}/${data_tag}/${setting}"
    mkdir -p "${work_dir}"

    echo "============================================================"
    echo "[single-image] cache reuse"
    echo "  model        : ${model_name}"
    echo "  dataset      : ${dataset_name}"
    echo "  setting      : ${setting}"
    echo "  prefill_mode : ${NANOVLLM_PREFILL_MODE}"
    echo "  recompute    : ${NANOVLLM_RECOMPUTE_STRATEGY}"
    echo "  v_norm       : ${CONDUIT_SCORE_USE_V_NORM}"
    echo "  layers       : ${CONDUIT_SCORE_LAYER_INDICES}"
    echo "  visible_gpu  : ${CUDA_VISIBLE_DEVICES:-all}"
    echo "  work_dir     : ${work_dir}"
    echo "============================================================"

    cmd=(torchrun "--nproc-per-node=${NUM_PROCS}" "--master-port=${MASTER_PORT}" run.py --model "${model_name}" --data "${dataset_name}" --work-dir "${work_dir}")
    if [ "${VERBOSE}" = "1" ]; then
        cmd+=(--verbose)
    fi
    if [ "${REUSE}" = "1" ]; then
        cmd+=(--reuse)
    fi
    "${cmd[@]}"
}

for model_name in ${MODELS}; do
    for dataset_name in ${DATAS}; do
        for setting in ${SETTINGS}; do
            run_eval "${model_name}" "${dataset_name}" "${setting}"
        done
    done
done
