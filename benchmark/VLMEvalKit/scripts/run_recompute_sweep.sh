#!/bin/bash
# M4: recompute-budget sweep on the single-image benchmarks.
#
#   bash scripts/run_recompute_sweep.sh
#   bash scripts/run_plot_recompute_sweeps.sh      # plot the curves
#
# By default this runs CONDUIT at r in {5%, 10%}.
#
# Note that kv_score is the *shared* scoring machinery, not CONDUIT itself:
#   kv_score + v_norm=false + c_j=0   ->  the ProphetKV baseline
#   kv_score + v_norm=true  + c_j=1   →  CONDUIT
# With a single image c_1 is always 1, so only v_norm=true is needed here
# (see the defaults below).
#
# To sweep a different baseline or budget:
#   RECOMPUTE_STRATEGIES="first cacheblend" RECOMPUTE_RATIOS="5 10 20" \
#     bash scripts/run_recompute_sweep.sh
#
# Output: ${OUTPUT_ROOT} (default ./outputs/recompute_sweep)
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VLMEVAL_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${VLMEVAL_ROOT}"

NUM_PROCS=${NUM_PROCS:-8}
MODELS=${MODELS:-"NanoVLLM-Qwen2.5-VL-3B-Instruct NanoVLLM-Qwen2.5-VL-7B-Instruct"}
DATAS=${DATAS:-"MMBench_DEV_EN_V11 OCRBench POPE"}
RECOMPUTE_STRATEGIES=${RECOMPUTE_STRATEGIES:-"kv_score"}
RECOMPUTE_RATIOS=${RECOMPUTE_RATIOS:-"5 10"}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs/recompute_sweep}
REUSE=${REUSE:-1}
VERBOSE=${VERBOSE:-1}
PRIORI_SEED=${PRIORI_SEED:-42}
# CONDUIT = the kv_score machinery + the ||V|| factor + the image
# coefficient c_j. With a single image c_1=1, so the bias stays at 0.0; the
# ||V|| factor must be enabled explicitly, otherwise the engine default of
# False applies and what actually runs is the ProphetKV baseline, not CONDUIT.
CONDUIT_SCORE_USE_V_NORM=${CONDUIT_SCORE_USE_V_NORM:-${CONDUIT_SCORE_USE_V_NORM:-"true"}}
CONDUIT_SCORE_IMAGE_BIAS_STRENGTH=${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH:-${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH:-"0.0"}}

# Strict recompute sweep: image_segment only, no priori / prompt-template /
# KV-patch confounders. Keep priori explicitly empty because config.py defaults
# NANOVLLM_IMAGE_PRIORI_MODE to random when it is not set.
export NANOVLLM_PREFILL_MODE="image_segment"
export CONDUIT_SCORE_ENABLED="false"
export CONDUIT_SCORE_USE_V_NORM="${CONDUIT_SCORE_USE_V_NORM}"
export CONDUIT_SCORE_IMAGE_BIAS_STRENGTH="${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}"
export NANOVLLM_IMAGE_PRIORI_MODE="none"
export NANOVLLM_IMAGE_PRIORI_SEED="${PRIORI_SEED}"

slugify() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:]]\+/-/g; s/^-//; s/-$//'
}

ratio_value() {
    local ratio="$1"
    ratio="${ratio%\%}"
    if [[ ! "${ratio}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Invalid recompute ratio: ${ratio}" >&2
        exit 1
    fi
    printf '%s' "${ratio}"
}

ratio_tag() {
    local ratio="$1"
    ratio=$(ratio_value "${ratio}")
    printf '%s' "${ratio}" | sed 's/\./p/g'
}

is_zero_like_float() {
    [[ "$1" =~ ^([0]+([.][0]*)?|[.][0]+)$ ]]
}

is_kv_score_runtime_recompute_strategy() {
    local spec="$1"
    local kind
    spec=$(printf '%s' "${spec}" | tr '[:upper:]' '[:lower:]')
    case "${spec}" in
        each=*|all=*|default=*)
            spec="${spec#*=}"
            ;;
    esac
    kind="${spec%%:*}"
    case "${kind}" in
        kv_score)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

run_eval() {
    local model_name="$1"
    local dataset_name="$2"
    local recompute_selector="$3"
    local ratio_input="$4"

    local model_tag
    local data_tag
    local strategy_tag
    local ratio
    local ratio_slug
    local recompute_strategy_value
    local work_tag
    local work_dir
    local cmd

    model_tag=$(slugify "${model_name}")
    data_tag=$(slugify "${dataset_name}")
    strategy_tag=$(slugify "${recompute_selector}")
    ratio=$(ratio_value "${ratio_input}")
    ratio_slug=$(ratio_tag "${ratio}")
    recompute_strategy_value="each=${recompute_selector}:${ratio}%"
    work_tag="image_segment_strategy_${strategy_tag}_ratio_${ratio_slug}"
    if ! is_zero_like_float "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}"; then
        work_tag="${work_tag}_imgbias_$(printf '%s' "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}" | sed 's/\./p/g')"
    fi
    work_dir="${OUTPUT_ROOT}/${model_tag}/${data_tag}/${work_tag}"

    export MODEL_NAME="${model_name}"
    export NANOVLLM_RECOMPUTE_STRATEGY="${recompute_strategy_value}"
    if is_kv_score_runtime_recompute_strategy "${recompute_strategy_value}"; then
        export CONDUIT_SCORE_ENABLED="true"
    else
        export CONDUIT_SCORE_ENABLED="false"
    fi
    export NANOVLLM_IMAGE_PRIORI_MODE="none"

    mkdir -p "${work_dir}"

    echo "============================================================"
    echo "[recompute] Image-segment recompute sweep"
    echo "  model              : ${model_name}"
    echo "  dataset            : ${dataset_name}"
    echo "  prefill_mode       : ${NANOVLLM_PREFILL_MODE}"
    echo "  recompute_selector : ${recompute_selector}"
    echo "  recompute_ratio    : ${ratio}%"
    echo "  recompute_strategy : ${NANOVLLM_RECOMPUTE_STRATEGY}"
    echo "  kv_score_enabled : ${CONDUIT_SCORE_ENABLED}"
    echo "  kv_score_imgbias : ${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}"
    echo "  priori_mode        : ${NANOVLLM_IMAGE_PRIORI_MODE}"
    echo "  work_dir           : ${work_dir}"
    echo "============================================================"

    cmd=(torchrun "--nproc-per-node=${NUM_PROCS}" run.py --model "${model_name}" --data "${dataset_name}" --work-dir "${work_dir}")
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
        for recompute_selector in ${RECOMPUTE_STRATEGIES}; do
            for recompute_ratio in ${RECOMPUTE_RATIOS}; do
                run_eval "${model_name}" "${dataset_name}" "${recompute_selector}" "${recompute_ratio}"
            done
        done
    done
done
