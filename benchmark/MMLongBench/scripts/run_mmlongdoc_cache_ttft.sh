#!/usr/bin/env bash
# M3: serving-latency comparison on MMLongBench-Doc (TTFT / prefill TFLOPs).
#
#   bash scripts/run_mmlongdoc_cache_ttft.sh
#   python scripts/export_cache_ttft_csv.py      # export the CSV
#
# The comparison arms are controlled by RECOMPUTE_STRATEGY. The default
# covers the paper's four baselines and two reference anchors:
#   none(cache reuse) / first(MPIC) / cacheblend / kvshare / prophetkv / conduit
#
# Latency numbers are very sensitive to the machine and to concurrent I/O.
# Always re-measure on an idle machine before reporting them.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT_DEFAULT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${ROOT_PATH:-${PROJECT_ROOT_DEFAULT}}}
ROOT_PATH=${ROOT_PATH:-${PROJECT_ROOT}}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-${PROJECT_ROOT}/benchmark/MMLongBench}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-"${PROJECT_ROOT}/models/internvl3-9b,${PROJECT_ROOT}/models/qwen2.5-vl-3b-instruct,${PROJECT_ROOT}/models/qwen2.5-vl-7b-instruct"}
MODEL_PATHS=${MODEL_PATHS:-${MODEL_PATH}}
MODEL_PATH_PRIMARY=${MODEL_PATHS%%,*}
CONFIG_PATH=${CONFIG_PATH:-configs/mmlongdoc_cache_ttft.yaml}
PREFILL_MODES=${PREFILL_MODES:-full,image_segment}
RECOMPUTE_STRATEGY=${RECOMPUTE_STRATEGY:-"none,cacheblend,first,kvshare,prophetkv,conduit"}
RECOMPUTE_STRATEGIES=${RECOMPUTE_STRATEGIES:-${RECOMPUTE_STRATEGY}}
RECOMPUTE_STRATEGY_PRIMARY=${RECOMPUTE_STRATEGIES%%,*}
RECOMPUTE_CASE_BUDGET=${RECOMPUTE_CASE_BUDGET:-10%}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
GPU_GROUP_SIZE=${GPU_GROUP_SIZE:-1}
GPU_GROUPS=${GPU_GROUPS:-}
CACHE_WARMUP_PASSES=${CACHE_WARMUP_PASSES:-1}
CACHE_WARMUP_GENERATION_MAX_LENGTH=${CACHE_WARMUP_GENERATION_MAX_LENGTH:-1}
CONDUIT_SCORE_USE_V_NORM=${CONDUIT_SCORE_USE_V_NORM:-}
CONDUIT_SCORE_IMAGE_BIAS_STRENGTH=${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH:-0.0}
OUTPUT_ROOT_MODEL_SEGMENT=$(basename "${MODEL_PATH_PRIMARY}")
if [[ "${MODEL_PATHS}" == *,* ]]; then
  OUTPUT_ROOT_MODEL_SEGMENT=multi_model
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-${BENCHMARK_ROOT}/output/cache_ttft/${OUTPUT_ROOT_MODEL_SEGMENT}/mmlongdoc_cache_ttft}
SKIP_EXISTING=${SKIP_EXISTING:-no}

SKIP_EXISTING_ARGS=()
case "$(printf '%s' "${SKIP_EXISTING}" | tr '[:upper:]' '[:lower:]')" in
  ""|auto)
    ;;
  1|true|yes|on)
    SKIP_EXISTING_ARGS+=(--skip-existing)
    ;;
  0|false|no|off)
    SKIP_EXISTING_ARGS+=(--no-skip-existing)
    ;;
  *)
    echo "Invalid SKIP_EXISTING=${SKIP_EXISTING}. Use auto/true/false." >&2
    exit 1
    ;;
esac

GPU_ARGS=()
if [[ -n "${GPU_GROUPS}" ]]; then
  GPU_ARGS+=(--gpu-groups "${GPU_GROUPS}")
elif [[ -n "${GPU_LIST}" ]]; then
  GPU_ARGS+=(--gpu-list "${GPU_LIST}" --gpu-group-size "${GPU_GROUP_SIZE}")
fi

CONDUIT_SCORE_ARGS=()
case "$(printf '%s' "${CONDUIT_SCORE_USE_V_NORM}" | tr '[:upper:]' '[:lower:]')" in
  "")
    ;;
  1|true|yes|on)
    CONDUIT_SCORE_ARGS+=(--kv_score_use_v_norm True)
    ;;
  0|false|no|off)
    ;;
  *)
    echo "Invalid CONDUIT_SCORE_USE_V_NORM=${CONDUIT_SCORE_USE_V_NORM}. Use true/false." >&2
    exit 1
    ;;
esac
if [[ -n "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}" ]]; then
  CONDUIT_SCORE_ARGS+=(
    --kv_score_image_bias_strength
    "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}"
  )
fi

cd "${BENCHMARK_ROOT}"
"${PYTHON_BIN}" scripts/mmlongdoc_cache_ttft.py \
  --model-name-or-paths "${MODEL_PATHS}" \
  --prefill-modes "${PREFILL_MODES}" \
  --recompute-strategies "${RECOMPUTE_STRATEGIES}" \
  --recompute-case-budget "${RECOMPUTE_CASE_BUDGET}" \
  --cache-warmup-passes "${CACHE_WARMUP_PASSES}" \
  --cache-warmup-generation-max-length "${CACHE_WARMUP_GENERATION_MAX_LENGTH}" \
  "${SKIP_EXISTING_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  "${CONDUIT_SCORE_ARGS[@]}" \
  --output-root "${OUTPUT_ROOT}" \
  --config "${CONFIG_PATH}" \
  --model_name_or_path "${MODEL_PATH_PRIMARY}" \
  --recompute_strategy "${RECOMPUTE_STRATEGY_PRIMARY}" \
  --test_file_root "${BENCHMARK_ROOT}/mmlb_data" \
  --image_file_root "${BENCHMARK_ROOT}/mmlb_image" \
  --use_nanovllm \
  --docqa_llm_judge False \
  "$@"
