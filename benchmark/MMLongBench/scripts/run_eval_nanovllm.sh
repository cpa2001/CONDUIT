#!/usr/bin/env bash
# Single-run entry point: run eval.py once with the given configuration,
# without any sweeping.
#
#   RECOMPUTE_STRATEGY="kv_score:5%" CONDUIT_SCORE_ENABLED=True \
#     bash scripts/run_eval_nanovllm.sh
#
# None of the paper's M2/M3/M4 runs go through this script
# (dispatch_nanovllm_sweep.py calls eval.py directly). It is kept so that a
# single configuration can be reproduced or debugged on its own -- the sweep
# scripts launch dozens of jobs at once, which is awkward when tracking down
# a problem.
#
# Main environment variables: MODEL_PATH / PREFILL_MODE /
# RECOMPUTE_STRATEGY / IMAGE_PRIORI_MODE / CONDUIT_SCORE_ENABLED. Their
# accepted values are visible in the defaults block below.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_PATH=${ROOT_PATH:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-${ROOT_PATH}/benchmark/MMLongBench}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-${ROOT_PATH}/models/qwen2.5-vl-3b-instruct}
MODEL_NAME=$(basename "${MODEL_PATH}")
cd "${BENCHMARK_ROOT}"

PREFILL_MODE=${PREFILL_MODE:-image_segment}
RECOMPUTE_STRATEGY=${RECOMPUTE_STRATEGY:-"kv_score:5%"}
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

if [[ -z "${CONDUIT_SCORE_ENABLED:-}" ]]; then
  if is_kv_score_runtime_recompute_strategy "${RECOMPUTE_STRATEGY}"; then
    CONDUIT_SCORE_ENABLED=True
  else
    CONDUIT_SCORE_ENABLED=False
  fi
fi
CONDUIT_SCORE_QUERY_FALLBACK=${CONDUIT_SCORE_QUERY_FALLBACK:-tail_text}
CONDUIT_SCORE_LAYER_IDX=${CONDUIT_SCORE_LAYER_IDX:-""}
CONDUIT_SCORE_LAYER_FROM_LAST=${CONDUIT_SCORE_LAYER_FROM_LAST:-""}
CONDUIT_SCORE_LAYER_SPLIT_PARTS=${CONDUIT_SCORE_LAYER_SPLIT_PARTS:-""}
CONDUIT_SCORE_LAYER_SPLIT_PART=${CONDUIT_SCORE_LAYER_SPLIT_PART:-""}
CONDUIT_SCORE_USE_V_NORM=${CONDUIT_SCORE_USE_V_NORM:-"True"}
CONDUIT_SCORE_IMAGE_BIAS_STRENGTH=${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH:-1.0}
IMAGE_PRIORI_MODE=${IMAGE_PRIORI_MODE:-chat_template}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-}
ENFORCE_EAGER=${ENFORCE_EAGER:-}
KVCACHE_BLOCK_SIZE=${KVCACHE_BLOCK_SIZE:-}
NUM_KVCACHE_BLOCKS=${NUM_KVCACHE_BLOCKS:-}
ENCODER_CACHE_RATIO=${ENCODER_CACHE_RATIO:-}
MAX_IMAGES=${MAX_IMAGES:-}
SAMPLER_BACKEND=${SAMPLER_BACKEND:-}
IMAGE_PRIORI_SEED=${IMAGE_PRIORI_SEED:-}

DO_SAMPLE=${DO_SAMPLE:-}
TEMPERATURE=${TEMPERATURE:-}
TOP_P=${TOP_P:-}

TORCH_PROFILE=${TORCH_PROFILE:-False}
TORCH_PROFILE_SKIP_FIRST=${TORCH_PROFILE_SKIP_FIRST:-5}
TORCH_PROFILE_WAIT=${TORCH_PROFILE_WAIT:-5}
TORCH_PROFILE_WARMUP=${TORCH_PROFILE_WARMUP:-5}
TORCH_PROFILE_ACTIVE=${TORCH_PROFILE_ACTIVE:-1}
TORCH_PROFILE_REPEAT=${TORCH_PROFILE_REPEAT:-1}
TORCH_PROFILE_RECORD_SHAPES=${TORCH_PROFILE_RECORD_SHAPES:-True}
TORCH_PROFILE_MEMORY=${TORCH_PROFILE_MEMORY:-True}
TORCH_PROFILE_WITH_STACK=${TORCH_PROFILE_WITH_STACK:-False}
CONFIG_NAME=${CONFIG_NAME:-mmlongdoc_8k}

RUN_NAME=${RECOMPUTE_STRATEGY}
if [[ -n "${DO_SAMPLE}" ]]; then
  RUN_NAME=${RUN_NAME}-samp${DO_SAMPLE}
fi
if [[ -n "${TEMPERATURE}" ]]; then
  RUN_NAME=${RUN_NAME}-t${TEMPERATURE//./p}
fi
if [[ -n "${TOP_P}" ]]; then
  RUN_NAME=${RUN_NAME}-p${TOP_P//./p}
fi
if [[ -n "${CONDUIT_SCORE_LAYER_IDX}" ]]; then
  RUN_NAME=${RUN_NAME}-scoreidx${CONDUIT_SCORE_LAYER_IDX}
fi
if [[ -n "${CONDUIT_SCORE_LAYER_FROM_LAST}" ]]; then
  RUN_NAME=${RUN_NAME}-scorelast${CONDUIT_SCORE_LAYER_FROM_LAST}
fi
if [[ -n "${CONDUIT_SCORE_LAYER_SPLIT_PARTS}" && -n "${CONDUIT_SCORE_LAYER_SPLIT_PART}" ]]; then
  RUN_NAME=${RUN_NAME}-scoresplit${CONDUIT_SCORE_LAYER_SPLIT_PART}of${CONDUIT_SCORE_LAYER_SPLIT_PARTS}
fi
case "$(printf '%s' "${CONDUIT_SCORE_USE_V_NORM}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    RUN_NAME=${RUN_NAME}-vnorm
    ;;
esac
if [[ ! "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}" =~ ^([0]+([.][0]*)?|[.][0]+)$ ]]; then
  IMAGE_SCORE_BIAS_TAG=${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH//./p}
  RUN_NAME=${RUN_NAME}-imgbias${IMAGE_SCORE_BIAS_TAG}
fi

CONDUIT_SCORE_RUNTIME_ARGS=()
if [[ -n "${CONDUIT_SCORE_LAYER_IDX}" ]]; then
  CONDUIT_SCORE_RUNTIME_ARGS+=(--kv_score_layer_idx "${CONDUIT_SCORE_LAYER_IDX}")
fi
if [[ -n "${CONDUIT_SCORE_LAYER_FROM_LAST}" ]]; then
  CONDUIT_SCORE_RUNTIME_ARGS+=(--kv_score_layer_from_last "${CONDUIT_SCORE_LAYER_FROM_LAST}")
fi
if [[ -n "${CONDUIT_SCORE_LAYER_SPLIT_PARTS}" ]]; then
  CONDUIT_SCORE_RUNTIME_ARGS+=(--kv_score_layer_split_parts "${CONDUIT_SCORE_LAYER_SPLIT_PARTS}")
fi
if [[ -n "${CONDUIT_SCORE_LAYER_SPLIT_PART}" ]]; then
  CONDUIT_SCORE_RUNTIME_ARGS+=(--kv_score_layer_split_part "${CONDUIT_SCORE_LAYER_SPLIT_PART}")
fi
case "$(printf '%s' "${CONDUIT_SCORE_USE_V_NORM}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    CONDUIT_SCORE_RUNTIME_ARGS+=(--kv_score_use_v_norm True)
    ;;
esac
if [[ -n "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}" ]]; then
  CONDUIT_SCORE_RUNTIME_ARGS+=(--kv_score_image_bias_strength "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}")
fi


GENERATION_ARGS=()
if [[ -n "${DO_SAMPLE}" ]]; then
  GENERATION_ARGS+=(--do_sample "${DO_SAMPLE}")
fi
if [[ -n "${TEMPERATURE}" ]]; then
  GENERATION_ARGS+=(--temperature "${TEMPERATURE}")
fi
if [[ -n "${TOP_P}" ]]; then
  GENERATION_ARGS+=(--top_p "${TOP_P}")
fi

NANOVLLM_CONFIG_ARGS=()
if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--max_num_seqs "${MAX_NUM_SEQS}")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}")
fi
if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--tensor_parallel_size "${TENSOR_PARALLEL_SIZE}")
fi
if [[ -n "${ENFORCE_EAGER}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--enforce_eager "${ENFORCE_EAGER}")
fi
if [[ -n "${KVCACHE_BLOCK_SIZE}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--kvcache_block_size "${KVCACHE_BLOCK_SIZE}")
fi
if [[ -n "${NUM_KVCACHE_BLOCKS}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--num_kvcache_blocks "${NUM_KVCACHE_BLOCKS}")
fi
if [[ -n "${ENCODER_CACHE_RATIO}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--encoder_cache_ratio "${ENCODER_CACHE_RATIO}")
fi
if [[ -n "${MAX_IMAGES}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--max_images "${MAX_IMAGES}")
fi
if [[ -n "${SAMPLER_BACKEND}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--sampler_backend "${SAMPLER_BACKEND}")
fi
if [[ -n "${IMAGE_PRIORI_SEED}" ]]; then
  NANOVLLM_CONFIG_ARGS+=(--image_priori_seed "${IMAGE_PRIORI_SEED}")
fi

OUTPUT_DIR=${BENCHMARK_ROOT}/output/profile/${MODEL_NAME}/${CONFIG_NAME}/${PREFILL_MODE}/${RUN_NAME}
TORCH_PROFILE_DIR=${TORCH_PROFILE_DIR:-${OUTPUT_DIR}/torch_profiler}
rm -rf "${OUTPUT_DIR}"

"${PYTHON_BIN}" eval.py \
  --config configs/${CONFIG_NAME}.yaml \
  --model_name_or_path ${MODEL_PATH} \
  --test_file_root ${BENCHMARK_ROOT}/mmlb_data \
  --image_file_root ${BENCHMARK_ROOT}/mmlb_image \
  --output_dir ${OUTPUT_DIR} \
  --use_nanovllm \
  --prefill_mode ${PREFILL_MODE} \
  --image_priori_mode ${IMAGE_PRIORI_MODE} \
  "${NANOVLLM_CONFIG_ARGS[@]}" \
  --recompute_strategy ${RECOMPUTE_STRATEGY} \
  --kv_score_enabled ${CONDUIT_SCORE_ENABLED} \
  --kv_score_query_fallback ${CONDUIT_SCORE_QUERY_FALLBACK} \
  "${CONDUIT_SCORE_RUNTIME_ARGS[@]}" \
