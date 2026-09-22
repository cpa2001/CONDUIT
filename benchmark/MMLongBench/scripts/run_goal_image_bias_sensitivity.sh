#!/usr/bin/env bash
# M4: sensitivity analysis for the image coefficient c_j (one arm of the
# paper's ablation table).
#
#   bash scripts/run_goal_image_bias_sensitivity.sh
#   bash scripts/export_goal_image_bias_sensitivity_csv.sh   # export the CSV
#
# Sweeps CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES, which corresponds to the
# paper's row "dropping the image coefficient sets c_j = 1".
#
# DRY_RUN=1 only prints the dispatch command that would run, without
# launching any job.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/lib_common.sh"
BENCHMARK_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd -- "${BENCHMARK_ROOT}/../.." && pwd)
DISPATCH_SCRIPT="${SCRIPT_DIR}/dispatch_nanovllm_sweep.py"

model_label() {
  local value="$1"
  value=${value//\\/\/}
  value=${value%/}
  sanitize_path_component "${value##*/}"
}

append_optional_arg() {
  local env_name="$1"
  local flag="$2"
  local value="${!env_name:-}"

  if [[ -n "${value}" ]]; then
    dispatch_cmd+=("${flag}" "${value}")
  fi
}

append_bool_flag() {
  local value="$1"
  local flag="$2"

  if is_enabled "${value}"; then
    dispatch_cmd+=("${flag}")
  fi
}

write_launch_manifest() {
  local stage_label="$1"
  local model_path="$2"
  local manifest_path="$3"
  shift 3
  local command=("$@")
  local git_commit=""
  local git_status_short=""

  git_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || true)
  git_status_short=$(git -C "${PROJECT_ROOT}" status --short 2>/dev/null || true)

  mkdir -p "$(dirname -- "${manifest_path}")"
  {
    printf 'goal_file=%s\n' "${PROJECT_ROOT}/GOAL.md"
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'stage=%s\n' "${stage_label}"
    printf 'model_path=%s\n' "${model_path}"
    printf 'git_commit=%s\n' "${git_commit}"
    printf 'root_path=%s\n' "${ROOT_PATH}"
    printf 'benchmark_root=%s\n' "${BENCHMARK_ROOT}"
    printf 'config_files=%s\n' "${CONFIG_FILES}"
    printf 'config_sweep=%s\n' "${CONFIG_SWEEP}"
    printf 'only_benchmarks=%s\n' "${ONLY_BENCHMARKS}"
    printf 'skip_benchmarks=%s\n' "${SKIP_BENCHMARKS}"
    printf 'tag=%s\n' "${TAG}"
    printf 'bias_values=%s\n' "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES}"
    printf 'seeds=%s\n' "${SEEDS}"
    printf 'prefill_mode=%s\n' "${PREFILL_MODE}"
    printf 'image_priori_mode=%s\n' "${IMAGE_PRIORI_MODE}"
    printf 'recompute_strategies=%s\n' "${RECOMPUTE_STRATEGIES}"
    printf 'ratio_values=%s\n' "${RATIO_VALUES}"
    printf 'do_sample=%s\n' "${DO_SAMPLE}"
    printf 'temperature=%s\n' "${TEMPERATURE}"
    printf 'top_p=%s\n' "${TOP_P}"
    printf 'kv_score_use_v_norm_values=%s\n' "${CONDUIT_SCORE_USE_V_NORM_VALUES}"
    printf 'gpu_list=%s\n' "${GPU_LIST}"
    printf 'gpu_group_size=%s\n' "${GPU_GROUP_SIZE}"
    printf 'gpu_groups=%s\n' "${GPU_GROUPS}"
    printf 'rerun=%s\n' "${RERUN}"
    printf 'preview_only=%s\n' "${PREVIEW_ONLY}"
    printf 'continue_on_error=%s\n' "${CONTINUE_ON_ERROR}"
    printf 'dispatch_command='
    print_command "${command[@]}"
    printf 'git_status_short<<EOF\n%s\nEOF\n' "${git_status_short}"
  } > "${manifest_path}"
}

build_dispatch_command() {
  local stage_label="$1"
  local model_path="$2"
  local config_file_args=()
  local env_commit=""

  read -r -a config_file_args <<< "${CONFIG_FILES}"
  env_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || true)

  dispatch_cmd=(
    "${PYTHON_BIN}"
    "${DISPATCH_SCRIPT}"
    --config-files "${config_file_args[@]}"
    --model_name_or_path "${model_path}"
    --test_file_root "${BENCHMARK_ROOT}/mmlb_data"
    --image_file_root "${BENCHMARK_ROOT}/mmlb_image"
    --prefill_mode "${PREFILL_MODE}"
    --image-priori-mode "${IMAGE_PRIORI_MODE}"
    --recompute-strategies "${RECOMPUTE_STRATEGIES}"
    --recompute-template "${RECOMPUTE_TEMPLATE}"
    --ratio-values "${RATIO_VALUES}"
    --seed-values "${SEEDS}"
    --extra-eval-args "${EXTRA_EVAL_ARGS}"
    --python "${PYTHON_BIN}"
    --tag "${TAG}"
    --env "CONDUIT_GOAL_FILE=GOAL.md"
    --env "CONDUIT_GOAL_RUN_ID=${RUN_ID}"
    --env "CONDUIT_GOAL_STAGE=${stage_label}"
    --env "CONDUIT_GOAL_COMMIT=${env_commit}"
  )

  if [[ "${CONFIG_SWEEP}" != "none" ]]; then
    dispatch_cmd+=(--config-sweep "${CONFIG_SWEEP}")
  fi

  append_optional_arg ONLY_BENCHMARKS --only-benchmarks
  append_optional_arg SKIP_BENCHMARKS --skip-benchmarks

  if [[ -n "${OUTPUT_ROOT}" ]]; then
    dispatch_cmd+=(--output-root "${OUTPUT_ROOT}")
  fi

  if [[ -n "${GPU_GROUPS}" ]]; then
    dispatch_cmd+=(--gpu-groups "${GPU_GROUPS}")
  else
    dispatch_cmd+=(--gpu-list "${GPU_LIST}" --gpu-group-size "${GPU_GROUP_SIZE}")
  fi

  append_optional_arg DO_SAMPLE --do_sample
  append_optional_arg TEMPERATURE --temperature
  append_optional_arg TOP_P --top_p
  append_optional_arg MAX_NUM_BATCHED_TOKENS --max-num-batched-tokens
  append_optional_arg MAX_NUM_SEQS --max-num-seqs
  append_optional_arg GPU_MEMORY_UTILIZATION --gpu-memory-utilization
  append_optional_arg TENSOR_PARALLEL_SIZE --tensor-parallel-size
  append_optional_arg ENFORCE_EAGER --enforce-eager
  append_optional_arg KVCACHE_BLOCK_SIZE --kvcache-block-size
  append_optional_arg NUM_KVCACHE_BLOCKS --num-kvcache-blocks
  append_optional_arg ENCODER_CACHE_RATIO --encoder-cache-ratio
  append_optional_arg MAX_IMAGES --max-images
  append_optional_arg SAMPLER_BACKEND --sampler-backend
  append_optional_arg IMAGE_PRIORI_SEED --image-priori-seed
  append_optional_arg CONDUIT_SCORE_LAYER_IDX --kv-score-layer-idx
  append_optional_arg CONDUIT_SCORE_LAYER_FROM_LAST --kv-score-layer-from-last
  append_optional_arg CONDUIT_SCORE_LAYER_SPLIT_PARTS --kv-score-layer-split-parts
  append_optional_arg CONDUIT_SCORE_LAYER_SPLIT_PART --kv-score-layer-split-part

  if [[ -n "${CONDUIT_SCORE_USE_V_NORM_VALUES}" ]]; then
    dispatch_cmd+=(--kv-score-use-v-norm-values "${CONDUIT_SCORE_USE_V_NORM_VALUES}")
  else
    append_optional_arg CONDUIT_SCORE_USE_V_NORM --kv-score-use-v-norm
  fi

  dispatch_cmd+=(
    --kv-score-image-bias-strength-values
    "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES}"
  )


  append_bool_flag "${RERUN}" --rerun
  append_bool_flag "${PREVIEW_ONLY}" --preview-only
  append_bool_flag "${CONTINUE_ON_ERROR}" --continue-on-error
}

run_model_stage() {
  local stage_label="$1"
  local model_path="$2"
  local label
  local manifest_path

  label=$(model_label "${model_path}")
  manifest_path="${MANIFEST_ROOT}/${RUN_ID}_${stage_label}_${label}.txt"
  build_dispatch_command "${stage_label}" "${model_path}"
  write_launch_manifest "${stage_label}" "${model_path}" "${manifest_path}" "${dispatch_cmd[@]}"

  echo "==> ${stage_label}: ${label}"
  echo "    manifest: ${manifest_path}"
  echo "    bias values: ${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES}"
  echo "    seeds: ${SEEDS}"
  if [[ -n "${ONLY_BENCHMARKS}" ]]; then
    echo "    only benchmarks: ${ONLY_BENCHMARKS}"
  fi
  if [[ -n "${SKIP_BENCHMARKS}" ]]; then
    echo "    skip benchmarks: ${SKIP_BENCHMARKS}"
  fi
  echo "    command:"
  print_command "${dispatch_cmd[@]}"
  "${dispatch_cmd[@]}"
}

set_default ROOT_PATH "${PROJECT_ROOT}"
set_default PYTHON_BIN "python"
set_default CONFIG_FILES "configs/docqa_all_8k.yaml configs/vrag_all_8k.yaml"
set_default CONFIG_SWEEP "entries"
set_default ONLY_BENCHMARKS "longdocurl,infoseek"
set_default SKIP_BENCHMARKS ""
set_default TAG "${ALL_CONFIG_TAG:-goal_image_bias_sensitivity_8k}"
set_default OUTPUT_ROOT ""
set_default PREFILL_MODE "image_segment"
set_default IMAGE_PRIORI_MODE "chat_template"
set_default RECOMPUTE_STRATEGIES "kv_score"
set_default RECOMPUTE_TEMPLATE "none"
set_default RATIO_VALUES "10"
set_default EXTRA_EVAL_ARGS "--docqa_llm_judge False"
set_default SEEDS "10,11,12,13,14,15"
set_default DO_SAMPLE "True"
set_default TEMPERATURE "0.2"
set_default TOP_P "0.9"
set_default CONDUIT_SCORE_USE_V_NORM_VALUES "True"
set_default CONDUIT_SCORE_USE_V_NORM ""
set_default CONDUIT_SCORE_LAYER_IDX ""
set_default CONDUIT_SCORE_LAYER_FROM_LAST ""
set_default CONDUIT_SCORE_LAYER_SPLIT_PARTS ""
set_default CONDUIT_SCORE_LAYER_SPLIT_PART ""
set_default GPU_LIST "0,1,2,3"
set_default GPU_GROUP_SIZE "1"
set_default GPU_GROUPS ""
set_default RERUN "0"
set_default DRY_RUN "0"
set_default CONTINUE_ON_ERROR "0"
set_default RUN_7B "0"
set_default RUN_9B "0"
set_default RUN_ALL_MODELS "0"
set_default MODEL_3B_PATH "${ROOT_PATH}/models/qwen2.5-vl-3b-instruct"
set_default MODEL_7B_PATH "${ROOT_PATH}/models/qwen2.5-vl-7b-instruct"
set_default MODEL_9B_PATH "${ROOT_PATH}/models/internvl3-9b"

if [[ -z "${PREVIEW_ONLY+x}" ]]; then
  PREVIEW_ONLY="${DRY_RUN}"
fi

if [[ -z "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES+x}" ]]; then
  if [[ -n "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH:-}" ]]; then
    CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES="${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}"
  else
    CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES="0.0,0.50"
  fi
fi

if is_enabled "${RUN_ALL_MODELS}"; then
  RUN_7B="1"
  RUN_9B="1"
fi

RUN_ID=${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")}
MANIFEST_BASE="${OUTPUT_ROOT:-${BENCHMARK_ROOT}/output}"
MANIFEST_ROOT=${MANIFEST_ROOT:-"${MANIFEST_BASE}/_summaries/$(sanitize_path_component "${TAG}")/launch_manifests"}
dispatch_cmd=()

if [[ ! -f "${DISPATCH_SCRIPT}" ]]; then
  echo "Missing dispatch script: ${DISPATCH_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${SEEDS}" ]]; then
  echo "SEEDS must be a comma-separated list, for example 10,11,12,13." >&2
  exit 1
fi

if [[ -z "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES}" ]]; then
  echo "CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES must not be empty." >&2
  exit 1
fi

run_model_stage "7b" "${MODEL_3B_PATH}"

