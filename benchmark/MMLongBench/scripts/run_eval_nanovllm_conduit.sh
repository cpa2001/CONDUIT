#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/lib_common.sh"
PROJECT_ROOT_DEFAULT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
SWEEP_SCRIPT="${SCRIPT_DIR}/run_eval_nanovllm_sweep_multi_gpu.sh"
REPORT_SCRIPT="${SCRIPT_DIR}/find_best_seed_metric.py"

PRESET=${PRESET:-main_table}
DEFAULT_RUN_MODE="eval"
DEFAULT_SEED=""
DEFAULT_SEEDS=""

usage() {
  cat <<'USAGE'
Usage: run_eval_nanovllm_conduit.sh [--preset PRESET]

PRESET values:
  main_table    The paper's long-document main table (default). Three
                backbones (Qwen2.5-VL-3B/7B, InternVL3-9B) x four benchmarks
                (longdocurl, mmlongdoc, slidevqa, infoseek) x r in {5%,10%}.
                Requires 8 GPUs.
  smoke         A fast single-model, single-benchmark configuration for
                checking the environment. **Not used for any paper metric.**
  seed_baseline Multi-seed sweep of the cache-reuse anchor, i.e. the control
                group: no ||V|| factor and no image coefficient (seeds 10-13)
  seed_conduit  Multi-seed sweep of CONDUIT: both factors on (seeds 10-13)

PRESET may also be given as an environment variable; --preset wins.
Everything else is overridden through environment variables -- see
apply_preset_defaults() in this file.
DRY_RUN=1 only prints the resolved configuration and the command that would
run, without launching any job.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --preset)
      [ $# -ge 2 ] || { echo "--preset requires a value" >&2; exit 2; }
      PRESET="$2"; shift 2 ;;
    --preset=*)
      PRESET="${1#--preset=}"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1 (use --help)" >&2; exit 2 ;;
  esac
done

apply_preset_defaults() {
  case "${PRESET}" in
    smoke)
      # A fast single-model, single-benchmark, single-budget configuration
      # for checking that the environment is set up. **Not used for any paper
      # metric** -- use --preset main_table for the main table.
      set_default ROOT_PATH "${PROJECT_ROOT_DEFAULT}"
      set_default MODEL_PATH "${ROOT_PATH}/models/internvl3-9b"
      set_default MODEL_PATHS "${MODEL_PATH}"
      set_default GPU_LIST "0"
      set_default RERUN "1"
      set_default ALL_CONFIG_TAG "smoke_8k"
      set_default RATIO_VALUES "5"
      set_default DO_SAMPLE "True"
      set_default TEMPERATURE "0.2"
      set_default TOP_P "0.9"
      set_default CONDUIT_SCORE_USE_V_NORM_VALUES "False"
      set_default CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES "0.0"
      set_default CONFIG_FILES "configs/mmlongdoc_8k.yaml"
      set_default EXPERIMENT_CASES "vanilla"
      DEFAULT_SEED="5"
      ;;
    main_table)
      # The paper's main table: three backbones x four benchmarks x
      # r in {5%, 10%}. Both CONDUIT factors are enabled below.
      set_default ROOT_PATH "${PROJECT_ROOT_DEFAULT}"
      set_default MODEL_PATH "${ROOT_PATH}/models/qwen2.5-vl-7b-instruct ${ROOT_PATH}/models/internvl3-9b ${ROOT_PATH}/models/qwen2.5-vl-3b-instruct"
      set_default MODEL_PATHS "${MODEL_PATH}"
      set_default GPU_LIST "0,1,2,3,4,5,6,7"
      set_default RERUN "0"
      set_default ALL_CONFIG_TAG "main_table_8k"
      set_default RATIO_VALUES "5,10"
      set_default CONDUIT_SCORE_USE_V_NORM_VALUES "True"
      set_default CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES "1.0"
      set_default CONFIG_FILES "configs/docqa_all_8k.yaml configs/infoseek_8k.yaml"
      set_default EXPERIMENT_CASES "vanilla"
      ;;
    seed_baseline)
      set_default ROOT_PATH "${PROJECT_ROOT_DEFAULT}"
      set_default BENCHMARK_ROOT "${ROOT_PATH}/benchmark/MMLongBench"
      set_default MODEL_PATH "${ROOT_PATH}/models/internvl3-9b"
      set_default MODEL_PATHS "${MODEL_PATH}"
      set_default GPU_LIST "4,5,6,7"
      set_default RERUN "0"
      set_default ALL_CONFIG_TAG "seed_baseline_8k"
      set_default RATIO_VALUES "10"
      set_default DO_SAMPLE "True"
      set_default TEMPERATURE "0.2"
      set_default TOP_P "0.9"
      set_default CONDUIT_SCORE_USE_V_NORM_VALUES "False"
      set_default CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES "0.0"
      set_default CONFIG_FILES "configs/infoseek_8k.yaml"
      set_default EXPERIMENT_CASES "vanilla"
      set_default ALLOW_MISSING_SUMMARIES "0"
      DEFAULT_RUN_MODE="seed_sweep"
      DEFAULT_SEEDS="10,11,12,13"
      ;;
    seed_conduit)
      set_default ROOT_PATH "${PROJECT_ROOT_DEFAULT}"
      set_default BENCHMARK_ROOT "${ROOT_PATH}/benchmark/MMLongBench"
      set_default MODEL_PATH "${ROOT_PATH}/models/internvl3-9b"
      set_default MODEL_PATHS "${MODEL_PATH}"
      set_default GPU_LIST "0,1,2,3"
      set_default RERUN "0"
      set_default ALL_CONFIG_TAG "seed_conduit_8k"
      set_default RATIO_VALUES "10"
      set_default DO_SAMPLE "True"
      set_default TEMPERATURE "0.2"
      set_default TOP_P "0.9"
      set_default CONDUIT_SCORE_USE_V_NORM_VALUES "True"
      set_default CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES "1.0"
      set_default CONFIG_FILES "configs/infoseek_8k.yaml"
      set_default EXPERIMENT_CASES "vanilla"
      set_default ALLOW_MISSING_SUMMARIES "0"
      DEFAULT_RUN_MODE="seed_sweep"
      DEFAULT_SEEDS="10,11,12,13"
      ;;
    *)
      echo "Unknown PRESET: ${PRESET}" >&2
      echo "Known presets: main_table, smoke, seed_baseline, seed_conduit" >&2
      exit 1
      ;;
  esac
}

apply_common_defaults() {
  set_default PYTHON_BIN "python"
  set_default BENCHMARK_ROOT "${ROOT_PATH}/benchmark/MMLongBench"
  set_default CONFIG_SWEEP "entries"
  set_default GPU_GROUP_SIZE "1"
  set_default GPU_GROUPS ""
  set_default DRY_RUN "0"
  set_default PREFILL_MODE "image_segment"
  set_default IMAGE_PRIORI_MODE "chat_template"
  set_default RECOMPUTE_STRATEGIES "kv_score"
  set_default RECOMPUTE_TEMPLATE "none"
  set_default EXTRA_EVAL_ARGS "--docqa_llm_judge False"
  set_default RUN_MODE "${DEFAULT_RUN_MODE}"
  set_default SEEDS "${DEFAULT_SEEDS}"
  if [[ -n "${SEEDS}" ]]; then
    set_default SEED ""
  else
    set_default SEED "${DEFAULT_SEED}"
  fi
  set_default ALLOW_MISSING_SUMMARIES "0"
}

validate_files() {
  if [[ ! -f "${SWEEP_SCRIPT}" ]]; then
    echo "Missing sweep script: ${SWEEP_SCRIPT}" >&2
    exit 1
  fi
  if [[ "${RUN_MODE}" == "seed_sweep" && ! -f "${REPORT_SCRIPT}" ]]; then
    echo "Missing seed metric report script: ${REPORT_SCRIPT}" >&2
    exit 1
  fi
}

validate_seed_settings() {
  local seed
  local seed_list=()

  if [[ -n "${SEED}" && -n "${SEEDS}" ]]; then
    echo "SEED and SEEDS are mutually exclusive." >&2
    exit 1
  fi

  if [[ "${RUN_MODE}" != "seed_sweep" ]]; then
    return
  fi
  if [[ -z "${SEEDS}" ]]; then
    echo "RUN_MODE=seed_sweep requires SEEDS." >&2
    exit 1
  fi

  read -r -a seed_list <<< "$(printf '%s' "${SEEDS}" | tr ',' ' ')"
  if [[ ${#seed_list[@]} -eq 0 ]]; then
    echo "SEEDS must contain at least one integer." >&2
    exit 1
  fi
  for seed in "${seed_list[@]}"; do
    if [[ -z "${seed}" ]]; then
      continue
    fi
    if [[ ! "${seed}" =~ ^-?[0-9]+$ ]]; then
      echo "Invalid seed value: ${seed}" >&2
      exit 1
    fi
  done
}

resolve_model_path() {
  local raw_path="$1"

  if [[ -z "${raw_path}" ]]; then
    echo ""
    return
  fi
  if [[ "${raw_path}" == /* ]]; then
    echo "${raw_path}"
    return
  fi
  if [[ -e "${ROOT_PATH}/models/${raw_path}" ]]; then
    echo "${ROOT_PATH}/models/${raw_path}"
    return
  fi

  echo "${raw_path}"
}

case_settings() {
  local case_name="$1"

  CASE_TAG=""
  SCORE_LAYER_FROM_LAST=""
  SCORE_LAYER_SPLIT_PARTS=""
  SCORE_LAYER_SPLIT_PART=""

  case "${case_name}" in
    vanilla)
      CASE_TAG="${ALL_CONFIG_TAG}"
      ;;
    split4)
      CASE_TAG="${ALL_CONFIG_TAG}_scoresplit4of4"
      SCORE_LAYER_SPLIT_PARTS="4"
      SCORE_LAYER_SPLIT_PART="4"
      ;;
    last1)
      CASE_TAG="${ALL_CONFIG_TAG}_scorelast1"
      SCORE_LAYER_FROM_LAST="1"
      ;;
    *)
      echo "Unknown EXPERIMENT_CASES entry: ${case_name}" >&2
      echo "Known cases: vanilla, split4, last1" >&2
      exit 1
      ;;
  esac
}

run_case() {
  local model_path="$1"
  local case_name="$2"
  local model_label="${model_path##*/}"

  case_settings "${case_name}"

  echo "==> Running preset ${PRESET}, mode ${RUN_MODE}, model ${model_label}, case ${case_name}, tag ${CASE_TAG}"

  if is_enabled "${DRY_RUN}"; then
    echo "    ROOT_PATH=${ROOT_PATH}"
    echo "    BENCHMARK_ROOT=${BENCHMARK_ROOT}"
    echo "    MODEL_PATH=${model_path}"
    echo "    CONFIG_FILES=${CONFIG_FILES}"
    echo "    GPU_LIST=${GPU_LIST}"
    echo "    GPU_GROUP_SIZE=${GPU_GROUP_SIZE}"
    echo "    GPU_GROUPS=${GPU_GROUPS}"
    echo "    RERUN=${RERUN}"
    echo "    RATIO_VALUES=${RATIO_VALUES}"
    echo "    SEED=${SEED}"
    echo "    SEEDS=${SEEDS}"
    echo "    DO_SAMPLE=${DO_SAMPLE:-}"
    echo "    TEMPERATURE=${TEMPERATURE:-}"
    echo "    TOP_P=${TOP_P:-}"
    echo "    CONDUIT_SCORE_USE_V_NORM_VALUES=${CONDUIT_SCORE_USE_V_NORM_VALUES:-}"
    echo "    CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES=${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES:-}"
    echo "    CONDUIT_SCORE_LAYER_FROM_LAST=${SCORE_LAYER_FROM_LAST}"
    echo "    CONDUIT_SCORE_LAYER_SPLIT_PARTS=${SCORE_LAYER_SPLIT_PARTS}"
    echo "    CONDUIT_SCORE_LAYER_SPLIT_PART=${SCORE_LAYER_SPLIT_PART}"
    return
  fi

  (
    export ROOT_PATH="${ROOT_PATH}"
    export BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
    export PYTHON_BIN="${PYTHON_BIN}"
    export MODEL_PATH="${model_path}"
    export MODEL_PATHS="${MODEL_PATHS}"
    export RERUN="${RERUN}"
    export CONFIG_SWEEP="${CONFIG_SWEEP}"
    export GPU_LIST="${GPU_LIST}"
    export GPU_GROUP_SIZE="${GPU_GROUP_SIZE}"
    export GPU_GROUPS="${GPU_GROUPS}"
    export ALL_CONFIG_TAG="${CASE_TAG}"
    export PREFILL_MODE="${PREFILL_MODE}"
    export IMAGE_PRIORI_MODE="${IMAGE_PRIORI_MODE}"
    export RECOMPUTE_STRATEGIES="${RECOMPUTE_STRATEGIES}"
    export RECOMPUTE_TEMPLATE="${RECOMPUTE_TEMPLATE}"
    export RATIO_VALUES="${RATIO_VALUES}"
    export EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS}"
    export DO_SAMPLE="${DO_SAMPLE:-}"
    export TEMPERATURE="${TEMPERATURE:-}"
    export TOP_P="${TOP_P:-}"
    export SEED="${SEED}"
    export SEEDS="${SEEDS}"
    export CONDUIT_SCORE_LAYER_IDX=""
    export CONDUIT_SCORE_LAYER_FROM_LAST="${SCORE_LAYER_FROM_LAST}"
    export CONDUIT_SCORE_LAYER_SPLIT_PARTS="${SCORE_LAYER_SPLIT_PARTS}"
    export CONDUIT_SCORE_LAYER_SPLIT_PART="${SCORE_LAYER_SPLIT_PART}"
    export CONDUIT_SCORE_USE_V_NORM_VALUES="${CONDUIT_SCORE_USE_V_NORM_VALUES:-}"
    export CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES="${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES:-}"
    export CONFIG_FILES="${CONFIG_FILES}"
    bash "${SWEEP_SCRIPT}"
  )
}

run_sweeps() {
  local model_path
  local resolved_model_path
  local case_name
  local model_path_list=()
  local case_list=()

  read -r -a model_path_list <<< "$(printf '%s' "${MODEL_PATHS}" | tr ',' ' ')"
  read -r -a case_list <<< "$(printf '%s' "${EXPERIMENT_CASES}" | tr ',' ' ')"

  if [[ ${#model_path_list[@]} -eq 0 ]]; then
    echo "No model paths were resolved from MODEL_PATHS=${MODEL_PATHS}" >&2
    exit 1
  fi
  if [[ ${#case_list[@]} -eq 0 ]]; then
    echo "No experiment cases were resolved from EXPERIMENT_CASES=${EXPERIMENT_CASES}" >&2
    exit 1
  fi

  for model_path in "${model_path_list[@]}"; do
    if [[ -z "${model_path}" ]]; then
      continue
    fi
    resolved_model_path="$(resolve_model_path "${model_path}")"
    for case_name in "${case_list[@]}"; do
      if [[ -z "${case_name}" ]]; then
        continue
      fi
      run_case "${resolved_model_path}" "${case_name}"
    done
  done
}

write_seed_report() {
  local base_config_tag="${ALL_CONFIG_TAG}"
  local sanitized_base_tag
  local report_args=()

  sanitized_base_tag=$(sanitize_path_component "${base_config_tag}")
  set_default RESULT_JSON "${BENCHMARK_ROOT}/output/_summaries/${sanitized_base_tag}_seed_sweep_best.json"

  if is_enabled "${DRY_RUN}"; then
    echo "==> DRY_RUN seed report"
    echo "    REPORT_SCRIPT=${REPORT_SCRIPT}"
    echo "    RESULT_JSON=${RESULT_JSON}"
    echo "    ALLOW_MISSING_SUMMARIES=${ALLOW_MISSING_SUMMARIES}"
    return
  fi

  report_args=(
    --root-path "${ROOT_PATH}"
    --benchmark-root "${BENCHMARK_ROOT}"
    --model-paths "${MODEL_PATHS}"
    --tag-prefix "${base_config_tag}"
    --seeds "${SEEDS}"
    --cases "${EXPERIMENT_CASES}"
    --image-priori-mode "${IMAGE_PRIORI_MODE}"
    --prefill-mode "${PREFILL_MODE}"
    --output-json "${RESULT_JSON}"
  )

  case "${ALLOW_MISSING_SUMMARIES}" in
    1|true|TRUE|yes|YES|on|ON)
      report_args+=(--allow-missing)
      ;;
  esac

  "${PYTHON_BIN}" "${REPORT_SCRIPT}" "${report_args[@]}"
}

apply_preset_defaults
apply_common_defaults
validate_files
validate_seed_settings

case "${RUN_MODE}" in
  eval)
    run_sweeps
    ;;
  seed_sweep)
    echo "==> Running seeds ${SEEDS} with base tag ${ALL_CONFIG_TAG}"
    run_sweeps
    write_seed_report
    ;;
  *)
    echo "Unknown RUN_MODE: ${RUN_MODE}" >&2
    echo "Known modes: eval, seed_sweep" >&2
    exit 1
    ;;
esac
