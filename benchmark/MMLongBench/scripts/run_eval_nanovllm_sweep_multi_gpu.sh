#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_PATH=${ROOT_PATH:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
PROJECT_ROOT=${PROJECT_ROOT:-${ROOT_PATH}}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-${PROJECT_ROOT}/benchmark/MMLongBench}
PREPARE_ENV_SCRIPT=${PREPARE_ENV_SCRIPT:-${PROJECT_ROOT}/prepare_env.sh}
WORKSPACE_ROOT=$(cd -- "${ROOT_PATH}/.." && pwd)
CONDA_ROOT=${CONDA_ROOT:-${WORKSPACE_ROOT}/miniconda3}

CURRENT_CHILD_PID=""

# prepare_environment() {
  
  # local conda_profile="${CONDA_ROOT}/etc/profile.d/conda.sh"

  # if [[ ! -f "${PREPARE_ENV_SCRIPT}" ]]; then
  #   echo "Missing prepare_env.sh: ${PREPARE_ENV_SCRIPT}" >&2
  #   exit 1
  # fi

  # if ! declare -F conda >/dev/null 2>&1; then
  #   if [[ ! -f "${conda_profile}" ]]; then
  #     echo "Missing conda shell hook: ${conda_profile}" >&2
  #     exit 1
  #   fi
  #   set +u
  #   # shellcheck disable=SC1090
  #   source "${conda_profile}"
  #   set -u
  # fi

  # set +u
  # # shellcheck disable=SC1090
  # source "${PREPARE_ENV_SCRIPT}"
  # set -u
# }

cleanup_active_child() {
  local child_pid="${CURRENT_CHILD_PID:-}"

  if [[ -z "${child_pid}" ]]; then
    return
  fi
  if ! kill -0 "${child_pid}" 2>/dev/null; then
    CURRENT_CHILD_PID=""
    return
  fi

  kill -TERM "${child_pid}" 2>/dev/null || true
  wait "${child_pid}" 2>/dev/null || true
  CURRENT_CHILD_PID=""
}

on_exit() {
  local exit_code="$1"

  trap - EXIT
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "run_eval_nanovllm_sweep_multi_gpu.sh exited with ${exit_code}, cleaning up active subprocesses." >&2
  fi
  cleanup_active_child
  exit "${exit_code}"
}

on_signal() {
  local signal_name="$1"
  local signal_number="$2"

  echo "Received ${signal_name}, stopping active subprocesses." >&2
  exit "$((128 + signal_number))"
}

trap 'on_exit $?' EXIT
trap 'on_signal INT 2' INT
trap 'on_signal TERM 15' TERM

# prepare_environment


export LLM_JUDGE_KEY=${LLM_JUDGE_KEY:-}
export LLM_JUDGE_ENDPOINT=${LLM_JUDGE_ENDPOINT:-}

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-${ROOT_PATH}/models/internvl3-9b}
CONFIG_SWEEP=${CONFIG_SWEEP:-entries}
GPU_LIST=${GPU_LIST:-1}
GPU_GROUP_SIZE=${GPU_GROUP_SIZE:-1}
RERUN=${RERUN:-1}
ALL_CONFIG_TAG=${ALL_CONFIG_TAG:-sweep_configs_8k_sweep_mml}
PREFILL_MODE=${PREFILL_MODE:-image_segment}
IMAGE_PRIORI_MODE=${IMAGE_PRIORI_MODE:-chat_template}
RECOMPUTE_STRATEGIES=${RECOMPUTE_STRATEGIES:-kv_score}
RECOMPUTE_TEMPLATE=${RECOMPUTE_TEMPLATE:-none}
RATIO_VALUES=${RATIO_VALUES:-10}
EXTRA_EVAL_ARGS=${EXTRA_EVAL_ARGS:---docqa_llm_judge False}
DO_SAMPLE=${DO_SAMPLE:-}
TEMPERATURE=${TEMPERATURE:-}
TOP_P=${TOP_P:-}
SEED=${SEED:-}
SEEDS=${SEEDS:-}
CONDUIT_SCORE_LAYER_IDX=${CONDUIT_SCORE_LAYER_IDX:-}
CONDUIT_SCORE_LAYER_FROM_LAST=${CONDUIT_SCORE_LAYER_FROM_LAST:-}
CONDUIT_SCORE_LAYER_SPLIT_PARTS=${CONDUIT_SCORE_LAYER_SPLIT_PARTS:-}
CONDUIT_SCORE_LAYER_SPLIT_PART=${CONDUIT_SCORE_LAYER_SPLIT_PART:-}
CONDUIT_SCORE_USE_V_NORM=${CONDUIT_SCORE_USE_V_NORM:-}
CONDUIT_SCORE_USE_V_NORM_VALUES=${CONDUIT_SCORE_USE_V_NORM_VALUES:-}
CONDUIT_SCORE_IMAGE_BIAS_STRENGTH=${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH:-0.0}
CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES=${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES:-0.0}

if [[ -n "${SEED}" && -n "${SEEDS}" ]]; then
  echo "SEED and SEEDS are mutually exclusive." >&2
  exit 1
fi

GENERATION_SWEEP_ARGS=()
if [[ -n "${DO_SAMPLE}" ]]; then
  GENERATION_SWEEP_ARGS+=(--do_sample "${DO_SAMPLE}")
fi
if [[ -n "${TEMPERATURE}" ]]; then
  GENERATION_SWEEP_ARGS+=(--temperature "${TEMPERATURE}")
fi
if [[ -n "${TOP_P}" ]]; then
  GENERATION_SWEEP_ARGS+=(--top_p "${TOP_P}")
fi
if [[ -n "${SEEDS}" ]]; then
  GENERATION_SWEEP_ARGS+=(--seed-values "${SEEDS}")
elif [[ -n "${SEED}" ]]; then
  GENERATION_SWEEP_ARGS+=(--seed "${SEED}")
fi

CONDUIT_SCORE_PROBE_SWEEP_ARGS=()
if [[ -n "${CONDUIT_SCORE_LAYER_IDX}" ]]; then
  CONDUIT_SCORE_PROBE_SWEEP_ARGS+=(--kv-score-layer-idx "${CONDUIT_SCORE_LAYER_IDX}")
fi
if [[ -n "${CONDUIT_SCORE_LAYER_FROM_LAST}" ]]; then
  CONDUIT_SCORE_PROBE_SWEEP_ARGS+=(--kv-score-layer-from-last "${CONDUIT_SCORE_LAYER_FROM_LAST}")
fi
if [[ -z "${CONDUIT_SCORE_LAYER_IDX}" && -z "${CONDUIT_SCORE_LAYER_FROM_LAST}" && -n "${CONDUIT_SCORE_LAYER_SPLIT_PARTS}" ]]; then
  CONDUIT_SCORE_PROBE_SWEEP_ARGS+=(--kv-score-layer-split-parts "${CONDUIT_SCORE_LAYER_SPLIT_PARTS}")
fi
if [[ -z "${CONDUIT_SCORE_LAYER_IDX}" && -z "${CONDUIT_SCORE_LAYER_FROM_LAST}" && -n "${CONDUIT_SCORE_LAYER_SPLIT_PART}" ]]; then
  CONDUIT_SCORE_PROBE_SWEEP_ARGS+=(--kv-score-layer-split-part "${CONDUIT_SCORE_LAYER_SPLIT_PART}")
fi
if [[ -n "${CONDUIT_SCORE_USE_V_NORM_VALUES}" ]]; then
  CONDUIT_SCORE_PROBE_SWEEP_ARGS+=(--kv-score-use-v-norm-values "${CONDUIT_SCORE_USE_V_NORM_VALUES}")
elif [[ -n "${CONDUIT_SCORE_USE_V_NORM}" ]]; then
  CONDUIT_SCORE_PROBE_SWEEP_ARGS+=(--kv-score-use-v-norm "${CONDUIT_SCORE_USE_V_NORM}")
fi
if [[ -n "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES}" ]]; then
  CONDUIT_SCORE_PROBE_SWEEP_ARGS+=(--kv-score-image-bias-strength-values "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH_VALUES}")
elif [[ -n "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}" ]]; then
  CONDUIT_SCORE_PROBE_SWEEP_ARGS+=(--kv-score-image-bias-strength "${CONDUIT_SCORE_IMAGE_BIAS_STRENGTH}")
fi

if [[ -n "${GPU_GROUPS:-}" ]]; then
  GPU_ARGS=(--gpu-groups "${GPU_GROUPS}")
else
  GPU_ARGS=(--gpu-list "${GPU_LIST}" --gpu-group-size "${GPU_GROUP_SIZE}")
fi

case "${RERUN}" in
  1|true|TRUE|yes|YES|on|ON)
    RERUN_ARGS=(--rerun)
    ;;
  0|false|FALSE|no|NO|off|OFF|"")
    RERUN_ARGS=()
    ;;
  *)
    echo "Invalid RERUN value: ${RERUN}" >&2
    exit 1
    ;;
esac

if [[ "${CONFIG_SWEEP}" != "none" ]]; then
  CONFIG_SWEEP_ARGS=(--config-sweep "${CONFIG_SWEEP}")
else
  CONFIG_SWEEP_ARGS=()
fi

if [[ -n "${CONFIG_FILES:-}" ]]; then
  read -r -a CONFIG_FILE_ARGS <<< "${CONFIG_FILES}"
else
  CONFIG_FILE_ARGS=(
    # configs/docqa_all_8k.yaml
    # configs/vh_all_8k.yaml
    # configs/vrag_all_8k.yaml
    configs/mmlongdoc_8k.yaml
    # configs/infoseek_8k.yaml
    # configs/icl_car_8k.yaml
  )
fi


set +e
"${PYTHON_BIN}" ${BENCHMARK_ROOT}/scripts/dispatch_nanovllm_sweep.py \
  --config-files "${CONFIG_FILE_ARGS[@]}" \
  "${CONFIG_SWEEP_ARGS[@]}" \
  --model_name_or_path ${MODEL_PATH} \
  --test_file_root ${BENCHMARK_ROOT}/mmlb_data \
  --image_file_root ${BENCHMARK_ROOT}/mmlb_image \
  --prefill_mode ${PREFILL_MODE} \
  --image-priori-mode ${IMAGE_PRIORI_MODE} \
  --recompute-strategies ${RECOMPUTE_STRATEGIES} \
  --recompute-template "${RECOMPUTE_TEMPLATE}" \
  --ratio-values ${RATIO_VALUES} \
  "${GENERATION_SWEEP_ARGS[@]}" \
  "${CONDUIT_SCORE_PROBE_SWEEP_ARGS[@]}" \
  --extra-eval-args "${EXTRA_EVAL_ARGS}" \
  "${GPU_ARGS[@]}" \
  --tag ${ALL_CONFIG_TAG} \
  "${RERUN_ARGS[@]}" &
CURRENT_CHILD_PID=$!
wait "${CURRENT_CHILD_PID}"
run_status=$?
CURRENT_CHILD_PID=""
set -e

exit "${run_status}"
