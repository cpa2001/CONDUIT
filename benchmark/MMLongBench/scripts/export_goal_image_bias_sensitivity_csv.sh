#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/lib_common.sh"
BENCHMARK_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
EXPORT_SCRIPT="${SCRIPT_DIR}/export_goal_image_bias_sensitivity_csv.py"

append_split_values() {
  local flag="$1"
  local raw_value="$2"
  local normalized
  local item

  normalized=${raw_value//,/ }
  for item in ${normalized}; do
    if [[ -n "${item}" ]]; then
      export_cmd+=("${flag}" "${item}")
    fi
  done
}

set_default PYTHON_BIN "python"
set_default TAG "goal_image_bias_sensitivity_8k"
set_default OUTPUT_ROOT "${BENCHMARK_ROOT}/output"
set_default IMAGE_PRIORI_MODE "chat_template"
set_default PREFILL_MODE "image_segment"
set_default OUTPUT_DIR "${OUTPUT_ROOT}/_summaries/${TAG}"
set_default MODEL_OUTPUT_DIRS ""
set_default SUMMARY_JSONS ""
set_default EXPECTED_BIAS_VALUES ""
set_default EXPECTED_SEEDS ""
set_default DETAIL_OUTPUT ""
set_default SUMMARY_OUTPUT ""
set_default MODEL_AVG_OUTPUT ""
set_default INCLUDE_MISSING_DETAIL "1"
set_default ALLOW_MISSING "0"
set_default PRECISION "6"

if [[ ! -f "${EXPORT_SCRIPT}" ]]; then
  echo "Missing export script: ${EXPORT_SCRIPT}" >&2
  exit 1
fi

export_cmd=(
  "${PYTHON_BIN}"
  "${EXPORT_SCRIPT}"
  --benchmark-root "${BENCHMARK_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --tag "${TAG}"
  --image-priori-mode "${IMAGE_PRIORI_MODE}"
  --prefill-mode "${PREFILL_MODE}"
  --output-dir "${OUTPUT_DIR}"
  --precision "${PRECISION}"
)

if [[ -n "${MODEL_OUTPUT_DIRS}" ]]; then
  append_split_values --model-output-dir "${MODEL_OUTPUT_DIRS}"
fi

if [[ -n "${SUMMARY_JSONS}" ]]; then
  append_split_values --summary-json "${SUMMARY_JSONS}"
fi

if [[ -n "${EXPECTED_BIAS_VALUES}" ]]; then
  export_cmd+=(--expected-bias-values "${EXPECTED_BIAS_VALUES}")
fi

if [[ -n "${EXPECTED_SEEDS}" ]]; then
  export_cmd+=(--expected-seeds "${EXPECTED_SEEDS}")
fi

if [[ -n "${DETAIL_OUTPUT}" ]]; then
  export_cmd+=(--detail-output "${DETAIL_OUTPUT}")
fi

if [[ -n "${SUMMARY_OUTPUT}" ]]; then
  export_cmd+=(--summary-output "${SUMMARY_OUTPUT}")
fi

if [[ -n "${MODEL_AVG_OUTPUT}" ]]; then
  export_cmd+=(--model-avg-output "${MODEL_AVG_OUTPUT}")
fi

if ! is_enabled "${INCLUDE_MISSING_DETAIL}"; then
  export_cmd+=(--no-missing-detail)
fi

if is_enabled "${ALLOW_MISSING}"; then
  export_cmd+=(--allow-missing)
fi

echo "==> Exporting goal image-bias sensitivity CSV"
echo "    tag: ${TAG}"
echo "    output dir: ${OUTPUT_DIR}"
echo "    command:"
print_command "${export_cmd[@]}"
"${export_cmd[@]}"
