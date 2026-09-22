#!/bin/bash
# Plotting step for M4: turn the output of run_recompute_sweep.sh into
# budget-versus-quality curves.
#
#   bash scripts/run_plot_recompute_sweeps.sh
#
# Reads ${OUTPUTS_ROOT} (default ../outputs) and writes to ${OUTPUT_DIR}
# (default ../figures/recompute_sweeps). FORMATS selects the output formats.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

OUTPUTS_ROOT=${OUTPUTS_ROOT:-"${PROJECT_ROOT}/outputs"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/figures/recompute_sweeps"}
MODELS=${MODELS:-""}
BENCHMARKS=${BENCHMARKS:-""}
FORMATS=${FORMATS:-"png pdf"}
NCOLS=${NCOLS:-3}
PYTHON_SCRIPT="${SCRIPT_DIR}/plot_recompute_sweeps.py"

CMD=(python "${PYTHON_SCRIPT}" "${OUTPUTS_ROOT}" --output-dir "${OUTPUT_DIR}" --ncols "${NCOLS}" --formats)

for fmt in ${FORMATS}; do
    CMD+=("${fmt}")
done

if [[ -n "${MODELS}" ]]; then
    CMD+=(--models)
    for model in ${MODELS}; do
        CMD+=("${model}")
    done
fi

if [[ -n "${BENCHMARKS}" ]]; then
    CMD+=(--benchmarks)
    for benchmark in ${BENCHMARKS}; do
        CMD+=("${benchmark}")
    done
fi

"${CMD[@]}"