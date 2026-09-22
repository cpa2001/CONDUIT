#!/usr/bin/env bash
# Shared helpers for the MMLongBench experiment scripts.
#
# Usage: source this file near the top of a script
#   SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
#   source "${SCRIPT_DIR}/lib_common.sh"
#
# These four functions used to be copied verbatim into
# run_eval_nanovllm_conduit.sh, run_goal_image_bias_sensitivity.sh and
# export_goal_image_bias_sensitivity_csv.sh. Extracting them here changes
# nothing about their behaviour.

# Assign a default only when the variable is unset. An empty string counts as
# set and is left alone. This is the usual idiom for "env vars override the
# script's defaults".
set_default() {
  local name="$1"
  local value="$2"

  if [[ -z "${!name+x}" ]]; then
    printf -v "${name}" '%s' "${value}"
  fi
}

# Treat the common spellings of "true" uniformly.
is_enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

# Turn a value that may contain / \ or spaces into something safe to use as
# a directory name.
sanitize_path_component() {
  local value="$1"
  value=${value//\//__}
  value=${value//\\/__}
  value=${value// /_}
  printf '%s\n' "${value}"
}

# Print a command, quoted well enough to be copy-pasted and replayed as is.
print_command() {
  printf '%q ' "$@"
  printf '\n'
}
