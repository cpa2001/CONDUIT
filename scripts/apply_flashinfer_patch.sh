#!/usr/bin/env bash
# Replace flashinfer 0.6.6's sparse.py with the version shipped in this repo.
# Idempotent: returns 0 immediately if the patch is already applied.
#
#   PYTHON=/path/to/python bash scripts/apply_flashinfer_patch.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PATCHED="${REPO_ROOT}/patches/flashinfer-0.6.6/sparse.py"
EXPECTED_MD5="b539dec2b4104b5491a53dcbf3c76a5b"

PYTHON=${PYTHON:-python}

if [ ! -f "${PATCHED}" ]; then
    echo "ERROR: patch file not found: ${PATCHED}" >&2
    exit 1
fi

actual_md5=$(md5sum "${PATCHED}" | awk '{print $1}')
if [ "${actual_md5}" != "${EXPECTED_MD5}" ]; then
    echo "ERROR: patch md5 mismatch: expected ${EXPECTED_MD5}, got ${actual_md5}" >&2
    exit 1
fi

fi_version=$("${PYTHON}" -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))' 2>/dev/null || echo "")
if [ -z "${fi_version}" ]; then
    echo "ERROR: flashinfer-python is not installed for ${PYTHON}" >&2
    echo "       run first: pip install flashinfer-python==0.6.6" >&2
    exit 1
fi
if [ "${fi_version}" != "0.6.6" ]; then
    echo "ERROR: this patch targets flashinfer 0.6.6 only, found ${fi_version}" >&2
    exit 1
fi

# Locate the package with find_spec rather than importing it: importing
# flashinfer triggers heavy module initialisation, and importing a package in
# the very script that is about to overwrite one of its files is unwise.
target=$("${PYTHON}" -c '
import importlib.util, os, sys
spec = importlib.util.find_spec("flashinfer")
if spec is None or not spec.submodule_search_locations:
    sys.exit("cannot locate the flashinfer package directory")
print(os.path.join(list(spec.submodule_search_locations)[0], "sparse.py"))
')
if [ ! -f "${target}" ]; then
    echo "ERROR: ${target} not found" >&2
    exit 1
fi

if cmp -s "${PATCHED}" "${target}"; then
    echo "Already patched, nothing to do: ${target}"
    exit 0
fi

backup="${target}.orig"
if [ ! -f "${backup}" ]; then
    cp -p "${target}" "${backup}"
    echo "Original file backed up to ${backup}"
fi

cp -p "${PATCHED}" "${target}"
if cmp -s "${PATCHED}" "${target}"; then
    echo "Patch applied: ${target}"
else
    echo "ERROR: verification after write failed" >&2
    exit 1
fi
