#!/usr/bin/env bash
# One codec process per physical CPU socket; paths stay relative to the caller.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$script_dir")"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python}" -m sigcomp.multi_cpu_socket "$@"
