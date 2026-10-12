#!/usr/bin/env bash
# Run from any directory; keep caller-relative input/output paths intact.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$script_dir")"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python}" -m sigcomp.multi_gpu "$@"
