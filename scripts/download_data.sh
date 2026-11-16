#!/usr/bin/env bash
# Convenience wrapper around the data acquisition commands.
#
#   scripts/download_data.sh              # resolve scenes + build the dataset
#   scripts/download_data.sh --scenes     # only resolve and verify the scenes
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

if [[ "${1:-}" == "--scenes" ]]; then
    shift
    "$PYTHON" commands.py download "$@"
else
    "$PYTHON" commands.py preprocess "$@"
fi
