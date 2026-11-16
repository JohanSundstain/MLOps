#!/usr/bin/env bash
# Build a TensorRT engine from the exported ONNX model.
#
#   scripts/build_tensorrt.sh                 # uses configs/tensorrt/default.yaml
#   scripts/build_tensorrt.sh --dry-run       # only prints the trtexec command
#
# TensorRT requires an NVIDIA GPU. The recommended way is the official container:
#   docker run --gpus all --rm -it -v "$PWD:/work" -w /work nvcr.io/nvidia/tensorrt:24.08-py3 \
#       scripts/build_tensorrt.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

"$PYTHON" commands.py export-tensorrt "$@"
