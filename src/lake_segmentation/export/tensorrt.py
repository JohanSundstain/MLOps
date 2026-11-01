"""TensorRT engine build from the exported ONNX model.

TensorRT is only available on NVIDIA GPUs, so this step is a thin, well-tested wrapper around
``trtexec``: it validates the inputs, builds the command, runs it and checks that an engine
file appeared. The same logic is exposed as ``scripts/build_tensorrt.sh``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from omegaconf import DictConfig

from lake_segmentation.logging_utils import get_logger

LOGGER = get_logger("export.tensorrt")


def build_trtexec_command(cfg: DictConfig) -> list[str]:
    """Assemble the ``trtexec`` command line from the config."""

    tensorrt_cfg = cfg.tensorrt
    onnx_path = Path(tensorrt_cfg.onnx_model)
    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. Run `python commands.py export-onnx` first."
        )
    command = [
        str(tensorrt_cfg.trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={Path(tensorrt_cfg.engine)}",
        f"--shapes={tensorrt_cfg.shapes}",
        f"--memPoolSize=workspace:{int(tensorrt_cfg.workspace_gb)}GiB",
    ]
    if bool(tensorrt_cfg.fp16):
        command.append("--fp16")
    if bool(tensorrt_cfg.int8):
        command.append("--int8")
    extra = str(tensorrt_cfg.get("extra_args", "")).strip()
    if extra:
        command.extend(extra.split())
    return command


def run_tensorrt_export(cfg: DictConfig, dry_run: bool = False) -> Path:
    """Build the TensorRT engine (or only print the command with ``dry_run``)."""

    command = build_trtexec_command(cfg)
    engine_path = Path(cfg.tensorrt.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        LOGGER.info("trtexec command: %s", " ".join(command))
        return engine_path

    if shutil.which(command[0]) is None:
        raise FileNotFoundError(
            f"{command[0]!r} is not available on this machine. TensorRT requires an NVIDIA GPU "
            "and the TensorRT tools; use the official container, e.g.\n"
            "  docker run --gpus all -it --rm -v $PWD:/work nvcr.io/nvidia/tensorrt:24.08-py3\n"
            "and run `python commands.py export-tensorrt` inside it."
        )

    LOGGER.info("Building TensorRT engine: %s", " ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        LOGGER.error("trtexec failed with code %d", completed.returncode)
        LOGGER.error("stderr tail: %s", completed.stderr[-2000:])
        raise RuntimeError("TensorRT engine build failed, see the logs above")

    if not engine_path.is_file():
        raise RuntimeError(f"trtexec finished but {engine_path} does not exist")
    size_mb = engine_path.stat().st_size / (1024 * 1024)
    LOGGER.info("TensorRT engine written to %s (%.1f MiB)", engine_path, size_mb)
    return engine_path


def run_engine_benchmark(cfg: DictConfig) -> str:
    """Benchmark an existing engine with ``trtexec --loadEngine``."""

    trtexec = str(cfg.tensorrt.trtexec)
    engine_path = Path(cfg.tensorrt.engine)
    if not engine_path.is_file():
        raise FileNotFoundError(f"Engine not found at {engine_path}")
    command = [trtexec, f"--loadEngine={engine_path}", "--iterations=50", "--warmUp=500"]
    LOGGER.info("Benchmarking: %s", " ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"trtexec benchmark failed: {completed.stderr[-1000:]}")
    return completed.stdout
