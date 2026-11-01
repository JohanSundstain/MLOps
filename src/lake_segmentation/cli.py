"""Command line interface.

One entry point (``commands.py`` at the repository root) exposes every stage of the project
with `Fire <https://github.com/google/python-fire>`_. Each command composes the Hydra config
through the compose API, so the CLI and the YAML configs never diverge::

    python commands.py download
    python commands.py preprocess --overrides="data.patch.max_patches_per_scene=10"
    python commands.py train --overrides="train.max_epochs=3,train.batch_size=4"
    python commands.py export-onnx
    python commands.py infer --image-a a.tif --image-b b.tif
"""

from __future__ import annotations

import json
from typing import Any

import fire
from omegaconf import DictConfig

from lake_segmentation.config import load_config, resolve_path, resolve_paths
from lake_segmentation.logging_utils import get_logger, setup_logging

LOGGER = get_logger("cli")


def _print_summary(title: str, payload: dict[str, Any]) -> None:
    """Pretty print a dictionary resulting from a command."""

    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print()


class LakeSegmentationCli:
    """Water segmentation of lakes on Sentinel-2 imagery."""

    def download(
        self,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
    ) -> None:
        """Resolve the best Sentinel-2 scene for every AOI and store the scenes manifest."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.data.pipeline import run_download

        _print_summary("download", run_download(cfg))

    def preprocess(
        self,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
    ) -> None:
        """Build the patch dataset, the split and the normalisation statistics."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.data.pipeline import run_preprocess

        _print_summary("preprocess", run_preprocess(cfg))

    def make_example(
        self,
        aoi: str = "chany",
        size: int = 1024,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
    ) -> None:
        """Cut a small example pair of GeoTIFFs for the inference demo."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.data.example import build_example_pair

        _print_summary("make-example", build_example_pair(cfg, aoi_name=aoi, size=size))

    def train(
        self,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
    ) -> None:
        """Train the U-Net, tune the threshold and export the artefacts."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.training.train import run_training

        _print_summary("train", run_training(cfg))

    def baseline(
        self,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
    ) -> None:
        """Evaluate the NDWI threshold baseline on the test split."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.baseline.ndwi import run_ndwi_baseline

        _print_summary("baseline", run_ndwi_baseline(cfg))

    def evaluate(
        self,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
    ) -> None:
        """Metrics of the trained model on the test split (uses the stored bundle)."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.training.evaluate import run_evaluation

        _print_summary("evaluate", run_evaluation(cfg))

    def export_onnx(
        self,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
    ) -> None:
        """Export the checkpoint to ONNX and validate it against PyTorch."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.export.onnx import describe_onnx, export_onnx

        path = export_onnx(cfg)
        _print_summary("export-onnx", {"path": str(path), **describe_onnx(path)})

    def export_tensorrt(
        self,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
        dry_run: bool = False,
    ) -> None:
        """Build a TensorRT engine from the ONNX model (requires an NVIDIA GPU)."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.export.tensorrt import run_tensorrt_export

        path = run_tensorrt_export(cfg, dry_run=dry_run)
        _print_summary("export-tensorrt", {"engine": str(path), "dry_run": dry_run})

    def infer(
        self,
        image_a: str,
        image_b: str | None = None,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
        output_dir: str | None = None,
        date_a: str | None = None,
        date_b: str | None = None,
    ) -> None:
        """Segment one or two GeoTIFFs and compute the water area change."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.inference.run import run_inference

        report = run_inference(
            cfg,
            image_a=image_a,
            image_b=image_b,
            output_dir=output_dir,
            date_a=date_a,
            date_b=date_b,
        )
        summary = {
            "first_date": report["first"],
            "second_date": report.get("second"),
            "change": report.get("change"),
            "figure": report.get("figure"),
            "report": report.get("report_path"),
        }
        _print_summary("infer", summary)

    def serve_mlflow(
        self,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
    ) -> None:
        """Publish the ONNX model to MLflow and print the serving command."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.serving.mlflow_model import publish_onnx_model, write_request_example

        model_uri = publish_onnx_model(cfg)
        request_path = write_request_example(cfg)
        _print_summary(
            "serve-mlflow",
            {"model_uri": model_uri, "request_example": str(request_path)},
        )

    def prepare_triton(
        self,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
    ) -> None:
        """Create the Triton model repository and print the server command."""

        cfg = load_config(config_name, overrides, config_dir)
        from lake_segmentation.serving.triton import client_snippet, prepare_repository

        repository = prepare_repository(cfg)
        _print_summary(
            "prepare-triton", {"repository": str(repository), "client": client_snippet(cfg)}
        )

    def show_config(
        self,
        config_name: str = "config",
        overrides: str = "",
        config_dir: str | None = None,
    ) -> None:
        """Print the fully resolved config (useful to debug overrides)."""

        from omegaconf import OmegaConf

        cfg: DictConfig = load_config(config_name, overrides, config_dir)
        print(OmegaConf.to_yaml(cfg, resolve=True))

    def paths(self, config_name: str = "config", overrides: str = "") -> None:
        """Print the resolved project paths."""

        cfg = load_config(config_name, overrides)
        _print_summary("paths", {key: str(value) for key, value in resolve_paths(cfg).items()})

    def dataset_info(self, config_name: str = "config", overrides: str = "") -> None:
        """Print the dataset summary (patch counts, split, normalisation)."""

        cfg = load_config(config_name, overrides)
        summary_path = resolve_path(cfg.paths.data_processed) / "dataset_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"{summary_path} does not exist, run `python commands.py preprocess` first"
            )
        print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))


def main() -> None:
    """Entry point used by ``commands.py`` and by the ``lake-segmentation`` script."""

    setup_logging()
    fire.Fire(LakeSegmentationCli)


if __name__ == "__main__":
    main()
