"""Import-level regression tests.

The whole package must be importable with the installed dependency versions: a renamed
upstream class (for example ``MLFlowLogger`` in Lightning 2.6) otherwise only shows up when a
command is executed for the first time.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import lake_segmentation


def _all_module_names() -> list[str]:
    names = [lake_segmentation.__name__]
    for module_info in pkgutil.walk_packages(
        lake_segmentation.__path__, prefix=f"{lake_segmentation.__name__}."
    ):
        names.append(module_info.name)
    return names


@pytest.mark.parametrize("module_name", _all_module_names())
def test_module_is_importable(module_name: str):
    importlib.import_module(module_name)


def test_training_entry_point_is_importable():
    from lake_segmentation.training.train import run_training

    assert callable(run_training)


def test_mlflow_logger_class_is_available():
    from lake_segmentation.training.loggers import MLflowLogger

    assert MLflowLogger is not None


def test_flatten_config_produces_flat_scalars():
    from lake_segmentation.training.loggers import flatten_config

    flat = flatten_config({"a": {"b": 1.5, "c": True}, "d": [1, 2], "e": None})
    assert flat["a.b"] == 1.5
    assert flat["a.c"] == "True"
    assert flat["d"] == "[1, 2]"
    assert flat["e"] == "None"


def test_unreachable_tracking_server_is_detected():
    from lake_segmentation.training.loggers import is_server_available

    assert is_server_available("http://127.0.0.1:9", timeout=0.3) is False


def test_streaming_metrics_match_numpy_definitions():
    import torch

    from lake_segmentation.metrics import evaluate_binary_masks
    from lake_segmentation.training.metrics import build_metric_collection

    predictions = torch.tensor([[[[1, 1], [0, 0]]], [[[0, 0], [1, 1]]]], dtype=torch.int32)
    targets = torch.tensor([[[[1, 0], [0, 0]]], [[[0, 0], [1, 1]]]], dtype=torch.int32)

    collection = build_metric_collection("test/", threshold=0.5)
    collection.update(predictions, targets)
    computed = collection.compute()
    reference = evaluate_binary_masks(
        [predictions[0, 0].numpy().astype(bool), predictions[1, 0].numpy().astype(bool)],
        [targets[0, 0].numpy().astype(bool), targets[1, 0].numpy().astype(bool)],
    )
    for name in ("iou", "dice", "precision", "recall"):
        assert float(computed[f"test/{name}"]) == pytest.approx(reference[name], rel=1e-6)


def test_non_mlflow_http_service_is_not_detected_as_tracking_server():
    """A Jupyter-like server answering HTML on /health must not be treated as MLflow."""

    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from lake_segmentation.training.loggers import is_server_available

    class HtmlHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self.send_response(404)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>404 Not Found</body></html>")

        def log_message(self, *args) -> None:  # keep the test output clean
            return

    server = HTTPServer(("127.0.0.1", 0), HtmlHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert is_server_available(f"http://127.0.0.1:{port}", timeout=2) is False
    finally:
        server.shutdown()
        thread.join(timeout=5)
