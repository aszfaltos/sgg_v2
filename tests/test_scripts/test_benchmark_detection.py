"""Unit tests for detection benchmark script.

Tests the CLI argument parsing, data loading, inference, and result formatting
for the detection benchmark script.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Ensure we import from project root scripts, not tests/scripts
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import will be added after implementation
# from scripts.benchmark_detection import main, parse_args, run_inference, format_results


class TestParseArgs:
    """Test CLI argument parsing."""

    @patch("sys.argv", ["benchmark_detection.py", "--detector", "fasterrcnn", "--split", "test"])
    def test_parse_args_defaults(self):
        """Test parsing with only required arguments."""
        from scripts.benchmark_detection import parse_args

        args = parse_args()

        assert args.detector == "fasterrcnn"
        assert args.split == "test"
        assert args.batch_size == 1  # Default
        assert args.device == "cuda" if torch.cuda.is_available() else "cpu"
        assert args.output is None  # Default
        assert args.backbone is None  # Default

    @patch(
        "sys.argv",
        [
            "benchmark_detection.py",
            "--detector",
            "efficientdet",
            "--variant",
            "d2",
            "--split",
            "train",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--output",
            "results.json",
        ],
    )
    def test_parse_args_all_options(self):
        """Test parsing with all arguments specified."""
        from scripts.benchmark_detection import parse_args

        args = parse_args()

        assert args.detector == "efficientdet"
        assert args.variant == "d2"
        assert args.split == "train"
        assert args.batch_size == 4
        assert args.device == "cpu"
        assert args.output == "results.json"


class TestRunInference:
    """Test inference loop."""

    def test_run_inference_with_mock_detector(self):
        """Test inference loop collects predictions and targets correctly."""
        from scripts.benchmark_detection import run_inference

        # Mock detector that returns fixed predictions
        mock_detector = MagicMock()
        # Configure to() and eval() to return self (common PyTorch pattern)
        mock_detector.to.return_value = mock_detector
        mock_detector.eval.return_value = mock_detector
        # Configure forward pass
        mock_output = MagicMock()
        mock_output.boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        mock_output.labels = [torch.tensor([1])]
        mock_output.scores = [torch.tensor([0.9])]
        mock_detector.return_value = mock_output

        # Mock dataset with 2 images
        mock_dataset = [
            (
                torch.rand(3, 100, 100),
                {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])},
            ),
            (
                torch.rand(3, 100, 100),
                {"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]), "labels": torch.tensor([2])},
            ),
        ]

        predictions, targets = run_inference(mock_detector, mock_dataset, device="cpu")

        # Check we got 2 predictions and 2 targets
        assert len(predictions) == 2
        assert len(targets) == 2

        # Check format
        assert "boxes" in predictions[0]
        assert "labels" in predictions[0]
        assert "scores" in predictions[0]
        assert "boxes" in targets[0]
        assert "labels" in targets[0]

        # Note: mock_detector() is called (forward pass), not mock_detector.to() or .eval()
        # So we check the return value (mock_output) was accessed correctly
        assert len(predictions) == 2  # Already checked above

    def test_run_inference_handles_empty_detections(self):
        """Test that empty detection outputs are handled correctly."""
        from scripts.benchmark_detection import run_inference

        # Mock detector that returns no detections
        mock_detector = MagicMock()
        # Configure to() and eval() to return self
        mock_detector.to.return_value = mock_detector
        mock_detector.eval.return_value = mock_detector
        # Configure forward pass with empty detections
        mock_output = MagicMock()
        mock_output.boxes = [torch.zeros((0, 4))]
        mock_output.labels = [torch.zeros((0,), dtype=torch.int64)]
        mock_output.scores = [torch.zeros((0,))]
        mock_detector.return_value = mock_output

        mock_dataset = [
            (
                torch.rand(3, 100, 100),
                {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])},
            ),
        ]

        predictions, targets = run_inference(mock_detector, mock_dataset, device="cpu")

        # Check we got predictions with empty tensors
        assert len(predictions) == 1
        assert predictions[0]["boxes"].shape == (0, 4)
        assert predictions[0]["labels"].shape == (0,)
        assert predictions[0]["scores"].shape == (0,)


class TestFormatResults:
    """Test result formatting."""

    def test_format_results_creates_valid_json(self):
        """Test that results are formatted into valid JSON structure."""
        from scripts.benchmark_detection import format_results

        metrics = {
            "mAP@0.5": 0.45,
            "mAP@0.5:0.95": 0.32,
            "AR@100": 0.52,
            "mAP_per_class": {1: 0.5, 2: 0.4},
        }

        config = {
            "detector": "fasterrcnn",
            "backbone": "resnet50",
            "split": "test",
            "batch_size": 4,
        }

        result = format_results(metrics, config)

        # Check top-level keys
        assert "detector" in result
        assert "split" in result
        assert "metrics" in result
        assert "config" in result
        assert "timestamp" in result

        # Check values
        assert result["detector"] == "fasterrcnn"
        assert result["split"] == "test"
        assert result["metrics"]["mAP@0.5"] == 0.45

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_format_results_includes_per_class_metrics(self):
        """Test that per-class metrics are included in results."""
        from scripts.benchmark_detection import format_results

        metrics = {
            "mAP@0.5": 0.45,
            "mAP@0.5:0.95": 0.32,
            "AR@100": 0.52,
            "mAP_per_class": {1: 0.5, 2: 0.4, 3: 0.6},
        }

        config = {"detector": "fasterrcnn"}

        result = format_results(metrics, config)

        assert "per_class" in result
        assert result["per_class"] == {1: 0.5, 2: 0.4, 3: 0.6}


class TestMainIntegration:
    """Integration tests for main function."""

    @pytest.mark.slow
    @patch("scripts.benchmark_detection.VRDDetectionDataset")
    @patch("scripts.benchmark_detection.create_detector")
    @patch("scripts.benchmark_detection.DetectionEvaluator")
    def test_main_runs_end_to_end(self, mock_evaluator_cls, mock_create_detector, mock_dataset_cls):
        """Test that main function runs end-to-end with mocked dependencies."""
        from scripts.benchmark_detection import main

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        mock_dataset.__getitem__.side_effect = [
            (
                torch.rand(3, 100, 100),
                {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])},
            ),
            (
                torch.rand(3, 100, 100),
                {"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]), "labels": torch.tensor([2])},
            ),
        ]
        mock_dataset_cls.return_value = mock_dataset

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.return_value = MagicMock(
            boxes=[torch.tensor([[10.0, 10.0, 50.0, 50.0]])],
            labels=[torch.tensor([1])],
            scores=[torch.tensor([0.9])],
        )
        mock_create_detector.return_value = mock_detector

        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.compute.return_value = {
            "mAP@0.5": 0.45,
            "mAP@0.5:0.95": 0.32,
            "AR@100": 0.52,
            "mAP_per_class": {},
        }
        mock_evaluator_cls.return_value = mock_evaluator

        # Create temp output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            # Run main
            main(
                detector="fasterrcnn",
                backbone="resnet50",
                variant=None,
                split="test",
                batch_size=1,
                device="cpu",
                output=output_path,
                dataset_root="datasets/vrd",
            )

            # Check that output file was created
            assert Path(output_path).exists()

            # Check that file contains valid JSON
            with open(output_path) as f:
                result = json.load(f)

            assert "metrics" in result
            assert result["metrics"]["mAP@0.5"] == 0.45

        finally:
            # Cleanup
            Path(output_path).unlink(missing_ok=True)

    def test_main_saves_results_to_file(self):
        """Test that results are saved to JSON file."""
        # This will be tested in the integration test above
        pass
