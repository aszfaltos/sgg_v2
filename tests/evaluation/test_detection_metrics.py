"""Tests for DetectionEvaluator.

Following TDD - these tests are written before the implementation.

Test strategy:
- Happy path: Normal predictions and targets
- Edge cases: Empty predictions, empty targets, perfect matches
- Correctness: Verify metric computation
- State management: Update, compute, reset cycle
"""

import pytest
import torch


@pytest.fixture
def sample_predictions():
    """Sample predictions in COCO format."""
    return [
        {
            "boxes": torch.tensor(
                [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]
            ),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([1, 2]),
        },
        {
            "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
            "scores": torch.tensor([0.7]),
            "labels": torch.tensor([1]),
        },
    ]


@pytest.fixture
def sample_targets():
    """Sample ground truth targets in COCO format."""
    return [
        {
            "boxes": torch.tensor(
                [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]
            ),
            "labels": torch.tensor([1, 2]),
        },
        {
            "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
            "labels": torch.tensor([1]),
        },
    ]


@pytest.fixture
def empty_predictions():
    """Empty predictions (no detections) - matches sample_targets length."""
    return [
        {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        },
        {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        },
    ]


@pytest.fixture
def empty_targets():
    """Empty targets (no ground truth) - matches sample_predictions length."""
    return [
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)},
    ]


class TestDetectionEvaluatorInit:
    """Test DetectionEvaluator initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)

        assert evaluator.num_classes == 10
        assert evaluator.iou_thresholds is None  # Use torchmetrics default

    def test_init_custom_iou_thresholds(self):
        """Test initialization with custom IoU thresholds."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        iou_thresholds = [0.5, 0.75, 0.9]
        evaluator = DetectionEvaluator(num_classes=10, iou_thresholds=iou_thresholds)

        assert evaluator.num_classes == 10
        assert evaluator.iou_thresholds == iou_thresholds

    def test_init_validates_num_classes(self):
        """Test that num_classes must be positive."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        with pytest.raises(ValueError, match="num_classes must be positive"):
            DetectionEvaluator(num_classes=0)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            DetectionEvaluator(num_classes=-1)


class TestDetectionEvaluatorUpdate:
    """Test DetectionEvaluator update method."""

    def test_update_single_batch(self, sample_predictions, sample_targets):
        """Test updating with a single batch."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(sample_predictions, sample_targets)

        # Should not raise, state should be accumulated
        # Actual verification happens in compute()

    def test_update_multiple_batches(self, sample_predictions, sample_targets):
        """Test updating with multiple batches."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(sample_predictions, sample_targets)
        evaluator.update(sample_predictions, sample_targets)

        # Should accumulate both batches

    def test_update_with_empty_predictions(self, empty_predictions, sample_targets):
        """Test updating when predictions are empty."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)
        # Should handle gracefully (mAP will be 0)
        evaluator.update(empty_predictions, sample_targets)

    def test_update_with_empty_targets(self, sample_predictions, empty_targets):
        """Test updating when targets are empty."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)
        # Should handle gracefully
        evaluator.update(sample_predictions, empty_targets)

    def test_update_validates_prediction_format(self):
        """Test that update validates prediction format."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)

        # Missing required key
        bad_preds = [{"boxes": torch.zeros((1, 4))}]  # Missing scores, labels
        targets = [
            {"boxes": torch.zeros((1, 4)), "labels": torch.zeros(1, dtype=torch.long)}
        ]

        with pytest.raises((KeyError, ValueError)):
            evaluator.update(bad_preds, targets)

    def test_update_validates_target_format(self):
        """Test that update validates target format."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)

        preds = [
            {
                "boxes": torch.zeros((1, 4)),
                "scores": torch.ones(1),
                "labels": torch.zeros(1, dtype=torch.long),
            }
        ]
        # Missing required key
        bad_targets = [{"boxes": torch.zeros((1, 4))}]  # Missing labels

        with pytest.raises((KeyError, ValueError)):
            evaluator.update(preds, bad_targets)


class TestDetectionEvaluatorCompute:
    """Test DetectionEvaluator compute method."""

    def test_compute_returns_expected_keys(self, sample_predictions, sample_targets):
        """Test that compute returns expected metric keys."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(sample_predictions, sample_targets)

        result = evaluator.compute()

        # Check for standard COCO metrics
        assert "mAP@0.5" in result
        assert "mAP@0.5:0.95" in result
        assert "AR@100" in result

    def test_compute_returns_float_values(self, sample_predictions, sample_targets):
        """Test that compute returns float values (not tensors)."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(sample_predictions, sample_targets)

        result = evaluator.compute()

        # All values should be float
        for key, value in result.items():
            if key != "mAP_per_class":  # Skip dict
                assert isinstance(value, float), f"{key} is not float: {type(value)}"

    def test_compute_perfect_predictions(self):
        """Test that perfect predictions yield mAP = 1.0."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        # Create perfect predictions (exact match)
        perfect_preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([1.0]),
                "labels": torch.tensor([1]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]),
            }
        ]

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(perfect_preds, targets)
        result = evaluator.compute()

        # Perfect match should give mAP = 1.0
        assert result["mAP@0.5"] == 1.0
        assert result["mAP@0.5:0.95"] == 1.0

    def test_compute_empty_predictions_zero_map(
        self, empty_predictions, sample_targets
    ):
        """Test that empty predictions yield mAP = 0.0."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(empty_predictions, sample_targets)
        result = evaluator.compute()

        # No predictions should give mAP = 0
        assert result["mAP@0.5"] == 0.0
        assert result["mAP@0.5:0.95"] == 0.0

    def test_compute_without_update_raises(self):
        """Test that compute without update raises an error."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)

        # Should raise because no data has been accumulated
        with pytest.raises((RuntimeError, ValueError)):
            evaluator.compute()

    def test_compute_per_class_metrics(self, sample_predictions, sample_targets):
        """Test that per-class metrics are returned when requested."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(sample_predictions, sample_targets)
        result = evaluator.compute()

        # Should have per-class metrics
        assert "mAP_per_class" in result
        assert isinstance(result["mAP_per_class"], dict)


class TestDetectionEvaluatorReset:
    """Test DetectionEvaluator reset method."""

    def test_reset_clears_state(self, sample_predictions, sample_targets):
        """Test that reset clears accumulated state."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(sample_predictions, sample_targets)
        _result1 = evaluator.compute()

        # Reset and update with different data
        evaluator.reset()

        # Should raise because state is cleared
        with pytest.raises((RuntimeError, ValueError)):
            evaluator.compute()

    def test_reset_allows_new_accumulation(self, sample_predictions, sample_targets):
        """Test that after reset, new data can be accumulated."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(sample_predictions, sample_targets)
        _result1 = evaluator.compute()

        # Reset and update with new data
        evaluator.reset()
        evaluator.update(sample_predictions, sample_targets)
        result2 = evaluator.compute()

        # Should compute successfully
        assert "mAP@0.5" in result2


class TestDetectionEvaluatorIntegration:
    """Integration tests for DetectionEvaluator."""

    def test_full_workflow(self, sample_predictions, sample_targets):
        """Test full workflow: init -> update -> compute -> reset."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        evaluator = DetectionEvaluator(num_classes=10)

        # Accumulate first batch
        evaluator.update(sample_predictions[:1], sample_targets[:1])
        # Accumulate second batch
        evaluator.update(sample_predictions[1:], sample_targets[1:])

        # Compute metrics
        result = evaluator.compute()
        assert isinstance(result, dict)
        assert result["mAP@0.5"] >= 0.0
        assert result["mAP@0.5"] <= 1.0

        # Reset for new evaluation
        evaluator.reset()

        # New accumulation
        evaluator.update(sample_predictions, sample_targets)
        result2 = evaluator.compute()
        assert isinstance(result2, dict)

    def test_high_iou_predictions(self):
        """Test predictions with high IoU but not perfect."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        # Slightly offset prediction (high IoU but not perfect)
        preds = [
            {
                "boxes": torch.tensor([[11.0, 11.0, 51.0, 51.0]]),  # Offset by 1
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]),
            }
        ]

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(preds, targets)
        result = evaluator.compute()

        # Should have high mAP@0.5 (IoU > 0.5)
        assert result["mAP@0.5"] > 0.9
        # But not perfect for stricter thresholds
        assert result["mAP@0.5:0.95"] < 1.0

    def test_multiple_classes(self):
        """Test evaluation with multiple object classes."""
        from src.evaluation.detection_metrics import DetectionEvaluator

        preds = [
            {
                "boxes": torch.tensor(
                    [
                        [10.0, 10.0, 50.0, 50.0],
                        [60.0, 60.0, 100.0, 100.0],
                        [120.0, 120.0, 160.0, 160.0],
                    ]
                ),
                "scores": torch.tensor([0.9, 0.8, 0.7]),
                "labels": torch.tensor([1, 2, 3]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor(
                    [
                        [10.0, 10.0, 50.0, 50.0],
                        [60.0, 60.0, 100.0, 100.0],
                        [120.0, 120.0, 160.0, 160.0],
                    ]
                ),
                "labels": torch.tensor([1, 2, 3]),
            }
        ]

        evaluator = DetectionEvaluator(num_classes=10)
        evaluator.update(preds, targets)
        result = evaluator.compute()

        # Perfect predictions across multiple classes
        assert result["mAP@0.5"] == 1.0
        # Should have per-class metrics for classes 1, 2, 3
        assert len(result["mAP_per_class"]) >= 3
