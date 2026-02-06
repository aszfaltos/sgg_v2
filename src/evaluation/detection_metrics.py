"""Detection evaluation metrics using torchmetrics.

This module provides a clean wrapper around torchmetrics.detection.MeanAveragePrecision
for computing COCO-style detection metrics (mAP, AR).

Example usage:
    >>> evaluator = DetectionEvaluator(num_classes=80)
    >>> for batch in dataloader:
    ...     predictions = model(batch["images"])
    ...     evaluator.update(predictions, batch["targets"])
    >>> results = evaluator.compute()
    >>> print(f"mAP@0.5: {results['mAP@0.5']:.3f}")
"""

from typing import Any

import torch
from torchmetrics.detection import MeanAveragePrecision


class DetectionEvaluator:
    """Evaluator for object detection metrics.

    Computes COCO-style metrics (mAP@0.5, mAP@0.5:0.95, AR@100) using torchmetrics.
    Handles empty predictions/targets gracefully and provides per-class metrics.

    Attributes:
        num_classes: Number of object classes (including background at index 0).
        iou_thresholds: IoU thresholds for mAP computation. If None, uses COCO defaults
            (0.50 to 0.95 in steps of 0.05).

    Example:
        >>> evaluator = DetectionEvaluator(num_classes=80)
        >>>
        >>> # Accumulate predictions over dataset
        >>> for images, targets in dataloader:
        ...     predictions = model(images)
        ...     evaluator.update(predictions, targets)
        >>>
        >>> # Compute final metrics
        >>> metrics = evaluator.compute()
        >>> print(f"mAP@0.5: {metrics['mAP@0.5']:.3f}")
        >>>
        >>> # Reset for new evaluation
        >>> evaluator.reset()
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: list[float] | None = None,
    ) -> None:
        """Initialize the detection evaluator.

        Args:
            num_classes: Number of object classes (must be positive). Includes background
                class at index 0.
            iou_thresholds: List of IoU thresholds for mAP computation. If None, uses
                COCO defaults: [0.50, 0.55, ..., 0.95] (10 thresholds).

        Raises:
            ValueError: If num_classes is not positive.

        Example:
            >>> # Standard COCO evaluation
            >>> evaluator = DetectionEvaluator(num_classes=80)
            >>>
            >>> # Custom IoU thresholds (e.g., only 0.5 and 0.75)
            >>> evaluator = DetectionEvaluator(
            ...     num_classes=80,
            ...     iou_thresholds=[0.5, 0.75]
            ... )
        """
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds

        # Initialize torchmetrics metric
        # box_format='xyxy' matches PyTorch convention: [x1, y1, x2, y2]
        # class_metrics=True enables per-class mAP computation
        self._metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=iou_thresholds,
            class_metrics=True,
        )

        # Track whether any data has been accumulated
        self._has_data = False

    def update(
        self,
        predictions: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        """Update metric state with predictions and targets.

        Accumulates predictions and targets for later metric computation. Can be called
        multiple times to process dataset in batches.

        Args:
            predictions: List of prediction dicts, one per image. Each dict must contain:
                - "boxes": Tensor[N, 4] in xyxy format [x1, y1, x2, y2]
                - "scores": Tensor[N] with detection confidence scores
                - "labels": Tensor[N] with predicted class labels (0-indexed)
            targets: List of target dicts, one per image. Each dict must contain:
                - "boxes": Tensor[M, 4] in xyxy format [x1, y1, x2, y2]
                - "labels": Tensor[M] with ground truth class labels (0-indexed)

        Raises:
            KeyError: If required keys are missing from predictions or targets.
            ValueError: If tensor shapes are invalid.

        Example:
            >>> predictions = [
            ...     {
            ...         "boxes": torch.tensor([[10., 10., 50., 50.]]),
            ...         "scores": torch.tensor([0.9]),
            ...         "labels": torch.tensor([1]),
            ...     }
            ... ]
            >>> targets = [
            ...     {
            ...         "boxes": torch.tensor([[10., 10., 50., 50.]]),
            ...         "labels": torch.tensor([1]),
            ...     }
            ... ]
            >>> evaluator.update(predictions, targets)
        """
        # Validate format
        self._validate_format(predictions, targets)

        # Update underlying metric
        self._metric.update(predictions, targets)
        self._has_data = True

    def compute(self) -> dict[str, Any]:
        """Compute detection metrics from accumulated predictions and targets.

        Computes COCO-style detection metrics after all data has been accumulated via
        update(). Returns standard metrics plus per-class metrics.

        Returns:
            Dictionary containing:
                - "mAP@0.5": mAP at IoU threshold 0.5
                - "mAP@0.5:0.95": mAP averaged over IoU thresholds 0.5 to 0.95
                - "AR@100": Average Recall with max 100 detections per image
                - "mAP_per_class": Dict mapping class index to per-class mAP
                - Additional COCO metrics (map_75, mar_1, mar_10, etc.)

        Raises:
            RuntimeError: If compute() is called before any update().
            ValueError: If no valid predictions/targets were accumulated.

        Example:
            >>> evaluator.update(predictions, targets)
            >>> metrics = evaluator.compute()
            >>> print(f"mAP@0.5: {metrics['mAP@0.5']:.3f}")
            >>> print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.3f}")
            >>> print(f"AR@100: {metrics['AR@100']:.3f}")
        """
        if not self._has_data:
            raise RuntimeError(
                "compute() called before any data was accumulated. "
                "Call update() at least once before compute()."
            )

        # Compute underlying torchmetrics result
        raw_result = self._metric.compute()

        # Convert to user-friendly format
        result = self._format_result(raw_result)

        return result

    def reset(self) -> None:
        """Reset metric state to start a new evaluation.

        Clears all accumulated predictions and targets. After reset(), compute() cannot
        be called until update() is called again.

        Example:
            >>> evaluator.update(predictions1, targets1)
            >>> metrics1 = evaluator.compute()
            >>>
            >>> # Start new evaluation
            >>> evaluator.reset()
            >>> evaluator.update(predictions2, targets2)
            >>> metrics2 = evaluator.compute()
        """
        self._metric.reset()
        self._has_data = False

    def _validate_format(
        self,
        predictions: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        """Validate prediction and target format.

        Args:
            predictions: List of prediction dicts.
            targets: List of target dicts.

        Raises:
            ValueError: If lengths don't match.
            KeyError: If required keys are missing.
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) must match "
                f"number of targets ({len(targets)})"
            )

        # Validate each prediction
        for i, pred in enumerate(predictions):
            required_pred_keys = {"boxes", "scores", "labels"}
            missing_keys = required_pred_keys - pred.keys()
            if missing_keys:
                raise KeyError(
                    f"Prediction {i} missing required keys: {missing_keys}. "
                    f"Required: {required_pred_keys}"
                )

        # Validate each target
        for i, target in enumerate(targets):
            required_target_keys = {"boxes", "labels"}
            missing_keys = required_target_keys - target.keys()
            if missing_keys:
                raise KeyError(
                    f"Target {i} missing required keys: {missing_keys}. "
                    f"Required: {required_target_keys}"
                )

    def _format_result(self, raw_result: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Format torchmetrics output to user-friendly format.

        Args:
            raw_result: Raw output from torchmetrics.detection.MeanAveragePrecision.

        Returns:
            Formatted dictionary with standard metric names and float values.
        """

        # Convert tensors to float, handle -1 (undefined) values
        def to_float(tensor: torch.Tensor) -> float:
            """Convert tensor to float, treating -1 as 0.0 (undefined)."""
            val = tensor.item()
            return 0.0 if val < 0 else val

        result: dict[str, Any] = {
            # Standard COCO metrics (renamed for clarity)
            "mAP@0.5": to_float(raw_result["map_50"]),
            "mAP@0.5:0.95": to_float(raw_result["map"]),
            "mAP@0.75": to_float(raw_result["map_75"]),
            "AR@1": to_float(raw_result["mar_1"]),
            "AR@10": to_float(raw_result["mar_10"]),
            "AR@100": to_float(raw_result["mar_100"]),
            # Size-specific metrics
            "mAP@small": to_float(raw_result["map_small"]),
            "mAP@medium": to_float(raw_result["map_medium"]),
            "mAP@large": to_float(raw_result["map_large"]),
            "AR@small": to_float(raw_result["mar_small"]),
            "AR@medium": to_float(raw_result["mar_medium"]),
            "AR@large": to_float(raw_result["mar_large"]),
        }

        # Per-class metrics
        if "map_per_class" in raw_result and raw_result["map_per_class"] is not None:
            classes = raw_result["classes"]
            map_per_class = raw_result["map_per_class"]

            # Handle both 0-d (single class) and 1-d (multiple classes) tensors
            if classes.dim() == 0:
                # Single class case
                result["mAP_per_class"] = {int(classes.item()): to_float(map_per_class)}
            else:
                # Multiple classes case
                result["mAP_per_class"] = {
                    int(classes[i].item()): to_float(map_per_class[i])
                    for i in range(len(classes))
                }
        else:
            result["mAP_per_class"] = {}

        return result
