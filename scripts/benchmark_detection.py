#!/usr/bin/env python3
"""Benchmark detection evaluation on VRD dataset.

This script evaluates object detection performance on the Visual Relationship
Detection (VRD) dataset using COCO-style metrics (mAP, AR).

Usage:
    # Faster R-CNN with ResNet-50
    uv run python scripts/benchmark_detection.py --detector fasterrcnn --split test

    # EfficientDet D2
    uv run python scripts/benchmark_detection.py --detector efficientdet --variant d2 --split test

    # Custom batch size and output file
    uv run python scripts/benchmark_detection.py \\
        --detector fasterrcnn \\
        --backbone resnet50 \\
        --split test \\
        --batch-size 4 \\
        --output results.json

Example output:
    Detection Benchmark Results
    ===========================
    Detector: fasterrcnn-resnet50
    Split: test
    Images: 1000

    Metrics:
    - mAP@0.5:      0.450
    - mAP@0.5:0.95: 0.320
    - AR@100:       0.520

    Results saved to: results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path if not already there (for direct script execution)
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.vrd_detection import VRDDetectionDataset  # noqa: E402
from src.evaluation.detection_metrics import DetectionEvaluator  # noqa: E402
from src.modules.detection import create_detector  # noqa: E402


def collate_fn_pad(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]], list[tuple[int, int]]]:
    """Collate function that pads images to max size in batch.

    Args:
        batch: List of (image, target) tuples from dataset.

    Returns:
        Tuple of:
        - images: (B, C, H_max, W_max) padded tensor
        - targets: List of target dicts (unchanged)
        - original_sizes: List of (H, W) tuples for each image
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Get max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    # Store original sizes for reference
    original_sizes = [(img.shape[1], img.shape[2]) for img in images]

    # Pad all images to max size (pad right and bottom)
    padded_images = []
    for img in images:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # F.pad expects (left, right, top, bottom) for 2D
        padded = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
        padded_images.append(padded)

    # Stack into batch tensor
    batched_images = torch.stack(padded_images, dim=0)

    return batched_images, targets, original_sizes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with defaults applied.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark object detection on VRD dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        choices=["fasterrcnn", "efficientdet"],
        help="Detector architecture",
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Dataset split to evaluate on",
    )

    # Detector-specific arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["resnet50", "resnet101"],
        help="Backbone architecture (for Faster R-CNN)",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["d2", "d3"],
        help="Model variant (for EfficientDet)",
    )

    # Evaluation settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)",
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (default: print to console only)",
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets/vrd",
        help="Path to VRD dataset root (default: datasets/vrd)",
    )

    return parser.parse_args()


def run_inference(
    detector: torch.nn.Module,
    dataset: VRDDetectionDataset,
    device: str,
    batch_size: int = 1,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Run inference on dataset and collect predictions and targets.

    Args:
        detector: SGG detector model.
        dataset: VRD detection dataset.
        device: Device to run inference on.
        batch_size: Number of images per batch.

    Returns:
        Tuple of (predictions, targets) where:
        - predictions: List of dicts with "boxes", "scores", "labels"
        - targets: List of dicts with "boxes", "labels"
    """
    detector.eval()
    detector = detector.to(device)

    # Create DataLoader with padding collate function
    # NUM_WORKERS=0 per project constraints (Python 3.13 multiprocessing issues)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_pad,
    )

    predictions: list[dict[str, Tensor]] = []
    targets: list[dict[str, Tensor]] = []

    # Run inference with progress bar
    with torch.no_grad():
        for images, batch_targets, _original_sizes in tqdm(
            loader, desc="Running inference", unit="batch"
        ):
            # Move batch to device
            images = images.to(device)

            # Run detector on batch
            output = detector(images)

            # Convert SGGDetectorOutput to evaluation format
            # output.boxes/labels/scores are lists (one per image in batch)
            for i in range(len(batch_targets)):
                pred = {
                    "boxes": output.boxes[i].cpu(),
                    "labels": output.labels[i].cpu(),
                    "scores": output.scores[i].cpu(),
                }
                predictions.append(pred)

                # Target is already in correct format
                tgt = {
                    "boxes": batch_targets[i]["boxes"].cpu(),
                    "labels": batch_targets[i]["labels"].cpu(),
                }
                targets.append(tgt)

    return predictions, targets


def format_results(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Format evaluation results into JSON-serializable structure.

    Args:
        metrics: Metrics from DetectionEvaluator.compute()
        config: Configuration dictionary with detector settings

    Returns:
        Dictionary with formatted results including:
        - detector: Detector name and config
        - split: Dataset split
        - metrics: Detection metrics (mAP, AR, etc.)
        - per_class: Per-class mAP scores
        - config: Full configuration
        - timestamp: ISO-format timestamp
    """
    # Extract per-class metrics
    per_class = metrics.pop("mAP_per_class", {})

    result = {
        "detector": config.get("detector", "unknown"),
        "backbone": config.get("backbone"),
        "variant": config.get("variant"),
        "split": config.get("split", "unknown"),
        "num_images": config.get("num_images", 0),
        "metrics": {
            # Main metrics
            "mAP@0.5": metrics.get("mAP@0.5", 0.0),
            "mAP@0.5:0.95": metrics.get("mAP@0.5:0.95", 0.0),
            "mAP@0.75": metrics.get("mAP@0.75", 0.0),
            "AR@1": metrics.get("AR@1", 0.0),
            "AR@10": metrics.get("AR@10", 0.0),
            "AR@100": metrics.get("AR@100", 0.0),
            # Size-specific metrics
            "mAP@small": metrics.get("mAP@small", 0.0),
            "mAP@medium": metrics.get("mAP@medium", 0.0),
            "mAP@large": metrics.get("mAP@large", 0.0),
            "AR@small": metrics.get("AR@small", 0.0),
            "AR@medium": metrics.get("AR@medium", 0.0),
            "AR@large": metrics.get("AR@large", 0.0),
        },
        "per_class": per_class,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    return result


def print_summary(results: dict[str, Any]) -> None:
    """Print formatted summary to console.

    Args:
        results: Formatted results dictionary.
    """
    # Build detector name
    detector_name = results["detector"]
    if results.get("backbone"):
        detector_name = f"{detector_name}-{results['backbone']}"
    elif results.get("variant"):
        detector_name = f"{detector_name}-{results['variant']}"

    print("\nDetection Benchmark Results")
    print("=" * 50)
    print(f"Detector: {detector_name}")
    print(f"Split: {results['split']}")
    print(f"Images: {results['num_images']}")
    print()

    # Print main metrics
    metrics = results["metrics"]
    print("Main Metrics:")
    print(f"  mAP@0.5:      {metrics['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"  mAP@0.75:     {metrics['mAP@0.75']:.4f}")
    print(f"  AR@1:         {metrics['AR@1']:.4f}")
    print(f"  AR@10:        {metrics['AR@10']:.4f}")
    print(f"  AR@100:       {metrics['AR@100']:.4f}")
    print()

    # Print size-specific metrics
    print("Size-Specific Metrics:")
    print(f"  mAP@small:    {metrics['mAP@small']:.4f}")
    print(f"  mAP@medium:   {metrics['mAP@medium']:.4f}")
    print(f"  mAP@large:    {metrics['mAP@large']:.4f}")
    print()

    # Print per-class summary
    per_class = results.get("per_class", {})
    if per_class:
        # Convert string keys back to ints if needed
        per_class_vals = [v for v in per_class.values() if v > 0]
        if per_class_vals:
            print("Per-Class Metrics:")
            print(f"  Classes evaluated: {len(per_class_vals)}")
            print(
                f"  Mean mAP:         {sum(per_class_vals) / len(per_class_vals):.4f}"
            )
            print(f"  Min mAP:          {min(per_class_vals):.4f}")
            print(f"  Max mAP:          {max(per_class_vals):.4f}")
            print()


def main(
    detector: str,
    backbone: str | None,
    variant: str | None,
    split: Literal["train", "test"],
    batch_size: int,
    device: str,
    output: str | None,
    dataset_root: str,
) -> None:
    """Run detection benchmark.

    Args:
        detector: Detector name ("fasterrcnn" or "efficientdet").
        backbone: Backbone name (for Faster R-CNN).
        variant: Variant name (for EfficientDet).
        split: Dataset split ("train" or "test").
        batch_size: Batch size for inference.
        device: Device to run on.
        output: Path to save JSON results (None = print only).
        dataset_root: Path to VRD dataset root.
    """
    # Load dataset
    print(f"Loading VRD {split} dataset from {dataset_root}...")
    dataset = VRDDetectionDataset(root=dataset_root, split=split)
    print(f"Loaded {len(dataset)} images")

    # Create detector with appropriate kwargs per detector type
    print(f"\nCreating detector: {detector}")

    if detector == "fasterrcnn":
        if variant:
            print("Warning: --variant is ignored for fasterrcnn (use --backbone)")
        detector_kwargs: dict[str, Any] = {
            "pretrained": True,
            "freeze": True,
            "min_score": 0.05,
            "max_detections_per_image": 100,
            "nms_thresh": 0.5,
        }
        if backbone:
            detector_kwargs["backbone"] = backbone

    elif detector == "efficientdet":
        if backbone:
            print("Warning: --backbone is ignored for efficientdet (use --variant)")
        detector_kwargs = {
            "pretrained": True,
            "freeze": True,
            "score_thresh": 0.05,
            "max_detections_per_image": 100,
        }
        if variant:
            detector_kwargs["variant"] = variant
    else:
        raise ValueError(f"Unknown detector: {detector}")

    model = create_detector(detector, **detector_kwargs)
    print(f"Detector created: {model.num_classes} classes")

    # Run inference
    print(f"\nRunning inference on {device} (batch_size={batch_size})...")
    predictions, targets = run_inference(
        model, dataset, device=device, batch_size=batch_size
    )

    # Evaluate
    print("\nComputing metrics...")
    evaluator = DetectionEvaluator(num_classes=dataset.num_classes)
    evaluator.update(predictions, targets)
    metrics = evaluator.compute()

    # Format results
    config = {
        "detector": detector,
        "backbone": backbone,
        "variant": variant,
        "split": split,
        "batch_size": batch_size,
        "device": device,
        "num_images": len(dataset),
        "dataset_root": dataset_root,
    }
    results = format_results(metrics, config)

    # Print summary
    print_summary(results)

    # Save to file if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()

    # Validate split (argparse already restricts choices, but cast for type checker)
    split = args.split
    if split not in ("train", "test"):
        raise ValueError(f"Invalid split: {split}")
    split_literal: Literal["train", "test"] = split  # type: ignore[assignment]

    main(
        detector=args.detector,
        backbone=args.backbone,
        variant=args.variant,
        split=split_literal,
        batch_size=args.batch_size,
        device=args.device,
        output=args.output,
        dataset_root=args.dataset_root,
    )
