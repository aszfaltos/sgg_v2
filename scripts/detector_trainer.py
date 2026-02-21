#!/usr/bin/env python3
"""Train object detectors on VRD dataset using PyTorch Lightning.

This script provides a command-line interface for training object detection models
(Faster R-CNN, EfficientDet) on the Visual Relationship Detection (VRD) dataset.

The training process:
1. Creates a trainable detector (unfrozen, with training mode enabled)
2. Loads VRD dataset with automatic train/val split
3. Wraps detector in DetectorLightningModule for training
4. Configures Lightning Trainer with callbacks and logging
5. Trains the model and saves checkpoints

Usage:
    # Train Faster R-CNN with default settings
    uv run python scripts/detector_trainer.py --detector fasterrcnn

    # Train EfficientDet D2 with custom hyperparameters
    uv run python scripts/detector_trainer.py \\
        --detector efficientdet \\
        --variant d2 \\
        --batch-size 4 \\
        --epochs 20 \\
        --lr 1e-4

    # Resume training from checkpoint
    uv run python scripts/detector_trainer.py \\
        --detector fasterrcnn \\
        --resume checkpoints/detectors/fasterrcnn_resnet50_fpn_v2/last.ckpt

    # Custom checkpoint directory
    uv run python scripts/detector_trainer.py \\
        --detector fasterrcnn \\
        --checkpoint-dir checkpoints/custom_experiment

Example output:
    Loading VRD dataset: datasets/vrd
    Creating detector: fasterrcnn (trainable=True, freeze=False)
    Checkpoint dir: checkpoints/detectors/fasterrcnn_resnet50_fpn_v2

    Training started...
    Epoch 1/20: train/loss=0.850, val/mAP@0.5=0.320
    Epoch 2/20: train/loss=0.720, val/mAP@0.5=0.380
    ...

    Training complete!
    Best checkpoint: checkpoints/detectors/fasterrcnn_resnet50_fpn_v2/best.ckpt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

# Add project root to path if not already there (for direct script execution)
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.modules.detection import SGGEfficientDet, SGGFasterRCNN  # noqa: E402
from src.trainer_lib import (  # noqa: E402
    DetectorLightningModule,
    VRDDetectionDataModule,
    create_aim_logger,
)

class CleanProgressBar(TQDMProgressBar):
    """Progress bar without v_num clutter."""

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


# EfficientDet image sizes by variant
EFFICIENTDET_SIZES = {
    "d0": (512, 512),
    "d1": (640, 640),
    "d2": (768, 768),
    "d3": (896, 896),
    "d4": (1024, 1024),
    "d5": (1280, 1280),
    "d6": (1280, 1280),
    "d7": (1536, 1536),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace with all configuration options.
    """
    parser = argparse.ArgumentParser(
        description="Train object detectors on VRD dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        choices=["fasterrcnn", "efficientdet"],
        help="Detector architecture to train",
    )

    # Detector-specific arguments
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"],
        help="EfficientDet variant (only for --detector=efficientdet, default: d2)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="vrd",
        choices=["vrd"],
        help="Dataset name (default: vrd)",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/vrd",
        help="Path to dataset root directory (default: datasets/vrd)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training and validation (default: 4)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1 = 10%%)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (auto-generated if not provided)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (optional)",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def generate_checkpoint_dir(detector: str, variant: str | None) -> Path:
    """Generate checkpoint directory path based on detector configuration.

    Args:
        detector: Detector name ("fasterrcnn" or "efficientdet").
        variant: Variant name for EfficientDet (None for Faster R-CNN).

    Returns:
        Path to checkpoint directory.

    Example:
        >>> generate_checkpoint_dir("fasterrcnn", None)
        Path("checkpoints/detectors/fasterrcnn_resnet50_fpn_v2")
        >>> generate_checkpoint_dir("efficientdet", "d2")
        Path("checkpoints/detectors/efficientdet_d2")
    """
    base = Path("checkpoints/detectors")

    if detector == "fasterrcnn":
        # Faster R-CNN always uses ResNet-50 FPN V2
        return base / "fasterrcnn_resnet50_fpn_v2"
    elif detector == "efficientdet":
        # Use variant if provided, otherwise default to d2
        variant_str = variant if variant else "d2"
        return base / f"efficientdet_{variant_str}"
    else:
        raise ValueError(f"Unknown detector: {detector}")


def create_detector_for_training(
    detector: str, variant: str | None
) -> SGGFasterRCNN | SGGEfficientDet:
    """Create trainable detector for training.

    Args:
        detector: Detector name ("fasterrcnn" or "efficientdet").
        variant: Variant name for EfficientDet (None for Faster R-CNN).

    Returns:
        Trainable detector instance.

    Raises:
        ValueError: If detector name is invalid.

    Note:
        Detectors are created with:
        - trainable=True: Returns loss dict during training
        - freeze=False: All parameters are trainable
        - pretrained=True: Start from COCO pretrained weights
        - num_classes: VRD-specific (101 for Faster R-CNN, 100 for EfficientDet)
    """
    if detector == "fasterrcnn":
        return SGGFasterRCNN(
            backbone="resnet50",
            num_classes=101,  # VRD: 100 classes + background
            trainable=True,
            freeze=False,
            pretrained=True,
        )
    elif detector == "efficientdet":
        variant_str = variant if variant else "d2"
        return SGGEfficientDet(
            variant=variant_str,  # type: ignore[arg-type]
            num_classes=100,  # VRD: 100 classes (no background)
            trainable=True,
            freeze=False,
            pretrained=True,
            box_loss_weight=50.0,  # Lower than default 50.0 to help classification learn
        )
    else:
        raise ValueError(f"Unknown detector: {detector}")


def create_datamodule(
    dataset: str,
    data_root: str,
    batch_size: int,
    val_split: float,
    target_size: tuple[int, int] | None,
    seed: int,
    detector: str,
) -> VRDDetectionDataModule:
    """Create Lightning DataModule for dataset.

    Args:
        dataset: Dataset name ("vrd").
        data_root: Path to dataset root directory.
        batch_size: Batch size for training and validation.
        val_split: Validation split ratio.
        target_size: Optional (height, width) to resize images to.
        seed: Random seed for reproducible splits.
        detector: Detector type ("fasterrcnn" or "efficientdet").

    Returns:
        Configured DataModule instance.

    Raises:
        ValueError: If dataset name is invalid.
    """
    # Both detectors need 1-indexed labels [1, N]:
    # - Faster R-CNN: torchvision convention (background=0)
    # - EfficientDet: effdet internally does (labels - 1), expects 1-indexed input
    background_class = True

    if dataset == "vrd":
        return VRDDetectionDataModule(
            root=data_root,
            batch_size=batch_size,
            val_split=val_split,
            target_size=target_size,
            seed=seed,
            background_class=background_class,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main(
    detector: str,
    variant: str | None,
    dataset: str,
    data_root: str,
    batch_size: int,
    epochs: int,
    lr: float,
    val_split: float,
    checkpoint_dir: str | None,
    resume: str | None,
    seed: int,
) -> None:
    """Train object detector on VRD dataset.

    Args:
        detector: Detector architecture ("fasterrcnn" or "efficientdet").
        variant: EfficientDet variant (None for Faster R-CNN).
        dataset: Dataset name ("vrd").
        data_root: Path to dataset root directory.
        batch_size: Batch size for training and validation.
        epochs: Number of training epochs.
        lr: Learning rate.
        val_split: Validation split ratio.
        checkpoint_dir: Directory to save checkpoints (auto-generated if None).
        resume: Path to checkpoint to resume from (optional).
        seed: Random seed for reproducibility.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Generate checkpoint directory if not provided
    if checkpoint_dir is None:
        checkpoint_path = generate_checkpoint_dir(detector, variant)
    else:
        checkpoint_path = Path(checkpoint_dir)

    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("Detector Training Configuration")
    print("=" * 60)
    print(f"Detector: {detector}" + (f" ({variant})" if variant else ""))
    print(f"Dataset: {dataset} (root: {data_root})")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Val split: {val_split}")
    print(f"Checkpoint dir: {checkpoint_path}")
    if resume:
        print(f"Resume from: {resume}")
    print(f"Seed: {seed}")
    print("=" * 60)
    print()

    # Create detector
    print(f"Creating {detector} detector...")
    model = create_detector_for_training(detector, variant)
    print(f"Detector created: {model.num_classes} classes")
    print()

    # Determine target size for DataModule (EfficientDet requires specific sizes)
    target_size = None
    if detector == "efficientdet":
        variant_str = variant if variant else "d2"
        target_size = EFFICIENTDET_SIZES[variant_str]
        print(f"EfficientDet target size: {target_size}")

    # Create DataModule
    print(f"Loading {dataset} dataset from {data_root}...")
    datamodule = create_datamodule(
        dataset=dataset,
        data_root=data_root,
        batch_size=batch_size,
        val_split=val_split,
        target_size=target_size,
        seed=seed,
        detector=detector,
    )
    print("DataModule created")
    print()

    # Load class names for debug visualization
    class_names = None
    objects_file = Path(data_root) / "objects.json"
    if objects_file.exists():
        with open(objects_file) as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} class names from {objects_file}")

    # Create Lightning Module
    print("Creating Lightning module...")
    debug_images_dir = checkpoint_path / "debug_images"
    lightning_module = DetectorLightningModule(
        model=model,
        learning_rate=lr,
        weight_decay=1e-4,
        warmup_epochs=5,
        debug_images_dir=debug_images_dir,
        num_debug_images=5,
        class_names=class_names,
        backbone_lr_factor=0.1,  # Backbone at 10% of head LR (already pretrained)
        trainable_backbone_layers=-1,  # Train all backbone layers
    )
    print(f"Lightning module created (debug images: {debug_images_dir})")
    print()

    # Create logger
    experiment_name = f"{detector}_detection"
    run_name = f"{detector}_{variant if variant else 'resnet50'}_bs{batch_size}_lr{lr}"
    logger = create_aim_logger(
        experiment_name=experiment_name,
        run_name=run_name,
        repo_path=".aim",
    )

    # Create callbacks
    callbacks = [
        # Progress bar without v_num
        CleanProgressBar(),
        # Model checkpointing - save best model based on validation mAP
        ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="best",
            monitor="val/mAP@0.5",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="epoch"),
        # Early stopping - stop if validation mAP doesn't improve
        EarlyStopping(
            monitor="val/mAP@0.5",
            patience=5,
            mode="max",
            verbose=True,
        ),
    ]

    # Create Trainer
    print("Creating Lightning Trainer...")
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",  # Auto-detect GPU/CPU
        devices="auto",  # Use all available devices
        logger=logger,
        callbacks=callbacks,
        precision="16-mixed" if torch.cuda.is_available() else "32",  # Mixed precision on GPU
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=8,  # Effective batch size = 4 * 4 = 16
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        enable_model_summary=True,
    )
    print("Trainer created")
    print()

    # Train
    print("Starting training...")
    print("=" * 60)
    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=resume,  # Resume from checkpoint if provided
    )
    print("=" * 60)
    print()

    # Print summary
    print("Training complete!")
    print(f"Best checkpoint: {checkpoint_path / 'best.ckpt'}")
    print(f"Last checkpoint: {checkpoint_path / 'last.ckpt'}")
    print(f"Logs: .aim/{experiment_name}/{run_name}/")
    print()
    print("To view logs with TensorBoard:")
    print("  tensorboard --logdir=.aim")


if __name__ == "__main__":
    args = parse_args()

    main(
        detector=args.detector,
        variant=args.variant,
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        seed=args.seed,
    )
