"""Example: Training a detector with PyTorch Lightning.

This example demonstrates how to use DetectorLightningModule to train
an object detector (Faster R-CNN or EfficientDet) on a custom dataset.

Usage:
    python examples/train_detector_example.py
"""

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from src.modules.detection import SGGFasterRCNN
from src.trainer_lib import DetectorLightningModule


class DummyDetectionDataset(Dataset):
    """Dummy detection dataset for demonstration."""

    def __init__(self, num_samples: int = 100, num_classes: int = 10):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        # Random image (3, H, W)
        image = torch.rand(3, 512, 512)

        # Random boxes and labels
        num_objects = torch.randint(1, 6, (1,)).item()
        boxes = torch.rand(num_objects, 4) * 512
        # Ensure x2 > x1 and y2 > y1
        boxes[:, 2:] = boxes[:, :2] + torch.rand(num_objects, 2) * 100
        labels = torch.randint(1, self.num_classes, (num_objects,))

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target


def collate_fn(batch: list) -> tuple[torch.Tensor, list[dict]]:
    """Collate function for detection batches.

    Args:
        batch: List of (image, target) tuples.

    Returns:
        Tuple of (images, targets) where images is a batched tensor
        and targets is a list of dicts.
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def main():
    """Main training loop."""
    print("Creating detector...")

    # Create trainable detector
    detector = SGGFasterRCNN(
        backbone="resnet50",
        pretrained=False,  # For demonstration, no pretrained weights
        freeze=False,  # Trainable
        trainable=True,  # Enable training mode
        num_classes=10,
    )

    # Wrap in Lightning module
    module = DetectorLightningModule(
        model=detector,
        learning_rate=1e-4,
        weight_decay=1e-4,
        scheduler="cosine",
        warmup_epochs=1,
    )

    print("Creating datasets...")

    # Create datasets
    train_dataset = DummyDetectionDataset(num_samples=100, num_classes=10)
    val_dataset = DummyDetectionDataset(num_samples=20, num_classes=10)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Don't use multiprocessing with Python 3.13
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print("Creating trainer...")

    # Create Lightning trainer
    trainer = Trainer(
        max_epochs=2,
        accelerator="auto",  # Use GPU if available
        devices=1,
        log_every_n_steps=10,
        # Disable some features for demonstration
        enable_checkpointing=False,
        logger=False,
    )

    print("Starting training...")

    # Train the model
    trainer.fit(module, train_loader, val_loader)

    print("Training complete!")

    # Metrics are logged during validation
    # In a real setup, use a logger like AimLogger to track metrics


if __name__ == "__main__":
    main()
