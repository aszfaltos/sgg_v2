"""Detection Lightning Module for training object detectors.

This module provides a PyTorch Lightning wrapper for training object detection models
(Faster R-CNN, EfficientDet) with standard detection losses and COCO-style mAP evaluation.

Example:
    >>> from src.modules.detection import SGGFasterRCNN
    >>> from src.trainer_lib.lightning_modules import DetectorLightningModule
    >>>
    >>> # Create trainable detector
    >>> detector = SGGFasterRCNN(num_classes=100, trainable=True, freeze=False)
    >>>
    >>> # Wrap in Lightning module
    >>> module = DetectorLightningModule(
    ...     model=detector,
    ...     learning_rate=1e-4,
    ...     scheduler="cosine",
    ... )
    >>>
    >>> # Train with Lightning Trainer
    >>> trainer = Trainer(max_epochs=10)
    >>> trainer.fit(module, datamodule=datamodule)
"""

from pathlib import Path
from typing import Any, Union, cast

import torch
from PIL import Image, ImageDraw
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.evaluation.detection_metrics import DetectionEvaluator
from src.modules.detection.base import SGGDetector


class DetectorLightningModule(LightningModule):
    """Lightning module for training object detectors.

    Handles training loop with loss computation, validation with mAP metrics,
    and optimizer/scheduler configuration. Compatible with both Faster R-CNN
    and EfficientDet architectures.

    Args:
        model: Object detection model (SGGFasterRCNN or SGGEfficientDet) in
            trainable mode (trainable=True, freeze=False).
        learning_rate: Base learning rate for optimizer. Default: 1e-4.
        weight_decay: L2 weight regularization. Default: 1e-4.
        warmup_epochs: Number of epochs for linear warmup (1% to 100% of base LR).

    Attributes:
        model: The wrapped detection model.
        learning_rate: Base learning rate.
        weight_decay: Weight decay coefficient.
        warmup_epochs: Warmup duration.
        val_predictions: Cache of validation predictions (cleared each epoch).
        val_targets: Cache of validation targets (cleared each epoch).

    Example:
        >>> # Setup detector and Lightning module
        >>> detector = SGGFasterRCNN(num_classes=100, trainable=True)
        >>> module = DetectorLightningModule(model=detector, learning_rate=1e-4)
        >>>
        >>> # Train with Lightning
        >>> trainer = Trainer(max_epochs=10, accelerator="gpu")
        >>> trainer.fit(module, train_dataloader, val_dataloader)
        >>>
        >>> # Validation metrics are logged automatically:
        >>> # - train/loss (total loss)
        >>> # - train/loss_classifier
        >>> # - train/loss_box_reg
        >>> # - val/mAP@0.5
        >>> # - val/mAP@0.5:0.95
        >>> # - val/AR@100

    Note:
        The model must be in trainable mode (trainable=True) and unfrozen (freeze=False).
        Training uses model.forward(images, targets) which returns a loss dict.
        Validation uses model.predict(images) which returns SGGDetectorOutput.
    """

    def __init__(
        self,
        model: SGGDetector,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        debug_images_dir: str | Path | None = None,
        num_debug_images: int = 5,
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize the detector Lightning module.

        Args:
            model: Detection model (SGGFasterRCNN or SGGEfficientDet).
            learning_rate: Base learning rate.
            weight_decay: L2 weight regularization.
            warmup_epochs: Number of warmup epochs (linear warmup then cosine decay).
            debug_images_dir: Directory to save debug visualization images. If None,
                no debug images are saved.
            num_debug_images: Number of debug images to save per validation epoch.
            class_names: List of class names for debug visualization. If None, shows
                numeric labels. Index 0 = first class (0-indexed for predictions,
                but ground truth labels are 1-indexed so subtract 1).
        """
        super().__init__()

        # Save hyperparameters (except model - too large for logging)
        self.save_hyperparameters(ignore=["model"])

        self.model: SGGDetector = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # Debug visualization
        self.debug_images_dir = Path(debug_images_dir) if debug_images_dir else None
        self.num_debug_images = num_debug_images
        self.class_names = class_names
        self._debug_images_saved = 0  # Counter for current epoch

        # Validation accumulation
        self.val_predictions: list[dict[str, Tensor]] = []
        self.val_targets: list[dict[str, Tensor]] = []

    def training_step(
        self, batch: tuple[Tensor, list[dict[str, Any]]], batch_idx: int
    ) -> Tensor:
        """Training step: forward pass returns loss dict, sum all losses.

        Args:
            batch: Tuple of (images, targets).
                - images: (B, 3, H, W) tensor of RGB images in [0, 1]
                - targets: List of dicts with "boxes" (N, 4) and "labels" (N,)
            batch_idx: Index of batch in epoch.

        Returns:
            Total loss (sum of all loss components).

        Note:
            Model must be in train mode. Loss components are logged individually.
        """
        images, targets = batch

        # Ensure model is in train mode
        self.model.train()

        # Forward pass returns loss dict
        loss_dict = self.model(images, targets)

        # Get total loss:
        # - EfficientDet returns {"loss": total, "class_loss": ..., "box_loss": ...}
        # - Faster R-CNN returns {"loss_classifier": ..., "loss_box_reg": ..., ...}
        if "loss" in loss_dict:
            # EfficientDet: use provided total directly
            total_loss = cast(Tensor, loss_dict["loss"])
        else:
            # Faster R-CNN: sum all loss components
            losses = list(loss_dict.values())
            total_loss = cast(Tensor, sum(losses))

        # Get batch size for logging
        batch_size = images.shape[0]

        # Log individual loss components (skip "loss" - logged separately as total)
        for loss_name, loss_value in loss_dict.items():
            if loss_name == "loss":
                continue
            self.log(
                f"train/{loss_name}",
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        # Log total loss
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Log learning rate
        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        return total_loss

    def validation_step(
        self, batch: tuple[Tensor, list[dict[str, Any]]], batch_idx: int
    ) -> None:
        """Validation step: run inference and accumulate predictions/targets.

        Args:
            batch: Tuple of (images, targets).
                - images: (B, 3, H, W) tensor of RGB images in [0, 1]
                - targets: List of dicts with "boxes" (N, 4) and "labels" (N,)
            batch_idx: Index of batch in epoch.

        Note:
            Model is put in eval mode. Uses predict() for inference.
            Predictions and targets are accumulated for mAP computation at epoch end.
        """
        images, targets = batch

        # Ensure model is in eval mode
        self.model.eval()

        # Use predict() for inference
        with torch.no_grad():
            output = self.model.predict(images)

        # Convert SGGDetectorOutput to list of dicts for evaluator
        predictions = [
            {
                "boxes": output.boxes[i],
                "labels": output.labels[i],
                "scores": output.scores[i],
            }
            for i in range(len(output))
        ]

        # Accumulate predictions and targets
        self.val_predictions.extend(predictions)
        self.val_targets.extend(targets)

        # Save debug images (first N images per epoch)
        if self.debug_images_dir and self._debug_images_saved < self.num_debug_images:
            self._save_debug_images(images, predictions, targets)

    def _save_debug_images(
        self,
        images: Tensor,
        predictions: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]],
    ) -> None:
        """Save debug visualization images with predictions and ground truth.

        Predictions are drawn in green, ground truth in red.

        Args:
            images: Batch of images (B, 3, H, W) normalized.
            predictions: List of prediction dicts with boxes, labels, scores.
            targets: List of target dicts with boxes, labels.
        """
        assert self.debug_images_dir is not None

        # Create output directory
        epoch = self.current_epoch
        output_dir = self.debug_images_dir / f"epoch_{epoch:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ImageNet denormalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)

        for i in range(len(images)):
            if self._debug_images_saved >= self.num_debug_images:
                break

            # Denormalize image
            img = images[i] * std[0] + mean[0]  # (3, H, W)
            img = img.clamp(0, 1)

            # Convert to PIL
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil_img)

            # Draw ground truth boxes (red) - text inside, top-left
            gt_boxes = targets[i]["boxes"].cpu()
            gt_labels = targets[i]["labels"].cpu()
            for box, label in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                # GT labels are 1-indexed, convert to 0-indexed for class_names lookup
                label_idx = int(label) - 1
                if self.class_names and 0 <= label_idx < len(self.class_names):
                    label_str = self.class_names[label_idx]
                else:
                    label_str = str(int(label))
                draw.text((x1 + 2, y1 + 2), label_str, fill="red")

            # Draw predicted boxes (green) - text inside, bottom-left
            pred_boxes = predictions[i]["boxes"].cpu()
            pred_labels = predictions[i]["labels"].cpu()
            pred_scores = predictions[i]["scores"].cpu()
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if score < 0.3:  # Skip low confidence
                    continue
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                # Pred labels are 0-indexed
                label_idx = int(label)
                if self.class_names and 0 <= label_idx < len(self.class_names):
                    label_str = f"{self.class_names[label_idx]} {score:.2f}"
                else:
                    label_str = f"{int(label)} {score:.2f}"
                draw.text((x1 + 2, y2 - 12), label_str, fill="green")

            # Save
            save_path = output_dir / f"img_{self._debug_images_saved:03d}.png"
            pil_img.save(save_path)
            self._debug_images_saved += 1

    def on_validation_epoch_start(self) -> None:
        """Reset debug image counter at start of validation epoch."""
        self._debug_images_saved = 0

    def on_validation_epoch_end(self) -> None:
        """Compute mAP metrics at end of validation epoch.

        Uses DetectionEvaluator to compute COCO-style metrics (mAP@0.5, mAP@0.5:0.95,
        AR@100) from accumulated predictions and targets.

        Note:
            Clears prediction/target cache after computing metrics.
        """
        if len(self.val_predictions) == 0:
            # No validation data - skip metrics
            return

        # Get num_classes - use cast to tell mypy we know it's an int
        # (SGGDetector defines it as an int property)
        num_classes = cast(int, self.model.num_classes)

        # Create evaluator
        evaluator = DetectionEvaluator(num_classes=num_classes)

        # Update with accumulated predictions/targets
        evaluator.update(self.val_predictions, self.val_targets)

        # Compute metrics
        metrics = evaluator.compute()

        # Log key metrics
        self.log("val/mAP@0.5", metrics["mAP@0.5"], prog_bar=True, logger=True)
        self.log(
            "val/mAP@0.5:0.95", metrics["mAP@0.5:0.95"], prog_bar=True, logger=True
        )
        self.log("val/AR@100", metrics["AR@100"], prog_bar=False, logger=True)

        # Log additional metrics (if available)
        if "mAP@0.75" in metrics:
            self.log("val/mAP@0.75", metrics["mAP@0.75"], prog_bar=False, logger=True)
        if "AR@1" in metrics:
            self.log("val/AR@1", metrics["AR@1"], prog_bar=False, logger=True)
        if "AR@10" in metrics:
            self.log("val/AR@10", metrics["AR@10"], prog_bar=False, logger=True)

        # Clear cache
        self.val_predictions = []
        self.val_targets = []

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> dict[str, Union[Optimizer, dict[str, Any]]]:
        """Configure optimizer and learning rate scheduler.

        Uses linear warmup (1% to 100% over warmup_epochs) followed by
        cosine annealing to 0 for the remaining epochs.

        Returns:
            Dict with optimizer and lr_scheduler config.
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Get max_epochs from trainer (default 100 if not available)
        trainer = getattr(self, "_trainer", None)
        max_epochs = (
            getattr(trainer, "max_epochs", 100) if trainer is not None else 100
        )

        # Linear warmup: 1% -> 100% of base LR over warmup_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )

        # Cosine decay: 100% -> 0% over remaining epochs
        cosine_epochs = max(1, max_epochs - self.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=0,
        )

        # Chain: warmup -> cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
