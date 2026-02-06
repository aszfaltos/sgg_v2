"""Integration tests for DetectorLightningModule with actual detector models.

These tests verify that the Lightning module works correctly with real SGGFasterRCNN
and SGGEfficientDet models (not mocks).
"""

import pytest
import torch

from src.modules.detection import SGGEfficientDet, SGGFasterRCNN
from src.trainer_lib.lightning_modules.detector import DetectorLightningModule


class TestDetectorLightningModuleIntegration:
    """Integration tests with real detector models."""

    @pytest.mark.parametrize("backbone", ["resnet50"])
    def test_faster_rcnn_training_step(self, backbone):
        """Should work with trainable Faster R-CNN."""
        # Arrange - create trainable detector
        detector = SGGFasterRCNN(
            backbone=backbone,
            pretrained=False,
            freeze=False,
            trainable=True,
            num_classes=10,
        )

        module = DetectorLightningModule(
            model=detector,
            learning_rate=1e-4,
        )

        # Create small batch
        images = torch.rand(1, 3, 512, 512)
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
                "labels": torch.tensor([1]),
            }
        ]
        batch = (images, targets)

        # Act
        loss = module.training_step(batch, batch_idx=0)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert loss.requires_grad

    @pytest.mark.parametrize("backbone", ["resnet50"])
    def test_faster_rcnn_validation_step(self, backbone):
        """Should work with frozen Faster R-CNN."""
        # Arrange - create frozen detector
        detector = SGGFasterRCNN(
            backbone=backbone,
            pretrained=False,
            freeze=True,
            trainable=False,
            num_classes=10,
        )

        module = DetectorLightningModule(
            model=detector,
            learning_rate=1e-4,
        )

        # Create small batch
        images = torch.rand(1, 3, 512, 512)
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
                "labels": torch.tensor([1]),
            }
        ]
        batch = (images, targets)

        # Act
        module.validation_step(batch, batch_idx=0)

        # Assert - predictions and targets accumulated
        assert len(module.val_predictions) == 1
        assert len(module.val_targets) == 1
        assert "boxes" in module.val_predictions[0]
        assert "labels" in module.val_predictions[0]
        assert "scores" in module.val_predictions[0]

    @pytest.mark.parametrize("variant", ["d0"])
    def test_efficientdet_training_step(self, variant):
        """Should work with trainable EfficientDet."""
        # Arrange - create trainable detector
        detector = SGGEfficientDet(
            variant=variant,
            pretrained=False,
            freeze=False,
            trainable=True,
            num_classes=10,
        )

        module = DetectorLightningModule(
            model=detector,
            learning_rate=1e-4,
        )

        # Get required image size for variant
        image_size = detector._image_size

        # Create batch with proper EfficientDet target format
        images = torch.rand(1, 3, image_size, image_size)
        targets = [
            {
                "bbox": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
                "cls": torch.tensor([1]),
                "img_scale": torch.tensor(1.0),
                "img_size": torch.tensor([image_size, image_size]),
            }
        ]
        batch = (images, targets)

        # Act
        loss = module.training_step(batch, batch_idx=0)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_configure_optimizers_with_real_model(self):
        """Should configure optimizers with real model parameters."""
        # Arrange
        detector = SGGFasterRCNN(
            backbone="resnet50",
            pretrained=False,
            freeze=False,
            trainable=True,
            num_classes=10,
        )

        module = DetectorLightningModule(
            model=detector,
            learning_rate=1e-4,
            weight_decay=1e-3,
        )

        # Act
        config = module.configure_optimizers()

        # Assert
        assert "optimizer" in config
        assert "lr_scheduler" in config
        optimizer = config["optimizer"]
        assert optimizer.__class__.__name__ == "AdamW"
        assert len(list(optimizer.param_groups)) > 0
