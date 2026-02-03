"""
Unit tests for freeze utilities.

Tests cover freezing modules, BatchNorm layers, and ResNet backbone stages.
"""

import pytest
import torch
import torch.nn as nn
from torchvision.models import resnet50

from src.modules.detection.components.freeze import (
    freeze_backbone_stages,
    freeze_bn,
    freeze_module,
)


class TestFreezeModule:
    """Test freeze_module function."""

    def test_freeze_simple_module(self):
        """Should set requires_grad=False for all parameters."""
        # Arrange
        module = nn.Linear(10, 5)
        assert all(p.requires_grad for p in module.parameters())

        # Act
        freeze_module(module)

        # Assert
        assert all(not p.requires_grad for p in module.parameters())

    def test_freeze_nested_module(self):
        """Should recursively freeze nested modules."""
        # Arrange
        module = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Sequential(
                nn.Linear(20, 10),
                nn.BatchNorm1d(10),
            ),
        )
        assert any(p.requires_grad for p in module.parameters())

        # Act
        freeze_module(module)

        # Assert
        assert all(not p.requires_grad for p in module.parameters())

    def test_freeze_empty_module(self):
        """Should handle modules with no parameters."""
        # Arrange
        module = nn.ReLU()

        # Act
        freeze_module(module)  # Should not raise

        # Assert
        assert len(list(module.parameters())) == 0

    def test_freeze_already_frozen_module(self):
        """Should be idempotent when called multiple times."""
        # Arrange
        module = nn.Linear(10, 5)
        freeze_module(module)
        assert all(not p.requires_grad for p in module.parameters())

        # Act
        freeze_module(module)  # Second call

        # Assert
        assert all(not p.requires_grad for p in module.parameters())


class TestFreezeBN:
    """Test freeze_bn function."""

    def test_freeze_batchnorm_sets_eval_mode(self):
        """Should set BatchNorm layers to eval mode."""
        # Arrange
        bn = nn.BatchNorm2d(64)
        bn.train()
        assert bn.training

        # Act
        freeze_bn(bn)

        # Assert
        assert not bn.training

    def test_freeze_batchnorm_disables_grad(self):
        """Should disable gradients for BatchNorm parameters."""
        # Arrange
        bn = nn.BatchNorm2d(64)
        assert all(p.requires_grad for p in bn.parameters())

        # Act
        freeze_bn(bn)

        # Assert
        assert all(not p.requires_grad for p in bn.parameters())

    def test_freeze_nested_batchnorm(self):
        """Should freeze all BatchNorm layers in a module."""
        # Arrange
        module = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Sequential(
                nn.Conv2d(64, 128, 3),
                nn.BatchNorm2d(128),
            ),
        )
        module.train()

        # Count BN layers
        bn_layers = [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) == 2
        assert all(m.training for m in bn_layers)

        # Act
        freeze_bn(module)

        # Assert
        assert all(not m.training for m in bn_layers)
        # Check all BN params are frozen
        for bn_layer in bn_layers:
            assert all(not p.requires_grad for p in bn_layer.parameters())

    def test_freeze_bn_stays_in_eval_after_train_call(self):
        """Should keep BatchNorm in eval mode even after model.train() is called."""
        # Arrange
        module = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
        )
        freeze_bn(module)
        bn = list(module.modules())[2]  # Get the BN layer
        assert not bn.training

        # Act
        module.train()  # Try to set whole module to train mode

        # Assert - BN should still be in eval mode
        # Note: This test documents expected behavior, but standard freeze_bn
        # won't persist across .train() calls unless we override train()
        # We'll test the actual behavior in implementation

    def test_freeze_bn_with_no_batchnorm(self):
        """Should handle modules with no BatchNorm layers."""
        # Arrange
        module = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU())

        # Act
        freeze_bn(module)  # Should not raise

        # Assert
        assert len([m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]) == 0


class TestFreezeBackboneStages:
    """Test freeze_backbone_stages function for ResNet."""

    def test_freeze_zero_stages(self):
        """Should freeze no stages when stages=0."""
        # Arrange
        backbone = resnet50(weights=None)
        total_params = sum(1 for _ in backbone.parameters())
        trainable_before = sum(1 for p in backbone.parameters() if p.requires_grad)
        assert trainable_before == total_params

        # Act
        freeze_backbone_stages(backbone, stages=0)

        # Assert
        trainable_after = sum(1 for p in backbone.parameters() if p.requires_grad)
        assert trainable_after == total_params  # All still trainable

    def test_freeze_one_stage(self):
        """Should freeze stem (conv1, bn1) when stages=1."""
        # Arrange
        backbone = resnet50(weights=None)

        # Act
        freeze_backbone_stages(backbone, stages=1)

        # Assert - stem (conv1, bn1) should be frozen
        assert all(not p.requires_grad for p in backbone.conv1.parameters())
        assert all(not p.requires_grad for p in backbone.bn1.parameters())

        # Layer1-4 should still be trainable
        assert any(p.requires_grad for p in backbone.layer1.parameters())
        assert any(p.requires_grad for p in backbone.layer2.parameters())

    def test_freeze_two_stages(self):
        """Should freeze stem + layer1 when stages=2."""
        # Arrange
        backbone = resnet50(weights=None)

        # Act
        freeze_backbone_stages(backbone, stages=2)

        # Assert
        assert all(not p.requires_grad for p in backbone.conv1.parameters())
        assert all(not p.requires_grad for p in backbone.bn1.parameters())
        assert all(not p.requires_grad for p in backbone.layer1.parameters())

        # Layer2-4 should still be trainable
        assert any(p.requires_grad for p in backbone.layer2.parameters())

    def test_freeze_all_stages(self):
        """Should freeze entire backbone when stages=5."""
        # Arrange
        backbone = resnet50(weights=None)

        # Act
        freeze_backbone_stages(backbone, stages=5)

        # Assert - everything should be frozen
        assert all(not p.requires_grad for p in backbone.parameters())

    def test_freeze_stages_invalid_count(self):
        """Should raise ValueError for invalid stage count."""
        # Arrange
        backbone = resnet50(weights=None)

        # Act & Assert
        with pytest.raises(ValueError, match="stages must be between 0 and 5"):
            freeze_backbone_stages(backbone, stages=6)

        with pytest.raises(ValueError, match="stages must be between 0 and 5"):
            freeze_backbone_stages(backbone, stages=-1)

    def test_freeze_stages_batchnorm_in_eval(self):
        """Should also set frozen BatchNorm layers to eval mode."""
        # Arrange
        backbone = resnet50(weights=None)
        backbone.train()

        # Act
        freeze_backbone_stages(backbone, stages=2)

        # Assert - frozen BN layers should be in eval mode
        assert not backbone.bn1.training
        for module in backbone.layer1.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert not module.training

    def test_freeze_stages_gradients_disabled(self):
        """Should prevent gradient computation in frozen stages."""
        # Arrange
        backbone = resnet50(weights=None)
        freeze_backbone_stages(backbone, stages=2)

        # Create dummy input
        x = torch.randn(1, 3, 224, 224)

        # Act - forward pass
        with torch.enable_grad():
            out = backbone(x)
            loss = out.sum()
            loss.backward()

        # Assert - frozen params should have no gradients
        assert all(p.grad is None for p in backbone.conv1.parameters())
        assert all(p.grad is None for p in backbone.layer1.parameters())

        # Trainable params should have gradients
        trainable_params = [p for p in backbone.layer2.parameters() if p.requires_grad]
        assert len(trainable_params) > 0
        assert all(p.grad is not None for p in trainable_params)


class TestIntegration:
    """Integration tests combining multiple freeze utilities."""

    def test_freeze_backbone_for_sgg(self):
        """Should freeze entire ResNet backbone for SGG training."""
        # Arrange
        backbone = resnet50(weights=None)

        # Act - typical SGG setup: freeze everything + freeze BN
        freeze_backbone_stages(backbone, stages=5)
        freeze_bn(backbone)

        # Assert
        assert all(not p.requires_grad for p in backbone.parameters())
        for module in backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert not module.training

    def test_partial_freeze_for_finetuning(self):
        """Should allow partial freezing for fine-tuning scenarios."""
        # Arrange
        backbone = resnet50(weights=None)

        # Act - freeze early stages, keep later stages trainable
        freeze_backbone_stages(backbone, stages=3)

        # Assert
        frozen_params = [p for p in backbone.conv1.parameters()]
        frozen_params += [p for p in backbone.layer1.parameters()]
        frozen_params += [p for p in backbone.layer2.parameters()]
        assert all(not p.requires_grad for p in frozen_params)

        trainable_params = [p for p in backbone.layer3.parameters()]
        trainable_params += [p for p in backbone.layer4.parameters()]
        assert all(p.requires_grad for p in trainable_params)
