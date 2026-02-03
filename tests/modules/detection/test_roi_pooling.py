"""Test ROI pooling component.

Tests for the ROI pooling wrapper that uses torchvision's MultiScaleRoIAlign
to extract fixed-size features from detected boxes across FPN levels.
"""

import pytest
import torch
from torch import nn


class TestROIPooler:
    """Test suite for ROIPooler component."""

    def test_can_import_roi_pooler(self) -> None:
        """Test that ROIPooler can be imported."""
        from src.modules.detection.components.roi_pooling import ROIPooler

        assert ROIPooler is not None

    def test_roi_pooler_initialization(self) -> None:
        """Test ROIPooler can be initialized with default parameters."""
        from src.modules.detection.components.roi_pooling import ROIPooler

        pooler = ROIPooler(
            output_size=7, scales=[0.25, 0.125, 0.0625, 0.03125], sampling_ratio=2
        )

        assert pooler is not None
        assert pooler.output_size == 7
        assert pooler.scales == [0.25, 0.125, 0.0625, 0.03125]
        assert pooler.sampling_ratio == 2

    def test_roi_pooler_forward_produces_correct_shape(self) -> None:
        """Test ROIPooler extracts features with expected output shape.

        Given:
            - Multi-scale FPN features (P2-P5) with different spatial resolutions
            - A list of bounding boxes per image
        When:
            - forward() is called with features and boxes
        Then:
            - Output shape is (total_boxes, channels, output_size, output_size)
        """
        from src.modules.detection.components.roi_pooling import ROIPooler

        pooler = ROIPooler(
            output_size=7, scales=[0.25, 0.125, 0.0625, 0.03125], sampling_ratio=2
        )

        # Create multi-scale FPN features (batch_size=2, channels=256)
        # P2: 1/4 scale (0.25), P3: 1/8 (0.125), P4: 1/16 (0.0625), P5: 1/32 (0.03125)
        features = {
            "0": torch.randn(2, 256, 100, 100),  # P2: H/4, W/4
            "1": torch.randn(2, 256, 50, 50),  # P3: H/8, W/8
            "2": torch.randn(2, 256, 25, 25),  # P4: H/16, W/16
            "3": torch.randn(2, 256, 13, 13),  # P5: H/32, W/32 (approx)
        }

        # Boxes format: [x1, y1, x2, y2] in image coordinates (0-400 for 400x400 image)
        # Image 1: 3 boxes, Image 2: 2 boxes
        boxes = [
            torch.tensor(
                [
                    [10.0, 10.0, 50.0, 50.0],
                    [100.0, 100.0, 200.0, 200.0],
                    [300.0, 300.0, 390.0, 390.0],
                ]
            ),
            torch.tensor([[50.0, 50.0, 150.0, 150.0], [200.0, 200.0, 350.0, 350.0]]),
        ]

        # Forward pass
        output = pooler(features, boxes)

        # Verify output shape: (total_boxes=5, channels=256, H=7, W=7)
        assert output.shape == (5, 256, 7, 7)

    def test_roi_pooler_with_single_box(self) -> None:
        """Test ROIPooler with single box per image (edge case).

        Given:
            - FPN features and a single box per image
        When:
            - forward() is called
        Then:
            - Output shape is (num_images, channels, output_size, output_size)
        """
        from src.modules.detection.components.roi_pooling import ROIPooler

        pooler = ROIPooler(output_size=7, scales=[0.25, 0.125], sampling_ratio=2)

        features = {
            "0": torch.randn(1, 256, 50, 50),
            "1": torch.randn(1, 256, 25, 25),
        }

        boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]

        output = pooler(features, boxes)

        assert output.shape == (1, 256, 7, 7)

    def test_roi_pooler_with_empty_boxes(self) -> None:
        """Test ROIPooler with empty box list (edge case).

        Given:
            - FPN features but no boxes for some images
        When:
            - forward() is called with empty box tensors
        Then:
            - Returns empty tensor with correct shape dimensions except batch
        """
        from src.modules.detection.components.roi_pooling import ROIPooler

        pooler = ROIPooler(output_size=7, scales=[0.25, 0.125], sampling_ratio=2)

        features = {
            "0": torch.randn(2, 256, 50, 50),
            "1": torch.randn(2, 256, 25, 25),
        }

        # Image 1: 1 box, Image 2: 0 boxes
        boxes = [
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            torch.empty((0, 4)),
        ]

        output = pooler(features, boxes)

        # Should have 1 output (only first image's box)
        assert output.shape == (1, 256, 7, 7)

    def test_roi_pooler_with_different_output_sizes(self) -> None:
        """Test ROIPooler with different output sizes.

        Given:
            - Different output_size parameters (5, 7, 14)
        When:
            - ROIPooler is instantiated and forward is called
        Then:
            - Output spatial dimensions match output_size
        """
        from src.modules.detection.components.roi_pooling import ROIPooler

        features = {
            "0": torch.randn(1, 256, 50, 50),
        }
        boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]

        for size in [5, 7, 14]:
            pooler = ROIPooler(output_size=size, scales=[0.25], sampling_ratio=2)
            output = pooler(features, boxes)
            assert output.shape == (1, 256, size, size)

    def test_roi_pooler_preserves_gradient_flow(self) -> None:
        """Test ROIPooler preserves gradients for backpropagation.

        Given:
            - FPN features with requires_grad=True
        When:
            - forward() and backward() are called
        Then:
            - Gradients flow back to input features
        """
        from src.modules.detection.components.roi_pooling import ROIPooler

        pooler = ROIPooler(output_size=7, scales=[0.25], sampling_ratio=2)

        features = {
            "0": torch.randn(1, 256, 50, 50, requires_grad=True),
        }
        boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]

        output = pooler(features, boxes)
        loss = output.sum()
        loss.backward()

        # Verify gradients exist
        assert features["0"].grad is not None
        assert features["0"].grad.shape == features["0"].shape

    def test_roi_pooler_default_fpn_scales(self) -> None:
        """Test ROIPooler with standard FPN scales from reference implementation.

        Given:
            - Reference FPN scales [0.25, 0.125, 0.0625, 0.03125] (P2-P5)
        When:
            - ROIPooler is created with these scales
        Then:
            - Pooler correctly routes boxes to appropriate feature levels
        """
        from src.modules.detection.components.roi_pooling import ROIPooler

        # Reference standard: P2-P5 scales
        pooler = ROIPooler(
            output_size=7,
            scales=[0.25, 0.125, 0.0625, 0.03125],  # P2, P3, P4, P5
            sampling_ratio=2,
        )

        # Create FPN features for a 512x512 image
        features = {
            "0": torch.randn(1, 256, 128, 128),  # P2: 512/4 = 128
            "1": torch.randn(1, 256, 64, 64),  # P3: 512/8 = 64
            "2": torch.randn(1, 256, 32, 32),  # P4: 512/16 = 32
            "3": torch.randn(1, 256, 16, 16),  # P5: 512/32 = 16
        }

        # Mix of box sizes (small, medium, large)
        boxes = [
            torch.tensor(
                [
                    [10.0, 10.0, 40.0, 40.0],  # Small box (30x30 pixels)
                    [100.0, 100.0, 300.0, 300.0],  # Large box (200x200 pixels)
                ]
            )
        ]

        output = pooler(features, boxes)

        assert output.shape == (2, 256, 7, 7)

    def test_roi_pooler_is_nn_module(self) -> None:
        """Test ROIPooler is a proper nn.Module.

        Given:
            - ROIPooler class
        When:
            - Instance is created
        Then:
            - Instance is a nn.Module subclass
            - Can be moved to different devices
            - Can be set to train/eval modes
        """
        from src.modules.detection.components.roi_pooling import ROIPooler

        pooler = ROIPooler(output_size=7, scales=[0.25], sampling_ratio=2)

        # Should be a proper nn.Module
        assert isinstance(pooler, nn.Module)

        # Should support train/eval modes
        pooler.train()
        assert pooler.training is True

        pooler.eval()
        assert pooler.training is False

    def test_roi_pooler_with_tuple_output_size(self) -> None:
        """Test ROIPooler with tuple output size (H, W).

        Given:
            - output_size as tuple (7, 14)
        When:
            - ROIPooler is instantiated and forward is called
        Then:
            - Output has shape (N, C, 7, 14)
        """
        from src.modules.detection.components.roi_pooling import ROIPooler

        pooler = ROIPooler(output_size=(7, 14), scales=[0.25], sampling_ratio=2)

        features = {"0": torch.randn(1, 256, 50, 50)}
        boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]

        output = pooler(features, boxes)

        assert output.shape == (1, 256, 7, 14)

    def test_roi_pooler_validates_box_format(self) -> None:
        """Test ROIPooler validates box tensor format.

        Given:
            - Boxes with wrong number of columns (not 4)
        When:
            - forward() is called
        Then:
            - Raises appropriate error
        """
        from src.modules.detection.components.roi_pooling import ROIPooler

        pooler = ROIPooler(output_size=7, scales=[0.25], sampling_ratio=2)

        features = {"0": torch.randn(1, 256, 50, 50)}

        # Wrong box format (5 columns instead of 4)
        invalid_boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.9]])]

        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            pooler(features, invalid_boxes)
