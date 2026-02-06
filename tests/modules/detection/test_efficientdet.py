"""Unit tests for SGGEfficientDet detector."""

import tempfile
from pathlib import Path

import pytest
import torch

from src.modules.detection import SGGDetectorOutput, SGGEfficientDet


class TestSGGEfficientDetInit:
    """Test SGGEfficientDet initialization."""

    def test_init_d2(self):
        """Should initialize with D2 variant."""
        detector = SGGEfficientDet(variant="d2", pretrained=False, freeze=True)
        assert detector is not None
        assert detector.variant == "d2"
        assert detector._freeze is True

    def test_init_d3(self):
        """Should initialize with D3 variant."""
        detector = SGGEfficientDet(variant="d3", pretrained=False, freeze=True)
        assert detector.variant == "d3"

    def test_init_invalid_variant_raises(self):
        """Should raise for invalid variant."""
        with pytest.raises(ValueError):
            SGGEfficientDet(variant="invalid", pretrained=False)

    def test_num_classes(self):
        """Should have 90 COCO classes (EfficientDet has no background)."""
        detector = SGGEfficientDet(variant="d2", pretrained=False, freeze=True)
        assert detector.num_classes == 90

    def test_roi_feature_dim_d2(self):
        """D2 should have 112 BiFPN channels."""
        detector = SGGEfficientDet(variant="d2", pretrained=False, freeze=True)
        assert detector.roi_feature_dim == (112, 7, 7)

    def test_roi_feature_dim_d3(self):
        """D3 should have 160 BiFPN channels."""
        detector = SGGEfficientDet(variant="d3", pretrained=False, freeze=True)
        assert detector.roi_feature_dim == (160, 7, 7)


class TestSGGEfficientDetFreeze:
    """Test freezing behavior."""

    def test_frozen_detector_has_no_gradients(self):
        """Frozen detector should have no trainable parameters."""
        detector = SGGEfficientDet(variant="d2", pretrained=False, freeze=True)
        for param in detector.model.parameters():
            assert not param.requires_grad

    def test_unfrozen_detector_has_gradients(self):
        """Unfrozen detector should have trainable parameters."""
        detector = SGGEfficientDet(variant="d2", pretrained=False, freeze=False)
        trainable = sum(p.requires_grad for p in detector.model.parameters())
        assert trainable > 0


class TestSGGEfficientDetForward:
    """Test forward pass."""

    @pytest.fixture
    def detector(self):
        """Create frozen detector for testing."""
        return SGGEfficientDet(variant="d2", pretrained=False, freeze=True)

    @pytest.fixture
    def sample_images(self):
        """Create sample batch of images (D2 requires 768x768)."""
        return torch.rand(2, 3, 768, 768)

    def test_forward_returns_sgg_output(self, detector, sample_images):
        """Should return SGGDetectorOutput."""
        output = detector.predict(sample_images)
        assert isinstance(output, SGGDetectorOutput)

    def test_forward_output_has_correct_batch_size(self, detector, sample_images):
        """Output lists should match batch size."""
        output = detector.predict(sample_images)
        assert len(output.boxes) == 2
        assert len(output.labels) == 2
        assert len(output.scores) == 2

    def test_forward_boxes_are_xyxy_format(self, detector, sample_images):
        """Boxes should be in xyxy format (4 values per box)."""
        output = detector.predict(sample_images)
        for boxes in output.boxes:
            if boxes.shape[0] > 0:
                assert boxes.shape[1] == 4

    def test_forward_labels_are_class_indices(self, detector, sample_images):
        """Labels should be integer class indices."""
        output = detector.predict(sample_images)
        for labels in output.labels:
            if labels.shape[0] > 0:
                assert labels.dtype in (torch.int64, torch.long)

    def test_forward_scores_are_probabilities(self, detector, sample_images):
        """Scores should be in [0, 1]."""
        output = detector.predict(sample_images)
        for scores in output.scores:
            if scores.shape[0] > 0:
                assert (scores >= 0).all()
                assert (scores <= 1).all()

    def test_forward_labels_are_valid_indices(self, detector, sample_images):
        """Labels should be valid class indices."""
        output = detector.predict(sample_images)
        for labels in output.labels:
            if labels.shape[0] > 0:
                assert (labels >= 0).all()
                assert (labels < detector.num_classes).all()

    def test_forward_roi_features_have_correct_channels(self, detector, sample_images):
        """ROI features should have BiFPN channels (112 for D2)."""
        output = detector.predict(sample_images)
        if output.roi_features.shape[0] > 0:
            assert output.roi_features.shape[1] == 112  # D2 BiFPN channels
            assert output.roi_features.shape[2] == 7
            assert output.roi_features.shape[3] == 7

    def test_forward_auto_resizes_images(self, detector):
        """Should auto-resize images to required size."""
        # Pass smaller images - should be auto-resized to 768x768
        small_images = torch.rand(1, 3, 512, 512)
        output = detector.predict(small_images)
        assert isinstance(output, SGGDetectorOutput)


class TestSGGEfficientDetRepr:
    """Test string representation."""

    def test_repr_contains_class_name(self):
        """Repr should contain class name."""
        detector = SGGEfficientDet(variant="d2", pretrained=False, freeze=True)
        assert "SGGEfficientDet" in repr(detector)

    def test_repr_contains_variant(self):
        """Repr should contain variant."""
        detector = SGGEfficientDet(variant="d2", pretrained=False, freeze=True)
        assert "d2" in repr(detector)


class TestSGGEfficientDetVariants:
    """Test all EfficientDet variants d0-d7."""

    @pytest.mark.parametrize(
        "variant,image_size,bifpn_channels",
        [
            ("d0", 512, 64),
            ("d1", 640, 88),
            ("d2", 768, 112),
            ("d3", 896, 160),
            ("d4", 1024, 224),
            ("d5", 1280, 288),
            ("d6", 1280, 384),
            ("d7", 1536, 384),
        ],
    )
    def test_variant_initialization(self, variant, image_size, bifpn_channels):
        """Should initialize all variants d0-d7 with correct params."""
        detector = SGGEfficientDet(variant=variant, pretrained=False, freeze=True)
        assert detector.variant == variant
        assert detector._image_size == image_size
        assert detector.roi_feature_dim == (bifpn_channels, 7, 7)

    @pytest.mark.parametrize("variant", ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"])
    def test_variant_predict(self, variant):
        """Should run predict for all variants."""
        detector = SGGEfficientDet(variant=variant, pretrained=False, freeze=True)
        # Use small images - will be auto-resized
        images = torch.rand(1, 3, 256, 256)
        output = detector.predict(images)
        assert isinstance(output, SGGDetectorOutput)


class TestSGGEfficientDetCustomNumClasses:
    """Test custom num_classes parameter."""

    def test_custom_num_classes(self):
        """Should initialize with custom num_classes."""
        detector = SGGEfficientDet(
            variant="d2", pretrained=False, freeze=True, num_classes=100
        )
        assert detector.num_classes == 100

    def test_default_num_classes_is_90(self):
        """Should default to 90 COCO classes."""
        detector = SGGEfficientDet(variant="d2", pretrained=False, freeze=True)
        assert detector.num_classes == 90


class TestSGGEfficientDetCheckpointLoading:
    """Test checkpoint loading."""

    def test_load_checkpoint(self):
        """Should load checkpoint from path."""
        # Create a detector and save its state
        detector1 = SGGEfficientDet(variant="d2", pretrained=False, freeze=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            torch.save(detector1.model.state_dict(), checkpoint_path)

            # Load checkpoint in new detector
            detector2 = SGGEfficientDet(
                variant="d2",
                pretrained=False,
                freeze=True,
                checkpoint_path=str(checkpoint_path),
            )

            # Verify parameters match
            for p1, p2 in zip(detector1.model.parameters(), detector2.model.parameters()):
                assert torch.allclose(p1, p2)

    def test_missing_checkpoint_raises(self):
        """Should raise error for missing checkpoint file."""
        with pytest.raises(FileNotFoundError):
            SGGEfficientDet(
                variant="d2",
                pretrained=False,
                freeze=True,
                checkpoint_path="/nonexistent/checkpoint.pth",
            )


class TestSGGEfficientDetTrainingMode:
    """Test training mode functionality."""

    @pytest.fixture
    def trainable_detector(self):
        """Create trainable detector."""
        return SGGEfficientDet(variant="d2", pretrained=False, freeze=False, trainable=True)

    @pytest.fixture
    def sample_images(self):
        """Sample batch of images."""
        return torch.rand(2, 3, 768, 768)

    @pytest.fixture
    def sample_targets(self):
        """Sample training targets in effdet format."""
        return [
            {
                "bbox": torch.tensor([[10, 20, 100, 200], [150, 50, 300, 250]], dtype=torch.float32),
                "cls": torch.tensor([5, 10], dtype=torch.long),
                "img_scale": torch.tensor(1.0),
                "img_size": torch.tensor([768, 768]),
            },
            {
                "bbox": torch.tensor([[30, 40, 120, 180]], dtype=torch.float32),
                "cls": torch.tensor([7], dtype=torch.long),
                "img_scale": torch.tensor(1.0),
                "img_size": torch.tensor([768, 768]),
            },
        ]

    def test_trainable_mode_has_gradients(self, trainable_detector):
        """Trainable detector should have trainable parameters."""
        trainable = sum(p.requires_grad for p in trainable_detector.model.parameters())
        assert trainable > 0

    def test_trainable_mode_forward_with_targets_returns_loss(
        self, trainable_detector, sample_images, sample_targets
    ):
        """Forward with targets should return loss dict in training mode."""
        trainable_detector.train()
        output = trainable_detector(sample_images, sample_targets)

        assert isinstance(output, dict)
        assert "loss" in output
        assert "class_loss" in output
        assert "box_loss" in output

        assert isinstance(output["loss"], torch.Tensor)
        assert output["loss"].requires_grad

    def test_predict_returns_detections(self, trainable_detector, sample_images):
        """predict() should return detections."""
        trainable_detector.eval()
        output = trainable_detector.predict(sample_images)

        assert isinstance(output, SGGDetectorOutput)
        assert len(output.boxes) == 2
        assert len(output.labels) == 2

    def test_forward_on_frozen_detector_raises(self, sample_images, sample_targets):
        """forward() on frozen detector should raise RuntimeError."""
        frozen_detector = SGGEfficientDet(variant="d2", pretrained=False, freeze=True)

        with pytest.raises(RuntimeError, match="forward\\(\\) requires trainable=True"):
            frozen_detector(sample_images, sample_targets)

    def test_trainable_mode_backward_updates_gradients(
        self, trainable_detector, sample_images, sample_targets
    ):
        """Should compute gradients in training mode."""
        trainable_detector.train()

        # Zero gradients
        trainable_detector.zero_grad()

        # Forward pass
        output = trainable_detector(sample_images, sample_targets)
        loss = output["loss"]

        # Backward pass
        loss.backward()

        # Check that gradients were computed
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in trainable_detector.model.parameters()
            if p.requires_grad
        )
        assert has_grad
