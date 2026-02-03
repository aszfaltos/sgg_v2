"""Unit tests for SGGEfficientDet detector."""

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
        output = detector(sample_images)
        assert isinstance(output, SGGDetectorOutput)

    def test_forward_output_has_correct_batch_size(self, detector, sample_images):
        """Output lists should match batch size."""
        output = detector(sample_images)
        assert len(output.boxes) == 2
        assert len(output.labels) == 2
        assert len(output.scores) == 2
        assert len(output.logits) == 2

    def test_forward_boxes_are_xyxy_format(self, detector, sample_images):
        """Boxes should be in xyxy format (4 values per box)."""
        output = detector(sample_images)
        for boxes in output.boxes:
            if boxes.shape[0] > 0:
                assert boxes.shape[1] == 4

    def test_forward_labels_are_class_indices(self, detector, sample_images):
        """Labels should be integer class indices."""
        output = detector(sample_images)
        for labels in output.labels:
            if labels.shape[0] > 0:
                assert labels.dtype in (torch.int64, torch.long)

    def test_forward_scores_are_probabilities(self, detector, sample_images):
        """Scores should be in [0, 1]."""
        output = detector(sample_images)
        for scores in output.scores:
            if scores.shape[0] > 0:
                assert (scores >= 0).all()
                assert (scores <= 1).all()

    def test_forward_logits_have_correct_shape(self, detector, sample_images):
        """Logits should be (N, num_classes)."""
        output = detector(sample_images)
        for logits in output.logits:
            if logits.shape[0] > 0:
                assert logits.shape[1] == detector.num_classes

    def test_forward_roi_features_have_correct_channels(self, detector, sample_images):
        """ROI features should have BiFPN channels (112 for D2)."""
        output = detector(sample_images)
        if output.roi_features.shape[0] > 0:
            assert output.roi_features.shape[1] == 112  # D2 BiFPN channels
            assert output.roi_features.shape[2] == 7
            assert output.roi_features.shape[3] == 7

    def test_forward_auto_resizes_images(self, detector):
        """Should auto-resize images to required size."""
        # Pass smaller images - should be auto-resized to 768x768
        small_images = torch.rand(1, 3, 512, 512)
        output = detector(small_images)
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
