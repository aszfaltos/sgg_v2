"""Unit tests for SGGFasterRCNN detector."""

import pytest
import torch

from src.modules.detection import SGGDetectorOutput, SGGFasterRCNN


class TestSGGFasterRCNNInit:
    """Test SGGFasterRCNN initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        detector = SGGFasterRCNN(pretrained=False, freeze=True)
        assert detector is not None
        assert detector.backbone_name == "resnet50"
        assert detector._freeze is True

    def test_init_resnet50(self):
        """Should initialize with ResNet-50 backbone."""
        detector = SGGFasterRCNN(backbone="resnet50", pretrained=False, freeze=True)
        assert detector.backbone_name == "resnet50"

    def test_init_resnet101(self):
        """Should initialize with ResNet-101 backbone."""
        detector = SGGFasterRCNN(backbone="resnet101", pretrained=False, freeze=True)
        assert detector.backbone_name == "resnet101"

    def test_init_invalid_backbone_raises(self):
        """Should raise for invalid backbone."""
        with pytest.raises(ValueError):
            SGGFasterRCNN(backbone="invalid", pretrained=False)

    def test_num_classes(self):
        """Should have 91 COCO classes."""
        detector = SGGFasterRCNN(pretrained=False, freeze=True)
        assert detector.num_classes == 91

    def test_roi_feature_dim(self):
        """Should have correct ROI feature dimensions."""
        detector = SGGFasterRCNN(pretrained=False, freeze=True)
        assert detector.roi_feature_dim == (256, 7, 7)


class TestSGGFasterRCNNFreeze:
    """Test freezing behavior."""

    def test_frozen_detector_has_no_gradients(self):
        """Frozen detector should have no trainable parameters."""
        detector = SGGFasterRCNN(pretrained=False, freeze=True)
        for param in detector.model.parameters():
            assert not param.requires_grad

    def test_unfrozen_detector_has_gradients(self):
        """Unfrozen detector should have trainable parameters."""
        detector = SGGFasterRCNN(pretrained=False, freeze=False)
        trainable = sum(p.requires_grad for p in detector.model.parameters())
        assert trainable > 0


class TestSGGFasterRCNNForward:
    """Test forward pass."""

    @pytest.fixture
    def detector(self):
        """Create frozen detector for testing."""
        return SGGFasterRCNN(pretrained=False, freeze=True)

    @pytest.fixture
    def sample_images(self):
        """Create sample batch of images."""
        return torch.rand(2, 3, 224, 224)

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
        """ROI features should have 256 channels."""
        output = detector(sample_images)
        if output.roi_features.shape[0] > 0:
            assert output.roi_features.shape[1] == 256
            assert output.roi_features.shape[2] == 7
            assert output.roi_features.shape[3] == 7

    def test_forward_with_no_grad(self, detector, sample_images):
        """Frozen detector should not track gradients."""
        with torch.no_grad():
            output = detector(sample_images)
        assert not output.roi_features.requires_grad


class TestSGGFasterRCNNRepr:
    """Test string representation."""

    def test_repr_contains_class_name(self):
        """Repr should contain class name."""
        detector = SGGFasterRCNN(pretrained=False, freeze=True)
        assert "SGGFasterRCNN" in repr(detector)

    def test_repr_contains_backbone(self):
        """Repr should contain backbone name."""
        detector = SGGFasterRCNN(backbone="resnet50", pretrained=False, freeze=True)
        assert "resnet50" in repr(detector)
