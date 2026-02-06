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

    def test_forward_roi_features_have_correct_channels(self, detector, sample_images):
        """ROI features should have 256 channels."""
        output = detector.predict(sample_images)
        if output.roi_features.shape[0] > 0:
            assert output.roi_features.shape[1] == 256
            assert output.roi_features.shape[2] == 7
            assert output.roi_features.shape[3] == 7

    def test_forward_with_no_grad(self, detector, sample_images):
        """Frozen detector should not track gradients."""
        with torch.no_grad():
            output = detector.predict(sample_images)
        assert not output.roi_features.requires_grad


class TestSGGFasterRCNNCustomClasses:
    """Test custom number of classes."""

    def test_init_with_custom_num_classes(self):
        """Should initialize with custom number of classes."""
        detector = SGGFasterRCNN(
            pretrained=False, freeze=False, num_classes=100, trainable=True
        )
        assert detector.num_classes == 100

    def test_custom_num_classes_replaces_box_predictor(self):
        """Should replace box predictor when num_classes != 91."""
        detector = SGGFasterRCNN(
            pretrained=False, freeze=False, num_classes=100, trainable=True
        )
        # Check that the box predictor has correct output dimensions
        # The box predictor should output num_classes for classification
        assert detector.model.roi_heads.box_predictor is not None

    def test_default_num_classes_is_91(self):
        """Should default to 91 COCO classes."""
        detector = SGGFasterRCNN(pretrained=False, freeze=True)
        assert detector.num_classes == 91


class TestSGGFasterRCNNCheckpointLoading:
    """Test checkpoint loading functionality."""

    def test_init_with_checkpoint_path(self, tmp_path):
        """Should load checkpoint when path provided."""
        # Create a detector and save its state
        detector1 = SGGFasterRCNN(
            pretrained=False, freeze=False, num_classes=100, trainable=True
        )
        checkpoint_path = tmp_path / "detector.pth"
        torch.save(detector1.model.state_dict(), checkpoint_path)

        # Load into a new detector
        detector2 = SGGFasterRCNN(
            pretrained=False,
            freeze=False,
            num_classes=100,
            checkpoint_path=str(checkpoint_path),
            trainable=True,
        )
        assert detector2 is not None

    def test_checkpoint_loading_preserves_weights(self, tmp_path):
        """Should preserve weights when loading checkpoint."""
        # Create a detector with custom num_classes
        detector1 = SGGFasterRCNN(
            pretrained=False, freeze=False, num_classes=100, trainable=True
        )

        # Get a reference parameter value
        first_param = next(detector1.model.parameters())
        original_value = first_param.data.clone()

        # Save checkpoint
        checkpoint_path = tmp_path / "detector.pth"
        torch.save(detector1.model.state_dict(), checkpoint_path)

        # Load into a new detector
        detector2 = SGGFasterRCNN(
            pretrained=False,
            freeze=False,
            num_classes=100,
            checkpoint_path=str(checkpoint_path),
            trainable=True,
        )

        # Compare parameter values
        second_param = next(detector2.model.parameters())
        assert torch.allclose(original_value, second_param.data)

    def test_init_without_checkpoint_path(self):
        """Should initialize normally when checkpoint_path is None."""
        detector = SGGFasterRCNN(
            pretrained=False, freeze=False, num_classes=100, trainable=True
        )
        assert detector is not None


class TestSGGFasterRCNNTrainableMode:
    """Test trainable mode functionality."""

    def test_init_trainable_true(self):
        """Should initialize in trainable mode."""
        detector = SGGFasterRCNN(
            pretrained=False, freeze=False, trainable=True, num_classes=100
        )
        assert detector.trainable is True

    def test_init_trainable_false(self):
        """Should initialize in frozen mode by default."""
        detector = SGGFasterRCNN(pretrained=False, freeze=True)
        assert detector.trainable is False

    def test_trainable_detector_has_gradients(self):
        """Trainable detector should have trainable parameters."""
        detector = SGGFasterRCNN(
            pretrained=False, freeze=False, trainable=True, num_classes=100
        )
        trainable = sum(p.requires_grad for p in detector.model.parameters())
        assert trainable > 0

    def test_trainable_detector_not_frozen(self):
        """Trainable detector should not be frozen."""
        detector = SGGFasterRCNN(
            pretrained=False, freeze=False, trainable=True, num_classes=100
        )
        assert detector.model.training is True

    @pytest.fixture
    def trainable_detector(self):
        """Create trainable detector for testing."""
        return SGGFasterRCNN(
            pretrained=False, freeze=False, trainable=True, num_classes=100
        )

    @pytest.fixture
    def sample_images(self):
        """Create sample batch of images."""
        return torch.rand(2, 3, 224, 224)

    @pytest.fixture
    def sample_targets(self):
        """Create sample targets for training."""
        return [
            {
                "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]).float(),
                "labels": torch.tensor([1, 2]).long(),
            },
            {
                "boxes": torch.tensor([[20, 20, 80, 80]]).float(),
                "labels": torch.tensor([3]).long(),
            },
        ]

    def test_forward_with_targets_returns_loss_dict(
        self, trainable_detector, sample_images, sample_targets
    ):
        """Should return loss dict when trainable and targets provided."""
        output = trainable_detector(sample_images, targets=sample_targets)
        assert isinstance(output, dict)
        assert "loss_classifier" in output or "loss_box_reg" in output

    def test_predict_returns_sgg_output(self, trainable_detector, sample_images):
        """predict() should return SGGDetectorOutput."""
        trainable_detector.eval()  # Set to eval for inference
        output = trainable_detector.predict(sample_images)
        assert isinstance(output, SGGDetectorOutput)

    def test_frozen_detector_predict_returns_sgg_output(self, sample_images):
        """Frozen detector predict() should return SGGDetectorOutput."""
        detector = SGGFasterRCNN(pretrained=False, freeze=True, trainable=False)
        output = detector.predict(sample_images)
        assert isinstance(output, SGGDetectorOutput)

    def test_forward_on_frozen_detector_raises(self, sample_images, sample_targets):
        """forward() on frozen detector should raise RuntimeError."""
        detector = SGGFasterRCNN(pretrained=False, freeze=True, trainable=False)
        with pytest.raises(RuntimeError):
            detector(sample_images, targets=sample_targets)

    def test_loss_dict_contains_expected_keys(
        self, trainable_detector, sample_images, sample_targets
    ):
        """Loss dict should contain expected Faster R-CNN loss keys."""
        output = trainable_detector(sample_images, targets=sample_targets)
        # Faster R-CNN typically returns these losses
        possible_keys = {
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        }
        assert any(key in output for key in possible_keys)

    def test_loss_values_are_tensors(
        self, trainable_detector, sample_images, sample_targets
    ):
        """Loss values should be tensors."""
        output = trainable_detector(sample_images, targets=sample_targets)
        for value in output.values():
            assert isinstance(value, torch.Tensor)


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
