"""Unit tests for default detector configurations.

Tests verify that all configuration dictionaries contain required keys
and have appropriate default values for instantiating detectors.
"""


from src.modules.detection.configs.defaults import (
    EFFICIENTDET_D2,
    EFFICIENTDET_D3,
    FASTERRCNN_R50,
    FASTERRCNN_R101,
)


class TestFasterRCNNConfigs:
    """Test Faster R-CNN configuration dictionaries."""

    def test_fasterrcnn_r50_has_required_keys(self):
        """Should contain all required configuration keys."""
        required_keys = {
            "backbone",
            "pretrained",
            "freeze",
            "min_score",
            "max_detections_per_image",
            "nms_thresh",
        }
        assert required_keys.issubset(FASTERRCNN_R50.keys())

    def test_fasterrcnn_r50_correct_values(self):
        """Should have correct default values for ResNet-50."""
        assert FASTERRCNN_R50["backbone"] == "resnet50"
        assert FASTERRCNN_R50["pretrained"] is True
        assert FASTERRCNN_R50["freeze"] is True
        assert FASTERRCNN_R50["min_score"] == 0.05
        assert FASTERRCNN_R50["max_detections_per_image"] == 100
        assert FASTERRCNN_R50["nms_thresh"] == 0.5

    def test_fasterrcnn_r101_has_required_keys(self):
        """Should contain all required configuration keys."""
        required_keys = {
            "backbone",
            "pretrained",
            "freeze",
            "min_score",
            "max_detections_per_image",
            "nms_thresh",
        }
        assert required_keys.issubset(FASTERRCNN_R101.keys())

    def test_fasterrcnn_r101_correct_values(self):
        """Should have correct default values for ResNet-101."""
        assert FASTERRCNN_R101["backbone"] == "resnet101"
        assert FASTERRCNN_R101["pretrained"] is True
        assert FASTERRCNN_R101["freeze"] is True

    def test_fasterrcnn_configs_frozen_by_default(self):
        """Should have freeze=True for SGG training."""
        assert FASTERRCNN_R50["freeze"] is True
        assert FASTERRCNN_R101["freeze"] is True


class TestEfficientDetConfigs:
    """Test EfficientDet configuration dictionaries."""

    def test_efficientdet_d2_has_required_keys(self):
        """Should contain all required configuration keys."""
        required_keys = {
            "variant",
            "pretrained",
            "freeze",
            "min_score",
            "max_detections_per_image",
            "nms_thresh",
        }
        assert required_keys.issubset(EFFICIENTDET_D2.keys())

    def test_efficientdet_d2_correct_values(self):
        """Should have correct default values for EfficientDet-D2."""
        assert EFFICIENTDET_D2["variant"] == "d2"
        assert EFFICIENTDET_D2["pretrained"] is True
        assert EFFICIENTDET_D2["freeze"] is True
        assert EFFICIENTDET_D2["min_score"] == 0.05
        assert EFFICIENTDET_D2["max_detections_per_image"] == 100
        assert EFFICIENTDET_D2["nms_thresh"] == 0.5

    def test_efficientdet_d3_has_required_keys(self):
        """Should contain all required configuration keys."""
        required_keys = {
            "variant",
            "pretrained",
            "freeze",
            "min_score",
            "max_detections_per_image",
            "nms_thresh",
        }
        assert required_keys.issubset(EFFICIENTDET_D3.keys())

    def test_efficientdet_d3_correct_values(self):
        """Should have correct default values for EfficientDet-D3."""
        assert EFFICIENTDET_D3["variant"] == "d3"
        assert EFFICIENTDET_D3["pretrained"] is True
        assert EFFICIENTDET_D3["freeze"] is True

    def test_efficientdet_configs_frozen_by_default(self):
        """Should have freeze=True for SGG training."""
        assert EFFICIENTDET_D2["freeze"] is True
        assert EFFICIENTDET_D3["freeze"] is True

    def test_efficientdet_configs_have_different_variants(self):
        """Should have different variant identifiers."""
        assert EFFICIENTDET_D2["variant"] == "d2"
        assert EFFICIENTDET_D3["variant"] == "d3"
        assert EFFICIENTDET_D2["variant"] != EFFICIENTDET_D3["variant"]


class TestConfigImmutability:
    """Test that configs are safe to use."""

    def test_fasterrcnn_configs_are_distinct_objects(self):
        """Should not share references between configs."""
        assert FASTERRCNN_R50 is not FASTERRCNN_R101
        assert FASTERRCNN_R50["backbone"] != FASTERRCNN_R101["backbone"]

    def test_efficientdet_configs_are_distinct_objects(self):
        """Should not share references between configs."""
        assert EFFICIENTDET_D2 is not EFFICIENTDET_D3
        assert EFFICIENTDET_D2["variant"] != EFFICIENTDET_D3["variant"]


class TestConfigValueTypes:
    """Test that config values have correct types."""

    def test_fasterrcnn_config_types(self):
        """Should have correct types for all values."""
        assert isinstance(FASTERRCNN_R50["backbone"], str)
        assert isinstance(FASTERRCNN_R50["pretrained"], bool)
        assert isinstance(FASTERRCNN_R50["freeze"], bool)
        assert isinstance(FASTERRCNN_R50["min_score"], float)
        assert isinstance(FASTERRCNN_R50["max_detections_per_image"], int)
        assert isinstance(FASTERRCNN_R50["nms_thresh"], float)

    def test_efficientdet_config_types(self):
        """Should have correct types for all values."""
        assert isinstance(EFFICIENTDET_D2["variant"], str)
        assert isinstance(EFFICIENTDET_D2["pretrained"], bool)
        assert isinstance(EFFICIENTDET_D2["freeze"], bool)
        assert isinstance(EFFICIENTDET_D2["min_score"], float)
        assert isinstance(EFFICIENTDET_D2["max_detections_per_image"], int)
        assert isinstance(EFFICIENTDET_D2["nms_thresh"], float)
