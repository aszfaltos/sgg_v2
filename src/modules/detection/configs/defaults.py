"""Default detector configurations.

Centralized hyperparameters for SGG detectors.

Usage:
    >>> from src.modules.detection import create_detector, FASTERRCNN_R50
    >>> detector = create_detector("fasterrcnn", **FASTERRCNN_R50)

    >>> # Or use configs directly
    >>> from src.modules.detection import SGGFasterRCNN, FASTERRCNN_R101
    >>> detector = SGGFasterRCNN(**FASTERRCNN_R101)
"""

from typing import Any

# Type alias for configuration dictionaries
DetectorConfig = dict[str, Any]

# ============================================================================
# Faster R-CNN Configurations
# ============================================================================

FASTERRCNN_R50: DetectorConfig = {
    "backbone": "resnet50",
    "pretrained": True,
    "freeze": True,
    "min_score": 0.05,
    "max_detections_per_image": 100,
    "nms_thresh": 0.5,
}

FASTERRCNN_R101: DetectorConfig = {
    "backbone": "resnet101",
    "pretrained": True,
    "freeze": True,
    "min_score": 0.05,
    "max_detections_per_image": 100,
    "nms_thresh": 0.5,
}

# ============================================================================
# EfficientDet Configurations
# ============================================================================

EFFICIENTDET_D2: DetectorConfig = {
    "variant": "d2",
    "pretrained": True,
    "freeze": True,
    "min_score": 0.05,
    "max_detections_per_image": 100,
    "nms_thresh": 0.5,
}

EFFICIENTDET_D3: DetectorConfig = {
    "variant": "d3",
    "pretrained": True,
    "freeze": True,
    "min_score": 0.05,
    "max_detections_per_image": 100,
    "nms_thresh": 0.5,
}

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "FASTERRCNN_R50",
    "FASTERRCNN_R101",
    "EFFICIENTDET_D2",
    "EFFICIENTDET_D3",
    "DetectorConfig",
]
