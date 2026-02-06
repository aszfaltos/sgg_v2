"""SGG Detection Module.

Provides frozen object detectors that return all outputs needed for
scene graph generation in a single forward pass.
"""

from .base import (
    SGGDetector,
    SGGDetectorOutput,
    create_detector,
    list_detectors,
    register_detector,
)
from .components.freeze import freeze_backbone_stages, freeze_bn, freeze_module
from .components.roi_pooling import ROIPooler
from .configs.defaults import (
    EFFICIENTDET_D2,
    EFFICIENTDET_D3,
    FASTERRCNN_R50,
    FASTERRCNN_R101,
)
from .efficientdet import SGGEfficientDet
from .faster_rcnn import SGGFasterRCNN

__all__ = [
    # Base classes
    "SGGDetector",
    "SGGDetectorOutput",
    # Factory
    "create_detector",
    "list_detectors",
    "register_detector",
    # Implementations
    "SGGFasterRCNN",
    "SGGEfficientDet",
    # Configs
    "FASTERRCNN_R50",
    "FASTERRCNN_R101",
    "EFFICIENTDET_D2",
    "EFFICIENTDET_D3",
    # Components
    "ROIPooler",
    # Utilities
    "freeze_module",
    "freeze_bn",
    "freeze_backbone_stages",
]
