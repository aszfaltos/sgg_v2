"""Default detector configurations.

Centralized hyperparameters for all detector variants.
"""

from .defaults import (
    EFFICIENTDET_D2,
    EFFICIENTDET_D3,
    FASTERRCNN_R50,
    FASTERRCNN_R101,
    DetectorConfig,
)

__all__ = [
    "FASTERRCNN_R50",
    "FASTERRCNN_R101",
    "EFFICIENTDET_D2",
    "EFFICIENTDET_D3",
    "DetectorConfig",
]
