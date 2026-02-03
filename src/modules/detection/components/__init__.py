"""Reusable detection components.

Shared building blocks for detector implementations:
- ROI pooling utilities
- Freezing utilities
"""

from .freeze import freeze_backbone_stages, freeze_bn, freeze_module
from .roi_pooling import ROIPooler

__all__ = [
    "ROIPooler",
    "freeze_module",
    "freeze_bn",
    "freeze_backbone_stages",
]
