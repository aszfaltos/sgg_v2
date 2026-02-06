"""Data loading and preprocessing modules.

Contains dataset classes for VRD and Visual Genome datasets,
as well as utilities for label mapping and data transformations.
"""

from src.data.vrd_detection import VRDDetectionDataset

__all__ = ["VRDDetectionDataset"]
