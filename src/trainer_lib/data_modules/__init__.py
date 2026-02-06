"""Lightning DataModules for training workflows.

Provides pre-configured DataModules for different tasks:
- VRDDetectionDataModule: Object detection on VRD dataset
"""

from src.trainer_lib.data_modules.detection import VRDDetectionDataModule

__all__ = ["VRDDetectionDataModule"]
