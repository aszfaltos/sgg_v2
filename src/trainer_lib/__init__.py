"""Trainer utilities for PyTorch Lightning.

Provides logging utilities, callbacks, and other training-related functionality.
"""

from src.trainer_lib.data_modules.detection import VRDDetectionDataModule
from src.trainer_lib.lightning_modules.detector import DetectorLightningModule
from src.trainer_lib.logging.aim_logger import create_aim_logger

__all__ = [
    "create_aim_logger",
    "DetectorLightningModule",
    "VRDDetectionDataModule",
]
