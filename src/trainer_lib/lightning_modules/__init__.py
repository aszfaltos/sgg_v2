"""PyTorch Lightning modules for training."""

from src.trainer_lib.lightning_modules.detector import DetectorLightningModule
from src.trainer_lib.lightning_modules.sgg import SGGLightningModule

__all__ = ["DetectorLightningModule", "SGGLightningModule"]
