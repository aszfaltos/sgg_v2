"""Base classes for SGG detectors.

Provides the abstract interface and output dataclass that all SGG detectors
must implement. Ensures consistent API across different detector architectures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

# Type alias for detector configuration
DetectorConfig = dict[str, Any]

# Registry for detector classes
_DETECTOR_REGISTRY: dict[str, type["SGGDetector"]] = {}


@dataclass
class SGGDetectorOutput:
    """Output from an SGG detector's forward pass.

    Contains all information needed for scene graph generation:
    - Detection results (boxes, labels, scores)
    - Pooled ROI features for graph message passing

    All list fields have one element per image in the batch.

    Attributes:
        boxes: List of (N_i, 4) tensors in xyxy format, absolute coordinates.
        labels: List of (N_i,) tensors with predicted class indices.
        scores: List of (N_i,) tensors with confidence scores in [0, 1].
        roi_features: (total_boxes, C, H, W) tensor with pooled ROI features,
            where total_boxes = sum(N_i) across all images.

    Example:
        >>> output = detector(images)  # images: (B, 3, H, W)
        >>> len(output)  # Number of images
        2
        >>> output.boxes[0].shape  # Detections for first image
        torch.Size([15, 4])
        >>> output.total_boxes  # Total detections across batch
        42
    """

    boxes: list[Tensor]
    labels: list[Tensor]
    scores: list[Tensor]
    roi_features: Tensor

    def __len__(self) -> int:
        """Return number of images in batch."""
        return len(self.boxes)

    @property
    def total_boxes(self) -> int:
        """Return total number of boxes across all images."""
        return sum(b.shape[0] for b in self.boxes)

    @property
    def device(self) -> torch.device:
        """Return device of tensors."""
        return self.roi_features.device

    def to(self, device: torch.device | str) -> "SGGDetectorOutput":
        """Move all tensors to specified device.

        Args:
            device: Target device.

        Returns:
            New SGGDetectorOutput with tensors on target device.
        """
        return SGGDetectorOutput(
            boxes=[b.to(device) for b in self.boxes],
            labels=[lbl.to(device) for lbl in self.labels],
            scores=[s.to(device) for s in self.scores],
            roi_features=self.roi_features.to(device),
        )


class SGGDetector(nn.Module, ABC):
    """Abstract base class for SGG detectors.

    SGG detectors are frozen object detectors that provide all outputs
    needed for scene graph generation in a single forward pass:
    - Bounding boxes for detected objects
    - Predicted class labels and confidence scores
    - Pooled ROI features (for graph message passing)

    Subclasses must implement:
    - forward(): Main detection method returning SGGDetectorOutput
    - num_classes: Number of object classes
    - roi_feature_dim: Dimensionality of ROI features

    Example:
        >>> detector = SGGFasterRCNN(pretrained=True, freeze=True)
        >>> images = torch.rand(2, 3, 800, 600)
        >>> output = detector(images)
        >>> output.boxes[0].shape  # (N, 4) boxes for first image
        >>> output.roi_features.shape  # (total_N, 256, 7, 7)
    """

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of object classes (including background if applicable)."""
        ...

    @property
    @abstractmethod
    def roi_feature_dim(self) -> tuple[int, int, int]:
        """Shape of ROI features: (channels, height, width)."""
        ...

    @abstractmethod
    def forward(self, images: Tensor) -> SGGDetectorOutput:
        """Run detection on images.

        Args:
            images: (B, 3, H, W) batch of RGB images, values in [0, 1].

        Returns:
            SGGDetectorOutput containing:
            - boxes: List of (N_i, 4) detection boxes per image
            - labels: List of (N_i,) predicted class indices
            - scores: List of (N_i,) confidence scores
            - roi_features: (total_boxes, C, H, W) pooled features
        """
        ...


def register_detector(name: str):
    """Decorator to register a detector class.

    Args:
        name: Name to register the detector under.

    Returns:
        Decorator function that registers the class.

    Example:
        >>> @register_detector("my_detector")
        ... class MyDetector(SGGDetector):
        ...     ...
    """

    def decorator(cls: type[SGGDetector]) -> type[SGGDetector]:
        if name in _DETECTOR_REGISTRY:
            raise ValueError(f"Detector '{name}' already registered")
        _DETECTOR_REGISTRY[name] = cls
        return cls

    return decorator


def create_detector(name: str, **kwargs: Any) -> SGGDetector:
    """Create a detector by name.

    Args:
        name: Registered detector name (e.g., "fasterrcnn", "efficientdet").
        **kwargs: Detector-specific configuration arguments.

    Returns:
        Instantiated SGGDetector.

    Raises:
        ValueError: If detector name is not registered.

    Example:
        >>> detector = create_detector("fasterrcnn", pretrained=True, freeze=True)
        >>> detector = create_detector("efficientdet", variant="d2")
    """
    if name not in _DETECTOR_REGISTRY:
        available = list(_DETECTOR_REGISTRY.keys())
        raise ValueError(f"Unknown detector '{name}'. Available: {available}")

    return _DETECTOR_REGISTRY[name](**kwargs)


def list_detectors() -> list[str]:
    """List available detector names.

    Returns:
        List of registered detector names.
    """
    return list(_DETECTOR_REGISTRY.keys())
