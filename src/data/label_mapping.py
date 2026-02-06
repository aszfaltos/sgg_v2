"""Label mapping utilities for VRD ↔ COCO class mapping.

VRD has 100 object classes, COCO has 80 object classes. COCO-pretrained
detectors output COCO class IDs, which need to be mapped to VRD classes
for evaluation on the VRD dataset.

For strict evaluation (class-aware metrics), only use detections for
classes present in both datasets. For IoU-based evaluation (localization
quality), class mapping may be optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import torch
from torch import Tensor


# COCO class names (0-indexed, 80 classes)
# Source: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/fasterrcnn.py
COCO_CLASSES: Final[list[str]] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class VRDCOCOMapper:
    """Map between VRD and COCO class labels.

    VRD has 100 classes, COCO has 80 classes. This mapper handles:
    - Bidirectional mapping between label spaces
    - Identifying shared classes
    - Filtering detections to shared classes only

    Label indexing:
    - VRD: 0-indexed (0-99), class names from objects.json
    - COCO: 0-indexed (0-79), standard COCO classes
    - Background class (if used): handled separately, not mapped

    Example:
        >>> mapper = VRDCOCOMapper()
        >>> mapper.get_shared_classes()
        ['person', 'car', 'bicycle', ...]
        >>> vrd_labels = torch.tensor([0, 8, 42])  # person, car, dog
        >>> coco_labels = mapper.vrd_to_coco(vrd_labels)
        >>> # Filter to shared classes only
        >>> boxes, labels, scores = mapper.filter_to_shared(boxes, labels, scores)
    """

    def __init__(self, vrd_root: str | Path = "datasets/vrd") -> None:
        """Initialize the mapper.

        Args:
            vrd_root: Path to VRD dataset root (contains objects.json)
        """
        self.vrd_root = Path(vrd_root)

        # Load VRD classes from objects.json
        objects_file = self.vrd_root / "objects.json"
        with open(objects_file, "r") as f:
            self.vrd_classes: list[str] = json.load(f)

        # COCO classes (80 classes)
        self.coco_classes: list[str] = COCO_CLASSES

        # Build mapping dictionaries
        self._build_mappings()

    def _build_mappings(self) -> None:
        """Build bidirectional mapping between VRD and COCO classes."""
        # Normalize class names for matching (lowercase, handle variations)
        vrd_normalized = {
            self._normalize(name): idx for idx, name in enumerate(self.vrd_classes)
        }
        coco_normalized = {
            self._normalize(name): idx for idx, name in enumerate(self.coco_classes)
        }

        # Find shared classes
        shared_normalized = set(vrd_normalized.keys()) & set(coco_normalized.keys())
        self._shared_classes = [
            self.vrd_classes[vrd_normalized[norm_name]]
            for norm_name in shared_normalized
        ]

        # Build VRD → COCO mapping (maps VRD index to COCO index)
        self._vrd_to_coco_map: dict[int, int] = {}
        for norm_name in shared_normalized:
            vrd_idx = vrd_normalized[norm_name]
            coco_idx = coco_normalized[norm_name]
            self._vrd_to_coco_map[vrd_idx] = coco_idx

        # Build COCO → VRD mapping (maps COCO index to VRD index)
        self._coco_to_vrd_map: dict[int, int] = {}
        for norm_name in shared_normalized:
            coco_idx = coco_normalized[norm_name]
            vrd_idx = vrd_normalized[norm_name]
            self._coco_to_vrd_map[coco_idx] = vrd_idx

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize class name for matching.

        Handles variations like:
        - "traffic light" vs "traffic_light"
        - "potted plant" vs "plant"
        - Case differences

        Args:
            name: Original class name

        Returns:
            Normalized class name
        """
        # Lowercase and replace underscores/hyphens with spaces
        normalized = name.lower().replace("_", " ").replace("-", " ")

        # Handle common variations
        # "potted plant" in COCO vs "plant" in VRD
        if normalized == "potted plant":
            return "plant"

        return normalized

    def get_shared_classes(self) -> list[str]:
        """Get list of classes present in both VRD and COCO.

        Returns:
            List of class names (VRD naming convention)
        """
        return self._shared_classes.copy()

    def vrd_to_coco(self, vrd_labels: Tensor) -> Tensor:
        """Map VRD labels to COCO labels.

        Args:
            vrd_labels: VRD class labels (0-indexed), shape (N,)

        Returns:
            COCO class labels (0-indexed), shape (N,)
            Unmapped classes are marked as -1
        """
        coco_labels = torch.full_like(vrd_labels, -1)

        for vrd_idx, coco_idx in self._vrd_to_coco_map.items():
            mask = vrd_labels == vrd_idx
            coco_labels[mask] = coco_idx

        return coco_labels

    def coco_to_vrd(self, coco_labels: Tensor) -> Tensor:
        """Map COCO labels to VRD labels.

        Args:
            coco_labels: COCO class labels (0-indexed), shape (N,)

        Returns:
            VRD class labels (0-indexed), shape (N,)
            Unmapped classes are marked as -1
        """
        vrd_labels = torch.full_like(coco_labels, -1)

        for coco_idx, vrd_idx in self._coco_to_vrd_map.items():
            mask = coco_labels == coco_idx
            vrd_labels[mask] = vrd_idx

        return vrd_labels

    def filter_to_shared(
        self, boxes: Tensor, labels: Tensor, scores: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Filter detections to only include shared classes.

        Removes detections with classes not present in both VRD and COCO.
        Useful for strict evaluation where only shared classes are considered.

        Args:
            boxes: Bounding boxes, shape (N, 4) in xyxy format
            labels: Class labels (VRD or COCO, depending on source), shape (N,)
            scores: Confidence scores, shape (N,)

        Returns:
            Tuple of (filtered_boxes, filtered_labels, filtered_scores)
            All tensors have shape (M,) where M <= N

        Note:
            Assumes labels are VRD labels. For COCO labels, convert first.
        """
        if len(labels) == 0:
            return boxes, labels, scores

        # Check which labels are in the shared set (have valid mapping)
        valid_mask = torch.zeros(len(labels), dtype=torch.bool)
        for vrd_idx in self._vrd_to_coco_map.keys():
            valid_mask |= labels == vrd_idx

        # Filter
        filtered_boxes = boxes[valid_mask]
        filtered_labels = labels[valid_mask]
        filtered_scores = scores[valid_mask]

        return filtered_boxes, filtered_labels, filtered_scores
