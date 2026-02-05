"""Faster R-CNN detector for Scene Graph Generation.

Uses torchvision's Faster R-CNN with ROI pooling from final detection boxes.
"""

from typing import Literal

import torch
from torch import Tensor
from torchvision.models.detection import (  # type: ignore[import-untyped]
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.backbone_utils import (  # type: ignore[import-untyped]
    resnet_fpn_backbone,
)

from src.modules.detection.base import (
    SGGDetector,
    SGGDetectorOutput,
    register_detector,
)
from src.modules.detection.components.freeze import freeze_bn, freeze_module


@register_detector("fasterrcnn")
class SGGFasterRCNN(SGGDetector):
    """Faster R-CNN for SGG using torchvision's full detector.

    ROI features are pooled from FPN features using final detection boxes.

    Args:
        backbone: ResNet variant ("resnet50" or "resnet101").
        pretrained: Load COCO pretrained weights.
        freeze: Freeze all detector parameters.
        min_score: Minimum score threshold for detections.
        max_detections_per_image: Maximum detections to return per image.
        nms_thresh: NMS IoU threshold.

    Attributes:
        num_classes: Number of COCO classes (91 including background).
        roi_feature_dim: Shape of ROI features (256, 7, 7).

    Note:
        Public attributes (e.g., ``backbone_name``) are for external inspection.
        Private attributes (e.g., ``_freeze``) are implementation details.

    Example:
        >>> detector = SGGFasterRCNN(backbone="resnet50", pretrained=True, freeze=True)
        >>> images = torch.rand(2, 3, 800, 600)
        >>> output = detector(images)
        >>> output.boxes[0].shape  # (N, 4) per-image boxes
        >>> output.labels[0].shape  # (N,) class indices
        >>> output.scores[0].shape  # (N,) confidence scores
        >>> output.roi_features.shape  # (total_N, 256, 7, 7)
    """

    def __init__(
        self,
        backbone: Literal["resnet50", "resnet101"] = "resnet50",
        pretrained: bool = True,
        freeze: bool = True,
        min_score: float = 0.05,
        max_detections_per_image: int = 100,
        nms_thresh: float = 0.5,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self._freeze = freeze
        self._min_score = min_score

        # Build model based on backbone
        if backbone == "resnet50":
            self.model = self._build_resnet50_model(
                pretrained=pretrained,
                min_score=min_score,
                max_detections=max_detections_per_image,
                nms_thresh=nms_thresh,
            )
        elif backbone == "resnet101":
            self.model = self._build_resnet101_model(
                pretrained=pretrained,
                min_score=min_score,
                max_detections=max_detections_per_image,
                nms_thresh=nms_thresh,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze if requested
        if freeze:
            freeze_module(self.model)
            freeze_bn(self.model)
            # torchvision requires eval mode for inference (asserts targets in train mode)
            self.model.eval()

    def _build_resnet50_model(
        self,
        pretrained: bool,
        min_score: float,
        max_detections: int,
        nms_thresh: float,
    ) -> FasterRCNN:
        """Build ResNet-50 FPN Faster R-CNN V2."""
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        return fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            box_score_thresh=min_score,
            box_nms_thresh=nms_thresh,
            box_detections_per_img=max_detections,
        )

    def _build_resnet101_model(
        self,
        pretrained: bool,
        min_score: float,
        max_detections: int,
        nms_thresh: float,
    ) -> FasterRCNN:
        """Build ResNet-101 FPN Faster R-CNN.

        torchvision doesn't have a simple API for ResNet-101, so we build manually.
        """
        # Build ResNet-101 FPN backbone
        weights_backbone = "DEFAULT" if pretrained else None
        backbone_model = resnet_fpn_backbone(
            backbone_name="resnet101",
            weights=weights_backbone,
            trainable_layers=0,  # Freeze all layers
        )

        # Build Faster R-CNN with this backbone
        model = FasterRCNN(
            backbone=backbone_model,
            num_classes=91,  # COCO classes
            box_score_thresh=min_score,
            box_nms_thresh=nms_thresh,
            box_detections_per_img=max_detections,
        )

        return model

    @property
    def num_classes(self) -> int:
        """COCO has 91 classes (including background)."""
        return 91

    @property
    def roi_feature_dim(self) -> tuple[int, int, int]:
        """FPN produces 256 channels, 7x7 spatial."""
        return (256, 7, 7)

    def forward(self, images: Tensor) -> SGGDetectorOutput:
        """Run Faster R-CNN and return SGGDetectorOutput.

        Args:
            images: (B, 3, H, W) batch of images, values in [0, 1].

        Returns:
            SGGDetectorOutput with detections and ROI features.
        """
        batch_size = images.shape[0]

        # Torchvision expects list[Tensor] format for batched inference
        image_list = [images[i] for i in range(batch_size)]

        detections = self.model(image_list)

        # Extract detection results
        boxes = [d["boxes"] for d in detections]
        labels = [d["labels"] for d in detections]
        scores = [d["scores"] for d in detections]

        # Extract FPN features for ROI pooling
        features = self.model.backbone(images)

        # Pool ROI features from final boxes
        # Image shapes for ROI pooling (all images same size in batch)
        image_shapes = [(images.shape[2], images.shape[3])] * batch_size
        roi_features = self._pool_roi_features(features, boxes, image_shapes)

        return SGGDetectorOutput(
            boxes=boxes,
            labels=labels,
            scores=scores,
            roi_features=roi_features,
        )

    def _pool_roi_features(
        self,
        features: dict[str, Tensor],
        boxes: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> Tensor:
        """Pool ROI features from FPN using final detection boxes.

        Uses torchvision's MultiScaleRoIAlign for multi-scale pooling.
        """
        # Check if any boxes exist
        total_boxes = sum(b.shape[0] for b in boxes)
        if total_boxes == 0:
            return torch.zeros(
                0, 256, 7, 7, device=next(iter(features.values())).device
            )

        roi_features = self.model.roi_heads.box_roi_pool(features, boxes, image_shapes)

        return roi_features

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"backbone={self.backbone_name!r}, "
            f"freeze={self._freeze}, "
            f"min_score={self._min_score})"
        )
