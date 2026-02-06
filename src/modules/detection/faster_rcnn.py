"""Faster R-CNN detector for Scene Graph Generation.

Uses torchvision's Faster R-CNN with ROI pooling from final detection boxes.
"""

from typing import Any, Literal

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
from torchvision.models.detection.faster_rcnn import (  # type: ignore[import-untyped]
    FastRCNNPredictor,
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
        num_classes: Number of object classes (default 91 for COCO).
        checkpoint_path: Path to checkpoint to load finetuned weights.
        trainable: Enable training mode (returns losses when targets provided).

    Attributes:
        backbone_name: Name of the backbone architecture.
        trainable: Whether the detector is in trainable mode.
        roi_feature_dim: Shape of ROI features (256, 7, 7).

    Note:
        When trainable=True and targets are provided, forward() returns a loss
        dict. Otherwise, it returns SGGDetectorOutput for inference.

    Example:
        >>> # Frozen detector for inference
        >>> detector = SGGFasterRCNN(backbone="resnet50", pretrained=True, freeze=True)
        >>> images = torch.rand(2, 3, 800, 600)
        >>> output = detector.predict(images)
        >>> output.boxes[0].shape  # (N, 4) per-image boxes

        >>> # Trainable detector for finetuning
        >>> detector = SGGFasterRCNN(num_classes=100, trainable=True, freeze=False)
        >>> targets = [{"boxes": ..., "labels": ...}]
        >>> losses = detector(images, targets)  # forward() returns loss dict
    """

    def __init__(
        self,
        backbone: Literal["resnet50", "resnet101"] = "resnet50",
        pretrained: bool = True,
        freeze: bool = True,
        min_score: float = 0.05,
        max_detections_per_image: int = 100,
        nms_thresh: float = 0.5,
        num_classes: int = 91,
        checkpoint_path: str | None = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self._freeze = freeze
        self._min_score = min_score
        self._num_classes = num_classes
        self.trainable = trainable

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

        # Replace box predictor if custom num_classes
        if num_classes != 91:
            self._replace_box_predictor(num_classes)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        # Freeze if requested
        if freeze:
            freeze_module(self.model)
            freeze_bn(self.model)
            # torchvision requires eval mode for inference (asserts targets in train mode)
            self.model.eval()
        else:
            # Trainable mode: keep in training mode
            self.model.train()

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

    def _replace_box_predictor(self, num_classes: int) -> None:
        """Replace the box predictor head with custom number of classes.

        Args:
            num_classes: Number of classes for the new predictor.
        """
        # Get the input feature dimension from the existing box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # Replace with new predictor
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint)

    @property
    def num_classes(self) -> int:
        """Number of object classes (including background)."""
        return self._num_classes

    @property
    def roi_feature_dim(self) -> tuple[int, int, int]:
        """FPN produces 256 channels, 7x7 spatial."""
        return (256, 7, 7)

    def forward(
        self, images: Tensor, targets: list[dict[str, Any]]
    ) -> dict[str, Tensor]:
        """Training forward pass - compute losses.

        Args:
            images: (B, 3, H, W) batch of images, values in [0, 1].
            targets: List of target dicts. Each dict should contain
                "boxes" (N, 4) and "labels" (N,) tensors.

        Returns:
            Dict of losses with keys: loss_classifier, loss_box_reg,
            loss_objectness, loss_rpn_box_reg.

        Raises:
            RuntimeError: If detector is not in trainable mode.
        """
        if not self.trainable:
            raise RuntimeError(
                "forward() requires trainable=True. Use predict() for inference."
            )

        batch_size = images.shape[0]
        image_list = [images[i] for i in range(batch_size)]
        return self.model(image_list, targets)  # type: ignore[no-any-return]

    def predict(self, images: Tensor) -> SGGDetectorOutput:
        """Inference forward pass - get detections and ROI features.

        Args:
            images: (B, 3, H, W) batch of images, values in [0, 1].

        Returns:
            SGGDetectorOutput with boxes, labels, scores, and roi_features.
        """
        batch_size = images.shape[0]

        # Torchvision expects list[Tensor] format
        image_list = [images[i] for i in range(batch_size)]

        # Run detection
        detections = self.model(image_list)

        # Extract detection results
        boxes = [d["boxes"] for d in detections]
        labels = [d["labels"] for d in detections]
        scores = [d["scores"] for d in detections]

        # Extract FPN features for ROI pooling
        features = self.model.backbone(images)

        # Pool ROI features from final boxes
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

        roi_features = self.model.roi_heads.box_roi_pool(features, boxes, image_shapes)  # type: ignore[no-any-return]

        return roi_features  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"backbone={self.backbone_name!r}, "
            f"freeze={self._freeze}, "
            f"min_score={self._min_score})"
        )
