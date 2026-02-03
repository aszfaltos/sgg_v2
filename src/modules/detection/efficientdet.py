"""EfficientDet detector for Scene Graph Generation.

Uses effdet's model components directly to get raw logits,
then decodes boxes and applies NMS while tracking kept indices.
"""

from typing import Literal

import torch
from torch import Tensor
from torchvision.ops import batched_nms  # type: ignore[import-untyped]

from src.modules.detection.base import (
    SGGDetector,
    SGGDetectorOutput,
    register_detector,
)
from src.modules.detection.components.freeze import freeze_bn, freeze_module
from src.modules.detection.components.roi_pooling import ROIPooler

try:
    from effdet import EfficientDet, get_efficientdet_config  # type: ignore[import-untyped]
    from effdet.anchors import Anchors  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError(
        "effdet library is required for EfficientDetBackbone. "
        "Install with: uv add effdet"
    ) from e


# BiFPN channel counts by variant
BIFPN_CHANNELS: dict[str, int] = {
    "d2": 112,
    "d3": 160,
}

# Required image sizes for each variant
IMAGE_SIZES: dict[str, int] = {
    "d2": 768,
    "d3": 896,
}


@register_detector("efficientdet")
class SGGEfficientDet(SGGDetector):
    """EfficientDet for SGG using model components directly.

    Uses class_net and box_net to get raw logits, then decodes boxes
    with anchors and applies NMS. Tracks which detections are kept
    to select corresponding logits (no reconstruction needed).

    Args:
        variant: EfficientDet variant ("d2" or "d3").
        pretrained: Load pretrained backbone weights.
        freeze: Freeze all detector parameters.
        min_score: Minimum score threshold for detections.
        max_detections_per_image: Maximum detections to return per image.
        nms_thresh: NMS IoU threshold.

    Attributes:
        num_classes: Number of COCO classes (90 for EfficientDet).
        roi_feature_dim: Shape of ROI features (channels, 7, 7).

    Example:
        >>> detector = SGGEfficientDet(variant="d2", pretrained=True, freeze=True)
        >>> images = torch.rand(2, 3, 768, 768)
        >>> output = detector(images)
        >>> output.boxes[0].shape  # (N, 4)
        >>> output.roi_features.shape  # (total_N, 112, 7, 7)
    """

    def __init__(
        self,
        variant: Literal["d2", "d3"] = "d2",
        pretrained: bool = True,
        freeze: bool = True,
        min_score: float = 0.05,
        max_detections_per_image: int = 100,
        nms_thresh: float = 0.5,
        roi_output_size: int = 7,
    ) -> None:
        super().__init__()

        if variant not in BIFPN_CHANNELS:
            raise ValueError(f"Unsupported variant: {variant}. Use 'd2' or 'd3'.")

        self.variant = variant
        self._freeze = freeze
        self._min_score = min_score
        self._max_detections = max_detections_per_image
        self._nms_thresh = nms_thresh
        self._image_size = IMAGE_SIZES[variant]
        self._fpn_channels = BIFPN_CHANNELS[variant]

        # Load EfficientDet model
        model_name = f"tf_efficientdet_{variant}"
        config = get_efficientdet_config(model_name)
        self._config = config

        # Create model (with pretrained backbone if requested)
        self.model = EfficientDet(config, pretrained_backbone=pretrained)

        # Create anchors for box decoding
        self.anchors = Anchors.from_config(config)

        # Create ROI pooler for BiFPN features (P3-P6)
        # Scales: [1/8, 1/16, 1/32, 1/64] for P3-P6
        self.roi_pooler = ROIPooler(
            output_size=roi_output_size,
            scales=[1 / 8, 1 / 16, 1 / 32, 1 / 64],
            sampling_ratio=2,
            canonical_scale=224,
            canonical_level=1,  # P4 is canonical
        )

        # Freeze if requested
        if freeze:
            freeze_module(self.model)
            freeze_bn(self.model)

    @property
    def num_classes(self) -> int:
        """EfficientDet uses 90 COCO classes (no background class)."""
        return self._config.num_classes

    @property
    def roi_feature_dim(self) -> tuple[int, int, int]:
        """BiFPN channels depend on variant, 7x7 spatial."""
        return (self._fpn_channels, 7, 7)

    def forward(self, images: Tensor) -> SGGDetectorOutput:
        """Run EfficientDet and return SGGDetectorOutput.

        Args:
            images: (B, 3, H, W) batch of images, values in [0, 1].

        Returns:
            SGGDetectorOutput with detections and ROI features.
        """
        batch_size = images.shape[0]
        device = images.device

        # Resize to required input size if needed
        if images.shape[2] != self._image_size or images.shape[3] != self._image_size:
            images = torch.nn.functional.interpolate(
                images,
                size=(self._image_size, self._image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Ensure eval mode for frozen detector
        was_training = self.model.training
        if self._freeze:
            self.model.eval()

        # Run model components
        with torch.no_grad() if self._freeze else torch.enable_grad():
            # Get backbone and BiFPN features
            backbone_features = self.model.backbone(images)
            bifpn_features = self.model.fpn(backbone_features)

            # Get raw logits from class_net and box_net
            # Returns List[Tensor] per FPN level
            class_outputs = self.model.class_net(bifpn_features)
            box_outputs = self.model.box_net(bifpn_features)

        # Restore training mode if needed
        if was_training and not self._freeze:
            self.model.train()

        # Decode boxes and apply NMS, tracking kept indices
        boxes, labels, scores, logits = self._decode_and_filter(
            class_outputs, box_outputs, batch_size, device
        )

        # Pool ROI features from final boxes
        # Use P3-P6 levels (indices 0-3)
        feature_dict = {str(i): bifpn_features[i] for i in range(4)}
        roi_features = self._pool_roi_features(feature_dict, boxes)

        return SGGDetectorOutput(
            boxes=boxes,
            labels=labels,
            scores=scores,
            logits=logits,
            roi_features=roi_features,
        )

    def _decode_and_filter(
        self,
        class_outputs: list[Tensor],
        box_outputs: list[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """Decode boxes, apply NMS, and gather logits for kept detections.

        Args:
            class_outputs: List of class logits per FPN level.
            box_outputs: List of box regression per FPN level.
            batch_size: Number of images in batch.
            device: Target device.

        Returns:
            Tuple of (boxes, labels, scores, logits) lists.
        """
        # Flatten class and box outputs across all levels
        # class_outputs[i] shape: (B, H_i, W_i, num_anchors * num_classes)
        # box_outputs[i] shape: (B, H_i, W_i, num_anchors * 4)

        # Get anchor boxes (pre-computed for this image size)
        anchor_boxes = self.anchors.boxes.to(device)
        # anchor_boxes shape: (num_anchors_total, 4)

        all_boxes: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_scores: list[Tensor] = []
        all_logits: list[Tensor] = []

        for batch_idx in range(batch_size):
            # Gather class logits and box regression for this image
            class_logits_list = []
            box_regression_list = []

            for cls_out, box_out in zip(class_outputs, box_outputs):
                # cls_out shape: (B, num_anchors * num_classes, H, W)
                cls_level = cls_out[batch_idx]  # (num_anchors * num_classes, H, W)
                box_level = box_out[batch_idx]  # (num_anchors * 4, H, W)

                # Reshape: (C, H, W) -> (H*W*num_anchors, num_classes or 4)
                cls_level = cls_level.permute(1, 2, 0).reshape(-1, self.num_classes)
                box_level = box_level.permute(1, 2, 0).reshape(-1, 4)

                class_logits_list.append(cls_level)
                box_regression_list.append(box_level)

            # Concatenate across all levels
            class_logits = torch.cat(class_logits_list, dim=0)  # (num_anchors_total, num_classes)
            box_regression = torch.cat(box_regression_list, dim=0)  # (num_anchors_total, 4)

            # Decode boxes from anchor offsets
            decoded_boxes = self._decode_boxes(anchor_boxes, box_regression)

            # Apply sigmoid to get scores
            class_scores = torch.sigmoid(class_logits)

            # Get max score and label per anchor
            max_scores, max_labels = class_scores.max(dim=1)

            # Filter by score threshold
            score_mask = max_scores > self._min_score
            filtered_boxes = decoded_boxes[score_mask]
            filtered_scores = max_scores[score_mask]
            filtered_labels = max_labels[score_mask]
            filtered_logits = class_logits[score_mask]

            if filtered_boxes.shape[0] == 0:
                all_boxes.append(torch.zeros(0, 4, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_logits.append(torch.zeros(0, self.num_classes, device=device))
                continue

            # Apply NMS per class
            keep = batched_nms(
                filtered_boxes, filtered_scores, filtered_labels, self._nms_thresh
            )

            # Limit detections
            keep = keep[: self._max_detections]

            all_boxes.append(filtered_boxes[keep])
            all_labels.append(filtered_labels[keep])
            all_scores.append(filtered_scores[keep])
            all_logits.append(filtered_logits[keep])  # Real logits, not reconstructed!

        return all_boxes, all_labels, all_scores, all_logits

    def _decode_boxes(self, anchors: Tensor, box_regression: Tensor) -> Tensor:
        """Decode box regression to absolute coordinates.

        Uses the standard box encoding: (dx, dy, dw, dh) relative to anchors.

        Args:
            anchors: (N, 4) anchor boxes in xyxy format.
            box_regression: (N, 4) predicted offsets.

        Returns:
            (N, 4) decoded boxes in xyxy format.
        """
        # Convert anchors to center format
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        # Decode
        dx = box_regression[:, 0]
        dy = box_regression[:, 1]
        dw = box_regression[:, 2]
        dh = box_regression[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        # Convert to xyxy format
        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        # Clamp to image bounds
        x1 = x1.clamp(min=0, max=self._image_size)
        y1 = y1.clamp(min=0, max=self._image_size)
        x2 = x2.clamp(min=0, max=self._image_size)
        y2 = y2.clamp(min=0, max=self._image_size)

        return torch.stack([x1, y1, x2, y2], dim=1)

    def _pool_roi_features(
        self,
        features: dict[str, Tensor],
        boxes: list[Tensor],
    ) -> Tensor:
        """Pool ROI features from BiFPN using final detection boxes."""
        total_boxes = sum(b.shape[0] for b in boxes)
        if total_boxes == 0:
            return torch.zeros(
                0,
                self._fpn_channels,
                7,
                7,
                device=next(iter(features.values())).device,
            )

        return self.roi_pooler(features, boxes)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"variant={self.variant!r}, "
            f"freeze={self._freeze}, "
            f"min_score={self._min_score}, "
            f"max_detections={self._max_detections})"
        )
