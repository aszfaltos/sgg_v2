"""EfficientDet detector for Scene Graph Generation.

Uses effdet's DetBenchPredict for inference with a hook to capture
BiFPN features for ROI pooling.
"""

from typing import Literal

import torch
from torch import Tensor

from src.modules.detection.base import (
    SGGDetector,
    SGGDetectorOutput,
    register_detector,
)
from src.modules.detection.components.freeze import freeze_bn, freeze_module
from src.modules.detection.components.roi_pooling import ROIPooler

try:
    from effdet import create_model_from_config, get_efficientdet_config  # type: ignore[import-untyped]
    from effdet.bench import DetBenchPredict  # type: ignore[import-untyped]
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
    """EfficientDet for SGG using DetBenchPredict.

    Uses effdet's built-in inference pipeline (DetBenchPredict) for detection,
    with a hook on the BiFPN to capture features for ROI pooling.

    Args:
        variant: EfficientDet variant ("d2" or "d3").
        pretrained: Load pretrained weights.
        freeze: Freeze all detector parameters.
        max_detections_per_image: Maximum detections to return per image (default: 100).
        score_thresh: Minimum score threshold for detections (default: 0.001).

    Attributes:
        num_classes: Number of COCO classes (90 for EfficientDet).
        roi_feature_dim: Shape of ROI features (channels, 7, 7).

    Note:
        Public attributes (e.g., ``variant``) are for external inspection.
        Private attributes (e.g., ``_freeze``) are implementation details.

    Example:
        >>> detector = SGGEfficientDet(variant="d2", pretrained=True, freeze=True)
        >>> images = torch.rand(2, 3, 768, 768)
        >>> output = detector(images)
        >>> output.boxes[0].shape  # (N, 4) per-image boxes
        >>> output.labels[0].shape  # (N,) class indices
        >>> output.scores[0].shape  # (N,) confidence scores
        >>> output.roi_features.shape  # (total_N, 112, 7, 7)
    """

    def __init__(
        self,
        variant: Literal["d2", "d3"] = "d2",
        pretrained: bool = True,
        freeze: bool = True,
        max_detections_per_image: int = 100,
        score_thresh: float = 0.001,
        roi_output_size: int = 7,
    ) -> None:
        super().__init__()

        if variant not in BIFPN_CHANNELS:
            raise ValueError(f"Unsupported variant: {variant}. Use 'd2' or 'd3'.")

        self.variant = variant
        self._freeze = freeze
        self._score_thresh = score_thresh
        self._image_size = IMAGE_SIZES[variant]
        self._fpn_channels = BIFPN_CHANNELS[variant]

        # Load EfficientDet model with DetBenchPredict wrapper
        model_name = f"tf_efficientdet_{variant}"
        config = get_efficientdet_config(model_name)
        config.max_det_per_image = max_detections_per_image
        self._num_classes = config.num_classes

        model = create_model_from_config(config, pretrained=pretrained)
        self.bench = DetBenchPredict(model)

        # Temporary storage for BiFPN features captured during forward pass.
        # Cleared at start of forward() and populated by _fpn_hook_fn.
        self._captured_fpn_features: list[Tensor] = []

        # Register hook to capture BiFPN features
        self._fpn_hook = self.bench.model.fpn.register_forward_hook(self._fpn_hook_fn)

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
            freeze_module(self.bench)
            freeze_bn(self.bench)
            self.bench.eval()

    def _fpn_hook_fn(
        self,
        module: torch.nn.Module,
        input: tuple[Tensor, ...],
        output: list[Tensor],
    ) -> None:
        """Capture BiFPN features during forward pass."""
        self._captured_fpn_features = list(output)

    @property
    def num_classes(self) -> int:
        """EfficientDet uses 90 COCO classes (no background class)."""
        return self._num_classes

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

        # Clear captured features
        self._captured_fpn_features = []

        # DetBenchPredict returns [B, max_det, 6] with format [x1, y1, x2, y2, score, class]
        detections = self.bench(images)

        # Parse detections into boxes, labels, scores
        boxes, labels, scores = self._parse_detections(detections, batch_size, device)

        # Pool ROI features from captured BiFPN features
        # Use P3-P6 levels (indices 0-3, skip P7 at index 4)
        feature_dict = {str(i): self._captured_fpn_features[i] for i in range(4)}
        roi_features = self._pool_roi_features(feature_dict, boxes)

        return SGGDetectorOutput(
            boxes=boxes,
            labels=labels,
            scores=scores,
            roi_features=roi_features,
        )

    def _parse_detections(
        self,
        detections: Tensor,
        batch_size: int,
        device: torch.device,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Parse DetBenchPredict output into separate lists.

        Args:
            detections: [B, max_det, 6] tensor with [x1, y1, x2, y2, score, class].
            batch_size: Number of images.
            device: Target device.

        Returns:
            Tuple of (boxes, labels, scores) lists.
        """
        all_boxes: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_scores: list[Tensor] = []

        for i in range(batch_size):
            det = detections[i]  # [max_det, 6]

            # Filter by score threshold
            scores = det[:, 4]
            mask = scores >= self._score_thresh

            boxes = det[mask, :4]  # [N, 4] xyxy format
            labels = det[mask, 5].long()  # [N] class indices
            scores = det[mask, 4]  # [N] confidence scores

            # Clip boxes to image bounds
            boxes = boxes.clamp(min=0, max=self._image_size)

            all_boxes.append(boxes)
            all_labels.append(labels)
            all_scores.append(scores)

        return all_boxes, all_labels, all_scores

    def _pool_roi_features(
        self,
        features: dict[str, Tensor],
        boxes: list[Tensor],
    ) -> Tensor:
        """Pool ROI features from BiFPN using final detection boxes."""
        total_boxes = sum(b.shape[0] for b in boxes)
        if total_boxes == 0:
            # Return empty tensor on same device as input features
            return torch.zeros(0, self._fpn_channels, 7, 7,
                device=next(iter(features.values())).device,
            )

        return self.roi_pooler(features, boxes)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"variant={self.variant!r}, "
            f"freeze={self._freeze}, "
            f"score_thresh={self._score_thresh})"
        )
