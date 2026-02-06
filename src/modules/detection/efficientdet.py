"""EfficientDet detector for Scene Graph Generation.

Uses effdet's DetBenchPredict for inference with a hook to capture
BiFPN features for ROI pooling, or DetBenchTrain for training mode.
"""

from pathlib import Path
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
    from effdet import create_model  # type: ignore[import-untyped]
    from effdet.bench import DetBenchPredict, DetBenchTrain  # type: ignore[import-untyped]
    from omegaconf import read_write  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError(
        "effdet library is required for EfficientDetBackbone. "
        "Install with: uv add effdet"
    ) from e


# BiFPN channel counts by variant (d0-d7)
BIFPN_CHANNELS: dict[str, int] = {
    "d0": 64,
    "d1": 88,
    "d2": 112,
    "d3": 160,
    "d4": 224,
    "d5": 288,
    "d6": 384,
    "d7": 384,
}

# Required image sizes for each variant (d0-d7)
IMAGE_SIZES: dict[str, int] = {
    "d0": 512,
    "d1": 640,
    "d2": 768,
    "d3": 896,
    "d4": 1024,
    "d5": 1280,
    "d6": 1280,
    "d7": 1536,
}


@register_detector("efficientdet")
class SGGEfficientDet(SGGDetector):
    """EfficientDet for SGG using DetBenchPredict or DetBenchTrain.

    Uses effdet's built-in inference pipeline (DetBenchPredict) for detection,
    with a hook on the BiFPN to capture features for ROI pooling. Can also be
    used in training mode with DetBenchTrain.

    Args:
        variant: EfficientDet variant ("d0" through "d7").
        pretrained: Load pretrained weights (default: True).
        freeze: Freeze all detector parameters (default: True).
        trainable: Enable training mode (default: False). Mutually exclusive with freeze.
        num_classes: Number of object classes (default: 90 for COCO).
        checkpoint_path: Path to checkpoint file to load (default: None).
        max_detections_per_image: Maximum detections to return per image (default: 100).
        score_thresh: Minimum score threshold for detections (default: 0.001).
        roi_output_size: Spatial size of ROI pooling output (default: 7).
        box_loss_weight: Weight for box regression loss (default: 50.0). Lower values
            (e.g., 10.0) help classification head learn faster when finetuning.

    Attributes:
        num_classes: Number of object classes.
        roi_feature_dim: Shape of ROI features (channels, 7, 7).

    Note:
        Public attributes (e.g., ``variant``) are for external inspection.
        Private attributes (e.g., ``_freeze``) are implementation details.

    Example:
        Inference mode:
        >>> detector = SGGEfficientDet(variant="d2", pretrained=True, freeze=True)
        >>> images = torch.rand(2, 3, 768, 768)
        >>> output = detector.predict(images)
        >>> output.boxes[0].shape  # (N, 4) per-image boxes

        Training mode (torchvision format - standard):
        >>> detector = SGGEfficientDet(variant="d2", pretrained=False, trainable=True)
        >>> targets = [{"boxes": boxes, "labels": labels}]  # Same as Faster R-CNN
        >>> loss_dict = detector(images, targets)
        >>> loss_dict["loss"].backward()

        Training mode (effdet format - also supported):
        >>> targets = [{"bbox": boxes, "cls": labels, "img_scale": 1.0, "img_size": [768, 768]}]
        >>> loss_dict = detector(images, targets)
    """

    def __init__(
        self,
        variant: Literal["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"] = "d2",
        pretrained: bool = True,
        freeze: bool = True,
        trainable: bool = False,
        num_classes: int = 90,
        checkpoint_path: str | None = None,
        max_detections_per_image: int = 100,
        score_thresh: float = 0.001,
        roi_output_size: int = 7,
        box_loss_weight: float = 50.0,
    ) -> None:
        super().__init__()

        if variant not in BIFPN_CHANNELS:
            valid = ", ".join(sorted(BIFPN_CHANNELS.keys()))
            raise ValueError(f"Unsupported variant: {variant}. Valid: {valid}")

        if freeze and trainable:
            raise ValueError("Cannot set both freeze=True and trainable=True")

        self.variant = variant
        self._freeze = freeze
        self._trainable = trainable
        self._score_thresh = score_thresh
        self._image_size = IMAGE_SIZES[variant]
        self._fpn_channels = BIFPN_CHANNELS[variant]
        self._num_classes = num_classes

        # Load EfficientDet model
        model_name = f"tf_efficientdet_{variant}"

        # Use create_model which supports custom num_classes
        if num_classes != 90:
            # Custom num_classes - need to specify in model creation
            model = create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
        else:
            # Default COCO classes - use pretrained directly
            model = create_model(
                model_name,
                pretrained=pretrained,
            )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            checkpoint_path_obj = Path(checkpoint_path)
            if not checkpoint_path_obj.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)

        # Store model for direct parameter access
        self.model = model

        # Adjust loss weights for finetuning (default 50.0 favors box regression)
        # Lower values help classification head learn faster on new datasets
        if box_loss_weight != 50.0:
            with read_write(model.config):
                model.config.box_loss_weight = box_loss_weight

        # Create both train and predict benches if trainable
        if trainable:
            self.bench_train = DetBenchTrain(model, create_labeler=True)
            self.bench_predict = DetBenchPredict(model)
        else:
            self.bench_predict = DetBenchPredict(model)
            self.bench_train = None

        # Temporary storage for BiFPN features captured during forward pass.
        # Cleared at start of forward() and populated by _fpn_hook_fn.
        self._captured_fpn_features: list[Tensor] = []

        # Register hook to capture BiFPN features from predict bench
        self._fpn_hook = self.bench_predict.model.fpn.register_forward_hook(
            self._fpn_hook_fn
        )

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
            self.bench_predict.eval()
            if self.bench_train is not None:
                self.bench_train.eval()

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
        """Number of object classes (no background class in EfficientDet)."""
        return self._num_classes

    @property
    def roi_feature_dim(self) -> tuple[int, int, int]:
        """BiFPN channels depend on variant, 7x7 spatial."""
        return (self._fpn_channels, 7, 7)

    def forward(
        self, images: Tensor, targets: list[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        """Training forward pass - compute losses.

        Args:
            images: (B, 3, H, W) batch of images, values in [0, 1].
            targets: List of target dicts. Accepts both formats:
                - Torchvision format: "boxes" (N, 4), "labels" (N,)
                - Effdet format: "bbox" (N, 4), "cls" (N,), "img_scale", "img_size"

        Returns:
            Dict with keys "loss", "class_loss", "box_loss".

        Raises:
            RuntimeError: If detector is not in trainable mode.
        """
        if not self._trainable:
            raise RuntimeError(
                "forward() requires trainable=True. Use predict() for inference."
            )

        batch_size = images.shape[0]

        # Resize to required input size if needed
        if (
            images.shape[2] != self._image_size
            or images.shape[3] != self._image_size
        ):
            images = torch.nn.functional.interpolate(
                images,
                size=(self._image_size, self._image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Convert targets from list to dict format expected by DetBenchTrain
        target_dict = self._prepare_training_targets(targets, batch_size)

        # DetBenchTrain returns dict with loss, class_loss, box_loss
        return self.bench_train(images, target_dict)  # type: ignore[no-any-return]

    def predict(self, images: Tensor) -> SGGDetectorOutput:
        """Inference forward pass - get detections and ROI features.

        Args:
            images: (B, 3, H, W) batch of images, values in [0, 1].

        Returns:
            SGGDetectorOutput with boxes, labels, scores, and roi_features.
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
        detections = self.bench_predict(images)

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

    def _prepare_training_targets(
        self, targets: list[dict[str, Tensor]], batch_size: int
    ) -> dict[str, Tensor]:
        """Convert list of target dicts to batched dict format for DetBenchTrain.

        Accepts both torchvision format (boxes, labels) and effdet format
        (bbox, cls, img_scale, img_size). Converts torchvision format to effdet.

        Args:
            targets: List of per-image target dicts. Each dict should contain either:
                - Torchvision format: "boxes" (N, 4), "labels" (N,)
                - Effdet format: "bbox" (N, 4), "cls" (N,), "img_scale", "img_size"
            batch_size: Number of images in batch.

        Returns:
            Dict with batched tensors: bbox, cls, img_scale, img_size.
        """
        # Convert torchvision format to effdet format if needed
        converted_targets = []
        for target in targets:
            if "boxes" in target and "bbox" not in target:
                # Torchvision format -> effdet format
                converted = {
                    "bbox": target["boxes"],
                    "cls": target["labels"],
                    "img_scale": target.get(
                        "img_scale", torch.tensor(1.0, device=target["boxes"].device)
                    ),
                    "img_size": target.get(
                        "img_size",
                        torch.tensor(
                            [self._image_size, self._image_size],
                            device=target["boxes"].device,
                        ),
                    ),
                }
                converted_targets.append(converted)
            else:
                # Already in effdet format
                converted_targets.append(target)

        # Handle empty targets
        if all(t["bbox"].shape[0] == 0 for t in converted_targets):
            # All empty - create dummy target with one zero box
            device = converted_targets[0]["bbox"].device
            return {
                "bbox": torch.zeros(batch_size, 1, 4, device=device),
                "cls": torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                "img_scale": torch.ones(batch_size, device=device),
                "img_size": torch.tensor(
                    [[self._image_size, self._image_size]] * batch_size, device=device
                ),
            }

        # Stack per-image targets into batch
        # Note: bbox needs to be padded since images may have different numbers of objects
        max_objs = max(t["bbox"].shape[0] for t in converted_targets)
        max_objs = max(max_objs, 1)  # At least 1 for padding

        bbox_batch = []
        cls_batch = []
        img_scale_batch = []
        img_size_batch = []

        for target in converted_targets:
            num_objs = target["bbox"].shape[0]

            # Pad bbox and cls to max_objs
            bbox = target["bbox"]
            cls = target["cls"]

            if num_objs < max_objs:
                # Pad with zeros
                bbox_pad = torch.zeros(
                    max_objs - num_objs, 4, dtype=bbox.dtype, device=bbox.device
                )
                cls_pad = torch.zeros(
                    max_objs - num_objs, dtype=cls.dtype, device=cls.device
                )
                bbox = torch.cat([bbox, bbox_pad], dim=0)
                cls = torch.cat([cls, cls_pad], dim=0)

            bbox_batch.append(bbox)
            cls_batch.append(cls)
            img_scale_batch.append(target["img_scale"])
            img_size_batch.append(target["img_size"])

        return {
            "bbox": torch.stack(bbox_batch),  # (B, max_objs, 4)
            "cls": torch.stack(cls_batch),  # (B, max_objs)
            "img_scale": torch.stack(img_scale_batch),  # (B,)
            "img_size": torch.stack(img_size_batch),  # (B, 2)
        }

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
            # effdet outputs 1-indexed labels [1, N], convert to 0-indexed [0, N-1]
            labels = det[mask, 5].long() - 1
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
            return torch.zeros(
                0,
                self._fpn_channels,
                7,
                7,
                device=next(iter(features.values())).device,
            )

        return self.roi_pooler(features, boxes)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        """String representation."""
        mode = (
            "trainable" if self._trainable else "frozen" if self._freeze else "unfrozen"
        )
        return (
            f"{self.__class__.__name__}("
            f"variant={self.variant!r}, "
            f"mode={mode}, "
            f"num_classes={self._num_classes}, "
            f"score_thresh={self._score_thresh})"
        )
