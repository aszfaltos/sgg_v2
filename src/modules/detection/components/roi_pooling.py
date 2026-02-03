"""ROI Pooling component.

Wraps torchvision's MultiScaleRoIAlign for extracting fixed-size features
from detected boxes across multi-scale FPN features.
"""

from collections import OrderedDict

from torch import Tensor, nn
from torchvision.ops import MultiScaleRoIAlign  # type: ignore[import-untyped]


class ROIPooler(nn.Module):
    """ROI Pooling wrapper using torchvision's MultiScaleRoIAlign.

    Extracts fixed-size feature maps from bounding boxes across multi-scale
    FPN feature pyramids (P2-P5). Automatically selects the appropriate
    feature level based on box size.

    Args:
        output_size: Output spatial size (H, W) or single int for square output.
            Standard value is 7 for 7x7 features.
        scales: Spatial scales of FPN levels relative to input image.
            Standard FPN: [0.25, 0.125, 0.0625, 0.03125] for P2-P5.
            These correspond to stride [4, 8, 16, 32].
        sampling_ratio: Number of sampling points per ROI bin.
            Higher values improve accuracy but increase computation.
            Standard value is 2.
        canonical_scale: Canonical box scale for level assignment.
            Boxes are assigned to levels based on their area relative to this.
        canonical_level: FPN level index corresponding to canonical_scale.

    Input:
        features: OrderedDict or dict of feature maps from FPN.
            Keys must be sequential strings: "0", "1", "2", "3" for P2-P5.
            Each value is a tensor of shape (B, C, H_i, W_i).
        boxes: List of bounding box tensors, one per image in batch.
            Each tensor has shape (N_i, 4) with format [x1, y1, x2, y2].
            Coordinates are in absolute image coordinates.

    Output:
        Tensor of shape (total_boxes, C, output_size, output_size).
        Features are extracted and pooled to fixed size.

    Example:
        >>> pooler = ROIPooler(
        ...     output_size=7,
        ...     scales=[0.25, 0.125, 0.0625, 0.03125],  # P2-P5
        ...     sampling_ratio=2
        ... )
        >>> features = {
        ...     "0": torch.randn(2, 256, 100, 100),  # P2
        ...     "1": torch.randn(2, 256, 50, 50),     # P3
        ...     "2": torch.randn(2, 256, 25, 25),     # P4
        ...     "3": torch.randn(2, 256, 13, 13),     # P5
        ... }
        >>> boxes = [
        ...     torch.tensor([[10., 10., 50., 50.]]),  # Image 1: 1 box
        ...     torch.tensor([[20., 20., 80., 80.]]),  # Image 2: 1 box
        ... ]
        >>> output = pooler(features, boxes)
        >>> output.shape
        torch.Size([2, 256, 7, 7])

    Reference:
        Feature Pyramid Networks for Object Detection (Lin et al., CVPR 2017)
        https://arxiv.org/abs/1612.03144
    """

    def __init__(
        self,
        output_size: int | tuple[int, int],
        scales: list[float],
        sampling_ratio: int,
        canonical_scale: int = 224,
        canonical_level: int = 4,
    ) -> None:
        """Initialize ROI Pooler.

        Args:
            output_size: Spatial size of output features.
            scales: FPN level scales (e.g., [0.25, 0.125, 0.0625, 0.03125]).
            sampling_ratio: Number of sampling points per bin.
            canonical_scale: Reference scale for level assignment (default: 224).
            canonical_level: FPN level for canonical scale (default: 4).
        """
        super().__init__()

        # Store configuration
        self.output_size = output_size
        self.scales = scales
        self.sampling_ratio = sampling_ratio
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

        # Create feature map names: "0", "1", "2", ... for P2, P3, P4, ...
        featmap_names = [str(i) for i in range(len(scales))]

        # Initialize torchvision's MultiScaleRoIAlign
        self._pooler = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=output_size,
            sampling_ratio=sampling_ratio,
            canonical_scale=canonical_scale,
            canonical_level=canonical_level,
        )

    def forward(
        self,
        features: dict[str, Tensor] | OrderedDict[str, Tensor],
        boxes: list[Tensor],
    ) -> Tensor:
        """Extract fixed-size features from boxes across FPN levels.

        Args:
            features: Multi-scale feature maps from FPN.
                Keys: "0", "1", "2", "3" for P2-P5.
                Values: Tensors of shape (B, C, H_i, W_i).
            boxes: List of box tensors, one per image.
                Each tensor shape: (N_i, 4) with [x1, y1, x2, y2] format.
                Coordinates in absolute image space.

        Returns:
            Tensor of shape (total_boxes, C, H_out, W_out) containing
            pooled features for all boxes across all images.

        Raises:
            ValueError: If box format is invalid (not N x 4).
            RuntimeError: If feature map sizes are inconsistent with scales.

        Note:
            - Empty box tensors (N_i=0) are handled gracefully.
            - Image sizes are inferred from the largest feature map (P2).
            - Boxes are automatically assigned to appropriate FPN levels.
        """
        # Validate box format
        for i, box_tensor in enumerate(boxes):
            if box_tensor.numel() > 0:  # Skip empty tensors
                if box_tensor.shape[-1] != 4:
                    raise ValueError(
                        f"Box tensor {i} has shape {box_tensor.shape}, "
                        f"but expected last dimension to be 4 (x1, y1, x2, y2). "
                        f"Got {box_tensor.shape[-1]} columns."
                    )

        # Infer image sizes from the highest-resolution feature map (P2, key "0")
        # P2 has scale 0.25 (stride 4), so image size = feature size * 4
        if "0" not in features:
            raise ValueError(
                f"Expected feature map '0' (P2) in features dict, "
                f"but got keys: {list(features.keys())}"
            )

        p2_features = features["0"]  # Shape: (B, C, H_P2, W_P2)
        batch_size = p2_features.shape[0]
        h_p2, w_p2 = p2_features.shape[2], p2_features.shape[3]

        # Image size = P2 size / P2 scale
        # Standard P2 scale is 0.25 (stride 4)
        p2_scale = self.scales[0]
        img_h = int(h_p2 / p2_scale)
        img_w = int(w_p2 / p2_scale)

        # Create image_shapes list: [(H, W), (H, W), ...]
        image_shapes = [(img_h, img_w)] * batch_size

        # Convert features dict to OrderedDict if needed
        if not isinstance(features, OrderedDict):
            # Ensure features are in correct order: "0", "1", "2", ...
            features = OrderedDict((k, features[k]) for k in sorted(features.keys()))

        # Call torchvision's MultiScaleRoIAlign
        pooled_features: Tensor = self._pooler(features, boxes, image_shapes)

        return pooled_features

    def __repr__(self) -> str:
        """String representation of ROIPooler."""
        return (
            f"{self.__class__.__name__}("
            f"output_size={self.output_size}, "
            f"scales={self.scales}, "
            f"sampling_ratio={self.sampling_ratio}, "
            f"canonical_scale={self.canonical_scale}, "
            f"canonical_level={self.canonical_level})"
        )
