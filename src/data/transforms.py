"""Detection data augmentation transforms.

Provides training and validation transforms for object detection that handle
both images and bounding boxes correctly. Uses torchvision.transforms.v2
which automatically applies transformations to boxes when needed.

Example:
    >>> train_transform = get_train_transforms()
    >>> image, target = dataset[0]
    >>> image_aug, target_aug = train_transform(image, target)
    >>> # image is augmented, boxes in target["boxes"] are transformed correctly
"""

import torch
from PIL.Image import Image
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2 as T


class DetectionTransform:
    """Callable transform for detection that handles images and boxes.

    Wraps a torchvision v2 Compose transform and handles conversion between
    plain tensors and tv_tensors.BoundingBoxes format.

    Args:
        composed: A torchvision v2 Compose transform.
    """

    def __init__(self, composed: T.Compose) -> None:
        self.composed = composed

    def __call__(
        self, image: Image, target: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply transforms to image and target.

        Args:
            image: PIL Image.
            target: Dict with "boxes" (Nx4 tensor, xyxy) and "labels" (N tensor).

        Returns:
            Tuple of (transformed_image, transformed_target).
        """
        boxes = target["boxes"]
        labels = target["labels"]

        # Get original image size for box format
        if isinstance(image, Image):
            img_size = (image.height, image.width)  # (H, W)
        else:
            img_size = image.shape[-2:]  # (H, W) for tensor

        # Convert boxes to torchvision.tv_tensors.BoundingBoxes format
        boxes_tv = tv_tensors.BoundingBoxes(
            boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=img_size,
        )

        # Apply transforms - v2 handles BoundingBoxes automatically
        transformed = self.composed({"image": image, "boxes": boxes_tv, "labels": labels})

        # Extract transformed data
        image_out = transformed["image"]
        boxes_out = transformed["boxes"]
        labels_out = transformed["labels"]

        # Convert back to plain tensors for target dict
        target_out = {
            "boxes": boxes_out.data if hasattr(boxes_out, "data") else boxes_out,
            "labels": labels_out,
        }

        return image_out, target_out


def get_train_transforms(
    target_size: tuple[int, int] | None = None,
) -> DetectionTransform:
    """Get training transforms with augmentation.

    Applies random horizontal flip, color jitter, and normalization.
    Both image and bounding boxes are transformed correctly.

    Args:
        target_size: Optional (height, width) to resize to. If None, no resizing.

    Returns:
        DetectionTransform that takes (image, target) and returns (image, target).
        Image should be PIL Image, target is dict with "boxes" (xyxy) and "labels".

    Example:
        >>> transform = get_train_transforms(target_size=(800, 1200))
        >>> image, target = transform(pil_image, {"boxes": boxes, "labels": labels})
    """
    transforms = []

    # Optional resize
    if target_size is not None:
        transforms.append(T.Resize(size=target_size, antialias=True))

    # Random horizontal flip (50% probability)
    transforms.append(T.RandomHorizontalFlip(p=0.5))

    # Color jitter for robustness
    transforms.append(
        T.ColorJitter(
            brightness=0.2,  # ±20% brightness
            contrast=0.2,  # ±20% contrast
            saturation=0.2,  # ±20% saturation
            hue=0.1,  # ±10% hue
        )
    )

    # Convert to tensor and normalize
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))  # Scale [0, 255] -> [0, 1]
    transforms.append(
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],  # ImageNet std
        )
    )

    return DetectionTransform(T.Compose(transforms))


def get_val_transforms(
    target_size: tuple[int, int] | None = None,
) -> DetectionTransform:
    """Get validation transforms (no augmentation).

    Applies only normalization to maintain consistency with training.
    No random augmentations are applied.

    Args:
        target_size: Optional (height, width) to resize to. If None, no resizing.

    Returns:
        DetectionTransform that takes (image, target) and returns (image, target).
        Image should be PIL Image, target is dict with "boxes" (xyxy) and "labels".

    Example:
        >>> transform = get_val_transforms(target_size=(800, 1200))
        >>> image, target = transform(pil_image, {"boxes": boxes, "labels": labels})
    """
    transforms = []

    # Optional resize
    if target_size is not None:
        transforms.append(T.Resize(size=target_size, antialias=True))

    # Convert to tensor and normalize (no augmentation)
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))  # Scale [0, 255] -> [0, 1]
    transforms.append(
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],  # ImageNet std
        )
    )

    return DetectionTransform(T.Compose(transforms))
