"""VRD Detection Dataset for object detection evaluation.

Loads Visual Relationship Detection (VRD) dataset images and ground truth
bounding boxes/labels for detection benchmarking.

Key conversions:
- Boxes: [ymin, ymax, xmin, xmax] -> [xmin, ymin, xmax, ymax] (xyxy)
- Labels: 0-indexed -> 1-indexed (if background_class=True, for Faster R-CNN)
        or kept 0-indexed (if background_class=False, for EfficientDet)
"""

import json
from pathlib import Path
from typing import Callable, Literal

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F


class VRDDetectionDataset:
    """Dataset for VRD object detection.

    Loads images and ground truth boxes/labels from VRD dataset for
    detection evaluation. Extracts all unique objects from relationship
    annotations and returns them in standard detection format.

    Attributes:
        root: Path to VRD dataset root directory.
        split: Dataset split ("train" or "test").
        transform: Optional transform to apply to images.
        background_class: Whether to use 1-indexed labels (for Faster R-CNN).
        num_classes: Number of object classes.
        class_names: List of class names.

    Example:
        >>> dataset = VRDDetectionDataset(root="datasets/vrd", split="train")
        >>> image, target = dataset[0]
        >>> image.shape  # (3, H, W)
        >>> target["boxes"].shape  # (N, 4) in xyxy format
        >>> target["labels"].shape  # (N,) with values in [1, 100] or [0, 99]
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        transform: Callable[[Image.Image], Tensor] | None = None,
        background_class: bool = True,
    ) -> None:
        """Initialize VRD detection dataset.

        Args:
            root: Path to VRD dataset root directory containing:
                - sg_train_images/ and sg_test_images/
                - annotations_train.json and annotations_test.json
                - objects.json
            split: Dataset split, either "train" or "test".
            transform: Optional callable to transform images. If None, applies
                default normalization (ToTensor + ImageNet stats).
            background_class: If True, labels are 1-indexed [1, 100] with
                background at 0 (for Faster R-CNN). If False, labels are
                0-indexed [0, 99] (for EfficientDet). Default True.

        Raises:
            ValueError: If split is not "train" or "test".
            FileNotFoundError: If required files are missing.
        """
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.root = Path(root)
        self.split = split
        self.transform = transform if transform is not None else self._default_transform
        self.background_class = background_class

        # Load annotations
        ann_file = self.root / f"annotations_{split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotations not found: {ann_file}")

        with open(ann_file) as f:
            self.annotations = json.load(f)

        # Load class names
        objects_file = self.root / "objects.json"
        if not objects_file.exists():
            raise FileNotFoundError(f"Objects file not found: {objects_file}")

        with open(objects_file) as f:
            objects = json.load(f)

        # Add background class at index 0 if using 1-indexed labels
        if background_class:
            self._class_names = ["background"] + objects
        else:
            self._class_names = objects

        # Build image list, filtering out non-existent images
        self.image_dir = self.root / f"sg_{split}_images"
        all_image_ids = list(self.annotations.keys())

        # Filter to only existing images
        self.image_ids = [
            img_id for img_id in all_image_ids if (self.image_dir / img_id).exists()
        ]

        # Warn if some images are missing
        num_missing = len(all_image_ids) - len(self.image_ids)
        if num_missing > 0:
            import warnings

            warnings.warn(
                f"{num_missing}/{len(all_image_ids)} images not found in {self.image_dir}. "
                f"Only {len(self.image_ids)} images will be used.",
                UserWarning,
                stacklevel=2,
            )

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        """Get image and target annotations.

        Args:
            idx: Index of image to retrieve.

        Returns:
            Tuple of (image, target) where:
            - image: (3, H, W) tensor of RGB image
            - target: Dict with keys:
                - "boxes": (N, 4) tensor in xyxy format
                - "labels": (N,) tensor with class indices [1, 100]
        """
        image_id = self.image_ids[idx]

        # Load image (existence already validated in __init__)
        image_path = self.image_dir / image_id
        image = Image.open(image_path).convert("RGB")

        # Extract unique objects from relationship annotations
        boxes = []
        labels = []
        seen_objects = set()  # Track (category, bbox) to avoid duplicates

        for relation in self.annotations[image_id]:
            # Process subject and object
            for obj_key in ("subject", "object"):
                obj = relation[obj_key]
                category = obj["category"]
                bbox = tuple(obj["bbox"])  # [ymin, ymax, xmin, xmax]

                # Skip if we've already seen this exact object
                obj_id = (category, bbox)
                if obj_id in seen_objects:
                    continue
                seen_objects.add(obj_id)

                # Convert bbox from [ymin, ymax, xmin, xmax] to [xmin, ymin, xmax, ymax]
                ymin, ymax, xmin, xmax = bbox
                boxes.append([xmin, ymin, xmax, ymax])

                # Convert label: 1-indexed for Faster R-CNN, 0-indexed for EfficientDet
                if self.background_class:
                    labels.append(category + 1)  # [1, 100]
                else:
                    labels.append(category)  # [0, 99]

        # Convert to tensors
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            # Handle images with no annotations
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        # Apply transform to image
        image_tensor = self.transform(image)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        return image_tensor, target

    @property
    def num_classes(self) -> int:
        """Return number of classes (101 with background, 100 without)."""
        return len(self._class_names)

    @property
    def class_names(self) -> list[str]:
        """Return list of class names (index 0 is background if background_class=True)."""
        return list(self._class_names)  # Cast to list for type safety

    @staticmethod
    def _default_transform(image: Image.Image) -> Tensor:
        """Default image transform: ToTensor + ImageNet normalization.

        Args:
            image: PIL image.

        Returns:
            Normalized tensor.
        """
        # Convert to tensor
        tensor: Tensor = F.to_tensor(image)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        normalized: Tensor = (tensor - mean) / std
        return normalized
