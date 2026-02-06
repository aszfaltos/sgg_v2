"""Lightning DataModule for VRD object detection.

Provides train/val dataloaders for VRD detection task with:
- Reproducible train/val splits using seeded random_split
- Padding collate function for variable-sized images
- Train and validation transforms
- Configurable batch size, num_workers, and target size

Example:
    >>> dm = VRDDetectionDataModule(
    ...     root="datasets/vrd",
    ...     batch_size=4,
    ...     val_split=0.1,
    ...     target_size=(800, 1200),
    ... )
    >>> dm.setup(stage="fit")
    >>> train_loader = dm.train_dataloader()
    >>> for images, targets in train_loader:
    ...     # images: (B, 3, H, W) padded to same size
    ...     # targets: list of dicts with "boxes" and "labels"
    ...     pass
"""

from pathlib import Path
from typing import Any, Callable

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.vrd_detection import VRDDetectionDataset


class VRDDetectionDataModule(LightningDataModule):
    """Lightning DataModule for VRD object detection.

    Handles data loading for VRD detection task with automatic train/val split,
    transform application, and collation for variable-sized images.

    Attributes:
        root: Path to VRD dataset root directory.
        batch_size: Batch size for dataloaders.
        val_split: Fraction of data to use for validation (0.0 to 1.0).
        num_workers: Number of worker processes for data loading.
        target_size: Optional (height, width) to resize images to.
        seed: Random seed for reproducible train/val split.
        train_dataset: Training dataset (available after setup).
        val_dataset: Validation dataset (available after setup).

    Example:
        >>> dm = VRDDetectionDataModule(
        ...     root="datasets/vrd",
        ...     batch_size=4,
        ...     val_split=0.1,
        ... )
        >>> dm.setup(stage="fit")
        >>> train_loader = dm.train_dataloader()
        >>> val_loader = dm.val_dataloader()
    """

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 4,
        val_split: float = 0.1,
        num_workers: int | None = None,
        target_size: tuple[int, int] | None = None,
        seed: int = 42,
        background_class: bool = True,
    ) -> None:
        """Initialize VRD Detection DataModule.

        Args:
            root: Path to VRD dataset root directory containing:
                - sg_train_images/
                - annotations_train.json
                - objects.json
            batch_size: Batch size for train and val dataloaders.
            val_split: Fraction of training data to use for validation.
                Must be in range [0.0, 1.0]. Default is 0.1 (10%).
            num_workers: Number of worker processes for data loading.
                If None, uses 11 (leaves 1 core for user).
            target_size: Optional (height, width) to resize images to.
                If None, images keep original sizes (and are padded in collate).
            seed: Random seed for reproducible train/val split.
            background_class: If True, labels are 1-indexed [1, 100] with
                background at 0 (for Faster R-CNN). If False, labels are
                0-indexed [0, 99] (for EfficientDet). Default True.
        """
        super().__init__()
        self.root = Path(root) if isinstance(root, str) else root
        self.batch_size = batch_size
        self.val_split = val_split
        self.target_size = target_size
        self.seed = seed
        self.background_class = background_class

        # Default num_workers: 11 (leave 1 core for user)
        self.num_workers = 11 if num_workers is None else num_workers

        # Datasets will be created in setup()
        self.train_dataset: Dataset[tuple[Tensor, dict[str, Tensor]]] | None = None
        self.val_dataset: Dataset[tuple[Tensor, dict[str, Tensor]]] | None = None

    def setup(self, stage: str) -> None:
        """Setup datasets for the given stage.

        Creates train and validation datasets by:
        1. Loading full VRD training split
        2. Applying random_split with configured val_split ratio
        3. Wrapping with appropriate transforms (train vs val)

        Args:
            stage: Stage to setup for ("fit", "validate", "test", or "predict").
                For detection, we only use "fit" stage with train/val split.
        """
        if stage == "fit" or stage is None:
            # Load full training split (we'll split into train/val)
            full_dataset = VRDDetectionDataset(
                root=str(self.root),
                split="train",
                transform=None,  # We'll apply transforms via wrapper
                background_class=self.background_class,
            )

            # Calculate split sizes
            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size

            # Create reproducible split using Subset
            generator = torch.Generator().manual_seed(self.seed)
            # Type ignore: VRDDetectionDataset is compatible but mypy can't infer
            subsets: list[Subset[tuple[Tensor, dict[str, Tensor]]]] = random_split(
                full_dataset,  # type: ignore[arg-type]
                [train_size, val_size],
                generator=generator,
            )
            train_subset: Subset[tuple[Tensor, dict[str, Tensor]]] = subsets[0]
            val_subset: Subset[tuple[Tensor, dict[str, Tensor]]] = subsets[1]

            # Create transform functions
            train_transform = get_train_transforms(target_size=self.target_size)
            val_transform = get_val_transforms(target_size=self.target_size)

            # Wrap datasets with transforms
            self.train_dataset = _TransformedSubset(
                full_dataset, list(train_subset.indices), train_transform
            )
            self.val_dataset = _TransformedSubset(
                full_dataset, list(val_subset.indices), val_transform
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Get training dataloader.

        Returns:
            DataLoader with training dataset, shuffling enabled, and padding collate.
            Yields batches of (images, targets) where images is (B, 3, H, W) and
            targets is a list of B dicts.
        """
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")

        # pin_memory only works on CUDA, not MPS
        pin_memory = torch.cuda.is_available()
        # persistent_workers speeds up worker initialization (only if num_workers > 0)
        persistent = self.num_workers > 0

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_fn_pad,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Get validation dataloader.

        Returns:
            DataLoader with validation dataset, no shuffling, and padding collate.
            Yields batches of (images, targets) where images is (B, 3, H, W) and
            targets is a list of B dicts.
        """
        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before val_dataloader()")

        # pin_memory only works on CUDA, not MPS
        pin_memory = torch.cuda.is_available()
        # persistent_workers speeds up worker initialization (only if num_workers > 0)
        persistent = self.num_workers > 0

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_fn_pad,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )


class _TransformedSubset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Subset of dataset with transforms applied.

    Wraps a subset of indices from a dataset and applies transforms
    after loading each sample. This allows different transforms for
    train and validation splits of the same underlying dataset.

    Args:
        dataset: Base dataset to wrap.
        indices: List of indices to include in this subset.
        transform: Transform function to apply to (image, target) pairs.
    """

    def __init__(
        self,
        dataset: VRDDetectionDataset,
        indices: list[int],
        transform: Callable[[Any, dict[str, Tensor]], tuple[Tensor, dict[str, Tensor]]],
    ) -> None:
        """Initialize transformed subset.

        Args:
            dataset: Base VRDDetectionDataset.
            indices: List of indices for this subset.
            transform: Callable that takes (image_pil, target_dict) and returns
                (image_tensor, target_dict).
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        """Return number of samples in subset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        """Get transformed sample from subset.

        Args:
            idx: Index into the subset (not the original dataset).

        Returns:
            Tuple of (image_tensor, target_dict) after transform.
        """
        # Map subset index to dataset index
        dataset_idx = self.indices[idx]

        # Get raw sample from base dataset
        # VRDDetectionDataset returns (image_tensor, target) with default transform
        # We need to get the PIL image before transform
        image_id = self.dataset.image_ids[dataset_idx]
        image_path = self.dataset.image_dir / image_id

        # Load PIL image
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # Extract target (same as VRDDetectionDataset.__getitem__)
        boxes = []
        labels = []
        seen_objects = set()

        for relation in self.dataset.annotations[image_id]:
            for obj_key in ("subject", "object"):
                obj = relation[obj_key]
                category = obj["category"]
                bbox = tuple(obj["bbox"])

                obj_id = (category, bbox)
                if obj_id in seen_objects:
                    continue
                seen_objects.add(obj_id)

                ymin, ymax, xmin, xmax = bbox
                boxes.append([xmin, ymin, xmax, ymax])
                # Use dataset's background_class setting for label indexing
                if self.dataset.background_class:
                    labels.append(category + 1)  # [1, 100] for Faster R-CNN
                else:
                    labels.append(category)  # [0, 99] for EfficientDet

        # Convert to tensors
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        # Apply transform
        image_transformed, target_transformed = self.transform(image, target)

        return image_transformed, target_transformed


def _collate_fn_pad(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Collate function that pads images to same size.

    VRD images have variable sizes, so we pad them to the max height and width
    in each batch. This allows stacking into a batched tensor.

    Args:
        batch: List of (image, target) tuples where:
            - image: (3, H, W) tensor
            - target: dict with "boxes" (N, 4) and "labels" (N,)

    Returns:
        Tuple of (images, targets) where:
        - images: (B, 3, H_max, W_max) batched tensor with padding
        - targets: list of B target dicts (unchanged)

    Example:
        >>> batch = [(img1, target1), (img2, target2)]  # img1: (3, 100, 80), img2: (3, 120, 90)
        >>> images, targets = _collate_fn_pad(batch)
        >>> images.shape  # (2, 3, 120, 90) - padded to max dimensions
    """
    # Separate images and targets
    images, targets = zip(*batch)

    # Find max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    # Pad each image to max dimensions
    # F.pad expects (left, right, top, bottom) padding
    padded_images = []
    for img in images:
        h, w = img.shape[1], img.shape[2]
        pad_h = max_h - h
        pad_w = max_w - w
        # Pad: (left, right, top, bottom)
        padded = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
        padded_images.append(padded)

    # Stack into batch tensor
    batched_images = torch.stack(padded_images, dim=0)

    # Return batched images and list of targets
    return batched_images, list(targets)
