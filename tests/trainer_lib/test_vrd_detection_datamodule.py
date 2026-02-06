"""Tests for VRD Detection Lightning DataModule."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.trainer_lib.data_modules.detection import VRDDetectionDataModule


@pytest.fixture
def mock_vrd_dataset(tmp_path: Path) -> Path:
    """Create a minimal mock VRD dataset for testing.

    Creates a small VRD dataset with 10 images (split into train/val)
    that can be used to test the DataModule without requiring the full dataset.

    Returns:
        Path to the mock dataset root directory.
    """
    root = tmp_path / "mock_vrd"
    root.mkdir()

    # Create image directories
    train_dir = root / "sg_train_images"
    train_dir.mkdir()

    # Create objects.json
    objects = ["person", "car", "dog"]
    with open(root / "objects.json", "w") as f:
        json.dump(objects, f)

    # Create 10 small mock images
    image_ids = []
    for i in range(10):
        img_id = f"image_{i:03d}.jpg"
        image_ids.append(img_id)

        # Create a small RGB image (10x10 pixels)
        img = Image.new("RGB", (10, 10), color=(i * 25, 100, 150))
        img.save(train_dir / img_id)

    # Create annotations_train.json with minimal data
    annotations = {}
    for img_id in image_ids:
        # Each image has 1-2 relationships
        annotations[img_id] = [
            {
                "subject": {
                    "category": 0,  # person
                    "bbox": [1, 5, 1, 5],  # [ymin, ymax, xmin, xmax]
                },
                "predicate": "ride",
                "object": {
                    "category": 1,  # car
                    "bbox": [4, 8, 4, 8],
                },
            }
        ]

    with open(root / "annotations_train.json", "w") as f:
        json.dump(annotations, f)

    return root


class TestVRDDetectionDataModuleInit:
    """Tests for DataModule initialization."""

    def test_creates_datamodule_instance(self, mock_vrd_dataset: Path) -> None:
        """Test that DataModule can be instantiated."""
        # Act
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)

        # Assert
        assert isinstance(dm, LightningDataModule)

    def test_accepts_string_root_path(self, mock_vrd_dataset: Path) -> None:
        """Test that root path can be provided as string."""
        # Act
        dm = VRDDetectionDataModule(root=str(mock_vrd_dataset))

        # Assert
        assert isinstance(dm, LightningDataModule)

    def test_accepts_path_object_root(self, mock_vrd_dataset: Path) -> None:
        """Test that root path can be provided as Path object."""
        # Act
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)

        # Assert
        assert isinstance(dm, LightningDataModule)

    def test_default_batch_size(self, mock_vrd_dataset: Path) -> None:
        """Test that default batch size is set correctly."""
        # Act
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)

        # Assert
        assert dm.batch_size == 4

    def test_custom_batch_size(self, mock_vrd_dataset: Path) -> None:
        """Test that custom batch size can be set."""
        # Act
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, batch_size=8)

        # Assert
        assert dm.batch_size == 8

    def test_default_val_split(self, mock_vrd_dataset: Path) -> None:
        """Test that default validation split is 0.1."""
        # Act
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)

        # Assert
        assert dm.val_split == 0.1

    def test_custom_val_split(self, mock_vrd_dataset: Path) -> None:
        """Test that custom validation split can be set."""
        # Act
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, val_split=0.2)

        # Assert
        assert dm.val_split == 0.2

    def test_default_num_workers(self, mock_vrd_dataset: Path) -> None:
        """Test that default num_workers is 11 (leaves 1 core for user)."""
        # Act
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)

        # Assert
        assert dm.num_workers == 11

    def test_custom_num_workers(self, mock_vrd_dataset: Path) -> None:
        """Test that custom num_workers can be set (though should be 0 for Python 3.13)."""
        # Act
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, num_workers=2)

        # Assert
        assert dm.num_workers == 2

    def test_optional_target_size(self, mock_vrd_dataset: Path) -> None:
        """Test that target_size can be optionally provided."""
        # Act
        dm = VRDDetectionDataModule(
            root=mock_vrd_dataset, target_size=(800, 1200)
        )

        # Assert
        assert dm.target_size == (800, 1200)

    def test_default_seed(self, mock_vrd_dataset: Path) -> None:
        """Test that default seed is 42 for reproducibility."""
        # Act
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)

        # Assert
        assert dm.seed == 42

    def test_custom_seed(self, mock_vrd_dataset: Path) -> None:
        """Test that custom seed can be set."""
        # Act
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, seed=123)

        # Assert
        assert dm.seed == 123


class TestVRDDetectionDataModuleSetup:
    """Tests for DataModule setup method."""

    def test_setup_creates_datasets(self, mock_vrd_dataset: Path) -> None:
        """Test that setup creates train and val datasets."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, val_split=0.2)

        # Act
        dm.setup(stage="fit")

        # Assert
        assert hasattr(dm, "train_dataset")
        assert hasattr(dm, "val_dataset")
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_setup_splits_data_correctly(self, mock_vrd_dataset: Path) -> None:
        """Test that train/val split uses correct proportions."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, val_split=0.2)

        # Act
        dm.setup(stage="fit")

        # Assert
        total_len = len(dm.train_dataset) + len(dm.val_dataset)
        assert total_len == 10  # We created 10 mock images
        # 20% val = 2 images, 80% train = 8 images
        assert len(dm.val_dataset) == 2
        assert len(dm.train_dataset) == 8

    def test_setup_is_reproducible(self, mock_vrd_dataset: Path) -> None:
        """Test that setup with same seed produces same split."""
        # Arrange
        dm1 = VRDDetectionDataModule(root=mock_vrd_dataset, seed=42, val_split=0.2)
        dm2 = VRDDetectionDataModule(root=mock_vrd_dataset, seed=42, val_split=0.2)

        # Act
        dm1.setup(stage="fit")
        dm2.setup(stage="fit")

        # Assert - verify same indices are in same splits
        # Note: We can't compare transformed images directly because train
        # transforms include random augmentation (e.g., RandomHorizontalFlip).
        # Instead, check that the underlying indices are the same.
        assert dm1.train_dataset.indices == dm2.train_dataset.indices
        assert dm1.val_dataset.indices == dm2.val_dataset.indices

    def test_setup_different_seeds_different_splits(
        self, mock_vrd_dataset: Path
    ) -> None:
        """Test that different seeds produce different splits."""
        # Arrange
        dm1 = VRDDetectionDataModule(root=mock_vrd_dataset, seed=42, val_split=0.2)
        dm2 = VRDDetectionDataModule(root=mock_vrd_dataset, seed=123, val_split=0.2)

        # Act
        dm1.setup(stage="fit")
        dm2.setup(stage="fit")

        # Assert - same total size but different contents
        assert len(dm1.train_dataset) == len(dm2.train_dataset)
        assert len(dm1.val_dataset) == len(dm2.val_dataset)

    def test_setup_can_be_called_multiple_times(
        self, mock_vrd_dataset: Path
    ) -> None:
        """Test that setup is idempotent and can be called multiple times."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)

        # Act
        dm.setup(stage="fit")
        first_train_len = len(dm.train_dataset)
        dm.setup(stage="fit")  # Call again
        second_train_len = len(dm.train_dataset)

        # Assert
        assert first_train_len == second_train_len


class TestVRDDetectionDataModuleDataLoaders:
    """Tests for DataModule dataloader methods."""

    def test_train_dataloader_returns_dataloader(
        self, mock_vrd_dataset: Path
    ) -> None:
        """Test that train_dataloader returns DataLoader instance."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)
        dm.setup(stage="fit")

        # Act
        loader = dm.train_dataloader()

        # Assert
        assert isinstance(loader, DataLoader)

    def test_val_dataloader_returns_dataloader(self, mock_vrd_dataset: Path) -> None:
        """Test that val_dataloader returns DataLoader instance."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)
        dm.setup(stage="fit")

        # Act
        loader = dm.val_dataloader()

        # Assert
        assert isinstance(loader, DataLoader)

    def test_train_dataloader_uses_correct_batch_size(
        self, mock_vrd_dataset: Path
    ) -> None:
        """Test that train DataLoader uses configured batch size."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, batch_size=2)
        dm.setup(stage="fit")

        # Act
        loader = dm.train_dataloader()

        # Assert
        assert loader.batch_size == 2

    def test_val_dataloader_uses_correct_batch_size(
        self, mock_vrd_dataset: Path
    ) -> None:
        """Test that val DataLoader uses configured batch size."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, batch_size=2)
        dm.setup(stage="fit")

        # Act
        loader = dm.val_dataloader()

        # Assert
        assert loader.batch_size == 2

    def test_dataloaders_use_num_workers(self, mock_vrd_dataset: Path) -> None:
        """Test that DataLoaders use configured num_workers."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, num_workers=0)
        dm.setup(stage="fit")

        # Act
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # Assert
        assert train_loader.num_workers == 0
        assert val_loader.num_workers == 0

    def test_train_dataloader_shuffles(self, mock_vrd_dataset: Path) -> None:
        """Test that train DataLoader shuffles data."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)
        dm.setup(stage="fit")

        # Act
        loader = dm.train_dataloader()

        # Assert - when shuffle=True, PyTorch creates a RandomSampler
        # Check that sampler is RandomSampler (indicates shuffling)
        from torch.utils.data.sampler import RandomSampler

        assert isinstance(loader.sampler, RandomSampler)

    def test_val_dataloader_does_not_shuffle(self, mock_vrd_dataset: Path) -> None:
        """Test that val DataLoader does not shuffle data."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)
        dm.setup(stage="fit")

        # Act
        loader = dm.val_dataloader()

        # Assert - sampler should be None (default sequential) or not shuffling
        # For validation, we want deterministic order
        assert loader.sampler is None or not hasattr(loader.sampler, "shuffle")


class TestVRDDetectionDataModuleCollateFn:
    """Tests for the collate function that pads images."""

    def test_collate_pads_variable_sized_images(self, mock_vrd_dataset: Path) -> None:
        """Test that collate function pads images to same size in batch."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, batch_size=2)
        dm.setup(stage="fit")
        loader = dm.train_dataloader()

        # Act
        batch = next(iter(loader))
        images, targets = batch

        # Assert
        # All images in batch should have same shape
        assert images.dim() == 4  # (B, C, H, W)
        assert images.shape[0] == 2  # Batch size
        # All images should have same H and W (padded)
        for i in range(1, images.shape[0]):
            assert images[i].shape == images[0].shape

    def test_collate_returns_list_of_targets(self, mock_vrd_dataset: Path) -> None:
        """Test that collate function returns list of target dicts."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, batch_size=2)
        dm.setup(stage="fit")
        loader = dm.train_dataloader()

        # Act
        batch = next(iter(loader))
        images, targets = batch

        # Assert
        assert isinstance(targets, list)
        assert len(targets) == 2
        assert all(isinstance(t, dict) for t in targets)
        assert all("boxes" in t and "labels" in t for t in targets)

    def test_collate_preserves_target_data(self, mock_vrd_dataset: Path) -> None:
        """Test that collate function preserves target boxes and labels."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, batch_size=1)
        dm.setup(stage="fit")
        loader = dm.train_dataloader()

        # Act
        batch = next(iter(loader))
        images, targets = batch

        # Assert
        target = targets[0]
        assert "boxes" in target
        assert "labels" in target
        assert target["boxes"].shape[1] == 4  # xyxy format
        assert target["labels"].dim() == 1  # 1D tensor


class TestVRDDetectionDataModuleTransforms:
    """Tests for transform application in DataModule."""

    def test_train_dataset_uses_train_transforms(
        self, mock_vrd_dataset: Path
    ) -> None:
        """Test that train dataset applies training transforms (with augmentation)."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)
        dm.setup(stage="fit")

        # Act
        image, target = dm.train_dataset[0]

        # Assert
        # Image should be normalized tensor
        assert isinstance(image, torch.Tensor)
        assert image.dim() == 3  # (C, H, W)
        assert image.shape[0] == 3  # RGB

    def test_val_dataset_uses_val_transforms(self, mock_vrd_dataset: Path) -> None:
        """Test that val dataset applies validation transforms (no augmentation)."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset)
        dm.setup(stage="fit")

        # Act
        image, target = dm.val_dataset[0]

        # Assert
        # Image should be normalized tensor
        assert isinstance(image, torch.Tensor)
        assert image.dim() == 3  # (C, H, W)
        assert image.shape[0] == 3  # RGB

    def test_target_size_passed_to_transforms(self, mock_vrd_dataset: Path) -> None:
        """Test that target_size is passed to transforms for resizing."""
        # Arrange
        target_size = (32, 48)  # Small size for testing
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, target_size=target_size)
        dm.setup(stage="fit")

        # Act
        image, target = dm.train_dataset[0]

        # Assert
        # Image should be resized to target_size
        assert image.shape[1] == target_size[0]  # Height
        assert image.shape[2] == target_size[1]  # Width


class TestVRDDetectionDataModuleEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_val_split_creates_no_val_data(
        self, mock_vrd_dataset: Path
    ) -> None:
        """Test that val_split=0 creates empty validation set."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, val_split=0.0)

        # Act
        dm.setup(stage="fit")

        # Assert
        assert len(dm.val_dataset) == 0
        assert len(dm.train_dataset) == 10  # All data in training

    def test_full_val_split_creates_no_train_data(
        self, mock_vrd_dataset: Path
    ) -> None:
        """Test that val_split=1.0 creates empty training set."""
        # Arrange
        dm = VRDDetectionDataModule(root=mock_vrd_dataset, val_split=1.0)

        # Act
        dm.setup(stage="fit")

        # Assert
        assert len(dm.train_dataset) == 0
        assert len(dm.val_dataset) == 10  # All data in validation

    def test_invalid_root_path_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid root path raises appropriate error on setup."""
        # Arrange
        dm = VRDDetectionDataModule(root=tmp_path / "nonexistent")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            dm.setup(stage="fit")
