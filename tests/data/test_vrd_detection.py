"""Unit tests for VRD Detection Dataset.

Tests cover:
- Dataset initialization (train/test splits)
- Data loading and format validation
- Box coordinate conversion (yxyx -> xyxy)
- Label indexing (0-indexed -> 1-indexed with background)
- Edge cases (empty annotations, invalid paths)
"""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor

from src.data.vrd_detection import VRDDetectionDataset


class TestVRDDetectionDatasetInit:
    """Test dataset initialization."""

    def test_init_train_split(self, vrd_dataset_root: Path) -> None:
        """Test initialization with train split."""
        dataset = VRDDetectionDataset(root=str(vrd_dataset_root), split="train")
        
        assert len(dataset) > 0
        assert dataset.split == "train"
        assert dataset.num_classes == 101  # 100 objects + background

    def test_init_test_split(self, vrd_dataset_root: Path) -> None:
        """Test initialization with test split."""
        dataset = VRDDetectionDataset(root=str(vrd_dataset_root), split="test")
        
        assert len(dataset) > 0
        assert dataset.split == "test"

    def test_init_invalid_split_raises_error(self, vrd_dataset_root: Path) -> None:
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be"):
            VRDDetectionDataset(root=str(vrd_dataset_root), split="validation")

    def test_init_missing_annotations_raises_error(self, tmp_path: Path) -> None:
        """Test that missing annotations file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            VRDDetectionDataset(root=str(tmp_path), split="train")


class TestVRDDetectionDatasetGetItem:
    """Test dataset item retrieval."""

    def test_getitem_returns_image_and_target(self, vrd_dataset_root: Path) -> None:
        """Test __getitem__ returns correct format."""
        dataset = VRDDetectionDataset(root=str(vrd_dataset_root), split="train")
        
        image, target = dataset[0]
        
        # Check image is a tensor
        assert isinstance(image, Tensor)
        assert image.ndim == 3  # (C, H, W)
        assert image.shape[0] == 3  # RGB
        
        # Check target has required keys
        assert isinstance(target, dict)
        assert "boxes" in target
        assert "labels" in target

    def test_boxes_are_xyxy_format(self, vrd_dataset_root: Path) -> None:
        """Test boxes are converted to xyxy format."""
        dataset = VRDDetectionDataset(root=str(vrd_dataset_root), split="train")
        
        _, target = dataset[0]
        boxes = target["boxes"]
        
        # Check shape
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        
        # Check coordinates are valid (xmin < xmax, ymin < ymax)
        assert torch.all(boxes[:, 0] < boxes[:, 2])  # xmin < xmax
        assert torch.all(boxes[:, 1] < boxes[:, 3])  # ymin < ymax
        
        # Check coordinates are non-negative
        assert torch.all(boxes >= 0)

    def test_labels_are_positive(self, vrd_dataset_root: Path) -> None:
        """Test labels are >= 1 (background is 0)."""
        dataset = VRDDetectionDataset(root=str(vrd_dataset_root), split="train")
        
        _, target = dataset[0]
        labels = target["labels"]
        
        assert labels.ndim == 1
        assert torch.all(labels >= 1)
        assert torch.all(labels <= 100)  # VRD has 100 object classes

    def test_consistent_num_boxes_and_labels(self, vrd_dataset_root: Path) -> None:
        """Test number of boxes matches number of labels."""
        dataset = VRDDetectionDataset(root=str(vrd_dataset_root), split="train")
        
        _, target = dataset[0]
        
        assert len(target["boxes"]) == len(target["labels"])


class TestVRDDetectionDatasetProperties:
    """Test dataset properties."""

    def test_num_classes(self, vrd_dataset_root: Path) -> None:
        """Test num_classes property returns 101."""
        dataset = VRDDetectionDataset(root=str(vrd_dataset_root), split="train")
        
        assert dataset.num_classes == 101

    def test_class_names_length(self, vrd_dataset_root: Path) -> None:
        """Test class_names has 101 entries (background + 100 objects)."""
        dataset = VRDDetectionDataset(root=str(vrd_dataset_root), split="train")
        
        assert len(dataset.class_names) == 101
        assert dataset.class_names[0] == "background"

    def test_class_names_match_objects_json(self, vrd_dataset_root: Path) -> None:
        """Test class names match objects.json."""
        dataset = VRDDetectionDataset(root=str(vrd_dataset_root), split="train")
        
        # Load objects.json
        objects_path = vrd_dataset_root / "objects.json"
        with open(objects_path) as f:
            objects = json.load(f)
        
        # Check alignment (skip background at index 0)
        assert dataset.class_names[1:] == objects


class TestVRDDetectionDatasetEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_images_with_no_boxes(self, vrd_dataset_with_empty_annotations: Path) -> None:
        """Test dataset handles images with no annotations gracefully."""
        dataset = VRDDetectionDataset(root=str(vrd_dataset_with_empty_annotations), split="train")
        
        image, target = dataset[0]
        
        # Should return empty tensors with correct shape
        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)

    def test_transform_is_applied(self, vrd_dataset_root: Path) -> None:
        """Test custom transform is applied to images."""
        def dummy_transform(img):
            """Convert to tensor and add 1 to all values."""
            from torchvision.transforms import functional as F
            return F.to_tensor(img) + 1.0
        
        dataset = VRDDetectionDataset(
            root=str(vrd_dataset_root), 
            split="train",
            transform=dummy_transform
        )
        
        image, _ = dataset[0]
        
        # Values should be > 1.0 due to transform
        assert torch.any(image > 1.0)


# Fixtures

@pytest.fixture
def vrd_dataset_root() -> Path:
    """Return path to VRD dataset root.
    
    This assumes the dataset is at the standard location.
    If not present, tests will be skipped.
    """
    dataset_path = Path("/Users/aszfalt/Projects/research/sgg_v2/datasets/vrd")
    
    if not dataset_path.exists():
        pytest.skip("VRD dataset not found at expected location")
    
    return dataset_path


@pytest.fixture
def vrd_dataset_with_empty_annotations(tmp_path: Path) -> Path:
    """Create a minimal VRD dataset with an image that has no annotations."""
    # Create directory structure
    images_dir = tmp_path / "sg_train_images"
    images_dir.mkdir()
    
    # Create a dummy image
    img = Image.new("RGB", (100, 100), color="red")
    img.save(images_dir / "test_image.jpg")
    
    # Create annotations with empty relationships (but image exists)
    annotations = {
        "test_image.jpg": []
    }
    
    with open(tmp_path / "annotations_train.json", "w") as f:
        json.dump(annotations, f)
    
    # Create objects.json
    objects = ["person", "car"]  # Just 2 classes for testing
    with open(tmp_path / "objects.json", "w") as f:
        json.dump(objects, f)
    
    return tmp_path
