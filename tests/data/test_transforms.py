"""Tests for detection data augmentation transforms."""

import torch
from PIL import Image

from src.data.transforms import get_train_transforms, get_val_transforms


class TestGetTrainTransforms:
    """Tests for training transforms."""

    def test_returns_callable(self):
        """get_train_transforms returns a callable."""
        transform = get_train_transforms()
        assert callable(transform)

    def test_handles_image_and_target(self):
        """Training transform handles both image and target dict."""
        transform = get_train_transforms()

        # Create sample image and target
        image = Image.new("RGB", (100, 80))
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }

        # Apply transform
        image_out, target_out = transform(image, target)

        # Check output types
        assert isinstance(image_out, torch.Tensor)
        assert isinstance(target_out, dict)
        assert "boxes" in target_out
        assert "labels" in target_out

    def test_normalizes_image(self):
        """Training transform normalizes image to ImageNet stats."""
        transform = get_train_transforms()

        # Create white image (RGB values = 1.0)
        image = Image.new("RGB", (100, 80), (255, 255, 255))
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

        image_out, _ = transform(image, target)

        # Check output is normalized (not in [0, 1] range)
        assert image_out.min() < 0 or image_out.max() > 1
        # Check ImageNet normalization roughly applied
        # White image (1.0) with ImageNet mean/std should give positive values
        assert image_out.shape == (3, 80, 100)  # (C, H, W)

    def test_preserves_boxes_format(self):
        """Training transform preserves xyxy box format."""
        transform = get_train_transforms()

        image = Image.new("RGB", (100, 80))
        boxes = torch.tensor(
            [[10.0, 20.0, 50.0, 60.0], [5.0, 5.0, 95.0, 75.0]], dtype=torch.float32
        )
        target = {"boxes": boxes, "labels": torch.tensor([1, 2], dtype=torch.int64)}

        _, target_out = transform(image, target)

        # Boxes should still be 4 coordinates per box
        assert target_out["boxes"].shape == (2, 4)
        # Boxes should be valid (xmin < xmax, ymin < ymax)
        for box in target_out["boxes"]:
            assert box[0] < box[2]  # xmin < xmax
            assert box[1] < box[3]  # ymin < ymax

    def test_applies_horizontal_flip_to_boxes(self):
        """Horizontal flip correctly transforms box coordinates."""
        # We need to test this multiple times since flip is random
        # Set seed for reproducibility
        torch.manual_seed(42)

        transform = get_train_transforms()

        image = Image.new("RGB", (100, 80))
        boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]], dtype=torch.float32)
        target = {"boxes": boxes, "labels": torch.tensor([1], dtype=torch.int64)}

        # Run multiple times to catch flips
        flipped_detected = False
        for _ in range(50):
            _, target_out = transform(image, target)
            box_out = target_out["boxes"][0]

            # Check if box was flipped (x coordinates change)
            # Original box: [10, 20, 50, 60]
            # If flipped with width=100: [50, 20, 90, 60] (100-50, y, 100-10, y)
            if not torch.allclose(box_out[:2], boxes[0][:2], atol=1e-4):
                flipped_detected = True
                # Y coordinates should stay the same
                assert torch.allclose(box_out[[1, 3]], boxes[0][[1, 3]], atol=1e-4)
                # X coordinates should be flipped
                assert box_out[0] != boxes[0][0]
                break

        # We should have detected at least one flip in 50 tries
        # (probability of no flips in 50 tries: 0.5^50 ≈ 0)
        assert flipped_detected, "No horizontal flip detected in 50 attempts"

    def test_preserves_labels(self):
        """Training transform preserves labels unchanged."""
        transform = get_train_transforms()

        image = Image.new("RGB", (100, 80))
        labels = torch.tensor([1, 2, 3], dtype=torch.int64)
        target = {
            "boxes": torch.tensor(
                [
                    [10.0, 20.0, 50.0, 60.0],
                    [5.0, 5.0, 20.0, 20.0],
                    [70.0, 50.0, 90.0, 70.0],
                ],
                dtype=torch.float32,
            ),
            "labels": labels,
        }

        _, target_out = transform(image, target)

        assert torch.equal(target_out["labels"], labels)

    def test_handles_empty_boxes(self):
        """Training transform handles empty bounding boxes."""
        transform = get_train_transforms()

        image = Image.new("RGB", (100, 80))
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

        image_out, target_out = transform(image, target)

        assert image_out.shape == (3, 80, 100)
        assert target_out["boxes"].shape == (0, 4)
        assert target_out["labels"].shape == (0,)

    def test_with_resize(self):
        """Training transform with target_size resizes image and boxes."""
        target_size = (120, 160)  # (H, W)
        transform = get_train_transforms(target_size=target_size)

        image = Image.new("RGB", (100, 80))  # (W, H)
        boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]], dtype=torch.float32)
        target = {"boxes": boxes, "labels": torch.tensor([1], dtype=torch.int64)}

        image_out, target_out = transform(image, target)

        # Image should be resized
        assert image_out.shape == (3, 120, 160)  # (C, H, W)

        # Boxes should be scaled proportionally
        # Original: 100x80, New: 160x120
        # Scale factors: x=160/100=1.6, y=120/80=1.5
        box_out = target_out["boxes"][0]
        expected_box = torch.tensor([16.0, 30.0, 80.0, 90.0], dtype=torch.float32)
        assert torch.allclose(box_out, expected_box, atol=1.0)


class TestGetValTransforms:
    """Tests for validation transforms."""

    def test_returns_callable(self):
        """get_val_transforms returns a callable."""
        transform = get_val_transforms()
        assert callable(transform)

    def test_handles_image_and_target(self):
        """Validation transform handles both image and target dict."""
        transform = get_val_transforms()

        image = Image.new("RGB", (100, 80))
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }

        image_out, target_out = transform(image, target)

        assert isinstance(image_out, torch.Tensor)
        assert isinstance(target_out, dict)
        assert "boxes" in target_out
        assert "labels" in target_out

    def test_normalizes_image(self):
        """Validation transform normalizes image to ImageNet stats."""
        transform = get_val_transforms()

        image = Image.new("RGB", (100, 80), (255, 255, 255))
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

        image_out, _ = transform(image, target)

        # Check output is normalized
        assert image_out.min() < 0 or image_out.max() > 1
        assert image_out.shape == (3, 80, 100)

    def test_no_random_augmentations(self):
        """Validation transform is deterministic (no random augmentations)."""
        transform = get_val_transforms()

        image = Image.new("RGB", (100, 80))
        boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]], dtype=torch.float32)
        target = {"boxes": boxes, "labels": torch.tensor([1], dtype=torch.int64)}

        # Run multiple times - should get same result
        results = []
        for _ in range(5):
            image_out, target_out = transform(image, target)
            results.append((image_out.clone(), target_out["boxes"].clone()))

        # All results should be identical
        for i in range(1, len(results)):
            assert torch.equal(results[0][0], results[i][0])
            assert torch.equal(results[0][1], results[i][1])

    def test_preserves_boxes_exactly(self):
        """Validation transform preserves boxes without modification (no augmentation)."""
        transform = get_val_transforms()

        image = Image.new("RGB", (100, 80))
        boxes = torch.tensor(
            [[10.0, 20.0, 50.0, 60.0], [5.0, 5.0, 95.0, 75.0]], dtype=torch.float32
        )
        target = {"boxes": boxes, "labels": torch.tensor([1, 2], dtype=torch.int64)}

        _, target_out = transform(image, target)

        # Boxes should be identical (no augmentation)
        assert torch.equal(target_out["boxes"], boxes)

    def test_with_resize(self):
        """Validation transform with target_size resizes image and boxes."""
        target_size = (120, 160)  # (H, W)
        transform = get_val_transforms(target_size=target_size)

        image = Image.new("RGB", (100, 80))  # (W, H)
        boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]], dtype=torch.float32)
        target = {"boxes": boxes, "labels": torch.tensor([1], dtype=torch.int64)}

        image_out, target_out = transform(image, target)

        # Image should be resized
        assert image_out.shape == (3, 120, 160)

        # Boxes should be scaled proportionally
        box_out = target_out["boxes"][0]
        expected_box = torch.tensor([16.0, 30.0, 80.0, 90.0], dtype=torch.float32)
        assert torch.allclose(box_out, expected_box, atol=1.0)

    def test_handles_empty_boxes(self):
        """Validation transform handles empty bounding boxes."""
        transform = get_val_transforms()

        image = Image.new("RGB", (100, 80))
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

        image_out, target_out = transform(image, target)

        assert image_out.shape == (3, 80, 100)
        assert target_out["boxes"].shape == (0, 4)
        assert target_out["labels"].shape == (0,)


class TestTransformsIntegration:
    """Integration tests with VRDDetectionDataset format."""

    def test_train_transforms_with_vrd_format(self):
        """Training transforms work with VRD dataset output format."""
        transform = get_train_transforms()

        # Simulate VRD dataset output
        image = Image.new("RGB", (100, 80))
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),  # 1-indexed
        }

        image_out, target_out = transform(image, target)

        # Should preserve data types
        assert image_out.dtype == torch.float32
        assert target_out["boxes"].dtype == torch.float32
        assert target_out["labels"].dtype == torch.int64

    def test_val_transforms_with_vrd_format(self):
        """Validation transforms work with VRD dataset output format."""
        transform = get_val_transforms()

        image = Image.new("RGB", (100, 80))
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }

        image_out, target_out = transform(image, target)

        assert image_out.dtype == torch.float32
        assert target_out["boxes"].dtype == torch.float32
        assert target_out["labels"].dtype == torch.int64
