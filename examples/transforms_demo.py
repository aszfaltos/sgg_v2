"""Demo of using transforms with VRD detection dataset.

Shows how to integrate training and validation transforms with the VRD dataset.
"""

from pathlib import Path


from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.vrd_detection import VRDDetectionDataset


def collate_fn(batch):
    """Collate function for variable-sized images and targets.

    Since VRD images have different sizes, we can't stack them directly.
    Instead, return lists of images and targets.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def main():
    # Setup paths
    vrd_root = Path("datasets/vrd")

    # Create transforms
    train_transform = get_train_transforms(target_size=(800, 1200))
    val_transform = get_val_transforms(target_size=(800, 1200))

    # Create datasets with transforms
    # Pass PIL image to transform, which will handle both image and target
    train_dataset = VRDDetectionDataset(
        root=str(vrd_root),
        split="train",
        transform=None,  # We'll apply transform manually to show the pattern
    )

    # Example: Manual transform application
    image, target = train_dataset[0]
    print(f"Original image shape: {image.shape}")
    print(f"Original boxes: {target['boxes']}")

    # For proper integration, create a wrapper
    class TransformedDataset:
        """Wrapper to apply (image, target) transforms to a detection dataset."""

        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # Get PIL image and target from base dataset
            image_pil = self.dataset[idx][0]  # This would need dataset modification
            target = self.dataset[idx][1]

            # Apply transform that handles both
            return self.transform(image_pil, target)

    # Better approach: modify VRDDetectionDataset to accept transform that gets
    # (PIL image, target) and returns (tensor, target)

    print("\nTransforms created successfully!")
    print(f"Train transform: {train_transform}")
    print(f"Val transform: {val_transform}")


if __name__ == "__main__":
    main()
