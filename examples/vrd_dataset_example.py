"""Example usage of VRD Detection Dataset.

Demonstrates:
- Loading train/test splits
- Iterating through dataset
- Using with DataLoader
- Custom transforms
"""

from torch.utils.data import DataLoader

from src.data import VRDDetectionDataset


def collate_fn(batch):
    """Collate function that handles variable-sized images."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def main():
    """Run dataset examples."""
    # Example 1: Basic usage
    print("Example 1: Basic dataset usage")
    print("-" * 50)
    
    dataset = VRDDetectionDataset(root="datasets/vrd", split="train")
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class names (first 5): {dataset.class_names[:5]}")
    
    # Load a sample
    image, target = dataset[0]
    print(f"\nSample image shape: {image.shape}")
    print(f"Number of boxes: {len(target['boxes'])}")
    print(f"Labels: {target['labels'].tolist()}")
    
    # Example 2: DataLoader usage
    print("\n\nExample 2: DataLoader usage")
    print("-" * 50)
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Required per project constraints
        collate_fn=collate_fn,
    )
    
    images, targets = next(iter(dataloader))
    print(f"Batch size: {len(images)}")
    print(f"Image 0 shape: {images[0].shape}")
    print(f"Image 1 shape: {images[1].shape}")
    print(f"Target 0 boxes: {len(targets[0]['boxes'])}")
    print(f"Target 1 boxes: {len(targets[1]['boxes'])}")
    
    # Example 3: Custom transform
    print("\n\nExample 3: Custom transform")
    print("-" * 50)
    
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((600, 800)),
        transforms.ToTensor(),
    ])
    
    dataset_with_transform = VRDDetectionDataset(
        root="datasets/vrd",
        split="train",
        transform=transform,
    )
    
    image, target = dataset_with_transform[0]
    print(f"Transformed image shape: {image.shape}")
    print("Expected shape: (3, 600, 800)")
    
    # Example 4: Statistics
    print("\n\nExample 4: Dataset statistics")
    print("-" * 50)
    
    test_dataset = VRDDetectionDataset(root="datasets/vrd", split="test")
    
    num_images = len(test_dataset)
    num_boxes = []
    all_labels = set()
    
    for i in range(min(100, num_images)):
        _, target = test_dataset[i]
        num_boxes.append(len(target["boxes"]))
        all_labels.update(target["labels"].tolist())
    
    print(f"Test dataset size: {num_images}")
    print(f"Average boxes per image (first 100): {sum(num_boxes) / len(num_boxes):.2f}")
    print(f"Min boxes: {min(num_boxes)}")
    print(f"Max boxes: {max(num_boxes)}")
    print(f"Unique classes seen: {len(all_labels)}")


if __name__ == "__main__":
    main()
