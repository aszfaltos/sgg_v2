"""Example usage of VRDDetectionDataModule.

Demonstrates how to use the VRD Detection DataModule with PyTorch Lightning
for training object detection models.
"""



from src.trainer_lib.data_modules import VRDDetectionDataModule


def main() -> None:
    """Example of using VRDDetectionDataModule with Lightning Trainer."""
    # Create DataModule
    dm = VRDDetectionDataModule(
        root="datasets/vrd",
        batch_size=4,
        val_split=0.1,  # 10% for validation
        num_workers=0,  # Python 3.13 constraint
        target_size=(800, 1200),  # Optional resize
        seed=42,  # Reproducible splits
    )

    # Setup datasets
    dm.setup(stage="fit")

    # Get dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Example: iterate through one batch
    for images, targets in train_loader:
        print("\nBatch shapes:")
        print(f"  Images: {images.shape}")  # (B, 3, H, W)
        print(f"  Targets: {len(targets)} dicts")
        print(f"  First target boxes: {targets[0]['boxes'].shape}")
        print(f"  First target labels: {targets[0]['labels'].shape}")
        break

    # Example: use with Lightning Trainer
    # trainer = Trainer(
    #     max_epochs=10,
    #     accelerator="auto",
    #     devices=1,
    #     callbacks=[
    #         ModelCheckpoint(
    #             monitor="val/mAP",
    #             mode="max",
    #             save_top_k=1,
    #         ),
    #     ],
    # )
    # trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
