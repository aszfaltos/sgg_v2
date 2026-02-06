"""Example of using the logger utility with PyTorch Lightning.

This demonstrates how to use create_aim_logger with a Lightning Trainer.
"""

from pytorch_lightning import Trainer

from src.trainer_lib import create_aim_logger


def main() -> None:
    """Example usage of the logger utility."""
    # Create a logger for the experiment
    logger = create_aim_logger(
        experiment_name="vrd_detection",
        run_name="baseline_fasterrcnn",
        repo_path=".aim",  # Default location
    )

    # Use with Lightning Trainer
    _ = Trainer(
        logger=logger,
        max_epochs=10,
        # ... other trainer args
    )

    # The trainer will automatically log:
    # - All metrics logged via self.log() in your LightningModule
    # - Hyperparameters via self.save_hyperparameters()
    # - Learning rate schedules
    # - System metrics

    # To view logs, run:
    # tensorboard --logdir=.aim

    print(f"Logger configured for experiment: {logger.name}")
    print(f"Log directory: {logger.log_dir}")


if __name__ == "__main__":
    main()
