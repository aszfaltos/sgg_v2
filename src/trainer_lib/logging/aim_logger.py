"""Experiment tracking logger utility for PyTorch Lightning.

Provides a convenient factory function to create and configure loggers
for use with PyTorch Lightning Trainer.

Note: Currently uses TensorBoard as Aim doesn't support Python 3.13 yet.
Will be migrated to Aim once aimrocks provides cp313 wheels.
"""

from pathlib import Path

from pytorch_lightning.loggers import Logger, TensorBoardLogger


def create_aim_logger(
    experiment_name: str,
    run_name: str | None = None,
    repo_path: str | Path = ".aim",
) -> Logger:
    """Create configured logger for Lightning training.

    Creates a TensorBoardLogger instance configured for PyTorch Lightning training.
    TensorBoard provides comprehensive experiment tracking including metrics,
    hyperparameters, and model graphs.

    The logger automatically tracks:
    - Training and validation metrics (loss, mAP, etc.)
    - Learning rate schedules
    - Hyperparameters
    - System metrics (GPU, CPU usage)
    - Model graphs and histograms

    Args:
        experiment_name: Name of the experiment. Used to organize runs
            in TensorBoard. For example: "vrd_detection", "vg_sgg".
        run_name: Optional name for this specific run. If None, TensorBoard will
            use a timestamp. Use descriptive names like "baseline",
            "with_attention", "lr_sweep_001" for easier identification.
        repo_path: Path to the logging directory. Can be a string or Path.
            Defaults to ".aim" for future Aim compatibility. Logs will be
            stored in repo_path/experiment_name/run_name/.

    Returns:
        Configured Logger instance ready to use with Lightning Trainer.

    Example:
        Basic usage with Trainer:
        >>> logger = create_aim_logger(
        ...     experiment_name="vrd_detection",
        ...     run_name="baseline_fasterrcnn",
        ... )
        >>> trainer = Trainer(logger=logger)

        Custom log location:
        >>> logger = create_aim_logger(
        ...     experiment_name="my_experiment",
        ...     repo_path="/data/experiments/logs",
        ... )

        Multiple experiments in same directory:
        >>> logger1 = create_aim_logger(experiment_name="exp1")
        >>> logger2 = create_aim_logger(experiment_name="exp2")
        >>> # Both experiments tracked in same base directory

    Notes:
        - The log directory is created automatically if it doesn't exist
        - Use `tensorboard --logdir=.aim` to visualize experiments
        - All metrics logged via `self.log()` in LightningModule are tracked
        - Hyperparameters are tracked via `self.save_hyperparameters()`
        - Future: Will migrate to Aim when Python 3.13 support is available
    """
    # Convert repo_path to Path object
    repo_path = Path(repo_path)

    # Create repository directory if it doesn't exist
    repo_path.mkdir(parents=True, exist_ok=True)

    # Create and return TensorBoardLogger
    # Note: Using version parameter to allow multiple runs with same name
    logger = TensorBoardLogger(
        save_dir=str(repo_path),
        name=experiment_name,
        version=run_name if run_name else None,
        default_hp_metric=False,  # Don't log hp_metric by default
    )

    return logger
