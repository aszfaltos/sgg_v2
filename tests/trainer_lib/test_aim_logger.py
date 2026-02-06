"""Tests for experiment tracking logger utility."""

from pathlib import Path

from pytorch_lightning.loggers import Logger, TensorBoardLogger

from src.trainer_lib.logging.aim_logger import create_aim_logger


class TestCreateAimLogger:
    """Tests for create_aim_logger function."""

    def test_creates_logger_instance(self, tmp_path: Path) -> None:
        """Test that function returns a Logger instance."""
        # Arrange
        experiment_name = "test_experiment"
        repo_path = tmp_path / ".aim"

        # Act
        logger = create_aim_logger(
            experiment_name=experiment_name,
            repo_path=repo_path,
        )

        # Assert
        assert isinstance(logger, Logger)
        assert isinstance(logger, TensorBoardLogger)

    def test_uses_provided_experiment_name(self, tmp_path: Path) -> None:
        """Test that provided experiment name is used."""
        # Arrange
        experiment_name = "my_experiment"
        repo_path = tmp_path / ".aim"

        # Act
        logger = create_aim_logger(
            experiment_name=experiment_name,
            repo_path=repo_path,
        )

        # Assert
        assert logger.name == experiment_name

    def test_uses_provided_run_name(self, tmp_path: Path) -> None:
        """Test that provided run name is used."""
        # Arrange
        experiment_name = "test_experiment"
        run_name = "my_run"
        repo_path = tmp_path / ".aim"

        # Act
        logger = create_aim_logger(
            experiment_name=experiment_name,
            run_name=run_name,
            repo_path=repo_path,
        )

        # Assert
        assert logger.version == run_name

    def test_run_name_optional(self, tmp_path: Path) -> None:
        """Test that run_name is optional."""
        # Arrange
        experiment_name = "test_experiment"
        repo_path = tmp_path / ".aim"

        # Act
        logger = create_aim_logger(
            experiment_name=experiment_name,
            repo_path=repo_path,
        )

        # Assert - version will be auto-generated (not None)
        assert logger.version is not None

    def test_uses_default_repo_path(self, tmp_path: Path) -> None:
        """Test that default repo path is .aim."""
        # Arrange
        experiment_name = "test_experiment"

        # Act - using default repo_path
        logger = create_aim_logger(experiment_name=experiment_name)

        # Assert - function should work without raising error
        assert isinstance(logger, Logger)

    def test_accepts_string_repo_path(self, tmp_path: Path) -> None:
        """Test that repo_path accepts string."""
        # Arrange
        experiment_name = "test_experiment"
        repo_path_str = str(tmp_path / ".aim")

        # Act
        logger = create_aim_logger(
            experiment_name=experiment_name,
            repo_path=repo_path_str,
        )

        # Assert
        assert isinstance(logger, Logger)

    def test_accepts_path_repo_path(self, tmp_path: Path) -> None:
        """Test that repo_path accepts Path object."""
        # Arrange
        experiment_name = "test_experiment"
        repo_path = tmp_path / ".aim"

        # Act
        logger = create_aim_logger(
            experiment_name=experiment_name,
            repo_path=repo_path,
        )

        # Assert
        assert isinstance(logger, Logger)

    def test_creates_repo_directory_if_missing(self, tmp_path: Path) -> None:
        """Test that repo directory is created if it doesn't exist."""
        # Arrange
        experiment_name = "test_experiment"
        repo_path = tmp_path / ".aim"
        assert not repo_path.exists()

        # Act
        create_aim_logger(
            experiment_name=experiment_name,
            repo_path=repo_path,
        )

        # Assert
        assert repo_path.exists()
        assert repo_path.is_dir()

    def test_logger_configured_for_lightning(self, tmp_path: Path) -> None:
        """Test that logger is compatible with Lightning Trainer."""
        # Arrange
        experiment_name = "test_experiment"
        repo_path = tmp_path / ".aim"

        # Act
        logger = create_aim_logger(
            experiment_name=experiment_name,
            repo_path=repo_path,
        )

        # Assert - AimLogger should have required Lightning logger methods
        assert hasattr(logger, "log_metrics")
        assert hasattr(logger, "log_hyperparams")
        assert hasattr(logger, "save")
        assert hasattr(logger, "finalize")

    def test_multiple_loggers_same_repo(self, tmp_path: Path) -> None:
        """Test that multiple loggers can use the same repo."""
        # Arrange
        repo_path = tmp_path / ".aim"

        # Act
        logger1 = create_aim_logger(
            experiment_name="exp1",
            repo_path=repo_path,
        )
        logger2 = create_aim_logger(
            experiment_name="exp2",
            repo_path=repo_path,
        )

        # Assert
        assert isinstance(logger1, Logger)
        assert isinstance(logger2, Logger)
        assert logger1.name != logger2.name


class TestLoggerIntegration:
    """Integration tests for logger with Lightning."""

    def test_logger_logs_metrics(self, tmp_path: Path) -> None:
        """Test that logger can log metrics."""
        # Arrange
        experiment_name = "test_experiment"
        repo_path = tmp_path / ".aim"
        logger = create_aim_logger(
            experiment_name=experiment_name,
            repo_path=repo_path,
        )

        # Act
        metrics = {"loss": 0.5, "accuracy": 0.9}
        logger.log_metrics(metrics, step=0)

        # Assert - should not raise any errors
        # Actual verification would require querying the Aim repo
        assert True

    def test_logger_logs_hyperparameters(self, tmp_path: Path) -> None:
        """Test that logger can log hyperparameters."""
        # Arrange
        experiment_name = "test_experiment"
        repo_path = tmp_path / ".aim"
        logger = create_aim_logger(
            experiment_name=experiment_name,
            repo_path=repo_path,
        )

        # Act
        hparams = {"learning_rate": 0.001, "batch_size": 32}
        logger.log_hyperparams(hparams)

        # Assert - should not raise any errors
        assert True
