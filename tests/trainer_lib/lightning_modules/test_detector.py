"""Unit tests for DetectorLightningModule."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import LightningModule

from src.trainer_lib.lightning_modules.detector import DetectorLightningModule


class TestDetectorLightningModuleInit:
    """Test DetectorLightningModule initialization."""

    def test_init_with_valid_model(self):
        """Should initialize with valid model and default parameters."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91

        # Act
        module = DetectorLightningModule(
            model=mock_model,
            learning_rate=1e-4,
        )

        # Assert
        assert module.model is mock_model
        assert module.learning_rate == 1e-4
        assert module.weight_decay == 1e-4
        assert module.scheduler == "cosine"
        assert module.warmup_epochs == 1

    def test_init_custom_hyperparameters(self):
        """Should initialize with custom hyperparameters."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 100

        # Act
        module = DetectorLightningModule(
            model=mock_model,
            learning_rate=5e-4,
            weight_decay=1e-3,
            scheduler="onecycle",
            warmup_epochs=3,
        )

        # Assert
        assert module.learning_rate == 5e-4
        assert module.weight_decay == 1e-3
        assert module.scheduler == "onecycle"
        assert module.warmup_epochs == 3

    def test_is_lightning_module(self):
        """Should be an instance of LightningModule."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91

        # Act
        module = DetectorLightningModule(model=mock_model)

        # Assert
        assert isinstance(module, LightningModule)

    def test_hparams_are_saved(self):
        """Should save hyperparameters for checkpointing."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91

        # Act
        module = DetectorLightningModule(
            model=mock_model,
            learning_rate=1e-4,
            weight_decay=1e-3,
        )

        # Assert - check that hparams were saved
        assert hasattr(module, "hparams")
        assert "learning_rate" in module.hparams
        assert module.hparams["learning_rate"] == 1e-4


class TestDetectorLightningModuleTrainingStep:
    """Test training_step method."""

    def test_training_step_returns_loss(self):
        """Should return scalar loss from training step."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91
        # Mock model to return loss dict
        mock_model.return_value = {
            "loss_classifier": torch.tensor(0.5),
            "loss_box_reg": torch.tensor(0.3),
            "loss_objectness": torch.tensor(0.2),
            "loss_rpn_box_reg": torch.tensor(0.1),
        }

        module = DetectorLightningModule(model=mock_model)

        # Create batch
        images = torch.rand(2, 3, 800, 600)
        targets = [
            {"boxes": torch.rand(5, 4), "labels": torch.randint(0, 91, (5,))},
            {"boxes": torch.rand(3, 4), "labels": torch.randint(0, 91, (3,))},
        ]
        batch = (images, targets)

        # Act
        loss = module.training_step(batch, batch_idx=0)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == pytest.approx(1.1, rel=1e-5)  # Sum of all losses
        mock_model.assert_called_once_with(images, targets)

    def test_training_step_model_in_train_mode(self):
        """Should ensure model is in train mode during training."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91
        mock_model.return_value = {
            "loss_classifier": torch.tensor(0.5),
        }
        mock_model.training = False

        module = DetectorLightningModule(model=mock_model)

        # Create batch
        images = torch.rand(2, 3, 800, 600)
        targets = [
            {"boxes": torch.rand(5, 4), "labels": torch.randint(0, 91, (5,))},
        ]
        batch = (images, targets)

        # Act
        module.training_step(batch, batch_idx=0)

        # Assert - model.train() should be called
        mock_model.train.assert_called()

    def test_training_step_logs_loss_components(self):
        """Should log individual loss components."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91
        mock_model.return_value = {
            "loss_classifier": torch.tensor(0.5),
            "loss_box_reg": torch.tensor(0.3),
        }

        module = DetectorLightningModule(model=mock_model)
        module.log = MagicMock()  # Mock the log method

        # Create batch
        images = torch.rand(2, 3, 800, 600)
        targets = [{"boxes": torch.rand(5, 4), "labels": torch.randint(0, 91, (5,))}]
        batch = (images, targets)

        # Act
        module.training_step(batch, batch_idx=0)

        # Assert - check that logging was called
        assert module.log.call_count >= 2  # At least loss_classifier and loss_box_reg


class TestDetectorLightningModuleValidationStep:
    """Test validation_step method."""

    def test_validation_step_accumulates_predictions(self):
        """Should accumulate predictions and targets."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91
        mock_output = MagicMock()
        mock_output.boxes = [torch.rand(5, 4)]
        mock_output.labels = [torch.randint(0, 91, (5,))]
        mock_output.scores = [torch.rand(5)]
        # Mock __len__ to return number of images
        mock_output.__len__ = MagicMock(return_value=1)
        mock_model.predict.return_value = mock_output

        module = DetectorLightningModule(model=mock_model)

        # Create batch
        images = torch.rand(1, 3, 800, 600)
        targets = [{"boxes": torch.rand(3, 4), "labels": torch.randint(0, 91, (3,))}]
        batch = (images, targets)

        # Act
        module.validation_step(batch, batch_idx=0)

        # Assert - should have accumulated predictions
        assert len(module.val_predictions) == 1
        assert len(module.val_targets) == 1
        mock_model.predict.assert_called_once_with(images)  # Uses predict() for inference

    def test_validation_step_model_in_eval_mode(self):
        """Should ensure model is in eval mode during validation."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91
        mock_output = MagicMock()
        mock_output.boxes = [torch.rand(5, 4)]
        mock_output.labels = [torch.randint(0, 91, (5,))]
        mock_output.scores = [torch.rand(5)]
        mock_model.predict.return_value = mock_output
        mock_model.training = True

        module = DetectorLightningModule(model=mock_model)

        # Create batch
        images = torch.rand(1, 3, 800, 600)
        targets = [{"boxes": torch.rand(3, 4), "labels": torch.randint(0, 91, (3,))}]
        batch = (images, targets)

        # Act
        module.validation_step(batch, batch_idx=0)

        # Assert - model.eval() should be called
        mock_model.eval.assert_called()


class TestDetectorLightningModuleValidationEpochEnd:
    """Test on_validation_epoch_end method."""

    @patch("src.trainer_lib.lightning_modules.detector.DetectionEvaluator")
    def test_validation_epoch_end_computes_metrics(self, mock_evaluator_class):
        """Should compute mAP metrics at end of validation epoch."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91

        module = DetectorLightningModule(model=mock_model)

        # Simulate accumulated predictions
        module.val_predictions = [
            {
                "boxes": torch.rand(5, 4),
                "labels": torch.randint(0, 91, (5,)),
                "scores": torch.rand(5),
            }
        ]
        module.val_targets = [
            {"boxes": torch.rand(3, 4), "labels": torch.randint(0, 91, (3,))}
        ]

        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.compute.return_value = {
            "mAP@0.5": 0.5,
            "mAP@0.5:0.95": 0.4,
            "AR@100": 0.6,
            "mAP@0.75": 0.45,
            "AR@1": 0.3,
            "AR@10": 0.5,
        }
        mock_evaluator_class.return_value = mock_evaluator

        module.log = MagicMock()  # Mock logging

        # Act
        module.on_validation_epoch_end()

        # Assert
        mock_evaluator_class.assert_called_once_with(num_classes=91)
        mock_evaluator.update.assert_called_once()
        mock_evaluator.compute.assert_called_once()
        assert module.log.call_count >= 3  # Should log mAP@0.5, mAP@0.5:0.95, AR@100

    @patch("src.trainer_lib.lightning_modules.detector.DetectionEvaluator")
    def test_validation_epoch_end_clears_cache(self, mock_evaluator_class):
        """Should clear predictions cache after computing metrics."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91

        module = DetectorLightningModule(model=mock_model)
        module.val_predictions = [{"boxes": torch.rand(5, 4)}]
        module.val_targets = [{"boxes": torch.rand(3, 4)}]

        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.compute.return_value = {
            "mAP@0.5": 0.5,
            "mAP@0.5:0.95": 0.4,
            "AR@100": 0.6,
            "mAP@0.75": 0.45,
            "AR@1": 0.3,
            "AR@10": 0.5,
        }
        mock_evaluator_class.return_value = mock_evaluator

        module.log = MagicMock()

        # Act
        module.on_validation_epoch_end()

        # Assert
        assert len(module.val_predictions) == 0
        assert len(module.val_targets) == 0


class TestDetectorLightningModuleOptimizer:
    """Test configure_optimizers method."""

    def test_configure_optimizers_returns_dict(self):
        """Should return optimizer and scheduler config dict."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.rand(10, 10))]

        module = DetectorLightningModule(
            model=mock_model,
            learning_rate=1e-4,
        )

        # Act
        config = module.configure_optimizers()

        # Assert
        assert isinstance(config, dict)
        assert "optimizer" in config
        assert "lr_scheduler" in config

    def test_configure_optimizers_adamw(self):
        """Should use AdamW optimizer."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91
        param = torch.nn.Parameter(torch.rand(10, 10))
        mock_model.parameters.return_value = [param]

        module = DetectorLightningModule(
            model=mock_model,
            learning_rate=1e-4,
            weight_decay=1e-3,
        )

        # Act
        config = module.configure_optimizers()

        # Assert
        optimizer = config["optimizer"]
        assert optimizer.__class__.__name__ == "AdamW"
        # Check learning rate and weight decay
        assert optimizer.defaults["lr"] == 1e-4
        assert optimizer.defaults["weight_decay"] == 1e-3

    def test_configure_optimizers_cosine_scheduler(self):
        """Should use cosine annealing scheduler."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.rand(10, 10))]

        module = DetectorLightningModule(
            model=mock_model,
            scheduler="cosine",
        )
        # Mock trainer for max_epochs
        module.trainer = MagicMock()
        module.trainer.max_epochs = 10

        # Act
        config = module.configure_optimizers()

        # Assert
        lr_scheduler_config = config["lr_scheduler"]
        assert lr_scheduler_config["scheduler"].__class__.__name__ == "CosineAnnealingLR"
        assert lr_scheduler_config["interval"] == "epoch"

    def test_configure_optimizers_onecycle_scheduler(self):
        """Should use OneCycleLR scheduler."""
        # Arrange
        mock_model = MagicMock()
        mock_model.num_classes = 91
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.rand(10, 10))]

        # Create module with trainer context
        module = DetectorLightningModule(
            model=mock_model,
            scheduler="onecycle",
            learning_rate=1e-4,
        )

        # Mock trainer and datamodule for total steps calculation
        mock_trainer = MagicMock()
        mock_trainer.max_epochs = 10
        mock_trainer.estimated_stepping_batches = 1000
        module.trainer = mock_trainer

        # Act
        config = module.configure_optimizers()

        # Assert
        lr_scheduler_config = config["lr_scheduler"]
        assert lr_scheduler_config["scheduler"].__class__.__name__ == "OneCycleLR"
        assert lr_scheduler_config["interval"] == "step"
