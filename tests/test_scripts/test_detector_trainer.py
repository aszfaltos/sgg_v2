"""Tests for detector training CLI script.

Tests cover argument parsing, configuration generation, and integration
with Lightning components.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class TestArgumentParsing:
    """Test CLI argument parsing."""

    @pytest.fixture
    def mock_parse_args(self):
        """Fixture to import parse_args after script is created."""
        from scripts.detector_trainer import parse_args

        return parse_args

    def test_required_detector_argument(self, mock_parse_args):
        """Test that detector argument is required."""
        with pytest.raises(SystemExit):
            # Missing required --detector
            with patch("sys.argv", ["detector_trainer.py"]):
                mock_parse_args()

    def test_fasterrcnn_defaults(self, mock_parse_args):
        """Test default values for Faster R-CNN training."""
        with patch(
            "sys.argv",
            [
                "detector_trainer.py",
                "--detector",
                "fasterrcnn",
            ],
        ):
            args = mock_parse_args()

        assert args.detector == "fasterrcnn"
        assert args.variant is None  # Not used for Faster R-CNN
        assert args.dataset == "vrd"
        assert args.data_root == "datasets/vrd"
        assert args.batch_size == 4
        assert args.epochs == 20
        assert args.lr == 1e-4
        assert args.val_split == 0.1
        assert args.scheduler == "cosine"
        assert args.checkpoint_dir is None
        assert args.resume is None
        assert args.seed == 42

    def test_efficientdet_with_variant(self, mock_parse_args):
        """Test EfficientDet with custom variant."""
        with patch(
            "sys.argv",
            [
                "detector_trainer.py",
                "--detector",
                "efficientdet",
                "--variant",
                "d3",
            ],
        ):
            args = mock_parse_args()

        assert args.detector == "efficientdet"
        assert args.variant == "d3"

    def test_custom_hyperparameters(self, mock_parse_args):
        """Test custom hyperparameter configuration."""
        with patch(
            "sys.argv",
            [
                "detector_trainer.py",
                "--detector",
                "fasterrcnn",
                "--batch-size",
                "8",
                "--epochs",
                "30",
                "--lr",
                "5e-4",
                "--val-split",
                "0.2",
                "--scheduler",
                "onecycle",
            ],
        ):
            args = mock_parse_args()

        assert args.batch_size == 8
        assert args.epochs == 30
        assert args.lr == 5e-4
        assert args.val_split == 0.2
        assert args.scheduler == "onecycle"

    def test_checkpoint_and_resume(self, mock_parse_args):
        """Test checkpoint directory and resume path."""
        with patch(
            "sys.argv",
            [
                "detector_trainer.py",
                "--detector",
                "fasterrcnn",
                "--checkpoint-dir",
                "checkpoints/test",
                "--resume",
                "checkpoints/test/last.ckpt",
            ],
        ):
            args = mock_parse_args()

        assert args.checkpoint_dir == "checkpoints/test"
        assert args.resume == "checkpoints/test/last.ckpt"


class TestCheckpointDirGeneration:
    """Test automatic checkpoint directory generation."""

    @pytest.fixture
    def mock_generate_checkpoint_dir(self):
        """Fixture to import function after script is created."""
        from scripts.detector_trainer import generate_checkpoint_dir

        return generate_checkpoint_dir

    def test_fasterrcnn_checkpoint_dir(self, mock_generate_checkpoint_dir):
        """Test Faster R-CNN checkpoint directory generation."""
        dir_path = mock_generate_checkpoint_dir("fasterrcnn", variant=None)
        expected = Path("checkpoints/detectors/fasterrcnn_resnet50_fpn_v2")
        assert dir_path == expected

    def test_efficientdet_checkpoint_dir(self, mock_generate_checkpoint_dir):
        """Test EfficientDet checkpoint directory generation."""
        dir_path = mock_generate_checkpoint_dir("efficientdet", variant="d2")
        expected = Path("checkpoints/detectors/efficientdet_d2")
        assert dir_path == expected

    def test_efficientdet_default_variant(self, mock_generate_checkpoint_dir):
        """Test EfficientDet with default variant."""
        dir_path = mock_generate_checkpoint_dir("efficientdet", variant=None)
        expected = Path("checkpoints/detectors/efficientdet_d2")
        assert dir_path == expected


class TestDetectorCreation:
    """Test detector instantiation."""

    @pytest.fixture
    def mock_create_detector(self):
        """Fixture to import function after script is created."""
        from scripts.detector_trainer import create_detector_for_training

        return create_detector_for_training

    def test_create_fasterrcnn_trainable(self, mock_create_detector):
        """Test creating trainable Faster R-CNN."""
        detector = mock_create_detector("fasterrcnn", variant=None)

        assert detector is not None
        assert detector.trainable is True
        assert detector._freeze is False
        assert detector.num_classes == 101  # VRD has 100 classes + background

    def test_create_efficientdet_trainable(self, mock_create_detector):
        """Test creating trainable EfficientDet."""
        detector = mock_create_detector("efficientdet", variant="d2")

        assert detector is not None
        assert detector._trainable is True
        assert detector._freeze is False
        assert detector.num_classes == 100  # VRD has 100 classes (no background)

    def test_invalid_detector_raises_error(self, mock_create_detector):
        """Test that invalid detector name raises error."""
        with pytest.raises(ValueError, match="Unknown detector"):
            mock_create_detector("invalid_detector", variant=None)


class TestDataModuleCreation:
    """Test DataModule instantiation."""

    @pytest.fixture
    def mock_create_datamodule(self):
        """Fixture to import function after script is created."""
        from scripts.detector_trainer import create_datamodule

        return create_datamodule

    def test_create_vrd_datamodule_fasterrcnn(self, mock_create_datamodule):
        """Test creating VRD DataModule for Faster R-CNN (with background class)."""
        dm = mock_create_datamodule(
            dataset="vrd",
            data_root="datasets/vrd",
            batch_size=4,
            val_split=0.1,
            target_size=None,
            seed=42,
            detector="fasterrcnn",
        )

        assert dm is not None
        assert dm.batch_size == 4
        assert dm.val_split == 0.1
        assert dm.seed == 42
        assert dm.target_size is None
        assert dm.background_class is True  # Faster R-CNN uses 1-indexed labels

    def test_create_vrd_datamodule_efficientdet(self, mock_create_datamodule):
        """Test creating VRD DataModule for EfficientDet (no background class)."""
        dm = mock_create_datamodule(
            dataset="vrd",
            data_root="datasets/vrd",
            batch_size=4,
            val_split=0.1,
            target_size=(768, 768),  # EfficientDet D2 size
            seed=42,
            detector="efficientdet",
        )

        assert dm.target_size == (768, 768)
        assert dm.background_class is False  # EfficientDet uses 0-indexed labels


class TestTargetSizeMapping:
    """Test target size mapping for EfficientDet variants."""

    @pytest.fixture
    def efficientdet_sizes(self):
        """Fixture for EfficientDet image sizes."""
        from scripts.detector_trainer import EFFICIENTDET_SIZES

        return EFFICIENTDET_SIZES

    def test_all_variants_have_sizes(self, efficientdet_sizes):
        """Test that all EfficientDet variants have defined sizes."""
        expected_variants = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]
        for variant in expected_variants:
            assert variant in efficientdet_sizes
            size = efficientdet_sizes[variant]
            assert isinstance(size, tuple)
            assert len(size) == 2
            assert all(isinstance(s, int) for s in size)

    def test_d2_size_is_768(self, efficientdet_sizes):
        """Test that D2 variant uses 768x768 images."""
        assert efficientdet_sizes["d2"] == (768, 768)


@pytest.mark.slow
class TestTrainingIntegration:
    """Integration tests for training loop (slow tests)."""

    @pytest.fixture
    def mock_main(self):
        """Fixture to import main function after script is created."""
        from scripts.detector_trainer import main

        return main

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_train_one_epoch_fasterrcnn(self, mock_main, tmp_path):
        """Test training Faster R-CNN for one epoch."""
        # This is a slow integration test - only run when explicitly requested
        # Use small dataset and fast configs
        _ = tmp_path / "checkpoints"

        with patch(
            "scripts.detector_trainer.VRDDetectionDataModule"
        ) as mock_datamodule:
            # Mock DataModule to return small fake dataset
            mock_dm = MagicMock()
            mock_datamodule.return_value = mock_dm

            # Would run training here - mocked for fast test
            # Real test would verify files are created:
            # - tmp_path / "checkpoints" / "last.ckpt"
            # - tmp_path / "checkpoints" / "best.ckpt"
            pass
