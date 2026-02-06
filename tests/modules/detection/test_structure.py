"""Test detection module structure and imports.

This test verifies that the basic directory structure is correctly set up
and that all modules can be imported without errors.
"""



class TestDetectionModuleStructure:
    """Test suite for detection module structure."""

    def test_can_import_detection_module(self) -> None:
        """Test that detection module can be imported."""
        from src.modules import detection

        assert detection is not None
        assert hasattr(detection, "__doc__")
        assert "SGG Detection Module" in detection.__doc__

    def test_can_import_components_module(self) -> None:
        """Test that components submodule can be imported."""
        from src.modules.detection import components

        assert components is not None
        assert hasattr(components, "__doc__")
        assert "Reusable detection components" in components.__doc__

    def test_can_import_configs_module(self) -> None:
        """Test that configs submodule can be imported."""
        from src.modules.detection import configs

        assert configs is not None
        assert hasattr(configs, "__doc__")
        assert "Default detector configurations" in configs.__doc__

    def test_wildcard_import_works(self) -> None:
        """Test that 'from src.modules.detection import *' works without errors."""
        namespace: dict = {}
        exec("from src.modules.detection import *", namespace)

        # Verify expected exports are present
        expected_exports = {
            "SGGDetector",
            "SGGDetectorOutput",
            "SGGFasterRCNN",
            "SGGEfficientDet",
            "create_detector",
            "FASTERRCNN_R50",
            "FASTERRCNN_R101",
            "EFFICIENTDET_D2",
            "EFFICIENTDET_D3",
        }

        for export in expected_exports:
            assert export in namespace, f"Expected export '{export}' not found"

    def test_all_modules_have_docstrings(self) -> None:
        """Test that all __init__.py files have proper docstrings."""
        from src.modules import detection
        from src.modules.detection import components, configs

        modules = [detection, components, configs]

        for module in modules:
            assert module.__doc__ is not None
            assert len(module.__doc__.strip()) > 0


class TestDetectionExports:
    """Test suite for detection module exports."""

    def test_can_import_detector_classes(self) -> None:
        """Test that detector classes can be imported."""
        from src.modules.detection import SGGEfficientDet, SGGFasterRCNN

        assert isinstance(SGGFasterRCNN, type)
        assert isinstance(SGGEfficientDet, type)

        import torch.nn as nn

        assert issubclass(SGGFasterRCNN, nn.Module)
        assert issubclass(SGGEfficientDet, nn.Module)

    def test_can_import_base_classes(self) -> None:
        """Test that base classes can be imported."""
        from dataclasses import fields

        from src.modules.detection import SGGDetector, SGGDetectorOutput

        import torch.nn as nn

        assert issubclass(SGGDetector, nn.Module)
        # Check dataclass fields
        field_names = {f.name for f in fields(SGGDetectorOutput)}
        assert "boxes" in field_names
        assert "labels" in field_names
        assert "scores" in field_names
        assert "roi_features" in field_names

    def test_can_import_factory_function(self) -> None:
        """Test that factory function can be imported."""
        from src.modules.detection import create_detector, list_detectors

        assert callable(create_detector)
        assert callable(list_detectors)

        # Verify registered detectors
        detectors = list_detectors()
        assert "fasterrcnn" in detectors
        assert "efficientdet" in detectors

    def test_can_import_default_configs(self) -> None:
        """Test that default configurations can be imported."""
        from src.modules.detection import (
            EFFICIENTDET_D2,
            EFFICIENTDET_D3,
            FASTERRCNN_R50,
            FASTERRCNN_R101,
        )

        # Verify they are dictionaries
        assert isinstance(FASTERRCNN_R50, dict)
        assert isinstance(FASTERRCNN_R101, dict)
        assert isinstance(EFFICIENTDET_D2, dict)
        assert isinstance(EFFICIENTDET_D3, dict)

        # Verify Faster R-CNN configs have expected keys
        for config in [FASTERRCNN_R50, FASTERRCNN_R101]:
            assert "backbone" in config
            assert "freeze" in config
            assert "pretrained" in config

        # Verify EfficientDet configs have expected keys
        for config in [EFFICIENTDET_D2, EFFICIENTDET_D3]:
            assert "variant" in config
            assert "freeze" in config
            assert "pretrained" in config

    def test_can_import_components(self) -> None:
        """Test that component utilities can be imported."""
        from src.modules.detection.components import (
            ROIPooler,
            freeze_backbone_stages,
            freeze_bn,
            freeze_module,
        )

        import torch.nn as nn

        assert issubclass(ROIPooler, nn.Module)
        assert callable(freeze_module)
        assert callable(freeze_bn)
        assert callable(freeze_backbone_stages)

    def test_all_exports_listed_in_all(self) -> None:
        """Test that __all__ is defined and contains all public exports."""
        from src.modules import detection
        from src.modules.detection import components

        # Check detection module __all__
        assert hasattr(detection, "__all__")
        detection_all = detection.__all__

        expected_detection = [
            "SGGDetector",
            "SGGDetectorOutput",
            "SGGFasterRCNN",
            "SGGEfficientDet",
            "create_detector",
            "FASTERRCNN_R50",
            "FASTERRCNN_R101",
            "EFFICIENTDET_D2",
            "EFFICIENTDET_D3",
        ]

        for export in expected_detection:
            assert export in detection_all, f"'{export}' missing from detection.__all__"

        # Check components module __all__
        assert hasattr(components, "__all__")
        components_all = components.__all__

        expected_components = [
            "ROIPooler",
            "freeze_module",
            "freeze_bn",
            "freeze_backbone_stages",
        ]

        for export in expected_components:
            assert export in components_all, f"'{export}' missing from components.__all__"

    def test_imports_are_accessible_from_module(self) -> None:
        """Test that all listed exports are actually accessible from the module."""
        from src.modules import detection
        from src.modules.detection import components

        # Verify all detection exports are accessible
        for export_name in detection.__all__:
            assert hasattr(
                detection, export_name
            ), f"Export '{export_name}' listed in __all__ but not accessible"

        # Verify all components exports are accessible
        for export_name in components.__all__:
            assert hasattr(
                components, export_name
            ), f"Export '{export_name}' listed in __all__ but not accessible"
