"""
Freeze utilities for detection backbones.

Provides functions to freeze model components during SGG training, where the
detection backbone is typically frozen and only the relation head is trainable.

Key functions:
- freeze_module: Recursively freeze all parameters in a module
- freeze_bn: Freeze BatchNorm layers (set to eval mode + disable gradients)
- freeze_backbone_stages: Freeze first N stages of a ResNet backbone

Usage:
    >>> from torchvision.models import resnet50
    >>> backbone = resnet50(weights='DEFAULT')
    >>> freeze_backbone_stages(backbone, stages=5)  # Freeze entire backbone
    >>> freeze_bn(backbone)  # Keep BN in eval mode
"""

from typing import Any

import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    """
    Recursively freeze all parameters in a module.

    Sets requires_grad=False for all parameters in the module and its children.
    This prevents gradient computation during backpropagation.

    Args:
        module: PyTorch module to freeze

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        >>> freeze_module(model)
        >>> all(not p.requires_grad for p in model.parameters())
        True
    """
    for param in module.parameters():
        param.requires_grad = False


def freeze_bn(module: nn.Module) -> None:
    """
    Freeze all BatchNorm layers in a module.

    Sets BatchNorm layers to eval mode and disables gradients for their parameters.
    This prevents updating running statistics during training.

    Note:
        This sets BN layers to eval mode at call time, but calling module.train()
        later will reset them to train mode. For persistent eval mode, consider
        overriding the train() method or using hooks.

    Args:
        module: PyTorch module (may contain BatchNorm layers)

    Example:
        >>> model = nn.Sequential(nn.Conv2d(3, 64, 3), nn.BatchNorm2d(64))
        >>> freeze_bn(model)
        >>> bn = list(model.modules())[2]
        >>> bn.training
        False
    """
    for child_module in module.modules():
        if isinstance(child_module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Set to eval mode (disables running stats update)
            child_module.eval()
            # Disable gradients
            freeze_module(child_module)


def freeze_backbone_stages(backbone: nn.Module, stages: int) -> None:
    """
    Freeze first N stages of a ResNet backbone.

    ResNet stages:
    - Stage 0: stem (conv1, bn1, maxpool)
    - Stage 1: layer1
    - Stage 2: layer2
    - Stage 3: layer3
    - Stage 4: layer4

    When stages=2, freezes stem + layer1 (FREEZE_CONV_BODY_AT=2 in reference impl).
    This is the minimum freezing level used in the original paper.

    When stages=5, freezes all feature extraction layers including avgpool and fc
    (entire backbone for SGG training).

    Args:
        backbone: ResNet backbone (must have conv1, bn1, layer1-4 attributes)
        stages: Number of stages to freeze (0-5)

    Raises:
        ValueError: If stages is not in range [0, 5]
        AttributeError: If backbone doesn't have expected ResNet structure

    Example:
        >>> from torchvision.models import resnet50
        >>> backbone = resnet50(weights='DEFAULT')
        >>> freeze_backbone_stages(backbone, stages=2)  # Freeze stem + layer1
        >>> all(not p.requires_grad for p in backbone.layer1.parameters())
        True
        >>> any(p.requires_grad for p in backbone.layer2.parameters())
        True
    """
    if not 0 <= stages <= 5:
        raise ValueError(f"stages must be between 0 and 5, got {stages}")

    # Define stage components (in order)
    stage_modules: list[tuple[int, list[Any]]] = [
        (1, ["conv1", "bn1"]),  # Stage 0: stem
        (2, ["layer1"]),  # Stage 1
        (3, ["layer2"]),  # Stage 2
        (4, ["layer3"]),  # Stage 3
        (5, ["layer4", "avgpool", "fc"]),  # Stage 4 + classifier head
    ]

    # Freeze stages
    for stage_idx, module_names in stage_modules:
        if stage_idx <= stages:
            for name in module_names:
                if not hasattr(backbone, name):
                    raise AttributeError(
                        f"Backbone missing expected attribute '{name}'. "
                        f"Is this a ResNet-style backbone?"
                    )
                module = getattr(backbone, name)
                freeze_module(module)
                # Also freeze BatchNorm in this stage
                freeze_bn(module)
