# Detection Module

Frozen object detection backbones for Scene Graph Generation. Extracts multi-scale features and ROI features from images.

## Available Detectors

| Detector | Params | COCO AP | Channels | ROI Pooling | RPN |
|----------|--------|---------|----------|-------------|-----|
| **ResNet-50-FPN** | ~32M | 37% | 256 | P2-P5 | Yes |
| **ResNet-101-FPN** | ~60M | 41% | 256 | P2-P5 | Yes |
| **EfficientDet-D2** | ~8M | 43% | 112 | P3-P6 | No |
| **EfficientDet-D3** | ~12M | 47% | 160 | P3-P6 | No |

**Recommendation**: Start with ResNet-101-FPN (matches reference), then evaluate EfficientDet for efficiency.

## Quick Start

### Faster R-CNN

```python
from src.modules.detection import FasterRCNNBackbone, RESNET101_FPN

detector = FasterRCNNBackbone(**RESNET101_FPN)

# Extract features
features = detector(images)  # dict: {"0": P2, "1": P3, ...}

# Get proposals (RPN)
proposals = detector.get_proposals(images)  # list[(N, 4)]

# Extract ROI features
roi_features = detector.extract_roi_features(features, proposals)  # (total, 256, 7, 7)
```

### EfficientDet

```python
from src.modules.detection import EfficientDetBackbone, EFFICIENTDET_D2

detector = EfficientDetBackbone(**EFFICIENTDET_D2)

features = detector(images)  # dict: {"0": P3, "1": P4, ...}

# No RPN - use ground truth or external proposals
boxes = [torch.tensor([[10., 10., 100., 100.]])]
roi_features = detector.extract_roi_features(features, boxes)  # (total, 112, 7, 7)
```

## Key Differences

| Aspect | Faster R-CNN | EfficientDet |
|--------|--------------|--------------|
| Feature channels | 256 (all levels) | 112 (D2) / 160 (D3) |
| Pyramid levels | P2-P6 | P3-P7 |
| Has RPN | Yes | No (`get_proposals()` raises error) |
| Input size | Any | 768×768 (D2) / 896×896 (D3) |
| Neck | FPN (unidirectional) | BiFPN (bidirectional) |

## Channel Projection

For modules expecting 256 channels with EfficientDet:

```python
class ProjectedEfficientDet(nn.Module):
    def __init__(self, variant="d2"):
        super().__init__()
        self.backbone = EfficientDetBackbone(variant=variant, pretrained=True, freeze=True)
        in_ch = 112 if variant == "d2" else 160
        self.proj = nn.Conv2d(in_ch, 256, kernel_size=1)

    def extract_roi_features(self, features, boxes):
        roi = self.backbone.extract_roi_features(features, boxes)
        return self.proj(roi)  # (N, 256, 7, 7)
```

## Freezing

The entire detector is frozen during SGG training (`freeze=True`):
- Backbone, FPN/BiFPN, RPN (if present)
- BatchNorm stays in eval mode

```python
# Verify frozen
assert all(not p.requires_grad for p in detector.parameters())
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM | Reduce batch size, use ResNet-50, or limit proposals |
| Channel mismatch | Add projection layer for EfficientDet (see above) |
| `NotImplementedError` on `get_proposals()` | EfficientDet has no RPN - use GT boxes or external proposals |
| BatchNorm updating | Call `freeze_bn(detector)` after `detector.train()` |

## Dependencies

- **Faster R-CNN**: `torchvision` (included with PyTorch)
- **EfficientDet**: `uv add effdet` (includes `timm`)
