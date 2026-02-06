# Lightning Modules

PyTorch Lightning modules for training detection and scene graph generation models.

## DetectorLightningModule

Lightning module for training object detectors (Faster R-CNN, EfficientDet) with COCO-style mAP evaluation.

### Features

- **Training loop**: Automatic loss computation and backpropagation
- **Validation loop**: Inference mode with mAP metrics (mAP@0.5, mAP@0.5:0.95, AR@100)
- **Optimizer**: AdamW with configurable learning rate and weight decay
- **Schedulers**:
  - Cosine annealing (smooth decay to 0)
  - OneCycleLR (triangular schedule with peak at 30% training)
- **Logging**: Automatic metric logging via Lightning's `self.log()`

### Usage

```python
from pytorch_lightning import Trainer
from src.modules.detection import SGGFasterRCNN
from src.trainer_lib import DetectorLightningModule

# 1. Create trainable detector
detector = SGGFasterRCNN(
    backbone="resnet50",
    pretrained=True,
    freeze=False,       # Must be unfrozen for training
    trainable=True,     # Enable training mode
    num_classes=100,
)

# 2. Wrap in Lightning module
module = DetectorLightningModule(
    model=detector,
    learning_rate=1e-4,
    weight_decay=1e-4,
    scheduler="cosine",
    warmup_epochs=1,
)

# 3. Train with Lightning Trainer
trainer = Trainer(max_epochs=10, accelerator="gpu")
trainer.fit(module, train_dataloader, val_dataloader)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | - | Detection model (SGGFasterRCNN or SGGEfficientDet) in trainable mode |
| `learning_rate` | `float` | `1e-4` | Base learning rate for AdamW optimizer |
| `weight_decay` | `float` | `1e-4` | L2 weight regularization coefficient |
| `scheduler` | `"cosine"` or `"onecycle"` | `"cosine"` | Learning rate scheduler type |
| `warmup_epochs` | `int` | `1` | Number of warmup epochs (planned, not yet implemented) |

### Batch Format

The module expects batches in the format `(images, targets)`:

```python
images: Tensor  # (B, 3, H, W) in [0, 1]
targets: list[dict]  # List of dicts with keys:
    - "boxes": Tensor (N, 4) in xyxy format
    - "labels": Tensor (N,) with class indices
```

For **EfficientDet training**, targets must include additional keys:
```python
targets: list[dict]
    - "bbox": Tensor (N, 4) in xyxy format
    - "cls": Tensor (N,) with class indices
    - "img_scale": Tensor (scalar) with image scale factor
    - "img_size": Tensor (2,) with [height, width]
```

### Logged Metrics

**Training:**
- `train/loss` - Total loss (sum of all components)
- `train/loss_classifier` - Classification loss
- `train/loss_box_reg` - Box regression loss
- `train/loss_objectness` - RPN objectness loss (Faster R-CNN)
- `train/loss_rpn_box_reg` - RPN box regression loss (Faster R-CNN)

**Validation:**
- `val/mAP@0.5` - mAP at IoU threshold 0.5
- `val/mAP@0.5:0.95` - mAP averaged over IoU 0.5 to 0.95
- `val/mAP@0.75` - mAP at IoU threshold 0.75
- `val/AR@1` - Average Recall with max 1 detection per image
- `val/AR@10` - Average Recall with max 10 detections per image
- `val/AR@100` - Average Recall with max 100 detections per image

### Schedulers

#### Cosine Annealing (`scheduler="cosine"`)

Smoothly decays learning rate to 0 over training:

```
LR(epoch) = lr * 0.5 * (1 + cos(π * epoch / max_epochs))
```

- **When to use**: Standard choice for most training runs
- **Interval**: Epoch-based (updates after each epoch)

#### OneCycleLR (`scheduler="onecycle"`)

Triangular schedule with learning rate peak at 30% of training:

```
Phase 1 (0-30%):  Warmup from lr/25 to lr
Phase 2 (30-100%): Decay from lr to lr/25
```

- **When to use**: Fast convergence, especially with large batch sizes
- **Interval**: Step-based (updates after each batch)

### Model Requirements

The model must:

1. Be in **trainable mode** (`trainable=True`, `freeze=False`)
2. Return a **loss dict** when targets are provided:
   ```python
   losses = model(images, targets)
   # Returns: {"loss_classifier": ..., "loss_box_reg": ...}
   ```
3. Return an **SGGDetectorOutput** when targets are not provided:
   ```python
   output = model(images)
   # Returns: SGGDetectorOutput(boxes=..., labels=..., scores=..., roi_features=...)
   ```

### Example with AIM Logger

```python
from src.trainer_lib import create_aim_logger, DetectorLightningModule

# Create AIM logger
logger = create_aim_logger(
    experiment_name="detector_training",
    run_name="faster_rcnn_resnet50",
)

# Create module
module = DetectorLightningModule(model=detector, learning_rate=1e-4)

# Train with logging
trainer = Trainer(
    max_epochs=10,
    logger=logger,
    accelerator="gpu",
)
trainer.fit(module, train_dataloader, val_dataloader)
```

### Implementation Details

#### Training Step

1. Ensures model is in `train()` mode
2. Calls `model(images, targets)` to get loss dict
3. Sums all loss components
4. Logs individual losses and total loss

#### Validation Step

1. Ensures model is in `eval()` mode
2. Calls `model(images)` without targets (inference mode)
3. Converts SGGDetectorOutput to list of prediction dicts
4. Accumulates predictions and targets

#### Validation Epoch End

1. Creates `DetectionEvaluator` with model's `num_classes`
2. Passes accumulated predictions/targets to evaluator
3. Computes COCO-style metrics
4. Logs metrics
5. Clears prediction/target cache

### Testing

Run unit tests:
```bash
uv run pytest tests/trainer_lib/lightning_modules/test_detector.py -v
```

Run integration tests with real models:
```bash
uv run pytest tests/trainer_lib/lightning_modules/test_detector_integration.py -v
```

### See Also

- [DetectionEvaluator](/src/evaluation/detection_metrics.py) - COCO metrics implementation
- [SGGFasterRCNN](/src/modules/detection/faster_rcnn.py) - Faster R-CNN detector
- [SGGEfficientDet](/src/modules/detection/efficientdet.py) - EfficientDet detector
- [Example script](/examples/train_detector_example.py) - Complete training example
