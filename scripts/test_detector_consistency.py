#!/usr/bin/env python3
"""Test detector consistency between training and inference paths.

Hooks into the Faster R-CNN training forward to extract the proposals and
classifications from the ROI head, decodes them into detections via
postprocess_detections, and compares with predict() output.

For EfficientDet, runs bench_predict in train mode (BN uses batch stats)
vs eval mode (BN uses running stats) since both benches share the same model.

Usage:
    uv run python scripts/test_detector_consistency.py --detector fasterrcnn
    uv run python scripts/test_detector_consistency.py --detector efficientdet --variant d0
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.vrd_detection import VRDDetectionDataset  # noqa: E402
from src.modules.detection import SGGEfficientDet, SGGFasterRCNN  # noqa: E402
from src.modules.detection.components.freeze import freeze_bn  # noqa: E402


def create_detector(
    detector: str, variant: str | None
) -> SGGFasterRCNN | SGGEfficientDet:
    if detector == "fasterrcnn":
        return SGGFasterRCNN(
            backbone="resnet50",
            num_classes=101,
            trainable=True,
            freeze=False,
            pretrained=True,
        )
    elif detector == "efficientdet":
        variant_str = variant or "d2"
        return SGGEfficientDet(
            variant=variant_str,  # type: ignore[arg-type]
            num_classes=100,
            trainable=True,
            freeze=False,
            pretrained=True,
        )
    else:
        raise ValueError(f"Unknown detector: {detector}")


def get_label_name(
    label_idx: int, class_names: list[str] | None, label_offset: int
) -> str:
    if class_names and 0 <= label_idx - label_offset < len(class_names):
        return class_names[label_idx - label_offset]
    return f"class_{label_idx}"


def sort_by_score(
    boxes: Tensor, labels: Tensor, scores: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    order = scores.argsort(descending=True)
    return boxes[order], labels[order], scores[order]


def print_detections(
    title: str,
    boxes: Tensor,
    labels: Tensor,
    scores: Tensor,
    class_names: list[str] | None,
    label_offset: int,
    max_show: int = 20,
) -> None:
    print(f"  {title}: {len(labels)} detections")
    for i in range(min(len(labels), max_show)):
        name = get_label_name(int(labels[i]), class_names, label_offset)
        b = boxes[i].tolist()
        print(
            f"    [{i:2d}] {name:<20s} score={scores[i]:.4f}  "
            f"box=[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]"
        )
    if len(labels) > max_show:
        print(f"    ... and {len(labels) - max_show} more")


def compare_outputs(
    boxes_a: Tensor,
    labels_a: Tensor,
    scores_a: Tensor,
    boxes_b: Tensor,
    labels_b: Tensor,
    scores_b: Tensor,
    class_names: list[str] | None,
    label_offset: int,
    max_show: int = 20,
) -> None:
    """Side-by-side comparison of train vs predict detections."""
    n_a, n_b = len(labels_a), len(labels_b)
    print(f"  Train path: {n_a} detections, Predict path: {n_b} detections")

    if n_a != n_b:
        print(f"  DIFFERENT count: {n_a} vs {n_b}")

    n = max(n_a, n_b)
    mismatched = 0
    for i in range(min(n, max_show)):
        if i < n_a and i < n_b:
            name_a = get_label_name(int(labels_a[i]), class_names, label_offset)
            name_b = get_label_name(int(labels_b[i]), class_names, label_offset)
            ba = boxes_a[i].tolist()
            bb = boxes_b[i].tolist()
            sa, sb = scores_a[i].item(), scores_b[i].item()
            label_same = int(labels_a[i]) == int(labels_b[i])
            box_diff = (boxes_a[i] - boxes_b[i]).abs().max().item()
            score_diff = abs(sa - sb)
            flag = "" if (label_same and box_diff < 0.1 and score_diff < 1e-4) else " <<<"
            if flag:
                mismatched += 1
            print(
                f"    [{i:2d}] train: {name_a:<15s} {sa:.4f} "
                f"[{ba[0]:6.1f},{ba[1]:6.1f},{ba[2]:6.1f},{ba[3]:6.1f}]  |  "
                f"pred: {name_b:<15s} {sb:.4f} "
                f"[{bb[0]:6.1f},{bb[1]:6.1f},{bb[2]:6.1f},{bb[3]:6.1f}]{flag}"
            )
        elif i < n_a:
            name_a = get_label_name(int(labels_a[i]), class_names, label_offset)
            ba = boxes_a[i].tolist()
            print(
                f"    [{i:2d}] train: {name_a:<15s} {scores_a[i]:.4f} "
                f"[{ba[0]:6.1f},{ba[1]:6.1f},{ba[2]:6.1f},{ba[3]:6.1f}]  |  "
                f"pred: ---"
            )
            mismatched += 1
        else:
            name_b = get_label_name(int(labels_b[i]), class_names, label_offset)
            bb = boxes_b[i].tolist()
            print(
                f"    [{i:2d}] train: ---"
                f"  |  pred: {name_b:<15s} {scores_b[i]:.4f} "
                f"[{bb[0]:6.1f},{bb[1]:6.1f},{bb[2]:6.1f},{bb[3]:6.1f}]"
            )
            mismatched += 1

    if n > max_show:
        print(f"    ... and {n - max_show} more")

    # Summary stats
    if n_a == n_b and n_a > 0:
        labels_match = torch.equal(labels_a, labels_b)
        box_max_diff = (boxes_a - boxes_b).abs().max().item()
        score_max_diff = (scores_a - scores_b).abs().max().item()
        print(f"\n  Summary:")
        print(f"    Labels match: {labels_match}")
        print(f"    Box max diff:   {box_max_diff:.6f}")
        print(f"    Score max diff: {score_max_diff:.6f}")
        if labels_match and box_max_diff < 1e-4 and score_max_diff < 1e-4:
            print("    VERDICT: IDENTICAL")
        else:
            print(f"    VERDICT: DIFFERENT ({mismatched} rows flagged)")


# --- Faster R-CNN: extract proposals from training path ---


def extract_fasterrcnn_train_detections(
    model: SGGFasterRCNN, image: Tensor, target: dict[str, Tensor]
) -> tuple[Tensor, Tensor, Tensor]:
    """Hook into Faster R-CNN training forward to extract decoded proposals.

    During training, roi_heads.select_training_samples subsamples ~512 proposals
    from ~1000 RPN proposals and matches them to GT. The box_predictor then
    classifies these. We capture the logits and proposals, then decode them
    into detections via postprocess_detections (same NMS as predict()).
    """
    captures: dict[str, list | Tensor] = {}

    def roi_pool_hook(module, input, output):  # type: ignore[no-untyped-def]
        # input = (features, proposals, image_shapes)
        # proposals here are AFTER select_training_samples in train mode
        captures["proposals"] = [p.detach().clone() for p in input[1]]
        captures["image_shapes"] = list(input[2])

    def predictor_hook(module, input, output):  # type: ignore[no-untyped-def]
        captures["class_logits"] = output[0].detach().clone()
        captures["box_regression"] = output[1].detach().clone()

    h1 = model.model.roi_heads.box_roi_pool.register_forward_hook(roi_pool_hook)
    h2 = model.model.roi_heads.box_predictor.register_forward_hook(predictor_hook)

    # Training forward — must be in train mode for select_training_samples
    # (which only runs in training mode), but freeze BN for stable stats
    # and DropPath is not used in ResNet so freeze_bn is sufficient here.
    model.model.train()
    freeze_bn(model.model)
    image_list = [image[i] for i in range(image.shape[0])]
    with torch.no_grad():
        loss_dict = model.model(image_list, [target])

    h1.remove()
    h2.remove()

    print(f"  Training losses:")
    for k, v in loss_dict.items():
        print(f"    {k}: {v.item():.4f}")
    print(f"  Sampled proposals: {captures['proposals'][0].shape[0]}")

    # Decode proposals into detections using the same NMS pipeline as predict()
    was_training = model.model.roi_heads.training
    model.model.roi_heads.training = False
    boxes_list, scores_list, labels_list = model.model.roi_heads.postprocess_detections(
        captures["class_logits"],
        captures["box_regression"],
        captures["proposals"],
        captures["image_shapes"],
    )
    model.model.roi_heads.training = was_training

    # Map boxes back to original image coordinates
    original_sizes = [(image.shape[2], image.shape[3])]
    result = [{"boxes": boxes_list[0], "scores": scores_list[0], "labels": labels_list[0]}]
    result = model.model.transform.postprocess(result, captures["image_shapes"], original_sizes)

    return result[0]["boxes"].cpu(), result[0]["labels"].cpu(), result[0]["scores"].cpu()


# --- EfficientDet: extract predictions from bench_train path ---


def extract_efficientdet_train_detections(
    model: SGGEfficientDet, image: Tensor, target: dict[str, Tensor]
) -> tuple[Tensor, Tensor, Tensor]:
    """Hook the shared model during bench_train forward, decode via bench_predict NMS.

    bench_train and bench_predict share the same underlying EfficientDet model.
    We hook model.forward() during bench_train to capture the raw (class_out, box_out),
    then monkey-patch model.forward to return those cached outputs so bench_predict's
    NMS runs on exactly what bench_train saw.
    """
    # Resize image to model's expected size (same as SGGEfficientDet.forward does)
    img = image
    if img.shape[2] != model._image_size or img.shape[3] != model._image_size:
        img = torch.nn.functional.interpolate(
            img,
            size=(model._image_size, model._image_size),
            mode="bilinear",
            align_corners=False,
        )

    # Hook to capture raw (class_out, box_out) during bench_train forward
    # These are lists of tensors (one per FPN level)
    captured: dict[str, tuple[list[Tensor], list[Tensor]]] = {}

    def model_hook(
        module: torch.nn.Module,
        input: tuple[Tensor, ...],
        output: tuple[list[Tensor], list[Tensor]],
    ) -> None:
        captured["output"] = (
            [t.detach().clone() for t in output[0]],
            [t.detach().clone() for t in output[1]],
        )

    h = model.model.register_forward_hook(model_hook)

    # Training forward via bench_train
    # Use eval mode for the underlying model so BN uses running stats AND
    # DropPath (stochastic depth) in the EfficientNet backbone is disabled.
    # freeze_bn() alone is not enough — DropPath with drop_path_rate=0.2
    # randomly drops paths in train mode, producing different features.
    model.model.eval()
    with torch.no_grad():
        loss_dict = model(image, [target])

    h.remove()

    print(f"  Training losses:")
    for k, v in loss_dict.items():
        if v.dim() == 0:
            print(f"    {k}: {v.item():.4f}")
        else:
            print(f"    {k}: tensor with {v.numel()} elements (mean={v.mean().item():.4f})")

    # Decode captured outputs via bench_predict's NMS pipeline.
    # Monkey-patch model.forward so bench_predict uses the exact outputs
    # that bench_train just computed (same BN state, same everything).
    original_forward = model.model.forward

    def cached_forward(*args: object, **kwargs: object) -> tuple[list[Tensor], list[Tensor]]:
        return captured["output"]

    model.model.forward = cached_forward  # type: ignore[assignment]

    with torch.no_grad():
        model._captured_fpn_features = []
        detections = model.bench_predict(img)

    model.model.forward = original_forward  # type: ignore[assignment]

    boxes, labels, scores = model._parse_detections(detections, image.shape[0], image.device)
    return boxes[0].cpu(), labels[0].cpu(), scores[0].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test detector consistency: training path vs predict()"
    )
    parser.add_argument(
        "--detector", type=str, default="fasterrcnn",
        choices=["fasterrcnn", "efficientdet"],
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        choices=["d0", "d1", "d2", "d3"],
    )
    parser.add_argument("--data-root", type=str, default="datasets/vrd")
    parser.add_argument("--image-idx", type=int, default=0)
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load class names
    objects_file = Path(args.data_root) / "objects.json"
    class_names = None
    if objects_file.exists():
        with open(objects_file) as f:
            class_names = json.load(f)

    is_fasterrcnn = args.detector == "fasterrcnn"
    # Both detectors output 1-indexed labels (background_class=True)
    label_offset = 1

    # Load dataset
    print(f"Loading VRD dataset from {args.data_root}...")
    dataset = VRDDetectionDataset(
        root=args.data_root, split="train", background_class=True,
    )
    print(f"  {len(dataset)} images, {dataset.num_classes} classes")

    # Get a single image
    image, target = dataset[args.image_idx]
    image = image.unsqueeze(0).to(device)
    target = {k: v.to(device) for k, v in target.items()}
    print(f"  Image {args.image_idx}: shape={image.shape}, "
          f"{target['boxes'].shape[0]} GT objects")

    # Print ground truth
    gt_labels = target["labels"].cpu()
    gt_boxes = target["boxes"].cpu()
    print(f"\nGround truth ({len(gt_labels)} objects):")
    for i in range(len(gt_labels)):
        name = get_label_name(int(gt_labels[i]), class_names, 1)
        b = gt_boxes[i].tolist()
        print(f"  [{i:2d}] {name:<20s} "
              f"box=[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")

    # Create detector
    print(f"\nCreating {args.detector} detector...")
    model = create_detector(args.detector, args.variant)
    model = model.to(device)
    print(f"  Created ({model.num_classes} classes)")

    # --- Predict path first (clean BN running stats, no contamination) ---
    print("\n" + "=" * 60)
    print("PREDICT PATH (eval → predict())")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        eval_output = model.predict(image)

    pred_boxes = eval_output.boxes[0].detach().cpu()
    pred_labels = eval_output.labels[0].detach().cpu()
    pred_scores = eval_output.scores[0].detach().cpu()
    pred_boxes, pred_labels, pred_scores = sort_by_score(
        pred_boxes, pred_labels, pred_scores
    )
    print_detections(
        "predict() output",
        pred_boxes, pred_labels, pred_scores, class_names, label_offset,
    )

    # --- Train path: extract proposals from training forward ---
    print("\n" + "=" * 60)
    print("TRAIN PATH (forward → extract proposals → decode)")
    print("=" * 60)

    if is_fasterrcnn:
        train_boxes, train_labels, train_scores = extract_fasterrcnn_train_detections(
            model, image, target  # type: ignore[arg-type]
        )
    else:
        train_boxes, train_labels, train_scores = extract_efficientdet_train_detections(
            model, image, target  # type: ignore[arg-type]
        )

    train_boxes, train_labels, train_scores = sort_by_score(
        train_boxes, train_labels, train_scores
    )
    print_detections(
        "Decoded from training forward",
        train_boxes, train_labels, train_scores, class_names, label_offset,
    )

    # --- Side-by-side comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON (train path vs predict path, <<< = different)")
    print("=" * 60)
    compare_outputs(
        train_boxes, train_labels, train_scores,
        pred_boxes, pred_labels, pred_scores,
        class_names, label_offset,
    )


if __name__ == "__main__":
    main()
