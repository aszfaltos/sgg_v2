#!/usr/bin/env python3
"""Precompute detector features for SGG training.

Extracts ROI features from a finetuned detector and saves them to HDF5 files.
Supports both ground truth boxes (for oracle/upper-bound experiments) and
predicted boxes (for standard SGG training).

Output is compatible with the batching design in batching.md: the SGG dataset
builds graph topology (edges, geometric features) at __getitem__ time from
the stored boxes.

File layout:
    datasets/vrd/features/{detector_name}/
        gt_train.h5     — GT box features for train split
        gt_test.h5      — GT box features for test split
        pred_train.h5   — Predicted box features for train split
        pred_test.h5    — Predicted box features for test split

HDF5 structure (per image group):
    /{image_id}/roi_features  (N, C, H, W) float32, lzf compressed
    /{image_id}/boxes         (N, 4) float32, xyxy
    /{image_id}/labels        (N,) int64, 1-indexed
    /{image_id}/scores        (N,) float32          [pred only]
    /{image_id}/relations     (R, 3) int64           [gt: all rels; pred: IoU-matched rels]

Usage:
    # Both GT and predicted features, train split, Faster R-CNN
    uv run python scripts/precompute_sgg_features.py \\
        --detector fasterrcnn --backbone resnet50 \\
        --checkpoint checkpoints/detectors/fasterrcnn_resnet50_fpn_v2/best.ckpt \\
        --split train --source both --batch-size 8

    # Predicted only, test split, EfficientDet
    uv run python scripts/precompute_sgg_features.py \\
        --detector efficientdet --variant d2 \\
        --checkpoint checkpoints/detectors/efficientdet_d2/best.ckpt \\
        --split test --source pred
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import h5py
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.vrd_detection import VRDDetectionDataset  # noqa: E402
from src.modules.detection import create_detector  # noqa: E402
from src.modules.detection.base import SGGDetector  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute detector features for SGG training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        choices=["fasterrcnn", "efficientdet"],
        help="Detector architecture",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to finetuned detector checkpoint",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="both",
        choices=["gt", "pred", "both"],
        help="Which features to extract (default: both)",
    )

    # Detector-specific
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["resnet50", "resnet101"],
        help="Backbone (Faster R-CNN only)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"],
        help="Variant (EfficientDet only)",
    )

    # Processing
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect)",
    )
    parser.add_argument("--dataset-root", type=str, default="datasets/vrd")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {dataset-root}/features/{detector_name})",
    )
    parser.add_argument("--num-classes", type=int, default=101)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process only first N images (for debugging)",
    )
    parser.add_argument(
        "--relation-iou-thresh",
        type=float,
        default=0.3,
        help="Min IoU for matching predicted boxes to GT boxes when carrying over "
             "GT relation labels (pred features only, default: 0.3)",
    )

    return parser.parse_args()


def _auto_device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Detector helpers
# ---------------------------------------------------------------------------


def get_detector_name(
    detector: str, backbone: str | None, variant: str | None
) -> str:
    """Generate detector directory name matching checkpoint convention."""
    if detector == "fasterrcnn":
        bb = backbone or "resnet50"
        return f"fasterrcnn_{bb}_fpn_v2"
    if detector == "efficientdet":
        v = variant or "d2"
        return f"efficientdet_{v}"
    raise ValueError(f"Unknown detector: {detector}")


def _load_state_dict_from_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Load model state dict from a checkpoint file.

    Handles both raw state dicts and Lightning checkpoints.  Lightning
    checkpoints wrap the model under ``state_dict`` with a ``model.``
    prefix (from ``DetectorLightningModule.model``).
    """
    ckpt: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    # Lightning checkpoint — extract and strip prefix
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        prefix = "model."
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        return sd  # type: ignore[no-any-return]

    # Raw state dict
    return ckpt


def create_frozen_detector(
    detector: str,
    backbone: str | None,
    variant: str | None,
    checkpoint: str,
    num_classes: int,
) -> SGGDetector:
    """Create a frozen detector with loaded checkpoint weights.

    Loads checkpoint separately to handle Lightning checkpoint format.
    Creates with trainable=True to match checkpoint structure (Lightning
    saves both bench_train and bench_predict for EfficientDet), then
    freezes after loading.
    """
    # Build detector with trainable=True to match checkpoint structure.
    # Lightning checkpoints from DetectorLightningModule save the full
    # SGGDetector including bench_train (for EfficientDet).
    kwargs: dict[str, Any] = {
        "freeze": False,
        "trainable": True,
        "num_classes": num_classes,
        "pretrained": False,  # weights come from checkpoint
    }
    if detector == "fasterrcnn":
        if backbone:
            kwargs["backbone"] = backbone
    elif detector == "efficientdet":
        if variant:
            kwargs["variant"] = variant
    else:
        raise ValueError(f"Unknown detector: {detector}")

    model = create_detector(detector, **kwargs)

    # Load checkpoint (handles Lightning and raw formats)
    state_dict = _load_state_dict_from_checkpoint(checkpoint)
    model.load_state_dict(state_dict)  # type: ignore[arg-type]

    # Freeze after loading
    from src.modules.detection.components.freeze import freeze_bn, freeze_module

    freeze_module(model)
    freeze_bn(model)
    model.eval()

    return model


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _collate_fn_pad(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Pad variable-size images to max dims in batch."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = []
    for img in images:
        _, h, w = img.shape
        padded.append(F.pad(img, (0, max_w - w, 0, max_h - h), value=0.0))

    return torch.stack(padded), targets


# ---------------------------------------------------------------------------
# GT annotation parsing
# ---------------------------------------------------------------------------


def parse_gt_annotations(
    relations: list[dict[str, Any]],
    background_class: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Parse VRD annotations into boxes, labels, and relation triplets.

    Uses the same deduplication order as VRDDetectionDataset so that
    object indices are consistent.

    Args:
        relations: Raw VRD annotation list for one image.
        background_class: If True, labels are 1-indexed [1, 100].

    Returns:
        boxes:     (N, 4) float32, xyxy format.
        labels:    (N,)   int64, class indices.
        rel_trips: (R, 3) int64, [subject_idx, object_idx, predicate].
    """
    boxes: list[list[float]] = []
    labels: list[int] = []
    seen: dict[tuple[int, tuple[int, ...]], int] = {}

    for rel in relations:
        for role in ("subject", "object"):
            obj = rel[role]
            category: int = obj["category"]
            bbox = tuple(obj["bbox"])  # (ymin, ymax, xmin, xmax)
            key = (category, bbox)
            if key not in seen:
                seen[key] = len(boxes)
                ymin, ymax, xmin, xmax = bbox
                boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                labels.append(category + 1 if background_class else category)

    triplets: list[list[int]] = []
    for rel in relations:
        sub_key = (rel["subject"]["category"], tuple(rel["subject"]["bbox"]))
        obj_key = (rel["object"]["category"], tuple(rel["object"]["bbox"]))
        triplets.append([seen[sub_key], seen[obj_key], rel["predicate"]])

    if boxes:
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
            torch.tensor(triplets, dtype=torch.int64),
        )
    return (
        torch.zeros(0, 4, dtype=torch.float32),
        torch.zeros(0, dtype=torch.int64),
        torch.zeros(0, 3, dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# IoU-based relation matching for predicted boxes
# ---------------------------------------------------------------------------


def match_relations_to_predictions(
    pred_boxes: Tensor,
    gt_boxes: Tensor,
    gt_relations: Tensor,
    iou_thresh: float = 0.5,
) -> Tensor:
    """Carry GT relation labels over to predicted box indices via IoU matching.

    For each GT box, the predicted box with the highest IoU is chosen as its
    match (if IoU >= iou_thresh).  GT relations where both subject and object
    have a matched predicted box are re-indexed and returned.

    This enables end-to-end SGG evaluation on predicted-box features: the
    stored relations let the evaluator compute recall using the predicted
    object set as the input graph.

    Args:
        pred_boxes:   (N_pred, 4) float32, xyxy.
        gt_boxes:     (N_gt,   4) float32, xyxy.
        gt_relations: (R,      3) int64, [sub_gt, obj_gt, predicate (0-indexed)].
        iou_thresh:   Minimum IoU for a predicted box to be matched to a GT box.

    Returns:
        (R', 3) int64 tensor [sub_pred, obj_pred, predicate (0-indexed)].
        Empty (0, 3) tensor when no pairs can be matched.
    """
    if (
        pred_boxes.shape[0] == 0
        or gt_boxes.shape[0] == 0
        or gt_relations.shape[0] == 0
    ):
        return torch.zeros(0, 3, dtype=torch.int64)

    # iou shape: (N_pred, N_gt)
    iou = box_iou(pred_boxes.float(), gt_boxes.float())

    # For each GT box, best-matching predicted box
    best_iou, best_pred = iou.max(dim=0)  # (N_gt,)

    gt_to_pred: dict[int, int] = {
        int(gt_idx): int(best_pred[gt_idx].item())
        for gt_idx in range(gt_boxes.shape[0])
        if best_iou[gt_idx].item() >= iou_thresh
    }

    matched: list[list[int]] = []
    for r in range(gt_relations.shape[0]):
        s_gt = int(gt_relations[r, 0].item())
        o_gt = int(gt_relations[r, 1].item())
        pred_class = int(gt_relations[r, 2].item())
        if s_gt in gt_to_pred and o_gt in gt_to_pred:
            matched.append([gt_to_pred[s_gt], gt_to_pred[o_gt], pred_class])

    if matched:
        return torch.tensor(matched, dtype=torch.int64)
    return torch.zeros(0, 3, dtype=torch.int64)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_gt_roi_features(
    detector: SGGDetector,
    images: Tensor,
    gt_boxes: list[Tensor],
) -> Tensor:
    """Extract ROI features at ground truth box locations.

    Runs the frozen detector backbone (+ FPN) and pools features at the
    given box coordinates.  Dispatches by detector type.

    Returns:
        (total_N, C, H, W) ROI feature tensor.
    """
    from src.modules.detection.efficientdet import SGGEfficientDet
    from src.modules.detection.faster_rcnn import SGGFasterRCNN

    total_boxes = sum(b.shape[0] for b in gt_boxes)
    if total_boxes == 0:
        c, h, w = detector.roi_feature_dim
        return torch.zeros(0, c, h, w, device=images.device)

    if isinstance(detector, SGGFasterRCNN):
        features = detector.model.backbone(images)
        image_shapes = [(images.shape[2], images.shape[3])] * images.shape[0]
        return detector._pool_roi_features(features, gt_boxes, image_shapes)

    if isinstance(detector, SGGEfficientDet):
        img_size = detector._image_size
        padded_h, padded_w = images.shape[2], images.shape[3]

        # Resize to the variant's expected input size
        if padded_h != img_size or padded_w != img_size:
            images = F.interpolate(
                images,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
            scale = torch.tensor(
                [img_size / padded_w, img_size / padded_h] * 2,
                device=images.device,
            )
            gt_boxes = [b * scale for b in gt_boxes]

        backbone_features = detector.model.backbone(images)
        fpn_features = detector.model.fpn(backbone_features)
        feature_dict = {str(i): fpn_features[i] for i in range(4)}
        return detector._pool_roi_features(feature_dict, gt_boxes)

    raise TypeError(f"Unsupported detector type: {type(detector)}")


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------


def save_image_to_hdf5(
    h5file: h5py.File,
    image_id: str,
    roi_features: Tensor,
    boxes: Tensor,
    labels: Tensor,
    *,
    scores: Tensor | None = None,
    relations: Tensor | None = None,
) -> None:
    """Write one image's data to an HDF5 group."""
    grp = h5file.create_group(image_id)

    rf = roi_features.numpy()
    grp.create_dataset(
        "roi_features",
        data=rf,
        compression="lzf",
        chunks=rf.shape if rf.shape[0] > 0 else None,
    )
    grp.create_dataset("boxes", data=boxes.numpy())
    grp.create_dataset("labels", data=labels.numpy())

    if scores is not None:
        grp.create_dataset("scores", data=scores.numpy())
    if relations is not None:
        grp.create_dataset("relations", data=relations.numpy())


def write_metadata(
    h5file: h5py.File,
    *,
    detector: str,
    backbone: str | None,
    variant: str | None,
    roi_feature_dim: tuple[int, int, int],
    num_classes: int,
    num_images: int,
    background_class: bool,
) -> None:
    """Write metadata attributes to the HDF5 root."""
    h5file.attrs["detector"] = detector
    if backbone is not None:
        h5file.attrs["backbone"] = backbone
    if variant is not None:
        h5file.attrs["variant"] = variant
    h5file.attrs["roi_feature_dim"] = list(roi_feature_dim)
    h5file.attrs["num_classes"] = num_classes
    h5file.attrs["num_images"] = num_images
    h5file.attrs["background_class"] = background_class


# ---------------------------------------------------------------------------
# Processing loops
# ---------------------------------------------------------------------------


def process_predicted(
    detector: SGGDetector,
    loader: DataLoader[tuple[Tensor, dict[str, Tensor]]],
    image_ids: list[str],
    output_path: Path,
    device: str,
    *,
    detector_name: str,
    backbone: str | None,
    variant: str | None,
    num_classes: int,
    background_class: bool,
    raw_annotations: dict[str, list[dict[str, Any]]],
    relation_iou_thresh: float = 0.5,
) -> None:
    """Extract and save predicted features with IoU-matched GT relations.

    Predicted boxes are matched to GT boxes via IoU (>= relation_iou_thresh).
    GT relation labels are carried over to predicted box pairs where both
    the subject and object have a matched predicted box.  This enables
    end-to-end SGG evaluation using the same recall computation as GT-box
    evaluation.

    Images where the detector returns no detections are saved with empty
    tensors and no relation dataset.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_detections = 0
    total_matched_rels = 0
    idx = 0

    with h5py.File(output_path, "w") as f:
        for images, _targets in tqdm(loader, desc="Predicted features"):
            images = images.to(device)
            batch_size = images.shape[0]

            with torch.no_grad():
                output = detector.predict(images)

            offset = 0
            for i in range(batch_size):
                img_id = image_ids[idx]
                idx += 1
                n_i = output.boxes[i].shape[0]
                roi_i = output.roi_features[offset : offset + n_i]
                pred_boxes_i = output.boxes[i].cpu()
                offset += n_i

                # Match GT relations to predicted box indices
                ann = raw_annotations.get(img_id, [])
                matched_rels: Tensor | None = None
                if ann and pred_boxes_i.shape[0] > 0:
                    gt_boxes_i, _, gt_rels_i = parse_gt_annotations(
                        ann, background_class
                    )
                    if gt_boxes_i.shape[0] > 0 and gt_rels_i.shape[0] > 0:
                        matched_rels = match_relations_to_predictions(
                            pred_boxes_i,
                            gt_boxes_i,
                            gt_rels_i,
                            iou_thresh=relation_iou_thresh,
                        )
                        total_matched_rels += matched_rels.shape[0]

                save_image_to_hdf5(
                    f,
                    img_id,
                    roi_i.cpu(),
                    pred_boxes_i,
                    output.labels[i].cpu(),
                    scores=output.scores[i].cpu(),
                    relations=matched_rels if (matched_rels is not None and matched_rels.shape[0] > 0) else None,
                )
                total_detections += n_i

        write_metadata(
            f,
            detector=detector_name,
            backbone=backbone,
            variant=variant,
            roi_feature_dim=detector.roi_feature_dim,
            num_classes=num_classes,
            num_images=idx,
            background_class=background_class,
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    avg_det = total_detections / max(idx, 1)
    avg_rel = total_matched_rels / max(idx, 1)
    print(f"  Saved {idx} images ({total_detections} detections, avg {avg_det:.1f}/img)")
    print(f"  Matched GT relations: {total_matched_rels} (avg {avg_rel:.1f}/img)")
    print(f"  File: {output_path} ({size_mb:.1f} MB)")


def process_gt(
    detector: SGGDetector,
    loader: DataLoader[tuple[Tensor, dict[str, Tensor]]],
    image_ids: list[str],
    raw_annotations: dict[str, list[dict[str, Any]]],
    output_path: Path,
    device: str,
    *,
    detector_name: str,
    backbone: str | None,
    variant: str | None,
    num_classes: int,
    background_class: bool,
) -> None:
    """Extract and save ground truth features."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_objects = 0
    total_relations = 0
    idx = 0

    with h5py.File(output_path, "w") as f:
        for images, _targets in tqdm(loader, desc="GT features"):
            images = images.to(device)
            batch_size = images.shape[0]

            # Parse GT annotations for this batch
            batch_boxes: list[Tensor] = []
            batch_labels: list[Tensor] = []
            batch_relations: list[Tensor] = []

            for i in range(batch_size):
                img_id = image_ids[idx + i]
                ann = raw_annotations.get(img_id, [])
                boxes_i, labels_i, rels_i = parse_gt_annotations(
                    ann, background_class
                )
                batch_boxes.append(boxes_i)
                batch_labels.append(labels_i)
                batch_relations.append(rels_i)

            # Move GT boxes to device for ROI pooling
            gt_boxes_device = [b.to(device) for b in batch_boxes]

            with torch.no_grad():
                roi_features = extract_gt_roi_features(
                    detector, images, gt_boxes_device
                )

            # Save per-image
            offset = 0
            for i in range(batch_size):
                img_id = image_ids[idx]
                idx += 1
                n_i = batch_boxes[i].shape[0]
                roi_i = roi_features[offset : offset + n_i]
                offset += n_i

                save_image_to_hdf5(
                    f,
                    img_id,
                    roi_i.cpu(),
                    batch_boxes[i],
                    batch_labels[i],
                    relations=batch_relations[i],
                )
                total_objects += n_i
                total_relations += batch_relations[i].shape[0]

        write_metadata(
            f,
            detector=detector_name,
            backbone=backbone,
            variant=variant,
            roi_feature_dim=detector.roi_feature_dim,
            num_classes=num_classes,
            num_images=idx,
            background_class=background_class,
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    avg_obj = total_objects / max(idx, 1)
    avg_rel = total_relations / max(idx, 1)
    print(
        f"  Saved {idx} images ({total_objects} objects avg {avg_obj:.1f}/img, "
        f"{total_relations} relations avg {avg_rel:.1f}/img)"
    )
    print(f"  File: {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    detector_type: str,
    split: Literal["train", "test"],
    checkpoint: str,
    source: Literal["gt", "pred", "both"],
    backbone: str | None,
    variant: str | None,
    batch_size: int,
    device: str,
    dataset_root: str,
    output_dir: str | None,
    num_classes: int,
    max_images: int | None,
    relation_iou_thresh: float = 0.3,
) -> None:
    """Run feature precomputation."""
    background_class = True
    det_name = get_detector_name(detector_type, backbone, variant)

    # Output directory
    out_dir = Path(output_dir) if output_dir else Path(dataset_root) / "features" / det_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading VRD {split} from {dataset_root}...")
    dataset = VRDDetectionDataset(
        root=dataset_root, split=split, background_class=background_class
    )
    image_ids = dataset.image_ids

    # Optionally limit for debugging
    ds: VRDDetectionDataset | Subset[Any]
    if max_images is not None and max_images < len(dataset):
        ds = Subset(dataset, list(range(max_images)))  # type: ignore[arg-type]
        image_ids = image_ids[:max_images]
        print(f"  Limited to first {max_images} images")
    else:
        ds = dataset
    print(f"  {len(ds)} images, {num_classes} classes")

    # Load raw annotations (needed for GT relations and pred IoU matching)
    ann_path = Path(dataset_root) / f"annotations_{split}.json"
    with open(ann_path) as f:
        raw_annotations: dict[str, list[dict[str, Any]]] = json.load(f)

    # Create detector
    print(f"Loading detector: {det_name} (checkpoint: {checkpoint})")
    model = create_frozen_detector(
        detector_type, backbone, variant, checkpoint, num_classes
    )
    model = model.to(device)
    print(f"  ROI feature dim: {model.roi_feature_dim}")

    # DataLoader (num_workers=0 per project constraint)
    loader: DataLoader[Any] = DataLoader(
        ds,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_fn_pad,
    )

    # Predicted features
    if source in ("pred", "both"):
        pred_path = out_dir / f"pred_{split}.h5"
        print(f"\nExtracting predicted features → {pred_path}")
        process_predicted(
            model,
            loader,
            image_ids,
            pred_path,
            device,
            detector_name=det_name,
            backbone=backbone,
            variant=variant,
            num_classes=num_classes,
            background_class=background_class,
            raw_annotations=raw_annotations,
            relation_iou_thresh=relation_iou_thresh,
        )

    # GT features
    if source in ("gt", "both"):
        gt_path = out_dir / f"gt_{split}.h5"
        print(f"\nExtracting GT features → {gt_path}")
        process_gt(
            model,
            loader,
            image_ids,
            raw_annotations,
            gt_path,
            device,
            detector_name=det_name,
            backbone=backbone,
            variant=variant,
            num_classes=num_classes,
            background_class=background_class,
        )

    print("\nDone.")


if __name__ == "__main__":
    args = parse_args()

    device = args.device or _auto_device()
    split = args.split
    if split not in ("train", "test"):
        raise ValueError(f"Invalid split: {split}")
    split_literal: Literal["train", "test"] = split  # type: ignore[assignment]
    source = args.source
    if source not in ("gt", "pred", "both"):
        raise ValueError(f"Invalid source: {source}")
    source_literal: Literal["gt", "pred", "both"] = source  # type: ignore[assignment]

    main(
        detector_type=args.detector,
        split=split_literal,
        checkpoint=args.checkpoint,
        source=source_literal,
        backbone=args.backbone,
        variant=args.variant,
        batch_size=args.batch_size,
        device=device,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        max_images=args.max_images,
        relation_iou_thresh=args.relation_iou_thresh,
    )
