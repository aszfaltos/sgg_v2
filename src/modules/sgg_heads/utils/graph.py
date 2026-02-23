"""Graph construction utilities for SGG heads.

Functions for building edge indices and computing geometric features
from detected bounding boxes. Used by all SGG heads.
"""

import torch
import torchvision.ops
from torch import Tensor


def build_edge_index(
    boxes: Tensor,
    dis_thresh: float = 0.5,
    iou_thresh: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Build directed edge index from detected boxes using spatial heuristics.

    An edge exists between i and j (i != j) when either:
      - normalised centre distance < dis_thresh, OR
      - IoU > iou_thresh

    Args:
        boxes: (N, 4) xyxy absolute coordinates for one image.
        dis_thresh: Distance threshold. Distance is normalised by sqrt(image area),
            where image area is inferred as max_x * max_y from box extents.
        iou_thresh: IoU threshold.

    Returns:
        Tuple of (subject_indices, object_indices), each (E,) int64.
        Both are empty tensors of shape (0,) when N == 0 or no edges pass.
    """
    N = boxes.shape[0]
    device = boxes.device

    if N == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    # Normalised centre distances
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2  # (N, 2)
    image_area = boxes[:, 2].max() * boxes[:, 3].max()
    diff = centres.unsqueeze(0) - centres.unsqueeze(1)  # (N, N, 2)
    dist = diff.norm(dim=-1) / image_area.sqrt().clamp(min=1e-6)  # (N, N)

    iou = torchvision.ops.box_iou(boxes, boxes)  # (N, N)

    mask = (dist < dis_thresh) | (iou > iou_thresh)  # (N, N)
    mask.fill_diagonal_(False)

    sub_idx, obj_idx = mask.nonzero(as_tuple=True)
    return sub_idx, obj_idx


def compute_geometric_encoding(
    boxes_sub: Tensor,
    boxes_obj: Tensor,
    image_wh: tuple[float, float],
) -> Tensor:
    """Compute 12-dim geometric feature for each (subject, object) pair.

    Encodes normalised box coordinates for both subject and object:
      [x1/W, y1/H, x2/W, y2/H, w/W, h/H] for each box, concatenated.

    Args:
        boxes_sub: (E, 4) xyxy for subject boxes.
        boxes_obj: (E, 4) xyxy for object boxes.
        image_wh: (width, height) for normalisation.

    Returns:
        (E, 12) geometric feature tensor.
    """
    W, H = image_wh

    def _encode(b: Tensor) -> Tensor:
        x1 = b[:, 0] / W
        y1 = b[:, 1] / H
        x2 = b[:, 2] / W
        y2 = b[:, 3] / H
        w = (b[:, 2] - b[:, 0]) / W
        h = (b[:, 3] - b[:, 1]) / H
        return torch.stack([x1, y1, x2, y2, w, h], dim=-1)  # (E, 6)

    return torch.cat([_encode(boxes_sub), _encode(boxes_obj)], dim=-1)  # (E, 12)


def compute_union_boxes(
    boxes_sub: Tensor,
    boxes_obj: Tensor,
) -> Tensor:
    """Compute union bounding box for each (subject, object) pair.

    Args:
        boxes_sub: (E, 4) xyxy for subject boxes.
        boxes_obj: (E, 4) xyxy for object boxes.

    Returns:
        (E, 4) xyxy union boxes.
    """
    xy_min = torch.minimum(boxes_sub[:, :2], boxes_obj[:, :2])  # (E, 2)
    xy_max = torch.maximum(boxes_sub[:, 2:], boxes_obj[:, 2:])  # (E, 2)
    return torch.cat([xy_min, xy_max], dim=-1)  # (E, 4)
