"""Graph construction utilities for SGG heads.

Functions for building edge indices and computing geometric features
from detected bounding boxes. Used by all SGG heads.
"""

import torch
from torch import Tensor


def build_edge_index(boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Build all directed edges between N detected boxes.

    Returns all N*(N-1) directed pairs (i→j for i≠j).  This matches the
    reference implementation which uses REQUIRE_BOX_OVERLAP=false for VRD/VG:
    no spatial filtering is applied so the relation head sees every candidate
    pair and learns which ones carry a relation.

    Args:
        boxes: (N, 4) xyxy absolute coordinates for one image.

    Returns:
        Tuple of (subject_indices, object_indices), each (E,) int64.
        Both are empty tensors of shape (0,) when N == 0.
    """
    N = boxes.shape[0]
    device = boxes.device

    if N == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    mask = torch.ones(N, N, dtype=torch.bool, device=device)
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
