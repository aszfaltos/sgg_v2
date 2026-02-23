"""Feature aggregation utilities for SGG heads.

Functions for aggregating edge messages to nodes during message passing.
Used by all SGG heads that operate on per-image variable-size graphs.
"""

import torch
from torch import Tensor


def aggregate_edge_to_node(
    e: Tensor,
    sub_idx: Tensor,
    obj_idx: Tensor,
    n_nodes: int,
) -> Tensor:
    """Aggregate edge features to nodes by mean pooling, direction-aware.

    For each node i, computes:
      - mean of all incoming edge features (edges where i is the object)
      - mean of all outgoing edge features (edges where i is the subject)

    Concatenates [mean_in ; mean_out] to preserve directionality, matching
    NMP paper Eq. 4: o²ᵢ = f¹_v([1/dⁱⁿ · Σ e¹_in ; 1/dᵒᵘᵗ · Σ e¹_out]).

    Nodes with no incoming or no outgoing edges receive zero vectors for that
    component (from clamp(min=1) division on zero count).

    Args:
        e: (E, d) edge feature matrix.
        sub_idx: (E,) int64 subject (source) node indices.
        obj_idx: (E,) int64 object (target) node indices.
        n_nodes: Number of nodes N.

    Returns:
        (N, 2*d) concatenated [mean_in ; mean_out] per node.
    """
    d = e.shape[-1]
    device = e.device

    # Outgoing: i is subject
    agg_out = torch.zeros(n_nodes, d, device=device)
    count_out = torch.zeros(n_nodes, 1, device=device)
    agg_out.index_add_(0, sub_idx, e)
    count_out.index_add_(0, sub_idx, torch.ones(e.shape[0], 1, device=device))
    mean_out = agg_out / count_out.clamp(min=1)  # (N, d)

    # Incoming: i is object
    agg_in = torch.zeros(n_nodes, d, device=device)
    count_in = torch.zeros(n_nodes, 1, device=device)
    agg_in.index_add_(0, obj_idx, e)
    count_in.index_add_(0, obj_idx, torch.ones(e.shape[0], 1, device=device))
    mean_in = agg_in / count_in.clamp(min=1)  # (N, d)

    return torch.cat([mean_in, mean_out], dim=-1)  # (N, 2*d)
