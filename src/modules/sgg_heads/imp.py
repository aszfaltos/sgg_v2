"""IMP (Iterative Message Passing) relation head.

Implements the base paper's extension of NMP:
  GRU cells replace stateless MLPs for node and edge updates, and
  four learned attention gates (α_s, α_o, β_out, β_in) control
  how much each neighbour's signal is incorporated each iteration.

Reference: "Semantic and Structural Graph Enhancements for Scene Graph
Generation" — extends NMP (Hu et al., arXiv:2208.04165) with GRU +
attention gates, corresponding to the IMP head (homogeneous graph, VRD).

Differences from NMPHead:
  - 2-layer MLPs → GRUCell for both node and edge updates
  - 4 scalar attention gates (Linear(2d→1) + Sigmoid)
  - num_iter rounds (default 3) instead of a single pass
  - Node aggregation uses sum (not mean) — GRU state handles degree variation

The graph topology and geometric encoding arrive precomputed in the batch
dict. All gate/GRU calls run in one shot over (total_N, d) / (total_E, d)
tensors with no per-image Python loop.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .base import SGGHead, SGGHeadOutput


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer MLP with ELU activation (square hidden layer)."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ELU(),
        nn.Linear(out_dim, out_dim),
        nn.ELU(),
    )


def _make_gate(d_hidden: int) -> nn.Sequential:
    """Scalar attention gate: Linear(2*d → 1) + Sigmoid."""
    return nn.Sequential(nn.Linear(2 * d_hidden, 1), nn.Sigmoid())


class IMPHead(SGGHead):
    """Iterative Message Passing head for scene graph generation.

    Extends NMPHead by replacing stateless MLPs with GRU cells and adding
    four learned scalar attention gates. Runs `num_iter` rounds of
    edge→node message passing, with node and edge representations
    accumulating context across rounds.

    Args:
        roi_feature_dim: (C, H, W) shape of ROI features from the detector
            (e.g. (256, 7, 7) for Faster R-CNN). Use detector.roi_feature_dim.
        num_predicates: Number of relation/predicate classes (e.g. 70 for VRD).
        semantic_dim: Dimension of word embeddings passed into forward()
            (300 for GloVe/Word2Vec, 768 for BERT, 384 for MiniLM).
        d_hidden: Hidden dimension for all layers.
        num_iter: Number of message passing iterations (default: 3).

    Example:
        >>> head = IMPHead(roi_feature_dim=(256, 7, 7), num_predicates=70)
        >>> embedding = torch.randn(101, 300)  # GloVe, 100 VRD classes + background
        >>> out = head(batch, embedding)
        >>> out.rel_logits[0].shape  # (E_0, 71)  — 70 predicates + 1 background
    """

    _GEO_DIM: int = 12  # output dim of compute_geometric_encoding

    def __init__(
        self,
        roi_feature_dim: tuple[int, int, int],
        num_predicates: int,
        semantic_dim: int = 300,
        d_hidden: int = 512,
        num_iter: int = 3,
    ) -> None:
        super().__init__()

        self._num_predicates = num_predicates
        self._num_iter = num_iter

        d_vis = roi_feature_dim[0]  # channel count after adaptive pool

        # ROI spatial collapse: (N, C, H, W) → (N, C)
        self._adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Node initialisation: embed [visual ; semantic] → hidden space
        self._f_emb = _make_mlp(d_vis + semantic_dim, d_hidden)

        # Edge initialisation: project node pair → hidden space
        self._f_edge_init = _make_mlp(2 * d_hidden, d_hidden)

        # GRU cells — one step at a time, iterate manually
        self._node_gru = nn.GRUCell(d_hidden, d_hidden)
        self._edge_gru = nn.GRUCell(d_hidden, d_hidden)

        # Attention gates (scalar ∈ [0,1] per edge or per-dim, broadcast over d)
        # Edge update gates: control subject/object contribution to edge
        self._gate_sub = _make_gate(d_hidden)  # α_s
        self._gate_obj = _make_gate(d_hidden)  # α_o
        # Node update gates: control outgoing/incoming edge contribution to node
        self._gate_out = _make_gate(d_hidden)  # β_out
        self._gate_in  = _make_gate(d_hidden)  # β_in

        # Classifier — appends spatial geo at classification time
        # Output: num_predicates + 1 classes (index 0 = no-relation background)
        self._rel_classifier = nn.Linear(d_hidden + self._GEO_DIM, num_predicates + 1)

    @property
    def num_predicates(self) -> int:
        return self._num_predicates

    def forward(
        self,
        batch: dict[str, Tensor],
        embedding_lookup: Tensor,
    ) -> SGGHeadOutput:
        """Run iterative message passing and classify relations.

        All gate and GRU calls operate on the full batch at once (no per-image
        loop). Graph topology and geometric encoding arrive precomputed in
        `batch`.

        Args:
            batch: Collated batch from sgg_collate with keys:
                roi_features, labels, sub_idx, obj_idx, geo,
                node_counts, edge_counts, image_offsets.
            embedding_lookup: (num_classes, semantic_dim) word embedding matrix,
                indexed by 1-indexed class labels.

        Returns:
            SGGHeadOutput with per-image relation logits and local pair indices.
        """
        roi_features: Tensor = batch["roi_features"]   # (total_N, C, H, W)
        labels: Tensor = batch["labels"]               # (total_N,)
        sub_idx: Tensor = batch["sub_idx"]             # (total_E,) global
        obj_idx: Tensor = batch["obj_idx"]             # (total_E,) global
        geo: Tensor = batch["geo"]                     # (total_E, 12)
        node_counts: Tensor = batch["node_counts"]     # (B,)
        edge_counts: Tensor = batch["edge_counts"]     # (B,)
        image_offsets: Tensor = batch["image_offsets"] # (B,)

        total_N = roi_features.shape[0]
        total_E = sub_idx.shape[0]

        # -- Handle empty batch -------------------------------------------
        if total_N == 0 or total_E == 0:
            B = node_counts.shape[0]
            empty_logits = [
                torch.zeros(0, self._num_predicates + 1, device=roi_features.device)
                for _ in range(B)
            ]
            empty_idx = [
                torch.zeros(0, dtype=torch.long, device=roi_features.device)
                for _ in range(B)
            ]
            return SGGHeadOutput(
                rel_logits=empty_logits,
                subject_indices=empty_idx,
                object_indices=empty_idx,
            )

        # -- Step 1: flatten ROI features (total_N, C, H, W) → (total_N, C) --
        vis = self._adaptive_pool(roi_features).flatten(1)  # (total_N, d_vis)

        # -- Step 2: semantic embeddings for all nodes --------------------
        sem = embedding_lookup[labels]  # (total_N, semantic_dim)

        # -- Step 3: initialise node hidden states ------------------------
        o_init = self._f_emb(torch.cat([vis, sem], dim=-1))  # (total_N, d)
        h_v = self._node_gru(o_init)                         # (total_N, d)

        # -- Step 4: initialise edge hidden states ------------------------
        e_init = self._f_edge_init(
            torch.cat([h_v[sub_idx], h_v[obj_idx]], dim=-1)
        )  # (total_E, d)
        h_e = self._edge_gru(e_init)  # (total_E, d)

        # -- Step 5: iterative message passing ----------------------------
        d = h_v.shape[1]
        for _ in range(self._num_iter):
            # -- Edge update --
            # α_s, α_o: scalar gates controlling subject/object → edge signal
            alpha_s = self._gate_sub(torch.cat([h_v[sub_idx], h_e], dim=-1))  # (E, 1)
            alpha_o = self._gate_obj(torch.cat([h_v[obj_idx], h_e], dim=-1))  # (E, 1)
            msg_e = alpha_s * h_v[sub_idx] + alpha_o * h_v[obj_idx]           # (E, d)
            h_e = self._edge_gru(msg_e, h_e)                                   # (E, d)

            # -- Node update --
            # β_out gates outgoing edge signals (edges where node is subject)
            # β_in  gates incoming edge signals (edges where node is object)
            beta_out = self._gate_out(torch.cat([h_v[sub_idx], h_e], dim=-1)) * h_e  # (E, d)
            beta_in  = self._gate_in(torch.cat([h_v[obj_idx], h_e], dim=-1)) * h_e   # (E, d)

            # Sum-aggregate gated edge signals to nodes
            agg = h_v.new_zeros(total_N, d)
            agg.index_add_(0, sub_idx, beta_out)  # outgoing: edges where node is subject
            agg.index_add_(0, obj_idx, beta_in)   # incoming: edges where node is object
            h_v = self._node_gru(agg, h_v)        # (total_N, d)

        # -- Step 6: classify with precomputed geo features ---------------
        logits = self._rel_classifier(
            torch.cat([h_e, geo], dim=-1)
        )  # (total_E, num_predicates + 1)

        # -- Step 7: split back to per-image lists ------------------------
        ec_list = edge_counts.tolist()
        off_list = image_offsets.tolist()

        rel_logits = list(logits.split(ec_list))

        sub_splits = sub_idx.split(ec_list)
        obj_splits = obj_idx.split(ec_list)
        subject_indices = [s - int(off) for s, off in zip(sub_splits, off_list)]
        object_indices = [o - int(off) for o, off in zip(obj_splits, off_list)]

        return SGGHeadOutput(
            rel_logits=rel_logits,
            subject_indices=subject_indices,
            object_indices=object_indices,
        )
