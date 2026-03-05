"""NMP (Neural Message Passing) relation head.

Implements the original NMP architecture from:
  Hu et al., "Neural Message Passing for Visual Relationship Detection",
  ICML Workshop 2019, arXiv:2208.04165.

Uses 2-layer MLPs with ELU activation throughout — no GRU, no attention gates.
Designed for the VRD dataset with a homogeneous object graph (objects as nodes,
relations as edges).

The graph (edge indices and geometric encoding) is built at dataset __getitem__
time and arrives precomputed in the batch dict. The forward pass runs all MLP
calls in one shot over (total_N, d) and (total_E, d) tensors with no per-image
Python loop — only the final split back to per-image lists uses a loop.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .base import SGGHead, SGGHeadOutput
from .utils.features import aggregate_edge_to_node


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer MLP with ELU activation (square hidden layer)."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ELU(),
        nn.Linear(out_dim, out_dim),
        nn.ELU(),
    )


class NMPHead(SGGHead):
    """Neural Message Passing head for scene graph generation.

    Builds a directed object graph per image and runs one round of message
    passing (node→edge→node→edge→fuse) before classifying each edge as a
    relation. Spatial features are appended at classification time, matching
    the original NMP paper design.

    The graph topology and geometric features arrive precomputed in the batch
    dict (produced by SGGPrecomputedDataset + sgg_collate), so all MLP calls
    operate over the full batch in one shot.

    Args:
        roi_feature_dim: (C, H, W) shape of ROI features from the detector
            (e.g. (256, 7, 7) for Faster R-CNN). Use detector.roi_feature_dim.
        num_predicates: Number of relation/predicate classes (e.g. 70 for VRD).
        semantic_dim: Dimension of word embeddings passed into forward()
            (300 for GloVe/Word2Vec, 768 for BERT, 384 for MiniLM).
        d_hidden: Hidden dimension for all MLP layers.

    Example:
        >>> head = NMPHead(roi_feature_dim=(256, 7, 7), num_predicates=70)
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
    ) -> None:
        super().__init__()

        self._num_predicates = num_predicates

        d_vis = roi_feature_dim[0]  # channel count after adaptive pool

        # ROI spatial collapse: (N, C, H, W) → (N, C)
        self._adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Eq. 2 — f_emb: embed [visual ; semantic] → hidden space
        self._f_emb = _make_mlp(d_vis + semantic_dim, d_hidden)

        # Eq. 3 — f1_e: local edge from node pair concat
        self._f1_e = _make_mlp(2 * d_hidden, d_hidden)

        # Eq. 4 — f1_v: node update from [mean_in ; mean_out]
        self._f1_v = _make_mlp(2 * d_hidden, d_hidden)

        # Eq. 5 — f2_e: global edge from updated node pair
        self._f2_e = _make_mlp(2 * d_hidden, d_hidden)

        # Eq. 6 — f_fusion: fuse local + global edge
        self._f_fusion = _make_mlp(2 * d_hidden, d_hidden)

        # Classifier — appends spatial lᵢⱼ at classification time (NMP paper design)
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
        """Run NMP message passing and classify relations.

        All MLP calls operate on the full batch at once (no per-image loop).
        Graph topology and geometric encoding arrive precomputed in `batch`.

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

        # -- Step 3: Eq. 2 — embed to hidden space (one MLP call) --------
        o1 = self._f_emb(torch.cat([vis, sem], dim=-1))  # (total_N, d_hidden)

        # -- Step 4: Eq. 3 — local edge features (one MLP call) ----------
        e1 = self._f1_e(
            torch.cat([o1[sub_idx], o1[obj_idx]], dim=-1)
        )  # (total_E, d_hidden)

        # -- Step 5: Eq. 4 — node update (one index_add_ pass) -----------
        agg = aggregate_edge_to_node(e1, sub_idx, obj_idx, total_N)  # (total_N, 2*d_hidden)
        o2 = self._f1_v(agg)  # (total_N, d_hidden)

        # -- Step 6: Eq. 5 — global edge features (one MLP call) ---------
        e2 = self._f2_e(
            torch.cat([o2[sub_idx], o2[obj_idx]], dim=-1)
        )  # (total_E, d_hidden)

        # -- Step 7: Eq. 6 — fuse local + global (one MLP call) ----------
        e_fused = self._f_fusion(torch.cat([e1, e2], dim=-1))  # (total_E, d_hidden)

        # -- Step 8: classify with precomputed geo features ---------------
        logits = self._rel_classifier(
            torch.cat([e_fused, geo], dim=-1)
        )  # (total_E, num_predicates + 1); index 0 = background/no-relation

        # -- Step 9: split back to per-image lists ------------------------
        # logits and edge indices split by edge_counts
        ec_list = edge_counts.tolist()
        off_list = image_offsets.tolist()

        rel_logits = list(logits.split(ec_list))

        # Convert global edge indices to image-local by subtracting node offset
        sub_splits = sub_idx.split(ec_list)
        obj_splits = obj_idx.split(ec_list)
        subject_indices = [s - int(off) for s, off in zip(sub_splits, off_list)]
        object_indices = [o - int(off) for o, off in zip(obj_splits, off_list)]

        return SGGHeadOutput(
            rel_logits=rel_logits,
            subject_indices=subject_indices,
            object_indices=object_indices,
        )
