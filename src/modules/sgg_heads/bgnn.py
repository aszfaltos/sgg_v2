"""BGNN (Bipartite Graph Neural Network) relation head.

Implements the bipartite entity-predicate graph from:
  Li et al., "Bipartite Graph Network with Adaptive Message Passing for
  Unbiased Scene Graph Generation", CVPR 2021.

As extended by the base paper with BERT/Word2Vec entity semantics.

Graph structure:
  - Entity nodes n_i  ∈ R^d  (one per detected object)
  - Predicate nodes r_{i→j} ∈ R^d  (one per candidate pair)
  - Bipartite edges: E_e2p (entity → predicate) and E_p2e (predicate → entity)

Two AMP update types per iteration (synchronous — both use values from l):

  Entity → Predicate (Eq. 5):
    d_s = σ(w_s^T [r ; n_sub])   subject affinity gate
    d_o = σ(w_o^T [r ; n_obj])   object affinity gate
    r^{l+1} = r^l + ReLU(d_s · W_r(n_sub) + d_o · W_r(n_obj))

  Predicate → Entity (Eq. 6, PCE-gated):
    s  = σ(MLP([r ; n_sub ; n_obj]))            predicate confidence
    γ  = clamp(α(s − β), 0, 1)                 learnable piecewise gate
    agg_sub[i] = mean_{k∈B_s(i)} γ_k d_s_k W_n(r_k)
    agg_obj[i] = mean_{k∈B_o(i)} γ_k d_o_k W_n(r_k)
    n^{l+1} = n^l + ReLU(agg_sub + agg_obj)

Predicate nodes are initialised from union visual features (precomputed and
stored in the HDF5 by scripts/precompute_sgg_features.py --save-union-features).
Entity nodes are initialised from visual + semantic + positional features.

Designed for Visual Genome (VG, 150 objects, 50 predicates).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import SGGHead, SGGHeadOutput


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer MLP with ELU activation."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ELU(),
        nn.Linear(out_dim, out_dim),
        nn.ELU(),
    )


class BGNNHead(SGGHead):
    """Bipartite Graph Neural Network head for scene graph generation.

    Uses separate entity and predicate nodes with confidence-gated
    Adaptive Message Passing (AMP) and PCE (Predicate Confidence Estimation)
    to suppress noisy candidate pairs from polluting entity representations.

    Requires HDF5 files precomputed with --save-union-features to provide
    union visual features for predicate node initialisation.

    Args:
        roi_feature_dim: (C, H, W) shape of ROI features from the detector.
            Use detector.roi_feature_dim.
        num_predicates: Number of relation/predicate classes (e.g. 50 for VG).
        semantic_dim: Dimension of word embeddings (300 for GloVe/Word2Vec,
            768 for BERT, 384 for MiniLM).
        d_hidden: Hidden dimension for all layers.
        num_iter: Number of AMP iterations (default: 2, BGNN paper best).

    Example:
        >>> head = BGNNHead(roi_feature_dim=(256, 7, 7), num_predicates=50)
        >>> embedding = torch.randn(151, 300)  # GloVe, 150 VG classes + background
        >>> out = head(batch, embedding)
        >>> out.rel_logits[0].shape  # (E_0, 51)  — 50 predicates + 1 background
    """

    _GEO_DIM: int = 12   # compute_geometric_encoding output dim
    _POS_DIM: int = 4    # per-node position feature (normalised box coords)

    def __init__(
        self,
        roi_feature_dim: tuple[int, int, int],
        num_predicates: int,
        semantic_dim: int = 300,
        d_hidden: int = 512,
        num_iter: int = 2,
    ) -> None:
        super().__init__()

        self._num_predicates = num_predicates
        self._num_iter = num_iter

        d_vis = roi_feature_dim[0]  # channel count after adaptive pool

        # Entity features: ROI spatial collapse (N, C, H, W) → (N, C)
        self._adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Entity node init: [visual ; semantic ; position] → d_hidden
        self._f_node = _make_mlp(d_vis + semantic_dim + self._POS_DIM, d_hidden)

        # Predicate node init: [union_visual ; geo] → d_hidden
        # union_features arrives pre-pooled as (E, C) — no spatial collapse needed
        self._f_pred = _make_mlp(d_vis + self._GEO_DIM, d_hidden)

        # AMP gates: entity → predicate affinity scalars
        self._gate_sub = nn.Sequential(nn.Linear(2 * d_hidden, 1), nn.Sigmoid())  # d_s
        self._gate_obj = nn.Sequential(nn.Linear(2 * d_hidden, 1), nn.Sigmoid())  # d_o

        # AMP weight matrices
        self._W_r = nn.Linear(d_hidden, d_hidden)  # entity message for predicate update
        self._W_n = nn.Linear(d_hidden, d_hidden)  # predicate message for entity update

        # PCE: predicate confidence estimator
        # s = σ(MLP([r ; n_sub ; n_obj])) ∈ (0, 1)
        self._pce_net = nn.Sequential(
            nn.Linear(3 * d_hidden, d_hidden),
            nn.ELU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )
        # Learnable piecewise gate parameters, init from BGNN paper
        self._pce_alpha = nn.Parameter(torch.tensor(2.2))
        self._pce_beta = nn.Parameter(torch.tensor(0.025))

        # Predicate classifier: index 0 = background/no-relation
        self._rel_classifier = nn.Linear(d_hidden, num_predicates + 1)

    @property
    def num_predicates(self) -> int:
        return self._num_predicates

    def forward(
        self,
        batch: dict[str, Tensor],
        embedding_lookup: Tensor,
    ) -> SGGHeadOutput:
        """Run bipartite AMP and classify relations.

        Both entity and predicate updates are synchronous: gates and messages
        are computed from iteration-l values before either node type is updated.

        Args:
            batch: Collated batch from sgg_collate. Must include 'union_features'
                (total_E, C) in addition to the standard keys.
            embedding_lookup: (num_classes, semantic_dim) word embedding matrix.

        Returns:
            SGGHeadOutput with per-image relation logits and local pair indices.
        """
        roi_features: Tensor = batch["roi_features"]    # (total_N, C, H, W)
        union_features: Tensor = batch["union_features"] # (total_E, C)
        boxes: Tensor = batch["boxes"]                  # (total_N, 4)
        labels: Tensor = batch["labels"]                # (total_N,)
        sub_idx: Tensor = batch["sub_idx"]              # (total_E,) global
        obj_idx: Tensor = batch["obj_idx"]              # (total_E,) global
        geo: Tensor = batch["geo"]                      # (total_E, 12)
        node_counts: Tensor = batch["node_counts"]      # (B,)
        edge_counts: Tensor = batch["edge_counts"]      # (B,)
        image_offsets: Tensor = batch["image_offsets"]  # (B,)

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

        # -- Step 1: Entity node init -------------------------------------
        vis = self._adaptive_pool(roi_features).flatten(1)  # (N, d_vis)
        sem = embedding_lookup[labels]                       # (N, semantic_dim)
        pos = boxes / 1000.0                                 # (N, 4) rough normalisation
        n = self._f_node(torch.cat([vis, sem, pos], dim=-1)) # (N, d_hidden)

        # -- Step 2: Predicate node init ----------------------------------
        # union_features: (E, C) — already global-average-pooled at precompute time
        r = self._f_pred(torch.cat([union_features, geo], dim=-1))  # (E, d_hidden)

        # -- Step 3: AMP iterations (synchronous update) ------------------
        d = n.shape[1]
        for _ in range(self._num_iter):
            # Compute all gates from old n, old r (synchronous)
            d_s = self._gate_sub(torch.cat([r, n[sub_idx]], dim=-1))    # (E, 1)
            d_o = self._gate_obj(torch.cat([r, n[obj_idx]], dim=-1))    # (E, 1)
            s = self._pce_net(
                torch.cat([r, n[sub_idx], n[obj_idx]], dim=-1)
            )  # (E, 1)
            gamma = (self._pce_alpha * (s - self._pce_beta)).clamp(0.0, 1.0)  # (E, 1)

            # Entity → Predicate (Eq. 5), uses old n
            r_new = r + F.relu(
                d_s * self._W_r(n[sub_idx]) + d_o * self._W_r(n[obj_idx])
            )  # (E, d)

            # Predicate → Entity (Eq. 6), uses old r
            msg = self._W_n(r)  # (E, d) — transform old r

            # Subject side: B_s(i) = predicates where entity i is subject
            weighted_sub = (gamma * d_s) * msg  # (E, d)
            agg_sub = n.new_zeros(total_N, d)
            cnt_sub = n.new_zeros(total_N, 1)
            agg_sub.index_add_(0, sub_idx, weighted_sub)
            cnt_sub.index_add_(0, sub_idx, torch.ones(total_E, 1, device=r.device))
            agg_sub = agg_sub / cnt_sub.clamp(min=1.0)

            # Object side: B_o(i) = predicates where entity i is object
            weighted_obj = (gamma * d_o) * msg  # (E, d)
            agg_obj = n.new_zeros(total_N, d)
            cnt_obj = n.new_zeros(total_N, 1)
            agg_obj.index_add_(0, obj_idx, weighted_obj)
            cnt_obj.index_add_(0, obj_idx, torch.ones(total_E, 1, device=r.device))
            agg_obj = agg_obj / cnt_obj.clamp(min=1.0)

            n = n + F.relu(agg_sub + agg_obj)
            r = r_new

        # -- Step 4: classify predicates ----------------------------------
        logits = self._rel_classifier(r)  # (total_E, num_predicates + 1)

        # -- Step 5: split back to per-image lists ------------------------
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
