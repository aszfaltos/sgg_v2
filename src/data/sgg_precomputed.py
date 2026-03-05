"""Dataset for precomputed SGG detector features.

Loads per-image ROI features, boxes, labels, and relations from an HDF5 file
(produced by scripts/precompute_sgg_features.py) and builds the candidate edge
graph at __getitem__ time so graph construction runs in parallel DataLoader workers.

HDF5 layout (per image group):
    /{image_id}/roi_features  (N, C, H, W) float32, lzf compressed
    /{image_id}/boxes         (N, 4)        float32, xyxy
    /{image_id}/labels        (N,)          int64, 1-indexed
    /{image_id}/scores        (N,)          float32  [predicted features only]
    /{image_id}/relations     (R, 3)        int64, [sub_idx, obj_idx, predicate]
                                            [GT features only; predicate is 0-indexed]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.modules.sgg_heads.utils.graph import build_edge_index, compute_geometric_encoding


class SGGPrecomputedDataset(Dataset[dict[str, Tensor]]):
    """Dataset that loads precomputed SGG detector features from HDF5.

    Builds the candidate edge graph (edge index + geometric encoding + relation
    labels) at __getitem__ time, enabling parallel construction across DataLoader
    workers.  All N*(N-1) directed pairs are used as candidate edges, matching
    the reference implementation (REQUIRE_BOX_OVERLAP=false).

    The h5py file handle is opened lazily per-worker to avoid issues with
    h5py not being fork-safe across multiple worker processes.

    Args:
        h5_path: Path to the HDF5 feature file.
        score_thresh: Minimum detection score to include a predicted box.
            Applied at __getitem__ time so the threshold can be swept without
            re-running precomputation.  GT features (no scores in HDF5) are
            never filtered.  Default: 0.0 (keep all boxes).

    Returns (from __getitem__):
        roi_features: (N, C, H, W) float32
        boxes:        (N, 4)       float32, xyxy
        labels:       (N,)         int64, 1-indexed
        scores:       (N,)         float32 (zeros if not present in file)
        sub_idx:      (E,)         int64, image-local subject indices
        obj_idx:      (E,)         int64, image-local object indices
        geo:          (E, 12)      float32, geometric encoding
        rel_labels:   (E,)         int64, 0=no relation, 1-indexed predicate otherwise

    Properties:
        node_counts: list[int] — number of objects per image, cached at init
            from HDF5 shape (unfiltered).  Used by MaxObjectsBatchSampler; when
            score_thresh > 0 the sampler is conservative (overestimates nodes)
            but correct.
    """

    def __init__(
        self,
        h5_path: str | Path,
        score_thresh: float = 0.0,
        num_pos: int | None = None,
        num_neg: int | None = None,
    ) -> None:
        super().__init__()
        self._h5_path = Path(h5_path)
        self._score_thresh = score_thresh
        self._num_pos = num_pos
        self._num_neg = num_neg

        # Lazy file handle: opened once per worker in __getitem__
        self._file: h5py.File | None = None
        self._worker_id: int | None = None

        # Scan image IDs and node counts once at init (shape access is O(1) in h5py)
        with h5py.File(self._h5_path, "r") as f:
            self.image_ids: list[str] = list(f.keys())
            self._node_counts: list[int] = [
                int(f[img_id]["roi_features"].shape[0])
                for img_id in self.image_ids
            ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_counts(self) -> list[int]:
        """Number of detected objects per image (cached at init, unfiltered)."""
        return self._node_counts

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        grp = self._get_group(idx)

        # -- Load stored tensors ------------------------------------------
        roi_features = torch.from_numpy(grp["roi_features"][:])  # (N, C, H, W)
        boxes = torch.from_numpy(grp["boxes"][:])  # (N, 4)
        labels = torch.from_numpy(grp["labels"][:])  # (N,)

        has_pred_scores = "scores" in grp
        if has_pred_scores:
            scores = torch.from_numpy(grp["scores"][:])  # (N,)
        else:
            scores = torch.zeros(boxes.shape[0], dtype=torch.float32)

        relations: Tensor
        if "relations" in grp and grp["relations"].shape[0] > 0:
            relations = torch.from_numpy(grp["relations"][:])  # (R, 3)
        else:
            relations = torch.zeros(0, 3, dtype=torch.int64)

        # -- Score filtering (pred features only) -------------------------
        # GT features have no scores dataset; never filter them.
        if self._score_thresh > 0.0 and has_pred_scores:
            keep_mask = scores >= self._score_thresh
            if keep_mask.any():
                keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
                # Build old→new index map for relation re-indexing
                old_to_new = torch.full((boxes.shape[0],), -1, dtype=torch.long)
                old_to_new[keep_idx] = torch.arange(keep_idx.shape[0], dtype=torch.long)

                roi_features = roi_features[keep_mask]
                boxes = boxes[keep_mask]
                labels = labels[keep_mask]
                scores = scores[keep_mask]

                # Re-index relations: drop any where sub or obj was filtered out
                if relations.shape[0] > 0:
                    sub_new = old_to_new[relations[:, 0]]
                    obj_new = old_to_new[relations[:, 1]]
                    valid = (sub_new >= 0) & (obj_new >= 0)
                    relations = torch.stack(
                        [sub_new[valid], obj_new[valid], relations[valid, 2]], dim=1
                    )
            else:
                # All boxes filtered out — treat as empty image
                roi_features = roi_features[:0]
                boxes = boxes[:0]
                labels = labels[:0]
                scores = scores[:0]
                relations = relations[:0]

        N = boxes.shape[0]

        # -- Handle empty image (N == 0) ----------------------------------
        if N == 0:
            empty_e = torch.zeros(0, dtype=torch.int64)
            return {
                "roi_features": roi_features,
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "sub_idx": empty_e,
                "obj_idx": empty_e,
                "geo": torch.zeros(0, 12, dtype=torch.float32),
                "rel_labels": empty_e,
            }

        # -- Build edge index (all N*(N-1) directed pairs) ----------------
        sub_idx, obj_idx = build_edge_index(boxes)

        # -- Relation labels ----------------------------------------------
        # relations: (R, 3) — [sub, obj, predicate (0-indexed)]
        # Target: 0 = no relation, 1..num_pred = predicate class (1-indexed)
        rel_map: dict[tuple[int, int], int] = {}
        for r in range(relations.shape[0]):
            s, o, p = int(relations[r, 0]), int(relations[r, 1]), int(relations[r, 2])
            rel_map[(s, o)] = p + 1  # convert to 1-indexed

        E = sub_idx.shape[0]
        rel_labels = torch.zeros(E, dtype=torch.int64)
        for k in range(E):
            key = (int(sub_idx[k].item()), int(obj_idx[k].item()))
            if key in rel_map:
                rel_labels[k] = rel_map[key]

        # -- Positive/negative edge sampling ------------------------------
        if self._num_pos is not None or self._num_neg is not None:
            pos_idx = (rel_labels > 0).nonzero(as_tuple=False).view(-1)
            neg_idx = (rel_labels == 0).nonzero(as_tuple=False).view(-1)

            if self._num_pos is not None and pos_idx.shape[0] > self._num_pos:
                perm = torch.randperm(pos_idx.shape[0])
                pos_idx = pos_idx[perm[: self._num_pos]]
            if self._num_neg is not None and neg_idx.shape[0] > self._num_neg:
                perm = torch.randperm(neg_idx.shape[0])
                neg_idx = neg_idx[perm[: self._num_neg]]

            keep, _ = torch.cat([pos_idx, neg_idx]).sort()
            sub_idx = sub_idx[keep]
            obj_idx = obj_idx[keep]
            rel_labels = rel_labels[keep]

        # -- Geometric encoding (computed after sampling for efficiency) ---
        if sub_idx.shape[0] > 0:
            W = float(boxes[:, 2].max().item())
            H = float(boxes[:, 3].max().item())
            geo = compute_geometric_encoding(
                boxes[sub_idx], boxes[obj_idx], (W, H)
            )  # (E, 12)
        else:
            geo = torch.zeros(0, 12, dtype=torch.float32)

        return {
            "roi_features": roi_features,
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "sub_idx": sub_idx,
            "obj_idx": obj_idx,
            "geo": geo,
            "rel_labels": rel_labels,
        }

    # ------------------------------------------------------------------
    # h5py file handle (lazy, per-worker)
    # ------------------------------------------------------------------

    def _get_group(self, idx: int) -> Any:
        """Return the h5py group for image at idx, opening the file if needed."""
        worker_info = torch.utils.data.get_worker_info()
        current_worker = worker_info.id if worker_info is not None else -1

        if self._file is None or self._worker_id != current_worker:
            if self._file is not None:
                self._file.close()
            self._file = h5py.File(self._h5_path, "r")
            self._worker_id = current_worker

        return self._file[self.image_ids[idx]]
