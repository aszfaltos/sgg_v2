"""SGG evaluation metrics: Recall@K and Mean Recall@K.

Implements the standard SGG evaluation protocol for Predicate Classification
(PredCls) mode: given candidate edges with scores, rank all (subject, object,
predicate) triplets by confidence and compute recall against ground truth
at cutoffs K = [20, 50, 100].

R@K  — fraction of GT relations recalled in the top-K predictions, averaged
       over images with at least one GT relation.

mR@K — same recall computation but averaged per predicate class first, then
       averaged across classes. Better reflects long-tail performance.

Reference targets:
    VRD: R@50 ≈ 19.90%, R@100 ≈ 23.58%
    VG:  R@50 ≈ 24.63%, R@100 ≈ 29.09%, mR@100 ≈ 6.02%
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.modules.sgg_heads.base import SGGHeadOutput


class SGGEvaluator:
    """Accumulates SGG predictions and computes R@K / mR@K metrics.

    Designed to be used in the same pattern as DetectionEvaluator:
    call update() in each validation step, then compute() at epoch end,
    then reset() to clear state.

    Args:
        num_predicates: Number of predicate classes (e.g. 70 for VRD).
        k_values: Top-K cutoffs to evaluate (default: [20, 50, 100]).

    Example:
        >>> evaluator = SGGEvaluator(num_predicates=70)
        >>> for batch in val_loader:
        ...     output = model(batch, embedding)
        ...     evaluator.update(output, batch)
        >>> metrics = evaluator.compute()
        >>> metrics["R@100"]  # e.g. 0.2358
    """

    def __init__(
        self,
        num_predicates: int,
        k_values: list[int] | None = None,
    ) -> None:
        self._num_predicates = num_predicates
        self._k_values = k_values if k_values is not None else [20, 50, 100]

        # Per-image recall values: {K: [recall_img0, recall_img1, ...]}
        self._recall_per_image: dict[int, list[float]] = {k: [] for k in self._k_values}

        # Per-predicate hit tracking for mR@K: {K: {pred_class: [hit, hit, ...]}}
        self._per_pred_hits: dict[int, dict[int, list[int]]] = {
            k: {p: [] for p in range(1, num_predicates + 1)}
            for k in self._k_values
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, output: SGGHeadOutput, batch: dict[str, Tensor]) -> None:
        """Accumulate predictions from one batch.

        Args:
            output: SGGHeadOutput from model forward pass (per-image lists).
            batch: Collated batch dict from sgg_collate. Must contain
                rel_labels (total_E,) and edge_counts (B,).
        """
        edge_counts: list[int] = batch["edge_counts"].tolist()
        rel_labels_per_image = batch["rel_labels"].split(edge_counts)

        for logits, sub_idx, obj_idx, rel_labels in zip(
            output.rel_logits,
            output.subject_indices,
            output.object_indices,
            rel_labels_per_image,
        ):
            self._update_image(logits, sub_idx, obj_idx, rel_labels)

    def compute(self) -> dict[str, float]:
        """Compute R@K and mR@K from accumulated predictions.

        Returns:
            Dict with keys "R@K" and "mR@K" for each K in k_values.
            Empty dict if no predictions were accumulated.
        """
        metrics: dict[str, float] = {}

        for k in self._k_values:
            recalls = self._recall_per_image[k]
            if recalls:
                metrics[f"R@{k}"] = sum(recalls) / len(recalls)

            # mR@K: average per-predicate recall over predicates that appear in GT
            per_pred_recalls: list[float] = []
            for p in range(1, self._num_predicates + 1):
                hits = self._per_pred_hits[k][p]
                if hits:
                    per_pred_recalls.append(sum(hits) / len(hits))
            if per_pred_recalls:
                metrics[f"mR@{k}"] = sum(per_pred_recalls) / len(per_pred_recalls)

        return metrics

    def reset(self) -> None:
        """Clear accumulated state."""
        self._recall_per_image = {k: [] for k in self._k_values}
        self._per_pred_hits = {
            k: {p: [] for p in range(1, self._num_predicates + 1)}
            for k in self._k_values
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_image(
        self,
        logits: Tensor,     # (E_i, num_predicates)
        sub_idx: Tensor,    # (E_i,) image-local
        obj_idx: Tensor,    # (E_i,) image-local
        rel_labels: Tensor, # (E_i,) 0=no_rel, 1..num_pred = predicate (1-indexed)
    ) -> None:
        """Process one image's predictions and accumulate recall."""
        # Build ground truth triplet set
        gt_mask = rel_labels > 0
        if not gt_mask.any():
            return  # No GT relations — skip (doesn't count toward average)

        gt_subs = sub_idx[gt_mask].tolist()
        gt_objs = obj_idx[gt_mask].tolist()
        gt_preds = rel_labels[gt_mask].tolist()
        gt_triplets: set[tuple[int, int, int]] = {
            (int(s), int(o), int(p)) for s, o, p in zip(gt_subs, gt_objs, gt_preds)
        }
        n_gt = len(gt_triplets)

        if logits.shape[0] == 0:
            # No candidate edges — zero recall
            for k in self._k_values:
                self._recall_per_image[k].append(0.0)
                for s, o, p in gt_triplets:
                    self._per_pred_hits[k][p].append(0)
            return

        E_i = logits.shape[0]
        n_pred = self._num_predicates

        # Expand edge indices across all predicate classes in one shot
        # (E_i * n_pred,) tensors — no Python loop
        sub_expanded = sub_idx.repeat_interleave(n_pred)  # (E_i * n_pred,)
        obj_expanded = obj_idx.repeat_interleave(n_pred)
        # predicate labels 1..n_pred, repeated for each edge
        pred_classes = torch.arange(1, n_pred + 1, device=logits.device).repeat(E_i)

        # Flatten scores: (E_i, n_pred) → (E_i * n_pred,)
        scores = logits.flatten()

        for k in self._k_values:
            actual_k = min(k, scores.shape[0])
            if actual_k < scores.shape[0]:
                _, topk_indices = torch.topk(scores, actual_k)
            else:
                topk_indices = torch.arange(scores.shape[0], device=scores.device)

            # Build predicted triplet set from top-K indices
            pred_triplets: set[tuple[int, int, int]] = {
                (int(sub_expanded[i]), int(obj_expanded[i]), int(pred_classes[i]))
                for i in topk_indices.tolist()
            }

            # Per-image recall
            hits = len(gt_triplets & pred_triplets)
            self._recall_per_image[k].append(hits / n_gt)

            # Per-predicate hits
            for s, o, p in gt_triplets:
                hit = 1 if (s, o, p) in pred_triplets else 0
                self._per_pred_hits[k][int(p)].append(hit)
