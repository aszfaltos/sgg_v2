"""DataModule and batching utilities for SGG training on precomputed features.

Design (from batching.md):
- Graph is built at dataset __getitem__ time (parallelised across DataLoader workers).
- MaxObjectsBatchSampler packs images greedily by total object count so batches never
  exceed a node budget and images are never split across batches.
- sgg_collate applies the global flat index: per-image edge indices are shifted by
  per-image node offsets, then all tensors are concatenated into one big disconnected
  graph. No padding needed.
- The resulting batch tensors have shapes (total_N, ...) and (total_E, ...) so all
  MLP calls in the SGG head run in one shot over the whole batch.
"""

from __future__ import annotations

from itertools import accumulate
from pathlib import Path
from typing import Any, Literal

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset, random_split

from src.data.sgg_precomputed import SGGPrecomputedDataset


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------


class MaxObjectsBatchSampler(Sampler[list[int]]):
    """Greedy bin-packing sampler that groups images by total object count.

    Images are packed into batches such that the total number of objects
    (nodes) per batch stays within `max_objects`. Images are never split
    across batches. A single image with more nodes than `max_objects` is
    yielded alone.

    Args:
        node_counts: Number of objects per dataset item.
        max_objects: Maximum total nodes per batch.
        shuffle: If True, randomise image order each epoch.
        seed: Base random seed (incremented each epoch for variety).
    """

    def __init__(
        self,
        node_counts: list[int],
        max_objects: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self._node_counts = node_counts
        self._max_objects = max_objects
        self._shuffle = shuffle
        self._seed = seed
        self._epoch: int = 0

        # Pre-compute approximate length (using unshuffled order)
        self._len = self._count_batches(list(range(len(node_counts))))

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling (call from LightningModule)."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self):  # type: ignore[override]
        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            indices = torch.randperm(len(self._node_counts), generator=g).tolist()
        else:
            indices = list(range(len(self._node_counts)))

        batch: list[int] = []
        batch_nodes = 0

        for i in indices:
            n = self._node_counts[i]
            if batch and batch_nodes + n > self._max_objects:
                yield batch
                batch = []
                batch_nodes = 0
            batch.append(i)
            batch_nodes += n

        if batch:
            yield batch

    # ------------------------------------------------------------------

    def _count_batches(self, indices: list[int]) -> int:
        count = 0
        batch_nodes = 0
        started = False
        for i in indices:
            n = self._node_counts[i]
            if started and batch_nodes + n > self._max_objects:
                count += 1
                batch_nodes = 0
            batch_nodes += n
            started = True
        if started:
            count += 1
        return count


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def sgg_collate(items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate SGG dataset items into a single batched graph.

    Applies the global flat index design: edge indices from each image are
    shifted by that image's node offset so the entire batch forms one big
    disconnected graph. No padding is used.

    Args:
        items: List of dicts from SGGPrecomputedDataset.__getitem__.

    Returns:
        Batched dict with keys:
            roi_features:  (total_N, C, H, W)
            boxes:         (total_N, 4)
            labels:        (total_N,)
            scores:        (total_N,)
            sub_idx:       (total_E,)  — global, offset-shifted
            obj_idx:       (total_E,)  — global, offset-shifted
            geo:           (total_E, 12)
            rel_labels:    (total_E,)
            node_counts:   (B,) int64
            image_offsets: (B,) int64
    """
    node_counts = [item["roi_features"].shape[0] for item in items]
    offsets = [0] + list(accumulate(node_counts[:-1]))

    sub_idx = torch.cat(
        [item["sub_idx"] + off for item, off in zip(items, offsets)]
    )
    obj_idx = torch.cat(
        [item["obj_idx"] + off for item, off in zip(items, offsets)]
    )

    edge_counts = [item["sub_idx"].shape[0] for item in items]

    batch = {
        "roi_features": torch.cat([item["roi_features"] for item in items]),
        "boxes": torch.cat([item["boxes"] for item in items]),
        "labels": torch.cat([item["labels"] for item in items]),
        "scores": torch.cat([item["scores"] for item in items]),
        "sub_idx": sub_idx,
        "obj_idx": obj_idx,
        "geo": torch.cat([item["geo"] for item in items]),
        "rel_labels": torch.cat([item["rel_labels"] for item in items]),
        "node_counts": torch.tensor(node_counts, dtype=torch.int64),
        "edge_counts": torch.tensor(edge_counts, dtype=torch.int64),
        "image_offsets": torch.tensor(offsets, dtype=torch.int64),
    }

    # Union features (BGNN only — present when HDF5 was built with --save-union-features)
    if "union_features" in items[0]:
        batch["union_features"] = torch.cat(
            [item["union_features"] for item in items]
        )  # (total_E, C)

    return batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subset_node_counts(subset: Subset) -> list[int]:  # type: ignore[type-arg]
    """Extract node_counts for a Subset in the correct index order."""
    full: SGGPrecomputedDataset = subset.dataset  # type: ignore[assignment]
    return [full.node_counts[i] for i in subset.indices]


def _get_node_counts(ds: Any) -> list[int]:
    """Extract node_counts from a Subset[SGGPrecomputedDataset] or ConcatDataset thereof."""
    if isinstance(ds, Subset) and isinstance(ds.dataset, SGGPrecomputedDataset):
        return _subset_node_counts(ds)
    if isinstance(ds, ConcatDataset):
        counts: list[int] = []
        for sub in ds.datasets:
            counts.extend(_get_node_counts(sub))
        return counts
    raise TypeError(f"Cannot extract node_counts from {type(ds)}")


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class SGGPrecomputedDataModule(LightningDataModule):
    """Lightning DataModule for SGG training on precomputed features.

    Training uses GT box features exclusively (cleanest signal for the relation
    head). Testing runs two loaders in sequence: GT boxes (oracle upper bound)
    and, optionally, predicted boxes (end-to-end performance).

    Training uses one of three sources controlled by ``train_source``:

    * ``"gt"``     — GT boxes only. Val uses GT. Measures relation head quality in
                     isolation (PredCls upper bound).
    * ``"pred"``   — Predicted boxes only. Val uses pred. Measures end-to-end SGDet
                     performance.
    * ``"hybrid"`` — Concatenates GT and pred training sets. Val uses pred, so
                     metrics reflect whether hybrid training improves SGDet.

    In all modes the val split indices are derived from the GT dataset (same image
    ordering as pred), and the test loaders are unchanged (GT + optional pred).

    Args:
        gt_train_h5: Path to the GT training HDF5 feature file. Always required;
            also determines the canonical train/val split indices.
        pred_train_h5: Path to the predicted-box training HDF5 file. Required when
            ``train_source`` is ``"pred"`` or ``"hybrid"``.
        train_source: Which box set(s) to train on (default: ``"gt"``).
        gt_test_h5: Path to the GT test HDF5 feature file (oracle evaluation).
        pred_test_h5: Path to the predicted-box test HDF5 file (end-to-end
            evaluation). If None, only GT test is run.
        max_objects: Maximum total objects (nodes) per batch.
        val_split: Fraction of training images to hold out for validation.
        num_workers: DataLoader worker count.
        score_thresh: Minimum detection score passed to SGGPrecomputedDataset.
            GT features (no scores in HDF5) are unaffected.
        num_pos_per_image: Max positive (relation) edges sampled per training image.
            0 = no limit (use all positives). Default: 250 (matches reference).
        num_neg_per_image: Max negative (no-relation) edges sampled per training image.
            0 = no limit. Default: 750 (matches reference 1000 pairs at 25% pos rate).
            Val/test datasets always use all edges regardless of these settings.
        seed: Random seed for train/val split and sampler shuffling.

    Example:
        >>> dm = SGGPrecomputedDataModule(
        ...     gt_train_h5="features/gt/train.h5",
        ...     gt_test_h5="features/gt/test.h5",
        ...     pred_test_h5="features/pred/test.h5",
        ...     max_objects=512,
        ... )
        >>> dm.setup("fit")
        >>> batch = next(iter(dm.train_dataloader()))
        >>> batch["roi_features"].shape  # (total_N, C, H, W)
    """

    def __init__(
        self,
        gt_train_h5: str | Path,
        gt_test_h5: str | Path,
        pred_train_h5: str | Path | None = None,
        pred_test_h5: str | Path | None = None,
        train_source: Literal["gt", "pred", "hybrid"] = "gt",
        max_objects: int = 512,
        val_split: float = 0.1,
        num_workers: int = 4,
        score_thresh: float = 0.0,
        num_pos_per_image: int = 250,
        num_neg_per_image: int = 750,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._gt_train_h5 = Path(gt_train_h5)
        self._pred_train_h5 = Path(pred_train_h5) if pred_train_h5 is not None else None
        self._train_source = train_source
        self._gt_test_h5 = Path(gt_test_h5)
        self._pred_test_h5 = Path(pred_test_h5) if pred_test_h5 is not None else None
        self._max_objects = max_objects
        self._val_split = val_split
        self._num_workers = num_workers
        self._score_thresh = score_thresh
        self._num_pos = num_pos_per_image if num_pos_per_image > 0 else None
        self._num_neg = num_neg_per_image if num_neg_per_image > 0 else None
        self._seed = seed

        self._train_ds: Subset | ConcatDataset | None = None  # type: ignore[type-arg]
        self._val_subset: Subset | None = None  # type: ignore[type-arg]
        self._gt_test_ds: SGGPrecomputedDataset | None = None
        self._pred_test_ds: SGGPrecomputedDataset | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            # gt_ds has no sampling — used for canonical split indices and GT val.
            gt_ds = SGGPrecomputedDataset(self._gt_train_h5, self._score_thresh)
            total = len(gt_ds)
            val_size = int(total * self._val_split)
            train_size = total - val_size

            # Canonical split indices always come from GT dataset.
            # Same indices are reused for pred dataset (same image ordering).
            generator = torch.Generator().manual_seed(self._seed)
            gt_train_sub, gt_val_sub = random_split(
                gt_ds, [train_size, val_size], generator=generator
            )
            train_indices = list(gt_train_sub.indices)
            val_indices = list(gt_val_sub.indices)

            if self._train_source == "gt":
                gt_ds_train = SGGPrecomputedDataset(
                    self._gt_train_h5, self._score_thresh, self._num_pos, self._num_neg
                )
                self._train_ds = Subset(gt_ds_train, train_indices)
                self._val_subset = Subset(gt_ds, val_indices)  # no sampling for val
            else:
                assert self._pred_train_h5 is not None
                if self._train_source == "pred":
                    pred_ds_train = SGGPrecomputedDataset(
                        self._pred_train_h5, self._score_thresh, self._num_pos, self._num_neg
                    )
                    pred_ds_val = SGGPrecomputedDataset(self._pred_train_h5, self._score_thresh)
                    self._train_ds = Subset(pred_ds_train, train_indices)
                    self._val_subset = Subset(pred_ds_val, val_indices)
                else:  # hybrid
                    gt_ds_train = SGGPrecomputedDataset(
                        self._gt_train_h5, self._score_thresh, self._num_pos, self._num_neg
                    )
                    pred_ds_train = SGGPrecomputedDataset(
                        self._pred_train_h5, self._score_thresh, self._num_pos, self._num_neg
                    )
                    pred_ds_val = SGGPrecomputedDataset(self._pred_train_h5, self._score_thresh)
                    self._train_ds = ConcatDataset([
                        Subset(gt_ds_train, train_indices),
                        Subset(pred_ds_train, train_indices),
                    ])
                    self._val_subset = Subset(pred_ds_val, val_indices)

        if stage in ("test", "predict", None):
            self._gt_test_ds = SGGPrecomputedDataset(
                self._gt_test_h5, self._score_thresh
            )
            if self._pred_test_h5 is not None:
                self._pred_test_ds = SGGPrecomputedDataset(
                    self._pred_test_h5, self._score_thresh
                )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader[Any]:
        if self._train_ds is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")

        sampler = MaxObjectsBatchSampler(
            _get_node_counts(self._train_ds),
            self._max_objects,
            shuffle=True,
            seed=self._seed,
        )
        pin_memory = torch.cuda.is_available()
        persistent = self._num_workers > 0
        return DataLoader(
            self._train_ds,
            batch_sampler=sampler,
            collate_fn=sgg_collate,
            num_workers=self._num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self._val_subset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader()")

        sampler = MaxObjectsBatchSampler(
            _subset_node_counts(self._val_subset),
            self._max_objects,
            shuffle=False,
            seed=self._seed,
        )
        pin_memory = torch.cuda.is_available()
        persistent = self._num_workers > 0
        return DataLoader(
            self._val_subset,
            batch_sampler=sampler,
            collate_fn=sgg_collate,
            num_workers=self._num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )

    def test_dataloader(self) -> list[DataLoader[Any]]:
        """Return [gt_loader] or [gt_loader, pred_loader].

        dataloader_idx 0 = GT boxes (oracle), 1 = predicted boxes (end-to-end).
        """
        if self._gt_test_ds is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")

        pin_memory = torch.cuda.is_available()
        persistent = self._num_workers > 0

        def _make_loader(ds: SGGPrecomputedDataset) -> DataLoader[Any]:
            sampler = MaxObjectsBatchSampler(
                ds.node_counts, self._max_objects, shuffle=False, seed=self._seed
            )
            return DataLoader(
                ds,
                batch_sampler=sampler,
                collate_fn=sgg_collate,
                num_workers=self._num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
            )

        loaders: list[DataLoader[Any]] = [_make_loader(self._gt_test_ds)]
        if self._pred_test_ds is not None:
            loaders.append(_make_loader(self._pred_test_ds))
        return loaders
