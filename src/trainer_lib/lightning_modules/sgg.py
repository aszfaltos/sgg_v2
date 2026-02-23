"""SGG Lightning Module for training scene graph generation heads.

Wraps any SGGHead subclass (NMPHead, future IMP/BGNN heads) in a PyTorch
Lightning module with:
- Cross-entropy loss on positive edges (ignoring no-relation edges)
- Differential learning rates: head at `learning_rate`, embeddings at `embedding_lr`
- Warmup + cosine LR schedule (matching DetectorLightningModule)
- R@K / mR@K evaluation via SGGEvaluator at validation epoch end

The word embedding table is stored as nn.Parameter in this module so it can
be trained with a separate (much lower) LR and moves to the correct device
automatically. It is passed into head.forward() each step.

Example:
    >>> from src.modules.sgg_heads.nmp import NMPHead
    >>> head = NMPHead(roi_feature_dim=(256, 7, 7), num_predicates=70)
    >>> embedding = torch.randn(101, 300)  # GloVe, 101 classes (bg + 100 VRD)
    >>> module = SGGLightningModule(model=head, embedding=embedding, num_predicates=70)
    >>> trainer = Trainer(max_epochs=20)
    >>> trainer.fit(module, datamodule=datamodule)
"""

from __future__ import annotations

from typing import Any, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.evaluation.sgg_metrics import SGGEvaluator
from src.modules.sgg_heads.base import SGGHead


class SGGLightningModule(LightningModule):
    """Lightning module for training SGG relation heads.

    Handles training with cross-entropy loss, validation with R@K / mR@K
    metrics, and optimizer/scheduler configuration with differential LRs
    for the graph head vs word embeddings.

    Args:
        model: SGG relation head (NMPHead or any SGGHead subclass).
        embedding: (num_classes, semantic_dim) word embedding matrix. Stored
            as nn.Parameter so it trains with `embedding_lr` and moves to
            the correct device automatically.
        num_predicates: Number of predicate classes (70 for VRD, varies for VG).
        learning_rate: Base LR for the graph head parameters. Default: 0.05.
        embedding_lr: LR for the word embedding table. Default: 1e-5 (much
            lower — embeddings are already pretrained).
        weight_decay: L2 regularization. Default: 1e-4.
        warmup_epochs: Epochs for linear warmup (1% → 100% of base LR).
        eval_k: Top-K cutoffs for R@K and mR@K metrics.

    Note:
        Loss is computed only on positive edges (edges with a GT relation).
        No-relation edges (rel_label == 0) are ignored. This matches the
        NMPHead output shape (num_predicates classes, no background logit).

        For test evaluation, set `test_prefix` before each `trainer.test()`
        call to control the metric namespace ("test_gt" or "test_pred").
        This allows two clean, independent test passes instead of using
        Lightning's multi-dataloader test (which has display artifacts).
    """

    def __init__(
        self,
        model: SGGHead,
        embedding: Tensor,
        num_predicates: int,
        learning_rate: float = 0.05,
        embedding_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        eval_k: list[int] | None = None,
    ) -> None:
        super().__init__()

        # Don't serialize model or embedding in hparams (too large)
        self.save_hyperparameters(ignore=["model", "embedding"])

        self.model: SGGHead = model
        # Store embedding as trainable parameter — moves with .to(device)
        self.embedding = nn.Parameter(embedding)

        self.num_predicates = num_predicates
        self.learning_rate = learning_rate
        self.embedding_lr = embedding_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self._eval_k = eval_k if eval_k is not None else [20, 50, 100]

        self._evaluator = SGGEvaluator(
            num_predicates=num_predicates,
            k_values=self._eval_k,
        )

        # Single test evaluator — prefix is set before each trainer.test() call
        self._test_evaluator = SGGEvaluator(
            num_predicates=num_predicates, k_values=self._eval_k
        )
        # Metric namespace for the current test pass; set by the caller
        self.test_prefix: str = "test_gt"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        """Forward pass, loss computation, and logging.

        Args:
            batch: Collated batch from sgg_collate.
            batch_idx: Index of batch in epoch.

        Returns:
            Scalar loss tensor.
        """
        output = self.model(batch, self.embedding)

        # Concatenate logits from all images: (total_E, num_predicates)
        all_logits = torch.cat(output.rel_logits)
        # Shift labels: 0=no_rel → -1 (ignore_index), 1..num_pred → 0-indexed
        all_labels = batch["rel_labels"] - 1  # (total_E,), -1=no relation

        # Skip batches with no positive edges to avoid NaN (cross_entropy
        # returns 0/0=NaN when every label is the ignore_index).
        if not (all_labels >= 0).any():
            return torch.tensor(0.0, device=all_logits.device, requires_grad=True)

        loss = F.cross_entropy(all_logits, all_labels, ignore_index=-1)

        total_e = int(batch["rel_labels"].shape[0])
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, batch_size=total_e,
        )

        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> None:
        """Run inference and accumulate predictions for epoch-end metrics.

        Args:
            batch: Collated batch from sgg_collate.
            batch_idx: Index of batch in epoch.
        """
        with torch.no_grad():
            output = self.model(batch, self.embedding)

        self._evaluator.update(output, batch)

    def on_validation_epoch_end(self) -> None:
        """Compute R@K / mR@K from accumulated predictions and log."""
        metrics = self._evaluator.compute()

        prog_bar_keys = {f"R@{k}" for k in self._eval_k}

        for key, value in metrics.items():
            self.log(
                f"val/{key}",
                value,
                prog_bar=(key in prog_bar_keys),
                logger=True,
            )

        self._evaluator.reset()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        """Accumulate predictions from one test batch.

        The metric namespace is determined by `self.test_prefix` (set by
        the caller before each `trainer.test()` invocation).

        Args:
            batch: Collated batch from sgg_collate.
            batch_idx: Batch index within the dataloader.
        """
        with torch.no_grad():
            output = self.model(batch, self.embedding)

        self._test_evaluator.update(output, batch)

    def on_test_epoch_end(self) -> None:
        """Log R@K / mR@K under `self.test_prefix` and reset evaluator."""
        prog_bar_keys = {f"R@{k}" for k in self._eval_k}

        metrics = self._test_evaluator.compute()
        for key, value in metrics.items():
            self.log(
                f"{self.test_prefix}/{key}",
                value,
                prog_bar=(key in prog_bar_keys),
                logger=True,
            )
        self._test_evaluator.reset()

    # ------------------------------------------------------------------
    # Sampler epoch sync
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Advance the MaxObjectsBatchSampler epoch for per-epoch shuffling."""
        try:
            dl = self.trainer.train_dataloader
            if dl is None:
                return
            sampler = dl.batch_sampler  # type: ignore[union-attr]
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.current_epoch)
        except Exception:
            pass  # Silently skip if sampler doesn't support it

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> dict[str, Union[Optimizer, dict[str, Any]]]:
        """Configure AdamW with warmup + cosine schedule.

        Two parameter groups:
        - head: graph head parameters at `learning_rate`
        - embedding: word embedding table at `embedding_lr`

        Returns:
            Lightning optimizer+scheduler config dict.
        """
        head_params = list(self.model.parameters())
        embedding_params = [self.embedding]

        for pg_name, pg in [("head", head_params), ("embedding", embedding_params)]:
            n = sum(p.numel() for p in pg)
            lr = self.learning_rate if pg_name == "head" else self.embedding_lr
            print(f"  {pg_name}: {n:,} params, lr={lr:.2e}")

        optimizer = AdamW(
            [
                {"params": head_params, "lr": self.learning_rate, "name": "head"},
                {"params": embedding_params, "lr": self.embedding_lr, "name": "embedding"},
            ],
            weight_decay=self.weight_decay,
        )

        trainer = getattr(self, "_trainer", None)
        max_epochs = (
            getattr(trainer, "max_epochs", 100) if trainer is not None else 100
        )

        # Linear warmup: 1% → 100% over warmup_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )

        # Cosine decay: 100% → 0% over remaining epochs
        cosine_epochs = max(1, max_epochs - self.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=0)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
