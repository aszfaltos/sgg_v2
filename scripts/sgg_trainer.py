#!/usr/bin/env python3
"""Train SGG relation heads on precomputed detector features.

This script trains scene graph generation heads (NMP and future variants)
on precomputed HDF5 features from a frozen detector.

The training process:
1. Load precomputed features from HDF5 (produced by precompute_sgg_features.py)
2. Load word embedding matrix (.pt file)
3. Create SGG head and wrap in SGGLightningModule
4. Configure Lightning Trainer with callbacks and logging
5. Train on GT or predicted box features (--train-source), validate on same split
6. Run final test on GT boxes (oracle) and predicted boxes (end-to-end)

Usage:
    # Train on GT boxes (oracle upper bound), test on both
    uv run python scripts/sgg_trainer.py \\
        --head nmp \\
        --gt-train-h5 datasets/vrd/features/efficientdet_d0/gt_train.h5 \\
        --gt-test-h5 datasets/vrd/features/efficientdet_d0/gt_test.h5 \\
        --pred-test-h5 datasets/vrd/features/efficientdet_d0/pred_test.h5 \\
        --embeddings datasets/vrd/embeddings/word2vec-google-news-300_objects.pt \\
        --num-predicates 70

    # Train on predicted boxes (end-to-end training), val/test on pred
    uv run python scripts/sgg_trainer.py \\
        --head nmp \\
        --gt-train-h5 datasets/vrd/features/efficientdet_d0/gt_train.h5 \\
        --pred-train-h5 datasets/vrd/features/efficientdet_d0/pred_train.h5 \\
        --gt-test-h5 datasets/vrd/features/efficientdet_d0/gt_test.h5 \\
        --pred-test-h5 datasets/vrd/features/efficientdet_d0/pred_test.h5 \\
        --embeddings datasets/vrd/embeddings/word2vec-google-news-300_objects.pt \\
        --num-predicates 70 \\
        --train-source pred

    # Train on GT + pred boxes (hybrid), val/test on pred (end-to-end)
    uv run python scripts/sgg_trainer.py \\
        --head nmp \\
        --gt-train-h5 datasets/vrd/features/efficientdet_d0/gt_train.h5 \\
        --pred-train-h5 datasets/vrd/features/efficientdet_d0/pred_train.h5 \\
        --gt-test-h5 datasets/vrd/features/efficientdet_d0/gt_test.h5 \\
        --pred-test-h5 datasets/vrd/features/efficientdet_d0/pred_test.h5 \\
        --embeddings datasets/vrd/embeddings/word2vec-google-news-300_objects.pt \\
        --num-predicates 70 \\
        --train-source hybrid

    # GT-only evaluation (no predicted-box features available)
    uv run python scripts/sgg_trainer.py \\
        --head nmp \\
        --gt-train-h5 datasets/vrd/features/efficientdet_d0/gt_train.h5 \\
        --gt-test-h5 datasets/vrd/features/efficientdet_d0/gt_test.h5 \\
        --embeddings datasets/vrd/embeddings/word2vec-google-news-300_objects.pt \\
        --num-predicates 70

    # Resume training from checkpoint
    uv run python scripts/sgg_trainer.py \\
        --head nmp \\
        --gt-train-h5 datasets/vrd/features/efficientdet_d0/gt_train.h5 \\
        --gt-test-h5 datasets/vrd/features/efficientdet_d0/gt_test.h5 \\
        --pred-test-h5 datasets/vrd/features/efficientdet_d0/pred_test.h5 \\
        --embeddings datasets/vrd/embeddings/word2vec-google-news-300_objects.pt \\
        --num-predicates 70 \\
        --resume checkpoints/sgg/nmp_gt_train/last.ckpt
"""

import argparse
import sys
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)

# Add project root to path for direct script execution
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.modules.sgg_heads.bgnn import BGNNHead  # noqa: E402
from src.modules.sgg_heads.imp import IMPHead  # noqa: E402
from src.modules.sgg_heads.nmp import NMPHead  # noqa: E402
from src.trainer_lib import (  # noqa: E402
    SGGLightningModule,
    SGGPrecomputedDataModule,
    create_aim_logger,
)


class CleanProgressBar(TQDMProgressBar):
    """Progress bar without v_num clutter."""

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SGG relation heads on precomputed features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--head",
        type=str,
        required=True,
        choices=["nmp", "imp", "bgnn"],
        help="SGG head architecture",
    )
    parser.add_argument(
        "--gt-train-h5",
        type=str,
        required=True,
        help="Path to GT training HDF5 feature file",
    )
    parser.add_argument(
        "--pred-train-h5",
        type=str,
        default=None,
        help="Path to predicted-box training HDF5 file (required when --train-source is "
             "'pred' or 'hybrid')",
    )
    parser.add_argument(
        "--train-source",
        type=str,
        default="gt",
        choices=["gt", "pred", "hybrid"],
        help="Which box set to train on: 'gt' (oracle, GT boxes only, default), "
             "'pred' (predicted boxes only, val on pred), or 'hybrid' (GT+pred concat, "
             "val on pred). --pred-train-h5 required for 'pred' and 'hybrid'.",
    )
    parser.add_argument(
        "--gt-test-h5",
        type=str,
        required=True,
        help="Path to GT test HDF5 feature file (oracle evaluation)",
    )
    parser.add_argument(
        "--pred-test-h5",
        type=str,
        default=None,
        help="Path to predicted-box test HDF5 file (end-to-end evaluation). "
             "If omitted, only GT-box test metrics are reported.",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to .pt embedding tensor of shape (num_classes, semantic_dim)",
    )

    # Model arguments
    parser.add_argument(
        "--num-predicates",
        type=int,
        default=70,
        help="Number of predicate classes (default: 70 for VRD)",
    )
    parser.add_argument(
        "--d-hidden",
        type=int,
        default=512,
        help="Hidden dimension for head MLPs (default: 512)",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=3,
        help="Message passing iterations for IMP/BGNN heads (default: 3 for IMP, 2 recommended for BGNN; ignored for NMP)",
    )

    # Batching
    parser.add_argument(
        "--max-objects",
        type=int,
        default=512,
        help="Max total objects (nodes) per batch (default: 512)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for head parameters (default: 0.001)",
    )
    parser.add_argument(
        "--embedding-lr",
        type=float,
        default=1e-5,
        help="Learning rate for word embeddings (default: 1e-5)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1 = 10%%)",
    )

    # Dataset score filtering
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.0,
        help="Minimum detection score to include a predicted box at train/val/test time. "
             "GT features are unaffected. 0.0 = keep all (default: 0.0)",
    )

    # Relation sampling
    parser.add_argument(
        "--num-pos",
        type=int,
        default=250,
        help="Max positive (relation) edges sampled per training image. "
             "0 = no limit (use all positive edges). (default: 250)",
    )
    parser.add_argument(
        "--num-neg",
        type=int,
        default=750,
        help="Max negative (no-relation) edges sampled per training image. "
             "0 = no limit. (default: 750)",
    )

    # Infrastructure
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (auto-generated if not provided)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and run test evaluation only (requires --resume)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def generate_checkpoint_dir(head: str, train_h5: str, train_source: str) -> Path:
    """Auto-generate checkpoint directory from head name, feature stem, and train source.

    Example:
        >>> generate_checkpoint_dir("nmp", "features/efficientdet_d0/gt_train.h5", "gt")
        Path("checkpoints/sgg/nmp_gt_train_gt")
    """
    stem = Path(train_h5).stem
    return Path("checkpoints") / "sgg" / f"{head}_{stem}_{train_source}"


def create_head(
    head: str,
    roi_feature_dim: tuple[int, int, int],
    num_predicates: int,
    semantic_dim: int,
    d_hidden: int,
    num_iter: int = 3,
) -> NMPHead | IMPHead | BGNNHead:
    """Create SGG head from name and config.

    Args:
        head: Head architecture name ("nmp", "imp", or "bgnn").
        roi_feature_dim: (C, H, W) shape of ROI features in the HDF5 file.
        num_predicates: Number of predicate classes.
        semantic_dim: Dimension of word embeddings.
        d_hidden: Hidden MLP dimension.
        num_iter: Message passing iterations (IMP/BGNN only).

    Returns:
        Configured SGGHead instance.
    """
    if head == "nmp":
        return NMPHead(
            roi_feature_dim=roi_feature_dim,
            num_predicates=num_predicates,
            semantic_dim=semantic_dim,
            d_hidden=d_hidden,
        )
    if head == "imp":
        return IMPHead(
            roi_feature_dim=roi_feature_dim,
            num_predicates=num_predicates,
            semantic_dim=semantic_dim,
            d_hidden=d_hidden,
            num_iter=num_iter,
        )
    if head == "bgnn":
        return BGNNHead(
            roi_feature_dim=roi_feature_dim,
            num_predicates=num_predicates,
            semantic_dim=semantic_dim,
            d_hidden=d_hidden,
            num_iter=num_iter,
        )
    raise ValueError(f"Unknown head: {head}")


def infer_roi_feature_dim(train_h5: str) -> tuple[int, int, int]:
    """Read ROI feature shape from HDF5 to configure the head.

    Args:
        train_h5: Path to the training HDF5 file.

    Returns:
        (C, H, W) tuple from first image's roi_features dataset.
    """
    import h5py

    with h5py.File(train_h5, "r") as f:
        first_key = next(iter(f.keys()))
        shape = f[first_key]["roi_features"].shape  # (N, C, H, W)
        return (int(shape[1]), int(shape[2]), int(shape[3]))


def main(
    head: str,
    gt_train_h5: str,
    pred_train_h5: str | None,
    train_source: str,
    gt_test_h5: str,
    pred_test_h5: str | None,
    embeddings: str,
    num_predicates: int,
    d_hidden: int,
    num_iter: int,
    max_objects: int,
    epochs: int,
    lr: float,
    embedding_lr: float,
    val_split: float,
    score_thresh: float,
    num_pos: int,
    num_neg: int,
    num_workers: int,
    checkpoint_dir: str | None,
    resume: str | None,
    test_only: bool,
    seed: int,
) -> None:
    """Train an SGG relation head on precomputed features."""
    if test_only and resume is None:
        raise ValueError("--test-only requires --resume <checkpoint_path>")

    if train_source in ("pred", "hybrid") and pred_train_h5 is None:
        raise ValueError(
            f"--pred-train-h5 is required when --train-source is '{train_source}'"
        )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint_path = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else generate_checkpoint_dir(head, str(gt_train_h5), train_source)
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # --- Print configuration ---
    print("=" * 60)
    print("SGG Head Training Configuration")
    print("=" * 60)
    print(f"Head:            {head}")
    print(f"Train source:    {train_source}")
    print(f"GT train:        {gt_train_h5}")
    if train_source in ("pred", "hybrid"):
        print(f"Pred train:      {pred_train_h5}")
    print(f"GT test:         {gt_test_h5}")
    print(f"Pred test:       {pred_test_h5 or '(none)'}")
    print(f"Embeddings:      {embeddings}")
    print(f"Num predicates:  {num_predicates}")
    print(f"Hidden dim:      {d_hidden}")
    print(f"Max objects:     {max_objects}")
    print(f"Epochs:          {epochs}")
    print(f"LR (head):       {lr}")
    print(f"LR (embedding):  {embedding_lr}")
    print(f"Val split:       {val_split}")
    print(f"Sampling:        pos={num_pos or 'all'}, neg={num_neg or 'all'} edges/image")
    print(f"Checkpoint dir:  {checkpoint_path}")
    if resume:
        print(f"Resume from:     {resume}")
    print(f"Seed:            {seed}")
    print("=" * 60)
    print()

    # --- Load embedding matrix ---
    print(f"Loading embeddings from {embeddings}...")
    embedding_tensor: torch.Tensor = torch.load(embeddings, weights_only=True)
    semantic_dim = embedding_tensor.shape[1]
    num_classes = embedding_tensor.shape[0]
    print(f"Embedding shape: {tuple(embedding_tensor.shape)}  ({num_classes} classes, {semantic_dim}-d)")
    print()

    # --- Infer ROI feature dimensions ---
    # Always read from GT: feature shape is a detector property, same for GT and pred.
    print(f"Reading ROI feature shape from {gt_train_h5}...")
    roi_feature_dim = infer_roi_feature_dim(gt_train_h5)
    print(f"ROI feature dim: {roi_feature_dim}  (C={roi_feature_dim[0]}, H={roi_feature_dim[1]}, W={roi_feature_dim[2]})")
    print()

    # --- Create head ---
    print(f"Creating {head} head...")
    sgg_head = create_head(
        head=head,
        roi_feature_dim=roi_feature_dim,
        num_predicates=num_predicates,
        semantic_dim=semantic_dim,
        d_hidden=d_hidden,
        num_iter=num_iter,
    )
    n_params = sum(p.numel() for p in sgg_head.parameters())
    print(f"Head parameters: {n_params:,}")
    print()

    # --- Create DataModule ---
    print("Creating DataModule...")
    datamodule = SGGPrecomputedDataModule(
        gt_train_h5=gt_train_h5,
        gt_test_h5=gt_test_h5,
        pred_train_h5=pred_train_h5,
        pred_test_h5=pred_test_h5,
        train_source=train_source,
        max_objects=max_objects,
        val_split=val_split,
        num_workers=num_workers,
        score_thresh=score_thresh,
        num_pos_per_image=num_pos,
        num_neg_per_image=num_neg,
        seed=seed,
    )
    print("DataModule created")
    print()

    # --- Create Lightning Module ---
    print("Creating Lightning module...")
    lightning_module = SGGLightningModule(
        model=sgg_head,
        embedding=embedding_tensor,
        num_predicates=num_predicates,
        learning_rate=lr,
        embedding_lr=embedding_lr,
        weight_decay=1e-4,
        warmup_epochs=5,
        eval_k=[20, 50, 100],
    )
    print("Lightning module created")
    print()

    # --- Logger ---
    train_stem = Path(gt_train_h5).stem
    experiment_name = f"{head}_sgg"
    run_name = f"{head}_{train_stem}_{train_source}_lr{lr}"
    logger = create_aim_logger(
        experiment_name=experiment_name,
        run_name=run_name,
        repo_path=".aim",
    )

    # --- Callbacks ---
    callbacks = [
        CleanProgressBar(),
        ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="best",
            monitor="val/R@100",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val/R@100",
            patience=10,
            mode="max",
            verbose=True,
        ),
    ]

    # --- Trainer ---
    print("Creating Lightning Trainer...")
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_model_summary=True,
    )
    print("Trainer created")
    print()

    # --- Train ---
    if not test_only:
        print("Starting training...")
        print("=" * 60)
        trainer.fit(
            lightning_module,
            datamodule=datamodule,
            ckpt_path=resume,
        )
        print("=" * 60)
        print()

        print("Training complete!")
        print(f"Best checkpoint: {checkpoint_path / 'best.ckpt'}")
        print(f"Last checkpoint: {checkpoint_path / 'last.ckpt'}")
        print()

    # Resolve checkpoint path for test
    test_ckpt = resume if test_only else "best"

    # --- Test (two separate passes to avoid Lightning multi-DL display artifact) ---
    print("Running final evaluation on test set...")
    datamodule.setup("test")
    test_loaders = datamodule.test_dataloader()

    # GT boxes (oracle upper bound)
    print("=" * 60)
    print("  GT boxes (oracle)")
    lightning_module.test_prefix = "test_gt"
    trainer.test(lightning_module, dataloaders=test_loaders[0], ckpt_path=test_ckpt)
    print("=" * 60)
    print()

    # Predicted boxes (end-to-end) — only if pred_test_h5 was provided
    if len(test_loaders) > 1:
        print("=" * 60)
        print("  Predicted boxes (end-to-end)")
        lightning_module.test_prefix = "test_pred"
        trainer.test(lightning_module, dataloaders=test_loaders[1], ckpt_path=test_ckpt)
        print("=" * 60)
        print()

    print("To view logs:")
    print("  tensorboard --logdir=.aim")


if __name__ == "__main__":
    args = parse_args()

    main(
        head=args.head,
        gt_train_h5=args.gt_train_h5,
        pred_train_h5=args.pred_train_h5,
        train_source=args.train_source,
        gt_test_h5=args.gt_test_h5,
        pred_test_h5=args.pred_test_h5,
        embeddings=args.embeddings,
        num_predicates=args.num_predicates,
        d_hidden=args.d_hidden,
        num_iter=args.num_iter,
        max_objects=args.max_objects,
        epochs=args.epochs,
        lr=args.lr,
        embedding_lr=args.embedding_lr,
        val_split=args.val_split,
        score_thresh=args.score_thresh,
        num_pos=args.num_pos,
        num_neg=args.num_neg,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        test_only=args.test_only,
        seed=args.seed,
    )
