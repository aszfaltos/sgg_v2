#!/usr/bin/env python3
"""Precompute Word2Vec embeddings for VRD object and predicate labels.

Downloads a pretrained word-vector model via gensim.downloader and builds
embedding matrices compatible with SGGLightningModule:

    objects:    (101, 300)  — row 0 = background zeros, rows 1-100 = Word2Vec
    predicates: (71,  300)  — row 0 = no-relation zeros, rows 1-70  = Word2Vec

The 1-indexed convention matches the detector output and sgg_collate batch dict:
``embedding_lookup[batch["labels"]]`` where labels ∈ [1, num_classes].

Multi-word labels (e.g. "traffic light", "sleep next to") are averaged over
their constituent word vectors. Words absent from the vocabulary contribute
zero and are not counted in the average; labels with no known words stay zero.

Outputs
-------
<dataset-dir>/embeddings/<model-name>_objects.pt      — (101, 300) float32
<dataset-dir>/embeddings/<model-name>_predicates.pt   — (71,  300) float32
<model-dir>/<model-name>.kv                           — gensim KeyedVectors

Usage
-----
    # First run: downloads word2vec-google-news-300 (~1.7 GB), caches .kv
    uv run python scripts/precompute_word2vec.py --dataset-dir datasets/vrd

    # Subsequent runs: loads from cache automatically
    uv run python scripts/precompute_word2vec.py --dataset-dir datasets/vrd

    # Use an existing .kv file (skip download entirely)
    uv run python scripts/precompute_word2vec.py \\
        --dataset-dir datasets/vrd \\
        --model-path models/word2vec/word2vec-google-news-300.kv

    # Use a lighter model
    uv run python scripts/precompute_word2vec.py \\
        --dataset-dir datasets/vrd \\
        --model glove-wiki-gigaword-300

Then pass to sgg_trainer.py:
    uv run python scripts/sgg_trainer.py \\
        --embeddings datasets/vrd/embeddings/word2vec-google-news-300_objects.pt \\
        ... other args
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_AVAILABLE_MODELS = [
    "word2vec-google-news-300",
    "glove-wiki-gigaword-300",
    "fasttext-wiki-news-subwords-300",
]


def load_labels(json_path: Path) -> list[str]:
    with open(json_path) as f:
        return json.load(f)


def embed_labels(labels: list[str], wv) -> torch.Tensor:
    """Build (N, dim) float32 tensor from a list of label strings.

    Each label is split on spaces; each word is looked up in ``wv``
    (lowercased first, original case as fallback). The per-label vector is
    the mean of all found word vectors. Labels with no in-vocabulary words
    are left as zero vectors.

    Args:
        labels: Class names, e.g. ``["person", "traffic light", ...]``.
        wv: Gensim ``KeyedVectors`` (or any mapping with ``vector_size``
            and ``__contains__`` / ``__getitem__``).

    Returns:
        Tensor of shape ``(len(labels), wv.vector_size)``, float32.
    """
    dim = wv.vector_size
    matrix = torch.zeros(len(labels), dim, dtype=torch.float32)
    for i, label in enumerate(labels):
        vectors = []
        for word in label.split():
            key = word.lower() if word.lower() in wv else word
            if key in wv:
                vectors.append(torch.from_numpy(wv[key].copy()))
        if vectors:
            matrix[i] = torch.stack(vectors).mean(0)
    return matrix


def build_embedding_matrix(labels: list[str], wv) -> torch.Tensor:
    """Return a ``(len(labels) + 1, dim)`` embedding matrix.

    Row 0 is an all-zero vector (background / no-relation placeholder).
    Row ``k`` (1-indexed) corresponds to ``labels[k - 1]``.  This matches
    the 1-indexed label convention used throughout sgg_v2.

    Args:
        labels: Class name list (0-indexed Python list, but 1-indexed in use).
        wv: Gensim ``KeyedVectors``.

    Returns:
        Tensor of shape ``(len(labels) + 1, dim)``, float32.
    """
    class_vecs = embed_labels(labels, wv)  # (N, dim)
    background = torch.zeros(1, wv.vector_size, dtype=torch.float32)
    return torch.cat([background, class_vecs], dim=0)


def oov_report(labels: list[str], wv) -> list[str]:
    """Return labels where every constituent word is absent from ``wv``."""
    return [
        label
        for label in labels
        if not any(w.lower() in wv or w in wv for w in label.split())
    ]


def load_word_vectors(model_name: str, model_path: str | None, model_dir: Path):
    """Load gensim KeyedVectors, downloading if necessary.

    Priority:
    1. ``model_path`` (explicit path to a .kv file) — no download.
    2. ``model_dir / model_name.kv`` (cached from a previous run).
    3. ``gensim.downloader.load(model_name)`` — downloads and caches.

    Args:
        model_name: Model name accepted by ``gensim.downloader``.
        model_path: Optional explicit path to an existing .kv file.
        model_dir: Directory used for caching downloaded models.

    Returns:
        Gensim KeyedVectors object.
    """
    from gensim.models import KeyedVectors

    if model_path is not None:
        path = Path(model_path)
        print(f"Loading word vectors from {path} ...")
        return KeyedVectors.load(str(path))

    cached = model_dir / f"{model_name}.kv"
    if cached.exists():
        print(f"Loading cached word vectors from {cached} ...")
        return KeyedVectors.load(str(cached))

    import gensim.downloader as gensim_downloader

    print(f"Downloading '{model_name}' via gensim.downloader ...")
    if "google-news-300" in model_name:
        print("  (word2vec-google-news-300 is ~1.7 GB — this will take a while)")
    wv = gensim_downloader.load(model_name)
    print(f"Saving KeyedVectors to {cached} ...")
    wv.save(str(cached))
    print(f"Saved. Future runs will load from cache.")
    return wv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute Word2Vec embeddings for VRD labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/vrd",
        help="VRD dataset directory with objects.json / predicates.json (default: datasets/vrd)",
    )
    parser.add_argument(
        "--model",
        default="word2vec-google-news-300",
        choices=_AVAILABLE_MODELS,
        help="Pretrained model name for gensim.downloader (default: word2vec-google-news-300)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        metavar="PATH",
        help="Path to an existing gensim .kv file; skips download and caching",
    )
    parser.add_argument(
        "--model-dir",
        default="models/word2vec",
        help="Directory to cache downloaded KeyedVectors (default: models/word2vec)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Output directory for .pt files (default: <dataset-dir>/embeddings)",
    )
    parser.add_argument(
        "--no-predicates",
        action="store_true",
        help="Skip building predicate embeddings (objects only)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "embeddings"
    model_dir = Path(args.model_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename stem from whichever model source is used
    if args.model_path is not None:
        model_stem = Path(args.model_path).stem
    else:
        model_stem = args.model

    print("=" * 60)
    print("Word2Vec Embedding Precomputation")
    print("=" * 60)
    print(f"Dataset dir: {dataset_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Model dir:   {model_dir}")
    print(f"Model:       {args.model_path or args.model}")
    print("=" * 60)
    print()

    # Load word vectors
    wv = load_word_vectors(args.model, args.model_path, model_dir)
    print(f"Vocab size: {len(wv):,}  |  Dim: {wv.vector_size}")
    print()

    # --- Objects ---
    objects_json = dataset_dir / "objects.json"
    objects = load_labels(objects_json)
    obj_matrix = build_embedding_matrix(objects, wv)
    obj_oov = oov_report(objects, wv)
    obj_path = output_dir / f"{model_stem}_objects.pt"
    torch.save(obj_matrix, obj_path)

    print(f"Objects ({len(objects)} classes + 1 background):")
    print(f"  Shape: {tuple(obj_matrix.shape)}")
    print(f"  OOV:   {len(obj_oov)}/{len(objects)}", end="")
    if obj_oov:
        print(f" → {obj_oov}")
    else:
        print(" (none)")
    print(f"  Saved: {obj_path}")
    print()

    # --- Predicates ---
    if not args.no_predicates:
        predicates_json = dataset_dir / "predicates.json"
        predicates = load_labels(predicates_json)
        pred_matrix = build_embedding_matrix(predicates, wv)
        pred_oov = oov_report(predicates, wv)
        pred_path = output_dir / f"{model_stem}_predicates.pt"
        torch.save(pred_matrix, pred_path)

        print(f"Predicates ({len(predicates)} classes + 1 no-relation):")
        print(f"  Shape: {tuple(pred_matrix.shape)}")
        print(f"  OOV:   {len(pred_oov)}/{len(predicates)}", end="")
        if pred_oov:
            print(f" → {pred_oov}")
        else:
            print(" (none)")
        print(f"  Saved: {pred_path}")
        print()

    print("Done!")
    print()
    print("Pass to sgg_trainer.py:")
    print(f"    --embeddings {obj_path}")


if __name__ == "__main__":
    main()
