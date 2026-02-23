"""Base classes for SGG relation heads.

Provides the abstract interface and output dataclass that all SGG heads
must implement. Ensures a consistent API for swapping heads at training time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class SGGHeadOutput:
    """Output from an SGG head's forward pass.

    Contains per-image relation predictions along with the pair indices needed
    to map each logit back to the detected boxes.

    All list fields have one element per image in the batch. Images with no
    detections or no valid edges produce empty tensors.

    Attributes:
        rel_logits: Per-image list of (E_i, num_predicates) unnormalized relation
            scores. E_i is the number of directed edges for image i.
        subject_indices: Per-image list of (E_i,) int64 indices into the
            per-image box list identifying the subject of each relation.
        object_indices: Per-image list of (E_i,) int64 indices into the
            per-image box list identifying the object of each relation.
    """

    rel_logits: list[Tensor]
    subject_indices: list[Tensor]
    object_indices: list[Tensor]

    def __len__(self) -> int:
        """Return number of images in batch."""
        return len(self.rel_logits)

    @property
    def total_edges(self) -> int:
        """Return total number of edges across all images."""
        return sum(t.shape[0] for t in self.rel_logits)

    @property
    def device(self) -> torch.device:
        """Return device of tensors."""
        return self.rel_logits[0].device if self.rel_logits else torch.device("cpu")

    def to(self, device: torch.device | str) -> "SGGHeadOutput":
        """Move all tensors to specified device."""
        return SGGHeadOutput(
            rel_logits=[t.to(device) for t in self.rel_logits],
            subject_indices=[t.to(device) for t in self.subject_indices],
            object_indices=[t.to(device) for t in self.object_indices],
        )


class SGGHead(nn.Module, ABC):
    """Abstract base class for SGG relation heads.

    SGG heads receive frozen detector output and produce relation predictions
    by running message passing over the object graph.

    The word embedding table is passed into forward() rather than stored in
    the head so that it can be shared, fine-tuned with a separate LR, and
    swapped between embedding types (GloVe, BERT, etc.) at the training level.

    Subclasses must implement:
        - forward(): Run message passing and return SGGHeadOutput.
        - num_predicates: Number of relation classes.
    """

    @property
    @abstractmethod
    def num_predicates(self) -> int:
        """Number of relation/predicate classes."""
        ...

    @abstractmethod
    def forward(
        self,
        batch: dict[str, Tensor],
        embedding_lookup: Tensor,
    ) -> SGGHeadOutput:
        """Run message passing and classify relations.

        Args:
            batch: Collated batch dict from sgg_collate. Expected keys:
                roi_features  (total_N, C, H, W) — pooled ROI features
                labels        (total_N,)          — 1-indexed class labels
                sub_idx       (total_E,)          — global subject indices
                obj_idx       (total_E,)          — global object indices
                geo           (total_E, 12)       — geometric encoding
                node_counts   (B,)                — objects per image
                edge_counts   (B,)                — edges per image
                image_offsets (B,)                — node offset per image
            embedding_lookup: (num_classes, semantic_dim) word embedding matrix.
                Indexed by 1-indexed class labels.

        Returns:
            SGGHeadOutput with per-image relation logits and local pair indices.
        """
        ...
