from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer


class GlobalOrthogonalRegularizerLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        weight: float = 1.0,
    ) -> None:
        """
        Global Orthogonal Regularizer (GOR) Loss encourages embeddings within a batch to be
        orthogonal to each other, promoting diverse representations that are spread out over
        the embedding space.

        This loss penalizes non-orthogonal embeddings by computing the sum of squared dot products
        between all pairs of embeddings in the batch. When embeddings are orthogonal, their dot
        product is zero, resulting in zero loss.

        The GOR loss formula is:

        .. math::

            \\mathcal{L}_{GOR} = \\frac{1}{B(B-1)} \\sum_{i,j: i \\neq j} (\\mathbf{q}_i^T \\mathbf{q}_j)^2
            + \\frac{1}{B(B-1)} \\sum_{i,j: i \\neq j} (\\mathbf{p}_i^{+T} \\mathbf{p}_j^{+})^2

        Where B is the batch size, :math:`\\mathbf{q}_i` are anchor embeddings, and :math:`\\mathbf{p}_i^{+}`
        are positive embeddings.

        Args:
            model: SentenceTransformer model
            weight: Weight (lambda) for the GOR regularization term. Higher values encourage
                more orthogonal embeddings. Default: 1.0. Typical values range from 0.01 to 1.0.

        References:
            - Learning Spread-out Local Feature Descriptors: https://arxiv.org/abs/1708.06320

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative, ...) tuples

        Inputs:
            +------------------------------------------+--------+
            | Texts                                    | Labels |
            +==========================================+========+
            | (anchor, positive) pairs                 | none   |
            +------------------------------------------+--------+
            | (anchor, positive, negative, ...) tuples | none   |
            +------------------------------------------+--------+

        Recommendations:
            - GOR is typically used as a regularizer combined with another loss like
              :class:`MultipleNegativesRankingLoss`. Use the ``compute_loss_from_embeddings``
              method for efficient combination (computes embeddings only once).
            - Typical weight values are between 0.01 and 1.0.

        Relations:
            - Commonly combined with :class:`MultipleNegativesRankingLoss` or :class:`CoSENTLoss`
              as a regularization term.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.GlobalOrthogonalRegularizerLoss(model, weight=0.1)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.weight = weight

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        """
        Compute GOR loss from pre-computed embeddings.

        This method enables efficient combination with other losses by allowing
        embeddings to be computed once and reused.

        Args:
            embeddings: List of embedding tensors. First element is anchors, second is positives.
            labels: Labels tensor (unused, included for API compatibility).

        Returns:
            GOR loss value (scalar tensor).
        """
        gor_loss = self._compute_gor(embeddings[0])  # anchors
        if len(embeddings) > 1:
            gor_loss = gor_loss + self._compute_gor(embeddings[1])  # positives
        return gor_loss * self.weight

    def _compute_gor(self, embeddings: Tensor) -> Tensor:
        """
        Compute Global Orthogonal Regularization for a single set of embeddings.

        The formula is: L_GOR = (1 / B(B-1)) * sum_{i,j: i!=j} (e_i^T e_j)^2

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)

        Returns:
            GOR loss value (scalar tensor)
        """
        batch_size = embeddings.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

        # Compute all pairwise dot products: (B, B) matrix
        dot_products = torch.mm(embeddings, embeddings.t())

        # Create mask to exclude diagonal elements (i != j)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

        # Extract off-diagonal elements, square them, and sum
        off_diagonal_squared = dot_products[mask] ** 2

        # Normalize by B(B-1), the number of off-diagonal pairs
        normalization = batch_size * (batch_size - 1)

        return off_diagonal_squared.sum() / normalization

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{zhang2017learning,
    title={Learning Spread-out Local Feature Descriptors},
    author={Zhang, Xu and Yu, Felix X and Kumar, Sanjiv and Chang, Shih-Fu},
    booktitle={ICCV},
    year={2017}
}
"""
