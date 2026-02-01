from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import GlobalOrthogonalRegularizerLoss, MultipleNegativesRankingLoss


@pytest.fixture
def mock_model():
    """Create a mock SentenceTransformer model."""
    model = Mock(spec=SentenceTransformer)
    return model


class TestGORLoss:
    """Tests for GlobalOrthogonalRegularizerLoss."""

    @pytest.mark.parametrize("weight", [0.1, 0.5, 1.0, 2.0])
    def test_weight_initialization(self, mock_model, weight):
        """Test that weight is set correctly during initialization."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model, weight=weight)
        assert loss.weight == weight
        assert loss.get_config_dict() == {"weight": weight}

    def test_default_weight(self, mock_model):
        """Test that default weight is 1.0."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model)
        assert loss.weight == 1.0

    def test_citation_property(self, mock_model):
        """Test that citation property returns valid BibTeX."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model)
        assert isinstance(loss.citation, str)
        assert "zhang2017learning" in loss.citation

    @pytest.mark.parametrize(
        "batch_size,expected_zero",
        [
            (1, True),  # Batch size 1 should return 0
            (2, False),  # Batch size > 1 should compute GOR
            (4, False),
        ],
    )
    def test_batch_size_handling(self, mock_model, batch_size, expected_zero):
        """Test GOR computation for different batch sizes."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model, weight=1.0)
        embeddings = torch.randn(batch_size, 768)
        result = loss._compute_gor(embeddings)

        if expected_zero:
            assert result.item() == 0.0
        else:
            assert result.item() >= 0.0

    @pytest.mark.parametrize(
        "embeddings,expected_approx",
        [
            # Orthogonal embeddings (one-hot style): loss should be ~0
            (
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                0.0,
            ),
            # Simple case for manual verification:
            # e0=[1,0], e1=[0.5,0.5], e2=[0,1]
            # dot(e0,e1)=0.5, dot(e0,e2)=0, dot(e1,e2)=0.5
            # sum of squared off-diagonal = 2*(0.25+0+0.25) = 1.0
            # normalized by 3*2=6 -> 1/6
            (
                torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
                1.0 / 6.0,
            ),
        ],
    )
    def test_gor_formula_correctness(self, mock_model, embeddings, expected_approx):
        """Verify GOR formula: (1/B(B-1)) * sum_{i!=j} (e_i^T e_j)^2."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model, weight=1.0)
        result = loss._compute_gor(embeddings)
        assert result.item() == pytest.approx(expected_approx, abs=1e-6)

    def test_identical_embeddings_high_loss(self, mock_model):
        """GOR loss should be high for identical normalized embeddings."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model, weight=1.0)

        # Create identical normalized embeddings
        single_embedding = torch.randn(768)
        single_embedding = single_embedding / single_embedding.norm()
        embeddings = single_embedding.unsqueeze(0).repeat(4, 1)

        result = loss._compute_gor(embeddings)
        # For normalized identical embeddings: dot product = 1, loss = 1
        assert result.item() == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize(
        "num_embedding_sets,weight",
        [
            (1, 1.0),  # Only anchors
            (2, 1.0),  # Anchors + positives
            (1, 0.5),  # Only anchors with weight
            (2, 0.1),  # Anchors + positives with weight
        ],
    )
    def test_compute_loss_from_embeddings(self, mock_model, num_embedding_sets, weight):
        """Test compute_loss_from_embeddings with different configurations."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model, weight=weight)

        embeddings = [torch.randn(4, 768) for _ in range(num_embedding_sets)]
        labels = torch.tensor([0, 1, 2, 3])

        result = loss.compute_loss_from_embeddings(embeddings, labels)

        # Compute expected value
        expected = sum(loss._compute_gor(emb) for emb in embeddings) * weight

        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([])
        assert result.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_forward_returns_scalar_tensor(self, mock_model):
        """Test that forward returns a scalar tensor."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model, weight=0.1)

        # Mock model to return embeddings
        mock_embeddings = torch.randn(2, 768)
        mock_model.return_value = {"sentence_embedding": mock_embeddings}

        sentence_features = [{"input_ids": torch.tensor([[1, 2], [3, 4]])}]
        labels = torch.tensor([0, 1])

        result = loss(sentence_features, labels)

        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([])
        assert result.item() >= 0

    def test_forward_calls_model_correctly(self, mock_model):
        """Test that forward calls the model with sentence features."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model, weight=1.0)

        mock_embeddings = torch.randn(2, 768)
        mock_model.return_value = {"sentence_embedding": mock_embeddings}

        sentence_features = [{"input_ids": torch.tensor([[1, 2], [3, 4]])}]
        labels = torch.tensor([0, 1])

        loss(sentence_features, labels)

        # Verify model was called once per sentence feature set
        assert mock_model.call_count == 1

    def test_forward_matches_compute_loss_from_embeddings(self, mock_model):
        """Test that forward produces same result as compute_loss_from_embeddings."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model, weight=0.1)

        mock_embeddings = torch.randn(2, 768)
        mock_model.return_value = {"sentence_embedding": mock_embeddings}

        sentence_features = [{"input_ids": torch.tensor([[1, 2], [3, 4]])}]
        labels = torch.tensor([0, 1])

        forward_result = loss(sentence_features, labels)
        embeddings_result = loss.compute_loss_from_embeddings([mock_embeddings], labels)

        assert forward_result.item() == pytest.approx(embeddings_result.item(), abs=1e-6)

    def test_gradients_flow_through_embeddings(self, mock_model):
        """Test that gradients flow through the loss computation."""
        loss = GlobalOrthogonalRegularizerLoss(model=mock_model, weight=0.1)

        # Create embeddings with gradients enabled
        embeddings = torch.randn(4, 768, requires_grad=True)

        result = loss._compute_gor(embeddings)
        result.backward()

        assert embeddings.grad is not None
        assert embeddings.grad.abs().sum() > 0

    def test_combine_with_mnrl_via_embeddings(self, mock_model):
        """Test efficient combination with MNRL using compute_loss_from_embeddings."""
        mnrl = MultipleNegativesRankingLoss(model=mock_model)
        gor = GlobalOrthogonalRegularizerLoss(model=mock_model, weight=0.1)

        # Create mock embeddings
        anchors = torch.randn(2, 768)
        positives = torch.randn(2, 768)
        embeddings = [anchors, positives]
        labels = torch.tensor([0, 1])

        # Compute both losses from embeddings
        mnrl_loss = mnrl.compute_loss_from_embeddings(embeddings, labels)
        gor_loss = gor.compute_loss_from_embeddings(embeddings, labels)
        total_loss = mnrl_loss + gor_loss

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.shape == torch.Size([])
        assert total_loss.item() > 0
