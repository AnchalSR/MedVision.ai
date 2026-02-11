"""
Tests for CLIP Encoder module.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestCLIPEncoder:
    """Test suite for the CLIPEncoder class."""

    def setup_method(self):
        from core.clip_encoder import CLIPEncoder
        self.encoder = CLIPEncoder()

    def test_mock_embedding_shape(self):
        """Test that mock embeddings have correct dimensionality."""
        emb = self.encoder._mock_embedding()
        assert emb.shape == (512,), f"Expected (512,), got {emb.shape}"

    def test_mock_embedding_normalized(self):
        """Test that mock embeddings are approximately unit-normalized."""
        emb = self.encoder._mock_embedding()
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5, f"Expected norm ~1.0, got {norm}"

    def test_mock_embedding_deterministic(self):
        """Test that mock embeddings are deterministic (same seed)."""
        emb1 = self.encoder._mock_embedding()
        emb2 = self.encoder._mock_embedding()
        np.testing.assert_array_equal(emb1, emb2)

    def test_compute_similarity_identical(self):
        """Test similarity of identical embeddings is ~1.0."""
        emb = self.encoder._mock_embedding()
        sim = self.encoder.compute_similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-5, f"Expected ~1.0, got {sim}"

    def test_compute_similarity_orthogonal(self):
        """Test similarity of orthogonal embeddings is ~0.0."""
        emb1 = np.zeros(512, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(512, dtype=np.float32)
        emb2[1] = 1.0
        sim = self.encoder.compute_similarity(emb1, emb2)
        assert abs(sim) < 1e-5, f"Expected ~0.0, got {sim}"

    def test_encode_text_returns_array(self):
        """Test that text encoding returns a numpy array."""
        # This will use mock since model won't be downloaded in tests
        self.encoder._model = "mock"
        result = self.encoder.encode_text("test medical image")
        assert isinstance(result, np.ndarray)
        assert result.shape[-1] == 512

    def test_encode_text_batch(self):
        """Test batch text encoding."""
        self.encoder._model = "mock"
        texts = ["chest xray", "brain mri", "knee scan"]
        result = self.encoder.encode_text(texts)
        assert result.shape == (3, 512)
