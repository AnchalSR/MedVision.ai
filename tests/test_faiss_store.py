"""
Tests for FAISS Vector Store module.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFAISSVectorStore:
    """Test suite for the FAISSVectorStore class."""

    def setup_method(self):
        from core.embeddings import FAISSVectorStore
        self.store = FAISSVectorStore(embedding_dim=512, index_path="/tmp/test_faiss")

    def test_mock_search_returns_results(self):
        """Test that mock search returns results."""
        results = self.store._mock_search(3)
        assert len(results) == 3
        for meta, score in results:
            assert "finding" in meta
            assert "description" in meta
            assert isinstance(score, float)

    def test_mock_search_limited_by_top_k(self):
        """Test that mock search respects top_k."""
        results = self.store._mock_search(2)
        assert len(results) == 2

    def test_mock_search_case_fields(self):
        """Test that mock cases have expected fields."""
        results = self.store._mock_search(5)
        for meta, score in results:
            assert "id" in meta
            assert "finding" in meta
            assert "description" in meta
            assert "modality" in meta

    def test_total_vectors_initially_zero(self):
        """Test that new store has zero vectors."""
        assert self.store.total_vectors == 0

    def test_add_and_count(self):
        """Test adding embeddings increases count."""
        embeddings = np.random.randn(5, 512).astype(np.float32)
        metadata = [{"id": f"test_{i}", "finding": f"Test {i}"} for i in range(5)]

        self.store._ensure_index()
        if self.store.is_mock:
            # Mock mode still tracks metadata
            self.store.add_embeddings(embeddings, metadata)
            assert self.store.total_vectors == 5
        else:
            total = self.store.add_embeddings(embeddings, metadata)
            assert total == 5

    def test_search_empty_index(self):
        """Test searching an empty index returns empty or mock results."""
        query = np.random.randn(512).astype(np.float32)
        results = self.store.search(query, top_k=3)
        # Either empty (real FAISS) or mock results
        assert isinstance(results, list)
