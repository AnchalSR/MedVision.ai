"""
Tests for the VQA Engine (end-to-end pipeline).
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestVQAEngine:
    """Test suite for the VQAEngine class."""

    def setup_method(self):
        from core.pipeline import VQAEngine
        self.engine = VQAEngine()

    def test_engine_creation(self):
        """Test that engine can be created."""
        assert self.engine is not None
        assert not self.engine._initialized

    def test_engine_initialization(self):
        """Test that engine initializes without error."""
        self.engine.initialize()
        assert self.engine._initialized

    def test_answer_with_pil_image(self):
        """Test answering a question with a PIL Image."""
        # Create a simple test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        result = self.engine.answer(img, "What do you see?")

        assert "answer" in result
        assert "confidence" in result
        assert "retrieved_cases" in result
        assert "modality" in result
        assert "processing_time" in result
        assert "query_id" in result
        assert result["status"] == "success"

    def test_answer_returns_nonempty(self):
        """Test that the answer is non-empty."""
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = self.engine.answer(img, "What abnormalities are visible?")
        assert len(result["answer"]) > 0

    def test_query_counting(self):
        """Test that queries are counted."""
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        self.engine.answer(img, "Test 1")
        self.engine.answer(img, "Test 2")
        assert self.engine._query_count == 2

    def test_get_status(self):
        """Test engine status reporting."""
        status = self.engine.get_status()
        assert "engine" in status
        assert "initialized" in status
        assert "total_queries" in status

    def test_stream_answer(self):
        """Test streaming answer generation."""
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        chunks = list(self.engine.answer_stream(img, "What is this?"))

        assert len(chunks) > 0
        # Should start with metadata
        assert chunks[0]["type"] == "metadata"
        # Should end with done
        assert chunks[-1]["type"] == "done"
        # Should have some token chunks
        token_chunks = [c for c in chunks if c["type"] == "token"]
        assert len(token_chunks) > 0
