"""
MedVision.ai - FAISS Vector Store
Manages a FAISS index for medical image-text embedding retrieval.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-backed vector store for storing and retrieving medical image embeddings
    alongside their metadata (descriptions, findings, modality tags).
    """

    def __init__(self, embedding_dim: int = None, index_path: str = None):
        from config.settings import CLIP_EMBEDDING_DIM, FAISS_INDEX_PATH

        self.embedding_dim = embedding_dim or CLIP_EMBEDDING_DIM
        self.index_path = Path(index_path or FAISS_INDEX_PATH)
        self._index = None
        self._metadata: List[Dict] = []
        self._faiss = None

    def _load_faiss(self):
        """Lazy-load FAISS library."""
        if self._faiss is not None:
            return
        try:
            import faiss
            self._faiss = faiss
            logger.info("FAISS library loaded successfully")
        except ImportError:
            logger.warning("FAISS not installed. Using mock vector store.")
            self._faiss = "mock"

    def _ensure_index(self):
        """Initialize or load the FAISS index."""
        self._load_faiss()

        if self._index is not None:
            return

        # Try to load existing index
        index_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "metadata.json"

        if index_file.exists() and meta_file.exists() and self._faiss != "mock":
            try:
                self._index = self._faiss.read_index(str(index_file))
                with open(meta_file, "r") as f:
                    self._metadata = json.load(f)
                logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")

        # Create new index
        if self._faiss != "mock":
            self._index = self._faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine sim)
            logger.info(f"Created new FAISS index (dim={self.embedding_dim})")
        else:
            self._index = "mock"

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict],
    ) -> int:
        """
        Add embeddings with associated metadata to the index.

        Args:
            embeddings: Array of shape (N, embedding_dim).
            metadata_list: List of dicts with keys like 'description', 'finding', 'modality'.

        Returns:
            Total number of vectors in the index.
        """
        self._ensure_index()

        if self._index == "mock":
            self._metadata.extend(metadata_list)
            return len(self._metadata)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalized = (embeddings / norms).astype(np.float32)

        self._index.add(normalized)
        self._metadata.extend(metadata_list)

        logger.info(f"Added {len(metadata_list)} vectors. Total: {self._index.ntotal}")
        return self._index.ntotal

    def search(
        self, query_embedding: np.ndarray, top_k: int = None
    ) -> List[Tuple[Dict, float]]:
        """
        Search for the most similar embeddings.

        Args:
            query_embedding: Query vector of shape (embedding_dim,).
            top_k: Number of results to return.

        Returns:
            List of (metadata_dict, similarity_score) tuples.
        """
        from config.settings import FAISS_TOP_K

        top_k = top_k or FAISS_TOP_K
        self._ensure_index()

        if self._index == "mock" or not self._metadata:
            return self._mock_search(top_k)

        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query) + 1e-8
        query = query / norm

        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        distances, indices = self._index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._metadata):
                results.append((self._metadata[idx], float(dist)))

        return results

    def save_index(self):
        """Persist the FAISS index and metadata to disk."""
        self._ensure_index()

        self.index_path.mkdir(parents=True, exist_ok=True)

        if self._index != "mock" and self._index is not None:
            index_file = self.index_path / "index.faiss"
            self._faiss.write_index(self._index, str(index_file))

        meta_file = self.index_path / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump(self._metadata, f, indent=2)

        logger.info(f"Saved FAISS index to {self.index_path}")

    def load_index(self) -> bool:
        """Load index from disk. Returns True if successful."""
        self._ensure_index()
        return self._index is not None and self._index != "mock"

    def build_from_knowledge_base(self, knowledge_base_path: str = None, encoder=None):
        """
        Build the vector index from the medical knowledge base JSON.
        Uses a CLIP encoder to generate embeddings for each entry's description.
        """
        from config.settings import KNOWLEDGE_BASE_PATH

        kb_path = knowledge_base_path or str(KNOWLEDGE_BASE_PATH)

        if not os.path.exists(kb_path):
            logger.warning(f"Knowledge base not found: {kb_path}. Using empty index.")
            self._ensure_index()
            return

        with open(kb_path, "r") as f:
            knowledge_base = json.load(f)

        entries = knowledge_base.get("cases", [])
        if not entries:
            logger.warning("Knowledge base is empty.")
            self._ensure_index()
            return

        # Generate text embeddings for each case description
        if encoder is None:
            from core.clip_encoder import CLIPEncoder
            encoder = CLIPEncoder()

        descriptions = [entry.get("description", "") for entry in entries]
        embeddings = encoder.encode_text(descriptions)

        self.add_embeddings(embeddings, entries)
        self.save_index()
        logger.info(f"Built FAISS index from {len(entries)} knowledge base entries")

    def _mock_search(self, top_k: int) -> List[Tuple[Dict, float]]:
        """Return mock search results for demo/testing."""
        mock_cases = [
            {
                "id": "case_001",
                "finding": "Normal chest anatomy",
                "description": "PA chest radiograph showing normal cardiomediastinal silhouette, clear lungs, and no acute osseous abnormalities.",
                "modality": "X-ray",
                "similarity": 0.92,
            },
            {
                "id": "case_002",
                "finding": "Mild cardiomegaly",
                "description": "Frontal chest X-ray demonstrating mild enlargement of the cardiac silhouette with a cardiothoracic ratio of approximately 0.55.",
                "modality": "X-ray",
                "similarity": 0.87,
            },
            {
                "id": "case_003",
                "finding": "Right lower lobe consolidation",
                "description": "Chest radiograph showing dense opacity in the right lower lobe consistent with pneumonic consolidation.",
                "modality": "X-ray",
                "similarity": 0.84,
            },
            {
                "id": "case_004",
                "finding": "Pleural effusion",
                "description": "Chest X-ray demonstrating blunting of the right costophrenic angle suggesting pleural effusion.",
                "modality": "X-ray",
                "similarity": 0.81,
            },
            {
                "id": "case_005",
                "finding": "Brain MRI - Normal study",
                "description": "Axial T2-weighted MRI of the brain showing normal grey-white matter differentiation with no acute intracranial pathology.",
                "modality": "MRI",
                "similarity": 0.78,
            },
        ]
        return [(case, case["similarity"]) for case in mock_cases[:top_k]]

    @property
    def total_vectors(self) -> int:
        """Total number of vectors in the index."""
        if self._index is None or self._index == "mock":
            return len(self._metadata)
        return self._index.ntotal

    @property
    def is_mock(self) -> bool:
        return self._index == "mock"
