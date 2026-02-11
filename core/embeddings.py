"""
MedVision.ai — Embeddings & Vector Store
Combines CLIP Image/Text Encoding and FAISS vector store for medical image retrieval.
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class CLIPEncoder:
    """Encodes medical images and text into a shared embedding space using CLIP."""

    def __init__(self, model_name: str = None, pretrained: str = None, device: str = None):
        from config.settings import CLIP_MODEL_NAME, CLIP_PRETRAINED, DEVICE, CLIP_EMBEDDING_DIM

        self.model_name = model_name or CLIP_MODEL_NAME
        self.pretrained = pretrained or CLIP_PRETRAINED
        self.device = device or DEVICE
        self.embedding_dim = CLIP_EMBEDDING_DIM
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the CLIP model with a timeout to avoid long hangs."""
        if self._model is not None:
            return

        # Skip real model loading if MEDVISION_MOCK env var is set
        if os.environ.get("MEDVISION_MOCK", "").lower() in ("1", "true", "yes"):
            logger.info("Mock mode forced via MEDVISION_MOCK env var")
            self._model = "mock"
            return

        result = {"model": None, "preprocess": None, "tokenizer": None, "error": None}

        def _do_load():
            try:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name, pretrained=self.pretrained
                )
                tokenizer = open_clip.get_tokenizer(self.model_name)
                model = model.to(self.device).eval()
                result["model"] = model
                result["preprocess"] = preprocess
                result["tokenizer"] = tokenizer
            except Exception as e:
                result["error"] = e

        logger.info(f"Loading CLIP model: {self.model_name} (timeout: 15s)...")
        load_thread = threading.Thread(target=_do_load, daemon=True)
        load_thread.start()
        load_thread.join(timeout=15)

        if result["model"] is not None:
            self._model = result["model"]
            self._preprocess = result["preprocess"]
            self._tokenizer = result["tokenizer"]
            logger.info(f"CLIP model loaded on {self.device}")
        else:
            reason = result["error"] or "Timed out (model download may be required)"
            logger.warning(f"CLIP model unavailable: {reason}. Using mock encoder.")
            self._model = "mock"

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, str, np.ndarray]) -> np.ndarray:
        """
        Encode a medical image into a CLIP embedding vector.
        
        Args:
            image: PIL Image, file path string, or numpy array.
            
        Returns:
            Normalized embedding vector of shape (embedding_dim,).
        """
        self._load_model()

        # Convert input to PIL Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        if self._model == "mock":
            return self._mock_embedding()

        preprocessed = self._preprocess(image).unsqueeze(0).to(self.device)
        embedding = self._model.encode_image(preprocessed)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text query into a CLIP embedding vector.
        
        Args:
            text: A string or list of strings.
            
        Returns:
            Normalized embedding vector(s).
        """
        self._load_model()

        if isinstance(text, str):
            text = [text]

        if self._model == "mock":
            return np.stack([self._mock_embedding() for _ in text])

        tokens = self._tokenizer(text).to(self.device)
        embeddings = self._model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    def compute_similarity(
        self, image_embedding: np.ndarray, text_embedding: np.ndarray
    ) -> float:
        """Compute cosine similarity between image and text embeddings."""
        img_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
        txt_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
        return float(np.dot(img_norm.flatten(), txt_norm.flatten()))

    def _mock_embedding(self) -> np.ndarray:
        """Return a deterministic mock embedding for demo/testing."""
        rng = np.random.RandomState(42)
        emb = rng.randn(self.embedding_dim).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_mock(self) -> bool:
        return self._model == "mock"


# ─────────────────────────────────────────────────────────────────────────────
# FAISS VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

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
            self._index = self._faiss.IndexFlatIP(self.embedding_dim)
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

        if encoder is None:
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
