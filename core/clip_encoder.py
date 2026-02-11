"""
MedVision.ai — CLIP Image & Text Encoder
Wraps OpenCLIP for medical image embedding extraction.
"""
import logging
from typing import Union, List

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


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

        import threading
        import os

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
        load_thread.join(timeout=15)  # 15 second timeout

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
