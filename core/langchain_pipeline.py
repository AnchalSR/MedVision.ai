"""
MedVision.ai - LangChain Multimodal Pipeline
Orchestrates CLIP, FAISS, and LLaMA into a unified reasoning chain.
"""
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MedVisionPipeline:
    """
    LangChain-style pipeline that orchestrates the full VQA workflow:
    Image -> CLIP Encoding -> FAISS Retrieval -> LLaMA Reasoning -> Diagnostic Answer.
    """

    def __init__(self):
        self._clip_encoder = None
        self._faiss_store = None
        self._llm_engine = None
        self._preprocessor = None
        self._initialized = False

    def initialize(self):
        """Initialize all pipeline components."""
        if self._initialized:
            return

        from core.clip_encoder import CLIPEncoder
        from core.faiss_store import FAISSVectorStore
        from core.llama_reasoning import LLaMAReasoningEngine
        from core.preprocessor import MedicalImagePreprocessor

        logger.info("Initializing MedVision Pipeline...")

        self._preprocessor = MedicalImagePreprocessor()
        self._clip_encoder = CLIPEncoder()
        self._faiss_store = FAISSVectorStore()
        self._llm_engine = LLaMAReasoningEngine()

        # Build FAISS index from knowledge base
        self._faiss_store.build_from_knowledge_base(encoder=self._clip_encoder)

        self._initialized = True
        logger.info("MedVision Pipeline initialized successfully")

    def run(self, image_source, question: str) -> Dict[str, Any]:
        """
        Execute the full VQA pipeline.

        Args:
            image_source: Image file path, bytes, or PIL Image.
            question: User's question about the image.

        Returns:
            Dict with keys: answer, confidence, retrieved_cases, modality, image_context.
        """
        self.initialize()

        # Step 1: Preprocess image
        logger.info("Step 1: Preprocessing image...")
        image = self._preprocessor.load_image(image_source)
        processed_image = self._preprocessor.preprocess(image)
        image_context = self._preprocessor.create_context_description(image)
        modality = self._preprocessor.detect_modality(image)

        # Step 2: Encode image with CLIP
        logger.info("Step 2: Encoding image with CLIP...")
        image_embedding = self._clip_encoder.encode_image(processed_image)

        # Step 3: Retrieve similar cases from FAISS
        logger.info("Step 3: Retrieving similar cases from FAISS...")
        search_results = self._faiss_store.search(image_embedding, top_k=5)
        retrieved_cases = [meta for meta, score in search_results]

        # Step 4: Compute image-question relevance
        logger.info("Step 4: Computing image-question relevance...")
        question_embedding = self._clip_encoder.encode_text(question)
        relevance_score = self._clip_encoder.compute_similarity(
            image_embedding, question_embedding[0]
        )

        # Step 5: Generate answer with LLaMA
        logger.info("Step 5: Generating diagnostic answer with LLaMA...")
        answer = self._llm_engine.generate(
            question=question,
            image_context=image_context,
            retrieved_cases=retrieved_cases,
        )

        # Compute overall confidence
        retrieval_confidence = search_results[0][1] if search_results else 0.0
        confidence = (relevance_score + retrieval_confidence) / 2

        result = {
            "answer": answer,
            "confidence": round(max(0, min(1, confidence)), 3),
            "retrieved_cases": retrieved_cases[:3],  # Top 3 for display
            "modality": modality,
            "image_context": image_context,
            "relevance_score": round(float(relevance_score), 3),
            "pipeline_status": {
                "clip": "loaded" if not self._clip_encoder.is_mock else "mock",
                "faiss": f"{self._faiss_store.total_vectors} vectors",
                "llm": "loaded" if not self._llm_engine.is_mock else "mock",
            },
        }

        logger.info(f"Pipeline complete. Confidence: {result['confidence']}")
        return result

    def run_stream(self, image_source, question: str):
        """
        Execute the pipeline with streaming LLM output.
        Yields dicts: first a 'metadata' dict, then 'token' dicts.
        """
        self.initialize()

        # Steps 1-4 same as run()
        image = self._preprocessor.load_image(image_source)
        processed_image = self._preprocessor.preprocess(image)
        image_context = self._preprocessor.create_context_description(image)
        modality = self._preprocessor.detect_modality(image)

        image_embedding = self._clip_encoder.encode_image(processed_image)
        search_results = self._faiss_store.search(image_embedding, top_k=5)
        retrieved_cases = [meta for meta, score in search_results]

        question_embedding = self._clip_encoder.encode_text(question)
        relevance_score = self._clip_encoder.compute_similarity(
            image_embedding, question_embedding[0]
        )

        retrieval_confidence = search_results[0][1] if search_results else 0.0
        confidence = (relevance_score + retrieval_confidence) / 2

        # Yield metadata first
        yield {
            "type": "metadata",
            "confidence": round(max(0, min(1, confidence)), 3),
            "retrieved_cases": retrieved_cases[:3],
            "modality": modality,
            "image_context": image_context,
        }

        # Stream answer tokens
        for token in self._llm_engine.generate_stream(
            question=question,
            image_context=image_context,
            retrieved_cases=retrieved_cases,
        ):
            yield {"type": "token", "content": token}

        yield {"type": "done"}

    def get_status(self) -> Dict[str, Any]:
        """Return the current status of all pipeline components."""
        status = {
            "initialized": self._initialized,
            "components": {},
        }

        if self._initialized:
            status["components"] = {
                "clip_encoder": {
                    "loaded": self._clip_encoder.is_loaded,
                    "mock": self._clip_encoder.is_mock,
                    "model": "ViT-B/32",
                },
                "faiss_store": {
                    "total_vectors": self._faiss_store.total_vectors,
                    "mock": self._faiss_store.is_mock,
                },
                "llm_engine": {
                    "loaded": self._llm_engine.is_loaded,
                    "mock": self._llm_engine.is_mock,
                    "model": self._llm_engine.model_name,
                },
                "preprocessor": {
                    "loaded": True,
                    "target_size": self._preprocessor.target_size,
                },
            }

        return status
