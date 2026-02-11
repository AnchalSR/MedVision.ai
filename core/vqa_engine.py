"""
MedVision.ai - VQA Engine
Top-level inference API that provides a simple interface to the full pipeline.
"""
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class VQAEngine:
    """
    Top-level Visual Question Answering engine.
    Provides a clean, simple API over the MedVision pipeline.
    """

    def __init__(self):
        from core.langchain_pipeline import MedVisionPipeline
        self._pipeline = MedVisionPipeline()
        self._initialized = False
        self._query_count = 0

    def initialize(self):
        """Pre-initialize all models. Call this at startup for faster first query."""
        if self._initialized:
            return
        logger.info("Initializing VQA Engine...")
        self._pipeline.initialize()
        self._initialized = True
        logger.info("VQA Engine ready")

    def answer(self, image_source, question: str) -> Dict[str, Any]:
        """
        Answer a question about a medical image.

        Args:
            image_source: Image file path, bytes, or PIL Image.
            question: Natural language question about the image.

        Returns:
            Dict with: answer, confidence, retrieved_cases, modality,
            image_context, processing_time, query_id.
        """
        start_time = time.time()
        self._query_count += 1
        query_id = f"q_{self._query_count:06d}"

        logger.info(f"[{query_id}] Processing: '{question[:80]}...'")

        try:
            result = self._pipeline.run(image_source, question)
            result["processing_time"] = round(time.time() - start_time, 2)
            result["query_id"] = query_id
            result["status"] = "success"

            logger.info(
                f"[{query_id}] Done in {result['processing_time']}s "
                f"(confidence: {result['confidence']})"
            )
            return result

        except Exception as e:
            logger.error(f"[{query_id}] Error: {e}")
            return {
                "answer": f"An error occurred during analysis: {str(e)}",
                "confidence": 0.0,
                "retrieved_cases": [],
                "modality": "unknown",
                "image_context": "",
                "processing_time": round(time.time() - start_time, 2),
                "query_id": query_id,
                "status": "error",
                "error": str(e),
            }

    def answer_stream(self, image_source, question: str):
        """
        Stream an answer to a question about a medical image.
        Yields dict chunks: metadata first, then token-by-token answer, then done.
        """
        self._query_count += 1
        query_id = f"q_{self._query_count:06d}"
        start_time = time.time()

        logger.info(f"[{query_id}] Streaming: '{question[:80]}...'")

        try:
            for chunk in self._pipeline.run_stream(image_source, question):
                chunk["query_id"] = query_id
                if chunk["type"] == "done":
                    chunk["processing_time"] = round(time.time() - start_time, 2)
                yield chunk
        except Exception as e:
            logger.error(f"[{query_id}] Stream error: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "query_id": query_id,
            }

    def get_status(self) -> Dict[str, Any]:
        """Get engine status and component health."""
        status = {
            "engine": "MedVision VQA Engine v1.0",
            "initialized": self._initialized,
            "total_queries": self._query_count,
        }
        if self._initialized:
            status["pipeline"] = self._pipeline.get_status()
        return status
