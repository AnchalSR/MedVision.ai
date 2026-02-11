"""
MedVision.ai - FastAPI Application
Main entry point for the web server.
"""
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Handles startup and shutdown logic.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("  MedVision.ai - Starting up...")
    logger.info("=" * 60)
    try:
        engine = get_vqa_engine()
        engine.initialize()
    except Exception as e:
        logger.error(f"VQA engine failed to initialize: {e}", exc_info=True)

        # Fallback lightweight mock engine so the server remains usable for demos/tests
        class MockEngine:
            def __init__(self):
                self._initialized = True

            def initialize(self):
                return

            def answer(self, image_source, question: str):
                return {
                    "answer": "Demo response: model unavailable, running in mock mode.",
                    "confidence": 0.0,
                    "retrieved_cases": [],
                    "modality": "unknown",
                    "image_context": "",
                    "processing_time": 0.0,
                    "query_id": "mock_000",
                    "status": "mock",
                }

            def answer_stream(self, image_source, question: str):
                # Simple streaming demo
                yield {"type": "metadata", "confidence": 0.0, "retrieved_cases": [], "modality": "unknown", "image_context": ""}
                for token in ("Demo", " ", "response", "."):
                    yield {"type": "token", "content": token}
                yield {"type": "done", "processing_time": 0.0}

            def get_status(self):
                return {"engine": "MockEngine", "initialized": True, "total_queries": 0}

        _vqa_engine = MockEngine()
        logger.info("Running with Mock VQA engine (models unavailable or failed to load).")
    logger.info("MedVision.ai is ready!")
    logger.info(f"Open http://localhost:8000/static/index.html in your browser")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown (if needed in future)
    logger.info("MedVision.ai - Shutting down...")


# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedVision.ai",
    description="AI Visual Question Answering System for Medical Imaging",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Routes
app.include_router(router, prefix="/api", tags=["API", "WebSocket"])

# ── VQA Engine (shared instance) ───────────────────────────────────────────
_vqa_engine = None


def get_vqa_engine():
    """Get or create the shared VQA engine instance."""
    global _vqa_engine
    if _vqa_engine is None:
        from core.pipeline import VQAEngine
        _vqa_engine = VQAEngine()
    return _vqa_engine


@app.get("/")
async def root():
    """Redirect to the main UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")
