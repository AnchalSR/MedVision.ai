"""
MedVision.ai - API Routes
REST API and WebSocket endpoints for medical image analysis.
"""
import asyncio
import base64
import io
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# REST API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Health check with model status."""
    from app.main import get_vqa_engine
    engine = get_vqa_engine()
    status = engine.get_status()
    return {
        "status": "healthy",
        "engine": status,
    }


@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(default="What do you see in this medical image? Provide a detailed analysis."),
):
    """
    Analyze a medical image with a question.

    - **image**: Medical image file (PNG, JPG, JPEG, BMP, TIFF)
    - **question**: Question about the image
    """
    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
    if image.content_type and image.content_type not in allowed_types:
        logger.warning(f"Unusual content type: {image.content_type}")

    try:
        # Read image bytes
        image_bytes = await image.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")

        if len(image_bytes) > 20 * 1024 * 1024:  # 20MB limit
            raise HTTPException(status_code=413, detail="Image too large (max 20MB)")

        # Run VQA pipeline in a thread to avoid blocking the event loop
        from app.main import get_vqa_engine
        engine = get_vqa_engine()
        result = await asyncio.to_thread(engine.answer, image_bytes, question)

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/sample-images")
async def list_sample_images():
    """List available sample/demo images."""
    from config.settings import SAMPLE_IMAGES_DIR

    samples = []
    if SAMPLE_IMAGES_DIR.exists():
        for f in sorted(SAMPLE_IMAGES_DIR.iterdir()):
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                samples.append({
                    "name": f.stem.replace("_", " ").title(),
                    "filename": f.name,
                    "path": f"/api/sample-image/{f.name}",
                })

    return {"samples": samples}


@router.get("/sample-image/{filename}")
async def get_sample_image(filename: str):
    """Serve a sample image file."""
    from config.settings import SAMPLE_IMAGES_DIR

    file_path = SAMPLE_IMAGES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")

    # Security: ensure the file is actually in the sample_images directory
    try:
        file_path.relative_to(SAMPLE_IMAGES_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(path=file_path, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for streaming medical image analysis.

    Protocol:
    1. Client sends JSON: {"image": "<base64>", "question": "..."}
    2. Server streams back JSON chunks:
       - {"type": "status", "message": "Processing..."}
       - {"type": "metadata", "confidence": 0.85, ...}
       - {"type": "token", "content": "word "}
       - {"type": "done", "processing_time": 1.23}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Receive analysis request
            data = await websocket.receive_text()
            request = json.loads(data)

            image_b64 = request.get("image", "")
            question = request.get("question", "What do you see in this medical image?")

            if not image_b64:
                # Text-only chat mode
                await websocket.send_json({"type": "status", "message": "Thinking..."})
                
                chunk_queue = queue.Queue()
                
                def run_text_chat():
                    try:
                        from app.main import get_vqa_engine
                        engine = get_vqa_engine()
                        llm = engine._pipeline._llm_engine
                        
                        # Stream from LLM directly
                        for token in llm.generate_stream(question, image_context="", retrieved_cases=[]):
                            chunk_queue.put({"type": "token", "content": token})
                        
                        chunk_queue.put({"type": "done"})
                        chunk_queue.put(None)
                    except Exception as e:
                        logger.error(f"Chat error: {e}", exc_info=True)
                        chunk_queue.put({"type": "error", "message": str(e)})
                        chunk_queue.put(None)

                thread = threading.Thread(target=run_text_chat, daemon=True)
                thread.start()
            
            else:
                # Image analysis mode
                # Decode base64 image
                try:
                    if "," in image_b64:
                        image_b64 = image_b64.split(",", 1)[1]
                    image_bytes = base64.b64decode(image_b64)
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Invalid image data: {e}"})
                    continue

                await websocket.send_json({"type": "status", "message": "Preprocessing image..."})

                # Run the blocking pipeline in a background thread,
                # streaming chunks back through a queue
                chunk_queue = queue.Queue()

                def run_pipeline():
                    try:
                        from app.main import get_vqa_engine
                        engine = get_vqa_engine()
                        for chunk in engine.answer_stream(image_bytes, question):
                            chunk_queue.put(chunk)
                        chunk_queue.put(None)  # Sentinel
                    except Exception as e:
                        logger.error(f"Pipeline error: {e}", exc_info=True)
                        chunk_queue.put({"type": "error", "message": str(e)})
                        chunk_queue.put(None)

                thread = threading.Thread(target=run_pipeline, daemon=True)
                thread.start()

            # Consume chunks from the queue and send via websocket
            while True:
                # Poll the queue without blocking the event loop
                chunk = await asyncio.to_thread(chunk_queue.get)
                if chunk is None:
                    break
                await websocket.send_json(chunk)
                if chunk.get("type") in ("done", "error"):
                    break

            thread.join(timeout=5)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
