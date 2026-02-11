"""
MedVision.ai - REST API Routes
Handles image upload, analysis, and health endpoints.
"""
import io
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()


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
    import asyncio
    from app.main import get_vqa_engine

    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
    if image.content_type and image.content_type not in allowed_types:
        # Still allow it but log a warning
        logger.warning(f"Unusual content type: {image.content_type}")

    try:
        # Read image bytes
        image_bytes = await image.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")

        if len(image_bytes) > 20 * 1024 * 1024:  # 20MB limit
            raise HTTPException(status_code=413, detail="Image too large (max 20MB)")

        # Run VQA pipeline in a thread to avoid blocking the event loop
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
    from fastapi.responses import FileResponse

    file_path = SAMPLE_IMAGES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")

    return FileResponse(str(file_path))


@router.get("/knowledge-base/stats")
async def knowledge_base_stats():
    """Get statistics about the knowledge base."""
    from config.settings import KNOWLEDGE_BASE_PATH
    import json

    if not KNOWLEDGE_BASE_PATH.exists():
        return {"total_cases": 0, "modalities": {}, "body_parts": {}}

    with open(KNOWLEDGE_BASE_PATH) as f:
        kb = json.load(f)

    cases = kb.get("cases", [])
    modalities = {}
    body_parts = {}
    severities = {}

    for case in cases:
        mod = case.get("modality", "unknown")
        modalities[mod] = modalities.get(mod, 0) + 1

        bp = case.get("body_part", "unknown")
        body_parts[bp] = body_parts.get(bp, 0) + 1

        sev = case.get("severity", "unknown")
        severities[sev] = severities.get(sev, 0) + 1

    return {
        "total_cases": len(cases),
        "modalities": modalities,
        "body_parts": body_parts,
        "severities": severities,
    }


@router.post("/chat")
async def chat_general(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    """
    General medical chat without image.
    """
    from app.main import get_vqa_engine
    import asyncio
    
    engine = get_vqa_engine()
    
    # Initial implementation: we can reuse the pipeline but without image
    # Or access the LLM directly.
    # The VQA engine's pipeline expects an image.
    # For now, let's use the underlying LLM engine directly or mock a blank image context.
    
    try:
        # Access internal LLM engine
        llm = engine._pipeline._llm_engine

        # Manage session history so chat feels conversational
        from app.chat import session_manager

        if not session_id:
            session_id = session_manager.create_session()

        session_manager.add_user_message(session_id, question)

        # Build a prompt from recent history (simple concatenation)
        history = session_manager.get_history(session_id)
        prompt_parts = []
        for msg in history:
            role = msg.get("role")
            text = msg.get("text")
            if role == "user":
                prompt_parts.append(f"User: {text}")
            else:
                prompt_parts.append(f"Assistant: {text}")

        prompt_parts.append(f"User: {question}")
        prompt = "\n".join(prompt_parts)

        # Run LLM generation in a thread-safe way
        response = await asyncio.to_thread(
            llm.generate,
            question=prompt,
            image_context="",
            retrieved_cases=[],
        )

        # Store bot reply in session
        session_manager.add_bot_message(session_id, response)

        return {
            "session_id": session_id,
            "answer": response,
            "confidence": 1.0,
            "status": "success",
            "mode": "text-only",
            "history_length": len(session_manager.get_history(session_id)),
        }
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
