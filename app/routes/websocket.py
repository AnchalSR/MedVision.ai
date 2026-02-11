"""
MedVision.ai - WebSocket Route
Real-time streaming analysis via WebSocket.
Uses asyncio.to_thread to avoid blocking the event loop.
"""
import asyncio
import json
import logging
import base64
import queue
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


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
