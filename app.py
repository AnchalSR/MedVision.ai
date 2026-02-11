"""
MedVision.ai — Quick Start Launcher
Run this script to start the full system: python app.py
"""
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print()
    print("=" * 60)
    print("  MedVision.ai — AI Visual Question Answering System")
    print("  Multimodal Medical Intelligence")
    print("=" * 60)
    print()
    # Tech Stack:
    print("    - CLIP (ViT-B/32)  : Vision Encoder")
    print("    - LLaMA / TinyLlama: Reasoning Engine")
    print("    - FAISS            : Vector Retrieval")
    print("    - LangChain        : Pipeline Orchestration")
    print("    - FastAPI          : Web Server")
    print()

    try:
        from config.settings import API_HOST, API_PORT
    except ImportError:
        API_HOST = "0.0.0.0"
        API_PORT = 7860

    print("  Starting server...")
    print(f"  Open: http://localhost:{API_PORT}")
    print("=" * 60)
    print()

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
