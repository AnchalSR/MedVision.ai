"""
MedVision.ai — Centralized Configuration
"""
import os
import torch
from pathlib import Path


# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ── CLIP Settings ───────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_EMBEDDING_DIM = 512
IMAGE_SIZE = 224

# ── LLaMA / Language Model Settings ────────────────────────────────────────
# Provider: "local", "openai", "openrouter", "groq", etc.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")

# API Keys
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Base URLs (for OpenAI-compatible endpoints)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model Name
# Common OpenRouter models: "meta-llama/llama-3-70b-instruct", "openai/gpt-4o", "anthropic/claude-3.5-sonnet"
LLM_MODEL_NAME = os.environ.get(
    "MEDVISION_LLM_MODEL",
    "meta-llama/llama-3-70b-instruct",  # Defaulting to Llama 3 70B via OpenRouter
)

LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.5
LLM_TOP_P = 0.9

# ── FAISS Settings ──────────────────────────────────────────────────────────
FAISS_TOP_K = 5
FAISS_NPROBE = 10

# ── API Settings ────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 7860))
MAX_UPLOAD_SIZE_MB = 20

# ── Medical Image Settings ──────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".dcm"}
XRAY_WINDOW_CENTER = 40
XRAY_WINDOW_WIDTH = 400

# ── System Prompt for LLM ──────────────────────────────────────────────────
MEDICAL_SYSTEM_PROMPT = """You are MedVision AI, an expert medical imaging assistant.
You analyze medical images (X-rays, MRIs, CT scans) and provide detailed, 
evidence-based diagnostic explanations.

Guidelines:
- Provide structured analysis with findings, impressions, and recommendations.
- Use professional medical terminology with plain-language explanations.
- Always note that AI analysis should be verified by qualified medical professionals.
- Reference the retrieved similar cases when available.
- Include confidence levels for your findings.

IMPORTANT DISCLAIMER: This is an AI-assisted analysis tool. All findings must be 
reviewed and confirmed by a qualified radiologist or physician before clinical use."""
