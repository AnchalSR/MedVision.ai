---
title: MedVision AI
emoji: ⚕️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# MedVision.ai

**AI Visual Question Answering System for Medical Imaging**

A multimodal AI agent that interprets X-rays, MRIs, and CT scans through integrated CLIP + LLaMA architecture, with FAISS-powered retrieval and LangChain orchestration.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)

---

## Architecture

```
User Upload (X-ray/MRI) 
    -> Preprocessor (Modality Detection)
    -> CLIP Encoder (ViT-B/32)
    -> FAISS Vector Store (Top-K Retrieval)
    -> LangChain Pipeline (Orchestration)
    -> LLaMA Reasoning (Diagnostic Answer + Confidence)
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vision Encoder | CLIP (ViT-B/32) | Image & text embedding |
| Language Model | LLaMA / TinyLlama | Diagnostic reasoning |
| Vector Store | FAISS | Image-text retrieval |
| Orchestration | LangChain | Pipeline composition |
| Backend | FastAPI + WebSocket | REST API & streaming |
| Frontend | Vanilla JS + CSS | Premium dark-mode UI |

## Quick Start

### 1. Install Dependencies

```bash
cd MedVision.ai
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python run.py
```

### 3. Open the UI

Navigate to **http://localhost:8000** in your browser.

## Project Structure

```
MedVision.ai/
├── app/                    # FastAPI web application
│   ├── main.py            # App entry point
│   ├── routes/            # API & WebSocket routes
│   └── static/            # Frontend (HTML/CSS/JS)
├── core/                   # AI pipeline modules
│   ├── clip_encoder.py    # CLIP image/text encoding
│   ├── llama_reasoning.py # LLaMA text generation
│   ├── faiss_store.py     # FAISS vector index
│   ├── langchain_pipeline.py # Pipeline orchestration
│   ├── preprocessor.py    # Medical image preprocessing
│   └── vqa_engine.py      # Top-level VQA API
├── data/
│   └── knowledge_base.json # 50+ medical cases
├── config/
│   └── settings.py        # Centralized configuration
├── tests/                  # Unit & integration tests
├── requirements.txt
├── run.py                  # One-command launcher
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Analyze image with question |
| GET | `/api/health` | System health check |
| GET | `/api/sample-images` | List demo images |
| GET | `/api/knowledge-base/stats` | KB statistics |
| WS | `/ws/analyze` | Streaming analysis |

## Configuration

Set environment variables to customize:

```bash
# Use a different language model
export MEDVISION_LLM_MODEL="meta-llama/Llama-2-7b-chat-hf"
```

## Disclaimer

> This is an AI research tool intended for educational and research purposes only.
> All AI-generated findings must be reviewed and confirmed by qualified medical professionals
> before any clinical use. This system is NOT a certified medical device.

## License

MIT
