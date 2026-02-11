# MedVision.ai Deployment Guide

This guide details how to deploy MedVision.ai to **Hugging Face Spaces** (Docker).

## 1. Prerequisites

- A Hugging Face account.
- An API Key from **OpenRouter** or **OpenAI** (for the AI reasoning).

## 2. Deployment Steps

1.  **Create a New Space**:
    - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
    - **Space Name**: `medvision-ai` (or similar).
    - **License**: `MIT`.
    - **SDK**: Select **Docker**. (This is simpler than Gradio for our custom UI).
    - Click **Create Space**.

2.  **Upload Files**:
    - You can upload files via the web interface or use `git`.
    - Upload the entire content of this folder to the Space.
    - **Important**: Ensure `Dockerfile`, `requirements.txt`, `app/`, `core/`, `config/`, and `data/` are uploaded.

3.  **Configure Secrets (Environment Variables)**:
    - In your Space settings, go to the **Settings** tab.
    - Scroll to **Variables and secrets**.
    - Add the following **Secrets** (for security):
        - `OPENROUTER_API_KEY`: Your OpenRouter key (starts with `sk-or-`).
        - OR `OPENAI_API_KEY`: Your OpenAI key (starts with `sk-`).
    - Add the following **Variables** (public config):
        - `LLM_PROVIDER`: `openrouter` (or `openai`).
        - `LLM_MODEL_NAME`: `meta-llama/llama-3-70b-instruct` (recommended) or `gpt-4o`.

4.  **Build & Run**:
    - Hugging Face will automatically build the Docker image.
    - Watch the **Logs** tab. It might take a few minutes to install dependencies (PyTorch, etc.).
    - Once "Running", click the **App** tab to see your deployed MedVision AI!

## 3. Troubleshooting

- **"Runtime Error"**: Check the Logs. If it says "Out of Memory", try switching to a larger hardware tier (Settings -> Hardware) or use a smaller vision model in `core/vqa_engine.py`.
- **"No API Key"**: Ensure you added the Secret correctly in the Settings tab.
- **"404 Not Found"**: properly set PORT env var to 7860 (Default for HF). Our Dockerfile handles this.

## 4. Local Development

To run locally:
1.  **Install**: `pip install -r requirements.txt`
2.  **Run**: `python app.py`
3.  **Visit**: `http://localhost:7860`
