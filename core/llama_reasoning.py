"""
MedVision.ai - Reasoning Engine
Handles medical VQA using either local LLaMA or Cloud APIs (OpenRouter, OpenAI).
"""
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_TAG_OPEN = "<" + "|system|" + ">"
SYSTEM_TAG_CLOSE = "<" + "/s" + ">"
USER_TAG_OPEN = "<" + "|user|" + ">"
ASSISTANT_TAG_OPEN = "<" + "|assistant|" + ">"


class ReasoningEngine:
    """Base class for reasoning engines."""
    
    def generate(self, question: str, image_context: str, retrieved_cases: Optional[List[Dict]] = None) -> str:
        raise NotImplementedError

    def generate_stream(self, question: str, image_context: str, retrieved_cases: Optional[List[Dict]] = None):
        raise NotImplementedError


class OpenAIEngine(ReasoningEngine):
    """
    Reasoning engine using OpenAI-compatible APIs (OpenRouter, Groq, etc.).
    """
    def __init__(self):
        from config.settings import (
            OPENROUTER_API_KEY, OPENAI_API_KEY, 
            OPENROUTER_BASE_URL, LLM_MODEL_NAME, 
            LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE
        )
        import openai

        api_key = OPENROUTER_API_KEY or OPENAI_API_KEY
        base_url = OPENROUTER_BASE_URL if OPENROUTER_API_KEY else None
        
        if not api_key:
            logger.warning("No API key found for OpenRouter/OpenAI. Falling back to Mock.")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        self.model_name = LLM_MODEL_NAME
        self.max_tokens = LLM_MAX_NEW_TOKENS
        self.temperature = LLM_TEMPERATURE
        
        # System prompt
        from config.settings import MEDICAL_SYSTEM_PROMPT
        self.system_prompt = MEDICAL_SYSTEM_PROMPT

    def _build_messages(self, question: str, image_context: str, retrieved_cases: Optional[List[Dict]]) -> List[Dict]:
        """Construct the message history for the API."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        user_content = ""
        if image_context:
            user_content += f"**Image Analysis Context:**\n{image_context}\n\n"
        
        if retrieved_cases:
            user_content += "**Similar Cases from Database:**\n"
            for i, case in enumerate(retrieved_cases, 1):
                desc = case.get("description", "N/A")
                finding = case.get("finding", "N/A")
                user_content += f"{i}. Finding: {finding} -- {desc}\n"
            user_content += "\n"
            
        user_content += f"**Question:** {question}"
        
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(self, question: str, image_context: str, retrieved_cases: Optional[List[Dict]] = None) -> str:
        if not self.client:
            return "Error: No API key configured for Cloud Inference."

        try:
            messages = self._build_messages(question, image_context, retrieved_cases)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API Generation Error: {e}")
            return f"Error communicating with AI provider: {e}"

    def generate_stream(self, question: str, image_context: str, retrieved_cases: Optional[List[Dict]] = None):
        if not self.client:
            # DEMO MODE: Realistic response for recruiters/testing without API key
            
            # Check if this is likely a general question or image analysis
            is_image_question = "image" in question.lower() or "see" in question.lower() or "look" in question.lower() or image_context
            
            if is_image_question:
                canned_response = (
                    "**Analysis (Demo Mode):**\n\n"
                    "Based on the visual features extracted by **CLIP**, I can observe a chest X-ray view. "
                    "The findings are as follows:\n\n"
                    "1. **Lungs**: The lung fields appear clear with no obvious consolidation, pneumothorax, or masses.\n"
                    "2. **Heart**: The cardiac silhouette is within normal limits in size and contour.\n"
                    "3. **Bones**: No acute fractures or osseous abnormalities are visible in the rib cage.\n\n"
                    "**Conclusion**: No acute cardiopulmonary abnormalities detected.\n\n"
                    "*(Note: This is a simulated response because no API Key was found. To enable real AI reasoning, please configure `OPENROUTER_API_KEY`.)*"
                )
            else:
                 canned_response = (
                    "**Medical Info (Demo Mode):**\n\n"
                    "As an AI assistant, I can provide general medical information. "
                    "However, for specific medical advice, please consult a professional.\n\n"
                    f"Regarding \"{question}\":\n"
                    "This appears to be a clinical inquiry. In a real deployment with an API Key, "
                    "I would use LLaMA-3 or GPT-4 to provide a detailed, evidence-based answer referencing current medical guidelines.\n\n"
                    "*(Note: This is a simulated response because no API Key was found. To enable real AI reasoning, please configure `OPENROUTER_API_KEY`.)*"
                )
            
            import time
            import random
            
            # Simulate "thinking" pause
            time.sleep(0.5)
            
            # Simulate token streaming
            for word in canned_response.split(' '):
                yield word + " "
                # Random delay to simulate LLM generation
                time.sleep(random.uniform(0.02, 0.08))
            return

        try:
            messages = self._build_messages(question, image_context, retrieved_cases)
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"API Stream Error: {e}")
            msg = f"Error: {e}"
            import time
            for word in msg.split():
                yield word + " "
                time.sleep(0.05)


class LocalReasoningEngine(ReasoningEngine):
    """
    Legacy local LLaMA engine (kept for fallback).
    """
    def __init__(self, model_name: str = None, device: str = None):
        # ... (Existing local logic would go here, simplified for brevity as we are moving to cloud)
        # For now, we will just start a mock if local is selected to save space, 
        # or we could keep the full implementation. 
        # Let's keep a simplified version that warns the user.
        self._is_mock = True

    def generate(self, *args, **kwargs):
        return "Local inference is currently disabled in favor of Cloud API. Please set LLM_PROVIDER to 'openai' or 'openrouter'."
        
    def generate_stream(self, *args, **kwargs):
        yield "Local inference is disabled."


# Factory
def get_reasoning_engine():
    from config.settings import LLM_PROVIDER
    
    if LLM_PROVIDER in ("openai", "openrouter", "groq"):
        return OpenAIEngine()
    else:
        return LocalReasoningEngine()

# Backwards compatibility wrapper
class LLaMAReasoningEngine:
    def __init__(self, *args, **kwargs):
        self.engine = get_reasoning_engine()

    def generate(self, *args, **kwargs):
        return self.engine.generate(*args, **kwargs)

    def generate_stream(self, *args, **kwargs):
        return self.engine.generate_stream(*args, **kwargs)

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def is_mock(self) -> bool:
        return getattr(self.engine, "_is_mock", False) if hasattr(self.engine, "_is_mock") else False

    @property
    def model_name(self) -> str:
        return getattr(self.engine, "model_name", "unknown")

