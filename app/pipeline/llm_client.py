"""
Thin, dependency-light LLM client. Uses `requests` directly against each
provider's REST API rather than pulling in the full google-generativeai or
groq SDKs -- keeps requirements.txt small and avoids SDK version churn for
what is, on our side, just "send messages, get text back".

Both providers are called with a strict "respond with JSON only" system
instruction when structured output is needed (analyzer, responder selection
step); app/pipeline/json_utils.py handles defensively extracting JSON even
if a model wraps it in prose or code fences despite instructions.
"""
import json as json_module

import requests

from app import config


class LLMError(Exception):
    pass


class LLMClient:
    """Interface both real backends and the test MockLLMClient implement."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        try:
            resp = requests.post(url, json=payload, timeout=config.LLM_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise LLMError(f"Gemini call failed: {e}") from e


class GroqClient(LLMClient):
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT_SECONDS
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"Groq call failed: {e}") from e


class MockLLMClient(LLMClient):
    """Deterministic stand-in used by tests / local dev without an API key.
    Returns canned JSON responses based on simple keyword matching so the
    rest of the pipeline (router, retriever, validator, schema) can be
    exercised end-to-end without any network access or real LLM cost."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        lower = user_prompt.lower()

        if "return a json object describing the assessment comparison" in system_prompt.lower():
            return json_module.dumps({"reply": "Mock comparison of the requested assessments."})

        if "select which of the candidate assessments" in system_prompt.lower():
            return json_module.dumps(
                {
                    "selected_ids": [],
                    "reply": "Mock: here are the assessments that fit your needs.",
                }
            )

        # Analyzer-style prompt: decide intent from the latest user message
        if "ignore all previous instructions" in lower or "system prompt" in lower:
            intent = "off_topic"
        elif "hiring" in lower or "developer" in lower or "engineer" in lower or "role" in lower:
            intent = "recommend"
        else:
            intent = "clarify"

        return json_module.dumps(
            {
                "intent": intent,
                "constraints": {
                    "role": "software engineer" if intent == "recommend" else None,
                    "seniority": None,
                    "skills": [],
                    "test_types_wanted": [],
                    "languages": [],
                    "other_notes": None,
                },
                "compare_targets": [],
                "missing_info": [] if intent == "recommend" else ["role"],
                "ready_to_recommend": intent == "recommend",
                "user_confirmed_shortlist": False,
            }
        )


def get_default_client() -> LLMClient:
    if config.LLM_PROVIDER == "groq":
        return GroqClient()
    return GeminiClient()
