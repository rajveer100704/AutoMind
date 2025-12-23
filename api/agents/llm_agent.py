# api/agents/llm_agent.py
"""
LLMReasoner — safe wrapper around Gemini.
Falls back to deterministic local plan if API unavailable.
"""

import os
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")


class LLMReasoner:
    def __init__(self, model: str = "models/gemini-2.5-flash", max_tokens: int = 256, temperature: float = 0.1):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if genai and GEMINI_KEY:
            try:
                genai.configure(api_key=GEMINI_KEY)
            except Exception:
                pass

    def prompt(self, prompt_text: str, max_output_tokens: Optional[int] = None) -> str:
        """
        Returns raw response; fallback is deterministic.
        """
        if genai and GEMINI_KEY:
            try:
                resp = genai.generate_text(
                    model=self.model,
                    input=prompt_text,
                    max_output_tokens=max_output_tokens or self.max_tokens
                )
                if hasattr(resp, "text") and resp.text:
                    return resp.text.strip()
            except Exception:
                pass

        return self._fallback(prompt_text)

    def think(self, step_name: str, context: Dict[str, Any], instructions: str) -> str:
        ctx = "\n".join([f"- {k}: {v}" for k, v in context.items()])
        prompt = (
            f"You are an expert DS agent.\n"
            f"Step: {step_name}\n"
            f"Context:\n{ctx}\n\n"
            f"Instructions:\n{instructions}\n"
            f"Give a short actionable plan (1–6 lines)."
        )
        return self.prompt(prompt)

    def validate(self, step_name: str, context: Dict[str, Any], observation: str) -> str:
        prompt = (
            f"Reviewing pipeline step '{step_name}'.\n"
            f"Context: {context}\n"
            f"Observation:\n{observation}\n\n"
            "Does this look correct? Give pass/fail + 1–3 fixes."
        )
        return self.prompt(prompt)

    def _fallback(self, text: str) -> str:
        """
        Deterministic fallback for offline mode.
        """
        t = text.lower()
        if "eda" in t:
            return "Plan: summary stats, missing %, correlation heatmap, distributions."
        if "preprocess" in t:
            return "Plan: median impute numeric, mode impute categorical, encode + scale."
        if "feature" in t:
            return "Plan: date parts, interactions, polynomial features."
        if "model" in t:
            return "Plan: compare tree models, linear baseline, tune top 2."
        return "Proceed safely with default steps and generate diagnostics."
