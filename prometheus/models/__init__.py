"""Model clients (local HF CausalLM, later OpenRouter)."""

from prometheus.models.llm import ChatResult, LocalLLM

__all__ = ["ChatResult", "LocalLLM"]
