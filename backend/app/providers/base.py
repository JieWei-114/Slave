"""
LLM Provider Base
Abstract interface for LLM backends (Ollama, OpenAI-compatible servers, ...)
Defines the streaming/non-streaming contract shared by all providers

"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class ProviderStreamError(RuntimeError):
    """Raised when a provider streaming API fails or returns an error payload."""


# Backwards-compatible alias (previously defined in app.services.ollama_service)
OllamaStreamError = ProviderStreamError


class LLMProvider(ABC):
    """Abstract LLM provider interface"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier (e.g. 'ollama', 'openai_compat')"""

    @abstractmethod
    def stream_chat(
        self, prompt: str, model: str, system: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream response from the LLM token by token.

        Yields plain token strings (same contract as the original stream_ollama).
        Raises ProviderStreamError on failure.

        """

    @abstractmethod
    async def generate_once(self, prompt: str, model: str, system: Optional[str] = None) -> str:
        """
        Get complete response from the LLM in one call (non-streaming).

        Returns '' on failure (same contract as the original call_ollama_once).

        """

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List model names available on this provider."""
