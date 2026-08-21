"""
Ollama Service (compatibility shim)
Delegates to the configured LLM provider (app.providers)
Kept so existing call sites (chat_service, memory_service, ...) work unchanged

"""

from typing import Optional

from app.providers import get_provider
from app.providers.base import OllamaStreamError, ProviderStreamError

__all__ = ['OllamaStreamError', 'ProviderStreamError', 'stream_ollama', 'call_ollama_once']


async def stream_ollama(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    images: Optional[list[str]] = None,
):
    """
    Stream response from the configured LLM provider token by token.

    images: optional base64-encoded images (no 'data:' prefix) for vision models.

    """
    async for token in get_provider().stream_chat(
        prompt=prompt, model=model, system=system, images=images
    ):
        yield token


async def call_ollama_once(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    json_schema: Optional[dict] = None,
) -> str:
    """
    Get complete response from the configured LLM provider in one call (non-streaming)

    """
    return await get_provider().generate_once(
        prompt=prompt, model=model, system=system, json_schema=json_schema
    )
