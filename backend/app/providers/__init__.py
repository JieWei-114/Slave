"""
LLM Provider Factory
Selects the active LLM provider from settings.LLM_PROVIDER

"""

import logging

from app.config.settings import settings
from app.providers.base import LLMProvider, OllamaStreamError, ProviderStreamError

logger = logging.getLogger(__name__)

__all__ = ['LLMProvider', 'OllamaStreamError', 'ProviderStreamError', 'get_provider']

_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """
    Get the configured LLM provider (cached singleton).

    settings.LLM_PROVIDER: 'ollama' (default) | 'openai_compat'

    """
    global _provider
    if _provider is None:
        name = (settings.LLM_PROVIDER or 'ollama').strip().lower()
        if name == 'openai_compat':
            from app.providers.openai_compat import OpenAICompatProvider

            _provider = OpenAICompatProvider()
        elif name == 'ollama':
            from app.providers.ollama_provider import OllamaProvider

            _provider = OllamaProvider()
        else:
            logger.warning("Unknown LLM_PROVIDER '%s', falling back to ollama", name)
            from app.providers.ollama_provider import OllamaProvider

            _provider = OllamaProvider()
        logger.info('LLM provider initialized: %s', _provider.name)
    return _provider
