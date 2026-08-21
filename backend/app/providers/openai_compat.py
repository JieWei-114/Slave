"""
OpenAI-Compatible Provider
Handles communication with any OpenAI-compatible API
(vLLM, llama.cpp server, LM Studio, OpenAI itself)

"""

import json
import logging
from typing import AsyncIterator, Optional

import httpx

from app.config.settings import settings
from app.providers.base import LLMProvider, ProviderStreamError

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    """LLM provider speaking the OpenAI-compatible API (/v1/chat/completions)"""

    def __init__(self):
        if not (settings.OPENAI_COMPAT_BASE_URL or '').strip():
            raise ValueError(
                "LLM_PROVIDER='openai_compat' requires OPENAI_COMPAT_BASE_URL to be set "
                '(e.g. http://localhost:8001 for vLLM, https://api.openai.com for OpenAI)'
            )

    @property
    def name(self) -> str:
        return 'openai_compat'

    def _base_url(self) -> str:
        return (settings.OPENAI_COMPAT_BASE_URL or '').rstrip('/')

    def _headers(self) -> dict:
        headers = {'Content-Type': 'application/json'}
        if settings.OPENAI_COMPAT_API_KEY:
            headers['Authorization'] = f'Bearer {settings.OPENAI_COMPAT_API_KEY}'
        return headers

    @staticmethod
    def _build_messages(prompt: str, system: Optional[str] = None) -> list[dict]:
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        return messages

    async def stream_chat(
        self, prompt: str, model: str, system: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream response token by token via SSE (stream=true)

        """
        logger.info(
            'openai_compat stream called with system=%s (type=%s)',
            'present' if system else 'None',
            type(system).__name__,
        )
        payload = {
            'model': model,
            'messages': self._build_messages(prompt, system),
            'stream': True,
        }

        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
                async with client.stream(
                    'POST',
                    f'{self._base_url()}/v1/chat/completions',
                    json=payload,
                    headers=self._headers(),
                ) as resp:
                    resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if not line or not line.startswith('data:'):
                            continue

                        data_str = line[len('data:') :].strip()
                        if data_str == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning(
                                'openai_compat stream returned non-JSON line: %s', line
                            )
                            continue

                        if 'error' in data:
                            logger.error('openai_compat stream error: %s', data['error'])
                            raise ProviderStreamError(
                                f'OpenAI-compatible stream error: {data["error"]}'
                            )

                        choices = data.get('choices') or []
                        if not choices:
                            continue

                        delta = choices[0].get('delta') or {}
                        token = delta.get('content')
                        if token:
                            yield token

                        if choices[0].get('finish_reason'):
                            break
        except httpx.HTTPError as exc:
            logger.error('openai_compat stream request failed: %s', exc)
            raise ProviderStreamError(f'OpenAI-compatible request failed: {exc}') from exc

    async def generate_once(self, prompt: str, model: str, system: Optional[str] = None) -> str:
        """
        Get complete response in one call (non-streaming)

        """
        payload = {
            'model': model,
            'messages': self._build_messages(prompt, system),
            'stream': False,
        }

        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
                resp = await client.post(
                    f'{self._base_url()}/v1/chat/completions',
                    json=payload,
                    headers=self._headers(),
                )
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error('openai_compat call failed: %s', exc)
            return ''

        try:
            choices = resp.json().get('choices') or []
            content = (choices[0].get('message') or {}).get('content', '') if choices else ''
            return (content or '').strip()
        except (ValueError, AttributeError, IndexError):
            logger.error('openai_compat returned invalid JSON body')
            return ''

    async def list_models(self) -> list[str]:
        """
        List models available on the server (/v1/models)

        """
        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
                resp = await client.get(
                    f'{self._base_url()}/v1/models',
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.error('openai_compat list models failed: %s', exc)
            return []

        return [m.get('id', '') for m in data.get('data', []) if m.get('id')]
