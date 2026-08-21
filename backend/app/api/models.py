"""
Model Management API

Hugging Face GGUF model discovery + Ollama pull/delete/list.

Ollama natively supports pulling GGUF models straight from the Hugging Face
hub via names like 'hf.co/{user}/{repo}:{quant}', so this module only needs
to (1) search the public HF hub for GGUF repos and (2) proxy Ollama's native
pull/delete APIs.

"""

import asyncio
import json
import logging
import re

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config.constants import (
    HTTP_BAD_GATEWAY,
    HTTP_BAD_REQUEST,
    HTTP_NOT_FOUND,
)
from app.config.settings import settings
from app.providers import get_provider

router = APIRouter(prefix='/models', tags=['models'])
logger = logging.getLogger(__name__)

HF_API_BASE = 'https://huggingface.co/api'
HF_TIMEOUT = 20.0

# Quant label inside GGUF filenames, e.g. Q4_K_M, Q8_0, IQ2_XS, F16, BF16
_QUANT_RE = re.compile(r'(?:IQ|Q)\d+(?:_[A-Z0-9]+)*|F16|F32|BF16', re.IGNORECASE)

# Tag subset worth surfacing in search results (skip noise like arxiv ids)
_INTERESTING_TAGS = {'gguf', 'text-generation', 'conversational', 'llama', 'mistral', 'qwen'}

# HF repo ids are '{user}/{repo}' — reject anything else (path traversal, etc.)
_REPO_ID_RE = re.compile(r'^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$')

HTTP_CONFLICT = 409

# Pulls are GB-scale: run one at a time, and reject duplicate pulls outright
_pull_semaphore = asyncio.Semaphore(1)
_pulls_in_flight: set[str] = set()


def _parse_quant(filename: str) -> str | None:
    """Extract a quant label (e.g. 'Q4_K_M') from a GGUF filename."""
    match = _QUANT_RE.search(filename)
    return match.group(0).upper() if match else None


@router.get('')
async def list_installed_models():
    """List models installed on the active LLM provider."""
    models = await get_provider().list_models()
    return {'provider': get_provider().name, 'models': models}


@router.get('/search')
async def search_hf_models(
    q: str = Query(..., min_length=1, description='Search query'),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Search the Hugging Face hub for GGUF model repos (sorted by downloads).

    """
    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
            resp = await client.get(
                f'{HF_API_BASE}/models',
                params={'search': q, 'filter': 'gguf', 'sort': 'downloads', 'limit': limit},
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.error('HF model search failed: %s', exc)
        raise HTTPException(status_code=HTTP_BAD_GATEWAY, detail='Hugging Face search failed')

    return [
        {
            'id': m.get('id') or m.get('modelId'),
            'downloads': m.get('downloads', 0),
            'likes': m.get('likes', 0),
            'updated_at': m.get('lastModified'),
            'tags': [t for t in m.get('tags', []) if t in _INTERESTING_TAGS],
        }
        for m in data
    ]


@router.get('/search/{repo_id:path}/files')
async def list_hf_gguf_files(repo_id: str):
    """
    List GGUF files (with parsed quant labels) for a Hugging Face repo.

    Each entry includes a ready-to-pull 'ollama_name' like
    'hf.co/{repo_id}:{quant}'.

    """
    if not _REPO_ID_RE.match(repo_id):
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail='Invalid repo id')

    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
            resp = await client.get(f'{HF_API_BASE}/models/{repo_id}')
            if resp.status_code == HTTP_NOT_FOUND:
                raise HTTPException(status_code=HTTP_NOT_FOUND, detail='Repo not found')
            resp.raise_for_status()
            data = resp.json()
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        logger.error('HF repo lookup failed for %s: %s', repo_id, exc)
        raise HTTPException(status_code=HTTP_BAD_GATEWAY, detail='Hugging Face lookup failed')

    files = []
    for sibling in data.get('siblings', []):
        filename = sibling.get('rfilename', '')
        if not filename.lower().endswith('.gguf'):
            continue
        quant = _parse_quant(filename)
        files.append(
            {
                'filename': filename,
                'quant': quant,
                'size_bytes': sibling.get('size'),
                'ollama_name': f'hf.co/{repo_id}:{quant}' if quant else f'hf.co/{repo_id}',
            }
        )

    return {'repo_id': repo_id, 'files': files}


class PullModelRequest(BaseModel):
    """JSON body for the model pull endpoint."""

    name: str


@router.post('/pull')
async def pull_model(payload: PullModelRequest):
    """
    Pull a model into Ollama, streaming progress via SSE.

    Accepts either a plain Ollama model name ('llama3.2:1b') or a Hugging
    Face GGUF reference ('hf.co/{user}/{repo}:{quant}'). Proxies Ollama's
    streaming /api/pull, forwarding each progress line as an SSE data event.

    """
    if (settings.LLM_PROVIDER or 'ollama').strip().lower() != 'ollama':
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail="Model pull requires LLM_PROVIDER='ollama' (pulls go through the Ollama server)",
        )

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail='Model name is required')

    if name in _pulls_in_flight:
        raise HTTPException(
            status_code=HTTP_CONFLICT, detail=f"Model '{name}' is already being pulled"
        )
    _pulls_in_flight.add(name)

    async def event_generator():
        # Pulls are GB-scale: no read timeout, keep connect timeout sane
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        try:
            # Serialize pulls: different names wait here for their turn
            async with _pull_semaphore, httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    'POST',
                    f'{settings.OLLAMA_URL}/api/pull',
                    json={'name': name, 'stream': True},
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode('utf-8', errors='replace')
                        logger.error('ollama pull failed (%s): %s', resp.status_code, body)
                        yield f'event: error\ndata: {json.dumps({"error": body.strip() or "pull failed"})}\n\n'
                        return

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            progress = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if progress.get('error'):
                            yield f'event: error\ndata: {json.dumps({"error": progress["error"]})}\n\n'
                            return

                        event = {'status': progress.get('status', '')}
                        if 'completed' in progress:
                            event['completed'] = progress['completed']
                        if 'total' in progress:
                            event['total'] = progress['total']
                        yield f'data: {json.dumps(event)}\n\n'

            yield f'event: done\ndata: {json.dumps({"name": name})}\n\n'
        except httpx.HTTPError as exc:
            logger.error('ollama pull stream failed for %s: %s', name, exc)
            yield f'event: error\ndata: {json.dumps({"error": str(exc)})}\n\n'
        finally:
            _pulls_in_flight.discard(name)

    return StreamingResponse(
        event_generator(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@router.delete('/{name:path}')
async def delete_model(name: str):
    """Delete an installed model from Ollama (proxy /api/delete)."""
    if (settings.LLM_PROVIDER or 'ollama').strip().lower() != 'ollama':
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail="Model delete requires LLM_PROVIDER='ollama'",
        )

    try:
        async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
            resp = await client.request(
                'DELETE', f'{settings.OLLAMA_URL}/api/delete', json={'name': name}
            )
    except httpx.HTTPError as exc:
        logger.error('ollama delete failed for %s: %s', name, exc)
        raise HTTPException(status_code=HTTP_BAD_GATEWAY, detail='Ollama delete failed')

    if resp.status_code == HTTP_NOT_FOUND:
        raise HTTPException(status_code=HTTP_NOT_FOUND, detail=f"Model '{name}' not found")
    if resp.status_code != 200:
        raise HTTPException(status_code=HTTP_BAD_GATEWAY, detail='Ollama delete failed')

    return {'deleted': name}
