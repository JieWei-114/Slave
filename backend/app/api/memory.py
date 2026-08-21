"""
Memory API

"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from app.config.constants import HTTP_BAD_REQUEST, HTTP_INTERNAL_ERROR
from app.models.dto import CreateMemoryRequest
from app.services.memory_service import (
    add_memory,
    compress_memories,
    delete_memory,
    list_all_memories,
    reindex_memories,
    search_memories,
    set_memory_enabled,
)

router = APIRouter(prefix='/memory', tags=['memory'])
logger = logging.getLogger(__name__)


@router.get('/')
def get_memories(
    session_id: str = Query(...),
):
    """
    List all enabled memories for a session, with id, content, created_at, source

    """
    try:
        if not session_id:
            raise HTTPException(status_code=HTTP_BAD_REQUEST, detail='session_id is required')
        logger.info('Listing memories for session %s', session_id)
        return list_all_memories(session_id)
    except Exception as e:
        logger.error('Failed to list memories: %s', e, exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to list memories')


@router.post('/')
def create_memory(payload: CreateMemoryRequest):
    """
    Create a new memory item for a session, object with assigned ID

    """
    try:
        logger.info('Creating memory for session %s', payload.session_id)
        return add_memory(
            payload.content,
            payload.session_id,
            category=payload.category or 'other',
        )
    except Exception as e:
        logger.error(
            'Failed to create memory for session %s: %s', payload.session_id, e, exc_info=True
        )
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to create memory')


@router.patch('/{memory_id}/enable')
def enable_memory(memory_id: str):
    """
    Enable a memory for use in context selection.

    """
    try:
        logger.info('Enabling memory %s', memory_id)
        set_memory_enabled(memory_id, True)
        return {'status': 'enabled'}
    except Exception as e:
        logger.error('Failed to enable memory %s: %s', memory_id, e, exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to enable memory')


@router.patch('/{memory_id}/disable')
def disable_memory(memory_id: str):
    """
    Disable a memory without deleting it.

    """
    try:
        logger.info('Disabling memory %s', memory_id)
        set_memory_enabled(memory_id, False)
        return {'status': 'disabled'}
    except Exception as e:
        logger.error('Failed to disable memory %s: %s', memory_id, e, exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to disable memory')


@router.delete('/{memory_id}')
def remove_memory(memory_id: str):
    """
    Permanently delete a memory item.

    """
    try:
        logger.info('Deleting memory %s', memory_id)
        delete_memory(memory_id)
    except ValueError:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail='Invalid memory id')
    except Exception as e:
        logger.error('Failed to delete memory %s: %s', memory_id, e, exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to delete memory')


@router.get('/search')
def search(
    q: str,
    limit: int = None,
    session_id: str = Query(...),
):
    """
    Semantic search for memories using embeddings.

    Finds memories related to query using vector similarity.
    More accurate than keyword search for finding relevant context.

    """
    try:
        if not session_id:
            raise HTTPException(status_code=HTTP_BAD_REQUEST, detail='session_id is required')
        logger.info('Searching memories for session %s (limit=%s)', session_id, limit)
        return search_memories(session_id, q, limit)
    except Exception as e:
        logger.error('Failed to search memories: %s', e, exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to search memories')


@router.post('/reindex')
async def reindex():
    """
    Reconcile the vector store with Mongo (manual trigger).

    Re-upserts all enabled, non-deprecated memories into the vector store.

    """
    try:
        logger.info('Manual memory reindex triggered')
        count = await asyncio.to_thread(reindex_memories)
        return {'reindexed': count}
    except Exception as e:
        logger.error('Failed to reindex memories: %s', e, exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to reindex memories')


@router.post('/compress')
async def compress(
    model: str,
    session_id: str = Query(...),
):
    """
    Compress and consolidate memories for a session.

    """
    try:
        if not session_id:
            raise HTTPException(status_code=HTTP_BAD_REQUEST, detail='session_id is required')
        logger.info('Compressing memories for session %s with model %s', session_id, model)
        result = await compress_memories(session_id, model)
        return result or {'status': 'skipped'}
    except Exception as e:
        logger.error('Failed to compress memories: %s', e, exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to compress memories')
