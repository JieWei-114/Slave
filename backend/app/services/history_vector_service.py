"""
History Vector Service
Embeds chat messages on save and retrieves semantically relevant older
messages (beyond the recency window) for prompt augmentation.

Storage lives in the 'history' vector-store namespace:
- Qdrant:  settings.QDRANT_HISTORY_COLLECTION
- Mongo:   dedicated 'message_vectors' collection (separate from the message
           docs so the sessions collection is never bloated with embeddings)

All functions are synchronous — async callers wrap them in asyncio.to_thread.
Failures are logged and swallowed by callers: semantic history must never
block or break the chat flow.

"""

import logging
from datetime import datetime

from app.config.settings import settings
from app.core.db import sessions_collection
from app.services.embedding_service import embed
from app.vector import get_vector_store

logger = logging.getLogger(__name__)

CONTENT_MAX = settings.HISTORY_VECTOR_CONTENT_MAX_CHARS


def _created_at_iso(created_at) -> str:
    if isinstance(created_at, datetime):
        return created_at.isoformat()
    return str(created_at or '')


def index_message(
    session_id: str,
    message_id: str,
    role: str,
    content: str,
    created_at,
) -> None:
    """
    Embed a message (truncated to ~1000 chars) and upsert it into the
    history vector store. Raises on failure — callers decide how to handle.

    """
    text = (content or '').strip()[:CONTENT_MAX]
    if not text:
        return

    vector = embed([text])[0]
    get_vector_store('history').upsert(
        message_id,
        vector,
        {
            'session_id': session_id,
            'role': role,
            'content': text,
            'created_at': _created_at_iso(created_at),
        },
    )


def search_history(
    session_id: str,
    query: str,
    exclude_ids: set[str] | None = None,
    exclude_created_ats: set[str] | None = None,
) -> list[dict]:
    """
    Embed `query` and search the session's history vectors, excluding
    messages already present in the recency window (by id or created_at).

    Returns up to HISTORY_VECTOR_TOP_K hits above HISTORY_VECTOR_THRESHOLD:
    [{'role', 'content', 'created_at'}]

    """
    query = (query or '').strip()
    if not query:
        return []

    exclude_ids = exclude_ids or set()
    exclude_created_ats = exclude_created_ats or set()

    vector = embed([query[:CONTENT_MAX]])[0]
    # Over-fetch so hits removed by the recency-window exclusion still leave
    # enough candidates for the top-k cut.
    fetch_limit = settings.HISTORY_VECTOR_TOP_K + len(exclude_ids) + 4
    hits = get_vector_store('history').search(
        vector, limit=fetch_limit, filter={'session_id': session_id}
    )

    results = []
    for hit in hits:
        if hit.get('score', 0.0) < settings.HISTORY_VECTOR_THRESHOLD:
            continue
        payload = hit.get('payload') or {}
        if hit.get('id') in exclude_ids:
            continue
        if payload.get('created_at') and payload['created_at'] in exclude_created_ats:
            continue
        results.append(
            {
                'role': payload.get('role', ''),
                'content': payload.get('content', ''),
                'created_at': payload.get('created_at', ''),
            }
        )
        if len(results) >= settings.HISTORY_VECTOR_TOP_K:
            break
    return results


def backfill_history_vectors(max_messages: int | None = None) -> int:
    """
    Embed + upsert messages that are not yet in the history vector store.

    Iterates sessions (most recently updated first), bounded to the
    `max_messages` most recent messages in total. Messages without a stable
    'id' are skipped (ids are added at save time going forward). Returns the
    number of messages indexed.

    """
    if max_messages is None:
        max_messages = settings.HISTORY_VECTOR_BACKFILL_MAX

    store = get_vector_store('history')
    scanned = 0
    indexed = 0

    cursor = sessions_collection.find(
        {'messages': {'$exists': True, '$ne': []}},
        {'_id': 0, 'id': 1, 'messages': 1},
    ).sort('updated_at', -1)

    for session in cursor:
        if scanned >= max_messages:
            break
        session_id = session.get('id')
        if not session_id or session_id == '__order__':
            continue

        messages = [m for m in session.get('messages', []) if m.get('id')]
        # Most recent messages first, within the global bound
        messages = messages[-(max_messages - scanned):]
        scanned += len(messages)
        if not messages:
            continue

        try:
            existing = store.existing_ids([m['id'] for m in messages])
        except Exception as e:
            logger.warning('History backfill: existing-id check failed: %s', e)
            existing = set()

        for msg in messages:
            if msg['id'] in existing:
                continue
            try:
                index_message(
                    session_id=session_id,
                    message_id=msg['id'],
                    role=msg.get('role', ''),
                    content=msg.get('content', ''),
                    created_at=msg.get('created_at'),
                )
                indexed += 1
            except Exception as e:
                logger.warning(
                    'History backfill: failed to index message %s: %s', msg['id'], e
                )

    logger.info('History vector backfill complete: %s messages indexed', indexed)
    return indexed
