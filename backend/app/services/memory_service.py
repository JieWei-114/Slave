"""
Memory Service

Handles storage, retrieval, and management of user memories (facts, preferences, file references).
Uses semantic search with embeddings for intelligent context matching

"""

import asyncio
import logging
import uuid
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId

from app.config.settings import settings
from app.core.db import (
    sessions_collection,
    synthesized_memory_collection,
)
from app.services.embedding_service import embed
from app.services.ollama_service import call_ollama_once
from app.vector import get_vector_store

logger = logging.getLogger(__name__)

# Import centralized settings
SEARCH_LIMIT = settings.MEMORY_SEARCH_LIMIT
SEARCH_THRESHOLD = settings.MEMORY_SEARCH_THRESHOLD
MAX_CHARS_PER_ITEM = settings.MEMORY_MAX_CHARS_PER_ITEM
MEMORY_DB_QUERY_LIMIT = settings.MEMORY_DB_QUERY_LIMIT
MEMORY_TEXT_FALLBACK_LIMIT = settings.MEMORY_TEXT_FALLBACK_LIMIT
MEMORY_KEY_TRUNCATION_LIMIT = settings.MEMORY_KEY_TRUNCATION_LIMIT
MEMORY_LOG_TRUNCATION_LIMIT = settings.MEMORY_LOG_TRUNCATION_LIMIT


def serialize_memory(doc: dict) -> dict:
    # Convert synthesized memory document to API-friendly format
    return {
        'id': doc.get('id') or str(doc.get('_id')),
        'content': doc.get('value') or doc.get('content') or doc.get('fact') or '',
        'enabled': doc.get('enabled', True),
        'created_at': doc.get('created_at'),
        'chat_sessionId': doc.get('session_id') or doc.get('chat_sessionId'),
        'category': doc.get('category', 'other'),
        'source': doc.get('source', 'manual'),
        'confidence': doc.get('confidence', settings.MEMORY_DEFAULT_CONFIDENCE),
    }


def _vector_payload(memory: dict) -> dict:
    # Build the payload stored alongside the vector (Mongo stays source of truth)
    return {
        'session_id': memory.get('session_id'),
        'enabled': memory.get('enabled', True),
        'is_deprecated': memory.get('is_deprecated', False),
        'value': memory.get('value', ''),
        'category': memory.get('category', 'other'),
        'source': memory.get('source', 'manual'),
    }


def get_session_memory_limit(session_id: str) -> int | None:
    """
    Get custom memory search limit for a session.

    Sessions can override default memory limit via rules.memorySearchLimit.
    Used to control how many memories are returned in search results.

    """
    try:
        session = sessions_collection.find_one({'id': session_id}, {'rules.memorySearchLimit': 1})
        if session and session.get('rules'):
            return session['rules'].get('memorySearchLimit')
        return None
    except Exception as e:
        logger.error(f'Error fetching memory limit for session {session_id}: {e}')
        return None


def add_memory(
    content: str,
    chat_sessionId: str,
    source: str = 'manual',
    category: str = 'other',
):
    if not content or not content.strip():
        raise ValueError('Content cannot be empty')

    if settings.MEMORY_MAX_CONTENT_LENGTH and len(content) > settings.MEMORY_MAX_CONTENT_LENGTH:
        content = content[: settings.MEMORY_MAX_CONTENT_LENGTH]

    try:
        embedding = embed([content])[0]
    except Exception as e:
        raise ValueError(f'Failed to generate embedding: {e}')

    memory = {
        'id': str(uuid.uuid4()),
        'session_id': chat_sessionId,
        'key': content.strip()[:MEMORY_KEY_TRUNCATION_LIMIT],
        'value': content.strip(),
        'embedding': embedding,
        'source': source,
        'category': category,
        'confidence': settings.MEMORY_DEFAULT_CONFIDENCE,
        'tags': [category],
        'enabled': True,
        'is_deprecated': False,
        'created_at': datetime.utcnow(),
        'last_referenced_at': datetime.utcnow(),
    }
    result = synthesized_memory_collection.insert_one(memory)
    memory['_id'] = result.inserted_id

    try:
        get_vector_store().upsert(memory['id'], embedding, _vector_payload(memory))
    except Exception as e:
        logger.warning(f'Failed to upsert memory vector {memory["id"]}: {e}')
        # Mark for later reconciliation so the vector store can catch up
        synthesized_memory_collection.update_one(
            {'_id': memory['_id']}, {'$set': {'needs_reindex': True}}
        )

    return serialize_memory(memory)


def list_all_memories(chat_sessionId: str):
    """
    Get all memories for a session (enabled and disabled).

    Sorted by creation date ascending.
    Used for listing UI to show all stored memories.

    """
    cursor = synthesized_memory_collection.find({'session_id': chat_sessionId}).sort(
        'created_at', 1
    )
    return [serialize_memory(doc) for doc in cursor]


def set_memory_enabled(memory_id: str, enabled: bool):
    """
    Enable or disable a memory without deleting it.

    Disabled memories are not returned in searches or context selection.

    """
    # Resolve the doc first so we can propagate the uuid 'id' (not the Mongo
    # _id hex string) to the vector store — Qdrant point ids are the uuid.
    doc = synthesized_memory_collection.find_one({'id': memory_id})
    if doc is None:
        try:
            doc = synthesized_memory_collection.find_one({'_id': ObjectId(memory_id)})
        except InvalidId:
            doc = None

    if doc is None:
        raise ValueError('Invalid memory id')

    synthesized_memory_collection.update_one({'_id': doc['_id']}, {'$set': {'enabled': enabled}})

    # Propagate to the vector store (no-op for Mongo)
    vector_id = doc.get('id') or str(doc['_id'])
    try:
        get_vector_store().set_enabled(vector_id, enabled)
    except Exception as e:
        logger.warning(f'Failed to update vector enabled flag for {vector_id}: {e}')


def list_enabled_memories(chat_sessionId: str):
    """
    Get only enabled memories for a session.

    Excludes disabled memories.
    Used for context selection in chat responses.

    """
    cursor = synthesized_memory_collection.find(
        {
            'session_id': chat_sessionId,
            'is_deprecated': {'$ne': True},
            '$or': [{'enabled': True}, {'enabled': {'$exists': False}}],
        }
    ).sort('created_at', 1)

    return [serialize_memory(doc) for doc in cursor]


def list_memories_by_category(chat_sessionId: str, category: str):
    """
    Get enabled memories for a session filtered by category.

    """
    cursor = synthesized_memory_collection.find(
        {
            'session_id': chat_sessionId,
            'category': category,
            'is_deprecated': {'$ne': True},
            '$or': [{'enabled': True}, {'enabled': {'$exists': False}}],
        }
    ).sort('created_at', 1)

    return [serialize_memory(doc) for doc in cursor]


def delete_memory(memory_id: str):
    """
    Permanently delete a memory item.

    """
    # Resolve the doc first so we can propagate the uuid 'id' (not the Mongo
    # _id hex string) to the vector store — Qdrant point ids are the uuid.
    doc = synthesized_memory_collection.find_one({'id': memory_id})
    if doc is None:
        try:
            doc = synthesized_memory_collection.find_one({'_id': ObjectId(memory_id)})
        except InvalidId:
            raise ValueError('Invalid memory id')

    if doc is None:
        raise ValueError('Invalid memory id')

    synthesized_memory_collection.delete_one({'_id': doc['_id']})

    # Propagate to the vector store (no-op for Mongo)
    vector_id = doc.get('id') or str(doc['_id'])
    try:
        get_vector_store().delete(vector_id)
    except Exception as e:
        logger.warning(f'Failed to delete vector for memory {vector_id}: {e}')


def delete_memories_for_session(chat_sessionId: str) -> int:
    """
    Delete all memories linked to a chat session.

    Called when a session is deleted to clean up associated memories.

    """
    result = synthesized_memory_collection.delete_many({'session_id': chat_sessionId})

    # Propagate to the vector store (no-op for Mongo)
    try:
        get_vector_store().delete_by_session(chat_sessionId)
    except Exception as e:
        logger.warning(f'Failed to delete vectors for session {chat_sessionId}: {e}')

    return result.deleted_count


def reindex_memories() -> int:
    """
    Reconcile the vector store with Mongo (source of truth).

    Iterates all enabled, non-deprecated memories, re-upserting each into the
    vector store (re-embedding only when no embedding is stored) and clearing
    any needs_reindex flag. Returns the number of memories reindexed.

    """
    store = get_vector_store()
    count = 0

    cursor = synthesized_memory_collection.find(
        {
            'is_deprecated': {'$ne': True},
            '$or': [{'enabled': True}, {'enabled': {'$exists': False}}],
        }
    )

    for doc in cursor:
        vector_id = doc.get('id') or str(doc['_id'])
        try:
            embedding = doc.get('embedding')
            if not embedding:
                content = doc.get('value') or doc.get('content') or doc.get('fact') or ''
                if not content.strip():
                    continue
                embedding = embed([content])[0]
                synthesized_memory_collection.update_one(
                    {'_id': doc['_id']}, {'$set': {'embedding': embedding}}
                )

            store.upsert(vector_id, embedding, _vector_payload(doc))
            synthesized_memory_collection.update_one(
                {'_id': doc['_id']}, {'$unset': {'needs_reindex': ''}}
            )
            count += 1
        except Exception as e:
            logger.warning(f'Failed to reindex memory {vector_id}: {e}')

    logger.info('Memory reindex complete: %s memories upserted', count)
    return count


def search_memories(chat_sessionId: str, query: str, limit: int = None, threshold: float = None):
    """
    Search memories using semantic similarity with embeddings.

    Implementation:
    1. Embed the query text using sentence-transformers
    2. Calculate cosine similarity to all memories
    3. Filter by confidence threshold
    4. Return top results sorted by similarity

    """
    if not query or not query.strip():
        return []

    # Use session-specific limit if set, otherwise use settings default
    if limit is None:
        session_limit = get_session_memory_limit(chat_sessionId)
        limit = session_limit or settings.MEMORY_SEARCH_LIMIT

    # Use threshold from settings (can be customized later per session)
    if threshold is None:
        threshold = settings.MEMORY_SEARCH_THRESHOLD

    logger.info(
        'Memory search start (query_len=%s, limit=%s, threshold=%s)',
        len(query),
        limit,
        threshold,
        extra={'session_id': chat_sessionId},
    )

    try:
        query_vec = embed([query])[0]
    except Exception as e:
        logger.error(f'Failed to embed query: {e}')
        return []

    try:
        hits = get_vector_store().search(
            query_vec,
            limit=MEMORY_DB_QUERY_LIMIT,
            filter={'session_id': chat_sessionId},
        )
    except Exception as e:
        logger.error(f'Vector search failed: {e}')
        return []

    scored = []
    for hit in hits:
        try:
            score = hit['score']
            if score >= threshold:
                doc = hit.get('payload') or {}
                if '_id' not in doc:
                    # Payload-only hit (e.g. Qdrant) — hydrate from Mongo (source of truth)
                    doc = synthesized_memory_collection.find_one({'id': hit['id']}) or doc

                # Re-check Mongo truth: skip disabled/deprecated memories the
                # vector store may not have caught up on yet
                if not doc.get('enabled', True) or doc.get('is_deprecated'):
                    continue

                # Truncate content to max chars per item
                content = doc.get('value') or doc.get('content', '')
                if len(content) > MAX_CHARS_PER_ITEM:
                    if doc.get('value'):
                        doc['value'] = content[:MAX_CHARS_PER_ITEM] + '...'
                    else:
                        doc['content'] = content[:MAX_CHARS_PER_ITEM] + '...'

                scored.append((score, doc))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)

    results = [serialize_memory(d[1]) for d in scored[:limit]]
    logger.info(
        'Memory search results (matched=%s, returned=%s)',
        len(scored),
        len(results),
        extra={'session_id': chat_sessionId},
    )
    return results


async def compress_memories(chat_sessionId: str, model: str):
    memories = list_enabled_memories(chat_sessionId)

    if len(memories) < 2:
        return None

    text = '\n'.join(f'- {m["content"]}' for m in memories)

    prompt = f"""
Summarize the following memories into a concise.
Structured in long-term memory.
Do not lose important facts.
Do not add explanation.

Memories:
{text}
"""

    # 1. Get the summary FIRST — never touch existing memories until we know
    #    we have a valid replacement.
    try:
        summary = await call_ollama_once(prompt, model)
    except Exception:
        logger.error('Memory compression failed: LLM call raised — nothing was modified')
        return None

    if not summary or not summary.strip():
        # call_ollama_once returns '' on failure — abort without disabling anything
        logger.error('Memory compression failed: empty summary — nothing was modified')
        return None

    # 2. Insert the new compressed memory first
    compressed = await asyncio.to_thread(
        add_memory,
        summary,
        chat_sessionId,
        'compress',
        'other',
    )

    # 3. Only disable old memories AFTER the compressed memory is safely stored
    for m in memories:
        try:
            await asyncio.to_thread(set_memory_enabled, m['id'], False)
        except Exception as e:
            logger.warning(f'Failed to disable old memory {m["id"]}: {e}')

    return compressed


def should_remember(user_text: str, assistant_text: str) -> bool:
    # Decide if a conversation turn should be saved as memory
    min_assistant = settings.MEMORY_MIN_ASSISTANT_LENGTH or 0
    min_conversation = settings.MEMORY_MIN_CONVERSATION_LENGTH or 0

    if len(assistant_text.strip()) < min_assistant:
        return False

    if len(user_text) + len(assistant_text) < min_conversation:
        return False

    # Check for reject patterns (case-insensitive, whole message)
    reject_patterns = ['dont remember']
    normalized = assistant_text.lower().strip()
    if any(pattern in normalized for pattern in reject_patterns):
        return False

    # Check for explicit remember requests
    accept_patterns = ['remember', 'save this', 'keep in mind']
    if any(pattern in normalized for pattern in accept_patterns):
        return True

    # Default: do NOT remember unless a pattern/heuristic indicated we should
    return False


async def summarize(text: str, model: str) -> str:
    prompt = f"""
Summarize the following content into a single concise fact.
Do NOT add explanation.

Content:
{text}
"""
    try:
        return await call_ollama_once(prompt, model)
    except Exception:
        return text[:MEMORY_TEXT_FALLBACK_LIMIT]  # Fallback to truncation


async def auto_memory_if_needed(
    chat_sessionId: str,
    user_text: str,
    assistant_text: str,
    model: str,
):
    if not should_remember(user_text, assistant_text):
        logger.debug('Auto memory criteria not met, skipping')
        return None

    logger.info('Auto memory triggered for this conversation')

    combined = f'User: {user_text}\nAssistant: {assistant_text}'

    summary = await summarize(combined, model)

    try:
        # add_memory does blocking embedding + Mongo work — keep it off the event loop
        return await asyncio.to_thread(
            add_memory,
            summary,
            chat_sessionId,
            'auto',
            'preference_or_fact',
        )
    except Exception as e:
        logger.error(f'Failed to add auto memory: {e}')
        return None


# ============================================================
# NEW: SYNTHESIZED MEMORY FUNCTIONS (Facts & Preferences)
# ============================================================


async def add_synthesized_memory(
    session_id: str,
    fact: str,
    category: str = 'general',
    confidence: float = None,
    source: str = 'user_statement',
    tags: list[str] | None = None,
    source_file: str | None = None,
) -> dict:
    """Store a synthesized memory item (fact or preference)"""
    if confidence is None:
        confidence = settings.MEMORY_DEFAULT_CONFIDENCE
    if tags is None:
        tags = []

    embedding = None
    try:
        if fact and fact.strip():
            # Blocking sentence-transformers call — run off the event loop
            embedding = (await asyncio.to_thread(embed, [fact]))[0]
    except Exception as e:
        logger.warning(f'Failed to embed synthesized memory: {e}')

    memory_item = {
        'id': str(uuid.uuid4()),
        'session_id': session_id,
        'category': category,
        'key': fact[:MEMORY_KEY_TRUNCATION_LIMIT],  # Use beginning as key
        'value': fact,
        'confidence': min(1.0, max(0.0, confidence)),  # Clamp 0-1
        'source': source,
        'tags': tags,
        'enabled': True,
        'created_at': datetime.utcnow(),
        'last_referenced_at': datetime.utcnow(),
        'is_deprecated': False,
    }

    if embedding is not None:
        memory_item['embedding'] = embedding

    if source_file:
        memory_item['source_file'] = source_file

    result = await asyncio.to_thread(synthesized_memory_collection.insert_one, memory_item)
    memory_item['_id'] = result.inserted_id

    if embedding is not None:
        try:
            # Blocking vector store call — run off the event loop
            await asyncio.to_thread(
                get_vector_store().upsert,
                memory_item['id'],
                embedding,
                _vector_payload(memory_item),
            )
        except Exception as e:
            logger.warning(f'Failed to upsert synthesized memory vector {memory_item["id"]}: {e}')
            # Mark for later reconciliation so the vector store can catch up
            await asyncio.to_thread(
                synthesized_memory_collection.update_one,
                {'_id': memory_item['_id']},
                {'$set': {'needs_reindex': True}},
            )

    logger.info(
        f'Synthesized memory added: {fact[:MEMORY_LOG_TRUNCATION_LIMIT]}... (confidence: {confidence})'
    )
    return memory_item
