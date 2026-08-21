"""
Vector Store Factory
Selects the active vector store from settings.VECTOR_STORE

Supports multiple namespaces so different data types live in separate
collections:
- 'memories' (default): synthesized memories (original behavior)
- 'history':  conversation-history message vectors (semantic history)

"""

import logging

from app.config.settings import settings
from app.vector.base import VectorStore

logger = logging.getLogger(__name__)

__all__ = ['VectorStore', 'get_vector_store']

_stores: dict[str, VectorStore] = {}


def get_vector_store(namespace: str = 'memories') -> VectorStore:
    """
    Get the configured vector store for a namespace (cached singleton per namespace).

    settings.VECTOR_STORE: 'mongo' (default) | 'qdrant'
    namespace: 'memories' (default) | 'history'

    """
    if namespace in _stores:
        return _stores[namespace]

    name = (settings.VECTOR_STORE or 'mongo').strip().lower()
    if name not in ('mongo', 'qdrant'):
        logger.warning("Unknown VECTOR_STORE '%s', falling back to mongo", name)
        name = 'mongo'

    if name == 'qdrant':
        from app.vector.qdrant_store import QdrantVectorStore

        if namespace == 'history':
            store = QdrantVectorStore(
                collection=settings.QDRANT_HISTORY_COLLECTION,
                filter_flags=False,  # history payloads have no enabled/is_deprecated
            )
        else:
            store = QdrantVectorStore()
    else:
        if namespace == 'history':
            from app.vector.mongo_store import MongoHistoryStore

            store = MongoHistoryStore()
        else:
            from app.vector.mongo_store import MongoVectorStore

            store = MongoVectorStore()

    _stores[namespace] = store
    logger.info(
        'Vector store initialized: %s (namespace=%s)', type(store).__name__, namespace
    )
    return store
