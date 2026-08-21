"""
Vector Store Factory
Selects the active vector store from settings.VECTOR_STORE

"""

import logging

from app.config.settings import settings
from app.vector.base import VectorStore

logger = logging.getLogger(__name__)

__all__ = ['VectorStore', 'get_vector_store']

_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Get the configured vector store (cached singleton).

    settings.VECTOR_STORE: 'mongo' (default) | 'qdrant'

    """
    global _store
    if _store is None:
        name = (settings.VECTOR_STORE or 'mongo').strip().lower()
        if name == 'qdrant':
            from app.vector.qdrant_store import QdrantVectorStore

            _store = QdrantVectorStore()
        elif name == 'mongo':
            from app.vector.mongo_store import MongoVectorStore

            _store = MongoVectorStore()
        else:
            logger.warning("Unknown VECTOR_STORE '%s', falling back to mongo", name)
            from app.vector.mongo_store import MongoVectorStore

            _store = MongoVectorStore()
        logger.info('Vector store initialized: %s', type(_store).__name__)
    return _store
