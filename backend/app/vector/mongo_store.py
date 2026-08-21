"""
Mongo Vector Store
Default vector store: embeddings live inside the memory documents in MongoDB,
similarity is computed with cosine similarity in Python (same behavior as the
original memory_service implementation)

"""

import logging
from typing import Optional

from app.core.db import synthesized_memory_collection
from app.services.embedding_service import cosine_similarity
from app.vector.base import VectorStore

logger = logging.getLogger(__name__)


class MongoVectorStore(VectorStore):
    """Vector store backed by the memories collection in MongoDB"""

    def upsert(self, id: str, vector: list[float], payload: dict) -> None:
        # No-op: the embedding is already stored inside the Mongo memory doc
        return None

    def search(
        self, vector: list[float], limit: int, filter: Optional[dict] = None
    ) -> list[dict]:
        """
        Fetch candidate docs from Mongo and score them with cosine similarity.

        Returns [{'id', 'score', 'payload'}] where payload is the full Mongo doc.

        """
        query = {
            'is_deprecated': {'$ne': True},
            '$or': [{'enabled': True}, {'enabled': {'$exists': False}}],
            'embedding': {'$exists': True},
        }
        if filter and filter.get('session_id'):
            query['session_id'] = filter['session_id']

        cursor = synthesized_memory_collection.find(query).limit(limit)

        results = []
        for doc in cursor:
            try:
                score = cosine_similarity(vector, doc['embedding'])
                results.append(
                    {'id': doc.get('id') or str(doc.get('_id')), 'score': score, 'payload': doc}
                )
            except Exception:
                continue
        return results

    def delete(self, id: str) -> None:
        # No-op: the Mongo doc (with its embedding) is deleted by memory_service
        return None

    def set_enabled(self, id: str, enabled: bool) -> None:
        # No-op: the enabled flag lives on the Mongo doc, updated by memory_service
        return None

    def delete_by_session(self, session_id: str) -> None:
        # No-op: the Mongo docs are deleted by memory_service
        return None
