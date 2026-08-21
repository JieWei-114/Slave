"""
Qdrant Vector Store
Vector store backed by a Qdrant server (settings.QDRANT_URL).
MongoDB remains the source of truth for memory metadata; Qdrant only stores
the embedding plus a small payload used for filtering

"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from app.config.settings import settings
from app.vector.base import VectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant"""

    def __init__(self, collection: Optional[str] = None, filter_flags: bool = True):
        """
        collection: Qdrant collection name (default: settings.QDRANT_COLLECTION).
        filter_flags: when True, searches only enabled/non-deprecated points
        (memory semantics). History vectors have no such flags -> False.

        """
        self._client = QdrantClient(url=settings.QDRANT_URL)
        self._collection = collection or settings.QDRANT_COLLECTION
        self._filter_flags = filter_flags
        self._ensured = False

    def _ensure_collection(self) -> None:
        # Auto-create the collection on first use
        if self._ensured:
            return
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(
                'Qdrant collection created: %s (dim=%s)',
                self._collection,
                settings.EMBEDDING_DIM,
            )
        self._ensured = True

    def _build_filter(self, filter: Optional[dict] = None) -> qmodels.Filter:
        # Only enabled, non-deprecated memories are searchable (memory namespace)
        must = []
        if self._filter_flags:
            must = [
                qmodels.FieldCondition(key='enabled', match=qmodels.MatchValue(value=True)),
                qmodels.FieldCondition(
                    key='is_deprecated', match=qmodels.MatchValue(value=False)
                ),
            ]
        if filter and filter.get('session_id'):
            must.append(
                qmodels.FieldCondition(
                    key='session_id', match=qmodels.MatchValue(value=filter['session_id'])
                )
            )
        return qmodels.Filter(must=must)

    def upsert(self, id: str, vector: list[float], payload: dict) -> None:
        self._ensure_collection()
        self._client.upsert(
            collection_name=self._collection,
            points=[qmodels.PointStruct(id=id, vector=vector, payload=payload)],
        )

    def search(
        self, vector: list[float], limit: int, filter: Optional[dict] = None
    ) -> list[dict]:
        self._ensure_collection()
        hits = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            query_filter=self._build_filter(filter),
            with_payload=True,
        ).points
        return [
            {'id': str(hit.id), 'score': float(hit.score), 'payload': hit.payload or {}}
            for hit in hits
        ]

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of ids already stored (used by history backfill)."""
        if not ids:
            return set()
        self._ensure_collection()
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        return {str(p.id) for p in points}

    def delete(self, id: str) -> None:
        self._ensure_collection()
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.PointIdsList(points=[id]),
        )

    def set_enabled(self, id: str, enabled: bool) -> None:
        self._ensure_collection()
        self._client.set_payload(
            collection_name=self._collection,
            payload={'enabled': enabled},
            points=[id],
        )

    def delete_by_session(self, session_id: str) -> None:
        self._ensure_collection()
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key='session_id', match=qmodels.MatchValue(value=session_id)
                        )
                    ]
                )
            ),
        )
