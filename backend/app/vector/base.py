"""
Vector Store Base
Abstract interface for vector storage/search backends (Mongo, Qdrant, ...)
MongoDB remains the source of truth for memory metadata; the vector store
only handles embedding storage and similarity search

"""

from abc import ABC, abstractmethod
from typing import Optional


class VectorStore(ABC):
    """
    Abstract vector store interface.

    All methods are synchronous — memory_service functions are synchronous
    and are already wrapped in asyncio.to_thread by async callers.

    """

    @abstractmethod
    def upsert(self, id: str, vector: list[float], payload: dict) -> None:
        """
        Store or update a vector with its payload.

        Payload keys used by memory_service: session_id, enabled,
        is_deprecated, value, category, source.

        """

    @abstractmethod
    def search(
        self, vector: list[float], limit: int, filter: Optional[dict] = None
    ) -> list[dict]:
        """
        Similarity search returning [{'id', 'score', 'payload'}].

        Only enabled, non-deprecated entries are returned.
        Supported filter keys: session_id.

        """

    @abstractmethod
    def delete(self, id: str) -> None:
        """Remove a vector by memory id."""

    @abstractmethod
    def set_enabled(self, id: str, enabled: bool) -> None:
        """Mark a vector as enabled/disabled (excluded from search when disabled)."""

    @abstractmethod
    def delete_by_session(self, session_id: str) -> None:
        """Remove all vectors belonging to a chat session."""
