"""
Chat Session Service

Handles all CRUD operations for chat sessions

"""

import logging
import uuid
from datetime import datetime

from app.core.db import sessions_collection
from app.models.dto import RulesConfig
from app.services.file_extraction_service import delete_file_attachments_for_session
from app.services.memory_service import delete_memories_for_session

logger = logging.getLogger(__name__)


def create_session(title: str) -> dict:
    """
    Create a new chat session with default rules.

    """
    now = datetime.utcnow()
    default_rules = RulesConfig().model_dump()

    session = {
        'id': str(uuid.uuid4()),
        'title': title,
        'messages': [],
        'created_at': now,
        'updated_at': now,
        'rules': default_rules,
    }

    sessions_collection.insert_one(session)

    return {
        'id': session['id'],
        'title': session['title'],
        'messages': [],
        'created_at': session['created_at'],
        'updated_at': session['updated_at'],
        'rules': session['rules'],
    }


def get_session(session_id: str) -> dict | None:
    """
    Retrieve a session by ID.

    """
    session = sessions_collection.find_one({'id': session_id}, {'_id': 0, 'pending_attachment': 0})
    return session


def get_session_rules(session_id: str) -> dict:
    """
    Get session-specific rules. Always returns a dict.

    """
    try:
        session = sessions_collection.find_one(
            {'id': session_id},
            {'_id': 0, 'rules': 1},
        )
        return session.get('rules', {}) if session else {}
    except Exception as e:
        logger.error(f'Error fetching session rules for {session_id}: {e}')
        return {}


def list_sessions() -> list[dict]:
    """
    List all sessions, ordered according to user's preference.

    """
    cursor = sessions_collection.find(
        {'id': {'$ne': '__order__'}},
        {
            '_id': 0,
            'id': 1,
            'title': 1,
            'updated_at': 1,
            'topic_break_at': 1,
            'topic_summary': 1,
        },
    )

    sessions = list(cursor)

    # Check for custom ordering
    order_doc = sessions_collection.find_one({'id': '__order__'}, {'_id': 0})
    if order_doc and 'sessionIds' in order_doc:
        ordered_ids = order_doc['sessionIds']
        session_map = {s['id']: s for s in sessions}

        result = []
        for sid in ordered_ids:
            if sid in session_map:
                result.append(session_map[sid])

        # Add any sessions not in order list
        for s in sessions:
            if s['id'] not in ordered_ids:
                result.append(s)

        return result

    # Default: sort by updated_at descending
    return sorted(sessions, key=lambda s: s.get('updated_at', ''), reverse=True)


def rename_session(session_id: str, title: str) -> dict | None:
    """
    Rename a session.

    """
    result = sessions_collection.update_one(
        {'id': session_id}, {'$set': {'title': title, 'updated_at': datetime.utcnow()}}
    )

    if result.matched_count == 0:
        return None

    return {'id': session_id, 'title': title}


def delete_session(session_id: str) -> bool:
    """
    Delete a session and all associated data.

    Also deletes:
    - Associated memories
    - File attachments
    - Removes from session order

    """
    result = sessions_collection.delete_one({'id': session_id})
    if result.deleted_count != 1:
        return False

    # Remove from session order
    sessions_collection.update_one(
        {'id': '__order__'},
        {'$pull': {'sessionIds': session_id}},
    )

    # Delete associated data
    delete_memories_for_session(session_id)
    delete_file_attachments_for_session(session_id)

    # Delete semantic-history vectors (best effort)
    try:
        from app.vector import get_vector_store

        get_vector_store('history').delete_by_session(session_id)
    except Exception:
        pass

    return True
