"""
Rules API

"""

import logging

from fastapi import APIRouter, HTTPException

from app.config.constants import HTTP_INTERNAL_ERROR
from app.config.settings import settings
from app.core.db import rules_collection, sessions_collection
from app.models.dto import RulesConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/rules', tags=['rules'])


@router.get('', response_model=RulesConfig)
async def get_rules():
    """
    Get global default rules that apply to all sessions.

    """
    try:
        logger.info('Fetching global rules')
        rules = rules_collection.find_one({}, {'_id': 0})

        if not rules:
            default_rules = RulesConfig()
            return default_rules

        return RulesConfig(**rules)

    except Exception as e:
        logger.error(f'Error fetching rules: {e}', exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to fetch rules')


@router.put('', response_model=RulesConfig)
async def update_rules(rules: RulesConfig):
    """
    Update global default rules.

    """
    try:
        rules_dict = rules.model_dump()
        rules_collection.update_one({}, {'$set': rules_dict}, upsert=True)
        logger.info(f'Rules updated: {rules_dict}')
        return rules

    except Exception as e:
        logger.error(f'Error updating rules: {e}', exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to update rules')


@router.get('/client-config')
def get_client_config():
    """
    Get client-side configuration from backend.

    Frontend calls this during app initialization to synchronize settings.
    This ensures frontend file upload limits match backend validation rules.

    NOTE: This static route MUST be declared before the parameterized
    /{session_id} route, otherwise FastAPI would match 'client-config'
    as a session_id and this endpoint would be unreachable.

    """
    try:
        logger.info('Fetching client config')
        # Sync frontend limits with backend settings
        return {
            'fileUpload': {
                'maxSizeMB': settings.FILE_UPLOAD_MAX_SIZE_MB,
                'allowedBinaryExtensions': settings.FILE_UPLOAD_ALLOWED_EXTENSIONS,
                'maxExtractChars': settings.FILE_UPLOAD_MAX_CHARS,
            }
        }
    except Exception as e:
        logger.error(f'Error fetching client config: {e}', exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to fetch client config')


@router.get('/{session_id}', response_model=RulesConfig)
async def get_session_rules(session_id: str):
    """
    Get rules for a specific session.

    """
    try:
        logger.info(f'Fetching session rules for {session_id}')
        session = sessions_collection.find_one({'id': session_id}, {'_id': 0})

        if not session:
            logger.warning(f'Session not found: {session_id}')
            return RulesConfig()

        rules = session.get('rules', {})
        logger.debug(f'Session id {session_id}')
        return RulesConfig(**rules) if rules else RulesConfig()

    except Exception as e:
        logger.error(f'Error fetching session rules: {e}', exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to fetch session rules')


@router.put('/{session_id}', response_model=RulesConfig)
async def update_session_rules(session_id: str, rules: RulesConfig):
    """
    Update rules for a specific session.

    """
    try:
        rules_dict = rules.model_dump()
        logger.info(f'Updating session rules for {session_id} with: {rules_dict}')

        result = sessions_collection.update_one(
            {'id': session_id},
            {'$set': {'rules': rules_dict}},
        )

        logger.info(
            f'Update result - matched: {result.matched_count}, modified: {result.modified_count}'
        )

        if result.matched_count == 0:
            logger.warning(f'Session not found: {session_id}')
            raise ValueError('Session not found')

        if result.modified_count == 0:
            logger.warning(f'No changes made to session {session_id}')

        logger.info(f'Session rules updated for {session_id}: {rules_dict}')

        # Verify the update by reading back
        updated_session = sessions_collection.find_one({'id': session_id}, {'_id': 0})
        logger.info(
            f'Verification - Updated session rules from DB: {updated_session.get("rules") if updated_session else "Session not found"}'
        )

        return rules

    except Exception as e:
        logger.error(f'Error updating session rules: {e}', exc_info=True)
        raise HTTPException(
            status_code=HTTP_INTERNAL_ERROR, detail='Failed to update session rules'
        )
