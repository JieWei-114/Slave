"""
Chat API

"""

import json
import logging
from datetime import datetime
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config.ai_models import AVAILABLE_MODELS
from app.config.constants import HTTP_BAD_REQUEST, HTTP_INTERNAL_ERROR, HTTP_NOT_FOUND
from app.config.settings import settings
from app.core.db import sessions_collection
from app.models.dto import (
    AttachFileRequest,
    CreateSessionRequest,
    RenameSessionRequest,
    ReorderSessionsRequest,
)
from app.services.chat_service import (
    create_session,
    delete_session,
    get_session,
    list_sessions,
    rename_session,
    stream_chat_reply,
)
from app.services.file_extraction_service import (
    delete_file_attachment_for_session,
    extract_text_from_file,
    list_file_attachments,
    store_file_attachment,
    truncate_content,
)

router = APIRouter(prefix='/chat', tags=['chat'])
logger = logging.getLogger(__name__)


def detect_file_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith('.pdf'):
        return 'PDF'
    if lower.endswith(('.docx', '.doc')):
        return 'Word'
    if lower.endswith(('.txt', '.md')):
        return 'Text'
    if lower.endswith(('.json', '.yaml', '.yml')):
        return 'Config'
    if lower.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
        return 'Code'
    return 'unknown'


@router.get('/models')
async def get_available_models():
    logger.info('Fetching available models')
    return AVAILABLE_MODELS


@router.get('/sessions')
def get_sessions():
    try:
        sessions = list_sessions()
        logger.info(f'Retrieved {len(sessions)} sessions')
        return sessions
    except Exception as e:
        logger.error(f'Failed to get sessions: {e}')
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to retrieve sessions')


@router.post('/session')
def create_chat_session(payload: CreateSessionRequest):
    try:
        logger.info(f'Creating session with title: {payload.title}')
        session = create_session(payload.title)
        logger.info(f'Session created: {session["id"]}')
        return session
    except Exception as e:
        logger.error(f'Failed to create session: {e}')
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to create session')


@router.get('/{session_id}')
def get_chat_session(session_id: str):
    try:
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=HTTP_NOT_FOUND, detail='Session not found')
        logger.info(f'Retrieved session: {session_id}')
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to get session {session_id}: {e}')
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to retrieve session')


class StreamMessageRequest(BaseModel):
    """JSON body for the chat streaming endpoint."""

    content: str
    model: str
    reasoning: bool = False


@router.post('/{session_id}/stream')
async def stream_message(session_id: str, payload_in: StreamMessageRequest):
    """
    Stream chat response using Server-Sent Events (SSE).

    POST with JSON body {content, model, reasoning}; responds with
    text/event-stream real-time token streaming.

    """
    content = payload_in.content
    model = payload_in.model
    reasoning = payload_in.reasoning
    try:
        logger.info(
            f'Streaming message for session {session_id}, content_len={len(content)}, model={model}, reasoning={reasoning}'
        )

        async def event_generator():
            try:
                async for chunk in stream_chat_reply(
                    session_id,
                    content,
                    model,
                    reasoning_enabled=reasoning,
                ):
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode('utf-8')

                    chunk = chunk.strip()
                    if not chunk:
                        continue

                    try:
                        payload = json.loads(chunk)
                        event_type = payload.get('type')

                        if event_type == 'done':
                            yield (f'event: done\ndata: {json.dumps(payload)}\n\n')

                        elif event_type == 'token':
                            yield (f'event: token\ndata: {json.dumps(payload["data"])}\n\n')

                        elif event_type == 'reasoning_token':
                            yield (
                                f'event: reasoning_token\ndata: {json.dumps(payload["data"])}\n\n'
                            )

                        elif event_type == 'error':
                            yield (f'event: error\ndata: {json.dumps(payload)}\n\n')

                        elif event_type:
                            # Generic passthrough for all other event types
                            # (answer_complete, verification_starting,
                            # verification_complete, reasoning_starting, ...)
                            yield (f'event: {event_type}\ndata: {json.dumps(payload)}\n\n')

                    except json.JSONDecodeError:
                        # normal token
                        yield f'data: {quote(chunk)}\n\n'
            except Exception as e:
                logger.error(f'Error in stream for session {session_id}: {e}')
                yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'

        logger.info(f'Stream started for session {session_id}')
        return StreamingResponse(
            event_generator(),
            media_type='text/event-stream',
        )
    except Exception as e:
        logger.error(f'Failed to stream message for session {session_id}: {e}')
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to stream message')


@router.get('/{session_id}/messages')
def get_messages(session_id: str, limit: int = None, before: Optional[str] = None):
    if limit is None:
        limit = settings.API_MESSAGES_DEFAULT_LIMIT

    session = sessions_collection.find_one({'id': session_id}, {'_id': 0, 'messages': 1})

    if not session:
        return []

    messages = session.get('messages', [])

    messages = sorted(messages, key=lambda m: m['created_at'])

    if before:
        try:
            before_dt = datetime.fromisoformat(before.replace('Z', ''))
        except ValueError:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Invalid 'before' timestamp: must be ISO 8601 format",
            )
        messages = [m for m in messages if m['created_at'] < before_dt]

    return messages[-limit:]


@router.patch('/{session_id}/rename')
def rename_chat_session(session_id: str, payload: RenameSessionRequest):
    try:
        logger.info(f'Renaming session {session_id} to: {payload.title}')
        updated = rename_session(session_id, payload.title)
        if not updated:
            raise HTTPException(status_code=HTTP_NOT_FOUND, detail='Session not found')
        logger.info(f'Session renamed: {session_id}')
        return updated
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to rename session {session_id}: {e}')
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to rename session')


@router.delete('/{session_id}')
def delete_chat_session(session_id: str):
    try:
        logger.info(f'Deleting session: {session_id}')
        ok = delete_session(session_id)
        if not ok:
            raise HTTPException(status_code=HTTP_NOT_FOUND, detail='Session not found')
        logger.info(f'Session deleted: {session_id}')
        return {'status': 'deleted'}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to delete session {session_id}: {e}')
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to delete session')


@router.post('/reorder')
def reorder_sessions(payload: ReorderSessionsRequest):
    try:
        sessions_collection.update_one(
            {'id': '__order__'},
            {
                '$set': {
                    'id': '__order__',
                    'sessionIds': payload.sessionIds,
                    'updated_at': datetime.utcnow(),
                }
            },
            upsert=True,
        )
        return {'status': 'reordered'}

    except Exception as e:
        raise HTTPException(
            status_code=HTTP_INTERNAL_ERROR, detail=f'Failed to reorder sessions: {str(e)}'
        )


@router.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and extract content from files (PDF, Word, text files).
    Returns extracted text content.
    """
    try:
        logger.info(f'Uploading file: {file.filename}')

        # Enforce allowed extensions server-side
        allowed_exts = tuple(ext.lower() for ext in settings.FILE_UPLOAD_ALLOWED_EXTENSIONS)
        if not (file.filename or '').lower().endswith(allowed_exts):
            logger.warning(f'File {file.filename} rejected: extension not allowed')
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail=f'File type not allowed. Allowed extensions: {", ".join(allowed_exts)}',
            )

        # Read file content
        file_content = await file.read()

        # Check file size
        max_size = settings.FILE_UPLOAD_MAX_SIZE_MB * 1024 * 1024
        if len(file_content) > max_size:
            logger.warning(f'File {file.filename} too large: {len(file_content)} bytes')
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail=f'File too large. Max {settings.FILE_UPLOAD_MAX_SIZE_MB}MB.',
            )

        # Extract text content
        try:
            extracted_text = extract_text_from_file(file_content, file.filename)
        except ValueError as e:
            logger.error(f'Failed to extract text from {file.filename}: {e}')
            raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=str(e))

        # Truncate if too long (based on settings)
        truncated_text = truncate_content(extracted_text, max_chars=settings.FILE_UPLOAD_MAX_CHARS)
        logger.info(f'File {file.filename} extracted: {len(truncated_text)} chars')

        return {
            'content': truncated_text,
            'filename': file.filename,
            'original_size': len(file_content),
            'extracted_length': len(truncated_text),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to process file {file.filename}: {e}')
        raise HTTPException(
            status_code=HTTP_INTERNAL_ERROR, detail=f'Failed to process file: {str(e)}'
        )


@router.post('/{session_id}/attachment')
def attach_file(session_id: str, payload: AttachFileRequest):
    """
    Attach extracted file content to a session.
    Stores in file_attachments collection for persistence.
    """
    try:
        session = sessions_collection.find_one({'id': session_id}, {'_id': 0, 'id': 1})
        if not session:
            raise HTTPException(status_code=HTTP_NOT_FOUND, detail='Session not found')

        filename = payload.filename.strip()
        content = payload.content.strip() if payload.content else ''

        if not filename or not content:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST, detail='filename and content are required'
            )

        truncated = truncate_content(content)
        logger.info(f'Attaching file {filename} to session {session_id}, length={len(truncated)}')

        # Detect file type and store in persistent collection
        file_type = detect_file_type(filename)
        file_record = store_file_attachment(
            session_id=session_id,
            filename=filename,
            content=truncated,
            file_type=file_type,
        )

        logger.info(f'File attached and stored: {filename} (ID: {file_record["id"]})')
        return {
            'status': 'attached',
            'file_id': file_record['id'],
            'filename': filename,
            'length': len(truncated),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to attach file to session {session_id}: {e}')
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to attach file')


@router.get('/{session_id}/files')
def list_files(session_id: str):
    """List file attachments for a session"""
    try:
        return list_file_attachments(session_id)
    except Exception as e:
        logger.error(f'Failed to list files for session {session_id}: {e}', exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to list files')


@router.delete('/{session_id}/files/{file_id}')
def delete_file(session_id: str, file_id: str):
    """Delete a file attachment for a session"""
    try:
        ok = delete_file_attachment_for_session(session_id, file_id)
        if not ok:
            raise HTTPException(status_code=HTTP_NOT_FOUND, detail='File not found')
        return {'status': 'deleted'}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to delete file {file_id}: {e}', exc_info=True)
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail='Failed to delete file')
