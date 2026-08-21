"""
Voice API (local STT + TTS)

Speech-to-text via faster-whisper and text-to-speech via Piper, both running
fully on-device (privacy-first — no cloud calls). Model inference is blocking,
so endpoints offload to a thread via asyncio.to_thread.
"""

import asyncio
import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.config.constants import HTTP_BAD_REQUEST, HTTP_INTERNAL_ERROR
from app.config.settings import settings
from app.services.voice_service import stt_ready, synthesize, transcribe, tts_ready

router = APIRouter(prefix='/voice', tags=['voice'])
logger = logging.getLogger(__name__)

HTTP_SERVICE_UNAVAILABLE = 503

HTTP_PAYLOAD_TOO_LARGE = 413

TRANSCRIBE_MAX_BYTES = 25 * 1024 * 1024  # 25MB audio upload limit
UPLOAD_CHUNK_BYTES = 1024 * 1024  # Stream uploads to disk in 1MB chunks
SPEAK_MAX_CHARS = 5000

ALLOWED_AUDIO_EXTENSIONS = ('.webm', '.wav', '.mp3', '.m4a', '.ogg')


def _ensure_voice_enabled() -> None:
    if not settings.VOICE_ENABLED:
        raise HTTPException(
            status_code=HTTP_SERVICE_UNAVAILABLE,
            detail='Voice features are disabled (VOICE_ENABLED=false)',
        )


@router.get('/config')
async def get_voice_config():
    """Voice feature status. Does NOT trigger model loads/downloads."""
    return {
        'enabled': settings.VOICE_ENABLED,
        'stt_model': settings.WHISPER_MODEL,
        'tts_voice': settings.PIPER_VOICE,
        'stt_ready': stt_ready(),
        'tts_ready': tts_ready(),
    }


@router.post('/transcribe')
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file (webm, wav, mp3, m4a, ogg) to text.

    Returns {text, language, duration}.

    """
    _ensure_voice_enabled()

    filename = file.filename or ''
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail=f'Audio type not allowed. Allowed: {", ".join(ALLOWED_AUDIO_EXTENSIONS)}',
        )

    tmp_path = None
    try:
        # Stream the upload to disk in chunks — never buffer the whole file
        # in RAM, and abort as soon as the size cap is exceeded.
        total_bytes = 0
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, 'wb') as tmp_file:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > TRANSCRIBE_MAX_BYTES:
                    raise HTTPException(
                        status_code=HTTP_PAYLOAD_TOO_LARGE,
                        detail=f'Audio too large. Max {TRANSCRIBE_MAX_BYTES // (1024 * 1024)}MB.',
                    )
                tmp_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=HTTP_BAD_REQUEST, detail='Empty audio file')

        logger.info('Transcribing %s (%d bytes)', filename, total_bytes)
        result = await asyncio.to_thread(transcribe, tmp_path)
        logger.info('Transcription done: %d chars', len(result['text']))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Transcription failed for {filename}: {e}', exc_info=True)
        raise HTTPException(
            status_code=HTTP_INTERNAL_ERROR, detail=f'Transcription failed: {str(e)}'
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


class SpeakRequest(BaseModel):
    """JSON body for the speech synthesis endpoint."""

    text: str


@router.post('/speak')
async def speak(payload: SpeakRequest):
    """Synthesize text to speech; returns audio/wav bytes."""
    _ensure_voice_enabled()

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail='text is required')
    if len(text) > SPEAK_MAX_CHARS:
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail=f'Text too long. Max {SPEAK_MAX_CHARS} characters.',
        )

    try:
        logger.info('Synthesizing speech: %d chars', len(text))
        wav_bytes = await asyncio.to_thread(synthesize, text)
        return Response(content=wav_bytes, media_type='audio/wav')
    except Exception as e:
        logger.error(f'Speech synthesis failed: {e}', exc_info=True)
        raise HTTPException(
            status_code=HTTP_INTERNAL_ERROR, detail=f'Speech synthesis failed: {str(e)}'
        )
