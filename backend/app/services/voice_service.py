"""
Voice Service (local STT + TTS)

Privacy-first speech pipeline running entirely on-device:
- STT: faster-whisper (CTranslate2 Whisper, CPU int8)
- TTS: Piper (ONNX voices from the rhasspy/piper-voices HF repo)

Models are lazy-loaded singletons; the first call downloads model files
into settings.VOICE_MODELS_DIR. Both transcribe() and synthesize() are
blocking — callers must run them via asyncio.to_thread.
"""

import io
import logging
import os
import re
import tempfile
import threading
import wave

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)

PIPER_VOICES_BASE = 'https://huggingface.co/rhasspy/piper-voices/resolve/main'
DOWNLOAD_TIMEOUT = 300.0

# Piper voice names look like '{locale}-{name}-{quality}', e.g. 'en_US-lessac-medium'
_VOICE_NAME_RE = re.compile(r'^([a-z]{2,3})_([A-Z]{2})-([a-z0-9_]+)-(x_low|low|medium|high)$')

_whisper_lock = threading.Lock()
_piper_lock = threading.Lock()
_whisper_model = None
_piper_voice = None


def _piper_voice_urls(voice_name: str) -> tuple[str, str]:
    """
    Derive the HF hub URLs for a Piper voice's .onnx and .onnx.json files.

    URL path pattern: {lang}/{locale}/{name}/{quality}/{voice_name}.onnx
    e.g. en/en_US/lessac/medium/en_US-lessac-medium.onnx

    """
    match = _VOICE_NAME_RE.match(voice_name)
    if not match:
        raise ValueError(
            f"Unsupported Piper voice name '{voice_name}': expected "
            "'{locale}-{name}-{quality}' like 'en_US-lessac-medium'"
        )
    lang, region, name, quality = match.groups()
    base = f'{PIPER_VOICES_BASE}/{lang}/{lang}_{region}/{name}/{quality}/{voice_name}.onnx'
    return base, f'{base}.json'


def _download_file(url: str, dest_path: str) -> None:
    """Stream-download url to dest_path via a temp file (atomic-ish rename)."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info('Downloading voice model file: %s', url)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest_path), suffix='.part')
    try:
        with os.fdopen(fd, 'wb') as tmp_file:
            with httpx.Client(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
                with client.stream('GET', url) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_bytes():
                        tmp_file.write(chunk)
        os.replace(tmp_path, dest_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _piper_model_paths() -> tuple[str, str]:
    """Local paths for the configured Piper voice's .onnx and .onnx.json."""
    onnx_path = os.path.join(settings.VOICE_MODELS_DIR, f'{settings.PIPER_VOICE}.onnx')
    return onnx_path, f'{onnx_path}.json'


def _ensure_piper_files() -> str:
    """Download the Piper voice files if missing; return the .onnx path."""
    onnx_path, json_path = _piper_model_paths()
    onnx_url, json_url = _piper_voice_urls(settings.PIPER_VOICE)
    if not os.path.exists(onnx_path):
        _download_file(onnx_url, onnx_path)
    if not os.path.exists(json_path):
        _download_file(json_url, json_path)
    return onnx_path


def get_whisper():
    """Lazy, thread-safe singleton for the faster-whisper model (CPU int8)."""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel

                logger.info('Loading whisper model: %s', settings.WHISPER_MODEL)
                _whisper_model = WhisperModel(
                    settings.WHISPER_MODEL,
                    device='cpu',
                    compute_type='int8',
                    download_root=settings.VOICE_MODELS_DIR,
                )
    return _whisper_model


def get_piper():
    """Lazy, thread-safe singleton for the Piper voice (downloads on first use)."""
    global _piper_voice
    if _piper_voice is None:
        with _piper_lock:
            if _piper_voice is None:
                from piper import PiperVoice

                onnx_path = _ensure_piper_files()
                logger.info('Loading piper voice: %s', settings.PIPER_VOICE)
                _piper_voice = PiperVoice.load(onnx_path)
    return _piper_voice


def stt_ready() -> bool:
    """True when the whisper model is loaded or its files are already cached."""
    if _whisper_model is not None:
        return True
    models_dir = settings.VOICE_MODELS_DIR
    if not os.path.isdir(models_dir):
        return False
    # faster-whisper caches HF snapshots like 'models--Systran--faster-whisper-base'
    marker = f'faster-whisper-{settings.WHISPER_MODEL}'
    return any(marker in entry for entry in os.listdir(models_dir))


def tts_ready() -> bool:
    """True when the piper voice is loaded or its files are already on disk."""
    if _piper_voice is not None:
        return True
    onnx_path, json_path = _piper_model_paths()
    return os.path.exists(onnx_path) and os.path.exists(json_path)


def transcribe(audio_path: str) -> dict:
    """
    Transcribe an audio file to text (blocking — run via asyncio.to_thread).

    Returns {text, language, duration}.

    """
    model = get_whisper()
    segments, info = model.transcribe(audio_path)
    text = ' '.join(segment.text.strip() for segment in segments).strip()
    return {
        'text': text,
        'language': info.language,
        'duration': info.duration,
    }


def synthesize(text: str) -> bytes:
    """
    Synthesize text to WAV bytes with Piper (blocking — run via asyncio.to_thread).

    Handles both Piper python APIs: the modern one where synthesize() yields
    AudioChunk objects, and the older one that writes into a wave file handle.

    """
    voice = get_piper()
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        if hasattr(voice, 'synthesize_wav'):
            # Modern API (piper >= 1.3): synthesize_wav writes into the handle
            voice.synthesize_wav(text, wav_file)
        else:
            result = voice.synthesize(text, wav_file)
            if result is not None:
                # AudioChunk-yielding variant: the wav handle was ignored, so
                # set params from the first chunk and write frames ourselves.
                params_set = False
                for chunk in result:
                    if not params_set:
                        wav_file.setnchannels(getattr(chunk, 'sample_channels', 1))
                        wav_file.setsampwidth(getattr(chunk, 'sample_width', 2))
                        wav_file.setframerate(getattr(chunk, 'sample_rate', 22050))
                        params_set = True
                    wav_file.writeframes(
                        getattr(chunk, 'audio_int16_bytes', None) or bytes(chunk)
                    )
    return buffer.getvalue()
