"""
Main application entry point
Sets up FastAPI app with CORS, routers, and health checks
"""

import asyncio
import logging
import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.memory import router as memory_router
from app.api.models import router as models_router
from app.api.rules import router as rule_router
from app.api.voice import router as voice_router
from app.api.web import router as web_router
from app.config.settings import settings
from app.core.auth import require_api_key
from app.core.db import client
from app.services.history_vector_service import backfill_history_vectors
from app.services.memory_service import reindex_memories

logger = logging.getLogger(__name__)

# Configure logging level from environment
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    force=True,
)

# Reduce noisy third-party debug logs
logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Initialize FastAPI application
app = FastAPI(title='Slave', version='1.0.0')

# Enable CORS for frontend communication.
# Never fall back to '*' — if CORS_ORIGINS is unset, use explicit localhost
# dev origins. allow_credentials is False because no cookies are used.
DEFAULT_DEV_ORIGINS = [
    'http://localhost:4200',
    'http://localhost:4000',
    'http://localhost:4173',
    # Tauri desktop shell (packaged app origins)
    'tauri://localhost',
    'http://tauri.localhost',
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS or DEFAULT_DEV_ORIGINS,
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


async def _reindex_vector_store() -> None:
    """Reconcile the vector store with Mongo without blocking startup."""
    try:
        count = await asyncio.to_thread(reindex_memories)
        logger.info('Startup vector reindex done: %s memories upserted', count)
    except Exception as e:
        logger.warning('Startup vector reindex failed: %s', e)


async def _backfill_history_vectors() -> None:
    """Index messages missing history vectors without blocking startup."""
    try:
        count = await asyncio.to_thread(backfill_history_vectors)
        logger.info('Startup history vector backfill done: %s messages indexed', count)
    except Exception as e:
        logger.warning('Startup history vector backfill failed: %s', e)


@app.on_event('startup')
async def startup_vector_reindex():
    # Only needed for external vector stores (Mongo store reads embeddings
    # straight from the source-of-truth collection)
    if (settings.VECTOR_STORE or 'mongo').strip().lower() != 'mongo':
        asyncio.create_task(_reindex_vector_store())

    # Semantic history: backfill message vectors for messages saved before
    # this feature existed (bounded; both backends use a separate collection)
    if settings.HISTORY_VECTOR_ENABLED:
        asyncio.create_task(_backfill_history_vectors())


@app.get('/')
async def read_root():
    """Root endpoint - welcome message"""
    return {'message': 'Welcome to my API!'}


@app.get('/health')
async def health_check():
    """Health check endpoint with database connectivity status"""
    try:
        # Ping MongoDB to check connection
        client.admin.command('ping')
        db_status = 'connected'
    except Exception:
        db_status = 'disconnected'

    return {
        'status': 'healthy' if db_status == 'connected' else 'degraded',
        'database': db_status,
        'version': '1.0.0',
    }


# Register API routers (all protected by optional API-key auth)
app.include_router(chat_router, dependencies=[Depends(require_api_key)])  # Chat sessions and streaming
app.include_router(memory_router, dependencies=[Depends(require_api_key)])  # Memory management
app.include_router(web_router, dependencies=[Depends(require_api_key)])  # Web search
app.include_router(rule_router, dependencies=[Depends(require_api_key)])  # Rules configuration
app.include_router(models_router, dependencies=[Depends(require_api_key)])  # HF model search + Ollama pull
app.include_router(voice_router, dependencies=[Depends(require_api_key)])  # Local voice STT + TTS
