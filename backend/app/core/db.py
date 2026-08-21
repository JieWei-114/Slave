"""
Database connection and collection setup
Configures MongoDB client with connection pooling and creates indexes

"""

import logging

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from app.config.settings import settings

logger = logging.getLogger(__name__)

try:
    # Initialize MongoDB client with connection pooling
    client = MongoClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=settings.MONGO_SERVER_SELECTION_TIMEOUT_MS,
        maxPoolSize=settings.MONGO_MAX_POOL_SIZE,
        minPoolSize=settings.MONGO_MIN_POOL_SIZE,
        directConnection=True,
    )
    # Test connection immediately
    client.admin.command('ping')
    logger.info('MongoDB connection established successfully')
    db = client[settings.DB_NAME]

except ConnectionFailure as e:
    logger.error(f'MongoDB connection failed: {e}')
    raise Exception('Failed to connect to MongoDB. Please ensure MongoDB is running.')

"""
Collection References

"""
# Core collections
sessions_collection = db['sessions']
rules_collection = db['rules']

# API quota tracking
serper_quota_collection = db['serper_quota']
tavily_quota_collection = db['tavily_quota']

# Source-aware collections (new architecture)
synthesized_memory_collection = db['synthesized_memory']
file_attachments_collection = db['file_attachments']

# Conversation-history message vectors (semantic history, mongo vector store)
message_vectors_collection = db['message_vectors']

"""
Index Creation

"""
try:
    # Synthesized memory indexes for fast queries
    synthesized_memory_collection.create_index([('session_id', 1), ('last_referenced_at', -1)])
    synthesized_memory_collection.create_index([('session_id', 1), ('category', 1)])
    synthesized_memory_collection.create_index([('session_id', 1), ('is_deprecated', 1)])

    # File attachments indexes for expiration and retrieval
    file_attachments_collection.create_index([('session_id', 1), ('expires_at', 1)])
    file_attachments_collection.create_index([('session_id', 1), ('uploaded_at', -1)])

    # Message vector indexes for semantic history lookup
    message_vectors_collection.create_index([('id', 1)], unique=True)
    message_vectors_collection.create_index([('session_id', 1), ('created_at', -1)])

    logger.info('Database indexes created successfully')
except Exception as e:
    logger.warning(f'Index creation warning: {e}')
