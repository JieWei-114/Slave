from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables"""

    # DATABASE SETTINGS
    MONGO_URI: Optional[str] = None # Set in .env
    # NOTE: default DB name renamed from 'myslave' to 'slave' as part of the
    # MySlave -> Slave rename. Existing deployments should set DB_NAME in .env
    # to keep pointing at their old database.
    DB_NAME: str = 'slave' # Override in .env

    # SECURITY SETTINGS
    # Optional API key. When non-empty, all API requests must include the
    # X-API-Key header with this value. Empty (default) = auth disabled (local dev).
    API_KEY: str = ''

    # MongoDB connection pool settings
    MONGO_MAX_POOL_SIZE: int = 100
    MONGO_MIN_POOL_SIZE: int = 10
    MONGO_SERVER_SELECTION_TIMEOUT_MS: int = 5000

    # LLM SETTINGS
    # Active LLM provider: 'ollama' (default) | 'openai_compat'
    LLM_PROVIDER: str = 'ollama'

    # Ollama (native API)
    OLLAMA_URL: Optional[str] = None # Set in .env
    OLLAMA_TIMEOUT: int = 120

    # OpenAI-compatible API (vLLM, llama.cpp server, LM Studio, OpenAI)
    # Base URL without /v1 suffix, e.g. http://localhost:8001
    OPENAI_COMPAT_BASE_URL: str = ''
    OPENAI_COMPAT_API_KEY: str = ''  # Sent as Authorization: Bearer when non-empty

    # VECTOR STORE SETTINGS
    # Active vector store: 'mongo' (default, cosine in Python) | 'qdrant'
    VECTOR_STORE: str = 'mongo'
    QDRANT_URL: str = 'http://qdrant:6333'
    QDRANT_COLLECTION: str = 'slave_memories'
    # Qdrant collection for conversation-history vectors (semantic history)
    QDRANT_HISTORY_COLLECTION: str = 'slave_history'
    # Embedding dimension (all-MiniLM-L6-v2 = 384)
    EMBEDDING_DIM: int = 384

    # SEMANTIC HISTORY SETTINGS
    # Embed chat messages on save and retrieve older, semantically relevant
    # messages beyond the recency window when the planner asks for them.
    HISTORY_VECTOR_ENABLED: bool = True
    # Max chars of message content embedded / stored in the history payload
    HISTORY_VECTOR_CONTENT_MAX_CHARS: int = 1000
    # Retrieval knobs
    HISTORY_VECTOR_TOP_K: int = 4
    HISTORY_VECTOR_THRESHOLD: float = 0.45
    # Startup backfill bound (most recent messages across sessions)
    HISTORY_VECTOR_BACKFILL_MAX: int = 2000

    # VISION SETTINGS
    # Max image upload size in MB (pre-base64-encode)
    VISION_IMAGE_MAX_MB: int = 10

    # VOICE SETTINGS (local STT + TTS, privacy-first)
    VOICE_ENABLED: bool = True
    # faster-whisper model size: tiny | base | small | medium | large-v3
    WHISPER_MODEL: str = 'base'
    # Piper voice name, e.g. 'en_US-lessac-medium' ({locale}-{name}-{quality})
    PIPER_VOICE: str = 'en_US-lessac-medium'
    # Where whisper + piper model files are downloaded/cached
    VOICE_MODELS_DIR: str = '/app/voice_models'

    # CORS SETTINGS
    CORS_ORIGINS: list[str] = Field(default_factory=list)

    # ============================================================
    # CENTRALIZED LIMIT SYSTEM
    # ============================================================
    # All limits defined once, used consistently across all services

    # --- WEB SEARCH LIMITS ---
    # How many results each provider fetches (per single search call)
    WEB_SEARCH_RESULTS_PER_PROVIDER: int = 10

    # Total limit for "advance search" (distributed across all enabled providers)
    WEB_SEARCH_ADVANCE_TOTAL: int = 40

    # How many results shown in final prompt (filtered from provider results)
    CHAT_WEB_RESULTS_LIMIT: int = 10

    # Max characters per web result snippet in prompt
    CHAT_WEB_SNIPPET_MAX_CHARS: int = 800

    # Total web search content allowed in prompt
    CHAT_WEB_TOTAL_MAX_CHARS: int = 6000

    # --- MEMORY LIMITS ---
    # How many memories to search/retrieve (default for all contexts)
    MEMORY_SEARCH_LIMIT: int = 10

    # Minimum similarity threshold for memory matching.
    # NOTE: all-MiniLM scores short facts vs. questions in the 0.40-0.50
    # band ("what is my name?" vs "my name is vey" = 0.47), so anything
    # above ~0.4 silently drops valid personal memories.
    MEMORY_SEARCH_THRESHOLD: float = 0.35

    # Max characters per memory item
    MEMORY_MAX_CHARS_PER_ITEM: int = 500

    # How many memories shown in final prompt
    CHAT_MEMORY_RESULTS_LIMIT: int = 10

    # Total memory content allowed in prompt
    CHAT_MEMORY_TOTAL_MAX_CHARS: int = 3000

    # --- CONVERSATION HISTORY LIMITS ---
    # How many recent messages to include
    CHAT_HISTORY_LIMIT: int = 6

    # Max characters per message in history
    CHAT_HISTORY_MAX_CHARS_PER_MSG: int = 500

    # Total history content allowed in prompt
    CHAT_HISTORY_TOTAL_MAX_CHARS: int = 5000

    # Total history content use for review
    CHAT_HISTORY_MAX_ASSISTANT_CONTEXT: int = 5

    # --- FILE UPLOAD LIMITS ---
    # Max file size for upload (in MB)
    FILE_UPLOAD_MAX_SIZE_MB: int = 10

    # Max characters extracted from file (at extraction time)
    FILE_UPLOAD_MAX_CHARS: int = 50000

    # Allowed binary file extensions for server-side extraction
    FILE_UPLOAD_ALLOWED_EXTENSIONS: list[str] = Field(
        default_factory=lambda: ['.pdf', '.doc', '.docx']
    )

    # Max file content shown in prompt
    CHAT_FILE_CONTENT_MAX_CHARS: int = 30000

    # --- URL EXTRACTION LIMITS ---
    # Total extracted content allowed in prompt
    CHAT_EXTRACT_TOTAL_MAX_CHARS: int = 8000

    # --- OVERALL PROMPT LIMIT ---
    # Final safety limit for entire prompt sent to model
    CHAT_PROMPT_MAX_TOTAL_CHARS: int = 100000

    # --- FEATURES ---
    CHAT_ENABLE_RESULT_RANKING: bool = True

    # Context planner: one small LLM call decides web queries, memory/history
    # retrieval phrases and relevant files before context is built.
    # When False, the static fallback plan is used (roughly the old behavior).
    PLANNER_ENABLED: bool = True

    # --- API DEFAULTS ---
    # Default limits for API endpoints (can be overridden by query params)
    # Note: API uses same limit as internal search for consistency
    API_MESSAGES_DEFAULT_LIMIT: int = 20

    # Content extraction and analysis
    EXTRACT_KEY_POINTS_MAX: int = 3  # Max key points to extract from URL content
    KEY_POINT_EXTRACTION_SAMPLE_SIZE: int = 30  # Sample first N sentences

    # ============================================================
    # PROVIDER-SPECIFIC SETTINGS
    # ============================================================
    # DuckDuckGo (DDG) - Free search engine
    DDG_TIMEOUT: float = 10.0
    DDG_LIMIT: int = 10

    # SearXNG - Self-hosted metasearch engine
    SEARXNG_URL: Optional[str] = None # Set in .env
    SEARXNG_TIMEOUT: float = 10.0
    SEARXNG_LIMIT: int = 10

    # Serper - Google Search API (paid, quota-limited)
    SERPER_URL: Optional[str] = None # Set in .env
    SERPER_API_KEY: Optional[str] = None  # Required: Set in .env
    SERPER_LIMIT: int = 10  # Results per search
    SERPER_TOTAL_LIMIT: int = 2500  # Monthly API quota
    SERPER_TIMEOUT: float = 20.0

    # Tavily - Research API (paid, quota-limited)
    TAVILY_URL: Optional[str] = None # Set in .env
    TAVILY_API_KEY: Optional[str] = None  # Required: Set in .env
    TAVILY_LIMIT: int = 5  # Results per search
    TAVILY_TIMEOUT: float = 20.0
    TAVILY_MONTHLY_LIMIT: int = 1000  # Monthly API quota

    # Auto-routing keywords for research-focused queries (Tavily)
    WEB_TAVILY_KEYWORDS: list[str] = Field(
        default_factory=lambda: [
            'research',
            'deep dive',
            'detailed',
            'analyze',
            'comprehensive',
            'thorough',
        ]
    )

    # WEB EXTRACTION SETTINGS
    TAVILY_EXTRACT_MAX_LENGTH: int = 10000
    TAVILY_EXTRACT_TIMEOUT: float = 20.0

    LOCAL_EXTRACT_MAX_CHARS: int = 20000
    LOCAL_EXTRACT_MAX_BYTES: int = 1_000_000
    LOCAL_EXTRACT_TIMEOUT: float = 10.0

    # ============================================================
    # MEMORY AUTO-SAVE SETTINGS
    # ============================================================
    MEMORY_MAX_CONTENT_LENGTH: int = 3000  # Max chars per memory entry
    MEMORY_MIN_ASSISTANT_LENGTH: int = 30  # Min chars in assistant response to remember
    MEMORY_MIN_CONVERSATION_LENGTH: int = 50  # Min combined user+assistant length to remember

    # Memory function defaults
    MEMORY_DEFAULT_CONFIDENCE: float = 0.95  # Default confidence for new memories

    # ============================================================
    # CONTEXT SOURCE CONFIDENCE LEVELS
    # ============================================================
    # Used by chat service to score different information sources
    CONFIDENCE_FILE: float = 0.99  # User-uploaded files
    CONFIDENCE_MEMORY: float = 0.85  # Stored memories
    CONFIDENCE_WEB: float = 0.65  # Web search results
    CONFIDENCE_HISTORY: float = (
        0.0  # Conversation history (contextual only, not counted in confidence)
    )
    CONFIDENCE_NONE: float = 0.3  # No context available

    # ============================================================
    # TEXT PROCESSING LIMITS
    # ============================================================
    TEXT_MIN_LENGTH_FOR_PROCESSING: int = 100  # Min chars to process text
    TEXT_MIN_SENTENCE_LENGTH: int = 10  # Min chars for valid sentence
    TEXT_SENTENCE_WEIGHT_DENOMINATOR: int = 150  # Used in sentence scoring
    TEXT_QUERY_TRUNCATION_LIMIT: int = 300  # Max chars for query preview
    TEXT_REASONING_TRUNCATION_LIMIT: int = 2000  # Max chars for reasoning storage

    # Memory processing
    MEMORY_DB_QUERY_LIMIT: int = 100  # Max results from DB query
    MEMORY_TEXT_FALLBACK_LIMIT: int = 500  # Fallback truncation
    MEMORY_KEY_TRUNCATION_LIMIT: int = 100  # Max chars for memory key
    MEMORY_LOG_TRUNCATION_LIMIT: int = 50  # Max chars in log messages

    # File attachment settings
    FILE_ATTACHMENT_MAX_CHARS: int = 100000
    FILE_ATTACHMENT_EXPIRY_DAYS: int = 30

    # ============================================================
    # FACTUAL GUARD THRESHOLDS
    # ============================================================
    # Used for post-answer validation (unverified entity detection)
    FACTUAL_GUARD_LOW_CAP: float = 0.6  # Confidence cap for low risk
    FACTUAL_GUARD_MED_CAP: float = 0.5  # Confidence cap for medium risk
    FACTUAL_GUARD_HIGH_CAP: float = 0.4  # Confidence cap for high risk
    FACTUAL_GUARD_MED_UNVERIFIED: int = 3  # Threshold for medium risk
    FACTUAL_GUARD_HIGH_UNVERIFIED: int = 6  # Threshold for high risk

    # ============================================================
    # TEXT PROCESSING WEIGHTS & SCORING
    # ============================================================
    # Sentence scoring weights (for key point extraction)
    SENTENCE_SCORE_POSITION_WEIGHT: float = 0.6  # Weight for sentence position
    SENTENCE_SCORE_LENGTH_WEIGHT: float = 0.4  # Weight for sentence length

    # Default confidence values
    CONFIDENCE_UNCERTAINTY: float = (
        0.7  # Threshold for uncertainty detection (used by entity_validation_service)
    )

    # ============================================================
    # SYSTEM INSTRUCTIONS
    # ============================================================
    CHAT_SYSTEM_INSTRUCTIONS: str = """
You are a helpful local AI assistant.

- Answer the user's question naturally and directly.
- The prompt may include context sections (UPLOADED FILE, CONVERSATION HISTORY,
  WEB SEARCH RESULTS, RELEVANT MEMORIES, CONVERSATION OVERVIEW). Use them when
  they are relevant; ignore them when they are not.
- Uploaded files are the most authoritative source, then memories, then web
  results. Conversation history is for continuity, not facts.
- RELEVANT MEMORIES are facts the user explicitly saved. They OVERRIDE
  conversation history: if an earlier assistant reply contradicts a memory
  (e.g. it said "I don't know" before the memory was saved), trust the
  memory and answer from it.
- When you rely on a context section, cite the source name inline, e.g.
  "According to report.pdf, ..." or "According to web search results, ...".
- If the answer is not in the context and you are not confident, say you
  don't know instead of guessing.
- Never invent file contents, URLs, or citations.
- Respond in the same language the user writes in.
""".strip()

    class Config:
        env_file = '.env'
        case_sensitive = True


settings = Settings()