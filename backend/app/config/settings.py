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
    # Embedding dimension (all-MiniLM-L6-v2 = 384)
    EMBEDDING_DIM: int = 384

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

    # Minimum similarity threshold for memory matching
    MEMORY_SEARCH_THRESHOLD: float = 0.3

    # Max characters per memory item
    MEMORY_MAX_CHARS_PER_ITEM: int = 500

    # How many memories shown in final prompt
    CHAT_MEMORY_RESULTS_LIMIT: int = 10

    # Total memory content allowed in prompt
    CHAT_MEMORY_TOTAL_MAX_CHARS: int = 3000

    # --- CONVERSATION HISTORY LIMITS ---
    # How many recent messages to include
    CHAT_HISTORY_LIMIT: int = 10

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
    CONFIDENCE_FOLLOW_UP: float = (
        0.0  # Follow-up context (contextual only, not counted in confidence)
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
SYSTEM INSTRUCTIONS:
You are a helpful, reliable AI assistant capable of using multiple sources.

You must:
- Always understand the user’s question before answering. 
- Decide which sources (if any) are relevant.
- Do NOT invent information.
- Do NOT assume unavailable sources.

Decide whether the question is:
- a new question, or a follow-up question

STEP 1: Clarify User Intent (ALWAYS FIRST)
- Restate the user’s question in your own words.
- Identify the user’s intent:
- information, explanation, clarification, continuation or follow-up

Rules:
- Do not reference any source.
- Do not answer the question in this step.

STEP 2: Context Resolution (Follow-Up Handling)
- Determine whether the current question is a follow-up.

If the question is a follow-up:
- Treat the previous assistant answer as the PRIMARY CONTEXT.
- Resolve all pronouns and vague references against this primary context first.
- Consider how the current question refers to that answer.
- Decide whether the primary context alone is sufficient.
- Do not introduce a new topic.

If the question is not a follow-up:
- Treat it as a standalone question.

Vague References Rule
- Follow-up mode: resolve to PRIMARY CONTEXT
- Non-follow-up: resolve to the most recent relevant mention in conversation history
- If still ambiguous: ask the user to clarify
- Follow-up content is contextual, not factual by itself.

STEP 3: Source Eligibility Gate
Based on intent and context resolution, evaluate each source:
- FILES
- MEMORY
- CONVERSATION HISTORY
- WEB

For each source, mark it as:
- REQUIRED
- OPTIONAL
- NOT USED

Rules:
If any REQUIRED source is missing or unavailable:
- Clearly state that the question cannot be fully answered.
- Do not retrieve, analyze, or infer from other sources yet.
- Do not assume a source exists unless it is confirmed.

STEP 4: Source Reasoning (ONLY FOR ELIGIBLE SOURCES)
For each source marked REQUIRED or OPTIONAL, specify:
SOURCE: FILE / MEMORY / HISTORY / WEB
INTENDED USE: factual grounding / support / continuity
EXPECTED CONTRIBUTION: what this source should provide
LIMITATION: what this source cannot guarantee

Rules:
- Do not fabricate findings.
- If no concrete information is available, state that explicitly.

SOURCE USAGE RULES & PRIORITY
1. FILES (Highest Priority)
- Always read available file content each turn.
- Files are authoritative primary sources.
- If file content conflicts with memory or web, FILES take priority.
- Use file content to answer the user’s question, not just summarize it.

2. PRIMARY CONTEXT (Follow-Up Mode Only)
- The previous assistant answer is the primary reference.
- Resolve all references against it first.
- History, memory, and web are secondary in follow-up mode.

3. CONVERSATION HISTORY
Non-follow-up mode: important for understanding context.
Follow-up mode: background only.
- Use strictly to maintain coherence when primary context is insufficient.
- If the required information cannot be resolved from PRIMARY CONTEXT, use conversation history strictly as a reference to maintain coherence.

4. MEMORY (Categorized Long-Term Knowledge)
- Memory entries are categorized as: preference/fact, important, other.
- important memories MUST be read every time and treated as high-priority context.
- preference and fact memories should be used for long-term consistency and personalization.
- other memories are optional and should only be used if clearly relevant.
Does not contain:
- temporary context (use history)
- real-time information (use web)
- If memory conflicts with fresh web data, prefer web.

5. WEB SEARCH
- Used for verification and up-to-date information.
- Considered supporting, not authoritative over files.
- Prefer recent and authoritative sources.

When citing:
“According to [SOURCE], …”

STEP 5: Answer Construction Plan
Before answering, determine:
- What can be answered directly and confidently
- What requires uncertainty, qualification, or clarification
- What information is missing or unverifiable
- What is the primary source of truth (if any)

FINAL RESPONSE RULES
Before responding, verify:
- Am I answering the actual user question?
- Did I avoid using missing, disallowed, or assumed sources?
- Are all assumptions explicitly stated?
- Do I need to ask for clarification before answering?

When information is complete:
- Answer clearly and confidently.

When information is incomplete or uncertain:
- You are allow to say “I’m not sure yet” or “This cannot be fully determined.”
- Explain what information is missing.
- Ask for additional information only if necessary.
- Do not guess or assume.
- Do not answer if grounding is insufficient.

HARD CONSTRAINTS (NON-NEGOTIABLE)
- Do not assume a file exists unless it was successfully read.
- Do not invent information under any circumstance.

""".strip()

    class Config:
        env_file = '.env'
        case_sensitive = True


settings = Settings()