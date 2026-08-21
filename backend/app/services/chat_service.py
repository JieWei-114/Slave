import asyncio
import json
import logging
import re
import uuid
from datetime import datetime

from app.config.prompt_templates import (
    FILE_CONTEXT_INSTRUCTION,
    HISTORY_CONTEXT_HEADER,
    MEMORY_CONTEXT_HEADER,
    REASONING_PHASE_SYSTEM,
    WEB_CONTEXT_HEADER,
)
from app.config.settings import settings
from app.core.db import sessions_collection
from app.models.dto import ContextSource
from app.services.chat_session_service import (
    get_session, 
    get_session_rules, 
    create_session, 
    delete_session, 
    list_sessions, 
    rename_session,
    )
from app.services.context_planner import build_fallback_plan, plan_context
from app.services.context_builder_service import (
    calculate_weighted_confidence,
    extract_file_content,
    extract_key_points,
    rank_search_results,
)
from app.services.entity_validation_service import (
    assess_factual_guard,
    detect_uncertainty,
    validate_entities,
)
from app.services.file_extraction_service import (
    get_file_attachment,
    list_file_attachments,
)
from app.services.history_vector_service import (
    index_message as index_history_message,
)
from app.services.history_vector_service import (
    search_history as search_history_vectors,
)
from app.services.memory_service import (
    add_synthesized_memory,
    auto_memory_if_needed,
    list_enabled_memories,
    list_memories_by_category,
    search_memories,
)
from app.services.ollama_service import ProviderStreamError, stream_ollama
from app.services.reasoning_veto_service import assess_reasoning_veto
from app.services.web_search_service import maybe_extract, maybe_web_search
from app.utils.reasoning_utils import ReasoningTracker

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ════════════════════════════════════════════════════════════════════════
# All limits and thresholds pulled from settings for easy customization

# Prompt generation
PROMPT_MAX_TOTAL = settings.CHAT_PROMPT_MAX_TOTAL_CHARS

# History limits
HISTORY_LIMIT = settings.CHAT_HISTORY_LIMIT
HISTORY_MAX_PER_MSG = settings.CHAT_HISTORY_MAX_CHARS_PER_MSG
HISTORY_TOTAL_MAX = settings.CHAT_HISTORY_TOTAL_MAX_CHARS
MAX_ASSISTANT_CONTEXT = settings.CHAT_HISTORY_MAX_ASSISTANT_CONTEXT

# Memory limits
MEMORY_RESULTS_LIMIT = settings.CHAT_MEMORY_RESULTS_LIMIT
# At or below this many enabled memories, inject all of them instead of
# similarity-searching (see _build_memory_context)
SMALL_MEMORY_CORPUS_LIMIT = 30
MEMORY_TOTAL_MAX = settings.CHAT_MEMORY_TOTAL_MAX_CHARS

# Web search limits
WEB_RESULTS_LIMIT = settings.CHAT_WEB_RESULTS_LIMIT
WEB_SNIPPET_MAX = settings.CHAT_WEB_SNIPPET_MAX_CHARS
WEB_TOTAL_MAX = settings.CHAT_WEB_TOTAL_MAX_CHARS

# URL extraction limits
EXTRACT_TOTAL_MAX = settings.CHAT_EXTRACT_TOTAL_MAX_CHARS

# File upload limits
FILE_CONTENT_MAX = settings.CHAT_FILE_CONTENT_MAX_CHARS

# Feature toggles
ENABLE_RANKING = settings.CHAT_ENABLE_RESULT_RANKING
SYSTEM_INSTRUCTIONS = settings.CHAT_SYSTEM_INSTRUCTIONS

# Confidence scoring (source-based)
CONFIDENCE_FILE = settings.CONFIDENCE_FILE
CONFIDENCE_MEMORY = settings.CONFIDENCE_MEMORY
CONFIDENCE_HISTORY = settings.CONFIDENCE_HISTORY
CONFIDENCE_WEB = settings.CONFIDENCE_WEB
CONFIDENCE_NONE = settings.CONFIDENCE_NONE

# Text processing thresholds
TEXT_MIN_LENGTH = settings.TEXT_MIN_LENGTH_FOR_PROCESSING
TEXT_MIN_SENTENCE_LENGTH = settings.TEXT_MIN_SENTENCE_LENGTH
QUERY_TRUNCATION_LIMIT = settings.TEXT_QUERY_TRUNCATION_LIMIT
REASONING_TRUNCATION_LIMIT = settings.TEXT_REASONING_TRUNCATION_LIMIT

# ════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - Supporting Utilities
# ════════════════════════════════════════════════════════════════════════


def _get_session_configuration(session_id: str) -> dict:
    """
    Load session-specific rules and configuration.

    """
    rules = get_session_rules(session_id)
    if not rules:
        return {}
    return rules


def _collect_all_files(session_id: str, inline_file: dict = None) -> list[dict]:
    """
    Gather all files for this session: inline + stored attachments.

    """
    files = []
    seen_keys = set()

    # Add inline file first (highest priority)
    if inline_file:
        key = f'inline:{inline_file.get("filename")}:{inline_file.get("length")}'
        seen_keys.add(key)
        files.append(inline_file)

    # Add stored attachments
    try:
        attachments = list_file_attachments(session_id)
        for att in attachments:
            att_id = att.get('id')
            if not att_id or att_id in seen_keys:
                continue

            full = get_file_attachment(att_id)
            if not full or not full.get('content'):
                continue

            file_info = {
                'id': full.get('id'),
                'filename': full.get('filename'),
                'content': full.get('content'),
                'length': full.get('size_chars') or len(full.get('content', '')),
                'type': full.get('file_type') or full.get('type', 'unknown'),
            }

            key = f'{file_info["id"]}:{file_info["filename"]}:{file_info["length"]}'
            if key not in seen_keys:
                seen_keys.add(key)
                files.append(file_info)
    except Exception as e:
        logger.warning(f'Failed to load file attachments: {e}', extra={'session_id': session_id})

    return files


async def _extract_url_content(user_content: str, session_id: str) -> tuple[str, list[str]]:
    """
    Extract and process web content from URLs in user message.

    If user pasted a URL, fetch and extract its content for direct reference.
    This is more reliable than relying on web search alone.

    """
    try:
        extracted = await maybe_extract(user_content, session_id=session_id)
        if not extracted or not extracted.strip():
            return None, []

        # Extract key phrases for smart search
        key_points = extract_key_points(extracted, max_points=settings.EXTRACT_KEY_POINTS_MAX)

        # Save to memory for future reference
        try:
            text_to_save = extracted[:5000] if settings.MEMORY_MAX_CONTENT_LENGTH else extracted
            await add_synthesized_memory(
                session_id=session_id,
                fact=text_to_save,
                source='url_extraction',
                category='important',
                confidence=0.9,
            )
        except Exception as e:
            logger.debug(f'Failed to save URL to memory: {e}')

        return extracted, key_points
    except Exception as e:
        logger.debug(f'URL extraction failed: {e}')
        return None, []


def _filter_messages_after_topic_break(messages: list[dict], topic_break_at) -> list[dict]:
    """
    Keep only messages created after the latest topic break (if any).

    Sessions with a `topic_break_at` marker treat earlier messages as a
    previous topic: they are excluded from context building so the model
    starts fresh (aside from the stored topic summary).

    """
    if not topic_break_at:
        return messages
    return [m for m in messages if m.get('created_at') and m['created_at'] > topic_break_at]


def retrieve_semantic_history(
    session_id: str,
    history_query: str | None,
    exclude_ids: set[str] | None = None,
    exclude_created_ats: set[str] | None = None,
) -> list[dict]:
    """
    Semantic retrieval over the full conversation history.

    Embeds `history_query` and searches the history vector store (filtered by
    session), excluding messages already in the recency window. Returns up to
    4 hits above the similarity threshold as [{role, content, created_at}].
    Blocking (embedding + store search) — async callers wrap in asyncio.to_thread.

    """
    if not settings.HISTORY_VECTOR_ENABLED or not (history_query or '').strip():
        return []
    try:
        return search_history_vectors(
            session_id=session_id,
            query=history_query,
            exclude_ids=exclude_ids,
            exclude_created_ats=exclude_created_ats,
        )
    except Exception as e:
        logger.warning(
            f'Semantic history retrieval failed: {e}', extra={'session_id': session_id}
        )
        return []


def _index_message_vector(session_id: str, message: dict) -> None:
    """Embed + upsert a saved message into the history vector store (best effort)."""
    if not settings.HISTORY_VECTOR_ENABLED:
        return
    try:
        index_history_message(
            session_id=session_id,
            message_id=message['id'],
            role=message.get('role', ''),
            content=message.get('content', ''),
            created_at=message.get('created_at'),
        )
    except Exception as e:
        logger.warning(
            f'Failed to index message vector: {e}', extra={'session_id': session_id}
        )


async def build_prompt_with_memory(
    user_content: str,
    chat_sessionId: str = 'default',
    plan: dict | None = None,
) -> tuple[str, str, dict]:
    """
    MAIN ENTRY POINT: Build Complete Augmented Prompt with Multi-Source Context

    This orchestrates ALL context sources, guided by the context planner's plan:
      1. Session configuration (rules, limits, custom instructions)
      2. File attachments (inline + stored, filtered by plan['relevant_files'])
      3. URL extraction (if URLs in message)
      4. Conversation history (recent window)
      5. Semantic memory (plan['memory_query'])
      6. Web search (plan['web_queries'])

    """

    # ─────────────────────────────────────────────────────────────────────
    # Step 0: Validate input
    # ─────────────────────────────────────────────────────────────────────
    if not user_content or not user_content.strip():
        return '', '', {}

    user_content = user_content.strip()
    logger.info(
        'PROMPT BUILD START Session: %s | Query: "%s" (%s chars)',
        chat_sessionId[:12],
        user_content[:50] + ('...' if len(user_content) > 50 else ''),
        len(user_content),
        extra={'session_id': chat_sessionId},
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Load session configuration
    # ─────────────────────────────────────────────────────────────────────
    config = _get_session_configuration(chat_sessionId)

    local_web_limit = config.get('webSearchLimit') or WEB_RESULTS_LIMIT
    local_history_limit = config.get('historyLimit') or HISTORY_LIMIT
    local_memory_limit = config.get('memorySearchLimit') or MEMORY_RESULTS_LIMIT
    local_file_limit = config.get('fileUploadMaxChars') or FILE_CONTENT_MAX
    custom_instructions = config.get('customInstructions', '')

    logger.info(
        'Config loaded: web=%s, hist=%s, mem=%s, file=%s',
        local_web_limit,
        local_history_limit,
        local_memory_limit,
        local_file_limit,
        extra={'session_id': chat_sessionId},
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Build system instructions
    # ─────────────────────────────────────────────────────────────────────
    system_instructions = SYSTEM_INSTRUCTIONS
    if custom_instructions:
        system_instructions += f'\n\nCUSTOM INSTRUCTIONS:\n{custom_instructions}'

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Extract inline file content
    # ─────────────────────────────────────────────────────────────────────
    clean_message, inline_file = extract_file_content(user_content)
    user_content = clean_message if inline_file else user_content

    if inline_file:
        logger.info(
            'Inline file: %s (%s chars)',
            inline_file.get('filename'),
            inline_file.get('length'),
            extra={'session_id': chat_sessionId},
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Collect all file attachments
    # ─────────────────────────────────────────────────────────────────────
    file_infos = _collect_all_files(chat_sessionId, inline_file)
    logger.info('Files collected: %s', len(file_infos), extra={'session_id': chat_sessionId})

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Resolve the context plan
    # ─────────────────────────────────────────────────────────────────────
    if plan is None:
        plan = build_fallback_plan(user_content, file_infos)

    # Filter files by the plan (fallback plan / inline files keep everything)
    if not plan.get('fallback') and file_infos:
        relevant_names = set(plan.get('relevant_files') or [])
        file_infos = [
            f
            for f in file_infos
            if not f.get('id')  # inline file from this message: always keep
            or f.get('filename') in relevant_names
        ]
        logger.info(
            'Files after plan filter: %s', len(file_infos), extra={'session_id': chat_sessionId}
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Extract URL content
    # ─────────────────────────────────────────────────────────────────────
    extracted_content, extracted_key_points = await _extract_url_content(
        user_content, chat_sessionId
    )
    if extracted_content:
        logger.info(
            'URL extracted: %s chars, %s key points',
            len(extracted_content),
            len(extracted_key_points),
            extra={'session_id': chat_sessionId},
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Assemble context blocks
    # ─────────────────────────────────────────────────────────────────────
    blocks = []  # All context blocks
    factual_blocks = []  # Factual sources only (for entity validation)
    sources_considered = {}  # Which sources have content?

    logger.info('  ASSEMBLING CONTEXT', extra={'session_id': chat_sessionId})

    # 7.1: Add URL extraction if present
    if extracted_content:
        display = extracted_content
        if EXTRACT_TOTAL_MAX and len(display) > EXTRACT_TOTAL_MAX:
            display = display[:EXTRACT_TOTAL_MAX] + '\n[Truncated]'

        blocks.append('EXTRACTED WEB CONTENT\n' + display)
        factual_blocks.append('EXTRACTED WEB CONTENT\n' + display)
        sources_considered['url-extract'] = CONFIDENCE_WEB
        logger.info(
            '  ✓ URL extraction: %s chars', len(display), extra={'session_id': chat_sessionId}
        )

    # 7.2: Build HISTORY context (contextual source)
    context_limits = {
        'history_messages': local_history_limit,
        'memory_items': local_memory_limit,
        'web_results': local_web_limit,
        'web_snippet_max': WEB_SNIPPET_MAX,
        'web_total_max': WEB_TOTAL_MAX,
        'file_max_chars': local_file_limit,
    }

    hist_result = await build_context_for_source(
        session_id=chat_sessionId,
        source=ContextSource.HISTORY,
        context_limits=context_limits,
    )
    if hist_result['content']:
        blocks.append(hist_result['content'])
        sources_considered['history'] = hist_result.get('confidence', CONFIDENCE_HISTORY)
        logger.info(
            'History: %s msgs',
            hist_result['metadata'].get('messages_count', 0),
            extra={'session_id': chat_sessionId},
        )

    # 7.2a: Semantic history retrieval (contextual, like history) — older
    # messages beyond the recency window matching the planner's history_query
    semantic_history_hits = []
    if plan.get('history_query'):
        semantic_history_hits = await asyncio.to_thread(
            retrieve_semantic_history,
            chat_sessionId,
            plan.get('history_query'),
            set(hist_result['metadata'].get('included_ids') or []),
            set(hist_result['metadata'].get('included_created_ats') or []),
        )
    if semantic_history_hits:
        sem_lines = []
        for hit in semantic_history_hits:
            role = (hit.get('role') or '').upper()
            date = (hit.get('created_at') or '')[:10]
            sem_lines.append(f'[{role} @ {date}] {hit.get("content", "")}')
        blocks.append(
            'RELEVANT EARLIER CONVERSATION (contextual, non-factual)\n' + '\n'.join(sem_lines)
        )
        sources_considered['semantic_history'] = CONFIDENCE_HISTORY
        logger.info(
            'Semantic history: %s msgs',
            len(semantic_history_hits),
            extra={'session_id': chat_sessionId},
        )

    # 7.2b: Add topic overview (contextual source, like history)
    topic_summary = ''
    try:
        session_doc = sessions_collection.find_one(
            {'id': chat_sessionId}, {'_id': 0, 'topic_summary': 1}
        )
        topic_summary = ((session_doc or {}).get('topic_summary') or '').strip()
    except Exception as e:
        logger.warning(f'Failed to load topic summary: {e}', extra={'session_id': chat_sessionId})

    if topic_summary:
        blocks.append(
            'CONVERSATION OVERVIEW (contextual, non-factual)\n'
            f'Overview of earlier conversation (before topic change): {topic_summary}'
        )
        sources_considered['overview'] = 0.0
        logger.info(
            'Overview: %s chars', len(topic_summary), extra={'session_id': chat_sessionId}
        )

    # 7.3: Build WEB context (factual source) — only the planner's queries
    web_result = await build_context_for_source(
        session_id=chat_sessionId,
        source=ContextSource.WEB,
        user_content=user_content,
        web_queries=plan.get('web_queries') or [],
        context_limits=context_limits,
    )
    if web_result['content']:
        blocks.append(web_result['content'])
        factual_blocks.append(web_result['content'])
        sources_considered['web'] = web_result.get('confidence', CONFIDENCE_WEB)
        logger.info(
            'Web: %s results',
            web_result['metadata'].get('results_count', 0),
            extra={'session_id': chat_sessionId},
        )

    # 7.4: Build MEMORY context (factual source) — planner's memory_query.
    # Search with BOTH the planner's phrase and the raw user message: the
    # planner (a small model) often words the query too abstractly to match.
    memory_query = plan.get('memory_query')
    if memory_query:
        mem_result = await build_context_for_source(
            session_id=chat_sessionId,
            source=ContextSource.MEMORY,
            user_content=[memory_query, user_content],
            context_limits=context_limits,
        )
    else:
        mem_result = {'content': '', 'confidence': 0.0, 'metadata': {'items_count': 0}}
    if mem_result['content']:
        blocks.append(mem_result['content'])
        factual_blocks.append(mem_result['content'])
        sources_considered['memory'] = mem_result.get('confidence', CONFIDENCE_MEMORY)
        logger.info(
            'Memory: %s items',
            mem_result['metadata'].get('items_count', 0),
            extra={'session_id': chat_sessionId},
        )

    # 7.5: Build FILE context (factual source - highest priority)
    if file_infos:
        file_blocks = []
        for finfo in file_infos:
            fres = await build_context_for_source(
                session_id=chat_sessionId,
                source=ContextSource.FILE,
                selected_file_id=finfo.get('id'),
                file_info=finfo,
                context_limits=context_limits,
            )
            if fres['content']:
                file_blocks.append(fres['content'])
                factual_blocks.append(fres['content'])
                sources_considered['file'] = fres.get('confidence', CONFIDENCE_FILE)
                logger.info('File: %s', finfo.get('filename'), extra={'session_id': chat_sessionId})

        if file_blocks:
            blocks = file_blocks + blocks

    # ─────────────────────────────────────────────────────────────────────
    # Step 9: Freeze source snapshot
    # ─────────────────────────────────────────────────────────────────────
    loaded_sources = {
        'file': {
            'available': len(file_infos) > 0,
            'count': len(file_infos),
            'files': file_infos,
        },
        'memory': {
            'available': 'memory' in sources_considered,
            'count': mem_result['metadata'].get('items_count', 0)
            if 'memory' in sources_considered
            else 0,
        },
        'web': {
            'available': 'web' in sources_considered,
            'count': web_result['metadata'].get('results_count', 0)
            if 'web' in sources_considered
            else 0,
        },
        'history': {
            'available': 'history' in sources_considered,
        },
        'semantic_history': {
            'available': len(semantic_history_hits) > 0,
            'count': len(semantic_history_hits),
        },
        'overview': {
            'available': bool(topic_summary),
        },
    }

    # ─────────────────────────────────────────────────────────────────────
    # Step 10: Calculate confidence (factual sources only)
    # ─────────────────────────────────────────────────────────────────────
    # Relevance scores (how relevant is this source to the query?)
    source_relevance = {}
    if 'file' in sources_considered:
        source_relevance['file'] = 1.0
    if 'memory' in sources_considered:
        source_relevance['memory'] = 0.9
    if 'web' in sources_considered:
        source_relevance['web'] = 0.8
    if 'url-extract' in sources_considered:
        source_relevance['url-extract'] = 0.85

    # Calculate weighted confidence from FACTUAL sources only
    overall_confidence = calculate_weighted_confidence(
        sources_considered=sources_considered,
        source_relevance=source_relevance,
        factual_sources_only=True,
        loaded_sources=loaded_sources,
    )

    logger.info(
        'Confidence calculated: %.2f (from factual sources)',
        overall_confidence,
        extra={'session_id': chat_sessionId},
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 11: Build final prompt
    # ─────────────────────────────────────────────────────────────────────
    context = '\n\n'.join(blocks) if blocks else 'No context available.'

    prompt = f"""{context}

{'=' * 80}

USER QUERY: {user_content}

A:
"""

    if len(prompt) > PROMPT_MAX_TOTAL:
        prompt = prompt[:PROMPT_MAX_TOTAL] + '\n[Truncated]'

    logger.info(
        'PROMPT BUILT %s chars | %s sources | confidence: %.2f',
        len(prompt),
        len(sources_considered),
        overall_confidence,
        extra={'session_id': chat_sessionId},
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 12: Package metadata for downstream processing
    # ─────────────────────────────────────────────────────────────────────
    metadata = {
        'source_used': 'combined',
        'sources_considered': sources_considered,
        'source_relevance': source_relevance,
        'confidence': overall_confidence,
        'loaded_sources': loaded_sources,
        'has_factual_content': any(s in sources_considered for s in ['file', 'memory', 'web']),
        'factual_context_blocks': factual_blocks,  # Keep for entity validation
        'plan': plan,
        'context_details': {
            'file_count': len(file_infos),
            'web_count': web_result['metadata'].get('results_count', 0)
            if 'web' in sources_considered
            else 0,
            'memory_count': mem_result['metadata'].get('items_count', 0)
            if 'memory' in sources_considered
            else 0,
        },
    }

    return system_instructions, prompt, metadata


# ════════════════════════════════════════════════════════════════════════════════
# SOURCE-SPECIFIC CONTEXT BUILDERS for individual source types (FILE, MEMORY, HISTORY, WEB, etc.)
# ════════════════════════════════════════════════════════════════════════════════


async def build_context_for_source(
    session_id: str,
    source: ContextSource,
    user_content: str = '',
    web_queries: list[str] | None = None,
    selected_file_id: str | None = None,
    file_info: dict | None = None,
    context_limits: dict | None = None,
) -> dict:
    """
    Assemble context from a single source.

    Each source type (FILE, MEMORY, HISTORY, WEB) has different assembly logic.
    This function routes to appropriate handler and returns standardized dict.

    Returns:
        {
            'content': str,          # Formatted context block
            'confidence': float,     # 0.0-1.0 confidence in this source
            'source': ContextSource, # Which source
            'metadata': dict,        # Source-specific metadata
            'warning': str | None,   # Warning if applicable
        }
    """
    if context_limits is None:
        context_limits = {
            'history_messages': HISTORY_LIMIT,
            'memory_items': MEMORY_RESULTS_LIMIT,
            'web_results': WEB_RESULTS_LIMIT,
            'web_snippet_max': WEB_SNIPPET_MAX,
            'web_total_max': WEB_TOTAL_MAX,
        }

    logger.debug('Building context: source=%s', source, extra={'session_id': session_id})

    try:
        # Route to appropriate source handler
        if source == ContextSource.FILE:
            return await _build_file_context(
                session_id, selected_file_id, file_info, context_limits
            )
        elif source == ContextSource.MEMORY:
            return await _build_memory_context(session_id, user_content, context_limits)
        elif source == ContextSource.HISTORY:
            return await _build_history_context(session_id, context_limits)
        elif source == ContextSource.WEB:
            return await _build_web_context(
                session_id, user_content, web_queries or [], context_limits
            )
        else:
            return {
                'content': '',
                'confidence': 0.0,
                'source': source,
                'metadata': {},
                'warning': f'Unknown source: {source}',
            }
    except Exception as e:
        logger.error(f'Error building context for {source}: {e}', exc_info=True)
        return {
            'content': '',
            'confidence': 0.0,
            'source': source,
            'metadata': {},
            'warning': f'Error: {str(e)}',
        }


async def _build_file_context(session_id: str, file_id: str, file_info: dict, limits: dict) -> dict:
    """Build context from uploaded file."""
    try:
        if not file_id and not file_info:
            return {
                'content': '',
                'confidence': 0.0,
                'source': ContextSource.FILE,
                'metadata': {},
                'warning': 'No file data',
            }

        # Get file content
        if file_id:
            file_data = get_file_attachment(file_id)
        else:
            file_data = file_info

        if not file_data or not file_data.get('content'):
            return {
                'content': '',
                'confidence': 0.0,
                'source': ContextSource.FILE,
                'metadata': {},
                'warning': 'File has no content',
            }

        content = file_data.get('content', '')
        max_len = limits.get('file_max_chars', FILE_CONTENT_MAX)
        if len(content) > max_len:
            content = content[:max_len] + '\n[Truncated]'

        formatted = f"""UPLOADED FILE: {file_data.get('filename', 'unknown')}
Type: {file_data.get('file_type', 'unknown')}
Size: {len(content)} characters

{content}

{FILE_CONTEXT_INSTRUCTION}"""

        return {
            'content': formatted,
            'confidence': CONFIDENCE_FILE,
            'source': ContextSource.FILE,
            'metadata': {'filename': file_data.get('filename')},
            'warning': None,
        }
    except Exception as e:
        logger.error(f'Error in _build_file_context: {e}')
        return {
            'content': '',
            'confidence': 0.0,
            'source': ContextSource.FILE,
            'metadata': {},
            'warning': str(e),
        }


async def _build_memory_context(session_id: str, query: str | list, limits: dict) -> dict:
    """Build context from semantic memories.

    query may be a single phrase or a list of phrases — every phrase is
    searched and hits are merged (planner phrase + raw user message).
    """
    try:
        queries = [q for q in (query if isinstance(query, list) else [query]) if q]
        # Run blocking embedding + Mongo work off the event loop.
        # Small corpus → inject ALL enabled memories: similarity search
        # can't bridge broad questions ("what do you know about me?") and
        # short facts ("male") — real cosine is ~0.15-0.3, below any sane
        # threshold. Only fall back to semantic search on large corpora.
        all_memories = await asyncio.to_thread(list_enabled_memories, session_id)
        if len(all_memories) <= SMALL_MEMORY_CORPUS_LIMIT:
            important_memories = all_memories
            memory_hits = []
        else:
            important_memories = await asyncio.to_thread(
                list_memories_by_category, session_id, 'important'
            )
            memory_hits = []
            for q in queries:
                memory_hits.extend(
                    await asyncio.to_thread(
                        search_memories,
                        session_id,
                        q,
                        limits.get('memory_items', MEMORY_RESULTS_LIMIT),
                    )
                )

        combined = []
        seen_ids = set()

        for m in important_memories:
            mem_id = m.get('id') or m.get('_id')
            if mem_id in seen_ids:
                continue
            seen_ids.add(mem_id)
            combined.append(m)

        for m in memory_hits:
            mem_id = m.get('id') or m.get('_id')
            if mem_id in seen_ids:
                continue
            seen_ids.add(mem_id)
            combined.append(m)

        if not combined:
            return {
                'content': '',
                'confidence': 0.3,
                'source': ContextSource.MEMORY,
                'metadata': {'items_count': 0},
                'warning': 'No memories found',
            }

        lines = []
        for m in combined:
            category = (m.get('category') or 'other').upper()
            source = (m.get('source') or 'manual').upper()
            content = m.get('content') or m.get('value') or m.get('fact') or ''
            if not content:
                continue
            lines.append(f'[MEMORY: {category} | {source}] {content}')

        if not lines:
            return {
                'content': '',
                'confidence': 0.3,
                'source': ContextSource.MEMORY,
                'metadata': {'items_count': 0},
                'warning': 'Empty memories',
            }

        formatted = MEMORY_CONTEXT_HEADER + '\n' + '\n\n'.join(lines)

        return {
            'content': formatted,
            'confidence': CONFIDENCE_MEMORY,
            'source': ContextSource.MEMORY,
            'metadata': {'items_count': len(lines), 'important_count': len(important_memories)},
            'warning': None,
        }
    except Exception as e:
        logger.error(f'Error in _build_memory_context: {e}')
        return {
            'content': '',
            'confidence': 0.3,
            'source': ContextSource.MEMORY,
            'metadata': {},
            'warning': str(e),
        }


def _truncate_at_word(text: str, max_chars: int) -> str:
    """Truncate text at a word boundary (last space) instead of mid-word."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(' ')
    if last_space > 0:
        cut = cut[:last_space]
    return cut + '…'


async def _build_history_context(session_id: str, limits: dict) -> dict:
    """Build context from conversation history."""
    try:
        session = get_session(session_id)
        if not session or not session.get('messages'):
            return {
                'content': '',
                'confidence': 0.0,
                'source': ContextSource.HISTORY,
                'metadata': {'messages_count': 0},
                'warning': None,
            }

        messages = _filter_messages_after_topic_break(
            session['messages'], session.get('topic_break_at')
        )
        to_include = messages[-limits.get('history_messages', HISTORY_LIMIT) :]

        if not to_include:
            return {
                'content': HISTORY_CONTEXT_HEADER + '\nStatus: FIRST CONVERSATION',
                'confidence': 0.0,
                'source': ContextSource.HISTORY,
                'metadata': {'messages_count': 0, 'first_conversation': True},
                'warning': None,
            }

        lines = []
        included_ids = []
        included_created_ats = []
        for msg in to_include:
            role = msg.get('role', '').upper()
            content = _truncate_at_word(msg.get('content', ''), HISTORY_MAX_PER_MSG)
            lines.append(f'{role}: {content}')
            if msg.get('id'):
                included_ids.append(msg['id'])
            created_at = msg.get('created_at')
            if isinstance(created_at, datetime):
                included_created_ats.append(created_at.isoformat())
            elif created_at:
                included_created_ats.append(str(created_at))

        formatted = HISTORY_CONTEXT_HEADER + '\n' + '\n'.join(lines)

        return {
            'content': formatted,
            'confidence': CONFIDENCE_HISTORY,
            'source': ContextSource.HISTORY,
            'metadata': {
                'messages_count': len(lines),
                # Used to exclude the recency window from semantic history
                'included_ids': included_ids,
                'included_created_ats': included_created_ats,
            },
            'warning': None,
        }
    except Exception as e:
        logger.error(f'Error in _build_history_context: {e}')
        return {
            'content': '',
            'confidence': 0.0,
            'source': ContextSource.HISTORY,
            'metadata': {},
            'warning': str(e),
        }


async def _build_web_context(
    session_id: str, query: str, web_queries: list[str], limits: dict
) -> dict:
    """Build context from web search results using the planner's queries only."""
    try:
        search_queries = [
            sq.strip() for sq in (web_queries or []) if isinstance(sq, str) and sq.strip()
        ]
        if not search_queries:
            # Planner decided web search is not needed
            return {
                'content': '',
                'confidence': 0.0,
                'source': ContextSource.WEB,
                'metadata': {'results_count': 0, 'skipped': True},
                'warning': None,
            }

        logger.debug(
            f'Web search with {len(search_queries)} queries', extra={'session_id': session_id}
        )

        results = []
        seen_results = set()
        for sq in search_queries[:3]:
            try:
                res = await maybe_web_search(sq, session_id=session_id)
                if res and isinstance(res, list):
                    for r in res[: limits.get('web_results', WEB_RESULTS_LIMIT)]:
                        # Dedupe merged results across queries (by url or snippet)
                        key = (
                            (r.get('url') or r.get('snippet', ''))
                            if isinstance(r, dict)
                            else str(r)
                        )
                        if key in seen_results:
                            continue
                        seen_results.add(key)
                        results.append(r)
                if len(results) >= limits.get('web_results', WEB_RESULTS_LIMIT):
                    break
            except Exception as e:
                logger.debug(f'Web search failed for query "{sq}": {e}')
                continue

        if not results:
            return {
                'content': '',
                'confidence': 0.2,
                'source': ContextSource.WEB,
                'metadata': {'results_count': 0},
                'warning': 'No web results',
            }

        # Rank results by relevance to original query (only if results are dicts)
        if results and isinstance(results[0], dict):
            ranked_results = rank_search_results(results, query)
            logger.debug(f'Ranked {len(ranked_results)} web results for query "{query[:50]}"')
        else:
            logger.warning(f'Results format unexpected: {type(results[0]) if results else "empty"}')
            ranked_results = results

        # Format results
        lines = []
        web_sources = set()
        total_chars = 0
        max_total = limits.get('web_total_max', WEB_TOTAL_MAX)

        for res in ranked_results[: limits.get('web_results', WEB_RESULTS_LIMIT)]:
            source = res.get('source', 'unknown')
            web_sources.add(source)

            snippet = res.get('snippet', '')[: limits.get('web_snippet_max', WEB_SNIPPET_MAX)]
            if total_chars + len(snippet) > max_total:
                break

            lines.append(f'[{source}] {snippet}')
            total_chars += len(snippet)

        if not lines:
            return {
                'content': '',
                'confidence': 0.2,
                'source': ContextSource.WEB,
                'metadata': {'results_count': 0},
                'warning': 'Web results too large',
            }

        formatted = WEB_CONTEXT_HEADER + '\n' + '\n\n'.join(lines)

        return {
            'content': formatted,
            'confidence': CONFIDENCE_WEB,
            'source': ContextSource.WEB,
            'metadata': {'results_count': len(lines), 'web_sources': list(web_sources)},
            'warning': None,
        }
    except Exception as e:
        logger.error(f'Error in _build_web_context: {e}')
        return {
            'content': '',
            'confidence': 0.2,
            'source': ContextSource.WEB,
            'metadata': {},
            'warning': str(e),
        }


# ════════════════════════════════════════════════════════════════════════════════
# RESPONSE GENERATION & VALIDATION
# ════════════════════════════════════════════════════════════════════════════════


def _format_plan_summary(plan: dict) -> str:
    """
    Render the planner's decisions as a short server-side text line so the plan
    is visibly part of the streamed reasoning (no model call involved).
    """
    plan = plan or {}
    web = plan.get('web_queries') or []
    memory = plan.get('memory_query') or ''
    history = plan.get('history_query') or ''
    files = plan.get('relevant_files') or []
    return (
        f'Plan: web={web} | '
        f"memory='{memory}' | "
        f"history='{history}' | "
        f'files={files}\n\n'
    )


def _build_thinking_prompt(final_prompt: str) -> str:
    """
    Build the think-first prompt: same context sections as the answer prompt
    plus a compact instruction to reason BEFORE answering.
    """
    return f'{final_prompt}\n\n{REASONING_PHASE_SYSTEM.strip()}\n'


def _build_answer_prompt_with_reasoning(final_prompt: str, reasoning_text: str) -> str:
    """
    Build the answer prompt: same context plus the full reasoning text, with an
    instruction to now produce ONLY the final user-facing answer.
    """
    return (
        f'{final_prompt}\n\n'
        f'YOUR ANALYSIS (use this to answer):\n{reasoning_text.strip()}\n\n'
        'Now give ONLY the final user-facing answer. '
        'No step numbering, no meta commentary about your analysis.\n'
    )


def rewrite_for_verification(
    answer: str,
    guard_eval: dict,
    uncertainty_flags: list[dict],
) -> str:
    """
    Apply system-level verifications to assistant response.

    - If HIGH factual risk: refuse with safe response
    - If MED/LOW risk: downgrade confident language
    - Otherwise: return as-is
    """
    if not answer:
        return answer

    risk = guard_eval.get('risk', 'NONE') if guard_eval else 'NONE'
    has_uncertainty = bool(uncertainty_flags)

    if risk == 'HIGH':
        return (
            "I can't reliably confirm this with available sources. "
            'Please provide more information or enable web search.'
        )

    if risk in {'MED', 'LOW'} or has_uncertainty:
        disclaimer = "I may be missing verification for some details. Here's my best effort:\n\n"

        toned = answer
        replacements = {
            r'\bdefinitely\b': 'likely',
            r'\bclearly\b': 'seems',
            r'\bwill\b': 'may',
        }
        for pattern, repl in replacements.items():
            toned = re.sub(pattern, repl, toned, flags=re.IGNORECASE)

        return disclaimer + toned

    return answer


async def stream_chat_reply(
    session_id: str,
    content: str,
    model: str,
    reasoning_enabled: bool = False,
):
    """
    Main orchestrator for streaming chat response generation.

    Flow:
      1. Build augmented prompt with all context
      2. Think-first reasoning pass (if enabled), streamed as reasoning tokens
      3. Stream answer from model (informed by the reasoning)
      4. Validate against factual sources
      5. Apply system-level verification
      6. Save to database
      7. Auto-save important memories
    """
    # ─────────────────────────────────────────────────────────────────────
    # Step 0: Initialize reasoning tracker
    # ─────────────────────────────────────────────────────────────────────
    user_message_id = str(uuid.uuid4())
    reasoning_tracker = ReasoningTracker(session_id, user_message_id)
    reasoning_start_time = datetime.utcnow()

    logger.info(
        'STREAM START - Session: %s | Model: %s',
        session_id,
        model,
        extra={'session_id': session_id},
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Save user message
    # ─────────────────────────────────────────────────────────────────────
    user_msg = {
        'id': user_message_id,  # stable id for semantic history vectors
        'role': 'user',
        'content': content,
        'created_at': datetime.utcnow(),
    }

    # Include attachment reference if file was uploaded for this session
    try:
        attachments = list_file_attachments(session_id)
        if attachments:
            # Use the most recent attachment
            latest_file = attachments[0]
            user_msg['attachment'] = {
                'filename': latest_file.get('filename', 'unknown'),
                'content': latest_file.get('content', '')[:500],  # Store snippet for UI display
            }
            logger.info(f'Including attachment in user message: {latest_file.get("filename")}')
    except Exception as e:
        logger.warning(f'Could not fetch attachments for message: {e}')

    await asyncio.to_thread(
        sessions_collection.update_one,
        {'id': session_id},
        {
            '$push': {'messages': user_msg},
            '$set': {'updated_at': datetime.utcnow()},
        },
    )
    logger.info('User message saved', extra={'session_id': session_id})

    # Embed the user message for semantic history (best effort, never blocks chat)
    await asyncio.to_thread(_index_message_vector, session_id, user_msg)

    reasoning_tracker.log_step(
        thought='User message received and saved',
        action='SAVE_MESSAGE',
        source='database',
        confidence=1.0,
        information=f'Message length: {len(content)} chars',
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 2a: Plan context retrieval (before building any context)
    # ─────────────────────────────────────────────────────────────────────
    yield json.dumps({'type': 'planning'})

    recent_messages = []
    try:
        session_doc = sessions_collection.find_one(
            {'id': session_id}, {'_id': 0, 'messages': 1, 'topic_break_at': 1}
        )
        if session_doc and session_doc.get('messages'):
            msgs = _filter_messages_after_topic_break(
                session_doc['messages'], session_doc.get('topic_break_at')
            )
            # Exclude the user message we just saved (it goes in separately)
            recent_messages = msgs[:-1][-4:]
    except Exception as e:
        logger.warning(f'Failed to load recent messages for planner: {e}')

    attachments_meta = []
    try:
        attachments_meta = [
            {'filename': a.get('filename')} for a in list_file_attachments(session_id)
        ]
    except Exception as e:
        logger.warning(f'Failed to load attachments for planner: {e}')

    plan = await plan_context(
        user_message=content,
        recent_messages=recent_messages,
        attachments_meta=attachments_meta,
        model=model,
    )
    yield json.dumps({'type': 'plan', 'data': plan})

    reasoning_tracker.log_step(
        thought='Context retrieval planned',
        action='PLAN_CONTEXT',
        source='planner',
        confidence=0.9,
        information=f'Web queries: {len(plan.get("web_queries") or [])}, '
        f'files: {len(plan.get("relevant_files") or [])}, '
        f'fallback: {plan.get("fallback", False)}',
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 2b: Build augmented prompt
    # ─────────────────────────────────────────────────────────────────────
    system_prompt, final_prompt, context_meta = await build_prompt_with_memory(
        user_content=content,
        chat_sessionId=session_id,
        plan=plan,
    )
    logger.info('Prompt built: %s chars', len(final_prompt), extra={'session_id': session_id})

    # Log context building to reasoning
    sources = context_meta.get('sources_considered', {})
    loaded_sources = context_meta.get('loaded_sources', {})

    if sources:
        sources_str = ', '.join(sources.keys())
        reasoning_tracker.log_step(
            thought=f'Context assembled from {sources_str}',
            action='BUILD_CONTEXT',
            source='multi',
            confidence=0.95,
            information=f'Sources: {sources_str}',
        )

    for source_name, confidence in sources.items():
        reasoning_tracker.log_source_evaluation(source_name, confidence)

    # Extract metadata
    overall_confidence = context_meta.get('confidence', 0.8)
    sources = context_meta.get('sources_considered', {})
    factual_blocks = context_meta.get('factual_context_blocks', [])
    loaded_sources = context_meta.get('loaded_sources', {})

    # ─────────────────────────────────────────────────────────────────────
    # Step 2c: Collect image attachments for vision
    # Images are sent only when the planner asked for vision OR the user
    # message arrived together with a newly uploaded image (first turn after
    # upload). Images have no text extraction — the base64 data IS the context.
    # ─────────────────────────────────────────────────────────────────────
    images_to_send: list[str] = []
    try:
        image_atts = [a for a in list_file_attachments(session_id) if a.get('is_image')]
        if image_atts:
            last_prev_created = (
                recent_messages[-1].get('created_at') if recent_messages else None
            )
            has_new_image = any(
                a.get('uploaded_at')
                and (last_prev_created is None or a['uploaded_at'] > last_prev_created)
                for a in image_atts
            )
            if plan.get('needs_vision') or has_new_image:
                for att in image_atts:
                    full = get_file_attachment(att.get('id'))
                    b64 = (full or {}).get('image_base64')
                    if b64:
                        images_to_send.append(b64)
        if images_to_send:
            logger.info(
                'Including %s image(s) for vision (needs_vision=%s)',
                len(images_to_send),
                plan.get('needs_vision'),
                extra={'session_id': session_id},
            )
    except Exception as e:
        logger.warning(f'Failed to collect image attachments: {e}')

    # ─────────────────────────────────────────────────────────────────────
    # Step 3a: REASONING PHASE FIRST (think before answering, if enabled)
    # ─────────────────────────────────────────────────────────────────────
    reasoning_text = ''

    if reasoning_enabled:
        yield json.dumps({'type': 'reasoning_starting', 'data': 'Thinking...'})

        # Prepend the planner's decisions as the first reasoning content
        # (server-side text, not a model call).
        plan_summary = _format_plan_summary(plan)
        reasoning_text += plan_summary
        yield json.dumps({'type': 'reasoning_token', 'data': plan_summary})

        thinking_prompt = _build_thinking_prompt(final_prompt)
        logger.info('Generating think-first reasoning', extra={'session_id': session_id})
        try:
            async for token in stream_ollama(
                prompt=thinking_prompt,
                model=model,
                system=(
                    'You are an AI assistant thinking through a question before answering. '
                    'Be clear, honest, and concise.'
                ),
                images=images_to_send or None,
            ):
                reasoning_text += token
                yield json.dumps({'type': 'reasoning_token', 'data': token})

            logger.info(
                'Reasoning generated: %s chars',
                len(reasoning_text),
                extra={'session_id': session_id},
            )
        except Exception as e:
            logger.error(f'Reasoning generation failed: {e}')
            reasoning_text += '[Reasoning generation failed]'

        reasoning_tracker.log_step(
            thought='Generated think-first analysis before answering',
            action='THINK_FIRST',
            source='model',
            confidence=0.9,
            information=f'Reasoning length: {len(reasoning_text)} chars',
        )
    else:
        logger.info('Reasoning generation skipped (disabled)', extra={'session_id': session_id})

    # ─────────────────────────────────────────────────────────────────────
    # Step 3b: Generate answer and STREAM IMMEDIATELY
    # ─────────────────────────────────────────────────────────────────────
    assistant_answer = ''

    answer_prompt = (
        _build_answer_prompt_with_reasoning(final_prompt, reasoning_text)
        if reasoning_enabled and reasoning_text.strip()
        else final_prompt
    )

    try:
        async for token in stream_ollama(
            prompt=answer_prompt,
            model=model,
            system=system_prompt,
            images=images_to_send or None,
        ):
            assistant_answer += token
            # Stream live tokens to user immediately
            yield json.dumps({'type': 'token', 'data': token})

        logger.info(
            'Answer streamed: %s chars', len(assistant_answer), extra={'session_id': session_id}
        )

        # Send signal that answer streaming is complete
        yield json.dumps({'type': 'answer_complete'})
    except ProviderStreamError as e:
        # Stream failed: notify caller and do NOT persist an assistant message.
        # Non-vision models reject image input — surface a helpful message.
        logger.error(f'Answer generation failed: {e}')
        if images_to_send:
            yield json.dumps(
                {
                    'type': 'error',
                    'message': (
                        'This model cannot view images — install a vision model '
                        '(e.g. qwen2.5vl or gemma3:4b) from the Models page. '
                        f'(Provider error: {e})'
                    ),
                }
            )
        else:
            yield json.dumps({'type': 'error', 'message': f'Generation failed: {e}'})
        return
    except Exception as e:
        # Stream failed: notify caller and do NOT persist an assistant message
        logger.error(f'Answer generation failed: {e}')
        yield json.dumps({'type': 'error', 'message': f'Generation failed: {e}'})
        return

    if not assistant_answer.strip():
        # Model produced no output — treat as failure, do not persist empty message
        logger.error('Answer generation produced empty output', extra={'session_id': session_id})
        yield json.dumps({'type': 'error', 'message': 'Generation failed: empty response'})
        return

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: AFTER STREAMING - Validate against factual sources
    # Show verification hint to user
    # ─────────────────────────────────────────────────────────────────────
    yield json.dumps(
        {'type': 'verification_starting', 'data': 'Verifying answer against sources...'}
    )

    unverified = validate_entities(
        answer=assistant_answer,
        context_blocks=context_meta.get('context_blocks', []),
        factual_blocks=factual_blocks,  # FACTUAL SOURCES ONLY
    )

    reasoning_tracker.log_step(
        thought='Validated answer for unverified claims',
        action='VALIDATE_ENTITIES',
        source='internal',
        confidence=0.9,
        information=f'Found {len(unverified)} unverified items',
    )

    for entity in unverified[:3]:  # Log top 3
        reasoning_tracker.log_uncertainty(f'Unverified: {entity}')

    guard_eval = assess_factual_guard(unverified)
    if guard_eval['risk'] != 'NONE':
        overall_confidence = min(overall_confidence, guard_eval['cap'])
        logger.warning(
            'Factual guard: %s risk (cap: %.2f)',
            guard_eval['risk'],
            guard_eval['cap'],
            extra={'session_id': session_id},
        )
        reasoning_tracker.log_uncertainty(
            f'Factual risk {guard_eval["risk"]}: confidence capped to {guard_eval["cap"]:.2f}'
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Detect uncertainty
    # ─────────────────────────────────────────────────────────────────────
    uncertainty_flags = detect_uncertainty(
        source_used='combined',
        confidence=overall_confidence,
        response_text=assistant_answer,
    )

    for flag in uncertainty_flags[:3]:  # Log top 3
        if isinstance(flag, dict):
            reasoning_tracker.log_uncertainty(flag.get('message', 'Unknown uncertainty'))
        else:
            reasoning_tracker.log_uncertainty(str(flag))

    # Send verification result to frontend
    yield json.dumps(
        {
            'type': 'verification_complete',
            'data': {
                'risk_level': guard_eval.get('risk', 'NONE'),
                'unverified_count': len(unverified),
                'confidence_cap': guard_eval.get('cap', 1.0),
                'has_uncertainties': len(uncertainty_flags) > 0,
            },
        }
    )

    reasoning_tracker.log_step(
        thought='Verified answer against factual sources',
        action='VERIFY_ANSWER',
        source='internal',
        confidence=0.95,
        information=f'Risk level: {guard_eval.get("risk", "NONE")}',
    )

    logger.info('Answer verified after streaming', extra={'session_id': session_id})

    # Reasoning veto assessment (reasoning_text produced in the think-first phase)
    reasoning_veto = (
        assess_reasoning_veto(reasoning_text, overall_confidence, assistant_answer)
        if reasoning_enabled
        else {
            'level': 'none',
            'signals': [],
            'confidence_cap': 1.0,
            'reason': 'Reasoning disabled',
            'should_refuse': False,
            'refusal_message': '',
        }
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Finalize reasoning chain
    # ─────────────────────────────────────────────────────────────────────
    reasoning_end_time = datetime.utcnow()
    duration_ms = (reasoning_end_time - reasoning_start_time).total_seconds() * 1000

    reasoning_chain = reasoning_tracker.finalize(
        final_answer=assistant_answer,
        final_confidence=overall_confidence,
        model_used=model,
        duration_ms=duration_ms,
    )

    logger.info(
        'Reasoning chain finalized: %s steps',
        len(reasoning_chain.reasoning_steps),
        extra={'session_id': session_id},
    )

    # Format reasoning for frontend
    reasoning_chain_for_frontend = reasoning_tracker.get_summary()

    # ─────────────────────────────────────────────────────────────────────
    # Step 8: Save assistant message with FULL verification details
    # ─────────────────────────────────────────────────────────────────────
    assistant_msg = {
        'id': str(uuid.uuid4()),  # stable id for semantic history vectors
        'role': 'assistant',
        'content': assistant_answer.strip(),
        'created_at': datetime.utcnow(),
        'meta': {
            'source_used': 'combined',
            'sources_considered': sources,
            'source_relevance': context_meta.get('source_relevance', {}),
            'sources_used': list(sources.keys()),
            'loaded_sources': loaded_sources,
            'has_factual_content': any(s in sources for s in ['file', 'memory', 'web']),
            'confidence_initial': overall_confidence,
            'confidence_final': overall_confidence * guard_eval.get('cap', 1.0),
            # Verification results for frontend display
            'factual_guard': {
                'risk': guard_eval.get('risk', 'NONE'),
                'cap': guard_eval.get('cap', 1.0),
                'unverified_entities': unverified[:5],
            },
            'uncertainty_flags': [
                flag.dict() if hasattr(flag, 'dict') else flag for flag in uncertainty_flags[:5]
            ],
            'reasoning_veto': reasoning_veto,
            # Reasoning details
            'reasoning': reasoning_text,
            'reasoning_chain': reasoning_chain_for_frontend,
            'reasoning_chain_full': reasoning_chain.dict(),
        },
    }

    try:
        await asyncio.to_thread(
            sessions_collection.update_one,
            {'id': session_id},
            {
                '$push': {'messages': assistant_msg},
                '$set': {'updated_at': datetime.utcnow()},
            },
        )
        logger.info(
            'Assistant message saved with verification details (confidence: %.2f, risk: %s)',
            overall_confidence,
            guard_eval.get('risk'),
            extra={'session_id': session_id},
        )
    except Exception as e:
        logger.error(f'Failed to save message: {e}')
        yield json.dumps({'type': 'error', 'message': f'Save failed: {e}'})
        return

    # Embed the assistant message for semantic history (best effort)
    await asyncio.to_thread(_index_message_vector, session_id, assistant_msg)

    # ─────────────────────────────────────────────────────────────────────
    # Step 9: Auto-memory
    # ─────────────────────────────────────────────────────────────────────
    try:
        await auto_memory_if_needed(
            chat_sessionId=session_id,
            user_text=content,
            assistant_text=assistant_answer,
            model=model,
        )
        logger.info('Auto-memory processed', extra={'session_id': session_id})
    except Exception as e:
        logger.warning(f'Auto-memory failed: {e}')

    # ─────────────────────────────────────────────────────────────────────
    # Step 10: Send completion with verification details
    # ─────────────────────────────────────────────────────────────────────
    logger.info(
        'STREAM COMPLETE - Sources: %s | Confidence: %.2f | Risk: %s | Reasoning steps: %s',
        len(sources),
        overall_confidence,
        guard_eval.get('risk', 'NONE'),
        len(reasoning_chain.reasoning_steps),
        extra={'session_id': session_id},
    )

    yield json.dumps(
        {
            'type': 'done',
            'metadata': {
                'source_used': 'combined',
                'sources_used': list(sources.keys()),
                'supplemented_with': list(sources.keys()),
                'plan': plan,
                'sources_considered': sources,
                'source_relevance': context_meta.get('source_relevance', {}),
                'loaded_sources': loaded_sources,
                'reasoning_veto': reasoning_veto,
                'confidence_initial': overall_confidence,
                'confidence_final': overall_confidence * guard_eval.get('cap', 1.0),
                # Verification details for context indicator
                'factual_guard': {
                    'risk': guard_eval.get('risk', 'NONE'),
                    'cap': guard_eval.get('cap', 1.0),
                    'unverified_entities': unverified[:5],
                },
                'uncertainty_flags': [
                    flag.dict() if hasattr(flag, 'dict') else flag for flag in uncertainty_flags[:5]
                ],
                'source_conflicts': [],  # TODO: Implement conflict detection service
                'reasoning_chain': reasoning_chain_for_frontend,
                'answer_length': len(assistant_answer),
                'has_factual_content': context_meta.get('has_factual_content', False),
            },
        }
    )
