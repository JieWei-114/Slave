import asyncio
import json
import logging
import re
import uuid
from datetime import datetime

from app.config.prompt_templates import (
    CONTINUATION_HINT_FOLLOWUP,
    FILE_CONTEXT_INSTRUCTION,
    HISTORY_CONTEXT_HEADER,
    MEMORY_CONTEXT_HEADER,
    PRIMARY_CONTEXT_HEADER,
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
from app.services.memory_service import (
    add_synthesized_memory,
    auto_memory_if_needed,
    list_memories_by_category,
    search_memories,
)
from app.services.ollama_service import stream_ollama
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
CONFIDENCE_FOLLOW_UP = settings.CONFIDENCE_FOLLOW_UP
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


def _get_primary_assistant_answer(session_id: str) -> str:
    """
    Retrieve most recent assistant response from conversation history.

    Used for follow-up detection and reference resolution.
    Messages before a topic break are ignored.

    """
    try:
        session = sessions_collection.find_one(
            {'id': session_id}, {'_id': 0, 'messages': 1, 'topic_break_at': 1}
        )
        if not session or not session.get('messages'):
            return None

        messages = _filter_messages_after_topic_break(
            session['messages'], session.get('topic_break_at')
        )
        for msg in reversed(messages):
            if msg.get('role') == 'assistant' and msg.get('content', '').strip():
                return msg['content']
        return None
    except Exception as e:
        logger.warning(f'Failed to retrieve primary answer: {e}')
        return None


async def build_prompt_with_memory(
    user_content: str,
    chat_sessionId: str = 'default',
) -> tuple[str, str, dict]:
    """
    MAIN ENTRY POINT: Build Complete Augmented Prompt with Multi-Source Context

    This orchestrates ALL context sources:
      1. Session configuration (rules, limits, custom instructions)
      2. File attachments (inline + stored)
      3. URL extraction (if URLs in message)
      4. Conversation history (for continuity)
      5. Semantic memory (past knowledge)
      6. Web search (factual grounding)
      7. Follow-up detection (reference resolution)

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
    follow_up_enabled = config.get('followUpEnabled', True)

    logger.info(
        'Config loaded: web=%s, hist=%s, mem=%s, file=%s, follow_up=%s',
        local_web_limit,
        local_history_limit,
        local_memory_limit,
        local_file_limit,
        follow_up_enabled,
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
    # Step 5: Get primary assistant answer for follow-up
    # ─────────────────────────────────────────────────────────────────────
    primary_answer = _get_primary_assistant_answer(chat_sessionId)
    if primary_answer:
        logger.info(
            'Primary answer: %s chars available',
            len(primary_answer),
            extra={'session_id': chat_sessionId},
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
        follow_up_enabled=follow_up_enabled,
    )
    if hist_result['content']:
        blocks.append(hist_result['content'])
        sources_considered['history'] = hist_result.get('confidence', CONFIDENCE_HISTORY)
        logger.info(
            'History: %s msgs',
            hist_result['metadata'].get('messages_count', 0),
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

    # 7.3: Build WEB context (factual source)
    web_result = await build_context_for_source(
        session_id=chat_sessionId,
        source=ContextSource.WEB,
        user_content=user_content,
        extracted_key_points=extracted_key_points,
        context_limits=context_limits,
        follow_up_enabled=follow_up_enabled,
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

    # 7.4: Build MEMORY context (factual source)
    mem_result = await build_context_for_source(
        session_id=chat_sessionId,
        source=ContextSource.MEMORY,
        user_content=user_content,
        context_limits=context_limits,
        follow_up_enabled=follow_up_enabled,
    )
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
                follow_up_enabled=follow_up_enabled,
            )
            if fres['content']:
                file_blocks.append(fres['content'])
                factual_blocks.append(fres['content'])
                sources_considered['file'] = fres.get('confidence', CONFIDENCE_FILE)
                logger.info('File: %s', finfo.get('filename'), extra={'session_id': chat_sessionId})

        if file_blocks:
            blocks = file_blocks + blocks

    # ─────────────────────────────────────────────────────────────────────
    # Step 8: Determine follow-up mode (after source collection)
    # ─────────────────────────────────────────────────────────────────────
    is_follow_up = False
    continuation_hint = ''

    if follow_up_enabled and primary_answer:
        is_follow_up = True
        continuation_hint = CONTINUATION_HINT_FOLLOWUP

        # Insert PRIMARY CONTEXT for reference resolution
        ins_at = 1 if file_infos else 0
        blocks.insert(ins_at, PRIMARY_CONTEXT_HEADER + '\n\n' + primary_answer)
        logger.info(
            'Follow-up mode ACTIVE: Primary context injected', extra={'session_id': chat_sessionId}
        )

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
        'overview': {
            'available': bool(topic_summary),
        },
        'follow_up': {
            'available': is_follow_up,
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
    summary = f"""CONTEXT SUMMARY
Query: {user_content[:100]}{'...' if len(user_content) > 100 else ''}
Sources: {', '.join(sources_considered.keys())}
Files: {len(file_infos)}
Confidence: {overall_confidence:.2f}"""

    context = '\n\n'.join(blocks) if blocks else 'No context available.'

    prompt = f"""{continuation_hint}

{'=' * 80}

{summary}

{'=' * 80}

{context}

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
        'followup_enabled': follow_up_enabled,
        'is_follow_up': is_follow_up,
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
    primary_assistant_answer: str | None = None,
    extracted_key_points: list[str] | None = None,
    selected_file_id: str | None = None,
    file_info: dict | None = None,
    context_limits: dict | None = None,
    follow_up_enabled: bool = True,
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
                session_id, user_content, extracted_key_points, context_limits
            )
        elif source == ContextSource.FOLLOW_UP:
            return _build_followup_context(primary_assistant_answer)
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


async def _build_memory_context(session_id: str, query: str, limits: dict) -> dict:
    """Build context from semantic memories."""
    try:
        # Run blocking embedding + Mongo work off the event loop
        important_memories = await asyncio.to_thread(
            list_memories_by_category, session_id, 'important'
        )
        memories = await asyncio.to_thread(
            search_memories,
            session_id,
            query,
            limits.get('memory_items', MEMORY_RESULTS_LIMIT),
        )

        combined = []
        seen_ids = set()

        for m in important_memories:
            mem_id = m.get('id') or m.get('_id')
            if mem_id in seen_ids:
                continue
            seen_ids.add(mem_id)
            combined.append(m)

        for m in memories:
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
        for msg in to_include:
            role = msg.get('role', '').upper()
            content = msg.get('content', '')[:HISTORY_MAX_PER_MSG]
            lines.append(f'{role}: {content}')

        formatted = HISTORY_CONTEXT_HEADER + '\n' + '\n'.join(lines)

        return {
            'content': formatted,
            'confidence': CONFIDENCE_HISTORY,
            'source': ContextSource.HISTORY,
            'metadata': {'messages_count': len(lines)},
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


async def _build_web_context(session_id: str, query: str, key_points: list, limits: dict) -> dict:
    """Build context from web search results."""
    try:
        # Build search queries: primary + key points
        search_queries = [query]
        if key_points and isinstance(key_points, list):
            # Only add if key_points are strings
            for kp in key_points[:2]:
                if isinstance(kp, str):
                    search_queries.append(kp)

        logger.debug(
            f'Web search with {len(search_queries)} queries', extra={'session_id': session_id}
        )

        results = []
        for sq in search_queries[:3]:
            if not isinstance(sq, str) or not sq.strip():
                logger.warning(
                    f'Skipping invalid search query: {type(sq)}', extra={'session_id': session_id}
                )
                continue

            try:
                res = await maybe_web_search(sq, session_id=session_id)
                if res and isinstance(res, list):
                    results.extend(res[: limits.get('web_results', WEB_RESULTS_LIMIT)])
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


def _build_followup_context(primary_answer: str) -> dict:
    """Build follow-up reference context."""
    if not primary_answer:
        return {
            'content': '',
            'confidence': 0.0,
            'source': ContextSource.FOLLOW_UP,
            'metadata': {},
            'warning': 'No previous answer',
        }

    return {
        'content': PRIMARY_CONTEXT_HEADER + '\n\n' + primary_answer,
        'confidence': CONFIDENCE_FOLLOW_UP,
        'source': ContextSource.FOLLOW_UP,
        'metadata': {'answer_length': len(primary_answer)},
        'warning': None,
    }


# ════════════════════════════════════════════════════════════════════════════════
# RESPONSE GENERATION & VALIDATION
# ════════════════════════════════════════════════════════════════════════════════


def _build_reasoning_prompt(
    user_query: str,
    assistant_answer: str,
    sources_used: dict,
    loaded_sources: dict,
    confidence: float,
    unverified_count: int,
    guard_eval: dict,
    is_follow_up: bool = False,
    primary_answer: str = None,
) -> str:
    """
    Build a prompt asking the model to explain its reasoning process.

    This generates a self-reflective prompt where the model analyzes
    how it arrived at the answer it just gave.
    """
    # Build source summary
    source_summary = []
    if 'file' in sources_used:
        file_count = loaded_sources.get('file', {}).get('count', 0)
        source_summary.append(f'- FILE: {file_count} file(s)')
    if 'memory' in sources_used:
        mem_count = loaded_sources.get('memory', {}).get('count', 0)
        source_summary.append(f'- MEMORY: {mem_count} item(s)')
    if 'web' in sources_used:
        web_count = loaded_sources.get('web', {}).get('count', 0)
        source_summary.append(f'- WEB: {web_count} result(s)')
    if 'history' in sources_used:
        source_summary.append('- HISTORY: Conversation context')
    if 'url-extract' in sources_used:
        source_summary.append('- URL: Extracted content')
    if is_follow_up and primary_answer:
        source_summary.append(
            f'- FOLLOW-UP MODE HAS BEEN APPLIED: Previous answer available as primary context ({len(primary_answer)} chars)'
        )

    sources_text = (
        '\n'.join(source_summary)
        if source_summary
        else 'No external sources (used training knowledge)'
    )

    risk = guard_eval.get('risk', 'NONE') if guard_eval else 'NONE'
    risk_text = (
        f'{risk} (found {unverified_count} unverified entities)' if unverified_count > 0 else 'NONE'
    )

    reasoning_phase = REASONING_PHASE_SYSTEM

    prompt = f"""You just answered a user's question. Now explain your reasoning process step by step.

USER ASKED: "{user_query}"

YOUR ANSWER WAS: "{assistant_answer[:800]}{'...' if len(assistant_answer) > 800 else ''}"

SOURCES YOU HAD AVAILABLE:
{sources_text}

YOUR CONFIDENCE: {confidence:.0%}
VERIFICATION RISK: {risk_text}

{reasoning_phase}
"""

    return prompt


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
      2. Stream answer from model
      3. Validate against factual sources
      4. Apply system-level verification
      5. Save to database
      6. Auto-save important memories
      7. Generate reasoning (if enabled)
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

    reasoning_tracker.log_step(
        thought='User message received and saved',
        action='SAVE_MESSAGE',
        source='database',
        confidence=1.0,
        information=f'Message length: {len(content)} chars',
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Build augmented prompt
    # ─────────────────────────────────────────────────────────────────────
    system_prompt, final_prompt, context_meta = await build_prompt_with_memory(
        user_content=content,
        chat_sessionId=session_id,
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
    # Step 3: Generate answer and STREAM IMMEDIATELY
    # ─────────────────────────────────────────────────────────────────────
    assistant_answer = ''

    try:
        async for token in stream_ollama(
            prompt=final_prompt,
            model=model,
            system=system_prompt,
        ):
            assistant_answer += token
            # Stream live tokens to user immediately
            yield json.dumps({'type': 'token', 'data': token})

        logger.info(
            'Answer streamed: %s chars', len(assistant_answer), extra={'session_id': session_id}
        )

        # Send signal that answer streaming is complete
        yield json.dumps({'type': 'answer_complete'})
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

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Generate reasoning by asking the MODEL to explain itself (if enabled)
    # ─────────────────────────────────────────────────────────────────────
    reasoning_text = ''

    if reasoning_enabled:
        yield json.dumps({'type': 'reasoning_starting', 'data': 'Generating reasoning...'})

        # Get follow-up information from metadata
        is_follow_up = context_meta.get('is_follow_up', False)
        primary_answer = _get_primary_assistant_answer(session_id) if is_follow_up else None

        reasoning_prompt = _build_reasoning_prompt(
            user_query=content,
            assistant_answer=assistant_answer,
            sources_used=sources,
            loaded_sources=loaded_sources,
            confidence=overall_confidence,
            unverified_count=len(unverified),
            guard_eval=guard_eval,
            is_follow_up=is_follow_up,
            primary_answer=primary_answer,
        )

        logger.info('Generating reasoning from model', extra={'session_id': session_id})
        try:
            async for token in stream_ollama(
                prompt=reasoning_prompt,
                model=model,
                system='You are an AI assistant explaining your reasoning process. Be clear, honest, and concise.',
            ):
                reasoning_text += token
                # Stream reasoning in real-time
                yield json.dumps({'type': 'reasoning_token', 'data': token})

            logger.info(
                'Reasoning generated: %s chars',
                len(reasoning_text),
                extra={'session_id': session_id},
            )
        except Exception as e:
            logger.error(f'Reasoning generation failed: {e}')
            reasoning_text = '[Reasoning generation failed]'
    else:
        logger.info('Reasoning generation skipped (disabled)', extra={'session_id': session_id})

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
