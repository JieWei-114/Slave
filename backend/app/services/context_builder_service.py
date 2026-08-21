"""
Context Builder Service

Handles context assembly from multiple sources:
- History, memory, web search, files, URL extraction
- Source layer separation (factual vs contextual)
- Follow-up mode resolution
- Confidence calculation
"""

import logging
import re

from app.config.settings import settings
from app.utils.text_utils import calculate_sentence_score, split_into_sentences

logger = logging.getLogger(__name__)

# Centralized limits from settings
PROMPT_MAX_TOTAL = settings.CHAT_PROMPT_MAX_TOTAL_CHARS
HISTORY_LIMIT = settings.CHAT_HISTORY_LIMIT
HISTORY_MAX_PER_MSG = settings.CHAT_HISTORY_MAX_CHARS_PER_MSG
HISTORY_TOTAL_MAX = settings.CHAT_HISTORY_TOTAL_MAX_CHARS
MAX_ASSISTANT_CONTEXT = settings.CHAT_HISTORY_MAX_ASSISTANT_CONTEXT
MEMORY_RESULTS_LIMIT = settings.CHAT_MEMORY_RESULTS_LIMIT
MEMORY_TOTAL_MAX = settings.CHAT_MEMORY_TOTAL_MAX_CHARS
WEB_RESULTS_LIMIT = settings.CHAT_WEB_RESULTS_LIMIT
WEB_SNIPPET_MAX = settings.CHAT_WEB_SNIPPET_MAX_CHARS
WEB_TOTAL_MAX = settings.CHAT_WEB_TOTAL_MAX_CHARS
EXTRACT_TOTAL_MAX = settings.CHAT_EXTRACT_TOTAL_MAX_CHARS
FILE_CONTENT_MAX = settings.CHAT_FILE_CONTENT_MAX_CHARS
ENABLE_RANKING = settings.CHAT_ENABLE_RESULT_RANKING
SYSTEM_INSTRUCTIONS = settings.CHAT_SYSTEM_INSTRUCTIONS
TEXT_MIN_LENGTH = settings.TEXT_MIN_LENGTH_FOR_PROCESSING
TEXT_MIN_SENTENCE_LENGTH = settings.TEXT_MIN_SENTENCE_LENGTH

# Confidence levels
CONFIDENCE_FILE = settings.CONFIDENCE_FILE
CONFIDENCE_MEMORY = settings.CONFIDENCE_MEMORY
CONFIDENCE_HISTORY = settings.CONFIDENCE_HISTORY
CONFIDENCE_WEB = settings.CONFIDENCE_WEB
CONFIDENCE_NONE = settings.CONFIDENCE_NONE


def extract_key_points(text: str, max_points: int = None) -> list[str]:
    """
    Extract key points/sentences from extracted content for smart web search.

    """
    if max_points is None:
        max_points = settings.EXTRACT_KEY_POINTS_MAX

    if not text or len(text) < TEXT_MIN_LENGTH:
        return []

    # Split into sentences
    sentences = split_into_sentences(text, min_length=TEXT_MIN_SENTENCE_LENGTH)

    if not sentences:
        return []

    # Score sentences using utility function
    scored = []
    total_sentences = min(len(sentences), settings.KEY_POINT_EXTRACTION_SAMPLE_SIZE)

    for idx, sent in enumerate(sentences[:total_sentences]):
        score = calculate_sentence_score(idx, len(sent), total_sentences)
        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    key_points = [s[1] for s in scored[:max_points]]

    logger.debug(f'Extracted {len(key_points)} key points from content')
    return key_points


def extract_file_content(user_content: str) -> tuple[str, dict | None]:
    """
    Extract uploaded file content from user message.

    """
    if not user_content:
        return user_content, None

    # Pattern: [Attached file: filename.ext]\nfile_content
    pattern = r'\n\n\[Attached file: ([^\]]+)\]\n(.+)'
    match = re.search(pattern, user_content, re.DOTALL)

    if not match:
        return user_content, None

    filename = match.group(1)
    file_content = match.group(2)

    # Remove file section from message
    clean_message = user_content[: match.start()].strip()

    file_info = {
        'filename': filename,
        'content': file_content,
        'length': len(file_content),
        'type': 'unknown',
    }

    # Detect file type from extension
    if filename.lower().endswith('.pdf'):
        file_info['type'] = 'PDF'
    elif filename.lower().endswith(('.docx', '.doc')):
        file_info['type'] = 'Word'
    elif filename.lower().endswith(('.txt', '.md')):
        file_info['type'] = 'Text'
    elif filename.lower().endswith(('.json', '.yaml', '.yml')):
        file_info['type'] = 'Config'
    elif filename.lower().endswith(('.py', '.js', '.ts', '.java', '.cpp')):
        file_info['type'] = 'Code'

    return clean_message, file_info


def rank_search_results(results: list[dict], query: str) -> list[dict]:
    """
    Rank search results by relevance to query.
    Uses simple keyword matching scoring.

    """
    if not ENABLE_RANKING or not results:
        return results

    query_terms = set(query.lower().split())
    # Remove common words that don't add relevance
    stop_words = {
        'the',
        'a',
        'an',
        'and',
        'or',
        'but',
        'in',
        'on',
        'at',
        'to',
        'for',
        'of',
        'with',
        'by',
    }
    query_terms = query_terms - stop_words

    if not query_terms:
        return results

    scored = []
    for r in results:
        # Score based on title + snippet content
        title = r.get('title', '').lower()
        snippet = r.get('snippet', '').lower()
        content = snippet

        text = f'{title} {title} {content}'  # Weight title 2x

        # Count term matches
        score = sum(1 for term in query_terms if term in text)

        # Boost score if terms appear in title
        title_boost = sum(2 for term in query_terms if term in title)

        total_score = score + title_boost
        scored.append((total_score, r))

    # Sort by score descending, keep original order for ties
    scored.sort(key=lambda x: x[0], reverse=True)

    return [r for _, r in scored]


def calculate_weighted_confidence(
    sources_considered: dict[str, float],
    source_relevance: dict[str, float] | None = None,
    factual_sources_only: bool = True,
    loaded_sources: dict | None = None,
) -> float:
    """
    Calculate confidence based on FACTUAL sources only.
    Excludes contextual helpers (history, follow-up) from confidence calculation.

    """
    if not sources_considered:
        return CONFIDENCE_NONE

    if source_relevance is None:
        source_relevance = {src: 1.0 for src in sources_considered.keys()}

    # Filter out contextual-only sources
    if factual_sources_only:
        factual_sources = {
            k: v
            for k, v in sources_considered.items()
            if k not in {'history', 'semantic_history', 'overview', 'follow-up'}
        }

        if not factual_sources:
            logger.info('No factual sources available, using default confidence')
            return CONFIDENCE_NONE

        sources_considered = factual_sources
        logger.info(
            'Factual sources for confidence: %s (excluded contextual: history, follow-up)',
            list(sources_considered.keys()),
        )

    # Check frozen sources to ensure they actually have content
    if loaded_sources:
        valid_factual_sources = {}
        for source_name, confidence in sources_considered.items():
            source_data = loaded_sources.get(source_name, {})

            # Verify source has actual content
            if source_name == 'file':
                if source_data.get('available') and source_data.get('count', 0) > 0:
                    valid_factual_sources[source_name] = confidence
            elif source_name == 'memory':
                if source_data.get('available') and source_data.get('count', 0) > 0:
                    valid_factual_sources[source_name] = confidence
            elif source_name == 'web':
                if source_data.get('available') and source_data.get('count', 0) > 0:
                    valid_factual_sources[source_name] = confidence
            else:
                # Other sources (url-extract, etc.)
                valid_factual_sources[source_name] = confidence

        if not valid_factual_sources:
            logger.warning('No valid factual sources with content')
            return CONFIDENCE_NONE

        sources_considered = valid_factual_sources

    # Weight by relevance × confidence
    weighted_scores = []
    for source_name, confidence in sources_considered.items():
        relevance = source_relevance.get(source_name, 0.5)
        weighted_score = confidence * relevance
        weighted_scores.append(weighted_score)

    # Use average of weighted scores
    avg_confidence = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0

    return min(max(avg_confidence, 0.0), 1.0)  # Clamp to [0, 1]
