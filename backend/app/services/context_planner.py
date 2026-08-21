"""
Context Planner

One small LLM call that decides, before context building, which sources are
worth retrieving for the current user message:
- web_queries: standalone web search phrases (empty = skip web)
- history_query: semantic retrieval phrase for past conversation (null = skip)
- memory_query: semantic retrieval phrase for long-term memory (null = skip)
- relevant_files: subset of uploaded filenames worth injecting
- needs_vision: whether the request needs image understanding

On any failure (or when settings.PLANNER_ENABLED is False) a static fallback
plan is returned that approximates the pre-planner behavior.
"""

import json
import logging
from typing import Optional

from app.config.settings import settings
from app.services.ollama_service import call_ollama_once

logger = logging.getLogger(__name__)

# JSON schema for constrained decoding (Ollama structured outputs) — small
# models won't follow "output JSON" prose instructions without it
PLAN_JSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'web_queries': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 3},
        'history_query': {'type': ['string', 'null']},
        'memory_query': {'type': ['string', 'null']},
        'relevant_files': {'type': 'array', 'items': {'type': 'string'}},
        'needs_vision': {'type': 'boolean'},
    },
    'required': ['web_queries', 'history_query', 'memory_query', 'relevant_files', 'needs_vision'],
}

PLANNER_PROMPT_TEMPLATE = """You are a context planner for an AI assistant. Decide what context to retrieve before answering the user.

RECENT CONVERSATION:
{recent}

UPLOADED FILES: {filenames}

USER MESSAGE: {user_message}

Output STRICT JSON only, no markdown, no explanation.

Example for "hi" or chitchat:
{{"web_queries": [], "history_query": null, "memory_query": null, "relevant_files": [], "needs_vision": false}}

Example for "how does the new EU AI act affect startups?":
{{"web_queries": ["EU AI Act startup obligations", "EU AI Act 2026 summary"], "history_query": "EU AI act discussion", "memory_query": null, "relevant_files": [], "needs_vision": false}}

Example for "what do you know about me?" or anything about the user themselves:
{{"web_queries": [], "history_query": "personal details the user shared", "memory_query": "user personal facts preferences name", "relevant_files": [], "needs_vision": false}}

Never use meta words from this prompt (like "recent conversation" or "user message") as web_queries. Set memory_query whenever the question involves the user's own identity, preferences, or past statements.

Rules:
- web_queries: up to 3 standalone web search phrases. Use [] for greetings, chitchat, or questions answerable from the conversation or files alone.
- history_query: a semantic retrieval phrase for finding relevant past conversation, or null if the recent messages are enough.
- memory_query: a semantic retrieval phrase for the user's long-term memory (preferences, facts about them), or null if irrelevant.
- relevant_files: subset of UPLOADED FILES actually relevant to the message. [] if none are relevant.
- needs_vision: true only if answering requires looking at an image."""


def build_fallback_plan(user_message: str, attachments_meta: list[dict]) -> dict:
    """Static plan approximating pre-planner behavior."""
    return {
        'web_queries': [user_message] if len(user_message) > 20 else [],
        'history_query': user_message,
        'memory_query': user_message,
        'relevant_files': [a.get('filename') for a in attachments_meta if a.get('filename')],
        'needs_vision': False,
        'fallback': True,
    }


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith('```'):
        # Drop opening fence (with optional language tag) and closing fence
        lines = text.split('\n')
        lines = lines[1:]
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    # Fall back to the outermost JSON object if extra prose surrounds it
    if not text.startswith('{'):
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


async def plan_context(
    user_message: str,
    recent_messages: list[dict],
    attachments_meta: list[dict],
    model: str,
) -> dict:
    """
    Ask the model for a retrieval plan. Never raises: on any failure the
    fallback plan is returned (marked with 'fallback': True).
    """
    if not settings.PLANNER_ENABLED:
        return build_fallback_plan(user_message, attachments_meta)

    filenames = [a.get('filename') for a in attachments_meta if a.get('filename')]

    recent_lines = []
    for msg in recent_messages[-4:]:
        role = (msg.get('role') or 'user').upper()
        content = (msg.get('content') or '')[:200]
        recent_lines.append(f'{role}: {content}')
    recent = '\n'.join(recent_lines) if recent_lines else '(none)'

    prompt = PLANNER_PROMPT_TEMPLATE.format(
        recent=recent,
        filenames=json.dumps(filenames) if filenames else '(none)',
        user_message=user_message,
    )

    try:
        raw = await call_ollama_once(
            prompt=prompt, model=model, system=None, json_schema=PLAN_JSON_SCHEMA
        )
        cleaned = _strip_fences(raw)
        # Small models may still wrap prose around the object — take the
        # outermost {...} span as a last resort before giving up.
        if not cleaned.startswith('{'):
            start, end = cleaned.find('{'), cleaned.rfind('}')
            if start != -1 and end > start:
                cleaned = cleaned[start : end + 1]
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError('planner output is not a JSON object')

        web_queries = parsed.get('web_queries') or []
        if not isinstance(web_queries, list):
            web_queries = []
        web_queries = [
            q.strip()
            for q in web_queries
            if isinstance(q, str) and q.strip() and q.strip().strip('.') != ''
        ][:3]

        def _clean_query(value: object) -> Optional[str]:
            if not isinstance(value, str):
                return None
            text = value.strip()
            # Drop empty strings and copied '...' placeholders
            if not text or text.strip('.') == '':
                return None
            return text

        history_query = _clean_query(parsed.get('history_query'))
        memory_query = _clean_query(parsed.get('memory_query'))

        relevant_files = parsed.get('relevant_files') or []
        if not isinstance(relevant_files, list):
            relevant_files = []
        # Keep only filenames that actually exist in the attachments
        relevant_files = [f for f in relevant_files if isinstance(f, str) and f in filenames]

        plan = {
            'web_queries': web_queries,
            'history_query': history_query,
            'memory_query': memory_query,
            'relevant_files': relevant_files,
            'needs_vision': bool(parsed.get('needs_vision', False)),
            'fallback': False,
        }
        logger.info('Context plan: %s', plan)
        return plan
    except Exception as e:
        logger.warning(f'Context planner failed, using fallback plan: {e}')
        return build_fallback_plan(user_message, attachments_meta)
