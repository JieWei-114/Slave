"""
Prompt templates and instruction snippets used by chat_service.

"""

# Used in build_context_for_source() when building FILE context block
FILE_CONTEXT_INSTRUCTION = "\n\nIMPORTANT: Read and analyze the above file carefully. Use its contents to answer the user's query."

# Used in build_context_for_source() when building HISTORY context block
HISTORY_CONTEXT_HEADER = 'CONVERSATION HISTORY'

# Used in build_context_for_source() when building WEB context block
WEB_CONTEXT_HEADER = 'WEB SEARCH RESULTS'

# Used in build_context_for_source() when building MEMORY context block
MEMORY_CONTEXT_HEADER = 'RELEVANT MEMORIES'

# ============================================================
# PHASE 1: INTERNAL REASONING PROMPTS
# ============================================================

# System-level instruction for reasoning phase
REASONING_PHASE_SYSTEM = """
Think step by step before answering.
1) What is the user really asking?
2) Which provided context sections are relevant and what do they say?
3) Any contradictions between sources (memories override conversation history)?
4) What should the answer be?
Be concise — a few short paragraphs. Do not write the final answer yet; this is
your private analysis.
"""
