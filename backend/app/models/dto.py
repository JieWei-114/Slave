from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class ContextSource(str, Enum):
    # Enumeration of context sources user can select
    FOLLOW_UP = 'follow-up'
    MEMORY = 'memory'
    FILE = 'file'
    HISTORY = 'history'
    WEB = 'web'


class ReasoningChainSummary(BaseModel):
    # Summary of reasoning chain used for UI display
    steps_count: int | None = None
    sources_used: list[str] = Field(default_factory=list)
    final_confidence: float | None = None
    uncertainty_flags: list[str] = Field(default_factory=list)
    duration_ms: float | None = None
    step_details: list[dict] = Field(default_factory=list)


class MessageMetadata(BaseModel):
    # Metadata attached to each message
    source_used: str
    sources_considered: dict[str, float]
    confidence: float | None = None
    supplemented_with: list[str] = Field(default_factory=list)
    web_search_queries: list[str] = Field(default_factory=list)
    file_referenced: str | None = None
    uncertainty_flags: list[dict] = Field(default_factory=list)

    # Confidence tracking
    confidence_initial: float | None = None
    confidence_final: float | None = None

    # Validation metadata
    reasoning: str | None = None
    reasoning_chain: ReasoningChainSummary | None = None
    reasoning_veto: dict | None = None
    factual_guard: dict | None = None
    source_conflicts: list[dict] = Field(default_factory=list)

    # Additional context
    source_relevance: dict[str, float] = Field(default_factory=dict)
    loaded_sources: dict | None = None
    has_factual_content: bool | None = None


class UncertaintyReport(BaseModel):
    # When assistant is unsure about something
    aspect: str
    confidence: float
    suggested_actions: list[str]


class CreateSessionRequest(BaseModel):
    title: str


class RenameSessionRequest(BaseModel):
    title: str


class CreateMemoryRequest(BaseModel):
    content: str
    session_id: str
    category: str | None = None


class RulesConfig(BaseModel):
    searxng: bool = True
    duckduckgo: bool = True
    tavily: bool = False
    serper: bool = False
    tavilyExtract: bool = False
    localExtract: bool = True

    advanceSearch: bool = False
    advanceExtract: bool = False

    webSearchLimit: int | None = None
    memorySearchLimit: int | None = None
    historyLimit: int | None = None
    fileUploadMaxChars: int | None = None

    customInstructions: str = ''
    followUpEnabled: bool = True
    reasoningEnabled: bool = False


class ReorderSessionsRequest(BaseModel):
    sessionIds: List[str]


class FileAttachment(BaseModel):
    # File attachment metadata
    id: str
    session_id: str
    filename: str
    file_type: str
    size_bytes: int
    size_chars: int
    content: str
    extracted_metadata: dict = Field(default_factory=lambda: {'key_points': [], 'structure': ''})
    uploaded_at: datetime
    expires_at: datetime | None = None
    suggested_facts: list[dict] = Field(default_factory=list)


class AttachFileRequest(BaseModel):
    filename: str
    content: str
