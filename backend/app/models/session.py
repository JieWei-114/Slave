from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class SessionRules(BaseModel):
    searxng: bool = True
    duckduckgo: bool = True
    tavily: bool = False
    serper: bool = False
    tavilyExtract: bool = False
    localExtract: bool = True
    followUpEnabled: bool = True
    reasoningEnabled: bool = False


class ChatSession(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    rules: SessionRules = Field(default_factory=SessionRules)
    topic_break_at: datetime | None = None
    topic_summary: str | None = None


class AssistantMeta(BaseModel):
    reasoning: str | None = None
    citations: list[dict] | None = None
    tools_used: list[str] | None = None


class MessageAttachment(BaseModel):
    filename: str
    content: str


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    meta: AssistantMeta | None = None
    attachment: MessageAttachment | None = None
