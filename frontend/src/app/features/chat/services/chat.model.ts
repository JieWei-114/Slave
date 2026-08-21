/**
 * Chat data models and types
 * These interfaces define the contract between frontend and backend API
 * Must stay synchronized with backend/app/models/dto.py
 */

import type { RulesConfig } from '../../rules/service/rules.model';

// Message role in conversation
export type ChatRole = 'user' | 'assistant';

/**
 * Metadata returned from backend for each assistant response
 * Provides transparency into context sources and confidence
 * Matches backend MessageMetadata model
 */
export interface MessageMetadata {
  source_used?: string; // Actual source used (combined, follow-up, memory, file, history, web)
  sources_considered?: Record<string, number>; // All sources evaluated with scores
  source_relevance?: Record<string, number>; // Source relevance scores
  confidence?: number; // Overall confidence (deprecated, use confidence_final)
  supplemented_with?: string[]; // Additional sources used to supplement
  uncertainty_flags?: UncertaintyReport[]; // List of uncertainties with details
  web_search_queries?: string[]; // Queries issued for web search
  file_referenced?: string | null; // Filename referenced in response

  has_factual_content?: boolean; // Whether FILE, MEMORY, or WEB were used
  loaded_sources?: {
    file?: { available: boolean; count: number; files?: any[] };
    memory?: { available: boolean; count: number };
    web?: { available: boolean; count: number };
    history?: { available: boolean };
    follow_up?: { available: boolean };
  };

  // Confidence tracking
  confidence_initial?: number; // Confidence before veto/guard
  confidence_final?: number; // Confidence after all reductions

  // Validation metadata
  reasoning?: string; // Raw LLM-generated reasoning explanation
  reasoning_streaming?: boolean; // UI-only flag while reasoning tokens stream
  reasoning_chain?: ReasoningChainSummary; // Structured reasoning steps
  reasoning_veto?: {
    level: string; // 'hard', 'soft', or 'none'
    signals?: string[]; // Detected veto signals
    confidence_cap?: number; // Confidence cap applied
    reason: string; // Explanation
  };
  factual_guard?: {
    risk: string; // 'high', 'med', 'low', or 'none'
    unverified_entities?: string[]; // Entities not found in sources
    cap: number; // Confidence cap applied
  };
  source_conflicts?: {
    sources: string[]; // Conflicting sources
    reason: string; // Why they conflict
    confidence_reduction?: number; // How much confidence was reduced
  }[];
}

/**
 * Single chat message
 * Stored in session and displayed in chat UI
 */
export interface ChatMessage {
  role: ChatRole; // 'user' or 'assistant'
  content: string; // Message text content
  created_at: string; // ISO timestamp
  meta?: MessageMetadata; // Metadata (only for assistant messages)
  attachment?: {
    filename: string;
    content: string;
  };
  remembered?: boolean; // UI flag for memory indicator
}

/**
 * Summary of reasoning chain used for UI display
 */
export interface ReasoningChainSummary {
  steps_count?: number;
  sources_used?: string[];
  final_confidence?: number;
  uncertainty_flags?: string[];
  duration_ms?: number | null;
  step_details?: {
    step: number;
    action: string;
    source: string;
    confidence: number;
  }[];
}

/**
 * Uncertainty report when assistant lacks confidence
 * Prompts user to provide more info or trigger web search
 * Matches backend UncertaintyReport model
 */
export interface UncertaintyReport {
  aspect: string; // What the assistant is uncertain about
  confidence: number; // Confidence level (0-1)
  suggested_actions: string[]; // Suggested actions: search_web, ask_user, use_history
}

/**
 * Chat session representation
 * Contains session metadata and message history
 */
export interface ChatSession {
  id: string; // Unique session identifier
  title: string; // Session title (editable)
  messages: ChatMessage[]; // Array of messages in conversation
  rules?: RulesConfig; // Session-specific rules configuration
  isTemp?: boolean; // True for local-only sessions not yet persisted to backend
  hasMore?: boolean; // Flag for pagination (more messages available)
  loadingMore?: boolean; // Loading state for pagination
}

/**
 * Lightweight session DTO for listing sessions
 * Used in sidebar session list
 * Matches backend response from GET /chat/sessions
 */
export interface ChatSessionDto {
  id: string;
  title: string;
  updated_at: string; // ISO timestamp
}

/**
 * AI model configuration
 * Available models are hardcoded but could be fetched from backend
 * Backend returns this from GET /chat/models
 */
export interface AIModel {
  id: string; // Model identifier for Ollama (e.g., 'gemma3:1b')
  name: string; // Display name
  description?: string; // Optional description
}

/**
 * Available AI models
 * Should be fetched from backend to stay synchronized
 */
export const AVAILABLE_MODELS: AIModel[] = [
  {
    id: 'gemma3:1b',
    name: 'Gemma 3 (1B)',
    description: 'Fast and efficient',
  },
  {
    id: 'qwen2.5:3b',
    name: 'qwen 2.5 (3B)',
    description: 'Balanced performance',
  },
];

// Default model used when no selection is made
export const DEFAULT_MODEL = AVAILABLE_MODELS[0];
