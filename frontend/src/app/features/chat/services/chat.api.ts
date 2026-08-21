/**
 * Chat API Service
 * Includes both REST API calls and Server-Sent Events (SSE) for streaming
 */
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';

import { AppConfigService } from '../../../core/services/app-config.services';
import { ChatSession, ChatSessionDto } from './chat.model';

@Injectable({ providedIn: 'root' })
export class ChatApi {
  private http = inject(HttpClient);
  private config = inject(AppConfigService);

  /**
   * Send a message to a chat session (non-streaming)
   */
  sendMessage(sessionId: string, content: string) {
    return this.http.post<{ reply: string }>(
      `${this.config.apiBaseUrl}/chat/${sessionId}/message`,
      { content },
    );
  }

  /**
   * Get all chat sessions (metadata only, no messages)
   */
  getSessions() {
    return this.http.get<ChatSessionDto[]>(`${this.config.apiBaseUrl}/chat/sessions`);
  }

  /**
   * Get full session data including messages
   */
  getSessionbyId(sessionId: string) {
    return this.http.get<any>(`${this.config.apiBaseUrl}/chat/${sessionId}`);
  }

  /**
   * Create a new chat session
   */
  createSession(title = 'New chat') {
    return this.http.post<ChatSession>(`${this.config.apiBaseUrl}/chat/session`, { title });
  }

  /**
   * Rename an existing chat session
   */
  renameSession(sessionId: string, title: string) {
    return this.http.patch<{ id: string; title: string }>(
      `${this.config.apiBaseUrl}/chat/${sessionId}/rename`,
      { title },
    );
  }

  /**
   * Delete a chat session permanently
   */
  deleteSession(sessionId: string) {
    return this.http.delete(`${this.config.apiBaseUrl}/chat/${sessionId}`);
  }

  /**
   * Start a new topic in a session — the backend summarises earlier
   * discussion and marks a topic break. 400 if the session has no messages.
   */
  newTopic(sessionId: string, model: string) {
    return this.http.post<{ topic_break_at: string; summary: string }>(
      `${this.config.apiBaseUrl}/chat/${sessionId}/new-topic`,
      { model },
    );
  }

  /**
   * Reorder chat sessions (for sidebar drag-and-drop)
   */
  reorderSessions(sessionIds: string[]) {
    return this.http.post<void>(`${this.config.apiBaseUrl}/chat/reorder`, { sessionIds });
  }

  /**
   * Stream AI response in real-time using Server-Sent Events (SSE)
   * Uses fetch() POST with a JSON body and hand-rolled SSE parsing over
   * the response ReadableStream. Returns a cancel function (AbortController).
   */
  streamMessage(
    sessionId: string,
    content: string,
    model: string,
    onToken: (t: string) => void,
    onReasoning: (r: string) => void,
    onDone: () => void,
    onMetadata?: (meta: any) => void,
    onVerification?: (status: { type: string; data?: any }) => void,
    reasoningEnabled = false,
    onError?: (err: unknown) => void,
    onPlan?: (plan: any) => void,
  ): () => void {
    const url = `${this.config.apiBaseUrl}/chat/${sessionId}/stream`;
    const controller = new AbortController();
    let finished = false;

    const finishDone = () => {
      if (finished) return;
      finished = true;
      onDone();
    };

    const finishError = (err: unknown) => {
      if (finished) return;
      finished = true;
      console.error('SSE error', err);
      onError?.(err);
    };

    // Dispatch a single parsed SSE event to the right callback
    const dispatch = (eventName: string, data: string): void => {
      const type = eventName || 'message';
      switch (type) {
        case 'token':
          onToken(JSON.parse(data));
          break;
        case 'answer_complete':
          onVerification?.({ type: 'answer_complete' });
          break;
        case 'verification_starting':
        case 'verification_complete':
        case 'reasoning_starting':
          onVerification?.({ type, data: data ? JSON.parse(data) : undefined });
          break;
        case 'planning':
          // Planner started — no payload of interest yet
          break;
        case 'plan': {
          if (onPlan) {
            const payload = data ? JSON.parse(data) : {};
            if (payload.data) onPlan(payload.data);
          }
          break;
        }
        case 'reasoning_token':
          onReasoning(JSON.parse(data));
          break;
        case 'done': {
          const payload = data ? JSON.parse(data) : {};
          if (payload.reasoning) {
            onReasoning(payload.reasoning);
          }
          if (payload.metadata && onMetadata) {
            onMetadata(payload.metadata);
          }
          controller.abort();
          finishDone();
          break;
        }
        case 'error': {
          let message = 'Stream error';
          try {
            const payload = data ? JSON.parse(data) : {};
            message = payload.message ?? payload.error ?? message;
          } catch {
            if (data) message = data;
          }
          controller.abort();
          finishError(new Error(message));
          break;
        }
        case 'message':
          // Backend fallback path sends URL-encoded plain tokens
          if (data) onToken(decodeURIComponent(data));
          break;
      }
    };

    // Minimal SSE parser: handles `event:` / `data:` lines, multi-line data,
    // and blank-line event boundaries.
    let eventName = '';
    let dataLines: string[] = [];
    const processLine = (line: string): void => {
      if (line === '') {
        if (dataLines.length > 0 || eventName) {
          dispatch(eventName, dataLines.join('\n'));
        }
        eventName = '';
        dataLines = [];
        return;
      }
      if (line.startsWith(':')) return; // comment
      const colon = line.indexOf(':');
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? '' : line.slice(colon + 1);
      if (value.startsWith(' ')) value = value.slice(1);
      if (field === 'event') {
        eventName = value;
      } else if (field === 'data') {
        dataLines.push(value);
      }
    };

    (async () => {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ content, model, reasoning: reasoningEnabled }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Stream request failed: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // If the buffer ends with '\r', defer it: it may be the first half of
        // a CRLF split across chunks (avoids a spurious empty-line dispatch)
        let deferredCR = '';
        if (buffer.endsWith('\r')) {
          deferredCR = '\r';
          buffer = buffer.slice(0, -1);
        }

        // Split into lines; keep the last partial line in the buffer
        let newlineIndex: number;
        while ((newlineIndex = buffer.search(/\r\n|\n|\r/)) !== -1) {
          const line = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + (buffer.startsWith('\r\n', newlineIndex) ? 2 : 1));
          processLine(line);
          if (finished) return;
        }
        buffer += deferredCR;
      }

      // Flush any trailing event without a final blank line
      buffer += decoder.decode();
      if (buffer) processLine(buffer);
      processLine('');
      finishDone();
    })().catch((err: unknown) => {
      if (controller.signal.aborted) {
        // Cancelled by user (or closed after 'done') — not an error, no callback
        finished = true;
        return;
      }
      finishError(err);
    });

    return () => controller.abort();
  }

  /**
   * Get messages from a session with pagination
   */
  getMessages(sessionId: string, limit = 20, before?: string) {
    let url = `${this.config.apiBaseUrl}/chat/${sessionId}/messages?limit=${limit}`;
    if (before) {
      url += `&before=${encodeURIComponent(before)}`;
    }
    return this.http.get<any[]>(url);
  }

  /**
   * Attach a file to the current chat session
   */
  attachFile(sessionId: string, payload: { filename: string; content: string }) {
    return this.http.post<{ status: string; filename: string; length: number }>(
      `${this.config.apiBaseUrl}/chat/${sessionId}/attachment`,
      payload,
    );
  }
}
