/**
 * Models API Service
 * REST endpoints for managing local AI models, plus a fetch-based SSE
 * reader for pull progress (mirrors the pattern in chat.api.ts).
 */
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';

import { AppConfigService } from '../../core/services/app-config.services';

export interface InstalledModel {
  name: string;
  size?: number;
  [key: string]: unknown;
}

export interface InstalledModelsResponse {
  provider: string;
  models: (InstalledModel | string)[];
}

export interface ModelSearchResult {
  repo_id?: string;
  id?: string;
  name?: string;
  [key: string]: unknown;
}

export interface ModelFile {
  filename: string;
  quant: string;
  size_bytes: number;
  ollama_name: string;
}

export interface ModelFilesResponse {
  repo_id: string;
  files: ModelFile[];
}

export interface PullProgress {
  status: string;
  completed?: number;
  total?: number;
}

@Injectable({ providedIn: 'root' })
export class ModelsApi {
  private http = inject(HttpClient);
  private config = inject(AppConfigService);

  /**
   * List installed models
   */
  getInstalled() {
    return this.http.get<InstalledModelsResponse>(`${this.config.apiBaseUrl}/models`);
  }

  /**
   * Search remote model repositories
   */
  search(q: string) {
    return this.http.get<ModelSearchResult[]>(
      `${this.config.apiBaseUrl}/models/search?q=${encodeURIComponent(q)}`,
    );
  }

  /**
   * List downloadable quant files for a repository
   */
  getFiles(repo: string) {
    const encodedRepo = repo.split('/').map(encodeURIComponent).join('/');
    return this.http.get<ModelFilesResponse>(
      `${this.config.apiBaseUrl}/models/search/${encodedRepo}/files`,
    );
  }

  /**
   * Delete an installed model
   */
  deleteModel(name: string) {
    return this.http.delete(`${this.config.apiBaseUrl}/models/${encodeURIComponent(name)}`);
  }

  /**
   * Pull (download) a model with SSE progress updates.
   * Uses fetch() POST with hand-rolled SSE parsing over the response
   * ReadableStream (same approach as chat.api.ts streamMessage).
   * Returns a cancel function (AbortController).
   */
  pullModel(
    name: string,
    onProgress: (p: PullProgress) => void,
    onDone: () => void,
    onError: (message: string) => void,
  ): () => void {
    const url = `${this.config.apiBaseUrl}/models/pull`;
    const controller = new AbortController();
    let finished = false;

    const finishDone = () => {
      if (finished) return;
      finished = true;
      onDone();
    };

    const finishError = (message: string) => {
      if (finished) return;
      finished = true;
      onError(message);
    };

    // Dispatch a single parsed SSE event to the right callback
    const dispatch = (eventName: string, data: string): void => {
      const type = eventName || 'message';
      switch (type) {
        case 'done':
          controller.abort();
          finishDone();
          break;
        case 'error': {
          let message = 'Model pull failed';
          try {
            const payload = data ? JSON.parse(data) : {};
            message = payload.message ?? payload.error ?? payload.detail ?? message;
          } catch {
            if (data) message = data;
          }
          controller.abort();
          finishError(message);
          break;
        }
        default: {
          // Progress payload: { status, completed?, total? }
          if (!data) return;
          try {
            onProgress(JSON.parse(data) as PullProgress);
          } catch {
            // Ignore malformed progress lines
          }
        }
      }
    };

    // Minimal SSE parser: `event:` / `data:` lines, blank-line boundaries
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
        body: JSON.stringify({ name }),
        signal: controller.signal,
      });

      if (response.status === 409) {
        finishError('This model is already being pulled.');
        return;
      }

      if (!response.ok || !response.body) {
        finishError(`Model pull failed (${response.status})`);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let newlineIndex: number;
        while ((newlineIndex = buffer.search(/\r\n|\n|\r/)) !== -1) {
          const line = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + (buffer.startsWith('\r\n', newlineIndex) ? 2 : 1));
          processLine(line);
          if (finished) return;
        }
      }

      // Flush any trailing event without a final blank line
      buffer += decoder.decode();
      if (buffer) processLine(buffer);
      processLine('');
      finishDone();
    })().catch((err: unknown) => {
      if (controller.signal.aborted) {
        finished = true;
        return;
      }
      finishError(err instanceof Error ? err.message : 'Model pull failed');
    });

    return () => controller.abort();
  }
}
