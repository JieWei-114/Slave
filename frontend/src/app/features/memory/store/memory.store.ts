/**
 * Memory Store
 */
import { Injectable, signal, inject } from '@angular/core';
import { MemoryApi } from '../service/memory.api';
import { Memory } from '../service/memory.model';

@Injectable({ providedIn: 'root' })
export class MemoryStore {
  // State
  memories = signal<Memory[]>([]);
  loading = signal(false);
  error = signal('');
  compressing = signal(false);

  private memoryApi = inject(MemoryApi);

  readonly currentSessionId = signal<string | null>(null);

  /**
   * Manually add a memory item
   */
  addManual(
    content: string,
    category: 'preference_or_fact' | 'important' | 'other' = 'other',
  ): void {
    const sessionId = this.currentSessionId();
    if (!sessionId) return;

    this.memoryApi.addMemory(content, sessionId, category).subscribe({
      next: (m) => {
        this.memories.update((list) => [...list, m as Memory]);
      },
      error: () => {
        this.error.set('Failed to save memory');
      },
    });
  }

  /**
   * Load all memories for a session
   */
  load(sessionId: string) {
    this.currentSessionId.set(sessionId);
    this.loading.set(true);
    this.error.set('');

    this.memoryApi.getMemories(sessionId).subscribe({
      next: (m) => {
        this.memories.set(m);
      },
      complete: () => this.loading.set(false),
      error: () => {
        this.error.set('Failed to load memories');
        this.loading.set(false);
      },
    });
  }

  /**
   * Toggle memory enabled/disabled state
   */
  toggle(m: Memory) {
    const action = m.enabled ? this.memoryApi.disable(m.id) : this.memoryApi.enable(m.id);

    action.subscribe({
      next: () => {
        this.memories.update((list) =>
          list.map((x) => (x.id === m.id ? { ...x, enabled: !x.enabled } : x)),
        );
      },
      error: () => {
        this.error.set(`Failed to ${m.enabled ? 'disable' : 'enable'} memory`);
      },
    });
  }

  /**
   * Delete a memory item
   */
  delete(m: Memory) {
    this.memoryApi.delete(m.id).subscribe({
      next: () => {
        this.memories.update((list) => list.filter((x) => x.id !== m.id));
      },
      error: () => {
        this.error.set('Failed to delete memory');
      },
    });
  }

  /**
   * Search memories by query string
   */
  search(q: string) {
    const sessionId = this.currentSessionId();
    if (!sessionId) return;

    this.loading.set(true);
    this.memoryApi.search(sessionId, q).subscribe({
      next: (res) => {
        this.memories.set(res);
      },
      complete: () => this.loading.set(false),
      error: () => {
        this.error.set('Memory search failed');
        this.loading.set(false);
      },
    });
  }

  clearError(): void {
    this.error.set('');
  }

  /**
   * Compress and synthesize memories using AI
   * Reduces memory count while preserving important information
   */
  compress(model: string) {
    const sessionId = this.currentSessionId();
    if (!sessionId || this.compressing()) return;

    this.compressing.set(true);

    this.memoryApi.compress(sessionId, model).subscribe({
      next: (m: any) => {
        if (m?.id) {
          this.memories.update((list) => [...list, m]);
        }
      },
      complete: () => {
        this.compressing.set(false);
        this.load(sessionId);
      },
      error: () => this.compressing.set(false),
    });
  }

  /**
   * Reload memories for current session
   */
  reload(sessionId: string) {
    this.loading.set(true);

    this.memoryApi.getMemories(sessionId).subscribe({
      next: (m) => {
        this.memories.set(m);
        this.loading.set(false);
      },
      error: () => {
        this.loading.set(false);
      },
    });
  }
}
