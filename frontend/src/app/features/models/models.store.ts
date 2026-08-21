/**
 * Models Store
 * Signal-based state for the models management page.
 */
import { Injectable, signal, computed, inject } from '@angular/core';

import { ModelsApi, InstalledModel, ModelSearchResult, ModelFile } from './models.api';

@Injectable({ providedIn: 'root' })
export class ModelsStore {
  private api = inject(ModelsApi);

  /**
   * State Signals
   */
  readonly installed = signal<InstalledModel[]>([]);
  readonly provider = signal('');
  readonly searchResults = signal<ModelSearchResult[]>([]);
  readonly files = signal<ModelFile[]>([]);
  readonly selectedRepo = signal('');

  readonly loading = signal(false);
  readonly searching = signal(false);
  readonly loadingFiles = signal(false);
  readonly error = signal('');

  // Active pull: model name + progress
  readonly pullingName = signal('');
  readonly pullStatus = signal('');
  readonly pullCompleted = signal(0);
  readonly pullTotal = signal(0);

  readonly isPulling = computed(() => !!this.pullingName());
  readonly pullPct = computed(() => {
    const total = this.pullTotal();
    if (!total) return 0;
    return Math.min(100, Math.round((this.pullCompleted() / total) * 100));
  });

  private cancelPull: (() => void) | null = null;

  clearError(): void {
    this.error.set('');
  }

  /**
   * Load installed models
   */
  loadInstalled(): void {
    this.loading.set(true);
    this.api.getInstalled().subscribe({
      next: (res) => {
        this.provider.set(res?.provider ?? '');
        const models = (res?.models ?? []).map((m) => (typeof m === 'string' ? { name: m } : m));
        this.installed.set(models);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('Failed to load installed models');
        this.loading.set(false);
      },
    });
  }

  /**
   * Search remote model repositories
   */
  search(q: string): void {
    const query = q.trim();
    if (!query) {
      this.searchResults.set([]);
      return;
    }

    this.searching.set(true);
    this.api.search(query).subscribe({
      next: (res) => {
        this.searchResults.set(res ?? []);
        this.searching.set(false);
      },
      error: () => {
        this.error.set('Model search failed');
        this.searching.set(false);
      },
    });
  }

  /**
   * Load quant files for a repository
   */
  loadFiles(repo: string): void {
    this.selectedRepo.set(repo);
    this.files.set([]);
    this.loadingFiles.set(true);

    this.api.getFiles(repo).subscribe({
      next: (res) => {
        this.files.set(res?.files ?? []);
        this.loadingFiles.set(false);
      },
      error: () => {
        this.error.set(`Failed to load files for ${repo}`);
        this.loadingFiles.set(false);
      },
    });
  }

  /**
   * Pull a model with streamed progress; refreshes the installed list on success
   */
  pull(name: string): void {
    if (this.isPulling()) {
      this.error.set(`Already pulling ${this.pullingName()} — wait for it to finish.`);
      return;
    }

    this.error.set('');
    this.pullingName.set(name);
    this.pullStatus.set('Starting…');
    this.pullCompleted.set(0);
    this.pullTotal.set(0);

    this.cancelPull = this.api.pullModel(
      name,
      (p) => {
        if (p.status) this.pullStatus.set(p.status);
        if (typeof p.completed === 'number') this.pullCompleted.set(p.completed);
        if (typeof p.total === 'number') this.pullTotal.set(p.total);
      },
      () => {
        this.resetPull();
        this.loadInstalled();
      },
      (message) => {
        this.resetPull();
        this.error.set(message);
      },
    );
  }

  /**
   * Cancel the active pull
   */
  stopPull(): void {
    this.cancelPull?.();
    this.resetPull();
  }

  private resetPull(): void {
    this.cancelPull = null;
    this.pullingName.set('');
    this.pullStatus.set('');
    this.pullCompleted.set(0);
    this.pullTotal.set(0);
  }

  /**
   * Delete an installed model
   */
  delete(name: string): void {
    this.api.deleteModel(name).subscribe({
      next: () => {
        this.installed.update((list) => list.filter((m) => m.name !== name));
      },
      error: () => {
        this.error.set(`Failed to delete ${name}`);
      },
    });
  }
}
