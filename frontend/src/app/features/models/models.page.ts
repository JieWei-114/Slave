/**
 * Models Page Component
 * Manage installed AI models: list, delete, search, and pull with progress.
 */
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ModelsStore } from './models.store';
import { ModelFile, ModelSearchResult } from './models.api';
import { ErrorBannerComponent } from '../../shared/ui/banner/error-banner.component';

@Component({
  selector: 'app-models-page',
  standalone: true,
  imports: [CommonModule, FormsModule, ErrorBannerComponent],
  templateUrl: './models.page.html',
  styleUrls: ['./models.page.css'],
})
export class ModelsPage implements OnInit {
  query = '';
  private searchTimer: ReturnType<typeof setTimeout> | null = null;

  /** Two-click delete confirmation: name of the model pending confirmation */
  confirmingDeleteName: string | null = null;
  private confirmTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(public store: ModelsStore) {}

  ngOnInit(): void {
    this.store.loadInstalled();
  }

  onSearchInput(): void {
    if (this.searchTimer) clearTimeout(this.searchTimer);
    this.searchTimer = setTimeout(() => this.store.search(this.query), 350);
  }

  repoId(result: ModelSearchResult): string {
    return String(result.repo_id ?? result.id ?? result.name ?? '');
  }

  selectResult(result: ModelSearchResult): void {
    const repo = this.repoId(result);
    if (repo) this.store.loadFiles(repo);
  }

  pullFile(file: ModelFile): void {
    this.store.pull(file.ollama_name);
  }

  requestDelete(name: string): void {
    if (this.confirmingDeleteName === name) {
      if (this.confirmTimer) clearTimeout(this.confirmTimer);
      this.confirmingDeleteName = null;
      this.store.delete(name);
      return;
    }
    this.confirmingDeleteName = name;
    if (this.confirmTimer) clearTimeout(this.confirmTimer);
    // Auto-reset the confirmation after a few seconds
    this.confirmTimer = setTimeout(() => (this.confirmingDeleteName = null), 3000);
  }

  formatBytes(bytes: number | undefined): string {
    if (!bytes || bytes <= 0) return '';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit++;
    }
    return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
  }

  trackByName(index: number, item: { name: string }): string {
    return item.name;
  }

  trackByFilename(index: number, item: ModelFile): string {
    return item.filename;
  }
}
