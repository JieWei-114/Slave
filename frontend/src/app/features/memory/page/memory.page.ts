import { Component, Input, effect, signal, Output, EventEmitter } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { MemoryStore } from '../store/memory.store';
import { Memory } from '../service/memory.model';
import { ChatStore } from '../../chat/store/chat.store';
import { AutoResizeTextareaDirective } from '../../../shared/directives/auto-resize-textarea.directive';
import { ErrorBannerComponent } from '../../../shared/ui/banner/error-banner.component';

@Component({
  selector: 'app-memory-panel',
  standalone: true,
  imports: [FormsModule, CommonModule, AutoResizeTextareaDirective, ErrorBannerComponent],
  templateUrl: './memory.page.html',
  styleUrls: ['./memory.page.css'],
})
export class MemoryPage {
  private _sessionId = signal<string | null>(null);
  private searchTimer: any;
  isCategoryDropdownOpen = false;

  /** Two-click delete confirmation: id of the memory pending confirmation */
  confirmingDeleteId: string | null = null;
  private confirmTimer: any;

  requestDelete(m: Memory): void {
    if (this.confirmingDeleteId === m.id) {
      clearTimeout(this.confirmTimer);
      this.confirmingDeleteId = null;
      this.store.delete(m);
      return;
    }
    this.confirmingDeleteId = m.id;
    clearTimeout(this.confirmTimer);
    // Auto-reset the confirmation after a few seconds
    this.confirmTimer = setTimeout(() => (this.confirmingDeleteId = null), 3000);
  }

  @Input({ required: true })
  set sessionId(value: string) {
    this._sessionId.set(value);
  }

  @Output() close = new EventEmitter<void>();

  newMemory = '';
  newCategory: 'preference_or_fact' | 'important' | 'other' = 'preference_or_fact';

  readonly categoryOptions: {
    value: 'preference_or_fact' | 'important' | 'other';
    label: string;
  }[] = [
    { value: 'preference_or_fact', label: 'Preference/Fact' },
    { value: 'important', label: 'Important' },
    { value: 'other', label: 'Other' },
  ];

  constructor(
    public store: MemoryStore,
    public chatStore: ChatStore,
  ) {
    effect(() => {
      const id = this._sessionId();
      if (!id) return;

      console.debug('Memory reload for session:', id);
      this.store.load(id);
    });
  }

  add() {
    if (!this.newMemory.trim()) return;
    this.store.addManual(this.newMemory, this.newCategory);
    this.newMemory = '';
  }

  compress() {
    this.store.compress(this.chatStore.currentModel().id);
  }

  onSearch(q: string) {
    clearTimeout(this.searchTimer);

    if (q.trim().length < 2) {
      const id = this._sessionId();
      if (id) {
        this.store.load(id);
      }
      return;
    }

    this.searchTimer = setTimeout(() => {
      this.store.search(q);
    }, 300);
  }

  toggleCategoryDropdown(event: MouseEvent) {
    event.stopPropagation();
    this.isCategoryDropdownOpen = !this.isCategoryDropdownOpen;
  }

  selectCategory(value: 'preference_or_fact' | 'important' | 'other') {
    this.newCategory = value;
    this.isCategoryDropdownOpen = false;
  }

  getCurrentCategoryLabel(): string {
    return this.categoryOptions.find((opt) => opt.value === this.newCategory)?.label || 'Select';
  }
}
