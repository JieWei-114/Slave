/**
 * Panel Component
 * Right-side panel with tabs for Memory, Web Search, and Rules
 */
import {
  Component,
  Input,
  Output,
  EventEmitter,
  AfterViewInit,
  ElementRef,
  Inject,
  PLATFORM_ID,
} from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { CommonModule } from '@angular/common';
import { MemoryPage } from '../../../features/memory/page/memory.page';
import { FilesPage } from '../../../features/file/page/files.page';
import { WebSearchComponent } from '../../../features/web/page/web-search.component';
import { RulesPanelComponent } from '../../../features/rules/page/rules.page';

type PanelTab = 'memory' | 'files' | 'web' | 'rules';

@Component({
  selector: 'app-panel',
  standalone: true,
  imports: [CommonModule, MemoryPage, FilesPage, WebSearchComponent, RulesPanelComponent],
  templateUrl: './panel.component.html',
  styleUrls: ['./panel.component.css'],
})
export class PanelComponent implements AfterViewInit {
  @Input() sessionId!: string;
  @Output() close = new EventEmitter<void>();

  activeTab: PanelTab = 'memory';
  private isResizing = false;
  private startX = 0;
  private startWidth = 0;

  constructor(
    private elementRef: ElementRef,
    @Inject(PLATFORM_ID) private platformId: object,
  ) {}

  ngAfterViewInit(): void {
    if (isPlatformBrowser(this.platformId)) {
      this.restorePanelWidth();
      this.setupPanelResize();
    }
  }

  /**
   * Restore the last user-dragged width so the panel opens at a stable size
   */
  private restorePanelWidth(): void {
    const saved = Number(localStorage.getItem('panel-width'));
    if (saved >= 390 && saved <= 600) {
      const panel = this.elementRef.nativeElement;
      panel.style.width = saved + 'px';
      panel.style.minWidth = saved + 'px';
    }
  }

  /**
   * Setup mouse event handlers for panel resizing
   * Allows user to drag the left edge to resize panel width
   */
  private setupPanelResize(): void {
    // Start resize on handle mousedown
    document.addEventListener('mousedown', (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.classList.contains('panel-resize-handle')) return;

      e.preventDefault();
      this.isResizing = true;
      this.startX = e.clientX;

      const panel = this.elementRef.nativeElement;
      if (!panel) return;

      this.startWidth = panel.offsetWidth;
      document.body.style.cursor = 'ew-resize';
      document.body.style.userSelect = 'none';
    });

    // Update width during drag
    document.addEventListener('mousemove', (e: MouseEvent) => {
      if (!this.isResizing) return;

      e.preventDefault();
      const panel = this.elementRef.nativeElement;
      if (!panel) return;

      // Drag left increases width (panel on right side)
      const delta = this.startX - e.clientX;
      const newWidth = this.startWidth + delta;

      // Enforce min/max width constraints
      if (newWidth >= 390 && newWidth <= 600) {
        panel.style.width = newWidth + 'px';
        panel.style.minWidth = newWidth + 'px';
      }
    });

    // End resize on mouseup
    document.addEventListener('mouseup', () => {
      if (this.isResizing) {
        this.isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        const width = this.elementRef.nativeElement?.offsetWidth;
        if (width) localStorage.setItem('panel-width', String(width));
      }
    });
  }

  /**
   * Switch active tab
   */
  selectTab(tab: PanelTab): void {
    this.activeTab = tab;
  }

  /**
   * Emit close event to parent
   */
  onClose(): void {
    this.close.emit();
  }
}
