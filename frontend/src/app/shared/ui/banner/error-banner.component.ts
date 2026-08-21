import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-error-banner',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './error-banner.component.html',
  styleUrls: ['./error-banner.component.css'],
})
export class ErrorBannerComponent {
  @Input() message = '';
  /** Show a ✕ button that emits (dismiss) */
  @Input() dismissible = false;
  @Output() dismiss = new EventEmitter<void>();
}
