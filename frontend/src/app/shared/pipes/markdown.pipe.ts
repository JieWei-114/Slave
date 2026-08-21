/**
 * Markdown Pipe
 * Renders markdown to sanitized HTML for assistant messages.
 * marked.parse() → DOMPurify.sanitize() → bypassSecurityTrustHtml on the
 * already-sanitized string. SSR-safe: DOMPurify requires a window, so on
 * the server we fall back to escaped plain text.
 */
import { Pipe, PipeTransform, inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

@Pipe({
  name: 'markdown',
  standalone: true,
})
export class MarkdownPipe implements PipeTransform {
  private sanitizer = inject(DomSanitizer);
  private platformId = inject(PLATFORM_ID);

  transform(value: string | null | undefined): SafeHtml {
    const text = value ?? '';

    // SSR fallback: DOMPurify needs a real window — emit escaped plain text
    if (!isPlatformBrowser(this.platformId) || typeof window === 'undefined') {
      return this.sanitizer.bypassSecurityTrustHtml(this.escapeHtml(text));
    }

    const html = marked.parse(text, { async: false, breaks: true, gfm: true });
    const clean = DOMPurify.sanitize(html);
    return this.sanitizer.bypassSecurityTrustHtml(clean);
  }

  private escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
}
