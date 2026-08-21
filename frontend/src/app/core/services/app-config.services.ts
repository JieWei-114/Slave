/**
 * Application configuration service
 * Provides centralized configuration for the entire app
 * including API endpoints and client-side settings
 */
import { Injectable } from '@angular/core';

/**
 * Client-side configuration structure
 * Loaded from backend to ensure consistent settings
 */
export interface ClientConfig {
  fileUpload: {
    maxSizeMB: number;
    allowedBinaryExtensions: string[];
    maxExtractChars: number;
  };
}

@Injectable({ providedIn: 'root' })
export class AppConfigService {
  // Application metadata
  readonly appName = 'Slave';

  /**
   * True when running inside the Tauri desktop shell.
   * SSR-safe: window is only touched when defined.
   */
  readonly isTauri =
    typeof window !== 'undefined' &&
    ('__TAURI_INTERNALS__' in window || '__TAURI__' in window);

  // Backend API endpoints (relative — proxied by nginx in prod, proxy.conf.json in dev).
  // In the Tauri desktop shell the app is served from the tauri:// protocol,
  // so relative URLs cannot reach the backend — talk to it directly instead.
  readonly apiBaseUrl = this.isTauri ? 'http://127.0.0.1:8000' : '/api';

  /**
   * WebSocket URL derived from the current origin (or the absolute
   * backend URL when running in Tauri).
   * Only valid in the browser; returns '' during SSR.
   */
  get wsUrl(): string {
    if (typeof window === 'undefined') return '';
    if (this.isTauri) return this.apiBaseUrl.replace(/^http/, 'ws');
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${window.location.host}${this.apiBaseUrl}`;
  }

  // Dynamic configuration loaded from backend
  private clientConfig: ClientConfig = {
    fileUpload: {
      maxSizeMB: 10,
      allowedBinaryExtensions: ['.pdf', '.doc', '.docx'],
      maxExtractChars: 50_000,
    },
  };

  /**
   * Load dynamic configuration from backend
   * Should be called during app initialization
   */
  async load(): Promise<void> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/client-config`);
      if (!response.ok) {
        throw new Error(`Failed to load client config: ${response.status}`);
      }
      this.clientConfig = await response.json();
    } catch (err) {
      console.warn('Client config load failed, using defaults', err);
    }
  }

  /**
   * Get max file upload size in bytes
   * Fallback to 10MB if config not loaded
   */
  get fileUploadMaxBytes(): number {
    return this.clientConfig
      ? this.clientConfig.fileUpload.maxSizeMB * 1024 * 1024
      : 10 * 1024 * 1024;
  }

  /**
   * Get list of allowed binary file extensions
   * (e.g., .pdf, .docx) that require server-side extraction
   */
  get binaryExtensions(): string[] {
    return this.clientConfig
      ? this.clientConfig.fileUpload.allowedBinaryExtensions
      : ['.pdf', '.doc', '.docx'];
  }

  /**
   * Get max file upload size in megabytes
   */
  get fileUploadMaxMB(): number {
    return this.clientConfig.fileUpload.maxSizeMB;
  }
}
