/**
 * Voice API Service
 * Speech-to-text (transcription) and text-to-speech endpoints.
 * Also holds the voice capability signals loaded once at chat page init.
 */
import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

import { AppConfigService } from '../../../core/services/app-config.services';

export interface VoiceConfig {
  enabled: boolean;
  stt_model: string;
  tts_voice: string;
  stt_ready: boolean;
  tts_ready: boolean;
}

@Injectable({ providedIn: 'root' })
export class VoiceApi {
  private http = inject(HttpClient);
  private config = inject(AppConfigService);

  /** Capability signals — default to disabled until getConfig() succeeds */
  readonly sttEnabled = signal(false);
  readonly ttsEnabled = signal(false);

  /** Model-readiness signals — false means first use will download the model */
  readonly sttReady = signal(false);
  readonly ttsReady = signal(false);

  private configLoaded = false;

  /**
   * Fetch voice capability config from backend
   */
  getConfig(): Promise<VoiceConfig> {
    return firstValueFrom(this.http.get<VoiceConfig>(`${this.config.apiBaseUrl}/voice/config`));
  }

  /**
   * Load config once and populate capability signals.
   * Any failure is treated as "voice disabled" — never blocks page load.
   */
  loadConfig(): void {
    if (this.configLoaded) return;
    this.configLoaded = true;

    this.getConfig()
      .then((cfg) => {
        // Not-ready models are still shown (first use downloads them);
        // only hide voice UI when the feature itself is disabled.
        this.sttEnabled.set(!!cfg.enabled);
        this.ttsEnabled.set(!!cfg.enabled);
        this.sttReady.set(!!cfg.stt_ready);
        this.ttsReady.set(!!cfg.tts_ready);
      })
      .catch(() => {
        this.sttEnabled.set(false);
        this.ttsEnabled.set(false);
        this.sttReady.set(false);
        this.ttsReady.set(false);
      });
  }

  /**
   * Transcribe recorded audio (audio/webm from MediaRecorder) to text
   */
  async transcribe(blob: Blob): Promise<{ text: string }> {
    const formData = new FormData();
    formData.append('file', blob, 'recording.webm');

    const result = await firstValueFrom(
      this.http.post<{ text: string; language: string; duration: number }>(
        `${this.config.apiBaseUrl}/voice/transcribe`,
        formData,
      ),
    );
    return { text: result.text };
  }

  /**
   * Synthesize speech for the given text — returns audio/wav bytes
   */
  speak(text: string): Promise<Blob> {
    return firstValueFrom(
      this.http.post(`${this.config.apiBaseUrl}/voice/speak`, { text }, { responseType: 'blob' }),
    );
  }
}
