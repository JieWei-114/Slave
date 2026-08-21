/**
 * Voice Playback Service
 * Plays TTS audio for message bubbles. Ensures only one
 * playback is active app-wide at any time.
 */
import { Injectable, inject, signal } from '@angular/core';

import { VoiceApi } from './voice.api';

@Injectable({ providedIn: 'root' })
export class VoicePlaybackService {
  private voiceApi = inject(VoiceApi);

  /** Key of the message currently playing (or null) */
  readonly playingKey = signal<string | null>(null);
  /** Key of the message whose audio is being fetched (or null) */
  readonly loadingKey = signal<string | null>(null);

  private audio: HTMLAudioElement | null = null;
  private objectUrl: string | null = null;

  /**
   * Toggle playback for a message: play if idle, stop if already playing.
   */
  async toggle(key: string, text: string): Promise<void> {
    // Clicking the currently-playing (or loading) message stops it
    if (this.playingKey() === key || this.loadingKey() === key) {
      this.stop();
      return;
    }

    // Stop any other playback first — only one at a time
    this.stop();

    this.loadingKey.set(key);
    try {
      const blob = await this.voiceApi.speak(text);

      // User may have cancelled while fetching
      if (this.loadingKey() !== key) return;

      this.objectUrl = URL.createObjectURL(blob);
      const audio = new Audio(this.objectUrl);
      this.audio = audio;
      audio.onended = () => this.stop();
      audio.onerror = () => this.stop();

      try {
        await audio.play();
      } catch {
        // Only tear down if this audio still belongs to this toggle call —
        // a newer toggle may have replaced it while play() was pending
        if (this.audio === audio) this.stop();
        return;
      }
      if (this.audio === audio) this.playingKey.set(key);
    } catch {
      this.stop();
    } finally {
      if (this.loadingKey() === key) this.loadingKey.set(null);
    }
  }

  /**
   * Stop playback and release resources
   */
  stop(): void {
    if (this.audio) {
      this.audio.pause();
      this.audio.onended = null;
      this.audio.onerror = null;
      this.audio = null;
    }
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }
    this.playingKey.set(null);
    this.loadingKey.set(null);
  }
}
