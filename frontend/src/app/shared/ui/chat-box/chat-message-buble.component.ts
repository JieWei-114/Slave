import { Component, Input, Output, EventEmitter, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ChatMessage } from '../../../features/chat/services/chat.model';
import { ChatStore } from '../../../features/chat/store/chat.store';
import { VoiceApi } from '../../../features/chat/services/voice.api';
import { VoicePlaybackService } from '../../../features/chat/services/voice-playback.service';
import { AutoResizeTextareaDirective } from '../../directives/auto-resize-textarea.directive';
import { ContextIndicatorComponent } from '../context-indicator/context-indicator.component';
import { MarkdownPipe } from '../../pipes/markdown.pipe';

@Component({
  selector: 'app-chat-message-bubble',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    AutoResizeTextareaDirective,
    ContextIndicatorComponent,
    MarkdownPipe,
  ],
  templateUrl: './chat-message-buble.component.html',
  styleUrls: ['./chat-message-buble.component.css'],
})
export class ChatMessageBubbleComponent implements OnChanges {
  @Input() message!: ChatMessage;
  @Input() isLastUserMessage = false;

  @Output() editAndResend = new EventEmitter<{
    content: string;
    attachment?: { filename: string; content: string };
  }>();

  /**
   * UI State
   */
  isEditing = false;
  editedContent = '';
  showReasoning = false;
  showMetadata = false;
  manuallyToggled = false; // Track if user manually toggled reasoning
  private previousReasoningLength = 0;
  private messageKey = '';

  constructor(
    private store: ChatStore,
    private voiceApi: VoiceApi,
    private voicePlayback: VoicePlaybackService,
  ) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (!changes['message']) return;

    const nextKey = this.message ? `${this.message.role}|${this.message.created_at}` : '';

    if (nextKey && nextKey !== this.messageKey) {
      // Reset per-message UI state when a different message is bound.
      this.showReasoning = false;
      this.manuallyToggled = false;
      this.showMetadata = false;
      this.previousReasoningLength = 0;
      this.messageKey = nextKey;
    }

    if (this.manuallyToggled) return;

    const currentReasoning = this.message?.meta?.reasoning || '';
    const currentLength = currentReasoning.length;

    // Auto-open only while reasoning is actively streaming and text is growing.
    if (
      this.message?.meta?.reasoning_streaming &&
      currentLength > this.previousReasoningLength &&
      currentLength > 0
    ) {
      this.showReasoning = true;
    }

    this.previousReasoningLength = currentLength;
  }

  /**
   * UI Actions
   */
  toggleReasoning(): void {
    this.showReasoning = !this.showReasoning;
    this.manuallyToggled = true; // Mark as manually toggled to prevent auto-opening
  }

  toggleMetadata(): void {
    this.showMetadata = !this.showMetadata;
  }

  remember(): void {
    if (!this.message || this.message.remembered) return;

    this.store.rememberMessage(this.message);
  }

  startEdit(): void {
    this.isEditing = true;
    this.editedContent = this.message.content ?? '';
  }

  cancelEdit(): void {
    this.isEditing = false;
    this.editedContent = '';
  }

  saveAndResend(): void {
    const trimmed = this.editedContent.trim();
    if (!trimmed) return;

    this.isEditing = false;
    this.editAndResend.emit({
      content: trimmed,
      attachment: this.message.attachment,
    });
  }

  onEnterKey(event: Event): void {
    const keyboardEvent = event as KeyboardEvent;

    if (keyboardEvent.ctrlKey) {
      keyboardEvent.preventDefault();
      this.saveAndResend();
    }
  }

  /**
   *  Template Helpers
   */
  get isUser(): boolean {
    return this.message?.role === 'user';
  }

  get isAssistant(): boolean {
    return this.message?.role === 'assistant';
  }

  get isReasoningEnabled(): boolean {
    return this.store.reasoningEnabled();
  }

  /**
   * Voice output (text-to-speech)
   */
  private get voiceKey(): string {
    return `${this.message?.role}|${this.message?.created_at}`;
  }

  get canSpeak(): boolean {
    return (
      this.isAssistant &&
      this.voiceApi.ttsEnabled() &&
      !!this.message?.content &&
      !this.store.loading() // hide while the response is still streaming
    );
  }

  get isSpeaking(): boolean {
    return this.voicePlayback.playingKey() === this.voiceKey;
  }

  get isSpeakLoading(): boolean {
    return this.voicePlayback.loadingKey() === this.voiceKey;
  }

  /** True while a slow first-time speak is likely downloading the voice model */
  isSpeakDownloadHint = false;
  private speakHintTimer: ReturnType<typeof setTimeout> | null = null;

  get speakTitle(): string {
    if (this.isSpeakLoading && this.isSpeakDownloadHint) {
      // "Downloading" only when the model truly isn't cached yet;
      // otherwise it's just slow synthesis on CPU.
      return this.voiceApi.ttsReady() ? 'Generating speech…' : 'Downloading voice model…';
    }
    if (this.isSpeaking) return 'Stop playback';
    if (!this.voiceApi.ttsReady()) return 'First use will download the voice model (~30s)';
    return 'Read message aloud';
  }

  toggleSpeak(): void {
    if (!this.message?.content) return;

    // Flip a hint after 3s: a long first fetch means the model is downloading
    this.clearSpeakHintTimer();
    this.speakHintTimer = setTimeout(() => {
      this.isSpeakDownloadHint = true;
    }, 3000);

    void this.voicePlayback.toggle(this.voiceKey, this.message.content).finally(() => {
      this.clearSpeakHintTimer();
      this.isSpeakDownloadHint = false;
    });
  }

  private clearSpeakHintTimer(): void {
    if (this.speakHintTimer) {
      clearTimeout(this.speakHintTimer);
      this.speakHintTimer = null;
    }
  }
}
