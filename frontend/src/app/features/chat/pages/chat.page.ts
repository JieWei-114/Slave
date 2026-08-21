/**
 * Chat Page Component
 * Main interface for chatting with AI
 */
import {
  Component,
  OnInit,
  OnDestroy,
  inject,
  signal,
  HostListener,
  ViewChild,
  ElementRef,
  AfterViewInit,
  effect,
  PLATFORM_ID,
} from '@angular/core';
import { isPlatformBrowser, CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { HttpClient } from '@angular/common/http';

import { ChatStore } from '../store/chat.store';
import { AVAILABLE_MODELS, AIModel } from '../services/chat.model';
import { AppButtonComponent } from '../../../shared/ui/button/app-button.component';
import { ChatMessageBubbleComponent } from '../../../shared/ui/chat-box/chat-message-buble.component';
import { AutoFocusDirective } from '../../../shared/directives/auto-focus.directive';
import { AutoScrollDirective } from '../../../shared/directives/auto-scroll.directive';
import { ErrorBannerComponent } from '../../../shared/ui/banner/error-banner.component';
import { SkeletonComponent } from '../../../shared/ui/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../../shared/ui/error-boundary/error-boundary.component';
import { AppConfigService } from '../../../core/services/app-config.services';
import { AutoResizeTextareaDirective } from '../../../shared/directives/auto-resize-textarea.directive';
import { VoiceApi } from '../services/voice.api';

@Component({
  selector: 'app-chat-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    AppButtonComponent,
    ChatMessageBubbleComponent,
    AutoFocusDirective,
    AutoScrollDirective,
    ErrorBannerComponent,
    SkeletonComponent,
    ErrorBoundaryComponent,
    AutoResizeTextareaDirective,
    RouterLink,
    // PrefixPipe
  ],
  templateUrl: './chat.page.html',
  styleUrls: ['./chat.page.css'],
})
export class ChatPage implements OnInit, AfterViewInit, OnDestroy {
  private http = inject(HttpClient);
  private config = inject(AppConfigService);
  private platformId = inject(PLATFORM_ID);
  voiceApi = inject(VoiceApi);

  @ViewChild('chatTextarea', { read: ElementRef }) textareaRef?: ElementRef<HTMLTextAreaElement>;

  models = signal<AIModel[]>(AVAILABLE_MODELS);
  noModelsInstalled = signal(false);
  isDropdownOpen = false;
  isErrorDismissed = signal(false);
  /** Whether the topic-break divider is expanded to show the stored summary */
  showTopicSummary = signal(false);
  selectedFileName = signal('');
  /** Object URL for a small thumbnail preview when the selected file is an image */
  imagePreviewUrl = signal('');
  fileError = signal('');
  isFileUploading = signal(false);
  private fileContent = signal('');
  private pendingFile: File | null = null;

  /** Image extensions accepted for upload (backend returns is_image for these) */
  private static readonly IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.gif'];

  // Voice input (speech-to-text) state
  isRecording = signal(false);
  isTranscribing = signal(false);
  /** True while a slow first transcription is likely downloading the voice model */
  voiceDownloadHint = signal(false);
  private voiceHintTimer: ReturnType<typeof setTimeout> | null = null;
  voiceError = signal('');
  micSupported = signal(false);
  private mediaRecorder: MediaRecorder | null = null;
  private mediaStream: MediaStream | null = null;
  private audioChunks: Blob[] = [];
  private recordingTimeout: ReturnType<typeof setTimeout> | null = null;
  private micStarting = false;

  /** Max recording length before auto-stop (ms) */
  private static readonly MAX_RECORDING_MS = 60_000;

  get message(): string {
    return this.store.draftMessage();
  }

  set message(value: string) {
    this.store.setDraftMessage(value);
  }

  constructor(
    public store: ChatStore,
    route: ActivatedRoute,
  ) {
    // Watch for route parameter changes (session ID)
    route.paramMap.subscribe((params: any) => {
      const id = params.get('id') ?? 'default';
      this.store.selectSession(id);
      this.isErrorDismissed.set(false);
      this.showTopicSummary.set(false);
    });

    // Auto-resize textarea when draft message changes
    effect(() => {
      this.store.draftMessage();
      this.triggerResize();
    });
  }

  ngOnInit(): void {
    // Load available AI models from backend
    this.loadModels();

    // Voice capabilities — browser only, never blocks page load
    if (isPlatformBrowser(this.platformId)) {
      this.micSupported.set(
        typeof navigator !== 'undefined' &&
          !!navigator.mediaDevices?.getUserMedia &&
          typeof MediaRecorder !== 'undefined',
      );
      this.voiceApi.loadConfig();
    }
  }

  /**
   * Toggle microphone recording: start on first click, stop + transcribe on second
   */
  async toggleRecording(): Promise<void> {
    if (this.isRecording()) {
      this.stopRecording();
      return;
    }

    if (!this.micSupported() || this.isTranscribing()) return;

    // Guard against double-click: bail if a start is already in flight
    if (this.micStarting) return;
    this.micStarting = true;

    this.voiceError.set('');
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      this.micStarting = false;
      this.voiceError.set('Microphone access denied. Check browser permissions.');
      return;
    }

    // If recording started elsewhere while we awaited, release this stream
    if (this.isRecording()) {
      stream.getTracks().forEach((track) => track.stop());
      this.micStarting = false;
      return;
    }

    const preferredType = 'audio/webm;codecs=opus';
    const options =
      typeof MediaRecorder.isTypeSupported === 'function' &&
      MediaRecorder.isTypeSupported(preferredType)
        ? { mimeType: preferredType }
        : undefined;

    this.mediaStream = stream;
    this.audioChunks = [];
    try {
      this.mediaRecorder = new MediaRecorder(stream, options);
    } catch {
      stream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
      this.micStarting = false;
      this.voiceError.set('Recording is not supported in this browser.');
      return;
    }

    this.mediaRecorder.ondataavailable = (e: BlobEvent) => {
      if (e.data.size > 0) this.audioChunks.push(e.data);
    };
    this.mediaRecorder.onstop = () => this.handleRecordingStopped();

    this.mediaRecorder.start();
    this.isRecording.set(true);
    this.micStarting = false;

    // Auto-stop after 60 seconds
    this.recordingTimeout = setTimeout(() => this.stopRecording(), ChatPage.MAX_RECORDING_MS);
  }

  /**
   * Stop the active recording (triggers transcription via onstop)
   */
  private stopRecording(): void {
    if (this.recordingTimeout) {
      clearTimeout(this.recordingTimeout);
      this.recordingTimeout = null;
    }
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }
    this.isRecording.set(false);
  }

  /**
   * Assemble recorded audio and transcribe it; append text to the draft
   */
  private async handleRecordingStopped(): Promise<void> {
    const mimeType = this.mediaRecorder?.mimeType || 'audio/webm';
    const blob = new Blob(this.audioChunks, { type: mimeType });
    this.releaseMicrophone();

    if (blob.size === 0) return;

    this.isTranscribing.set(true);
    // Flip a hint after 3s: a long first transcription means the model is downloading
    this.clearVoiceHintTimer();
    this.voiceHintTimer = setTimeout(() => this.voiceDownloadHint.set(true), 3000);
    try {
      const { text } = await this.voiceApi.transcribe(blob);
      this.voiceApi.sttReady.set(true);
      const transcribed = (text ?? '').trim();
      if (transcribed) {
        const draft = this.store.draftMessage();
        this.store.setDraftMessage(draft ? `${draft} ${transcribed}` : transcribed);
        this.triggerResize();
      }
    } catch {
      this.voiceError.set('Transcription failed. Please try again.');
    } finally {
      this.clearVoiceHintTimer();
      this.voiceDownloadHint.set(false);
      this.isTranscribing.set(false);
    }
  }

  private clearVoiceHintTimer(): void {
    if (this.voiceHintTimer) {
      clearTimeout(this.voiceHintTimer);
      this.voiceHintTimer = null;
    }
  }

  /** Tooltip for the mic button, aware of model readiness and downloads */
  get micTitle(): string {
    if (this.isTranscribing() && this.voiceDownloadHint()) {
      // Only claim "downloading" when the model genuinely wasn't cached;
      // otherwise a slow CPU transcription would show a misleading message.
      return this.voiceApi.sttReady() ? 'Transcribing…' : 'Downloading voice model…';
    }
    if (this.isRecording()) return 'Stop recording';
    if (!this.voiceApi.sttReady()) return 'First use will download the voice model (~30s)';
    return 'Record voice message';
  }

  /**
   * Release microphone tracks and recorder resources
   */
  private releaseMicrophone(): void {
    this.mediaStream?.getTracks().forEach((track) => track.stop());
    this.mediaStream = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
  }

  ngAfterViewInit(): void {
    // Lifecycle hook - resize effect is in constructor
  }

  ngOnDestroy(): void {
    this.clearVoiceHintTimer();
    // Clean up mic resources: no transcription should fire after destroy
    if (this.recordingTimeout) {
      clearTimeout(this.recordingTimeout);
      this.recordingTimeout = null;
    }
    if (this.mediaRecorder) {
      this.mediaRecorder.onstop = null;
      if (this.mediaRecorder.state !== 'inactive') {
        this.mediaRecorder.stop();
      }
    }
    this.releaseMicrophone();
    this.isRecording.set(false);
    this.revokeImagePreview();
  }

  /**
   * Toggle the expanded topic summary under the "New topic" divider.
   * Only expands when the current session has a stored summary.
   */
  toggleTopicSummary(): void {
    if (!this.store.currentSession()?.topic_summary) return;
    this.showTopicSummary.update((v) => !v);
  }

  /**
   * TrackBy for the messages *ngFor
   */
  trackByMessage(index: number, msg: { role: string; created_at: string }): string {
    return `${msg.role}:${msg.created_at}`;
  }

  /**
   * Programmatically trigger textarea auto-resize
   * Called when content is inserted programmatically
   */
  triggerResize(): void {
    // Only run in browser environment
    if (!isPlatformBrowser(this.platformId) || typeof window === 'undefined') return;

    // Defer to next tick to ensure DOM is updated
    setTimeout(() => {
      const textarea = this.textareaRef?.nativeElement;
      if (textarea) {
        textarea.style.height = 'auto';
        const style = window.getComputedStyle(textarea);
        const maxHeight = parseFloat(style.maxHeight || '0');
        const max = Number.isFinite(maxHeight) && maxHeight > 0 ? maxHeight : Infinity;
        const nextHeight = Math.min(textarea.scrollHeight, max);
        textarea.style.height = `${nextHeight}px`;
        textarea.style.overflowY = textarea.scrollHeight > max ? 'auto' : 'hidden';
      }
    });
  }

  /**
   * Load available AI models from backend API.
   * An empty list is respected (no silent fallback) and flips the
   * first-run guidance flag; only a failed fetch falls back to defaults.
   */
  private loadModels(): void {
    this.http.get<AIModel[]>(`${this.config.apiBaseUrl}/chat/models`).subscribe({
      next: (data: AIModel[]) => {
        const list = data ?? [];
        this.models.set(list);
        this.store.availableModels.set(list);
        this.noModelsInstalled.set(list.length === 0);
        // If the selected model isn't actually installed, snap to the
        // first installed one so sending never 404s on the provider.
        if (list.length > 0 && !list.some((m) => m.id === this.store.currentModel().id)) {
          this.store.setModel(list[0]);
        }
      },
      error: () => {
        console.warn('Failed to load models from API, using defaults');
        this.models.set(AVAILABLE_MODELS);
        this.noModelsInstalled.set(true);
      },
    });
  }

  /**
   * Send message to AI with optional file attachment
   * Validates content and handles file attachments
   */
  send(): void {
    this.isErrorDismissed.set(false);
    const content = this.store.draftMessage().trim();

    if (!content) return;

    if (this.isFileUploading()) {
      this.fileError.set('Please wait for file extraction to finish.');
      return;
    }

    if (this.pendingFile && !this.fileContent()) {
      this.isFileUploading.set(true);
      this.fileError.set('Extracting content...');
      const fileToUpload = this.pendingFile;
      this.uploadFileToBackend(fileToUpload).subscribe({
        next: (response: { content: string; filename: string }) => {
          this.selectedFileName.set(response.filename);
          this.fileContent.set(response.content);
          this.fileError.set('');
          this.isFileUploading.set(false);
          this.pendingFile = null;
          this.performSend(content);
        },
        error: (err: any) => {
          this.fileError.set(err.error?.detail || 'Failed to extract file content.');
          this.selectedFileName.set('');
          this.fileContent.set('');
          this.isFileUploading.set(false);
          this.pendingFile = null;
        },
      });
      return;
    }

    this.performSend(content);
  }

  private performSend(content: string): void {
    // Attach file content if selected
    const attachment = this.fileContent()
      ? { filename: this.selectedFileName(), content: this.fileContent() }
      : undefined;

    this.store.sendMessage(content, attachment);

    // Clear draft and file selection
    this.store.clearDraft();
    this.selectedFileName.set('');
    this.fileContent.set('');
    this.fileError.set('');
    this.pendingFile = null;
    this.revokeImagePreview();
  }

  /**
   * Handle AI model selection change
   */
  onModelChange(modelId: string): void {
    const model = this.models().find((m: any) => m.id === modelId);
    if (model) {
      this.store.setModel(model);
    }
  }

  /**
   * Handle Enter key in textarea
   * Enter = send, Shift+Enter = new line
   */
  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  /**
   * Show the typing indicator while the AI is working and no
   * assistant tokens have arrived yet.
   */
  get isTyping(): boolean {
    if (!this.store.loading()) return false;
    const messages = this.store.messageList();
    if (!messages.length) return false;
    const last = messages[messages.length - 1];
    return last.role === 'user' || (last.role === 'assistant' && !last.content);
  }

  /**
   * Handle scroll event for infinite loading
   * Load older messages when user scrolls to top
   */
  onScroll(e: Event) {
    const el = e.target as HTMLElement;
    if (el.scrollTop < 20) {
      this.store.loadOlderMessages(this.store.currentSessionId()!);
    }
  }

  /**
   * Handle file selection from input
   * Supports both text files (read in browser) and binary files (uploaded to backend)
   */
  onFileSelected(event: Event): void {
    this.fileError.set('');
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    // Validate file size
    const maxBytes = this.config.fileUploadMaxBytes;
    if (file.size > maxBytes) {
      this.fileError.set('File too large. Max 10MB.');
      input.value = '';
      return;
    }

    // Check if image file (uploaded to backend, previewed as a thumbnail)
    const isImage =
      file.type.startsWith('image/') ||
      ChatPage.IMAGE_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext));

    // Check if binary file (requires server-side extraction)
    const isBinary = this.config.binaryExtensions.some((ext: any) =>
      file.name.toLowerCase().endsWith(ext),
    );

    this.isFileUploading.set(true);
    this.revokeImagePreview();

    if (isImage) {
      // Defer image upload until Send; show a thumbnail preview meanwhile
      this.pendingFile = file;
      this.selectedFileName.set(file.name);
      this.fileContent.set('');
      this.imagePreviewUrl.set(URL.createObjectURL(file));
      this.isFileUploading.set(false);
      input.value = '';
    } else if (isBinary) {
      // Defer binary file upload until Send
      this.pendingFile = file;
      this.selectedFileName.set(file.name);
      this.fileContent.set('');
      this.fileError.set('File ready. Click Send to upload and include it.');
      this.isFileUploading.set(false);
      input.value = '';
    } else {
      // Read text files directly in browser
      const reader = new FileReader();
      reader.onload = () => {
        const raw = String(reader.result ?? '');
        const content = raw.trim();
        if (!content) {
          this.fileError.set('File is empty or unreadable.');
          this.isFileUploading.set(false);
          input.value = '';
          return;
        }

        this.selectedFileName.set(file.name);
        this.fileContent.set(content);
        this.isFileUploading.set(false);
        input.value = '';
      };

      reader.onerror = () => {
        this.fileError.set('Failed to read file.');
        this.isFileUploading.set(false);
        input.value = '';
      };

      reader.readAsText(file);
    }
  }

  /**
   * Upload binary file to backend for text extraction
   * Backend handles PDF, Word, and other complex formats
   */
  private uploadFileToBackend(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post<{
      content: string;
      filename: string;
      is_image?: boolean;
    }>(`${this.config.apiBaseUrl}/chat/upload`, formData);
  }

  retryLoadSessions(): void {
    this.isErrorDismissed.set(false);
    this.store.loadSessions();
  }

  dismissSessionError(): void {
    this.store.sessionLoadError.set('');
  }

  toggleDropdown(event: MouseEvent) {
    event.stopPropagation();
    this.isDropdownOpen = !this.isDropdownOpen;
  }

  selectModel(model: any) {
    this.onModelChange(model.id);
    this.isDropdownOpen = false;
  }

  @HostListener('document:click')
  onClickOutside() {
    this.isDropdownOpen = false;
  }

  dismissError() {
    this.isErrorDismissed.set(true);
    this.store.error.set('');
  }

  clearFile(): void {
    this.selectedFileName.set('');
    this.fileContent.set('');
    this.fileError.set('');
    this.isFileUploading.set(false);
    this.pendingFile = null;
    this.revokeImagePreview();
  }

  /** Release the thumbnail object URL (removal / send / destroy) */
  private revokeImagePreview(): void {
    const url = this.imagePreviewUrl();
    if (url) {
      URL.revokeObjectURL(url);
      this.imagePreviewUrl.set('');
    }
  }

  isLastUserMessage(msg: any, index: number): boolean {
    const messages = this.store.messageList();
    if (msg.role !== 'user') return false;

    // Find the last user message index
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        return i === index;
      }
    }
    return false;
  }

  onEditAndResend(editedContent: string, attachment?: { filename: string; content: string }): void {
    if (!editedContent.trim() || this.store.loading()) return;

    // Remove the last user message and any assistant responses after it
    const messages = this.store.messageList();
    let lastUserIndex = -1;

    // Find last user message
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserIndex = i;
        break;
      }
    }

    if (lastUserIndex !== -1) {
      // Remove messages from last user message onwards
      this.store.removeMessagesFrom(lastUserIndex);
    }

    // Send the edited message with attachment if present
    this.store.sendMessage(editedContent, attachment);
  }
}
