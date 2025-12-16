/**
 * FarmEyes Chat Module
 * ====================
 * Handles the chat interface, message sending, and voice input.
 * 
 * Updated: Inline "Listening..." indicator with timer (no full-screen overlay)
 * Updated: TTS Listen button on assistant messages
 */

const Chat = {
    // State
    isLoading: false,
    messages: [],
    
    // Voice recording state
    recordingTimer: null,
    recordingSeconds: 0,
    
    // Message ID counter for TTS
    messageIdCounter: 0,
    
    // DOM Elements
    elements: {},
    
    /**
     * Initialize chat module
     */
    init() {
        this.cacheElements();
        this.bindEvents();
        this.initVoiceInput();
        this.initTTS();
        this.createTTSPlayer();
        console.log('[Chat] Initialized');
    },
    
    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            // Header
            btnBack: document.getElementById('btn-back-diagnosis'),
            btnLanguage: document.getElementById('btn-chat-language'),
            chatLangDisplay: document.getElementById('chat-lang-display'),
            
            // Context banner
            contextBanner: document.getElementById('chat-context-banner'),
            contextDiseaseName: document.getElementById('context-disease-name'),
            contextConfidence: document.getElementById('context-confidence'),
            contextSeverity: document.getElementById('context-severity'),
            
            // Messages
            messagesContainer: document.getElementById('chat-messages'),
            chatWelcome: document.getElementById('chat-welcome'),
            
            // Input
            chatInput: document.getElementById('chat-input'),
            btnVoice: document.getElementById('btn-voice-input'),
            btnSend: document.getElementById('btn-send-message'),
            chatInputBox: document.querySelector('.chat-input-box'),
            
            // Voice overlay (keep reference but won't use full-screen)
            voiceOverlay: document.getElementById('voice-overlay'),
            btnStopVoice: document.getElementById('btn-stop-voice')
        };
    },
    
    /**
     * Bind event handlers
     */
    bindEvents() {
        const { btnBack, chatInput, btnVoice, btnSend, btnStopVoice } = this.elements;
        
        // Back button
        btnBack?.addEventListener('click', () => App.navigateToDiagnosis());
        
        // Input events
        chatInput?.addEventListener('input', () => this.handleInputChange());
        chatInput?.addEventListener('keydown', (e) => this.handleKeyDown(e));
        
        // Send button
        btnSend?.addEventListener('click', () => this.sendMessage());
        
        // Voice buttons
        btnVoice?.addEventListener('click', () => this.toggleVoiceRecording());
        btnStopVoice?.addEventListener('click', () => this.stopVoiceRecording());
        
        // Auto-resize input
        chatInput?.addEventListener('input', () => this.autoResizeInput());
    },
    
    /**
     * Initialize voice input
     */
    initVoiceInput() {
        VoiceInput.init({
            onTranscription: (text, result) => {
                this.handleVoiceTranscription(text, result);
            },
            onError: (error) => {
                App.showToast(error, 'error');
                this.hideListeningIndicator();
            },
            onRecordingStart: () => {
                this.showListeningIndicator();
            },
            onRecordingStop: () => {
                this.hideListeningIndicator();
            }
        });
    },
    
    /**
     * Initialize TTS (Text-to-Speech)
     */
    initTTS() {
        TTS.init({
            onPlayStart: () => {
                console.log('[Chat] TTS playback started');
            },
            onPlayEnd: () => {
                console.log('[Chat] TTS playback ended');
                this.updateListenButtons();
            },
            onError: (error) => {
                App.showToast(error, 'error');
                this.updateListenButtons();
            }
        });
    },
    
    /**
     * Create floating TTS player element
     */
    createTTSPlayer() {
        // Check if player already exists
        if (document.getElementById('tts-player')) return;
        
        const player = document.createElement('div');
        player.id = 'tts-player';
        player.className = 'tts-player';
        player.innerHTML = `
            <div class="tts-player-header">
                <div class="tts-player-title">
                    <span class="tts-player-title-icon">🔊</span>
                    <span>Now Playing</span>
                </div>
                <button class="btn-tts-close" id="tts-close" title="Close">×</button>
            </div>
            <div class="tts-progress-container" id="tts-progress-container">
                <div class="tts-progress-bar" id="tts-progress"></div>
            </div>
            <div class="tts-controls">
                <div class="tts-playback-controls">
                    <button class="btn-tts-control" id="tts-play-pause" title="Play/Pause">
                        <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                            <polygon points="5,3 19,12 5,21"/>
                        </svg>
                    </button>
                    <button class="btn-tts-control stop" id="tts-stop" title="Stop">
                        <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16">
                            <rect x="6" y="6" width="12" height="12" rx="2"/>
                        </svg>
                    </button>
                    <span class="tts-time" id="tts-time">0:00</span>
                </div>
                <div class="tts-speed-controls">
                    <span class="tts-speed-label">Speed:</span>
                    <button class="tts-speed-btn" data-rate="0.75">0.75x</button>
                    <button class="tts-speed-btn active" data-rate="1">1x</button>
                    <button class="tts-speed-btn" data-rate="1.25">1.25x</button>
                    <button class="tts-speed-btn" data-rate="1.5">1.5x</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(player);
        
        // Bind player events
        document.getElementById('tts-close')?.addEventListener('click', () => {
            TTS.stop();
        });
        
        document.getElementById('tts-play-pause')?.addEventListener('click', () => {
            TTS.togglePlayPause();
        });
        
        document.getElementById('tts-stop')?.addEventListener('click', () => {
            TTS.stop();
        });
        
        // Speed buttons
        document.querySelectorAll('.tts-speed-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const rate = parseFloat(btn.dataset.rate);
                TTS.setPlaybackRate(rate);
            });
        });
        
        console.log('[Chat] TTS player created');
    },
    
    // =========================================================================
    // CHAT PAGE LIFECYCLE
    // =========================================================================
    
    /**
     * Called when chat page becomes active
     */
    async onPageEnter() {
        console.log('[Chat] Page entered');
        
        // Update context banner
        this.updateContextBanner();
        
        // Update language display
        this.updateLanguageDisplay();
        
        // Load chat history or get welcome
        await this.loadChat();
        
        // Focus input
        this.elements.chatInput?.focus();
    },
    
    /**
     * Called when leaving chat page
     */
    onPageLeave() {
        console.log('[Chat] Page left');
        // Stop any ongoing recording
        if (VoiceInput.getIsRecording()) {
            VoiceInput.cancelRecording();
            this.hideListeningIndicator();
        }
    },
    
    /**
     * Update the context banner with diagnosis info
     */
    updateContextBanner() {
        const diagnosis = Diagnosis.getDiagnosis();
        
        if (!diagnosis) {
            this.elements.contextBanner?.classList.add('hidden');
            return;
        }
        
        this.elements.contextBanner?.classList.remove('hidden');
        
        const { detection } = diagnosis;
        this.elements.contextDiseaseName.textContent = detection.disease_name || 'Unknown';
        this.elements.contextConfidence.textContent = `${Math.round(detection.confidence_percent || 0)}%`;
        this.elements.contextSeverity.textContent = I18n.getSeverity(detection.severity_level || 'unknown');
    },
    
    /**
     * Update language display
     */
    updateLanguageDisplay() {
        const lang = I18n.getLanguage();
        if (this.elements.chatLangDisplay) {
            this.elements.chatLangDisplay.textContent = lang.toUpperCase();
        }
    },
    
    /**
     * Load chat history or welcome message
     */
    async loadChat() {
        // Clear existing messages
        this.clearMessages();
        
        try {
            // Try to get existing history
            const history = await FarmEyesAPI.getChatHistory();
            
            if (history.success && history.messages && history.messages.length > 0) {
                // Display existing messages
                this.messages = history.messages;
                this.displayMessages(history.messages);
            } else {
                // Get welcome message
                const welcome = await FarmEyesAPI.getChatWelcome(I18n.getLanguage());
                
                if (welcome.success && welcome.response) {
                    this.addMessage('assistant', welcome.response);
                } else {
                    // Show default welcome
                    this.showWelcome();
                }
            }
        } catch (error) {
            console.error('[Chat] Load failed:', error);
            this.showWelcome();
        }
    },
    
    // =========================================================================
    // MESSAGE HANDLING
    // =========================================================================
    
    /**
     * Handle input change
     */
    handleInputChange() {
        const text = this.elements.chatInput?.value?.trim();
        this.elements.btnSend.disabled = !text || this.isLoading;
    },
    
    /**
     * Handle keyboard input
     * @param {KeyboardEvent} event
     */
    handleKeyDown(event) {
        // Send on Enter (without Shift)
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    },
    
    /**
     * Auto-resize textarea
     */
    autoResizeInput() {
        const input = this.elements.chatInput;
        if (!input) return;
        
        input.style.height = 'auto';
        const newHeight = Math.min(input.scrollHeight, 150);
        input.style.height = `${newHeight}px`;
    },
    
    /**
     * Send a chat message
     */
    async sendMessage() {
        const input = this.elements.chatInput;
        const message = input?.value?.trim();
        
        if (!message || this.isLoading) return;
        
        // Clear input
        input.value = '';
        this.autoResizeInput();
        this.handleInputChange();
        
        // Add user message to UI
        this.addMessage('user', message);
        
        // Send to API
        this.isLoading = true;
        this.showTypingIndicator();
        
        try {
            const response = await FarmEyesAPI.sendChatMessage(message, I18n.getLanguage());
            
            if (response.success) {
                this.addMessage('assistant', response.response);
            } else {
                throw new Error(response.error || 'Failed to get response');
            }
        } catch (error) {
            console.error('[Chat] Send failed:', error);
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            App.showToast(error.message, 'error');
        } finally {
            this.isLoading = false;
            this.hideTypingIndicator();
            this.handleInputChange();
        }
    },
    
    /**
     * Add a message to the chat
     * @param {string} role - 'user' or 'assistant'
     * @param {string} content - Message content
     */
    addMessage(role, content) {
        // Hide welcome if visible
        this.elements.chatWelcome?.classList.add('hidden');
        
        // Create message element
        const messageEl = this.createMessageElement(role, content);
        
        // Add to container
        this.elements.messagesContainer?.appendChild(messageEl);
        
        // Store in array
        this.messages.push({ role, content, timestamp: new Date().toISOString() });
        
        // Scroll to bottom
        this.scrollToBottom();
    },
    
    /**
     * Create message DOM element
     * @param {string} role
     * @param {string} content
     * @returns {HTMLElement}
     */
    createMessageElement(role, content) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        
        // Generate unique message ID for TTS caching
        const messageId = `msg_${++this.messageIdCounter}_${Date.now()}`;
        div.dataset.messageId = messageId;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? '👤' : '🌱';
        
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-content-wrapper';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        contentWrapper.appendChild(contentDiv);
        
        // Add Listen button for assistant messages
        if (role === 'assistant') {
            const listenBtn = document.createElement('button');
            listenBtn.className = 'btn-listen';
            listenBtn.dataset.messageId = messageId;
            listenBtn.dataset.text = content;
            listenBtn.innerHTML = `
                <span class="btn-listen-icon">🔊</span>
                <span class="btn-listen-text">Listen</span>
            `;
            listenBtn.title = 'Listen to this message';
            listenBtn.addEventListener('click', () => this.handleListenClick(listenBtn, content, messageId));
            
            contentWrapper.appendChild(listenBtn);
        }
        
        div.appendChild(avatar);
        div.appendChild(contentWrapper);
        
        return div;
    },
    
    /**
     * Handle Listen button click
     * @param {HTMLElement} button - The listen button
     * @param {string} text - Message text
     * @param {string} messageId - Unique message ID
     */
    async handleListenClick(button, text, messageId) {
        // If already playing this message, toggle pause
        if (TTS.currentMessageId === messageId) {
            if (TTS.getIsPlaying()) {
                TTS.pause();
                button.innerHTML = `<span class="btn-listen-icon">▶️</span><span class="btn-listen-text">Resume</span>`;
            } else if (TTS.getIsPaused()) {
                TTS.resume();
                button.innerHTML = `<span class="btn-listen-icon">⏸️</span><span class="btn-listen-text">Pause</span>`;
            }
            return;
        }
        
        // Reset all other listen buttons
        this.updateListenButtons();
        
        // Show loading state
        button.classList.add('loading');
        button.innerHTML = `<span class="btn-listen-icon">🔊</span><span class="btn-listen-text">Loading...</span>`;
        
        // Get current language
        const language = I18n.getLanguage();
        
        // Start TTS
        const success = await TTS.speak(text, language, messageId);
        
        // Update button state
        button.classList.remove('loading');
        
        if (success) {
            button.classList.add('playing');
            button.innerHTML = `<span class="btn-listen-icon">⏸️</span><span class="btn-listen-text">Pause</span>`;
        } else {
            button.innerHTML = `<span class="btn-listen-icon">🔊</span><span class="btn-listen-text">Listen</span>`;
        }
    },
    
    /**
     * Update all listen buttons to default state
     */
    updateListenButtons() {
        document.querySelectorAll('.btn-listen').forEach(btn => {
            btn.classList.remove('loading', 'playing');
            btn.innerHTML = `<span class="btn-listen-icon">🔊</span><span class="btn-listen-text">Listen</span>`;
        });
    },
    
    /**
     * Display multiple messages
     * @param {Array} messages
     */
    displayMessages(messages) {
        this.elements.chatWelcome?.classList.add('hidden');
        
        messages.forEach(msg => {
            const messageEl = this.createMessageElement(msg.role, msg.content);
            this.elements.messagesContainer?.appendChild(messageEl);
        });
        
        this.scrollToBottom();
    },
    
    /**
     * Clear all messages
     */
    clearMessages() {
        if (this.elements.messagesContainer) {
            this.elements.messagesContainer.innerHTML = '';
            // Re-add welcome
            const welcome = document.createElement('div');
            welcome.id = 'chat-welcome';
            welcome.className = 'chat-welcome';
            welcome.innerHTML = `
                <div class="welcome-icon">🌱</div>
                <p class="welcome-text">Start a conversation about your diagnosis</p>
            `;
            this.elements.messagesContainer.appendChild(welcome);
            this.elements.chatWelcome = welcome;
        }
        this.messages = [];
    },
    
    /**
     * Show welcome screen
     */
    showWelcome() {
        this.elements.chatWelcome?.classList.remove('hidden');
    },
    
    /**
     * Show typing indicator
     */
    showTypingIndicator() {
        // Remove existing indicator
        this.hideTypingIndicator();
        
        const indicator = document.createElement('div');
        indicator.className = 'message assistant typing-message';
        indicator.innerHTML = `
            <div class="message-avatar">🌱</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        
        this.elements.messagesContainer?.appendChild(indicator);
        this.scrollToBottom();
    },
    
    /**
     * Hide typing indicator
     */
    hideTypingIndicator() {
        const indicator = this.elements.messagesContainer?.querySelector('.typing-message');
        indicator?.remove();
    },
    
    /**
     * Scroll chat to bottom
     */
    scrollToBottom() {
        const container = this.elements.messagesContainer;
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    },
    
    // =========================================================================
    // VOICE INPUT - INLINE LISTENING INDICATOR
    // =========================================================================
    
    /**
     * Toggle voice recording
     */
    async toggleVoiceRecording() {
        if (!VoiceInput.isSupported()) {
            App.showToast('Voice input is not supported in this browser', 'error');
            return;
        }
        
        if (VoiceInput.getIsRecording()) {
            this.stopVoiceRecording();
        } else {
            await this.startVoiceRecording();
        }
    },
    
    /**
     * Start voice recording
     */
    async startVoiceRecording() {
        const started = await VoiceInput.startRecording();
        
        if (!started) {
            // Error handled by VoiceInput callback
            return;
        }
    },
    
    /**
     * Stop voice recording
     */
    stopVoiceRecording() {
        VoiceInput.stopRecording();
    },
    
    /**
     * Handle voice transcription result
     * @param {string} text - Transcribed text
     * @param {object} result - Full result object
     */
    handleVoiceTranscription(text, result) {
        if (!text) {
            App.showToast('Could not understand audio. Please try again.', 'warning');
            return;
        }
        
        // Put text in input
        if (this.elements.chatInput) {
            this.elements.chatInput.value = text;
            this.autoResizeInput();
            this.handleInputChange();
            
            // Optionally auto-send
            // this.sendMessage();
        }
        
        // Show language detected
        if (result.language) {
            console.log('[Chat] Detected language:', result.language);
        }
    },
    
    /**
     * Show inline listening indicator in chat bar
     * Replaces the textarea with a listening indicator + timer
     */
    showListeningIndicator() {
        const inputBox = this.elements.chatInputBox;
        const textarea = this.elements.chatInput;
        const btnVoice = this.elements.btnVoice;
        const btnSend = this.elements.btnSend;
        
        if (!inputBox) return;
        
        // Hide textarea and send button
        textarea?.classList.add('hidden');
        btnSend?.classList.add('hidden');
        
        // Update voice button to stop style
        btnVoice?.classList.add('recording');
        
        // Create listening indicator
        const listeningIndicator = document.createElement('div');
        listeningIndicator.id = 'listening-indicator';
        listeningIndicator.className = 'listening-indicator';
        listeningIndicator.innerHTML = `
            <div class="listening-pulse"></div>
            <span class="listening-text">Listening...</span>
            <span class="listening-timer">0:00</span>
            <button class="btn-stop-inline" title="Stop Recording">
                <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18">
                    <rect x="6" y="6" width="12" height="12" rx="2"/>
                </svg>
            </button>
        `;
        
        // Insert before voice button
        inputBox.insertBefore(listeningIndicator, btnVoice);
        
        // Bind stop button
        const btnStop = listeningIndicator.querySelector('.btn-stop-inline');
        btnStop?.addEventListener('click', () => this.stopVoiceRecording());
        
        // Start timer
        this.recordingSeconds = 0;
        this.updateRecordingTimer();
        this.recordingTimer = setInterval(() => {
            this.recordingSeconds++;
            this.updateRecordingTimer();
        }, 1000);
        
        console.log('[Chat] Listening indicator shown');
    },
    
    /**
     * Hide inline listening indicator
     */
    hideListeningIndicator() {
        const inputBox = this.elements.chatInputBox;
        const textarea = this.elements.chatInput;
        const btnVoice = this.elements.btnVoice;
        const btnSend = this.elements.btnSend;
        
        // Remove listening indicator
        const indicator = document.getElementById('listening-indicator');
        indicator?.remove();
        
        // Show textarea and send button
        textarea?.classList.remove('hidden');
        btnSend?.classList.remove('hidden');
        
        // Update voice button
        btnVoice?.classList.remove('recording');
        
        // Stop timer
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
        this.recordingSeconds = 0;
        
        console.log('[Chat] Listening indicator hidden');
    },
    
    /**
     * Update the recording timer display
     */
    updateRecordingTimer() {
        const timerEl = document.querySelector('.listening-timer');
        if (timerEl) {
            const minutes = Math.floor(this.recordingSeconds / 60);
            const seconds = this.recordingSeconds % 60;
            timerEl.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    },
    
    // =========================================================================
    // LEGACY OVERLAY METHODS (kept for compatibility but not used)
    // =========================================================================
    
    /**
     * Show voice recording overlay (LEGACY - not used)
     */
    showVoiceOverlay() {
        // Replaced by showListeningIndicator()
        this.showListeningIndicator();
    },
    
    /**
     * Hide voice recording overlay (LEGACY - not used)
     */
    hideVoiceOverlay() {
        // Replaced by hideListeningIndicator()
        this.hideListeningIndicator();
    }
};

// Export for use in other modules
window.Chat = Chat;
