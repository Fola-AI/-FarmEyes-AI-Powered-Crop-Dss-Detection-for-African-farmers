/**
 * FarmEyes TTS Module
 * ===================
 * Text-to-Speech functionality using Meta MMS-TTS via HuggingFace API.
 * 
 * Features:
 * - Play/Pause/Stop controls
 * - Speed control (0.75x, 1x, 1.25x, 1.5x)
 * - Audio caching (browser session)
 * - Web Speech API fallback for English
 * - Floating player for last message
 */

const TTS = {
    // ==========================================================================
    // STATE
    // ==========================================================================
    
    // Current audio state
    isPlaying: false,
    isPaused: false,
    currentAudio: null,
    currentMessageId: null,
    
    // Playback settings
    playbackRate: 1.0,
    
    // Cache for generated audio (session-based)
    audioCache: new Map(),
    
    // Web Speech API fallback
    speechSynthesis: window.speechSynthesis || null,
    
    // Callbacks
    onPlayStart: null,
    onPlayEnd: null,
    onError: null,
    
    // ==========================================================================
    // INITIALIZATION
    // ==========================================================================
    
    /**
     * Initialize TTS module
     * @param {object} callbacks - Optional callback functions
     */
    init(callbacks = {}) {
        this.onPlayStart = callbacks.onPlayStart || (() => {});
        this.onPlayEnd = callbacks.onPlayEnd || (() => {});
        this.onError = callbacks.onError || ((err) => console.error('[TTS]', err));
        
        // Check Web Speech API availability
        if (this.speechSynthesis) {
            console.log('[TTS] Web Speech API available for fallback');
        }
        
        console.log('[TTS] Initialized');
    },
    
    // ==========================================================================
    // PUBLIC API
    // ==========================================================================
    
    /**
     * Speak text using TTS
     * @param {string} text - Text to speak
     * @param {string} language - Language code (en, ha, yo, ig)
     * @param {string} messageId - Unique message identifier for caching
     * @returns {Promise<boolean>} Success status
     */
    async speak(text, language = 'en', messageId = null) {
        // Stop any current playback
        this.stop();
        
        // Generate cache key
        const cacheKey = messageId || this.generateCacheKey(text, language);
        
        // Check cache first
        if (this.audioCache.has(cacheKey)) {
            console.log('[TTS] Using cached audio');
            return this.playFromCache(cacheKey);
        }
        
        // Try MMS-TTS API
        try {
            console.log(`[TTS] Synthesizing: lang=${language}, length=${text.length}`);
            
            const result = await this.synthesizeAPI(text, language);
            
            if (result.success && result.audio_base64) {
                // Cache the audio
                this.audioCache.set(cacheKey, {
                    audio_base64: result.audio_base64,
                    content_type: result.content_type,
                    language: language
                });
                
                // Play it
                return this.playAudio(result.audio_base64, result.content_type, cacheKey);
            } else {
                throw new Error(result.error || 'TTS synthesis failed');
            }
            
        } catch (error) {
            console.error('[TTS] API failed:', error);
            
            // Try fallback for English
            if (language === 'en' && this.speechSynthesis) {
                console.log('[TTS] Falling back to Web Speech API');
                return this.speakWithWebSpeech(text);
            }
            
            // No fallback available
            this.onError(`Voice playback failed: ${error.message}`);
            return false;
        }
    },
    
    /**
     * Play audio from cache
     * @param {string} cacheKey - Cache key
     * @returns {Promise<boolean>} Success status
     */
    async playFromCache(cacheKey) {
        const cached = this.audioCache.get(cacheKey);
        if (!cached) return false;
        
        return this.playAudio(cached.audio_base64, cached.content_type, cacheKey);
    },
    
    /**
     * Pause current playback
     */
    pause() {
        if (this.currentAudio && this.isPlaying) {
            this.currentAudio.pause();
            this.isPlaying = false;
            this.isPaused = true;
            console.log('[TTS] Paused');
            this.updatePlayerUI('paused');
        }
    },
    
    /**
     * Resume paused playback
     */
    resume() {
        if (this.currentAudio && this.isPaused) {
            this.currentAudio.play();
            this.isPlaying = true;
            this.isPaused = false;
            console.log('[TTS] Resumed');
            this.updatePlayerUI('playing');
        }
    },
    
    /**
     * Toggle play/pause
     */
    togglePlayPause() {
        if (this.isPlaying) {
            this.pause();
        } else if (this.isPaused) {
            this.resume();
        }
    },
    
    /**
     * Stop playback completely
     */
    stop() {
        // Stop HTML5 Audio
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
        }
        
        // Stop Web Speech API
        if (this.speechSynthesis) {
            this.speechSynthesis.cancel();
        }
        
        this.isPlaying = false;
        this.isPaused = false;
        this.currentMessageId = null;
        
        console.log('[TTS] Stopped');
        this.updatePlayerUI('stopped');
        this.onPlayEnd();
    },
    
    /**
     * Set playback speed
     * @param {number} rate - Playback rate (0.5 - 2.0)
     */
    setPlaybackRate(rate) {
        this.playbackRate = Math.max(0.5, Math.min(2.0, rate));
        
        if (this.currentAudio) {
            this.currentAudio.playbackRate = this.playbackRate;
        }
        
        console.log(`[TTS] Playback rate: ${this.playbackRate}x`);
        this.updateSpeedButtonsUI();
    },
    
    /**
     * Check if currently playing
     * @returns {boolean}
     */
    getIsPlaying() {
        return this.isPlaying;
    },
    
    /**
     * Check if paused
     * @returns {boolean}
     */
    getIsPaused() {
        return this.isPaused;
    },
    
    /**
     * Get current playback rate
     * @returns {number}
     */
    getPlaybackRate() {
        return this.playbackRate;
    },
    
    // ==========================================================================
    // API COMMUNICATION
    // ==========================================================================
    
    /**
     * Call TTS API to synthesize speech
     * @param {string} text - Text to synthesize
     * @param {string} language - Language code
     * @returns {Promise<object>} API response
     */
    async synthesizeAPI(text, language) {
        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                language: language
            })
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        return await response.json();
    },
    
    // ==========================================================================
    // AUDIO PLAYBACK
    // ==========================================================================
    
    /**
     * Play audio from base64 data
     * @param {string} audioBase64 - Base64 encoded audio
     * @param {string} contentType - MIME type
     * @param {string} messageId - Message identifier
     * @returns {Promise<boolean>} Success status
     */
    async playAudio(audioBase64, contentType, messageId) {
        try {
            // Create blob from base64
            const binaryString = atob(audioBase64);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            const blob = new Blob([bytes], { type: contentType });
            const audioUrl = URL.createObjectURL(blob);
            
            // Create audio element
            this.currentAudio = new Audio(audioUrl);
            this.currentAudio.playbackRate = this.playbackRate;
            this.currentMessageId = messageId;
            
            // Set up event handlers
            this.currentAudio.onplay = () => {
                this.isPlaying = true;
                this.isPaused = false;
                this.onPlayStart();
                this.updatePlayerUI('playing');
            };
            
            this.currentAudio.onpause = () => {
                if (!this.currentAudio.ended) {
                    this.updatePlayerUI('paused');
                }
            };
            
            this.currentAudio.onended = () => {
                this.isPlaying = false;
                this.isPaused = false;
                this.currentMessageId = null;
                URL.revokeObjectURL(audioUrl);
                this.onPlayEnd();
                this.updatePlayerUI('stopped');
                console.log('[TTS] Playback ended');
            };
            
            this.currentAudio.onerror = (e) => {
                console.error('[TTS] Audio error:', e);
                this.stop();
                this.onError('Audio playback failed');
            };
            
            // Update UI for time tracking
            this.currentAudio.ontimeupdate = () => {
                this.updateProgressUI();
            };
            
            // Start playback
            await this.currentAudio.play();
            console.log('[TTS] Playing audio');
            
            return true;
            
        } catch (error) {
            console.error('[TTS] Play error:', error);
            this.onError('Failed to play audio');
            return false;
        }
    },
    
    // ==========================================================================
    // WEB SPEECH API FALLBACK (English only)
    // ==========================================================================
    
    /**
     * Speak using Web Speech API (fallback for English)
     * @param {string} text - Text to speak
     * @returns {Promise<boolean>} Success status
     */
    async speakWithWebSpeech(text) {
        return new Promise((resolve) => {
            if (!this.speechSynthesis) {
                this.onError('Web Speech API not available');
                resolve(false);
                return;
            }
            
            // Cancel any ongoing speech
            this.speechSynthesis.cancel();
            
            // Create utterance
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            utterance.rate = this.playbackRate;
            utterance.pitch = 1.0;
            
            // Get English voice
            const voices = this.speechSynthesis.getVoices();
            const englishVoice = voices.find(v => v.lang.startsWith('en'));
            if (englishVoice) {
                utterance.voice = englishVoice;
            }
            
            // Event handlers
            utterance.onstart = () => {
                this.isPlaying = true;
                this.isPaused = false;
                this.onPlayStart();
                this.updatePlayerUI('playing');
            };
            
            utterance.onend = () => {
                this.isPlaying = false;
                this.isPaused = false;
                this.onPlayEnd();
                this.updatePlayerUI('stopped');
                resolve(true);
            };
            
            utterance.onerror = (e) => {
                console.error('[TTS] Web Speech error:', e);
                this.onError('Speech synthesis failed');
                resolve(false);
            };
            
            // Speak
            this.speechSynthesis.speak(utterance);
            console.log('[TTS] Using Web Speech API');
        });
    },
    
    // ==========================================================================
    // UI HELPERS
    // ==========================================================================
    
    /**
     * Generate cache key from text and language
     * @param {string} text - Text content
     * @param {string} language - Language code
     * @returns {string} Cache key
     */
    generateCacheKey(text, language) {
        // Simple hash-like key
        const hash = text.slice(0, 50).replace(/\s+/g, '_');
        return `${language}_${hash}_${text.length}`;
    },
    
    /**
     * Update player UI state
     * @param {string} state - 'playing', 'paused', 'stopped'
     */
    updatePlayerUI(state) {
        const player = document.getElementById('tts-player');
        const btnPlayPause = document.getElementById('tts-play-pause');
        
        if (!player) return;
        
        switch (state) {
            case 'playing':
                player.classList.add('active');
                if (btnPlayPause) {
                    btnPlayPause.innerHTML = this.getPauseIcon();
                    btnPlayPause.title = 'Pause';
                }
                break;
                
            case 'paused':
                player.classList.add('active');
                if (btnPlayPause) {
                    btnPlayPause.innerHTML = this.getPlayIcon();
                    btnPlayPause.title = 'Resume';
                }
                break;
                
            case 'stopped':
                player.classList.remove('active');
                this.resetProgressUI();
                break;
        }
    },
    
    /**
     * Update progress bar and time display
     */
    updateProgressUI() {
        if (!this.currentAudio) return;
        
        const progress = document.getElementById('tts-progress');
        const timeDisplay = document.getElementById('tts-time');
        
        const current = this.currentAudio.currentTime;
        const duration = this.currentAudio.duration || 0;
        
        if (progress && duration > 0) {
            const percent = (current / duration) * 100;
            progress.style.width = `${percent}%`;
        }
        
        if (timeDisplay && duration > 0) {
            const formatTime = (t) => {
                const mins = Math.floor(t / 60);
                const secs = Math.floor(t % 60);
                return `${mins}:${secs.toString().padStart(2, '0')}`;
            };
            timeDisplay.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
        }
    },
    
    /**
     * Reset progress UI
     */
    resetProgressUI() {
        const progress = document.getElementById('tts-progress');
        const timeDisplay = document.getElementById('tts-time');
        
        if (progress) progress.style.width = '0%';
        if (timeDisplay) timeDisplay.textContent = '0:00';
    },
    
    /**
     * Update speed buttons to show active state
     */
    updateSpeedButtonsUI() {
        const buttons = document.querySelectorAll('.tts-speed-btn');
        buttons.forEach(btn => {
            const rate = parseFloat(btn.dataset.rate);
            btn.classList.toggle('active', rate === this.playbackRate);
        });
    },
    
    /**
     * Get play icon SVG
     */
    getPlayIcon() {
        return `<svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
            <polygon points="5,3 19,12 5,21"/>
        </svg>`;
    },
    
    /**
     * Get pause icon SVG
     */
    getPauseIcon() {
        return `<svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
            <rect x="6" y="4" width="4" height="16"/>
            <rect x="14" y="4" width="4" height="16"/>
        </svg>`;
    },
    
    /**
     * Clear audio cache
     */
    clearCache() {
        this.audioCache.clear();
        console.log('[TTS] Cache cleared');
    }
};

// Export for use in other modules
window.TTS = TTS;
