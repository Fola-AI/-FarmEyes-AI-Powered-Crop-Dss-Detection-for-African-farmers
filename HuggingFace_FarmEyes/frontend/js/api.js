/**
 * FarmEyes API Client
 * ===================
 * Handles all communication with the FastAPI backend.
 * Provides clean async methods for detection, chat, and transcription.
 */

const FarmEyesAPI = {
    // Base URL - auto-detect based on environment
    baseUrl: window.location.origin,
    
    // Current session ID
    sessionId: null,
    
    // Current language
    language: 'en',
    
    /**
     * Initialize API client
     */
    async init() {
        // Try to get existing session from storage
        this.sessionId = localStorage.getItem('farmeyes_session');
        this.language = localStorage.getItem('farmeyes_language') || 'en';
        
        // Create new session if none exists
        if (!this.sessionId) {
            await this.createSession(this.language);
        }
        
        console.log('[API] Initialized with session:', this.sessionId?.substring(0, 8));
        return this;
    },
    
    /**
     * Make an API request
     * @param {string} endpoint - API endpoint
     * @param {object} options - Fetch options
     * @returns {Promise<object>} Response data
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const defaultOptions = {
            headers: {
                'Accept': 'application/json',
            },
        };
        
        // Merge options
        const fetchOptions = { ...defaultOptions, ...options };
        
        // Add Content-Type for JSON body
        if (options.body && !(options.body instanceof FormData)) {
            fetchOptions.headers['Content-Type'] = 'application/json';
            fetchOptions.body = JSON.stringify(options.body);
        }
        
        try {
            const response = await fetch(url, fetchOptions);
            
            // Handle non-JSON responses
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return { success: true };
            }
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || data.error || `HTTP ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error('[API] Request failed:', endpoint, error);
            throw error;
        }
    },
    
    // =========================================================================
    // SESSION MANAGEMENT
    // =========================================================================
    
    /**
     * Create a new session
     * @param {string} language - Language code
     * @returns {Promise<object>} Session data
     */
    async createSession(language = 'en') {
        const data = await this.request(`/api/session?language=${language}`);
        
        if (data.success && data.session_id) {
            this.sessionId = data.session_id;
            this.language = language;
            localStorage.setItem('farmeyes_session', this.sessionId);
            localStorage.setItem('farmeyes_language', language);
            console.log('[API] Session created:', this.sessionId.substring(0, 8));
        }
        
        return data;
    },
    
    /**
     * Get session info
     * @returns {Promise<object>} Session info
     */
    async getSession() {
        if (!this.sessionId) {
            return { success: false, error: 'No session' };
        }
        return this.request(`/api/session/${this.sessionId}`);
    },
    
    /**
     * Update session language
     * @param {string} language - New language code
     * @returns {Promise<object>} Updated session
     */
    async setLanguage(language) {
        if (!this.sessionId) {
            await this.createSession(language);
            return { success: true };
        }
        
        const data = await this.request(`/api/session/${this.sessionId}/language?language=${language}`, {
            method: 'PUT'
        });
        
        if (data.success) {
            this.language = language;
            localStorage.setItem('farmeyes_language', language);
        }
        
        return data;
    },
    
    /**
     * Clear current session and create new one
     * @returns {Promise<object>} New session data
     */
    async resetSession() {
        if (this.sessionId) {
            try {
                await this.request(`/api/session/${this.sessionId}`, { method: 'DELETE' });
            } catch (e) {
                // Ignore errors on delete
            }
        }
        
        localStorage.removeItem('farmeyes_session');
        this.sessionId = null;
        
        return this.createSession(this.language);
    },
    
    // =========================================================================
    // DISEASE DETECTION
    // =========================================================================
    
    /**
     * Analyze crop image for disease detection
     * @param {File} imageFile - Image file to analyze
     * @param {string} language - Language for results
     * @returns {Promise<object>} Detection results
     */
    async detectDisease(imageFile, language = null) {
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('language', language || this.language);
        formData.append('session_id', this.sessionId || '');
        
        const data = await this.request('/api/detect/', {
            method: 'POST',
            body: formData
        });
        
        // Update session ID if returned
        if (data.session_id) {
            this.sessionId = data.session_id;
            localStorage.setItem('farmeyes_session', this.sessionId);
        }
        
        return data;
    },
    
    /**
     * Analyze base64 encoded image
     * @param {string} base64Image - Base64 encoded image
     * @param {string} language - Language for results
     * @returns {Promise<object>} Detection results
     */
    async detectDiseaseBase64(base64Image, language = null) {
        const data = await this.request('/api/detect/base64', {
            method: 'POST',
            body: {
                image_base64: base64Image,
                language: language || this.language,
                session_id: this.sessionId
            }
        });
        
        if (data.session_id) {
            this.sessionId = data.session_id;
            localStorage.setItem('farmeyes_session', this.sessionId);
        }
        
        return data;
    },
    
    /**
     * Get detection service status
     * @returns {Promise<object>} Service status
     */
    async getDetectionStatus() {
        return this.request('/api/detect/status');
    },
    
    /**
     * Get supported disease classes
     * @returns {Promise<object>} Classes info
     */
    async getClasses() {
        return this.request('/api/detect/classes');
    },
    
    /**
     * Clear current diagnosis
     * @returns {Promise<object>} Result
     */
    async clearDiagnosis() {
        if (!this.sessionId) {
            return { success: false, error: 'No session' };
        }
        return this.request(`/api/detect/session/${this.sessionId}`, {
            method: 'DELETE'
        });
    },
    
    // =========================================================================
    // CHAT
    // =========================================================================
    
    /**
     * Send chat message
     * @param {string} message - User message
     * @param {string} language - Response language
     * @returns {Promise<object>} Chat response
     */
    async sendChatMessage(message, language = null) {
        if (!this.sessionId) {
            await this.createSession(language || this.language);
        }
        
        return this.request('/api/chat/', {
            method: 'POST',
            body: {
                session_id: this.sessionId,
                message: message,
                language: language || this.language
            }
        });
    },
    
    /**
     * Get welcome message for chat
     * @param {string} language - Language code
     * @returns {Promise<object>} Welcome message
     */
    async getChatWelcome(language = null) {
        if (!this.sessionId) {
            return { success: false, error: 'No session' };
        }
        
        const lang = language || this.language;
        return this.request(`/api/chat/welcome?session_id=${this.sessionId}&language=${lang}`);
    },
    
    /**
     * Get chat history
     * @param {number} limit - Max messages to return
     * @returns {Promise<object>} Chat history
     */
    async getChatHistory(limit = 50) {
        if (!this.sessionId) {
            return { success: false, messages: [] };
        }
        
        return this.request(`/api/chat/history?session_id=${this.sessionId}&limit=${limit}`);
    },
    
    /**
     * Clear chat history
     * @returns {Promise<object>} Result
     */
    async clearChatHistory() {
        if (!this.sessionId) {
            return { success: false };
        }
        
        return this.request(`/api/chat/history?session_id=${this.sessionId}`, {
            method: 'DELETE'
        });
    },
    
    /**
     * Get current diagnosis context
     * @returns {Promise<object>} Diagnosis context
     */
    async getChatContext() {
        if (!this.sessionId) {
            return { success: false };
        }
        
        return this.request(`/api/chat/context?session_id=${this.sessionId}`);
    },
    
    // =========================================================================
    // VOICE TRANSCRIPTION
    // =========================================================================
    
    /**
     * Transcribe audio file
     * @param {File|Blob} audioFile - Audio file to transcribe
     * @param {string} languageHint - Language hint
     * @returns {Promise<object>} Transcription result
     */
    async transcribeAudio(audioFile, languageHint = null) {
        const formData = new FormData();
        formData.append('file', audioFile, audioFile.name || 'audio.wav');
        
        if (languageHint) {
            formData.append('language_hint', languageHint);
        }
        
        return this.request('/api/transcribe/', {
            method: 'POST',
            body: formData
        });
    },
    
    /**
     * Transcribe base64 audio
     * @param {string} base64Audio - Base64 encoded audio
     * @param {string} filename - Original filename
     * @param {string} languageHint - Language hint
     * @returns {Promise<object>} Transcription result
     */
    async transcribeBase64(base64Audio, filename = 'audio.wav', languageHint = null) {
        return this.request('/api/transcribe/base64', {
            method: 'POST',
            body: {
                audio_base64: base64Audio,
                filename: filename,
                language_hint: languageHint
            }
        });
    },
    
    /**
     * Get transcription service status
     * @returns {Promise<object>} Service status
     */
    async getTranscriptionStatus() {
        return this.request('/api/transcribe/status');
    },
    
    /**
     * Pre-load Whisper model
     * @returns {Promise<object>} Result
     */
    async loadWhisperModel() {
        return this.request('/api/transcribe/load-model', {
            method: 'POST'
        });
    },
    
    // =========================================================================
    // UI TRANSLATIONS
    // =========================================================================
    
    /**
     * Get UI translations
     * @param {string} language - Language code
     * @returns {Promise<object>} Translations
     */
    async getTranslations(language = null) {
        const lang = language || this.language;
        return this.request(`/api/translations?language=${lang}`);
    },
    
    // =========================================================================
    // HEALTH CHECK
    // =========================================================================
    
    /**
     * Check API health
     * @returns {Promise<object>} Health status
     */
    async healthCheck() {
        return this.request('/health');
    },
    
    /**
     * Get API info
     * @returns {Promise<object>} API information
     */
    async getApiInfo() {
        return this.request('/api');
    }
};

// Export for use in other modules
window.FarmEyesAPI = FarmEyesAPI;
