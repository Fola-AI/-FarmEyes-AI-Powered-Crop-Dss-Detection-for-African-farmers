/**
 * FarmEyes Voice Input - Robust Implementation
 * =============================================
 * Handles voice recording and transcription using Web Audio API
 * and backend Whisper service.
 * 
 * Pipeline: Voice → Whisper → Text → N-ATLaS → Response
 * 
 * Features:
 * - Comprehensive browser compatibility detection
 * - Detailed error logging for debugging
 * - Secure context verification
 * - Graceful fallbacks for older browsers
 * - Safari and Chrome-specific handling
 * 
 * @author FarmEyes Team
 * @version 2.0.0
 */

const VoiceInput = {
    // ==========================================================================
    // STATE MANAGEMENT
    // ==========================================================================
    
    // Recording state
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    stream: null,
    
    // Configuration
    maxDuration: 30000,        // 30 seconds max recording
    minDuration: 500,          // 0.5 seconds minimum
    recordingTimer: null,
    recordingStartTime: null,
    
    // Browser capabilities (cached after first check)
    _capabilities: null,
    
    // Callbacks
    onTranscription: null,
    onError: null,
    onRecordingStart: null,
    onRecordingStop: null,
    onPermissionDenied: null,
    
    // ==========================================================================
    // INITIALIZATION
    // ==========================================================================
    
    /**
     * Initialize voice input with callbacks
     * @param {object} callbacks - Callback functions
     * @param {function} callbacks.onTranscription - Called with transcribed text
     * @param {function} callbacks.onError - Called on errors
     * @param {function} callbacks.onRecordingStart - Called when recording starts
     * @param {function} callbacks.onRecordingStop - Called when recording stops
     * @param {function} callbacks.onPermissionDenied - Called when mic permission denied
     */
    init(callbacks = {}) {
        // Set callbacks with defaults
        this.onTranscription = callbacks.onTranscription || ((text) => {
            console.log('[Voice] Transcription:', text);
        });
        this.onError = callbacks.onError || ((err) => {
            console.error('[Voice] Error:', err);
        });
        this.onRecordingStart = callbacks.onRecordingStart || (() => {
            console.log('[Voice] Recording started');
        });
        this.onRecordingStop = callbacks.onRecordingStop || (() => {
            console.log('[Voice] Recording stopped');
        });
        this.onPermissionDenied = callbacks.onPermissionDenied || (() => {
            console.warn('[Voice] Permission denied');
        });
        
        // Check and cache browser capabilities
        this._capabilities = this.checkCapabilities();
        
        // Log initialization status
        console.log('[Voice] Initialized with capabilities:', this._capabilities);
        
        return this._capabilities.supported;
    },
    
    // ==========================================================================
    // BROWSER CAPABILITY DETECTION
    // ==========================================================================
    
    /**
     * Comprehensive browser capability check
     * Returns detailed information about what's supported
     * @returns {object} Capability report
     */
    checkCapabilities() {
        const capabilities = {
            supported: false,
            secureContext: false,
            mediaDevices: false,
            getUserMedia: false,
            mediaRecorder: false,
            audioContext: false,
            supportedMimeTypes: [],
            browser: this.detectBrowser(),
            issues: []
        };
        
        // Check 1: Secure Context (required for getUserMedia)
        // localhost is considered secure, but let's verify
        capabilities.secureContext = this.isSecureContext();
        if (!capabilities.secureContext) {
            capabilities.issues.push('Not in a secure context (HTTPS or localhost required)');
            console.warn('[Voice] ❌ Not in secure context. URL:', window.location.href);
        } else {
            console.log('[Voice] ✓ Secure context verified');
        }
        
        // Check 2: navigator.mediaDevices exists
        capabilities.mediaDevices = !!(navigator.mediaDevices);
        if (!capabilities.mediaDevices) {
            capabilities.issues.push('navigator.mediaDevices not available');
            console.warn('[Voice] ❌ navigator.mediaDevices is undefined');
            
            // Try to diagnose why
            if (typeof navigator === 'undefined') {
                console.error('[Voice] navigator object is undefined');
            } else {
                console.log('[Voice] navigator exists, but mediaDevices is:', navigator.mediaDevices);
            }
        } else {
            console.log('[Voice] ✓ navigator.mediaDevices available');
        }
        
        // Check 3: getUserMedia function exists
        if (capabilities.mediaDevices) {
            capabilities.getUserMedia = !!(navigator.mediaDevices.getUserMedia);
            if (!capabilities.getUserMedia) {
                capabilities.issues.push('getUserMedia not available');
                console.warn('[Voice] ❌ getUserMedia not found on mediaDevices');
            } else {
                console.log('[Voice] ✓ getUserMedia available');
            }
        }
        
        // Check 4: MediaRecorder API exists
        capabilities.mediaRecorder = !!(window.MediaRecorder);
        if (!capabilities.mediaRecorder) {
            capabilities.issues.push('MediaRecorder API not available');
            console.warn('[Voice] ❌ MediaRecorder not available');
        } else {
            console.log('[Voice] ✓ MediaRecorder available');
            
            // Check supported MIME types
            capabilities.supportedMimeTypes = this.getSupportedMimeTypes();
            console.log('[Voice] Supported MIME types:', capabilities.supportedMimeTypes);
        }
        
        // Check 5: AudioContext (optional but useful)
        capabilities.audioContext = !!(window.AudioContext || window.webkitAudioContext);
        if (capabilities.audioContext) {
            console.log('[Voice] ✓ AudioContext available');
        }
        
        // Final determination
        capabilities.supported = (
            capabilities.secureContext &&
            capabilities.mediaDevices &&
            capabilities.getUserMedia &&
            capabilities.mediaRecorder
        );
        
        if (capabilities.supported) {
            console.log('[Voice] ✅ All capabilities supported - voice input ready');
        } else {
            console.error('[Voice] ❌ Voice input NOT supported. Issues:', capabilities.issues);
        }
        
        return capabilities;
    },
    
    /**
     * Check if we're in a secure context
     * @returns {boolean} Is secure context
     */
    isSecureContext() {
        // Modern browsers have window.isSecureContext
        if (typeof window.isSecureContext === 'boolean') {
            return window.isSecureContext;
        }
        
        // Fallback check for older browsers
        const protocol = window.location.protocol;
        const hostname = window.location.hostname;
        
        // HTTPS is always secure
        if (protocol === 'https:') {
            return true;
        }
        
        // localhost and 127.0.0.1 are considered secure even over HTTP
        if (protocol === 'http:') {
            if (hostname === 'localhost' || 
                hostname === '127.0.0.1' || 
                hostname === '[::1]' ||
                hostname.endsWith('.localhost')) {
                return true;
            }
        }
        
        // file:// protocol - depends on browser
        if (protocol === 'file:') {
            console.warn('[Voice] file:// protocol detected - may not support getUserMedia');
            return false;
        }
        
        return false;
    },
    
    /**
     * Detect browser type and version
     * @returns {object} Browser info
     */
    detectBrowser() {
        const ua = navigator.userAgent;
        let browser = { name: 'unknown', version: 'unknown' };
        
        if (ua.includes('Chrome') && !ua.includes('Edg')) {
            const match = ua.match(/Chrome\/(\d+)/);
            browser = { name: 'chrome', version: match ? match[1] : 'unknown' };
        } else if (ua.includes('Safari') && !ua.includes('Chrome')) {
            const match = ua.match(/Version\/(\d+)/);
            browser = { name: 'safari', version: match ? match[1] : 'unknown' };
        } else if (ua.includes('Firefox')) {
            const match = ua.match(/Firefox\/(\d+)/);
            browser = { name: 'firefox', version: match ? match[1] : 'unknown' };
        } else if (ua.includes('Edg')) {
            const match = ua.match(/Edg\/(\d+)/);
            browser = { name: 'edge', version: match ? match[1] : 'unknown' };
        }
        
        console.log('[Voice] Detected browser:', browser.name, browser.version);
        return browser;
    },
    
    /**
     * Get all supported MIME types for MediaRecorder
     * @returns {string[]} Array of supported MIME types
     */
    getSupportedMimeTypes() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/ogg',
            'audio/mp4',
            'audio/mp4;codecs=mp4a.40.2',
            'audio/mpeg',
            'audio/wav',
            'audio/aac'
        ];
        
        return types.filter(type => {
            try {
                return MediaRecorder.isTypeSupported(type);
            } catch (e) {
                return false;
            }
        });
    },
    
    // ==========================================================================
    // PUBLIC API - COMPATIBILITY CHECK
    // ==========================================================================
    
    /**
     * Simple check if voice input is supported
     * @returns {boolean} Support status
     */
    isSupported() {
        // Use cached capabilities if available
        if (this._capabilities) {
            return this._capabilities.supported;
        }
        
        // Quick check without full diagnostics
        return !!(
            this.isSecureContext() &&
            navigator.mediaDevices &&
            navigator.mediaDevices.getUserMedia &&
            window.MediaRecorder
        );
    },
    
    /**
     * Get detailed capability report
     * @returns {object} Capability details
     */
    getCapabilities() {
        if (!this._capabilities) {
            this._capabilities = this.checkCapabilities();
        }
        return this._capabilities;
    },
    
    /**
     * Get human-readable error message for unsupported browsers
     * @returns {string} Error message
     */
    getUnsupportedMessage() {
        const caps = this.getCapabilities();
        
        if (caps.supported) {
            return null;
        }
        
        // Provide specific guidance based on what's missing
        if (!caps.secureContext) {
            return 'Voice input requires a secure connection. Please access via HTTPS or localhost.';
        }
        
        if (!caps.mediaDevices || !caps.getUserMedia) {
            if (caps.browser.name === 'safari') {
                return 'Voice input requires Safari 11 or later. Please update your browser.';
            }
            return 'Your browser does not support voice input. Please use Chrome, Firefox, or Edge.';
        }
        
        if (!caps.mediaRecorder) {
            return 'Your browser does not support audio recording. Please use a modern browser.';
        }
        
        return 'Voice input is not supported in this browser configuration.';
    },
    
    // ==========================================================================
    // PUBLIC API - PERMISSIONS
    // ==========================================================================
    
    /**
     * Request microphone permission
     * @returns {Promise<boolean>} Permission granted
     */
    async requestPermission() {
        // Check capabilities first
        if (!this.isSupported()) {
            const message = this.getUnsupportedMessage();
            console.error('[Voice] Cannot request permission:', message);
            this.onError(message);
            return false;
        }
        
        try {
            console.log('[Voice] Requesting microphone permission...');
            
            // Request permission with audio constraints
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: { ideal: 16000 },
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            // Permission granted - stop the test stream immediately
            stream.getTracks().forEach(track => {
                track.stop();
                console.log('[Voice] Test track stopped:', track.label);
            });
            
            console.log('[Voice] ✅ Microphone permission granted');
            return true;
            
        } catch (error) {
            console.error('[Voice] Permission error:', error.name, error.message);
            
            // Handle specific error types
            let errorMessage;
            
            switch (error.name) {
                case 'NotAllowedError':
                case 'PermissionDeniedError':
                    errorMessage = 'Microphone access denied. Please allow microphone permission in your browser settings.';
                    this.onPermissionDenied();
                    break;
                    
                case 'NotFoundError':
                case 'DevicesNotFoundError':
                    errorMessage = 'No microphone found. Please connect a microphone and try again.';
                    break;
                    
                case 'NotReadableError':
                case 'TrackStartError':
                    errorMessage = 'Microphone is in use by another application. Please close other apps using the microphone.';
                    break;
                    
                case 'OverconstrainedError':
                    // Try again with simpler constraints
                    console.log('[Voice] Retrying with basic audio constraints...');
                    return await this.requestPermissionBasic();
                    
                case 'AbortError':
                    errorMessage = 'Microphone access was aborted. Please try again.';
                    break;
                    
                case 'SecurityError':
                    errorMessage = 'Microphone access blocked due to security policy. Please use HTTPS.';
                    break;
                    
                default:
                    errorMessage = `Microphone error: ${error.message || error.name}`;
            }
            
            this.onError(errorMessage);
            return false;
        }
    },
    
    /**
     * Fallback permission request with basic constraints
     * @returns {Promise<boolean>} Permission granted
     */
    async requestPermissionBasic() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            console.log('[Voice] ✅ Microphone permission granted (basic)');
            return true;
        } catch (error) {
            console.error('[Voice] Basic permission also failed:', error);
            this.onError('Microphone access denied. Please check browser permissions.');
            this.onPermissionDenied();
            return false;
        }
    },
    
    /**
     * Check current permission status without prompting
     * @returns {Promise<string>} Permission state: 'granted', 'denied', 'prompt', or 'unknown'
     */
    async checkPermissionStatus() {
        try {
            // Use Permissions API if available
            if (navigator.permissions && navigator.permissions.query) {
                const result = await navigator.permissions.query({ name: 'microphone' });
                console.log('[Voice] Permission status:', result.state);
                return result.state;
            }
        } catch (e) {
            // Permissions API not supported or microphone not queryable
            console.log('[Voice] Permissions API not available for microphone');
        }
        
        return 'unknown';
    },
    
    // ==========================================================================
    // PUBLIC API - RECORDING
    // ==========================================================================
    
    /**
     * Start recording audio
     * @returns {Promise<boolean>} Started successfully
     */
    async startRecording() {
        // Prevent double recording
        if (this.isRecording) {
            console.warn('[Voice] Already recording');
            return false;
        }
        
        // Check support
        if (!this.isSupported()) {
            const message = this.getUnsupportedMessage();
            this.onError(message);
            return false;
        }
        
        try {
            console.log('[Voice] Starting recording...');
            
            // Get audio stream with optimal settings for speech
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: { ideal: 16000, min: 8000, max: 48000 },
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            console.log('[Voice] Audio stream acquired');
            
            // Get best MIME type for this browser
            const mimeType = this.getBestMimeType();
            console.log('[Voice] Using MIME type:', mimeType || 'default');
            
            // Create MediaRecorder with options
            const options = {};
            if (mimeType) {
                options.mimeType = mimeType;
            }
            
            // Add bitrate for better quality/size balance
            options.audioBitsPerSecond = 128000;
            
            try {
                this.mediaRecorder = new MediaRecorder(this.stream, options);
            } catch (e) {
                // Fallback without options if it fails
                console.warn('[Voice] MediaRecorder with options failed, using defaults');
                this.mediaRecorder = new MediaRecorder(this.stream);
            }
            
            this.audioChunks = [];
            
            // Handle data available
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    console.log('[Voice] Chunk received:', event.data.size, 'bytes');
                }
            };
            
            // Handle recording stop
            this.mediaRecorder.onstop = () => {
                console.log('[Voice] MediaRecorder stopped, processing...');
                this.processRecording();
            };
            
            // Handle errors
            this.mediaRecorder.onerror = (event) => {
                console.error('[Voice] MediaRecorder error:', event.error);
                this.onError('Recording error: ' + (event.error?.message || 'Unknown error'));
                this.cleanup();
            };
            
            // Start recording - collect data every 500ms for responsive UI
            this.mediaRecorder.start(500);
            this.isRecording = true;
            this.recordingStartTime = Date.now();
            
            // Set maximum duration timer
            this.recordingTimer = setTimeout(() => {
                if (this.isRecording) {
                    console.log('[Voice] Maximum duration reached, stopping...');
                    this.stopRecording();
                }
            }, this.maxDuration);
            
            // Notify callback
            this.onRecordingStart();
            console.log('[Voice] ✅ Recording started');
            
            return true;
            
        } catch (error) {
            console.error('[Voice] Start recording failed:', error);
            
            // Handle specific errors
            if (error.name === 'NotAllowedError') {
                this.onError('Microphone permission denied. Please allow access.');
                this.onPermissionDenied();
            } else if (error.name === 'NotFoundError') {
                this.onError('No microphone found. Please connect a microphone.');
            } else {
                this.onError('Failed to start recording: ' + (error.message || error.name));
            }
            
            this.cleanup();
            return false;
        }
    },
    
    /**
     * Stop recording
     */
    stopRecording() {
        if (!this.isRecording) {
            console.log('[Voice] Not recording, nothing to stop');
            return;
        }
        
        console.log('[Voice] Stopping recording...');
        
        // Clear max duration timer
        if (this.recordingTimer) {
            clearTimeout(this.recordingTimer);
            this.recordingTimer = null;
        }
        
        // Check recording duration
        const duration = Date.now() - (this.recordingStartTime || Date.now());
        if (duration < this.minDuration) {
            console.warn('[Voice] Recording too short:', duration, 'ms');
        }
        
        // Stop the MediaRecorder (triggers onstop -> processRecording)
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            try {
                this.mediaRecorder.stop();
            } catch (e) {
                console.warn('[Voice] Error stopping MediaRecorder:', e);
            }
        }
        
        // Stop all audio tracks
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                track.stop();
                console.log('[Voice] Track stopped:', track.label);
            });
        }
        
        this.isRecording = false;
        this.onRecordingStop();
        console.log('[Voice] Recording stopped after', duration, 'ms');
    },
    
    /**
     * Toggle recording state
     * @returns {Promise<boolean>} New recording state (true = now recording)
     */
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
            return false;
        } else {
            return await this.startRecording();
        }
    },
    
    /**
     * Cancel recording without processing
     */
    cancelRecording() {
        console.log('[Voice] Cancelling recording...');
        
        // Clear timer
        if (this.recordingTimer) {
            clearTimeout(this.recordingTimer);
            this.recordingTimer = null;
        }
        
        // Remove the onstop handler to prevent processing
        if (this.mediaRecorder) {
            this.mediaRecorder.onstop = null;
            
            if (this.mediaRecorder.state !== 'inactive') {
                try {
                    this.mediaRecorder.stop();
                } catch (e) {
                    // Ignore errors during cancel
                }
            }
        }
        
        this.cleanup();
        this.onRecordingStop();
        console.log('[Voice] Recording cancelled');
    },
    
    // ==========================================================================
    // AUDIO PROCESSING
    // ==========================================================================
    
    /**
     * Process recorded audio and send for transcription
     */
    async processRecording() {
        // Check if we have audio data
        if (!this.audioChunks || this.audioChunks.length === 0) {
            console.warn('[Voice] No audio chunks to process');
            this.onError('No audio recorded. Please try again.');
            this.cleanup();
            return;
        }
        
        try {
            // Create blob from chunks
            const mimeType = this.mediaRecorder?.mimeType || 'audio/webm';
            const audioBlob = new Blob(this.audioChunks, { type: mimeType });
            
            console.log('[Voice] Processing audio blob:', {
                size: audioBlob.size,
                type: mimeType,
                chunks: this.audioChunks.length
            });
            
            // Validate blob size
            if (audioBlob.size < 1000) {
                console.warn('[Voice] Audio blob too small:', audioBlob.size);
                this.onError('Recording too short. Please speak longer.');
                this.cleanup();
                return;
            }
            
            // Get file extension
            const extension = this.getExtensionFromMimeType(mimeType);
            const filename = `recording_${Date.now()}.${extension}`;
            
            // Create File object for upload
            const audioFile = new File([audioBlob], filename, { type: mimeType });
            
            console.log('[Voice] Sending for transcription:', filename);
            
            // Get current language for hint
            const languageHint = typeof I18n !== 'undefined' ? I18n.getLanguage() : 'en';
            
            // Send to backend
            const result = await this.sendForTranscription(audioFile, languageHint);
            
            if (result.success && result.text) {
                console.log('[Voice] ✅ Transcription successful:', result.text);
                this.onTranscription(result.text, result);
            } else {
                const errorMsg = result.error || 'Transcription failed. Please try again.';
                console.error('[Voice] Transcription failed:', errorMsg);
                this.onError(errorMsg);
            }
            
        } catch (error) {
            console.error('[Voice] Processing error:', error);
            this.onError('Failed to process recording: ' + (error.message || 'Unknown error'));
        } finally {
            this.cleanup();
        }
    },
    
    /**
     * Send audio file to backend for transcription
     * @param {File} audioFile - Audio file to transcribe
     * @param {string} languageHint - Language hint (en, ha, yo, ig)
     * @returns {Promise<object>} Transcription result
     */
    async sendForTranscription(audioFile, languageHint = 'en') {
        // Check if FarmEyesAPI is available
        if (typeof FarmEyesAPI !== 'undefined' && FarmEyesAPI.transcribeAudio) {
            return await FarmEyesAPI.transcribeAudio(audioFile, languageHint);
        }
        
        // Fallback: Direct API call
        console.log('[Voice] Using direct API call for transcription');
        
        try {
            const formData = new FormData();
            formData.append('audio', audioFile);
            formData.append('language_hint', languageHint);
            
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            
            return await response.json();
            
        } catch (error) {
            console.error('[Voice] Transcription API error:', error);
            return {
                success: false,
                error: error.message || 'Failed to connect to transcription service'
            };
        }
    },
    
    // ==========================================================================
    // UTILITY METHODS
    // ==========================================================================
    
    /**
     * Get the best MIME type for the current browser
     * @returns {string|null} Best supported MIME type
     */
    getBestMimeType() {
        // Preferred order: webm with opus is best for speech
        const preferredTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/ogg',
            'audio/mp4',
            'audio/wav'
        ];
        
        for (const type of preferredTypes) {
            try {
                if (MediaRecorder.isTypeSupported(type)) {
                    return type;
                }
            } catch (e) {
                // Continue to next type
            }
        }
        
        return null;
    },
    
    /**
     * Get file extension from MIME type
     * @param {string} mimeType - MIME type
     * @returns {string} File extension
     */
    getExtensionFromMimeType(mimeType) {
        const mimeToExt = {
            'audio/webm': 'webm',
            'audio/webm;codecs=opus': 'webm',
            'audio/ogg': 'ogg',
            'audio/ogg;codecs=opus': 'ogg',
            'audio/mp4': 'm4a',
            'audio/mp4;codecs=mp4a.40.2': 'm4a',
            'audio/mpeg': 'mp3',
            'audio/wav': 'wav',
            'audio/aac': 'aac'
        };
        
        // Handle MIME types with parameters
        const baseMime = mimeType.split(';')[0];
        return mimeToExt[mimeType] || mimeToExt[baseMime] || 'webm';
    },
    
    /**
     * Cleanup all resources
     */
    cleanup() {
        console.log('[Voice] Cleaning up resources...');
        
        // Clear audio chunks
        this.audioChunks = [];
        
        // Clear MediaRecorder
        if (this.mediaRecorder) {
            this.mediaRecorder.ondataavailable = null;
            this.mediaRecorder.onstop = null;
            this.mediaRecorder.onerror = null;
            this.mediaRecorder = null;
        }
        
        // Stop and clear stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (e) {
                    // Ignore errors during cleanup
                }
            });
            this.stream = null;
        }
        
        // Clear timer
        if (this.recordingTimer) {
            clearTimeout(this.recordingTimer);
            this.recordingTimer = null;
        }
        
        // Reset state
        this.isRecording = false;
        this.recordingStartTime = null;
    },
    
    /**
     * Get current recording state
     * @returns {boolean} Is currently recording
     */
    getIsRecording() {
        return this.isRecording;
    },
    
    /**
     * Get recording duration in milliseconds
     * @returns {number} Duration in ms, or 0 if not recording
     */
    getRecordingDuration() {
        if (!this.isRecording || !this.recordingStartTime) {
            return 0;
        }
        return Date.now() - this.recordingStartTime;
    },
    
    // ==========================================================================
    // DIAGNOSTIC METHODS
    // ==========================================================================
    
    /**
     * Run full diagnostic and log to console
     * Useful for debugging issues
     */
    runDiagnostic() {
        console.group('[Voice] Running Diagnostic');
        
        console.log('=== Browser Info ===');
        console.log('User Agent:', navigator.userAgent);
        console.log('Platform:', navigator.platform);
        
        console.log('\n=== Security Context ===');
        console.log('URL:', window.location.href);
        console.log('Protocol:', window.location.protocol);
        console.log('Hostname:', window.location.hostname);
        console.log('isSecureContext:', window.isSecureContext);
        console.log('Our check:', this.isSecureContext());
        
        console.log('\n=== API Availability ===');
        console.log('navigator:', typeof navigator);
        console.log('navigator.mediaDevices:', typeof navigator.mediaDevices);
        console.log('getUserMedia:', typeof navigator.mediaDevices?.getUserMedia);
        console.log('MediaRecorder:', typeof window.MediaRecorder);
        console.log('AudioContext:', typeof (window.AudioContext || window.webkitAudioContext));
        
        console.log('\n=== MediaRecorder MIME Types ===');
        if (window.MediaRecorder) {
            const types = this.getSupportedMimeTypes();
            types.forEach(type => console.log('  ✓', type));
            if (types.length === 0) {
                console.log('  ❌ No supported MIME types');
            }
        }
        
        console.log('\n=== Full Capabilities ===');
        const caps = this.checkCapabilities();
        console.log('Supported:', caps.supported);
        console.log('Issues:', caps.issues);
        
        console.groupEnd();
        
        return caps;
    }
};

// ==========================================================================
// EXPORT
// ==========================================================================

// Export for use in other modules
window.VoiceInput = VoiceInput;

// Auto-run diagnostic in development
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    // Delay diagnostic to ensure page is fully loaded
    setTimeout(() => {
        console.log('[Voice] Development mode - running diagnostic...');
        VoiceInput.runDiagnostic();
    }, 1000);
}
