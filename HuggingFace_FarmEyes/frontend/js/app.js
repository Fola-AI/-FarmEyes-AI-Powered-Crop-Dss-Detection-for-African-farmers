/**
 * FarmEyes Main Application
 * =========================
 * Main controller - navigation and language handling.
 */

const App = {
    currentPage: 'language',
    selectedLanguage: null,
    isInitialized: false,
    elements: {},
    
    /**
     * Initialize
     */
    async init() {
        console.log('[App] Initializing FarmEyes...');
        
        this.cacheElements();
        
        await FarmEyesAPI.init();
        await I18n.init('en');
        
        this.bindEvents();
        
        Diagnosis.init();
        Chat.init();
        
        // Always start with language page
        this.navigateToPage('language');
        
        this.isInitialized = true;
        console.log('[App] Ready!');
    },
    
    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            pageLanguage: document.getElementById('page-language'),
            pageDiagnosis: document.getElementById('page-diagnosis'),
            pageChat: document.getElementById('page-chat'),
            
            languageButtons: document.querySelectorAll('.language-btn'),
            btnContinue: document.getElementById('btn-continue-language'),
            
            btnLanguageToggle: document.getElementById('btn-language-toggle'),
            currentLangDisplay: document.getElementById('current-lang-display'),
            languageMenu: document.getElementById('language-menu'),
            languageDropdownItems: document.querySelectorAll('#language-menu .dropdown-item'),
            
            loadingOverlay: document.getElementById('loading-overlay'),
            loadingText: document.getElementById('loading-text'),
            toastContainer: document.getElementById('toast-container')
        };
    },
    
    /**
     * Bind events
     */
    bindEvents() {
        // Language selector buttons
        this.elements.languageButtons.forEach(btn => {
            btn.addEventListener('click', () => this.selectLanguage(btn.dataset.lang));
        });
        
        // Continue button
        this.elements.btnContinue?.addEventListener('click', () => this.onLanguageContinue());
        
        // Language dropdown
        this.elements.btnLanguageToggle?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleLanguageMenu();
        });
        
        this.elements.languageDropdownItems.forEach(item => {
            item.addEventListener('click', () => this.changeLanguage(item.dataset.lang));
        });
        
        document.addEventListener('click', () => this.closeLanguageMenu());
        
        // Back button in chat
        document.getElementById('btn-back-diagnosis')?.addEventListener('click', () => {
            this.navigateToDiagnosis();
        });
        
        window.addEventListener('languageChanged', (e) => {
            this.onLanguageChanged(e.detail.language);
        });
    },
    
    /**
     * Select language on language page
     */
    selectLanguage(lang) {
        this.selectedLanguage = lang;
        
        this.elements.languageButtons.forEach(btn => {
            btn.classList.toggle('selected', btn.dataset.lang === lang);
        });
        
        if (this.elements.btnContinue) {
            this.elements.btnContinue.disabled = false;
        }
        
        console.log('[App] Language selected:', lang);
    },
    
    /**
     * Continue after language selection
     */
    async onLanguageContinue() {
        if (!this.selectedLanguage) return;
        
        this.showLoading('Setting up...');
        
        try {
            await I18n.setLanguage(this.selectedLanguage);
            this.navigateToDiagnosis();
        } catch (error) {
            console.error('[App] Language setup failed:', error);
            this.showToast('Failed to set language', 'error');
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Change language from dropdown
     */
    async changeLanguage(lang) {
        if (lang === I18n.getLanguage()) {
            this.closeLanguageMenu();
            return;
        }
        
        this.closeLanguageMenu();
        this.showLoading('Changing language...');
        
        try {
            await I18n.setLanguage(lang);
        } catch (error) {
            console.error('[App] Language change failed:', error);
            this.showToast('Failed to change language', 'error');
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Handle language change event
     */
    onLanguageChanged(language) {
        // Update header display
        if (this.elements.currentLangDisplay) {
            this.elements.currentLangDisplay.textContent = language.toUpperCase();
        }
        
        // Update chat display
        const chatLangDisplay = document.getElementById('chat-lang-display');
        if (chatLangDisplay) {
            chatLangDisplay.textContent = language.toUpperCase();
        }
        
        // Update dropdown active state
        this.elements.languageDropdownItems.forEach(item => {
            item.classList.toggle('active', item.dataset.lang === language);
        });
        
        console.log('[App] Language updated to:', language);
    },
    
    toggleLanguageMenu() {
        this.elements.languageMenu?.classList.toggle('hidden');
    },
    
    closeLanguageMenu() {
        this.elements.languageMenu?.classList.add('hidden');
    },
    
    /**
     * Navigate to page
     */
    navigateToPage(pageName) {
        const pages = ['language', 'diagnosis', 'chat'];
        if (!pages.includes(pageName)) return;
        
        // Hide all
        this.elements.pageLanguage?.classList.remove('active');
        this.elements.pageDiagnosis?.classList.remove('active');
        this.elements.pageChat?.classList.remove('active');
        
        // Show target
        const target = document.getElementById(`page-${pageName}`);
        target?.classList.add('active');
        
        // Lifecycle
        if (this.currentPage === 'chat' && pageName !== 'chat') {
            Chat.onPageLeave?.();
        }
        if (pageName === 'chat') {
            Chat.onPageEnter?.();
        }
        
        this.currentPage = pageName;
        console.log('[App] Page:', pageName);
    },
    
    navigateToDiagnosis() {
        this.navigateToPage('diagnosis');
    },
    
    navigateToChat() {
        if (!Diagnosis.hasDiagnosis()) {
            this.showToast('Please analyze an image first', 'warning');
            return;
        }
        this.navigateToPage('chat');
    },
    
    /**
     * Loading overlay
     */
    showLoading(message = 'Loading...') {
        if (this.elements.loadingText) {
            this.elements.loadingText.textContent = message;
        }
        this.elements.loadingOverlay?.classList.remove('hidden');
    },
    
    hideLoading() {
        this.elements.loadingOverlay?.classList.add('hidden');
    },
    
    /**
     * Toast notifications
     */
    showToast(message, type = 'info', duration = 4000) {
        const container = this.elements.toastContainer;
        if (!container) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-message">${message}</span>
            <span class="toast-close" onclick="this.parentElement.remove()">✕</span>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },
    
    getCurrentPage() {
        return this.currentPage;
    },
    
    isReady() {
        return this.isInitialized;
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    App.init().catch(error => {
        console.error('[App] Init failed:', error);
    });
});

window.App = App;
