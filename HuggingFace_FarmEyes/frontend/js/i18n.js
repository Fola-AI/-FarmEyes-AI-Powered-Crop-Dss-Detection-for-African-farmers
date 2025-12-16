/**
 * FarmEyes Internationalization (i18n)
 * =====================================
 * Static translations for UI elements.
 * Always works - no API dependency.
 */

const I18n = {
    currentLanguage: 'en',
    
    // Static translations - embedded for reliability
    translations: {
        en: {
            // Buttons
            "buttons.continue": "Continue",
            "buttons.analyze": "Analyze Crop",
            "buttons.new_scan": "+ New Scan",
            "buttons.back": "Back",
            "buttons.chat": "Chat with Assistant",
            "buttons.stop": "Stop",
            
            // Diagnosis page
            "diagnosis.upload_title": "Upload Crop Image",
            "diagnosis.upload_desc": "Take a clear photo of the affected leaf or plant",
            "diagnosis.click_or_drag": "Click or drag image here",
            "diagnosis.analyzing": "Analyzing your crop...",
            
            // Results
            "results.title": "Diagnosis Results",
            "results.confidence": "Confidence:",
            "results.transmission": "How It Spreads",
            "results.yield_impact": "Yield Impact",
            "results.recovery": "Recovery Chance",
            
            // Tabs
            "tabs.symptoms": "Symptoms",
            "tabs.treatment": "Treatment",
            "tabs.prevention": "Prevention",
            
            // Treatment
            "treatment.immediate": "Immediate Actions",
            "treatment.chemical": "Chemical Treatment",
            "treatment.cost": "Estimated Cost:",
            
            // Chat
            "chat.discussing": "Discussing:",
            "chat.welcome": "Ask me anything about your diagnosis, treatments, or prevention tips.",
            "chat.placeholder": "Ask about your diagnosis...",
            "chat.disclaimer": "FarmEyes provides guidance only. Consult experts for serious cases.",
            
            // Voice
            "voice.listening": "Listening...",
            
            // Severity
            "severity_levels.very_high": "Very High",
            "severity_levels.high": "High",
            "severity_levels.medium": "Medium",
            "severity_levels.low": "Low",
            
            // Crops
            "crops.cassava": "Cassava",
            "crops.cocoa": "Cocoa",
            "crops.tomato": "Tomato"
        },
        
        ha: {
            // Buttons
            "buttons.continue": "Ci gaba",
            "buttons.analyze": "Bincika Amfanin Gona",
            "buttons.new_scan": "+ Sabon Duba",
            "buttons.back": "Koma",
            "buttons.chat": "Yi magana da Mataimaki",
            "buttons.stop": "Daina",
            
            // Diagnosis page
            "diagnosis.upload_title": "Ɗora Hoton Amfanin Gona",
            "diagnosis.upload_desc": "Ɗauki hoto mai kyau na ganyen da ya kamu",
            "diagnosis.click_or_drag": "Danna ko ja hoto nan",
            "diagnosis.analyzing": "Ana bincika amfanin gonar ku...",
            
            // Results
            "results.title": "Sakamakon Bincike",
            "results.confidence": "Tabbaci:",
            "results.transmission": "Yadda Yake Yaɗuwa",
            "results.yield_impact": "Tasirin Amfanin Gona",
            "results.recovery": "Damar Murmurewa",
            
            // Tabs
            "tabs.symptoms": "Alamomi",
            "tabs.treatment": "Magani",
            "tabs.prevention": "Rigakafi",
            
            // Treatment
            "treatment.immediate": "Matakai na Gaggawa",
            "treatment.chemical": "Maganin Sinadari",
            "treatment.cost": "Ƙiyasin Farashi:",
            
            // Chat
            "chat.discussing": "Muna tattaunawa:",
            "chat.welcome": "Tambaye ni komai game da binciken ku.",
            "chat.placeholder": "Tambaya game da binciken ku...",
            "chat.disclaimer": "FarmEyes yana ba da jagora kawai.",
            
            // Voice
            "voice.listening": "Ana saurara...",
            
            // Severity
            "severity_levels.very_high": "Mai Tsanani Sosai",
            "severity_levels.high": "Mai Tsanani",
            "severity_levels.medium": "Matsakaici",
            "severity_levels.low": "Ƙasa",
            
            // Crops
            "crops.cassava": "Rogo",
            "crops.cocoa": "Koko",
            "crops.tomato": "Tumatir"
        },
        
        yo: {
            // Buttons
            "buttons.continue": "Tẹ̀síwájú",
            "buttons.analyze": "Ṣe Àyẹ̀wò Ohun Ọ̀gbìn",
            "buttons.new_scan": "+ Àyẹ̀wò Tuntun",
            "buttons.back": "Padà",
            "buttons.chat": "Bá Olùrànlọ́wọ́ sọ̀rọ̀",
            "buttons.stop": "Dúró",
            
            // Diagnosis page
            "diagnosis.upload_title": "Gbé Àwòrán Ohun Ọ̀gbìn Sókè",
            "diagnosis.upload_desc": "Ya àwòrán tó ṣe kedere ti ewé tó ní àrùn",
            "diagnosis.click_or_drag": "Tẹ tàbí fà àwòrán síbí",
            "diagnosis.analyzing": "A ń ṣe àyẹ̀wò ohun ọ̀gbìn yín...",
            
            // Results
            "results.title": "Àbájáde Àyẹ̀wò",
            "results.confidence": "Ìgbẹ́kẹ̀lé:",
            "results.transmission": "Bí Ó Ṣe Ń Tàn Kálẹ̀",
            "results.yield_impact": "Ipa Lórí Èso",
            "results.recovery": "Àǹfààní Ìmúlàradà",
            
            // Tabs
            "tabs.symptoms": "Àmì Àrùn",
            "tabs.treatment": "Ìtọ́jú",
            "tabs.prevention": "Ìdènà",
            
            // Treatment
            "treatment.immediate": "Ìgbésẹ̀ Lẹ́sẹ̀kẹsẹ̀",
            "treatment.chemical": "Ìtọ́jú Kẹ́míkà",
            "treatment.cost": "Iye Owó Tí A Ṣe Àfojúsùn:",
            
            // Chat
            "chat.discussing": "A ń sọ̀rọ̀ nípa:",
            "chat.welcome": "Bi mi nípa àyẹ̀wò rẹ, ìtọ́jú, tàbí ìdènà.",
            "chat.placeholder": "Béèrè nípa àyẹ̀wò rẹ...",
            "chat.disclaimer": "FarmEyes pèsè ìtọ́sọ́nà nìkan.",
            
            // Voice
            "voice.listening": "A ń gbọ́...",
            
            // Severity
            "severity_levels.very_high": "Ga Jù",
            "severity_levels.high": "Ga",
            "severity_levels.medium": "Àárín",
            "severity_levels.low": "Kéré",
            
            // Crops
            "crops.cassava": "Ẹ̀gẹ́",
            "crops.cocoa": "Koko",
            "crops.tomato": "Tòmátì"
        },
        
        ig: {
            // Buttons
            "buttons.continue": "Gaa n'ihu",
            "buttons.analyze": "Nyochaa Ihe Ọkụkụ",
            "buttons.new_scan": "+ Nyocha Ọhụụ",
            "buttons.back": "Laghachi",
            "buttons.chat": "Soro Onye enyemaka",
            "buttons.stop": "Kwụsị",
            
            // Diagnosis page
            "diagnosis.upload_title": "Bulite Foto Ihe Ọkụkụ",
            "diagnosis.upload_desc": "See foto doro anya nke akwụkwọ nke nwere nsogbu",
            "diagnosis.click_or_drag": "Pịa ma ọ bụ dọrọ foto ebe a",
            "diagnosis.analyzing": "Anyị na-enyocha ihe ọkụkụ gị...",
            
            // Results
            "results.title": "Nsonaazụ Nyocha",
            "results.confidence": "Ntụkwasị Obi:",
            "results.transmission": "Otu Ọ Si Agbasa",
            "results.yield_impact": "Mmetụta Ọnụ Ego",
            "results.recovery": "Ohere Ịlaghachi",
            
            // Tabs
            "tabs.symptoms": "Ihe Ngosi",
            "tabs.treatment": "Ọgwụgwọ",
            "tabs.prevention": "Mgbochi",
            
            // Treatment
            "treatment.immediate": "Ihe Ọsịịsọ",
            "treatment.chemical": "Ọgwụgwọ Kemịkalụ",
            "treatment.cost": "Ego A Tụrụ Anya:",
            
            // Chat
            "chat.discussing": "Anyị na-atụ:",
            "chat.welcome": "Jụọ m ihe ọ bụla gbasara nyocha gị.",
            "chat.placeholder": "Jụọ maka nyocha gị...",
            "chat.disclaimer": "FarmEyes na-enye nduzi nọọ.",
            
            // Voice
            "voice.listening": "Anyị na-ege...",
            
            // Severity
            "severity_levels.very_high": "Dị Elu Nnọọ",
            "severity_levels.high": "Dị Elu",
            "severity_levels.medium": "Etiti",
            "severity_levels.low": "Dị Ala",
            
            // Crops
            "crops.cassava": "Akpụ",
            "crops.cocoa": "Koko",
            "crops.tomato": "Tomato"
        }
    },
    
    /**
     * Initialize
     */
    async init(language = 'en') {
        this.currentLanguage = language;
        this.applyTranslations();
        console.log('[I18n] Initialized:', language);
    },
    
    /**
     * Set language
     */
    async setLanguage(language) {
        if (!['en', 'ha', 'yo', 'ig'].includes(language)) {
            language = 'en';
        }
        
        this.currentLanguage = language;
        localStorage.setItem('farmeyes_language', language);
        
        // Update API
        try {
            await FarmEyesAPI.setLanguage(language);
        } catch (e) {
            console.warn('[I18n] API update failed:', e);
        }
        
        this.applyTranslations();
        
        window.dispatchEvent(new CustomEvent('languageChanged', { detail: { language } }));
        
        console.log('[I18n] Language changed:', language);
    },
    
    /**
     * Get translation
     */
    t(key, params = {}) {
        const langData = this.translations[this.currentLanguage] || this.translations.en;
        let value = langData[key] || this.translations.en[key] || key;
        
        // Interpolate
        if (typeof value === 'string' && Object.keys(params).length > 0) {
            value = value.replace(/\{(\w+)\}/g, (match, k) => params[k] !== undefined ? params[k] : match);
        }
        
        return value;
    },
    
    /**
     * Apply translations to DOM
     */
    applyTranslations() {
        // Text content
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            const translation = this.t(key);
            if (translation !== key) {
                el.textContent = translation;
            }
        });
        
        // Placeholders
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            const translation = this.t(key);
            if (translation !== key) {
                el.placeholder = translation;
            }
        });
        
        document.documentElement.lang = this.currentLanguage;
    },
    
    getLanguage() {
        return this.currentLanguage;
    },
    
    formatCurrency(amount) {
        if (amount == null) return '';
        return new Intl.NumberFormat('en-NG', {
            style: 'currency',
            currency: 'NGN',
            minimumFractionDigits: 0
        }).format(amount);
    },
    
    getSeverity(level) {
        if (!level) return 'Unknown';
        const key = `severity_levels.${level.toLowerCase().replace(/\s+/g, '_')}`;
        const t = this.t(key);
        return t !== key ? t : level;
    },
    
    getCropName(crop) {
        if (!crop) return '';
        const key = `crops.${crop.toLowerCase()}`;
        const t = this.t(key);
        return t !== key ? t : crop;
    }
};

window.I18n = I18n;
