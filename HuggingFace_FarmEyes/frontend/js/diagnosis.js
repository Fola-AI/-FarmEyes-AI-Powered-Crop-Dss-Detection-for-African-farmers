/**
 * FarmEyes Diagnosis Module
 * =========================
 * Handles image upload, disease detection, and results display.
 * Fixed to correctly map API response structure to UI elements.
 */

const Diagnosis = {
    // State
    currentImage: null,
    currentDiagnosis: null,
    isAnalyzing: false,
    
    // DOM Elements (cached)
    elements: {},
    
    /**
     * Initialize diagnosis module
     */
    init() {
        this.cacheElements();
        this.bindEvents();
        console.log('[Diagnosis] Initialized');
    },
    
    /**
     * Cache DOM elements for performance
     */
    cacheElements() {
        this.elements = {
            // Upload
            uploadZone: document.getElementById('upload-zone'),
            fileInput: document.getElementById('file-input'),
            imagePreviewContainer: document.getElementById('image-preview-container'),
            imagePreview: document.getElementById('image-preview'),
            btnRemoveImage: document.getElementById('btn-remove-image'),
            btnAnalyze: document.getElementById('btn-analyze'),
            analyzingLoader: document.getElementById('analyzing-loader'),
            
            // Sections
            uploadSection: document.getElementById('upload-section'),
            resultsSection: document.getElementById('results-section'),
            
            // Results - Disease Card
            btnNewScan: document.getElementById('btn-new-scan'),
            diseaseIcon: document.getElementById('disease-icon'),
            diseaseName: document.getElementById('disease-name'),
            cropType: document.getElementById('crop-type'),
            confidenceBar: document.getElementById('confidence-bar'),
            confidenceValue: document.getElementById('confidence-value'),
            severityBadge: document.getElementById('severity-badge'),
            
            // Tabs
            tabButtons: document.querySelectorAll('.tab-btn'),
            tabSymptoms: document.getElementById('tab-symptoms'),
            tabTreatment: document.getElementById('tab-treatment'),
            tabPrevention: document.getElementById('tab-prevention'),
            
            // Symptoms tab content
            symptomsList: document.getElementById('symptoms-list'),
            transmissionList: document.getElementById('transmission-list'),
            yieldImpactText: document.getElementById('yield-impact-text'),
            recoveryBar: document.getElementById('recovery-bar'),
            recoveryText: document.getElementById('recovery-text'),
            
            // Treatment tab content
            immediateActionsList: document.getElementById('immediate-actions-list'),
            chemicalTreatments: document.getElementById('chemical-treatments'),
            costEstimate: document.getElementById('cost-estimate'),
            
            // Prevention tab content
            preventionList: document.getElementById('prevention-list'),
            
            // Chat button
            btnOpenChat: document.getElementById('btn-open-chat')
        };
    },
    
    /**
     * Bind event handlers
     */
    bindEvents() {
        const { uploadZone, fileInput, btnRemoveImage, btnAnalyze, 
                btnNewScan, tabButtons, btnOpenChat } = this.elements;
        
        // Upload zone click
        uploadZone?.addEventListener('click', () => fileInput?.click());
        
        // File input change
        fileInput?.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop
        uploadZone?.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadZone?.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadZone?.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Remove image
        btnRemoveImage?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.removeImage();
        });
        
        // Analyze button
        btnAnalyze?.addEventListener('click', () => this.analyzeImage());
        
        // New scan button
        btnNewScan?.addEventListener('click', () => this.clearResults());
        
        // Tab switching
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });
        
        // Open chat
        btnOpenChat?.addEventListener('click', () => {
            if (this.currentDiagnosis) {
                App.navigateToChat();
            }
        });
    },
    
    // =========================================================================
    // IMAGE HANDLING
    // =========================================================================
    
    handleFileSelect(event) {
        const file = event.target.files?.[0];
        if (file) {
            this.loadImage(file);
        }
    },
    
    handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
        this.elements.uploadZone?.classList.add('dragover');
    },
    
    handleDragLeave(event) {
        event.preventDefault();
        event.stopPropagation();
        this.elements.uploadZone?.classList.remove('dragover');
    },
    
    handleDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        this.elements.uploadZone?.classList.remove('dragover');
        
        const file = event.dataTransfer?.files?.[0];
        if (file && file.type.startsWith('image/')) {
            this.loadImage(file);
        } else {
            App.showToast('Please drop an image file', 'error');
        }
    },
    
    loadImage(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            App.showToast('Invalid image format. Use JPG, PNG, or WEBP.', 'error');
            return;
        }
        
        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            App.showToast('Image too large. Maximum 10MB.', 'error');
            return;
        }
        
        this.currentImage = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.elements.imagePreview.src = e.target.result;
            this.elements.uploadZone?.classList.add('hidden');
            this.elements.imagePreviewContainer?.classList.remove('hidden');
            this.elements.btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
        
        console.log('[Diagnosis] Image loaded:', file.name);
    },
    
    removeImage() {
        this.currentImage = null;
        this.elements.imagePreview.src = '';
        this.elements.uploadZone?.classList.remove('hidden');
        this.elements.imagePreviewContainer?.classList.add('hidden');
        this.elements.btnAnalyze.disabled = true;
        this.elements.fileInput.value = '';
        
        console.log('[Diagnosis] Image removed');
    },
    
    // =========================================================================
    // ANALYSIS
    // =========================================================================
    
    async analyzeImage() {
        if (!this.currentImage || this.isAnalyzing) {
            return;
        }
        
        this.isAnalyzing = true;
        this.showAnalyzing(true);
        
        try {
            console.log('[Diagnosis] Starting analysis...');
            
            const result = await FarmEyesAPI.detectDisease(
                this.currentImage, 
                I18n.getLanguage()
            );
            
            console.log('[Diagnosis] API Response:', result);
            
            if (result.success) {
                this.currentDiagnosis = result;
                this.displayResults(result);
                console.log('[Diagnosis] Analysis complete:', result.detection?.disease_name);
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
            
        } catch (error) {
            console.error('[Diagnosis] Analysis failed:', error);
            App.showToast(error.message || 'Analysis failed. Please try again.', 'error');
        } finally {
            this.isAnalyzing = false;
            this.showAnalyzing(false);
        }
    },
    
    showAnalyzing(show) {
        const { btnAnalyze, analyzingLoader } = this.elements;
        
        if (show) {
            btnAnalyze?.classList.add('hidden');
            analyzingLoader?.classList.remove('hidden');
        } else {
            btnAnalyze?.classList.remove('hidden');
            analyzingLoader?.classList.add('hidden');
        }
    },
    
    // =========================================================================
    // RESULTS DISPLAY - FIXED MAPPING
    // =========================================================================
    
    displayResults(result) {
        const { detection, diagnosis } = result;
        
        console.log('[Diagnosis] Displaying results:', { detection, diagnosis });
        
        // Show results section
        this.elements.resultsSection?.classList.remove('hidden');
        
        // Disease header
        this.elements.diseaseIcon.textContent = this.getDiseaseIcon(detection.crop_type);
        
        // Disease name - check multiple possible locations
        const diseaseName = diagnosis?.disease?.name || detection?.disease_name || 'Unknown Disease';
        this.elements.diseaseName.textContent = diseaseName;
        
        // Crop type
        this.elements.cropType.textContent = this.formatCropName(detection?.crop_type);
        
        // Confidence
        const confidencePercent = detection?.confidence_percent || (detection?.confidence * 100) || 0;
        this.elements.confidenceBar.style.width = `${confidencePercent}%`;
        this.elements.confidenceValue.textContent = `${Math.round(confidencePercent)}%`;
        
        // Severity
        const severity = diagnosis?.disease?.severity?.level || detection?.severity_level || 'unknown';
        this.elements.severityBadge.textContent = this.formatSeverity(severity);
        this.elements.severityBadge.className = `severity-badge ${severity.toLowerCase().replace(/\s+/g, '-')}`;
        
        // === SYMPTOMS TAB ===
        // Symptoms from diagnosis.symptoms array
        const symptoms = diagnosis?.symptoms || [];
        this.populateList(this.elements.symptomsList, symptoms);
        
        // Transmission from diagnosis.transmission array
        const transmission = diagnosis?.transmission || [];
        this.populateList(this.elements.transmissionList, transmission);
        
        // Yield impact
        const yieldImpact = diagnosis?.yield_impact;
        if (yieldImpact && this.elements.yieldImpactText) {
            const minLoss = yieldImpact.min_percent || 0;
            const maxLoss = yieldImpact.max_percent || 0;
            this.elements.yieldImpactText.textContent = `${minLoss}% - ${maxLoss}% potential yield loss`;
        }
        
        // Recovery/Health projection
        const projection = diagnosis?.current_projection || diagnosis?.health_projection;
        if (projection && this.elements.recoveryBar) {
            const recovery = projection.recovery_chance_percent || projection.recovery_chance || 0;
            this.elements.recoveryBar.style.width = `${recovery}%`;
            if (this.elements.recoveryText) {
                this.elements.recoveryText.textContent = `${recovery}% recovery chance`;
            }
        }
        
        // === TREATMENT TAB ===
        // Immediate actions from diagnosis.treatments.immediate_actions
        const treatments = diagnosis?.treatments || {};
        const immediateActions = treatments.immediate_actions || [];
        this.populateActionsList(this.elements.immediateActionsList, immediateActions);
        
        // Chemical treatments
        const chemicalTreatments = treatments.chemical || [];
        this.populateChemicalTreatments(chemicalTreatments);
        
        // Cost estimate
        const costs = diagnosis?.costs;
        if (costs && this.elements.costEstimate) {
            const minCost = costs.min_ngn || 0;
            const maxCost = costs.max_ngn || 0;
            if (minCost && maxCost) {
                this.elements.costEstimate.textContent = `₦${minCost.toLocaleString()} - ₦${maxCost.toLocaleString()}`;
            } else {
                this.elements.costEstimate.textContent = 'Contact local supplier';
            }
        }
        
        // === PREVENTION TAB ===
        // Prevention tips from diagnosis.prevention array
        const prevention = diagnosis?.prevention || [];
        this.populateList(this.elements.preventionList, prevention);
        
        // Scroll to results
        this.elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
    },
    
    /**
     * Populate a simple list with items
     */
    populateList(listElement, items) {
        if (!listElement) return;
        
        listElement.innerHTML = '';
        
        if (!items || items.length === 0) {
            const li = document.createElement('li');
            li.textContent = 'No information available';
            li.style.fontStyle = 'italic';
            li.style.color = 'var(--text-muted)';
            listElement.appendChild(li);
            return;
        }
        
        items.slice(0, 6).forEach(item => {
            const li = document.createElement('li');
            // Handle both string items and object items
            if (typeof item === 'string') {
                li.textContent = item;
            } else if (typeof item === 'object') {
                li.textContent = item.text || item.description || item.name || JSON.stringify(item);
            }
            listElement.appendChild(li);
        });
    },
    
    /**
     * Populate immediate actions list
     */
    populateActionsList(listElement, actions) {
        if (!listElement) return;
        
        listElement.innerHTML = '';
        
        if (!actions || actions.length === 0) {
            const li = document.createElement('li');
            li.textContent = 'Consult agricultural expert for guidance';
            listElement.appendChild(li);
            return;
        }
        
        actions.slice(0, 5).forEach(action => {
            const li = document.createElement('li');
            if (typeof action === 'string') {
                li.textContent = action;
            } else if (typeof action === 'object') {
                li.textContent = action.action || action.description || action.text || '';
            }
            listElement.appendChild(li);
        });
    },
    
    /**
     * Populate chemical treatments
     */
    populateChemicalTreatments(treatments) {
        const container = this.elements.chemicalTreatments;
        if (!container) return;
        
        container.innerHTML = '';
        
        if (!treatments || treatments.length === 0) {
            const div = document.createElement('div');
            div.className = 'treatment-item';
            div.innerHTML = '<span>Consult local agricultural store</span>';
            container.appendChild(div);
            return;
        }
        
        treatments.slice(0, 4).forEach(treatment => {
            const div = document.createElement('div');
            div.className = 'treatment-item';
            
            const name = treatment.product || treatment.product_name || treatment.name || 'Treatment';
            const dosage = treatment.dosage || treatment.application || '';
            const costMin = treatment.cost_min || treatment.cost_ngn_min || '';
            const costMax = treatment.cost_max || treatment.cost_ngn_max || '';
            
            let costText = '';
            if (costMin && costMax) {
                costText = ` - ₦${costMin.toLocaleString()} to ₦${costMax.toLocaleString()}`;
            }
            
            div.innerHTML = `
                <strong>${name}</strong>
                <span>${dosage}${costText}</span>
            `;
            
            container.appendChild(div);
        });
    },
    
    /**
     * Get disease icon based on crop type
     */
    getDiseaseIcon(cropType) {
        const icons = {
            cassava: '🌿',
            cocoa: '🍫',
            tomato: '🍅'
        };
        return icons[cropType?.toLowerCase()] || '🌱';
    },
    
    /**
     * Format crop name
     */
    formatCropName(cropType) {
        if (!cropType) return 'Unknown';
        return cropType.charAt(0).toUpperCase() + cropType.slice(1).toLowerCase();
    },
    
    /**
     * Format severity level
     */
    formatSeverity(severity) {
        if (!severity) return 'Unknown';
        return severity.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    },
    
    /**
     * Switch between tabs
     */
    switchTab(tabName) {
        // Update button states
        this.elements.tabButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        
        // Update content visibility
        const tabs = ['symptoms', 'treatment', 'prevention'];
        tabs.forEach(tab => {
            const tabElement = this.elements[`tab${tab.charAt(0).toUpperCase() + tab.slice(1)}`];
            if (tabElement) {
                tabElement.classList.toggle('active', tab === tabName);
                tabElement.classList.toggle('hidden', tab !== tabName);
            }
        });
    },
    
    /**
     * Clear results and reset for new scan
     */
    clearResults() {
        this.currentDiagnosis = null;
        this.elements.resultsSection?.classList.add('hidden');
        this.removeImage();
        
        // Clear API diagnosis
        FarmEyesAPI.clearDiagnosis().catch(() => {});
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        console.log('[Diagnosis] Results cleared');
    },
    
    /**
     * Get current diagnosis data
     */
    getDiagnosis() {
        return this.currentDiagnosis;
    },
    
    /**
     * Check if there's a valid diagnosis
     */
    hasDiagnosis() {
        return this.currentDiagnosis !== null;
    }
};

// Export
window.Diagnosis = Diagnosis;
