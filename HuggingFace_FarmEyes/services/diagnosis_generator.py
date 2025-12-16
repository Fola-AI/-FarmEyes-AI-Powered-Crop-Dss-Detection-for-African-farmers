"""
FarmEyes Diagnosis Generator Service
====================================
Generates complete multilingual diagnosis reports by combining:
- Disease detection results (from YOLO model)
- Knowledge base information (symptoms, treatments, costs)
- N-ATLaS translations (Hausa, Yoruba, Igbo)

Produces farmer-friendly reports with actionable treatment recommendations.
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DIAGNOSIS REPORT DATACLASS
# =============================================================================

@dataclass
class DiagnosisReport:
    """
    Complete diagnosis report with all information translated to user's language.
    """
    # Metadata
    report_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    language: str = "en"
    
    # Detection summary
    crop_type: str = ""
    is_healthy: bool = False
    confidence: float = 0.0
    confidence_level: str = ""  # high, medium, low
    
    # Disease information (translated)
    disease_name: str = ""
    disease_name_scientific: str = ""
    disease_category: str = ""
    
    # Severity (translated)
    severity_level: str = ""
    severity_scale: int = 0
    severity_description: str = ""
    
    # Summary message (translated)
    summary_message: str = ""
    
    # Symptoms (translated)
    symptoms: List[str] = field(default_factory=list)
    
    # How it spreads (translated)
    transmission: List[str] = field(default_factory=list)
    
    # Yield impact
    yield_loss_min: int = 0
    yield_loss_max: int = 0
    yield_loss_message: str = ""
    
    # Treatments (translated)
    immediate_actions: List[Dict] = field(default_factory=list)
    chemical_treatments: List[Dict] = field(default_factory=list)
    organic_treatments: List[Dict] = field(default_factory=list)
    traditional_treatments: List[Dict] = field(default_factory=list)
    resistant_varieties: List[Dict] = field(default_factory=list)
    
    # Costs
    treatment_cost_min: int = 0
    treatment_cost_max: int = 0
    cost_message: str = ""
    
    # Prevention (translated)
    prevention_tips: List[str] = field(default_factory=list)
    
    # Health projection (translated)
    health_projection: Dict = field(default_factory=dict)
    current_projection: Dict = field(default_factory=dict)
    
    # Expert contact
    expert_institution: str = ""
    expert_location: str = ""
    expert_services: str = ""
    
    # For healthy plants (translated)
    healthy_message: str = ""
    maintenance_tips: List[str] = field(default_factory=list)
    expected_yield: Dict = field(default_factory=dict)
    
    # Original detection result (for reference)
    raw_detection: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON serialization."""
        return {
            "metadata": {
                "report_id": self.report_id,
                "timestamp": self.timestamp,
                "language": self.language
            },
            "detection": {
                "crop_type": self.crop_type,
                "is_healthy": self.is_healthy,
                "confidence": round(self.confidence, 4),
                "confidence_percent": round(self.confidence * 100, 1),
                "confidence_level": self.confidence_level
            },
            "disease": {
                "name": self.disease_name,
                "scientific_name": self.disease_name_scientific,
                "category": self.disease_category,
                "severity": {
                    "level": self.severity_level,
                    "scale": self.severity_scale,
                    "description": self.severity_description
                }
            },
            "summary": self.summary_message,
            "symptoms": self.symptoms,
            "transmission": self.transmission,
            "yield_impact": {
                "min_percent": self.yield_loss_min,
                "max_percent": self.yield_loss_max,
                "message": self.yield_loss_message
            },
            "treatments": {
                "immediate_actions": self.immediate_actions,
                "chemical": self.chemical_treatments,
                "organic": self.organic_treatments,
                "traditional": self.traditional_treatments,
                "resistant_varieties": self.resistant_varieties
            },
            "costs": {
                "min_ngn": self.treatment_cost_min,
                "max_ngn": self.treatment_cost_max,
                "message": self.cost_message
            },
            "prevention": self.prevention_tips,
            "health_projection": self.health_projection,
            "current_projection": self.current_projection,
            "expert_contact": {
                "institution": self.expert_institution,
                "location": self.expert_location,
                "services": self.expert_services
            },
            "healthy_plant": {
                "message": self.healthy_message,
                "maintenance_tips": self.maintenance_tips,
                "expected_yield": self.expected_yield
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def get_short_summary(self) -> str:
        """Get a brief one-line summary."""
        if self.is_healthy:
            return f"✅ {self.healthy_message}"
        else:
            return f"⚠️ {self.disease_name} - {self.severity_level} severity ({self.confidence:.0%} confidence)"


# =============================================================================
# DIAGNOSIS GENERATOR SERVICE
# =============================================================================

class DiagnosisGenerator:
    """
    Generates complete multilingual diagnosis reports.
    Combines disease detection, knowledge base, and translation services.
    """
    
    def __init__(self, auto_load_models: bool = False):
        """
        Initialize the diagnosis generator.
        
        Args:
            auto_load_models: Whether to load ML models immediately
        """
        self._disease_detector = None
        self._translator = None
        self._report_counter = 0
        
        # Load services
        self._init_services(auto_load_models)
        
        logger.info("DiagnosisGenerator initialized")
    
    def _init_services(self, auto_load_models: bool) -> None:
        """Initialize required services."""
        from services.disease_detector import get_disease_detector
        from services.translator import get_translator
        
        self._disease_detector = get_disease_detector()
        self._translator = get_translator()
        
        if auto_load_models:
            self._disease_detector.ensure_model_loaded()
            self._translator.ensure_model_loaded()
    
    def _generate_report_id(self) -> str:
        """Generate a unique report ID."""
        self._report_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"FE-{timestamp}-{self._report_counter:04d}"
    
    # =========================================================================
    # MAIN GENERATION METHODS
    # =========================================================================
    
    def generate(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        language: str = "en"
    ) -> DiagnosisReport:
        """
        Generate a complete diagnosis report from an image.
        
        Args:
            image: Input crop image (path, PIL Image, or numpy array)
            language: Target language code (en, ha, yo, ig)
            
        Returns:
            Complete DiagnosisReport in specified language
        """
        # Ensure models are loaded
        self._disease_detector.ensure_model_loaded()
        
        # Detect disease
        logger.info("Running disease detection...")
        detection_result = self._disease_detector.detect(image)
        
        # Generate report from detection result
        report = self._build_report(detection_result, language)
        
        return report
    
    def generate_with_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        language: str = "en"
    ) -> Tuple[DiagnosisReport, Image.Image]:
        """
        Generate diagnosis report and return annotated image.
        
        Args:
            image: Input crop image
            language: Target language code
            
        Returns:
            Tuple of (DiagnosisReport, annotated PIL Image)
        """
        # Ensure models are loaded
        self._disease_detector.ensure_model_loaded()
        
        # Detect disease with image
        logger.info("Running disease detection with visualization...")
        detection_result, annotated_image = self._disease_detector.detect_with_image(image)
        
        # Generate report
        report = self._build_report(detection_result, language)
        
        return report, annotated_image
    
    def generate_from_detection(
        self,
        detection_result,
        language: str = "en"
    ) -> DiagnosisReport:
        """
        Generate report from an existing detection result.
        
        Args:
            detection_result: DetectionResult from disease detector
            language: Target language code
            
        Returns:
            Complete DiagnosisReport
        """
        return self._build_report(detection_result, language)
    
    # =========================================================================
    # REPORT BUILDING
    # =========================================================================
    
    def _build_report(
        self,
        detection_result,
        language: str
    ) -> DiagnosisReport:
        """
        Build a complete diagnosis report from detection result.
        
        Args:
            detection_result: DetectionResult from disease detector
            language: Target language code
            
        Returns:
            Complete DiagnosisReport
        """
        report = DiagnosisReport(
            report_id=self._generate_report_id(),
            language=language,
            raw_detection=detection_result.to_dict()
        )
        
        # Basic detection info
        report.crop_type = detection_result.crop_type
        report.is_healthy = detection_result.is_healthy
        report.confidence = detection_result.confidence
        report.confidence_level = detection_result.get_confidence_level()
        
        if detection_result.is_healthy:
            self._build_healthy_report(report, detection_result, language)
        else:
            self._build_disease_report(report, detection_result, language)
        
        return report
    
    def _build_healthy_report(
        self,
        report: DiagnosisReport,
        detection_result,
        language: str
    ) -> None:
        """Build report for healthy plant detection."""
        # Disease name (translated)
        report.disease_name = self._translate(
            detection_result.display_name or "Healthy Plant",
            language
        )
        
        # Summary message
        summary_en = f"Great news! Your {detection_result.crop_type} plant appears to be healthy. Continue with good farming practices to maintain plant health."
        report.summary_message = self._translate(summary_en, language)
        
        # Healthy message
        if detection_result.healthy_message:
            report.healthy_message = self._translate(
                detection_result.healthy_message,
                language
            )
        else:
            report.healthy_message = report.summary_message
        
        # Maintenance tips
        if detection_result.maintenance_tips:
            report.maintenance_tips = self._translate_list(
                detection_result.maintenance_tips[:6],
                language
            )
        else:
            # Default tips
            default_tips = [
                "Continue regular monitoring for early disease detection",
                "Maintain proper watering and fertilization",
                "Keep the field free of weeds",
                "Practice crop rotation",
                "Use disease-free planting materials"
            ]
            report.maintenance_tips = self._translate_list(default_tips, language)
        
        # Expected yield
        report.expected_yield = detection_result.expected_yield
    
    def _build_disease_report(
        self,
        report: DiagnosisReport,
        detection_result,
        language: str
    ) -> None:
        """Build report for disease detection."""
        # Disease information
        report.disease_name = self._translate(
            detection_result.display_name,
            language
        )
        report.disease_name_scientific = detection_result.scientific_name
        report.disease_category = detection_result.category
        
        # Severity
        report.severity_level = self._translate(
            detection_result.severity_level.replace("_", " ").title(),
            language
        )
        report.severity_scale = detection_result.severity_scale
        report.severity_description = self._translate(
            detection_result.severity_description,
            language
        )
        
        # Summary message
        summary_en = self._create_summary_message(detection_result)
        report.summary_message = self._translate(summary_en, language)
        
        # Symptoms
        if detection_result.symptoms:
            report.symptoms = self._translate_list(
                detection_result.symptoms[:6],
                language
            )
        
        # Transmission
        if detection_result.transmission:
            report.transmission = self._translate_list(
                detection_result.transmission[:5],
                language
            )
        
        # Yield impact
        report.yield_loss_min = detection_result.yield_loss_min
        report.yield_loss_max = detection_result.yield_loss_max
        yield_msg_en = f"This disease can cause {detection_result.yield_loss_min}% to {detection_result.yield_loss_max}% yield loss if not treated."
        report.yield_loss_message = self._translate(yield_msg_en, language)
        
        # Treatments
        self._build_treatment_sections(report, detection_result, language)
        
        # Costs
        report.treatment_cost_min = detection_result.treatment_cost_min
        report.treatment_cost_max = detection_result.treatment_cost_max
        cost_msg_en = f"Estimated treatment cost: ₦{detection_result.treatment_cost_min:,} to ₦{detection_result.treatment_cost_max:,} per hectare."
        report.cost_message = self._translate(cost_msg_en, language)
        
        # Prevention
        if detection_result.prevention:
            report.prevention_tips = self._translate_list(
                detection_result.prevention[:6],
                language
            )
        
        # Health projection
        self._build_health_projection(report, detection_result, language)
        
        # Expert contact
        expert = detection_result.expert_contact
        if expert:
            report.expert_institution = expert.get("institution", "")
            report.expert_location = expert.get("location", "")
            report.expert_services = expert.get("services", "")
    
    def _create_summary_message(self, detection_result) -> str:
        """Create English summary message for disease detection."""
        severity = detection_result.severity_level.replace("_", " ")
        confidence_pct = int(detection_result.confidence * 100)
        
        if detection_result.confidence >= 0.85:
            confidence_text = "high confidence"
        elif detection_result.confidence >= 0.60:
            confidence_text = "moderate confidence"
        else:
            confidence_text = "low confidence"
        
        return (
            f"We detected {detection_result.display_name} in your {detection_result.crop_type} "
            f"with {confidence_text} ({confidence_pct}%). "
            f"This is a {severity} severity disease. "
            f"Please follow the treatment recommendations below to protect your crop."
        )
    
    def _build_treatment_sections(
        self,
        report: DiagnosisReport,
        detection_result,
        language: str
    ) -> None:
        """Build all treatment sections of the report."""
        treatments = detection_result.treatments
        
        # Immediate actions (cultural practices)
        cultural = treatments.get("cultural", [])
        for t in cultural[:4]:
            action = {
                "action": self._translate(t.get("method", ""), language),
                "description": self._translate(t.get("description", ""), language),
                "effectiveness": t.get("effectiveness", ""),
                "timing": t.get("timing", "")
            }
            report.immediate_actions.append(action)
        
        # Chemical treatments
        chemical = treatments.get("chemical", [])
        for t in chemical[:3]:
            treatment = {
                "product_name": t.get("product_name", ""),
                "local_brands": t.get("local_brands", []),
                "dosage": t.get("dosage", ""),
                "frequency": t.get("frequency", ""),
                "application_method": self._translate(
                    t.get("application_method", ""),
                    language
                ),
                "cost_min": t.get("cost_ngn_min", 0),
                "cost_max": t.get("cost_ngn_max", 0),
                "effectiveness": t.get("effectiveness", ""),
                "safety_precautions": self._translate_list(
                    t.get("safety_precautions", [])[:3],
                    language
                )
            }
            report.chemical_treatments.append(treatment)
        
        # Biological/organic treatments
        biological = treatments.get("biological", [])
        for t in biological[:2]:
            treatment = {
                "method": self._translate(t.get("method", ""), language),
                "description": self._translate(t.get("description", ""), language),
                "effectiveness": t.get("effectiveness", ""),
                "source": t.get("source", "")
            }
            report.organic_treatments.append(treatment)
        
        # Traditional treatments
        traditional = treatments.get("traditional", [])
        for t in traditional[:3]:
            treatment = {
                "method": self._translate(t.get("method", ""), language),
                "description": self._translate(t.get("description", ""), language),
                "cost": t.get("cost_ngn", 0),
                "effectiveness": t.get("effectiveness", "")
            }
            report.traditional_treatments.append(treatment)
        
        # Resistant varieties
        varieties = treatments.get("resistant_varieties", [])
        for v in varieties[:3]:
            variety = {
                "name": v.get("variety_name", ""),
                "resistance_level": v.get("resistance_level", ""),
                "source": v.get("source", ""),
                "cost": v.get("cost_ngn_per_bundle", 0),
                "notes": self._translate(v.get("notes", ""), language)
            }
            report.resistant_varieties.append(variety)
    
    def _build_health_projection(
        self,
        report: DiagnosisReport,
        detection_result,
        language: str
    ) -> None:
        """Build health projection section."""
        projection = detection_result.health_projection
        
        if not projection:
            return
        
        # Translate all projection stages
        for stage, info in projection.items():
            if isinstance(info, dict):
                report.health_projection[stage] = {
                    "recovery_chance_percent": info.get("recovery_chance_percent", 0),
                    "message": self._translate(info.get("message", ""), language)
                }
        
        # Set current projection based on confidence
        # Higher confidence often correlates with more visible/advanced symptoms
        if detection_result.confidence >= 0.85:
            # Clear symptoms suggest moderate to severe infection
            stage = "moderate_infection"
        elif detection_result.confidence >= 0.60:
            # Some symptoms visible - likely early detection
            stage = "early_detection"
        else:
            # Low confidence - could be very early
            stage = "early_detection"
        
        if stage in report.health_projection:
            report.current_projection = report.health_projection[stage]
    
    # =========================================================================
    # TRANSLATION HELPERS
    # =========================================================================
    
    def _translate(self, text: str, language: str) -> str:
        """Translate text to target language."""
        if not text or language == "en":
            return text
        
        try:
            return self._translator.translate(text, language)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text
    
    def _translate_list(self, texts: List[str], language: str) -> List[str]:
        """Translate a list of texts."""
        if not texts or language == "en":
            return texts
        
        try:
            return self._translator.translate_batch(texts, language)
        except Exception as e:
            logger.warning(f"Batch translation failed: {e}")
            return texts
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def validate_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[bool, str]:
        """
        Validate an image before diagnosis.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (is_valid, message)
        """
        return self._disease_detector.validate_image(image)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for reports."""
        return self._translator.get_supported_languages()
    
    def get_supported_crops(self) -> List[str]:
        """Get list of supported crops."""
        return ["cassava", "cocoa", "tomato"]
    
    def get_service_status(self) -> Dict:
        """Get status of underlying services."""
        return {
            "disease_detector": {
                "model_loaded": self._disease_detector._yolo_model is not None and 
                               self._disease_detector._yolo_model.is_loaded if self._disease_detector._yolo_model else False,
                "knowledge_base_loaded": self._disease_detector._knowledge_base is not None
            },
            "translator": {
                "model_loaded": self._translator.is_model_loaded,
                "cache_stats": self._translator.get_cache_stats()
            }
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_generator_instance: Optional[DiagnosisGenerator] = None


def get_diagnosis_generator() -> DiagnosisGenerator:
    """
    Get the singleton diagnosis generator instance.
    
    Returns:
        DiagnosisGenerator instance
    """
    global _generator_instance
    
    if _generator_instance is None:
        _generator_instance = DiagnosisGenerator(auto_load_models=False)
    
    return _generator_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_diagnosis(
    image: Union[str, Path, Image.Image, np.ndarray],
    language: str = "en"
) -> DiagnosisReport:
    """
    Generate a diagnosis report from an image.
    
    Args:
        image: Input crop image
        language: Target language code (en, ha, yo, ig)
        
    Returns:
        Complete DiagnosisReport
    """
    generator = get_diagnosis_generator()
    return generator.generate(image, language)


def generate_diagnosis_with_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    language: str = "en"
) -> Tuple[DiagnosisReport, Image.Image]:
    """
    Generate diagnosis report with annotated image.
    
    Args:
        image: Input crop image
        language: Target language code
        
    Returns:
        Tuple of (DiagnosisReport, annotated Image)
    """
    generator = get_diagnosis_generator()
    return generator.generate_with_image(image, language)


# =============================================================================
# MAIN - Test the service
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Diagnosis Generator Service Test")
    print("=" * 60)
    
    # Initialize generator
    print("\n1. Initializing Diagnosis Generator...")
    generator = DiagnosisGenerator(auto_load_models=False)
    
    # Check supported languages
    print("\n2. Supported Languages:")
    for code, name in generator.get_supported_languages().items():
        print(f"   {code}: {name}")
    
    # Check supported crops
    print("\n3. Supported Crops:")
    for crop in generator.get_supported_crops():
        print(f"   - {crop}")
    
    # Check service status
    print("\n4. Service Status:")
    status = generator.get_service_status()
    print(f"   Disease Detector - Knowledge Base Loaded: {status['disease_detector']['knowledge_base_loaded']}")
    print(f"   Disease Detector - Model Loaded: {status['disease_detector']['model_loaded']}")
    print(f"   Translator - Model Loaded: {status['translator']['model_loaded']}")
    
    # Test DiagnosisReport structure
    print("\n5. Testing DiagnosisReport Structure...")
    test_report = DiagnosisReport(
        report_id="TEST-001",
        language="en",
        crop_type="cassava",
        is_healthy=False,
        confidence=0.92,
        confidence_level="high",
        disease_name="Cassava Mosaic Disease",
        severity_level="Very High",
        summary_message="Test summary message"
    )
    print(f"   Report ID: {test_report.report_id}")
    print(f"   Short Summary: {test_report.get_short_summary()}")
    
    print("\n6. To generate actual diagnosis (requires models):")
    print("   >>> report = generator.generate('/path/to/image.jpg', 'ha')")
    print("   >>> print(report.summary_message)")
    print("   >>> print(report.to_json())")
    
    print("\n" + "=" * 60)
    print("✅ Diagnosis Generator Service test completed!")
    print("=" * 60)
