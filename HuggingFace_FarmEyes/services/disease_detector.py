"""
FarmEyes Disease Detector Service
=================================
Service layer that combines YOLOv11 disease detection with knowledge base
lookup to provide comprehensive disease information including symptoms,
treatments, costs, and prevention methods.

This service acts as the bridge between:
- YOLOv11 model (disease detection) - 6 classes
- Knowledge base (disease information)
- N-ATLaS model (translation - handled by translator service)

6-Class Model:
    0: Cassava Bacteria Blight
    1: Cassava Mosaic Virus
    2: Cocoa Monilia Disease
    3: Cocoa Phytophthora Disease
    4: Tomato Gray Mold Disease
    5: Tomato Wilt Disease
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
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
# DETECTION RESULT DATACLASS
# =============================================================================

@dataclass
class DetectionResult:
    """
    Complete detection result with disease information from knowledge base.
    """
    # Detection info
    class_index: int
    class_name: str
    disease_key: str
    confidence: float
    crop_type: str
    is_healthy: bool  # Always False in 6-class model (no healthy classes)
    
    # Disease details from knowledge base
    display_name: str = ""
    scientific_name: str = ""
    category: str = ""  # bacterial, viral, fungal, oomycete
    
    # Severity
    severity_level: str = ""
    severity_scale: int = 0
    severity_description: str = ""
    
    # Symptoms
    symptoms: List[str] = field(default_factory=list)
    
    # How it spreads
    transmission: List[str] = field(default_factory=list)
    
    # Yield impact
    yield_loss_min: int = 0
    yield_loss_max: int = 0
    yield_loss_description: str = ""
    
    # Treatments
    treatments: Dict = field(default_factory=dict)
    
    # Costs
    treatment_cost_min: int = 0
    treatment_cost_max: int = 0
    cost_unit: str = "per hectare"
    
    # Prevention
    prevention: List[str] = field(default_factory=list)
    
    # Health projection
    health_projection: Dict = field(default_factory=dict)
    
    # Expert contact
    expert_contact: Dict = field(default_factory=dict)
    
    # For healthy plants (not used in 6-class model)
    maintenance_tips: List[str] = field(default_factory=list)
    expected_yield: Dict = field(default_factory=dict)
    healthy_message: str = ""
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "detection": {
                "class_index": self.class_index,
                "class_name": self.class_name,
                "disease_key": self.disease_key,
                "confidence": round(self.confidence, 4),
                "confidence_percent": round(self.confidence * 100, 1),
                "crop_type": self.crop_type,
                "is_healthy": self.is_healthy
            },
            "disease_info": {
                "display_name": self.display_name,
                "scientific_name": self.scientific_name,
                "category": self.category,
                "severity": {
                    "level": self.severity_level,
                    "scale": self.severity_scale,
                    "description": self.severity_description
                },
                "symptoms": self.symptoms,
                "transmission": self.transmission,
                "yield_loss": {
                    "min_percent": self.yield_loss_min,
                    "max_percent": self.yield_loss_max,
                    "description": self.yield_loss_description
                }
            },
            "treatments": self.treatments,
            "costs": {
                "min_ngn": self.treatment_cost_min,
                "max_ngn": self.treatment_cost_max,
                "unit": self.cost_unit
            },
            "prevention": self.prevention,
            "health_projection": self.health_projection,
            "expert_contact": self.expert_contact,
            "healthy_plant": {
                "maintenance_tips": self.maintenance_tips,
                "expected_yield": self.expected_yield,
                "message": self.healthy_message
            },
            "timestamp": self.timestamp
        }
    
    def get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.confidence >= 0.85:
            return "high"
        elif self.confidence >= 0.60:
            return "medium"
        elif self.confidence >= 0.40:
            return "low"
        else:
            return "very_low"
    
    def get_summary(self) -> str:
        """Get a brief summary of the detection."""
        if self.is_healthy:
            return f"Your {self.crop_type} plant appears healthy ({self.confidence:.0%} confidence)."
        else:
            return f"Detected {self.display_name} in your {self.crop_type} ({self.confidence:.0%} confidence). Severity: {self.severity_level}."


# =============================================================================
# DISEASE DETECTOR SERVICE
# =============================================================================

class DiseaseDetectorService:
    """
    Service for detecting crop diseases and retrieving comprehensive information.
    Combines YOLO model predictions with knowledge base data.
    
    Supports 6 disease classes (no healthy classes):
        - Cassava: Bacterial Blight, Mosaic Virus
        - Cocoa: Monilia Disease, Phytophthora Disease
        - Tomato: Gray Mold, Wilt Disease
    """
    
    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        auto_load_model: bool = False
    ):
        """
        Initialize the disease detector service.
        
        Args:
            knowledge_base_path: Path to knowledge_base.json
            auto_load_model: Whether to load YOLO model immediately
        """
        from config import KNOWLEDGE_BASE_PATH
        
        self.knowledge_base_path = knowledge_base_path or str(KNOWLEDGE_BASE_PATH)
        self._knowledge_base: Optional[Dict] = None
        self._yolo_model = None
        
        # Load knowledge base
        self._load_knowledge_base()
        
        # Optionally load YOLO model
        if auto_load_model:
            self._load_yolo_model()
        
        logger.info("DiseaseDetectorService initialized")
    
    # =========================================================================
    # LOADING METHODS
    # =========================================================================
    
    def _load_knowledge_base(self) -> None:
        """Load the disease knowledge base from JSON file."""
        try:
            kb_path = Path(self.knowledge_base_path)
            
            if not kb_path.exists():
                logger.error(f"Knowledge base not found at {kb_path}")
                raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
            
            with open(kb_path, 'r', encoding='utf-8') as f:
                self._knowledge_base = json.load(f)
            
            # Validate structure
            if "diseases" not in self._knowledge_base:
                raise ValueError("Invalid knowledge base: missing 'diseases' key")
            
            disease_count = len(self._knowledge_base["diseases"])
            logger.info(f"Knowledge base loaded: {disease_count} diseases")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in knowledge base: {e}")
            raise ValueError(f"Could not parse knowledge base: {e}")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            raise
    
    def _load_yolo_model(self) -> None:
        """Load the YOLO model for inference."""
        from models.yolo_model import get_yolo_model
        
        self._yolo_model = get_yolo_model()
        self._yolo_model.load_model()
        logger.info("YOLO model loaded")
    
    def ensure_model_loaded(self) -> None:
        """Ensure YOLO model is loaded before inference."""
        if self._yolo_model is None:
            self._load_yolo_model()
        elif not self._yolo_model.is_loaded:
            self._yolo_model.load_model()
    
    # =========================================================================
    # KNOWLEDGE BASE ACCESS
    # =========================================================================
    
    def get_disease_info(self, disease_key: str) -> Optional[Dict]:
        """
        Get disease information from knowledge base.
        
        Args:
            disease_key: Key identifying the disease (e.g., 'cassava_mosaic_virus')
            
        Returns:
            Disease information dictionary or None if not found
        """
        if self._knowledge_base is None:
            logger.warning("Knowledge base not loaded")
            return None
        
        diseases = self._knowledge_base.get("diseases", {})
        return diseases.get(disease_key)
    
    def get_all_diseases(self) -> Dict:
        """Get all diseases from knowledge base."""
        if self._knowledge_base is None:
            return {}
        return self._knowledge_base.get("diseases", {})
    
    def get_diseases_by_crop(self, crop: str) -> Dict:
        """
        Get all diseases for a specific crop.
        
        Args:
            crop: Crop type ('cassava', 'cocoa', 'tomato')
            
        Returns:
            Dictionary of diseases for the specified crop
        """
        all_diseases = self.get_all_diseases()
        return {
            key: info for key, info in all_diseases.items()
            if info.get("crop") == crop
        }
    
    # =========================================================================
    # DETECTION METHODS
    # =========================================================================
    
    def detect(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> DetectionResult:
        """
        Detect disease in an image and return comprehensive information.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            DetectionResult with disease information
        """
        self.ensure_model_loaded()
        
        # Get YOLO prediction
        from models.yolo_model import PredictionResult
        prediction = self._yolo_model.predict(image)
        
        # Build detection result with knowledge base info
        return self._build_detection_result(prediction)
    
    def detect_with_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[DetectionResult, Image.Image]:
        """
        Detect disease and return annotated image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (DetectionResult, annotated PIL Image)
        """
        self.ensure_model_loaded()
        
        # Get YOLO prediction with visualization
        prediction, annotated_image = self._yolo_model.predict_with_visualization(image)
        
        # Build detection result
        result = self._build_detection_result(prediction)
        
        return result, annotated_image
    
    def _build_detection_result(self, prediction) -> DetectionResult:
        """
        Build a DetectionResult from a YOLO prediction and knowledge base.
        
        Args:
            prediction: PredictionResult from YOLO model
            
        Returns:
            DetectionResult with complete disease information
        """
        # Get disease info from knowledge base
        disease_info = self.get_disease_info(prediction.disease_key)
        
        if disease_info is None:
            # Return basic result without knowledge base info
            return DetectionResult(
                class_index=prediction.class_index,
                class_name=prediction.class_name,
                disease_key=prediction.disease_key,
                confidence=prediction.confidence,
                crop_type=prediction.crop_type,
                is_healthy=prediction.is_healthy,
                display_name=prediction.class_name
            )
        
        # Extract severity info
        severity = disease_info.get("severity", {})
        
        # Extract yield loss info
        yield_loss = disease_info.get("yield_loss", {})
        
        # Extract treatment costs
        treatment_cost = disease_info.get("total_treatment_cost", {})
        
        # Build complete result
        result = DetectionResult(
            # Basic detection info
            class_index=prediction.class_index,
            class_name=prediction.class_name,
            disease_key=prediction.disease_key,
            confidence=prediction.confidence,
            crop_type=prediction.crop_type,
            is_healthy=prediction.is_healthy,
            
            # Disease details
            display_name=disease_info.get("display_name", prediction.class_name),
            scientific_name=disease_info.get("scientific_name", ""),
            category=disease_info.get("category", ""),
            
            # Severity
            severity_level=severity.get("level", ""),
            severity_scale=severity.get("scale", 0),
            severity_description=severity.get("description", ""),
            
            # Symptoms and transmission
            symptoms=disease_info.get("symptoms", []),
            transmission=disease_info.get("how_it_spreads", []),
            
            # Yield impact
            yield_loss_min=yield_loss.get("min_percent", 0),
            yield_loss_max=yield_loss.get("max_percent", 0),
            yield_loss_description=yield_loss.get("description", ""),
            
            # Treatments
            treatments=disease_info.get("treatments", {}),
            
            # Costs
            treatment_cost_min=treatment_cost.get("min_ngn", 0),
            treatment_cost_max=treatment_cost.get("max_ngn", 0),
            cost_unit=treatment_cost.get("per", "hectare"),
            
            # Prevention
            prevention=disease_info.get("prevention", []),
            
            # Health projection
            health_projection=disease_info.get("health_projection", {}),
            
            # Expert contact
            expert_contact=disease_info.get("expert_contact", {}),
            
            # For healthy plants (not used in 6-class model)
            maintenance_tips=disease_info.get("maintenance_tips", []),
            expected_yield=disease_info.get("expected_yield", {}),
            healthy_message=disease_info.get("message", "")
        )
        
        return result
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_treatment_summary(
        self,
        detection_result: DetectionResult,
        include_traditional: bool = True
    ) -> Dict:
        """
        Get a summarized treatment plan from detection result.
        
        Args:
            detection_result: The detection result
            include_traditional: Whether to include traditional remedies
            
        Returns:
            Summarized treatment dictionary
        """
        treatments = detection_result.treatments
        
        summary = {
            "immediate_actions": [],
            "chemical_options": [],
            "organic_options": [],
            "traditional_options": [],
            "resistant_varieties": [],
            "estimated_cost": {
                "min_ngn": detection_result.treatment_cost_min,
                "max_ngn": detection_result.treatment_cost_max,
                "unit": detection_result.cost_unit
            }
        }
        
        # Cultural practices (immediate actions)
        cultural = treatments.get("cultural", [])
        for t in cultural[:3]:  # Top 3
            summary["immediate_actions"].append({
                "action": t.get("method", ""),
                "description": t.get("description", ""),
                "effectiveness": t.get("effectiveness", "")
            })
        
        # Chemical treatments
        chemical = treatments.get("chemical", [])
        for t in chemical[:2]:  # Top 2
            summary["chemical_options"].append({
                "product": t.get("product_name", ""),
                "brands": t.get("local_brands", []),
                "dosage": t.get("dosage", ""),
                "cost_min": t.get("cost_ngn_min", 0),
                "cost_max": t.get("cost_ngn_max", 0)
            })
        
        # Biological/organic options
        biological = treatments.get("biological", [])
        for t in biological:
            summary["organic_options"].append({
                "method": t.get("method", ""),
                "description": t.get("description", ""),
                "effectiveness": t.get("effectiveness", "")
            })
        
        # Traditional methods
        if include_traditional:
            traditional = treatments.get("traditional", [])
            for t in traditional:
                summary["traditional_options"].append({
                    "method": t.get("method", ""),
                    "description": t.get("description", ""),
                    "cost": t.get("cost_ngn", 0)
                })
        
        # Resistant varieties
        resistant = treatments.get("resistant_varieties", [])
        for v in resistant[:3]:  # Top 3
            summary["resistant_varieties"].append({
                "name": v.get("variety_name", ""),
                "resistance": v.get("resistance_level", ""),
                "source": v.get("source", ""),
                "cost": v.get("cost_ngn_per_bundle", 0)
            })
        
        return summary
    
    def get_health_projection_for_stage(
        self,
        detection_result: DetectionResult,
        infection_stage: str = "early_detection"
    ) -> Dict:
        """
        Get health projection for a specific infection stage.
        
        Args:
            detection_result: The detection result
            infection_stage: Stage of infection (early_detection, moderate_infection, severe_infection)
            
        Returns:
            Health projection dictionary
        """
        projections = detection_result.health_projection
        
        if infection_stage in projections:
            return projections[infection_stage]
        
        return {
            "recovery_chance_percent": 0,
            "message": "Unable to determine health projection."
        }
    
    def validate_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[bool, str]:
        """
        Validate an image before detection.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (is_valid, message)
        """
        self.ensure_model_loaded()
        
        try:
            # Use YOLO model's preprocessing and validation
            pil_image = self._yolo_model.preprocess_image(image)
            return self._yolo_model.validate_image(pil_image)
        except Exception as e:
            return False, str(e)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_service_instance: Optional[DiseaseDetectorService] = None


def get_disease_detector() -> DiseaseDetectorService:
    """
    Get the singleton disease detector service instance.
    
    Returns:
        DiseaseDetectorService instance
    """
    global _service_instance
    
    if _service_instance is None:
        _service_instance = DiseaseDetectorService(auto_load_model=False)
    
    return _service_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_crop_disease(
    image: Union[str, Path, Image.Image, np.ndarray]
) -> DetectionResult:
    """
    Convenience function to detect disease in a crop image.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        
    Returns:
        DetectionResult with complete disease information
    """
    service = get_disease_detector()
    return service.detect(image)


def detect_crop_disease_with_image(
    image: Union[str, Path, Image.Image, np.ndarray]
) -> Tuple[DetectionResult, Image.Image]:
    """
    Detect disease and return annotated image.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (DetectionResult, annotated Image)
    """
    service = get_disease_detector()
    return service.detect_with_image(image)


def get_disease_information(disease_key: str) -> Optional[Dict]:
    """
    Get disease information from knowledge base.
    
    Args:
        disease_key: Key identifying the disease
        
    Returns:
        Disease information dictionary
    """
    service = get_disease_detector()
    return service.get_disease_info(disease_key)


# =============================================================================
# MAIN - Test the service
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Disease Detector Service Test (6-Class Model)")
    print("=" * 60)
    
    # Initialize service
    print("\n1. Initializing Disease Detector Service...")
    service = DiseaseDetectorService(auto_load_model=False)
    
    # Test knowledge base access
    print("\n2. Testing knowledge base access...")
    
    # Get all diseases
    diseases = service.get_all_diseases()
    print(f"   Total diseases in knowledge base: {len(diseases)}")
    
    # Get diseases by crop
    for crop in ["cassava", "cocoa", "tomato"]:
        crop_diseases = service.get_diseases_by_crop(crop)
        print(f"   {crop.capitalize()} diseases: {len(crop_diseases)}")
    
    # Test getting specific disease info (updated for 6-class model)
    print("\n3. Testing disease info retrieval...")
    test_keys = [
        "cassava_bacterial_blight",
        "cassava_mosaic_virus",
        "cocoa_monilia_disease",
        "cocoa_phytophthora_disease",
        "tomato_gray_mold",
        "tomato_wilt_disease"
    ]
    
    for key in test_keys:
        info = service.get_disease_info(key)
        if info:
            print(f"   ✓ {key}: {info.get('display_name', 'N/A')}")
        else:
            print(f"   ✗ {key}: Not found")
    
    # Test with sample disease (without actual image)
    print("\n4. Testing DetectionResult structure...")
    
    # Create a mock detection result
    mock_result = DetectionResult(
        class_index=1,
        class_name="Cassava Mosaic Virus",
        disease_key="cassava_mosaic_virus",
        confidence=0.92,
        crop_type="cassava",
        is_healthy=False,
        display_name="Cassava Mosaic Virus",
        severity_level="very_high",
        severity_scale=5
    )
    
    print(f"   Summary: {mock_result.get_summary()}")
    print(f"   Confidence level: {mock_result.get_confidence_level()}")
    
    print("\n5. To test with actual image detection, run:")
    print("   >>> service.ensure_model_loaded()")
    print("   >>> result = service.detect('path/to/image.jpg')")
    print("   >>> print(result.to_dict())")
    
    print("\n" + "=" * 60)
    print("✅ Disease Detector Service test completed!")
    print("=" * 60)
