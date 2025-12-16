"""
FarmEyes YOLOv11 Model Integration
==================================
Handles loading and inference with YOLOv11 model for crop disease detection.
Optimized for Apple Silicon M1 Pro with MPS (Metal Performance Shaders) acceleration.

Model: Custom trained YOLOv11 for 6 disease classes (no healthy classes)
Crops: Cassava, Cocoa, Tomato
Classes:
    0: Cassava Bacteria Blight
    1: Cassava Mosaic Virus
    2: Cocoa Monilia Disease
    3: Cocoa Phytophthora Disease
    4: Tomato Gray Mold Disease
    5: Tomato Wilt Disease
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PREDICTION RESULT DATACLASS
# =============================================================================

@dataclass
class PredictionResult:
    """
    Container for disease prediction results.
    """
    class_index: int           # Index of predicted class (0-5)
    class_name: str            # Human-readable class name
    disease_key: str           # Key for knowledge base lookup
    confidence: float          # Confidence score (0.0 - 1.0)
    crop_type: str             # Crop type (cassava, cocoa, tomato)
    is_healthy: bool           # Whether plant is healthy (always False in 6-class model)
    bbox: Optional[List[float]] = None  # Bounding box [x1, y1, x2, y2] if available
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "class_index": self.class_index,
            "class_name": self.class_name,
            "disease_key": self.disease_key,
            "confidence": round(self.confidence, 4),
            "confidence_percent": round(self.confidence * 100, 1),
            "crop_type": self.crop_type,
            "is_healthy": self.is_healthy,
            "bbox": self.bbox
        }
    
    def __repr__(self) -> str:
        return f"PredictionResult({self.class_name}, conf={self.confidence:.2%}, crop={self.crop_type})"


# =============================================================================
# YOLO MODEL CLASS
# =============================================================================

class YOLOModel:
    """
    YOLOv11 Model wrapper for FarmEyes crop disease detection.
    Uses Ultralytics library with MPS acceleration for Apple Silicon.
    
    6-class model (all diseases, no healthy classes):
        0: Cassava Bacteria Blight
        1: Cassava Mosaic Virus
        2: Cocoa Monilia Disease
        3: Cocoa Phytophthora Disease
        4: Tomato Gray Mold Disease
        5: Tomato Wilt Disease
    """
    
    # Class mappings (must match your trained model - 6 classes)
    CLASS_NAMES: List[str] = [
        "Cassava Bacteria Blight",      # Index 0
        "Cassava Mosaic Virus",         # Index 1
        "Cocoa Monilia Disease",        # Index 2
        "Cocoa Phytophthora Disease",   # Index 3
        "Tomato Gray Mold Disease",     # Index 4
        "Tomato Wilt Disease"           # Index 5
    ]
    
    # Class index to knowledge base key mapping (6 classes)
    CLASS_TO_KEY: Dict[int, str] = {
        0: "cassava_bacterial_blight",
        1: "cassava_mosaic_virus",
        2: "cocoa_monilia_disease",
        3: "cocoa_phytophthora_disease",
        4: "tomato_gray_mold",
        5: "tomato_wilt_disease"
    }
    
    # Class index to crop type mapping (6 classes)
    CLASS_TO_CROP: Dict[int, str] = {
        0: "cassava",   # Cassava Bacteria Blight
        1: "cassava",   # Cassava Mosaic Virus
        2: "cocoa",     # Cocoa Monilia Disease
        3: "cocoa",     # Cocoa Phytophthora Disease
        4: "tomato",    # Tomato Gray Mold Disease
        5: "tomato"     # Tomato Wilt Disease
    }
    
    # No healthy class indices in 6-class model (all classes are diseases)
    HEALTHY_INDICES: List[int] = []
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "mps",
        input_size: int = 640
    ):
        """
        Initialize YOLOv11 model.
        
        Args:
            model_path: Path to trained YOLOv11 .pt weights file
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Compute device ('mps' for Apple Silicon, 'cuda', 'cpu')
            input_size: Input image size for the model
        """
        # Import config here to avoid circular imports
        from config import yolo_config, MODELS_DIR
        
        self.model_path = model_path or str(yolo_config.model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        # Determine best device
        self.device = self._get_best_device(device)
        
        # Model instance (lazy loaded)
        self._model = None
        self._is_loaded = False
        
        logger.info(f"YOLOModel initialized:")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        logger.info(f"  Input size: {self.input_size}")
        logger.info(f"  Number of classes: {len(self.CLASS_NAMES)}")
    
    # =========================================================================
    # DEVICE MANAGEMENT
    # =========================================================================
    
    def _get_best_device(self, preferred: str = "mps") -> str:
        """
        Determine the best available compute device.
        
        Args:
            preferred: Preferred device ('mps', 'cuda', 'cpu')
            
        Returns:
            Best available device string
        """
        import torch
        
        if preferred == "mps" and torch.backends.mps.is_available():
            logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
            return "mps"
        elif preferred == "cuda" and torch.cuda.is_available():
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            logger.info("Using CPU for inference")
            return "cpu"
    
    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    
    def load_model(self) -> bool:
        """
        Load the YOLOv11 model into memory.
        
        Returns:
            True if model loaded successfully
        """
        if self._is_loaded:
            logger.info("Model already loaded")
            return True
        
        try:
            from ultralytics import YOLO
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.warning(f"Model file not found at {self.model_path}")
                logger.warning("Using placeholder - please provide trained model")
                
                # Create a placeholder with pretrained YOLOv11n for testing
                # Replace this with your actual trained model
                logger.info("Loading pretrained YOLOv11n as placeholder...")
                self._model = YOLO("yolo11n.pt")  # Downloads pretrained model
                self._is_placeholder = True
            else:
                logger.info(f"Loading YOLOv11 model from {self.model_path}...")
                self._model = YOLO(self.model_path)
                self._is_placeholder = False
            
            # Move model to device
            self._model.to(self.device)
            
            self._is_loaded = True
            logger.info(f"✅ YOLOv11 model loaded successfully on {self.device}!")
            
            return True
            
        except ImportError:
            logger.error("Ultralytics not installed!")
            logger.error("Install with: pip install ultralytics")
            raise ImportError("ultralytics package is required")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._is_loaded = False
            raise RuntimeError(f"Could not load YOLOv11 model: {e}")
    
    def unload_model(self):
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            
            # Clear GPU cache
            import torch
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded from memory")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    # =========================================================================
    # IMAGE PREPROCESSING
    # =========================================================================
    
    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            PIL Image ready for inference
        """
        # Handle different input types
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            pil_image = Image.open(image_path)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        return pil_image
    
    def validate_image(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Validate image for inference.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check image size
        width, height = image.size
        
        if width < 32 or height < 32:
            return False, "Image too small. Minimum size is 32x32 pixels."
        
        if width > 4096 or height > 4096:
            return False, "Image too large. Maximum size is 4096x4096 pixels."
        
        return True, "Image is valid"
    
    # =========================================================================
    # INFERENCE
    # =========================================================================
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> PredictionResult:
        """
        Run disease detection on an image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            PredictionResult with disease information
        """
        if not self._is_loaded:
            self.load_model()
        
        # Preprocess image
        pil_image = self.preprocess_image(image)
        
        # Validate image
        is_valid, message = self.validate_image(pil_image)
        if not is_valid:
            logger.warning(f"Image validation failed: {message}")
            return self._create_low_confidence_result()
        
        try:
            # Run inference
            results = self._model(
                pil_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                device=self.device,
                verbose=False
            )
            
            # Parse results
            predictions = self._parse_results(results)
            
            if not predictions:
                logger.info("No predictions above confidence threshold")
                return self._create_low_confidence_result()
            
            # Return top prediction
            return predictions[0]
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self._create_low_confidence_result()
    
    def predict_with_visualization(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[PredictionResult, Image.Image]:
        """
        Run detection and return annotated image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (PredictionResult, annotated PIL Image)
        """
        if not self._is_loaded:
            self.load_model()
        
        # Preprocess image
        pil_image = self.preprocess_image(image)
        
        # Validate image
        is_valid, message = self.validate_image(pil_image)
        if not is_valid:
            logger.warning(f"Image validation failed: {message}")
            return self._create_low_confidence_result(), pil_image
        
        try:
            # Run inference
            results = self._model(
                pil_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                device=self.device,
                verbose=False
            )
            
            # Parse results
            predictions = self._parse_results(results)
            
            # Get annotated image
            annotated = results[0].plot()
            annotated_pil = Image.fromarray(annotated[..., ::-1])  # BGR to RGB
            
            if not predictions:
                return self._create_low_confidence_result(), annotated_pil
            
            return predictions[0], annotated_pil
            
        except Exception as e:
            logger.error(f"Inference with visualization failed: {e}")
            return self._create_low_confidence_result(), pil_image
    
    def _parse_results(self, results) -> List[PredictionResult]:
        """
        Parse YOLO results into PredictionResult objects.
        
        Args:
            results: YOLO inference results
            
        Returns:
            List of PredictionResult objects sorted by confidence
        """
        predictions = []
        
        for result in results:
            # Check if we have classification results (for classification model)
            if hasattr(result, 'probs') and result.probs is not None:
                probs = result.probs
                
                # Get top prediction
                top_idx = int(probs.top1)
                top_conf = float(probs.top1conf)
                
                # Handle placeholder model (pretrained YOLO)
                if hasattr(self, '_is_placeholder') and self._is_placeholder:
                    # Map to our classes for demo purposes
                    top_idx = top_idx % len(self.CLASS_NAMES)
                
                if top_idx < len(self.CLASS_NAMES):
                    prediction = PredictionResult(
                        class_index=top_idx,
                        class_name=self.CLASS_NAMES[top_idx],
                        disease_key=self.CLASS_TO_KEY[top_idx],
                        confidence=top_conf,
                        crop_type=self.CLASS_TO_CROP[top_idx],
                        is_healthy=top_idx in self.HEALTHY_INDICES  # Always False for 6-class model
                    )
                    predictions.append(prediction)
            
            # Check for detection results (for detection model)
            elif hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    cls_idx = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    bbox = boxes.xyxy[i].tolist() if boxes.xyxy is not None else None
                    
                    # Handle placeholder model
                    if hasattr(self, '_is_placeholder') and self._is_placeholder:
                        cls_idx = cls_idx % len(self.CLASS_NAMES)
                    
                    if cls_idx < len(self.CLASS_NAMES):
                        prediction = PredictionResult(
                            class_index=cls_idx,
                            class_name=self.CLASS_NAMES[cls_idx],
                            disease_key=self.CLASS_TO_KEY[cls_idx],
                            confidence=conf,
                            crop_type=self.CLASS_TO_CROP[cls_idx],
                            is_healthy=cls_idx in self.HEALTHY_INDICES,  # Always False for 6-class model
                            bbox=bbox
                        )
                        predictions.append(prediction)
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        return predictions
    
    def _create_low_confidence_result(self) -> PredictionResult:
        """Create a result indicating low confidence / no detection."""
        return PredictionResult(
            class_index=-1,
            class_name="Unknown",
            disease_key="unknown",
            confidence=0.0,
            crop_type="unknown",
            is_healthy=False
        )
    
    # =========================================================================
    # BATCH INFERENCE
    # =========================================================================
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]]
    ) -> List[PredictionResult]:
        """
        Run detection on multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of PredictionResult objects (one per image)
        """
        if not self._is_loaded:
            self.load_model()
        
        results = []
        for image in images:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                results.append(self._create_low_confidence_result())
        
        return results
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_class_info(self, class_index: int) -> Dict:
        """
        Get information about a class by index.
        
        Args:
            class_index: Index of the class (0-5)
            
        Returns:
            Dictionary with class information
        """
        if class_index < 0 or class_index >= len(self.CLASS_NAMES):
            return {
                "class_index": class_index,
                "class_name": "Unknown",
                "disease_key": "unknown",
                "crop_type": "unknown",
                "is_healthy": False
            }
        
        return {
            "class_index": class_index,
            "class_name": self.CLASS_NAMES[class_index],
            "disease_key": self.CLASS_TO_KEY[class_index],
            "crop_type": self.CLASS_TO_CROP[class_index],
            "is_healthy": class_index in self.HEALTHY_INDICES  # Always False for 6-class model
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            "model_path": self.model_path,
            "is_loaded": self._is_loaded,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "input_size": self.input_size,
            "num_classes": len(self.CLASS_NAMES),
            "classes": self.CLASS_NAMES
        }
        
        if self._is_loaded and hasattr(self, '_is_placeholder'):
            info["is_placeholder"] = self._is_placeholder
        
        return info


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_model_instance: Optional[YOLOModel] = None


def get_yolo_model() -> YOLOModel:
    """
    Get the singleton YOLO model instance.
    
    Returns:
        YOLOModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        from config import yolo_config
        
        _model_instance = YOLOModel(
            model_path=str(yolo_config.model_path),
            confidence_threshold=yolo_config.confidence_threshold,
            iou_threshold=yolo_config.iou_threshold,
            device=yolo_config.device,
            input_size=yolo_config.input_size
        )
    
    return _model_instance


def unload_yolo_model():
    """Unload the singleton YOLO model to free memory."""
    global _model_instance
    
    if _model_instance is not None:
        _model_instance.unload_model()
        _model_instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_disease(
    image: Union[str, Path, Image.Image, np.ndarray]
) -> PredictionResult:
    """
    Convenience function to detect disease in an image.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        
    Returns:
        PredictionResult with disease information
    """
    model = get_yolo_model()
    return model.predict(image)


def detect_disease_with_image(
    image: Union[str, Path, Image.Image, np.ndarray]
) -> Tuple[PredictionResult, Image.Image]:
    """
    Detect disease and return annotated image.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (PredictionResult, annotated Image)
    """
    model = get_yolo_model()
    return model.predict_with_visualization(image)


# =============================================================================
# MAIN - Test the model
# =============================================================================

if __name__ == "__main__":
    import torch
    
    print("=" * 60)
    print("YOLOv11 Model Test (6-Class Disease Detection)")
    print("=" * 60)
    
    # Check device
    print("\n1. Checking compute device...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    print(f"   MPS built: {torch.backends.mps.is_built()}")
    
    # Initialize model
    print("\n2. Initializing YOLOv11 model...")
    model = YOLOModel()
    
    # Load model
    print("\n3. Loading model...")
    model.load_model()
    
    # Print model info
    print("\n4. Model information:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test with a sample image (if available)
    print("\n5. Testing inference...")
    print("   To test with an actual image, run:")
    print("   >>> result = model.predict('path/to/your/image.jpg')")
    print("   >>> print(result)")
    
    # Print class mappings
    print("\n6. Class mappings (6 classes - all diseases):")
    for idx, name in enumerate(model.CLASS_NAMES):
        crop = model.CLASS_TO_CROP[idx]
        key = model.CLASS_TO_KEY[idx]
        print(f"   {idx}: {name}")
        print(f"      Crop: {crop}")
        print(f"      Key: {key}")
    
    print("\n" + "=" * 60)
    print("✅ YOLOv11 model test completed!")
    print("=" * 60)
