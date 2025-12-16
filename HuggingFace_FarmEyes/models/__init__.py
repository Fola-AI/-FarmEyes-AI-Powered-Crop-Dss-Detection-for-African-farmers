"""
FarmEyes Models Package
=======================
AI model wrappers for the FarmEyes application.

Models:
- natlas_model: N-ATLaS hybrid model (API + GGUF) for translation and chat
- yolo_model: YOLOv11 for crop disease detection
"""

from models.natlas_model import (
    NATLaSModel,
    HuggingFaceAPIClient,
    LocalGGUFModel,
    get_natlas_model,
    unload_natlas_model,
    translate_text,
    translate_batch,
    LANGUAGE_NAMES,
    NATIVE_LANGUAGE_NAMES
)

from models.yolo_model import (
    YOLOModel,
    PredictionResult,
    get_yolo_model,
    unload_yolo_model,
    detect_disease,
    detect_disease_with_image
)

__all__ = [
    # N-ATLaS
    "NATLaSModel",
    "HuggingFaceAPIClient",
    "LocalGGUFModel",
    "get_natlas_model",
    "unload_natlas_model",
    "translate_text",
    "translate_batch",
    "LANGUAGE_NAMES",
    "NATIVE_LANGUAGE_NAMES",
    
    # YOLO
    "YOLOModel",
    "PredictionResult",
    "get_yolo_model",
    "unload_yolo_model",
    "detect_disease",
    "detect_disease_with_image"
]
