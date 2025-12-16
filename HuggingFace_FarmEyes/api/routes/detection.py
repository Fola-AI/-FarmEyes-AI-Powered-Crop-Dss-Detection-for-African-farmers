"""
FarmEyes Detection API Routes
=============================
REST API endpoints for crop disease detection.

Endpoints:
- POST /api/detect - Analyze crop image for diseases
- GET /api/detect/status - Check model status
- GET /api/detect/classes - Get supported disease classes
"""

import sys
import io
import base64
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/detect", tags=["Detection"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class DetectionRequest(BaseModel):
    """Request model for detection with base64 image."""
    image_base64: str = Field(..., description="Base64 encoded image data")
    language: str = Field(default="en", description="Language code (en, ha, yo, ig)")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")


class DetectionResponse(BaseModel):
    """Response model for disease detection."""
    success: bool
    session_id: str
    detection: dict
    diagnosis: dict
    language: str
    timestamp: str


class StatusResponse(BaseModel):
    """Response model for service status."""
    status: str
    yolo_loaded: bool
    natlas_loaded: bool
    knowledge_base_loaded: bool
    supported_languages: list
    supported_crops: list


class ClassesResponse(BaseModel):
    """Response model for supported classes."""
    total_classes: int
    classes: list
    crops: dict


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_base64_image(base64_string: str) -> bytes:
    """
    Decode base64 image string to bytes.
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        Image bytes
    """
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    try:
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")


def validate_image_format(filename: str) -> bool:
    """
    Validate image file format.
    
    Args:
        filename: Image filename
        
    Returns:
        True if valid format
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ext = Path(filename).suffix.lower()
    return ext in valid_extensions


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/", response_model=DetectionResponse)
async def detect_disease(
    file: UploadFile = File(..., description="Crop image file"),
    language: str = Form(default="en", description="Language code"),
    session_id: Optional[str] = Form(default=None, description="Session ID")
):
    """
    Detect crop disease from uploaded image.
    
    Analyzes the image using YOLOv11 model and returns:
    - Disease detection results
    - Complete diagnosis with treatments
    - All content translated to selected language
    
    Supported formats: JPG, JPEG, PNG, WEBP, BMP
    Maximum file size: 10MB
    """
    try:
        # Validate file format
        if not file.filename or not validate_image_format(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Supported: JPG, JPEG, PNG, WEBP, BMP"
            )
        
        # Read file content
        contents = await file.read()
        
        # Validate file size (10MB max)
        max_size = 10 * 1024 * 1024
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB"
            )
        
        # Validate language
        valid_languages = ["en", "ha", "yo", "ig"]
        if language not in valid_languages:
            language = "en"
        
        # Import services
        from services.session_manager import get_session_manager, DiagnosisContext
        from services.diagnosis_generator import generate_diagnosis_with_image
        from PIL import Image
        
        # Get or create session
        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id, language)
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Generate diagnosis
        logger.info(f"Processing detection for session {session.session_id[:8]}...")
        report, annotated_image = generate_diagnosis_with_image(image, language)
        
        # Update session with diagnosis context
        diagnosis_context = DiagnosisContext.from_diagnosis_report(report)
        session_manager.update_diagnosis(session.session_id, diagnosis_context)
        
        # Build response
        response_data = {
            "success": True,
            "session_id": session.session_id,
            "detection": {
                "disease_name": report.disease_name,
                "crop_type": report.crop_type,
                "confidence": report.confidence,
                "confidence_percent": round(report.confidence * 100, 1),
                "severity_level": report.severity_level,
                "is_healthy": report.is_healthy
            },
            "diagnosis": report.to_dict(),
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Detection complete: {report.disease_name} ({report.confidence:.1%})")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/base64", response_model=DetectionResponse)
async def detect_disease_base64(request: DetectionRequest):
    """
    Detect crop disease from base64 encoded image.
    
    Alternative endpoint for clients that prefer sending
    images as base64 strings rather than file uploads.
    """
    try:
        # Decode base64 image
        try:
            image_bytes = decode_base64_image(request.image_base64)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Validate size
        max_size = 10 * 1024 * 1024
        if len(image_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Maximum size: {max_size // (1024*1024)}MB"
            )
        
        # Validate language
        valid_languages = ["en", "ha", "yo", "ig"]
        language = request.language if request.language in valid_languages else "en"
        
        # Import services
        from services.session_manager import get_session_manager, DiagnosisContext
        from services.diagnosis_generator import generate_diagnosis_with_image
        from PIL import Image
        
        # Get or create session
        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(request.session_id, language)
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate diagnosis
        logger.info(f"Processing base64 detection for session {session.session_id[:8]}...")
        report, annotated_image = generate_diagnosis_with_image(image, language)
        
        # Update session
        diagnosis_context = DiagnosisContext.from_diagnosis_report(report)
        session_manager.update_diagnosis(session.session_id, diagnosis_context)
        
        # Build response
        response_data = {
            "success": True,
            "session_id": session.session_id,
            "detection": {
                "disease_name": report.disease_name,
                "crop_type": report.crop_type,
                "confidence": report.confidence,
                "confidence_percent": round(report.confidence * 100, 1),
                "severity_level": report.severity_level,
                "is_healthy": report.is_healthy
            },
            "diagnosis": report.to_dict(),
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.get("/status", response_model=StatusResponse)
async def get_detection_status():
    """
    Get status of detection service.
    
    Returns information about:
    - Model loading status
    - Supported languages
    - Supported crops
    """
    try:
        # Try to get service status
        status_info = {
            "status": "operational",
            "yolo_loaded": False,
            "natlas_loaded": False,
            "knowledge_base_loaded": False,
            "supported_languages": ["en", "ha", "yo", "ig"],
            "supported_crops": ["cassava", "cocoa", "tomato"]
        }
        
        try:
            from services.disease_detector import get_disease_detector
            detector = get_disease_detector()
            status_info["knowledge_base_loaded"] = detector._knowledge_base is not None
            status_info["yolo_loaded"] = (
                detector._yolo_model is not None and 
                detector._yolo_model._is_loaded
            )
        except Exception as e:
            logger.warning(f"Could not get detector status: {e}")
        
        try:
            from models.natlas_model import get_natlas_model
            natlas = get_natlas_model()
            status_info["natlas_loaded"] = natlas.is_loaded
        except Exception as e:
            logger.warning(f"Could not get N-ATLaS status: {e}")
        
        return JSONResponse(content=status_info)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/classes", response_model=ClassesResponse)
async def get_supported_classes():
    """
    Get list of supported disease classes.
    
    Returns:
    - Total number of classes
    - Class names with indices
    - Mapping of crops to class indices
    """
    try:
        from config import CLASS_NAMES, CROP_TYPES, CLASS_TO_CROP, CLASS_INDEX_TO_KEY
        
        classes_list = []
        for idx, name in enumerate(CLASS_NAMES):
            classes_list.append({
                "index": idx,
                "name": name,
                "key": CLASS_INDEX_TO_KEY.get(idx, ""),
                "crop": CLASS_TO_CROP.get(idx, "unknown")
            })
        
        return JSONResponse(content={
            "total_classes": len(CLASS_NAMES),
            "classes": classes_list,
            "crops": CROP_TYPES
        })
        
    except Exception as e:
        logger.error(f"Get classes failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session_diagnosis(session_id: str):
    """
    Clear diagnosis data for a session.
    
    Clears the current diagnosis and chat history,
    allowing user to start fresh with a new image.
    """
    try:
        from services.session_manager import get_session_manager
        
        session_manager = get_session_manager()
        success = session_manager.clear_diagnosis(session_id)
        
        if success:
            return JSONResponse(content={
                "success": True,
                "message": "Diagnosis cleared",
                "session_id": session_id
            })
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
