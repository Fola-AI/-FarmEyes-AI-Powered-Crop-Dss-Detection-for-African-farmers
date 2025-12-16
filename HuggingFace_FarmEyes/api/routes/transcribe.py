"""
FarmEyes Transcribe API Routes
==============================
REST API endpoints for speech-to-text transcription.

Endpoints:
- POST /api/transcribe - Transcribe audio to text
- GET /api/transcribe/status - Check Whisper model status
- GET /api/transcribe/formats - Get supported audio formats
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
router = APIRouter(prefix="/api/transcribe", tags=["Transcription"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class TranscribeRequest(BaseModel):
    """Request model for base64 audio transcription."""
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    filename: str = Field(default="audio.wav", description="Original filename for format detection")
    language_hint: Optional[str] = Field(default=None, description="Language hint (en, ha, yo, ig)")


class TranscribeResponse(BaseModel):
    """Response model for transcription."""
    success: bool
    text: str
    language: Optional[str] = None
    confidence: float = 0.0
    duration: float = 0.0
    processing_time: Optional[float] = None


class StatusResponse(BaseModel):
    """Response model for service status."""
    status: str
    model_loaded: bool
    model_size: str
    device: str
    supported_formats: list


class FormatsResponse(BaseModel):
    """Response model for supported formats."""
    formats: list
    max_file_size_mb: int
    max_duration_seconds: int


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_base64_audio(base64_string: str) -> bytes:
    """
    Decode base64 audio string to bytes.
    
    Args:
        base64_string: Base64 encoded audio
        
    Returns:
        Audio bytes
    """
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    try:
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 audio: {e}")


def validate_audio_format(filename: str) -> bool:
    """
    Validate audio file format.
    
    Args:
        filename: Audio filename
        
    Returns:
        True if valid format
    """
    valid_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
    ext = Path(filename).suffix.lower()
    return ext in valid_extensions


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file"),
    language_hint: Optional[str] = Form(default=None, description="Language hint (en, ha, yo, ig)")
):
    """
    Transcribe audio file to text.
    
    Uses OpenAI Whisper model for accurate speech-to-text,
    with special optimization for Nigerian languages.
    
    Supported formats: WAV, MP3, M4A, OGG, FLAC, WEBM
    Maximum file size: 5MB
    Maximum duration: 30 seconds
    
    Language hints improve accuracy:
    - en: English
    - ha: Hausa
    - yo: Yoruba
    - ig: Igbo
    """
    try:
        # Validate file format
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not validate_audio_format(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid audio format. Supported: WAV, MP3, M4A, OGG, FLAC, WEBM"
            )
        
        # Read file content
        contents = await file.read()
        
        # Validate file size (5MB max)
        max_size = 5 * 1024 * 1024
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB"
            )
        
        # Validate language hint
        valid_languages = ["en", "ha", "yo", "ig"]
        if language_hint and language_hint not in valid_languages:
            language_hint = None
        
        # Import Whisper service
        from services.whisper_service import get_whisper_service
        
        whisper_service = get_whisper_service()
        
        # Transcribe
        logger.info(f"Transcribing audio: {file.filename}")
        result = whisper_service.transcribe_bytes(
            audio_bytes=contents,
            filename=file.filename,
            language_hint=language_hint
        )
        
        if not result.get("success", False):
            error_msg = result.get("error", "Transcription failed")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Build response
        response_data = {
            "success": True,
            "text": result.get("text", ""),
            "language": result.get("language"),
            "confidence": result.get("confidence", 0.0),
            "duration": result.get("duration", 0.0),
            "processing_time": result.get("processing_time")
        }
        
        logger.info(f"Transcription complete: {len(response_data['text'])} chars")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/base64", response_model=TranscribeResponse)
async def transcribe_audio_base64(request: TranscribeRequest):
    """
    Transcribe base64 encoded audio to text.
    
    Alternative endpoint for clients that prefer sending
    audio as base64 strings (e.g., from web recordings).
    """
    try:
        # Decode base64 audio
        try:
            audio_bytes = decode_base64_audio(request.audio_base64)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Validate size (5MB max)
        max_size = 5 * 1024 * 1024
        if len(audio_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too large. Maximum size: {max_size // (1024*1024)}MB"
            )
        
        # Validate format from filename
        if not validate_audio_format(request.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid audio format. Supported: WAV, MP3, M4A, OGG, FLAC, WEBM"
            )
        
        # Validate language hint
        valid_languages = ["en", "ha", "yo", "ig"]
        language_hint = request.language_hint
        if language_hint and language_hint not in valid_languages:
            language_hint = None
        
        # Import Whisper service
        from services.whisper_service import get_whisper_service
        
        whisper_service = get_whisper_service()
        
        # Transcribe
        logger.info(f"Transcribing base64 audio: {request.filename}")
        result = whisper_service.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=request.filename,
            language_hint=language_hint
        )
        
        if not result.get("success", False):
            error_msg = result.get("error", "Transcription failed")
            raise HTTPException(status_code=500, detail=error_msg)
        
        response_data = {
            "success": True,
            "text": result.get("text", ""),
            "language": result.get("language"),
            "confidence": result.get("confidence", 0.0),
            "duration": result.get("duration", 0.0),
            "processing_time": result.get("processing_time")
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.get("/status", response_model=StatusResponse)
async def get_transcription_status():
    """
    Get status of transcription service.
    
    Returns information about:
    - Whisper model loading status
    - Model size and device
    - Supported formats
    """
    try:
        from services.whisper_service import get_whisper_service
        
        whisper_service = get_whisper_service()
        info = whisper_service.get_model_info()
        
        return JSONResponse(content={
            "status": "operational" if info.get("is_loaded") else "model_not_loaded",
            "model_loaded": info.get("is_loaded", False),
            "model_size": info.get("model_size", "base"),
            "device": info.get("device", "cpu"),
            "supported_formats": info.get("supported_formats", [])
        })
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/formats", response_model=FormatsResponse)
async def get_supported_formats():
    """
    Get supported audio formats and limits.
    
    Returns:
    - List of supported audio formats
    - Maximum file size
    - Maximum audio duration
    """
    try:
        from services.whisper_service import AudioProcessor
        
        return JSONResponse(content={
            "formats": list(AudioProcessor.SUPPORTED_FORMATS),
            "max_file_size_mb": 5,
            "max_duration_seconds": AudioProcessor.MAX_DURATION
        })
        
    except Exception as e:
        logger.error(f"Get formats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-language")
async def detect_audio_language(
    file: UploadFile = File(..., description="Audio file")
):
    """
    Detect language in audio file.
    
    Uses Whisper's language detection to identify
    the spoken language without full transcription.
    Faster than full transcription for language detection.
    """
    try:
        # Validate file
        if not file.filename or not validate_audio_format(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid audio format"
            )
        
        # Read content
        contents = await file.read()
        
        # Validate size
        max_size = 5 * 1024 * 1024
        if len(contents) > max_size:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Import service
        from services.whisper_service import get_whisper_service
        
        whisper_service = get_whisper_service()
        
        # Detect language
        result = whisper_service.detect_language(contents)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Language detection failed")
            )
        
        return JSONResponse(content={
            "success": True,
            "language": result.get("language"),
            "confidence": result.get("confidence", 0.0),
            "top_languages": result.get("all_probabilities", {})
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-model")
async def load_whisper_model():
    """
    Explicitly load the Whisper model.
    
    Useful for warming up the model before user
    starts using voice input. Model loads automatically
    on first use, but pre-loading improves UX.
    """
    try:
        from services.whisper_service import get_whisper_service
        
        whisper_service = get_whisper_service()
        
        if whisper_service.is_loaded:
            return JSONResponse(content={
                "success": True,
                "message": "Model already loaded",
                "model_size": whisper_service.model_size
            })
        
        # Load model
        logger.info("Pre-loading Whisper model...")
        success = whisper_service.load_model()
        
        if success:
            return JSONResponse(content={
                "success": True,
                "message": "Model loaded successfully",
                "model_size": whisper_service.model_size
            })
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to load Whisper model"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
