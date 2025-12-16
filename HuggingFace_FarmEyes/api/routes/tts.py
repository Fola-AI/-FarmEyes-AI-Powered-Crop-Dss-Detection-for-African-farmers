"""
FarmEyes TTS API Routes
=======================
REST API endpoints for text-to-speech synthesis.

Endpoints:
- POST /api/tts - Synthesize text to speech
- GET /api/tts/languages - Get supported languages
- GET /api/tts/status - Check TTS service status
"""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/tts", tags=["Text-to-Speech"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class TTSRequest(BaseModel):
    """Request model for TTS synthesis."""
    text: str = Field(..., description="Text to convert to speech", max_length=2000)
    language: str = Field(default="en", description="Language code (en, ha, yo, ig)")


class TTSResponse(BaseModel):
    """Response model for TTS synthesis."""
    success: bool
    audio_base64: Optional[str] = None
    content_type: str = "audio/flac"
    duration: float = 0.0
    language: str = "en"
    text_length: int = 0
    processing_time: Optional[float] = None
    error: Optional[str] = None


class LanguagesResponse(BaseModel):
    """Response model for supported languages."""
    success: bool
    languages: dict


class StatusResponse(BaseModel):
    """Response model for service status."""
    success: bool
    status: str
    has_token: bool
    supported_languages: list


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("", response_model=TTSResponse)
@router.post("/", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize text to speech.
    
    Converts the provided text to audio using Meta MMS-TTS.
    Returns base64 encoded audio data.
    
    Supported languages:
    - en: English
    - ha: Hausa
    - yo: Yoruba
    - ig: Igbo
    """
    try:
        from services.tts_service import get_tts_service
        
        logger.info(f"TTS request: lang={request.language}, text_len={len(request.text)}")
        
        # Get TTS service
        tts_service = get_tts_service()
        
        # Check language support
        if not tts_service.is_language_supported(request.language):
            raise HTTPException(
                status_code=400,
                detail=f"Language '{request.language}' is not supported. Use: en, ha, yo, ig"
            )
        
        # Synthesize
        result = tts_service.synthesize(request.text, request.language)
        
        if result["success"]:
            return TTSResponse(
                success=True,
                audio_base64=result["audio_base64"],
                content_type=result.get("content_type", "audio/flac"),
                duration=result.get("duration", 0.0),
                language=result["language"],
                text_length=result.get("text_length", len(request.text)),
                processing_time=result.get("processing_time")
            )
        else:
            # Return error but don't raise exception (for fallback handling)
            return TTSResponse(
                success=False,
                language=request.language,
                text_length=len(request.text),
                error=result.get("error", "TTS synthesis failed")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/languages", response_model=LanguagesResponse)
async def get_supported_languages():
    """
    Get list of supported TTS languages.
    
    Returns language codes and their display names.
    """
    try:
        from services.tts_service import get_tts_service
        
        tts_service = get_tts_service()
        languages = tts_service.get_supported_languages()
        
        return LanguagesResponse(
            success=True,
            languages=languages
        )
        
    except Exception as e:
        logger.error(f"Get languages failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=StatusResponse)
async def get_tts_status():
    """
    Get TTS service status.
    
    Returns whether the service is configured and ready.
    """
    try:
        from services.tts_service import get_tts_service
        
        tts_service = get_tts_service()
        has_token = bool(tts_service.hf_token)
        languages = list(tts_service.get_supported_languages().keys())
        
        status = "ready" if has_token else "no_token"
        
        return StatusResponse(
            success=True,
            status=status,
            has_token=has_token,
            supported_languages=languages
        )
        
    except Exception as e:
        logger.error(f"Get status failed: {e}")
        return StatusResponse(
            success=False,
            status="error",
            has_token=False,
            supported_languages=[]
        )
