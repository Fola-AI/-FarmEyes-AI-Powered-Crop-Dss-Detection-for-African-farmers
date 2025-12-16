"""
FarmEyes Chat API Routes
========================
REST API endpoints for contextual agricultural chat.

Endpoints:
- POST /api/chat - Send message and get response
- GET /api/chat/welcome - Get welcome message for chat page
- GET /api/chat/history - Get chat history for session
- DELETE /api/chat/history - Clear chat history
"""

import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/chat", tags=["Chat"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Request model for chat message."""
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    language: str = Field(default="en", description="Response language (en, ha, yo, ig)")


class ChatResponse(BaseModel):
    """Response model for chat message."""
    success: bool
    response: str
    session_id: str
    language: str
    is_redirect: bool = False
    context: Optional[dict] = None
    timestamp: str


class WelcomeResponse(BaseModel):
    """Response model for welcome message."""
    success: bool
    response: str
    session_id: str
    language: str
    context: Optional[dict] = None
    is_welcome: bool = True


class HistoryResponse(BaseModel):
    """Response model for chat history."""
    success: bool
    session_id: str
    messages: List[dict]
    total_messages: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/", response_model=ChatResponse)
async def send_chat_message(request: ChatRequest):
    """
    Send a chat message and get AI response.
    
    The assistant will:
    - Answer questions about the diagnosed disease
    - Provide related agricultural advice
    - Respond in the user's preferred language
    - Redirect off-topic questions politely
    
    Requires an active session with a diagnosis.
    """
    try:
        # Validate language
        valid_languages = ["en", "ha", "yo", "ig"]
        language = request.language if request.language in valid_languages else "en"
        
        # Validate message
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(message) > 2000:
            raise HTTPException(status_code=400, detail="Message too long (max 2000 characters)")
        
        # Import chat service
        from services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        
        # Get response
        logger.info(f"Chat request from session {request.session_id[:8]}...")
        result = chat_service.chat(
            session_id=request.session_id,
            message=message,
            language=language
        )
        
        if not result.get("success", False):
            # Handle specific error cases
            error_type = result.get("error", "unknown")
            
            if error_type == "no_diagnosis":
                raise HTTPException(
                    status_code=400,
                    detail=result.get("response", "Please analyze an image first")
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.get("response", "Failed to generate response")
                )
        
        # Build response
        response_data = {
            "success": True,
            "response": result.get("response", ""),
            "session_id": result.get("session_id", request.session_id),
            "language": result.get("language", language),
            "is_redirect": result.get("is_redirect", False),
            "context": result.get("context"),
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.get("/welcome", response_model=WelcomeResponse)
async def get_welcome_message(
    session_id: str = Query(..., description="Session ID"),
    language: str = Query(default="en", description="Language code")
):
    """
    Get welcome message for chat page.
    
    Returns a personalized welcome message based on the
    current diagnosis in the session. Should be called
    when user navigates to the chat page.
    """
    try:
        # Validate language
        valid_languages = ["en", "ha", "yo", "ig"]
        language = language if language in valid_languages else "en"
        
        # Import chat service
        from services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        
        # Get welcome message
        result = chat_service.get_welcome_message(session_id, language)
        
        if not result.get("success", False):
            error_type = result.get("error", "unknown")
            
            if error_type == "no_diagnosis":
                raise HTTPException(
                    status_code=400,
                    detail=result.get("response", "Please analyze an image first")
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate welcome message"
                )
        
        response_data = {
            "success": True,
            "response": result.get("response", ""),
            "session_id": result.get("session_id", session_id),
            "language": result.get("language", language),
            "context": result.get("context"),
            "is_welcome": True
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get welcome failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=HistoryResponse)
async def get_chat_history(
    session_id: str = Query(..., description="Session ID"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum messages to return")
):
    """
    Get chat history for a session.
    
    Returns all messages in the current chat session,
    useful for restoring chat state when user returns
    to the chat page.
    """
    try:
        from services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        messages = chat_service.get_history(session_id)
        
        # Apply limit
        if len(messages) > limit:
            messages = messages[-limit:]
        
        return JSONResponse(content={
            "success": True,
            "session_id": session_id,
            "messages": messages,
            "total_messages": len(messages)
        })
        
    except Exception as e:
        logger.error(f"Get history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history")
async def clear_chat_history(
    session_id: str = Query(..., description="Session ID")
):
    """
    Clear chat history for a session.
    
    Removes all messages but keeps the diagnosis context,
    allowing user to start a fresh conversation about
    the same diagnosis.
    """
    try:
        from services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        success = chat_service.clear_history(session_id)
        
        if success:
            return JSONResponse(content={
                "success": True,
                "message": "Chat history cleared",
                "session_id": session_id
            })
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context")
async def get_diagnosis_context(
    session_id: str = Query(..., description="Session ID")
):
    """
    Get current diagnosis context for chat.
    
    Returns the diagnosis information being used as
    context for the chat assistant. Useful for displaying
    context banner in chat UI.
    """
    try:
        from services.session_manager import get_session_manager
        
        session_manager = get_session_manager()
        diagnosis = session_manager.get_diagnosis(session_id)
        
        if not diagnosis or not diagnosis.is_valid():
            raise HTTPException(
                status_code=404,
                detail="No diagnosis found for this session"
            )
        
        return JSONResponse(content={
            "success": True,
            "session_id": session_id,
            "context": diagnosis.to_dict(),
            "context_string": diagnosis.get_context_string()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get context failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice")
async def chat_with_voice(
    session_id: str = Query(..., description="Session ID"),
    language: str = Query(default="en", description="Language code"),
    text: str = Query(..., description="Transcribed text from voice")
):
    """
    Send chat message from voice input.
    
    Convenience endpoint that accepts already-transcribed
    text from the voice input system. The transcription
    is done separately via /api/transcribe.
    
    This is the final step in the voice chat pipeline:
    Voice → Whisper → Text → This endpoint → Response
    """
    try:
        # Create request and use main chat endpoint logic
        request = ChatRequest(
            session_id=session_id,
            message=text,
            language=language
        )
        
        return await send_chat_message(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
