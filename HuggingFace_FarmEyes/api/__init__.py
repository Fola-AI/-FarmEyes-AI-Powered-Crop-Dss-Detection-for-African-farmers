"""
FarmEyes API Routes Package
===========================
REST API endpoint modules for the FarmEyes application.

Endpoints:
- /api/detect - Disease detection from images
- /api/chat - Contextual chat with N-ATLaS
- /api/transcribe - Voice-to-text with Whisper
- /api/session - Session management
- /api/translate - Text translation
"""

from api.routes.detection import router as detection_router
from api.routes.chat import router as chat_router
from api.routes.transcribe import router as transcribe_router

__all__ = [
    "detection_router",
    "chat_router",
    "transcribe_router"
]
