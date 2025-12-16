"""
FarmEyes API Routes
===================
Individual route modules for REST API endpoints.
"""

from api.routes.detection import router as detection_router
from api.routes.chat import router as chat_router
from api.routes.transcribe import router as transcribe_router

__all__ = [
    "detection_router",
    "chat_router", 
    "transcribe_router"
]
