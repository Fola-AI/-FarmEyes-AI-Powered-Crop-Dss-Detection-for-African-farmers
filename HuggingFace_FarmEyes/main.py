"""
FarmEyes Main Application
=========================
FastAPI backend server for FarmEyes crop disease detection.

FIXED: 
- Preloads GGUF model at startup for better performance
- Serves static files correctly for frontend

Run: python main.py
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    FIXED: Preloads GGUF model for better chat performance.
    """
    # STARTUP
    logger.info("=" * 60)
    logger.info("🌱 FarmEyes Starting Up...")
    logger.info("=" * 60)
    
    # Print config
    try:
        from config import print_config_summary
        print_config_summary()
    except ImportError as e:
        logger.warning(f"Could not load config: {e}")
    
    # Initialize session manager
    try:
        from services.session_manager import get_session_manager
        get_session_manager()
        logger.info("✅ Session manager initialized")
    except Exception as e:
        logger.warning(f"Session manager init failed: {e}")
    
    # PRELOAD GGUF MODEL FOR PERFORMANCE
    try:
        from models.natlas_model import get_natlas_model
        logger.info("🔄 Preloading N-ATLaS GGUF model...")
        model = get_natlas_model(auto_load_local=True)
        if model.local_model.is_loaded:
            logger.info("✅ N-ATLaS GGUF model preloaded successfully!")
        else:
            logger.warning("⚠️ GGUF model not loaded - will load on first use")
    except Exception as e:
        logger.warning(f"⚠️ GGUF model preload failed: {e}")
        logger.warning("   Model will load on first use (slower first request)")
    
    logger.info("=" * 60)
    logger.info("🚀 FarmEyes Ready!")
    logger.info("=" * 60)
    
    yield  # Application runs
    
    # SHUTDOWN
    logger.info("=" * 60)
    logger.info("🛑 FarmEyes Shutting Down...")
    logger.info("=" * 60)
    
    try:
        from services.whisper_service import unload_whisper_service
        unload_whisper_service()
    except Exception:
        pass
    
    try:
        from models.natlas_model import unload_natlas_model
        unload_natlas_model()
    except Exception:
        pass
    
    logger.info("👋 Goodbye!")


# =============================================================================
# CREATE APPLICATION
# =============================================================================

app = FastAPI(
    title="FarmEyes API",
    description="AI-Powered Crop Disease Detection for African Farmers",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST LOGGING MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    # Skip logging for static files
    if not request.url.path.startswith("/static"):
        duration = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.1f}ms")
    
    return response


# =============================================================================
# INCLUDE API ROUTERS
# =============================================================================

try:
    from api.routes.detection import router as detection_router
    app.include_router(detection_router)
    logger.info("✅ Detection routes loaded")
except ImportError as e:
    logger.error(f"Failed to load detection routes: {e}")

try:
    from api.routes.chat import router as chat_router
    app.include_router(chat_router)
    logger.info("✅ Chat routes loaded")
except ImportError as e:
    logger.error(f"Failed to load chat routes: {e}")

try:
    from api.routes.transcribe import router as transcribe_router
    app.include_router(transcribe_router)
    logger.info("✅ Transcribe routes loaded")
except ImportError as e:
    logger.error(f"Failed to load transcribe routes: {e}")

try:
    from api.routes.tts import router as tts_router
    app.include_router(tts_router)
    logger.info("✅ TTS routes loaded")
except ImportError as e:
    logger.error(f"Failed to load TTS routes: {e}")


# =============================================================================
# STATIC FILES
# =============================================================================

# Mount static files for CSS, JS
static_dir = PROJECT_ROOT / "frontend"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"✅ Static files mounted from: {static_dir}")
else:
    logger.warning(f"⚠️ Frontend directory not found: {static_dir}")


# =============================================================================
# ROOT ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend application."""
    index_path = PROJECT_ROOT / "frontend" / "index.html"
    
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>FarmEyes</title></head>
        <body>
            <h1>🌱 FarmEyes API</h1>
            <p>Frontend not found. API is running.</p>
            <p>Visit <a href="/api/docs">/api/docs</a> for API documentation.</p>
        </body>
        </html>
        """)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "FarmEyes",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": "FarmEyes API",
        "version": "2.0.0",
        "description": "AI-Powered Crop Disease Detection for African Farmers",
        "endpoints": {
            "detection": "/api/detect",
            "chat": "/api/chat",
            "transcribe": "/api/transcribe",
            "docs": "/api/docs"
        },
        "supported_languages": ["en", "ha", "yo", "ig"],
        "supported_crops": ["cassava", "cocoa", "tomato"]
    }


# =============================================================================
# SESSION ENDPOINTS
# =============================================================================

@app.get("/api/session")
async def create_session(language: str = "en"):
    """Create a new session."""
    try:
        from services.session_manager import get_session_manager
        
        session_manager = get_session_manager()
        session = session_manager.create_session(language)
        
        # Note: created_at is already an ISO format string from session_manager
        return {
            "success": True,
            "session_id": session.session_id,
            "language": session.language,
            "created_at": session.created_at
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    try:
        from services.session_manager import get_session_manager
        
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "session_id": session.session_id,
            "language": session.language,
            "has_diagnosis": session.diagnosis is not None,
            "chat_messages": len(session.chat_history),
            "created_at": session.created_at,  # Already ISO string
            "last_accessed": session.last_accessed  # Unix timestamp float
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/session/{session_id}/language")
async def update_session_language(session_id: str, language: str = "en"):
    """Update session language."""
    try:
        from services.session_manager import get_session_manager
        
        valid_languages = ["en", "ha", "yo", "ig"]
        if language not in valid_languages:
            raise HTTPException(status_code=400, detail=f"Invalid language. Use: {valid_languages}")
        
        session_manager = get_session_manager()
        success = session_manager.set_language(session_id, language)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "session_id": session_id,
            "language": language
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    try:
        from services.session_manager import get_session_manager
        
        session_manager = get_session_manager()
        success = session_manager.delete_session(session_id)
        
        return {
            "success": success,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TRANSLATIONS ENDPOINT
# =============================================================================

@app.get("/api/translations")
async def get_translations(language: str = "en"):
    """Get UI translations."""
    try:
        translations_path = PROJECT_ROOT / "static" / "ui_translations.json"
        
        if translations_path.exists():
            import json
            with open(translations_path, "r", encoding="utf-8") as f:
                all_translations = json.load(f)
            
            lang_translations = all_translations.get(language, all_translations.get("en", {}))
            
            return {
                "success": True,
                "language": language,
                "translations": lang_translations
            }
        else:
            return {
                "success": False,
                "language": language,
                "translations": {},
                "error": "Translations file not found"
            }
    except Exception as e:
        return {
            "success": False,
            "language": language,
            "translations": {},
            "error": str(e)
        }


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors - serve SPA for non-API routes."""
    if not request.url.path.startswith("/api"):
        index_path = PROJECT_ROOT / "frontend" / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
    
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": request.url.path}
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc: Exception):
    """Handle 500 errors."""
    logger.error(f"Server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Check if running on HuggingFace Spaces
    is_spaces = os.environ.get("SPACE_ID") is not None
    
    if is_spaces:
        # HuggingFace Spaces config - must use 0.0.0.0 for external access
        host = "0.0.0.0"
        port = 7860
        reload = False
    else:
        # Local development config
        # FIXED: Use 127.0.0.1 instead of 0.0.0.0 for secure context
        # This allows navigator.mediaDevices (microphone) to work in Chrome
        # Access via http://localhost:7860 (NOT http://0.0.0.0:7860)
        host = os.environ.get("HOST", "127.0.0.1")
        port = int(os.environ.get("PORT", 7860))
        reload = os.environ.get("RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Access the app at: http://localhost:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
