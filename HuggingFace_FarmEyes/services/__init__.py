"""
FarmEyes Services Package
=========================
Service layer modules for the FarmEyes application.

Services:
- session_manager: Session state and chat memory management
- chat_service: Contextual agricultural chatbot
- whisper_service: Speech-to-text for voice input
- disease_detector: Disease detection with knowledge base
- translator: N-ATLaS translation service
- diagnosis_generator: Complete diagnosis report generation
"""

# Import services for easy access
from services.session_manager import (
    SessionManager,
    UserSession,
    DiagnosisContext,
    ChatMessage,
    get_session_manager,
    create_session,
    get_session,
    get_or_create_session
)

from services.chat_service import (
    ChatService,
    get_chat_service,
    chat,
    get_welcome
)

from services.whisper_service import (
    WhisperService,
    AudioProcessor,
    get_whisper_service,
    transcribe_audio,
    transcribe_bytes
)

# These will be imported from existing files
# from services.disease_detector import (
#     DiseaseDetectorService,
#     DetectionResult,
#     get_disease_detector,
#     detect_crop_disease
# )

# from services.translator import (
#     TranslatorService,
#     get_translator,
#     translate_text
# )

# from services.diagnosis_generator import (
#     DiagnosisGenerator,
#     DiagnosisReport,
#     get_diagnosis_generator,
#     generate_diagnosis
# )

__all__ = [
    # Session management
    "SessionManager",
    "UserSession",
    "DiagnosisContext",
    "ChatMessage",
    "get_session_manager",
    "create_session",
    "get_session",
    "get_or_create_session",
    
    # Chat service
    "ChatService",
    "get_chat_service",
    "chat",
    "get_welcome",
    
    # Whisper service
    "WhisperService",
    "AudioProcessor",
    "get_whisper_service",
    "transcribe_audio",
    "transcribe_bytes",
]
