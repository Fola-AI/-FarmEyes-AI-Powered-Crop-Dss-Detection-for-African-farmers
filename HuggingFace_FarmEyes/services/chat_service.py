"""
FarmEyes Chat Service
=====================
Contextual chat using N-ATLaS GGUF model.
"""

import sys
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CHAT SERVICE
# =============================================================================

class ChatService:
    """Chat service using N-ATLaS model."""
    
    # Welcome messages per language
    WELCOME_MESSAGES = {
        "en": "Hello! I'm your FarmEyes assistant. I've analyzed your {crop} and detected **{disease}** with {confidence}% confidence. How can I help you understand or treat this condition?",
        "ha": "Sannu! Ni ne mataimaki na FarmEyes. Na bincika {crop} ɗin ku kuma na gano **{disease}** da tabbaci {confidence}%. Yaya zan taimaka?",
        "yo": "Pẹlẹ o! Mo jẹ́ olùrànlọ́wọ́ FarmEyes rẹ. Mo ti ṣàyẹ̀wò {crop} rẹ mo sì rí **{disease}** pẹ̀lú {confidence}%. Báwo ni mo ṣe lè ràn ọ́ lọ́wọ́?",
        "ig": "Nnọọ! Abụ m onye enyemaka FarmEyes gị. Enyochala m {crop} gị ma chọpụta **{disease}** na {confidence}%. Kedu ka m ga-esi nyere gị aka?"
    }
    
    # No diagnosis messages
    NO_DIAGNOSIS_MESSAGES = {
        "en": "Please analyze a crop image first before chatting. Upload an image on the Diagnosis page.",
        "ha": "Da fatan za a fara bincika hoton amfanin gona kafin tattaunawa.",
        "yo": "Jọ̀wọ́ ṣe àyẹ̀wò àwòrán ohun ọ̀gbìn kan kọ́kọ́.",
        "ig": "Biko nyochaa foto ihe ọkụkụ mbụ tupu nkata."
    }
    
    # Error messages
    ERROR_MESSAGES = {
        "en": "I'm having trouble responding right now. Please try again.",
        "ha": "Ina da matsala wajen amsa yanzu. Da fatan za a sake gwadawa.",
        "yo": "Mo ń ní ìṣòro láti dáhùn báyìí. Jọ̀wọ́ gbìyànjú lẹ́ẹ̀kan sí i.",
        "ig": "Enwere m nsogbu ịza ugbu a. Biko nwaa ọzọ."
    }
    
    def __init__(self, auto_load_model: bool = True):
        self._natlas_model = None
        self._session_manager = None
        self._is_initialized = False
        
        if auto_load_model:
            self._initialize()
    
    def _initialize(self):
        """Initialize dependencies."""
        if self._is_initialized:
            return
        
        try:
            from models.natlas_model import get_natlas_model
            self._natlas_model = get_natlas_model(auto_load_local=True)
            
            from services.session_manager import get_session_manager
            self._session_manager = get_session_manager()
            
            self._is_initialized = True
            logger.info("ChatService initialized")
            
        except Exception as e:
            logger.error(f"ChatService init failed: {e}")
            raise
    
    def ensure_initialized(self):
        if not self._is_initialized:
            self._initialize()
    
    def chat(
        self,
        session_id: str,
        message: str,
        diagnosis_context: Optional[Dict] = None,
        language: str = "en"
    ) -> Dict:
        """Process chat message and return response."""
        try:
            self.ensure_initialized()
            
            if not message or not message.strip():
                return {
                    "success": False,
                    "response": "Please enter a message.",
                    "error": "empty_message",
                    "language": language
                }
            
            message = message.strip()
            
            # Get session
            session = self._session_manager.get_session(session_id)
            if not session:
                session = self._session_manager.create_session(language)
                session_id = session.session_id
            
            # Get diagnosis context
            context = self._get_context(session, diagnosis_context)
            
            if not context or not context.get("image_analyzed"):
                return {
                    "success": False,
                    "response": self.NO_DIAGNOSIS_MESSAGES.get(language, self.NO_DIAGNOSIS_MESSAGES["en"]),
                    "error": "no_diagnosis",
                    "language": language
                }
            
            # Generate response using the model
            logger.info(f"Generating chat response for: {message[:50]}...")
            
            response_text = self._natlas_model.chat_response(
                message=message,
                context=context,
                language=language
            )
            
            if not response_text:
                logger.warning("Model returned empty response")
                return {
                    "success": False,
                    "response": self.ERROR_MESSAGES.get(language, self.ERROR_MESSAGES["en"]),
                    "error": "generation_failed",
                    "language": language
                }
            
            # Store in history
            try:
                self._session_manager.add_chat_message(session_id, "user", message)
                self._session_manager.add_chat_message(session_id, "assistant", response_text)
            except Exception as e:
                logger.warning(f"Failed to save chat history: {e}")
            
            return {
                "success": True,
                "response": response_text,
                "session_id": session_id,
                "language": language,
                "context": {
                    "crop_type": context.get("crop_type"),
                    "disease_name": context.get("disease_name")
                }
            }
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "response": self.ERROR_MESSAGES.get(language, self.ERROR_MESSAGES["en"]),
                "error": str(e),
                "language": language
            }
    
    def get_welcome_message(self, session_id: str, language: str = "en") -> Dict:
        """Get welcome message for chat page."""
        try:
            self.ensure_initialized()
            
            session = self._session_manager.get_session(session_id)
            if not session:
                return {
                    "success": False,
                    "response": self.NO_DIAGNOSIS_MESSAGES.get(language, self.NO_DIAGNOSIS_MESSAGES["en"]),
                    "error": "no_diagnosis"
                }
            
            context = self._get_context(session, None)
            
            if not context or not context.get("image_analyzed"):
                return {
                    "success": False,
                    "response": self.NO_DIAGNOSIS_MESSAGES.get(language, self.NO_DIAGNOSIS_MESSAGES["en"]),
                    "error": "no_diagnosis"
                }
            
            # Format welcome message
            crop = context.get("crop_type", "crop").capitalize()
            disease = context.get("disease_name", "disease")
            confidence = context.get("confidence", 0)
            if confidence <= 1:
                confidence = int(confidence * 100)
            
            template = self.WELCOME_MESSAGES.get(language, self.WELCOME_MESSAGES["en"])
            welcome = template.format(crop=crop, disease=disease, confidence=confidence)
            
            return {
                "success": True,
                "response": welcome,
                "session_id": session_id,
                "language": language,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Welcome message error: {e}")
            return {
                "success": False,
                "response": str(e),
                "error": "error"
            }
    
    def _get_context(self, session, provided_context: Optional[Dict]) -> Optional[Dict]:
        """Get diagnosis context."""
        if provided_context and provided_context.get("image_analyzed"):
            return provided_context
        
        if session and hasattr(session, 'diagnosis') and session.diagnosis:
            try:
                if hasattr(session.diagnosis, 'is_valid') and session.diagnosis.is_valid():
                    return session.diagnosis.to_dict()
                elif hasattr(session.diagnosis, 'to_dict'):
                    ctx = session.diagnosis.to_dict()
                    if ctx.get("image_analyzed"):
                        return ctx
            except:
                pass
        
        return None
    
    def clear_history(self, session_id: str) -> bool:
        """Clear chat history."""
        self.ensure_initialized()
        try:
            return self._session_manager.clear_chat_history(session_id)
        except:
            return False
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get chat history."""
        self.ensure_initialized()
        try:
            messages = self._session_manager.get_chat_history(session_id)
            return [msg.to_dict() if hasattr(msg, 'to_dict') else msg for msg in messages]
        except:
            return []


# =============================================================================
# SINGLETON
# =============================================================================

_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get singleton ChatService."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(auto_load_model=True)
    return _chat_service


def chat(session_id: str, message: str, language: str = "en", diagnosis_context: Optional[Dict] = None) -> Dict:
    """Convenience function."""
    return get_chat_service().chat(session_id, message, diagnosis_context, language)


def get_welcome(session_id: str, language: str = "en") -> Dict:
    """Convenience function."""
    return get_chat_service().get_welcome_message(session_id, language)
