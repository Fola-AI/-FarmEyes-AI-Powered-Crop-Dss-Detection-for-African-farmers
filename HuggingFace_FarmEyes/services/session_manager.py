"""
FarmEyes Session Manager
========================
Manages user sessions for maintaining:
- Diagnosis context across chat interactions
- Chat history within a session
- Language preferences
- Session lifecycle (creation, access, cleanup)

Thread-safe implementation for concurrent API requests.
Optimized for memory efficiency with automatic cleanup.
"""

import uuid
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CHAT MESSAGE DATACLASS
# =============================================================================

@dataclass
class ChatMessage:
    """
    Represents a single chat message.
    
    Attributes:
        role: 'user' or 'assistant'
        content: Message text
        timestamp: When message was created
        language: Language code (en, ha, yo, ig)
    """
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    language: str = "en"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "language": self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        """Create ChatMessage from dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            language=data.get("language", "en")
        )


# =============================================================================
# DIAGNOSIS CONTEXT DATACLASS
# =============================================================================

@dataclass
class DiagnosisContext:
    """
    Stores the current diagnosis context for chat.
    
    This is populated after a successful disease detection
    and used to provide context for the chat assistant.
    """
    # Detection info
    disease_key: str = ""
    disease_name: str = ""
    crop_type: str = ""
    confidence: float = 0.0
    severity_level: str = ""
    
    # Additional context
    symptoms: List[str] = field(default_factory=list)
    treatment_summary: str = ""
    prevention_tips: List[str] = field(default_factory=list)
    cost_estimate: str = ""
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    image_analyzed: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "disease_key": self.disease_key,
            "disease_name": self.disease_name,
            "crop_type": self.crop_type,
            "confidence": self.confidence,
            "severity_level": self.severity_level,
            "symptoms": self.symptoms,
            "treatment_summary": self.treatment_summary,
            "prevention_tips": self.prevention_tips,
            "cost_estimate": self.cost_estimate,
            "timestamp": self.timestamp,
            "image_analyzed": self.image_analyzed
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DiagnosisContext':
        """Create DiagnosisContext from dictionary."""
        return cls(
            disease_key=data.get("disease_key", ""),
            disease_name=data.get("disease_name", ""),
            crop_type=data.get("crop_type", ""),
            confidence=data.get("confidence", 0.0),
            severity_level=data.get("severity_level", ""),
            symptoms=data.get("symptoms", []),
            treatment_summary=data.get("treatment_summary", ""),
            prevention_tips=data.get("prevention_tips", []),
            cost_estimate=data.get("cost_estimate", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            image_analyzed=data.get("image_analyzed", False)
        )
    
    @classmethod
    def from_diagnosis_report(cls, report) -> 'DiagnosisContext':
        """
        Create DiagnosisContext from a DiagnosisReport object.
        
        Args:
            report: DiagnosisReport from diagnosis_generator
            
        Returns:
            DiagnosisContext populated with report data
        """
        # Extract treatment summary
        treatment_parts = []
        if report.immediate_actions:
            actions = [a.get("action", "") for a in report.immediate_actions[:2]]
            treatment_parts.append(f"Immediate: {', '.join(actions)}")
        if report.chemical_treatments:
            chemicals = [c.get("product", "") for c in report.chemical_treatments[:1]]
            treatment_parts.append(f"Chemical: {', '.join(chemicals)}")
        
        treatment_summary = ". ".join(treatment_parts) if treatment_parts else "Consult expert"
        
        # Extract cost estimate
        if report.treatment_cost_min > 0 and report.treatment_cost_max > 0:
            cost_estimate = f"₦{report.treatment_cost_min:,} - ₦{report.treatment_cost_max:,} per hectare"
        else:
            cost_estimate = "Cost varies by treatment"
        
        return cls(
            disease_key=report.raw_detection.get("detection", {}).get("disease_key", ""),
            disease_name=report.disease_name,
            crop_type=report.crop_type,
            confidence=report.confidence,
            severity_level=report.severity_level,
            symptoms=report.symptoms[:5] if report.symptoms else [],
            treatment_summary=treatment_summary,
            prevention_tips=report.prevention_tips[:3] if report.prevention_tips else [],
            cost_estimate=cost_estimate,
            image_analyzed=True
        )
    
    def get_context_string(self) -> str:
        """
        Generate a context string for the chat assistant.
        
        Returns:
            Formatted string with diagnosis context
        """
        if not self.image_analyzed:
            return "No diagnosis has been made yet."
        
        symptoms_str = ", ".join(self.symptoms[:3]) if self.symptoms else "Not specified"
        
        return f"""DIAGNOSIS CONTEXT:
- Crop: {self.crop_type.capitalize()}
- Disease: {self.disease_name}
- Confidence: {self.confidence * 100:.1f}%
- Severity: {self.severity_level}
- Key Symptoms: {symptoms_str}
- Treatment: {self.treatment_summary}
- Estimated Cost: {self.cost_estimate}"""
    
    def is_valid(self) -> bool:
        """Check if diagnosis context has valid data."""
        return self.image_analyzed and self.disease_name and self.crop_type


# =============================================================================
# USER SESSION DATACLASS
# =============================================================================

@dataclass
class UserSession:
    """
    Represents a complete user session.
    
    Contains:
    - Session identification
    - Language preference
    - Current diagnosis context
    - Chat history
    - Timestamps for lifecycle management
    """
    # Session identification
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # User preferences
    language: str = "en"
    
    # Diagnosis context
    diagnosis: DiagnosisContext = field(default_factory=DiagnosisContext)
    
    # Chat history
    chat_history: List[ChatMessage] = field(default_factory=list)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "language": self.language,
            "diagnosis": self.diagnosis.to_dict(),
            "chat_history": [msg.to_dict() for msg in self.chat_history],
            "created_at": self.created_at,
            "last_accessed": self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserSession':
        """Create UserSession from dictionary."""
        session = cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            language=data.get("language", "en"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_accessed=data.get("last_accessed", time.time())
        )
        
        # Load diagnosis
        if "diagnosis" in data:
            session.diagnosis = DiagnosisContext.from_dict(data["diagnosis"])
        
        # Load chat history
        if "chat_history" in data:
            session.chat_history = [
                ChatMessage.from_dict(msg) for msg in data["chat_history"]
            ]
        
        return session
    
    def add_message(self, role: str, content: str) -> ChatMessage:
        """
        Add a message to chat history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message text
            
        Returns:
            The created ChatMessage
        """
        message = ChatMessage(
            role=role,
            content=content,
            language=self.language
        )
        self.chat_history.append(message)
        self.touch()  # Update last accessed
        return message
    
    def get_chat_history_for_prompt(self, max_messages: int = 10) -> str:
        """
        Get recent chat history formatted for LLM prompt.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted string of recent conversation
        """
        if not self.chat_history:
            return ""
        
        recent = self.chat_history[-max_messages:]
        formatted = []
        
        for msg in recent:
            role_label = "Farmer" if msg.role == "user" else "FarmEyes"
            formatted.append(f"{role_label}: {msg.content}")
        
        return "\n".join(formatted)
    
    def clear_chat_history(self):
        """Clear all chat messages but keep diagnosis context."""
        self.chat_history = []
        self.touch()
    
    def clear_diagnosis(self):
        """Clear diagnosis context and chat history."""
        self.diagnosis = DiagnosisContext()
        self.chat_history = []
        self.touch()
    
    def touch(self):
        """Update last accessed timestamp."""
        self.last_accessed = time.time()
    
    def is_expired(self, lifetime_seconds: int) -> bool:
        """
        Check if session has expired.
        
        Args:
            lifetime_seconds: Session lifetime in seconds
            
        Returns:
            True if session has expired
        """
        return (time.time() - self.last_accessed) > lifetime_seconds


# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    """
    Thread-safe session manager for FarmEyes application.
    
    Handles:
    - Session creation and retrieval
    - Diagnosis context management
    - Chat history management
    - Automatic cleanup of expired sessions
    
    Usage:
        manager = SessionManager()
        session = manager.create_session("ha")  # Create with Hausa
        manager.update_diagnosis(session.session_id, diagnosis_report)
        manager.add_chat_message(session.session_id, "user", "How to treat?")
    """
    
    def __init__(
        self,
        session_lifetime: int = 3600,
        max_sessions: int = 1000,
        max_chat_history: int = 50,
        cleanup_interval: int = 300
    ):
        """
        Initialize the session manager.
        
        Args:
            session_lifetime: Session lifetime in seconds (default: 1 hour)
            max_sessions: Maximum number of concurrent sessions
            max_chat_history: Maximum messages per session
            cleanup_interval: Interval for cleanup checks in seconds
        """
        self._sessions: Dict[str, UserSession] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Configuration
        self.session_lifetime = session_lifetime
        self.max_sessions = max_sessions
        self.max_chat_history = max_chat_history
        self.cleanup_interval = cleanup_interval
        
        # Cleanup tracking
        self._last_cleanup = time.time()
        
        logger.info(f"SessionManager initialized: lifetime={session_lifetime}s, max={max_sessions}")
    
    # =========================================================================
    # SESSION LIFECYCLE
    # =========================================================================
    
    def create_session(self, language: str = "en") -> UserSession:
        """
        Create a new user session.
        
        Args:
            language: Initial language preference
            
        Returns:
            New UserSession object
        """
        with self._lock:
            # Run cleanup if needed
            self._maybe_cleanup()
            
            # Check session limit
            if len(self._sessions) >= self.max_sessions:
                self._force_cleanup()
            
            # Create new session
            session = UserSession(language=language)
            self._sessions[session.session_id] = session
            
            logger.info(f"Session created: {session.session_id[:8]}... (lang={language})")
            return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            UserSession if found and not expired, None otherwise
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if session is None:
                return None
            
            # Check if expired
            if session.is_expired(self.session_lifetime):
                self._remove_session(session_id)
                return None
            
            # Update access time
            session.touch()
            return session
    
    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        language: str = "en"
    ) -> UserSession:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Optional session ID to retrieve
            language: Language for new session if created
            
        Returns:
            Existing or new UserSession
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(language)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            True if session was deleted, False if not found
        """
        with self._lock:
            return self._remove_session(session_id)
    
    def _remove_session(self, session_id: str) -> bool:
        """Internal method to remove a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Session removed: {session_id[:8]}...")
            return True
        return False
    
    # =========================================================================
    # LANGUAGE MANAGEMENT
    # =========================================================================
    
    def set_language(self, session_id: str, language: str) -> bool:
        """
        Update session language preference.
        
        Args:
            session_id: The session ID
            language: New language code (en, ha, yo, ig)
            
        Returns:
            True if updated, False if session not found
        """
        session = self.get_session(session_id)
        if session:
            session.language = language
            logger.debug(f"Session {session_id[:8]}... language set to {language}")
            return True
        return False
    
    def get_language(self, session_id: str) -> str:
        """
        Get session language preference.
        
        Args:
            session_id: The session ID
            
        Returns:
            Language code or 'en' default
        """
        session = self.get_session(session_id)
        return session.language if session else "en"
    
    # =========================================================================
    # DIAGNOSIS MANAGEMENT
    # =========================================================================
    
    def update_diagnosis(
        self,
        session_id: str,
        diagnosis_context: DiagnosisContext
    ) -> bool:
        """
        Update diagnosis context for a session.
        
        Args:
            session_id: The session ID
            diagnosis_context: New diagnosis context
            
        Returns:
            True if updated, False if session not found
        """
        session = self.get_session(session_id)
        if session:
            session.diagnosis = diagnosis_context
            # Clear old chat history when new diagnosis is made
            session.chat_history = []
            logger.info(f"Session {session_id[:8]}... diagnosis updated: {diagnosis_context.disease_name}")
            return True
        return False
    
    def update_diagnosis_from_report(self, session_id: str, report) -> bool:
        """
        Update diagnosis from a DiagnosisReport object.
        
        Args:
            session_id: The session ID
            report: DiagnosisReport from diagnosis_generator
            
        Returns:
            True if updated, False if session not found
        """
        context = DiagnosisContext.from_diagnosis_report(report)
        return self.update_diagnosis(session_id, context)
    
    def get_diagnosis(self, session_id: str) -> Optional[DiagnosisContext]:
        """
        Get current diagnosis context for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            DiagnosisContext or None
        """
        session = self.get_session(session_id)
        return session.diagnosis if session else None
    
    def has_diagnosis(self, session_id: str) -> bool:
        """
        Check if session has a valid diagnosis.
        
        Args:
            session_id: The session ID
            
        Returns:
            True if valid diagnosis exists
        """
        session = self.get_session(session_id)
        return session is not None and session.diagnosis.is_valid()
    
    def clear_diagnosis(self, session_id: str) -> bool:
        """
        Clear diagnosis and chat history for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            True if cleared, False if session not found
        """
        session = self.get_session(session_id)
        if session:
            session.clear_diagnosis()
            logger.info(f"Session {session_id[:8]}... diagnosis cleared")
            return True
        return False
    
    # =========================================================================
    # CHAT MANAGEMENT
    # =========================================================================
    
    def add_chat_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> Optional[ChatMessage]:
        """
        Add a message to session chat history.
        
        Args:
            session_id: The session ID
            role: 'user' or 'assistant'
            content: Message text
            
        Returns:
            Created ChatMessage or None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        # Enforce chat history limit
        if len(session.chat_history) >= self.max_chat_history:
            # Remove oldest messages (keep recent context)
            session.chat_history = session.chat_history[-(self.max_chat_history - 1):]
        
        return session.add_message(role, content)
    
    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """
        Get chat history for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            List of ChatMessage objects
        """
        session = self.get_session(session_id)
        return session.chat_history if session else []
    
    def get_chat_context(self, session_id: str, max_messages: int = 10) -> str:
        """
        Get formatted chat context for LLM prompt.
        
        Args:
            session_id: The session ID
            max_messages: Maximum recent messages to include
            
        Returns:
            Formatted conversation string
        """
        session = self.get_session(session_id)
        if not session:
            return ""
        return session.get_chat_history_for_prompt(max_messages)
    
    def clear_chat_history(self, session_id: str) -> bool:
        """
        Clear chat history but keep diagnosis.
        
        Args:
            session_id: The session ID
            
        Returns:
            True if cleared, False if session not found
        """
        session = self.get_session(session_id)
        if session:
            session.clear_chat_history()
            return True
        return False
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def _maybe_cleanup(self):
        """Run cleanup if enough time has passed since last cleanup."""
        if (time.time() - self._last_cleanup) > self.cleanup_interval:
            self._cleanup_expired()
    
    def _cleanup_expired(self):
        """Remove all expired sessions."""
        expired = []
        
        for session_id, session in self._sessions.items():
            if session.is_expired(self.session_lifetime):
                expired.append(session_id)
        
        for session_id in expired:
            self._remove_session(session_id)
        
        self._last_cleanup = time.time()
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def _force_cleanup(self):
        """Force cleanup when session limit is reached."""
        # Remove oldest sessions until we're under 80% capacity
        target_count = int(self.max_sessions * 0.8)
        
        # Sort by last accessed time
        sorted_sessions = sorted(
            self._sessions.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest
        while len(self._sessions) > target_count and sorted_sessions:
            session_id, _ = sorted_sessions.pop(0)
            self._remove_session(session_id)
        
        logger.warning(f"Force cleanup: reduced sessions to {len(self._sessions)}")
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get session manager statistics.
        
        Returns:
            Dictionary with stats
        """
        with self._lock:
            active_count = 0
            with_diagnosis = 0
            total_messages = 0
            
            for session in self._sessions.values():
                if not session.is_expired(self.session_lifetime):
                    active_count += 1
                    if session.diagnosis.is_valid():
                        with_diagnosis += 1
                    total_messages += len(session.chat_history)
            
            return {
                "total_sessions": len(self._sessions),
                "active_sessions": active_count,
                "sessions_with_diagnosis": with_diagnosis,
                "total_chat_messages": total_messages,
                "max_sessions": self.max_sessions,
                "session_lifetime_seconds": self.session_lifetime
            }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get the singleton SessionManager instance.
    
    Returns:
        SessionManager instance
    """
    global _session_manager
    
    if _session_manager is None:
        # Import config here to avoid circular imports
        try:
            from config import session_config
            _session_manager = SessionManager(
                session_lifetime=session_config.session_lifetime,
                max_sessions=session_config.max_sessions,
                max_chat_history=session_config.max_chat_history,
                cleanup_interval=session_config.cleanup_interval
            )
        except ImportError:
            # Use defaults if config not available
            _session_manager = SessionManager()
    
    return _session_manager


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_session(language: str = "en") -> UserSession:
    """Create a new session."""
    return get_session_manager().create_session(language)


def get_session(session_id: str) -> Optional[UserSession]:
    """Get a session by ID."""
    return get_session_manager().get_session(session_id)


def get_or_create_session(
    session_id: Optional[str] = None,
    language: str = "en"
) -> UserSession:
    """Get existing session or create new one."""
    return get_session_manager().get_or_create_session(session_id, language)


# =============================================================================
# MAIN - Test the session manager
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Session Manager Test")
    print("=" * 60)
    
    # Create manager
    manager = SessionManager(session_lifetime=60)  # 1 minute for testing
    
    # Test session creation
    print("\n1. Creating sessions...")
    session1 = manager.create_session("en")
    session2 = manager.create_session("ha")
    print(f"   Session 1: {session1.session_id[:8]}... (en)")
    print(f"   Session 2: {session2.session_id[:8]}... (ha)")
    
    # Test diagnosis update
    print("\n2. Updating diagnosis...")
    diagnosis = DiagnosisContext(
        disease_key="cassava_mosaic_virus",
        disease_name="Cassava Mosaic Virus",
        crop_type="cassava",
        confidence=0.92,
        severity_level="high",
        symptoms=["Yellow patches", "Leaf curling"],
        treatment_summary="Remove infected plants, use resistant varieties",
        image_analyzed=True
    )
    manager.update_diagnosis(session1.session_id, diagnosis)
    print(f"   Diagnosis set: {diagnosis.disease_name}")
    
    # Test chat messages
    print("\n3. Adding chat messages...")
    manager.add_chat_message(session1.session_id, "user", "How do I treat this?")
    manager.add_chat_message(session1.session_id, "assistant", "First, remove infected plants...")
    print(f"   Messages added: {len(manager.get_chat_history(session1.session_id))}")
    
    # Test context retrieval
    print("\n4. Getting context...")
    print(f"   Diagnosis context:\n{manager.get_diagnosis(session1.session_id).get_context_string()}")
    
    # Test stats
    print("\n5. Manager stats:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Session Manager test completed!")
    print("=" * 60)
