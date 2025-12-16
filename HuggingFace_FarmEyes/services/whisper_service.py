"""
FarmEyes Whisper Service
========================
Speech-to-text service using OpenAI Whisper for voice input.

Features:
- Supports multiple audio formats (wav, mp3, m4a, ogg, flac, webm)
- Optimized for Nigerian languages (Hausa, Yoruba, Igbo, English)
- Works offline after model download
- Automatic audio preprocessing
- Memory-efficient processing

Pipeline: Voice → Whisper → Text → N-ATLaS → Response
"""

import sys
import os
import io
import tempfile
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

class AudioProcessor:
    """
    Audio preprocessing utilities for Whisper.
    
    Handles:
    - Format validation
    - File size checks
    - Temporary file management
    """
    
    # Whisper expects 16kHz sample rate
    TARGET_SAMPLE_RATE = 16000
    
    # Maximum audio duration (seconds)
    MAX_DURATION = 30
    
    # Supported formats
    SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
    
    @classmethod
    def validate_file(cls, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, message)
        """
        path = Path(file_path)
        
        # Check existence
        if not path.exists():
            return False, "Audio file not found"
        
        # Check format
        if path.suffix.lower() not in cls.SUPPORTED_FORMATS:
            return False, f"Unsupported format. Supported: {', '.join(cls.SUPPORTED_FORMATS)}"
        
        # Check file size (5MB max)
        max_size = 5 * 1024 * 1024
        if path.stat().st_size > max_size:
            return False, f"File too large. Maximum size: {max_size // (1024*1024)}MB"
        
        return True, "Valid"
    
    @classmethod
    def validate_bytes(cls, audio_bytes: bytes, filename: str = "audio.wav") -> Tuple[bool, str]:
        """
        Validate audio bytes.
        
        Args:
            audio_bytes: Raw audio data
            filename: Original filename for format detection
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not audio_bytes:
            return False, "No audio data provided"
        
        # Check size
        max_size = 5 * 1024 * 1024
        if len(audio_bytes) > max_size:
            return False, f"Audio too large. Maximum size: {max_size // (1024*1024)}MB"
        
        # Check format from filename
        ext = Path(filename).suffix.lower()
        if ext and ext not in cls.SUPPORTED_FORMATS:
            return False, f"Unsupported format. Supported: {', '.join(cls.SUPPORTED_FORMATS)}"
        
        return True, "Valid"
    
    @classmethod
    def save_temp_file(cls, audio_bytes: bytes, suffix: str = ".wav") -> str:
        """
        Save audio bytes to temporary file.
        
        Args:
            audio_bytes: Raw audio data
            suffix: File extension
            
        Returns:
            Path to temporary file
        """
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(audio_bytes)
            return temp_path
        except Exception as e:
            os.unlink(temp_path)
            raise e
    
    @classmethod
    def cleanup_temp_file(cls, file_path: str):
        """
        Clean up temporary file.
        
        Args:
            file_path: Path to temporary file
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file: {e}")


# =============================================================================
# WHISPER SERVICE
# =============================================================================

class WhisperService:
    """
    Speech-to-text service using OpenAI Whisper.
    
    Optimized for:
    - Nigerian languages (Hausa, Yoruba, Igbo)
    - Agricultural terminology
    - Mobile/web audio recording quality
    
    Usage:
        service = WhisperService()
        result = service.transcribe(audio_bytes)
        text = result["text"]
    """
    
    # Language hints for Whisper
    LANGUAGE_HINTS = {
        "en": "english",
        "ha": "hausa",
        "yo": "yoruba",
        "ig": "igbo"
    }
    
    # Model sizes
    MODEL_SIZES = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large"
    }
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        download_root: Optional[str] = None
    ):
        """
        Initialize Whisper service.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Compute device (cpu, cuda)
            download_root: Custom download directory for model
        """
        self.model_size = model_size if model_size in self.MODEL_SIZES else "base"
        self.device = device
        self.download_root = download_root
        
        self._model = None
        self._is_loaded = False
        
        logger.info(f"WhisperService initialized: model={self.model_size}, device={self.device}")
    
    # =========================================================================
    # MODEL MANAGEMENT
    # =========================================================================
    
    def load_model(self) -> bool:
        """
        Load Whisper model into memory.
        
        Returns:
            True if model loaded successfully
        """
        if self._is_loaded:
            return True
        
        try:
            import whisper
            
            logger.info(f"Loading Whisper {self.model_size} model...")
            start_time = time.time()
            
            # Load model
            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=self.download_root
            )
            
            elapsed = time.time() - start_time
            self._is_loaded = True
            
            logger.info(f"✅ Whisper model loaded in {elapsed:.2f}s")
            return True
            
        except ImportError:
            logger.error("Whisper not installed! Run: pip install openai-whisper")
            return False
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False
    
    def unload_model(self):
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            logger.info("Whisper model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def ensure_loaded(self) -> bool:
        """Ensure model is loaded before use."""
        if not self._is_loaded:
            return self.load_model()
        return True
    
    # =========================================================================
    # TRANSCRIPTION
    # =========================================================================
    
    def transcribe(
        self,
        audio: Union[str, bytes, Path],
        language_hint: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio file path or bytes
            language_hint: Optional language hint (en, ha, yo, ig)
            task: "transcribe" or "translate" (translate to English)
            
        Returns:
            Dict with transcription result:
            {
                "success": bool,
                "text": str,
                "language": str (detected),
                "duration": float (seconds),
                "confidence": float (0-1)
            }
        """
        if not self.ensure_loaded():
            return {
                "success": False,
                "text": "",
                "error": "Model not loaded",
                "language": None,
                "duration": 0,
                "confidence": 0
            }
        
        temp_file = None
        
        try:
            # Handle input type
            if isinstance(audio, bytes):
                temp_file = AudioProcessor.save_temp_file(audio)
                audio_path = temp_file
            elif isinstance(audio, (str, Path)):
                audio_path = str(audio)
            else:
                raise ValueError("Audio must be file path or bytes")
            
            # Validate file
            is_valid, message = AudioProcessor.validate_file(audio_path)
            if not is_valid:
                return {
                    "success": False,
                    "text": "",
                    "error": message,
                    "language": None,
                    "duration": 0,
                    "confidence": 0
                }
            
            # Prepare transcription options
            options = {
                "task": task,
                "fp16": False,  # Use FP32 for CPU compatibility
            }
            
            # Add language hint if provided
            if language_hint and language_hint in self.LANGUAGE_HINTS:
                options["language"] = self.LANGUAGE_HINTS[language_hint]
            
            # Transcribe
            logger.info(f"Transcribing audio: {audio_path}")
            start_time = time.time()
            
            result = self._model.transcribe(audio_path, **options)
            
            elapsed = time.time() - start_time
            logger.info(f"Transcription completed in {elapsed:.2f}s")
            
            # Extract text
            text = result.get("text", "").strip()
            detected_language = result.get("language", "unknown")
            
            # Calculate rough confidence from segments
            segments = result.get("segments", [])
            if segments:
                avg_confidence = sum(
                    s.get("no_speech_prob", 0) for s in segments
                ) / len(segments)
                confidence = 1.0 - avg_confidence
            else:
                confidence = 0.5
            
            # Get audio duration
            duration = 0
            if segments:
                duration = segments[-1].get("end", 0)
            
            return {
                "success": True,
                "text": text,
                "language": detected_language,
                "duration": duration,
                "confidence": min(1.0, max(0.0, confidence)),
                "processing_time": elapsed
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e),
                "language": None,
                "duration": 0,
                "confidence": 0
            }
            
        finally:
            if temp_file:
                AudioProcessor.cleanup_temp_file(temp_file)
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_hint: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_bytes: Raw audio data
            filename: Original filename (for format detection)
            language_hint: Optional language hint
            
        Returns:
            Transcription result dict
        """
        is_valid, message = AudioProcessor.validate_bytes(audio_bytes, filename)
        if not is_valid:
            return {
                "success": False,
                "text": "",
                "error": message,
                "language": None,
                "duration": 0,
                "confidence": 0
            }
        
        ext = Path(filename).suffix.lower() or ".wav"
        
        temp_file = None
        try:
            temp_file = AudioProcessor.save_temp_file(audio_bytes, suffix=ext)
            return self.transcribe(temp_file, language_hint=language_hint)
        finally:
            if temp_file:
                AudioProcessor.cleanup_temp_file(temp_file)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def detect_language(self, audio: Union[str, bytes, Path]) -> Dict:
        """
        Detect language in audio without full transcription.
        
        Args:
            audio: Audio file path or bytes
            
        Returns:
            Dict with detected language info
        """
        if not self.ensure_loaded():
            return {
                "success": False,
                "language": None,
                "error": "Model not loaded"
            }
        
        temp_file = None
        
        try:
            if isinstance(audio, bytes):
                temp_file = AudioProcessor.save_temp_file(audio)
                audio_path = temp_file
            else:
                audio_path = str(audio)
            
            import whisper
            
            audio_data = whisper.load_audio(audio_path)
            audio_data = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio_data).to(self._model.device)
            _, probs = self._model.detect_language(mel)
            
            detected = max(probs, key=probs.get)
            confidence = probs[detected]
            
            return {
                "success": True,
                "language": detected,
                "confidence": confidence,
                "all_probabilities": dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                "success": False,
                "language": None,
                "error": str(e)
            }
            
        finally:
            if temp_file:
                AudioProcessor.cleanup_temp_file(temp_file)
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model."""
        info = {
            "model_size": self.model_size,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "supported_formats": list(AudioProcessor.SUPPORTED_FORMATS),
            "max_duration_seconds": AudioProcessor.MAX_DURATION
        }
        
        if self._is_loaded and self._model is not None:
            info["model_dims"] = {
                "n_mels": self._model.dims.n_mels,
                "n_audio_ctx": self._model.dims.n_audio_ctx,
                "n_audio_state": self._model.dims.n_audio_state,
                "n_text_ctx": self._model.dims.n_text_ctx,
                "n_text_state": self._model.dims.n_text_state,
            }
        
        return info


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_whisper_service: Optional[WhisperService] = None


def get_whisper_service() -> WhisperService:
    """
    Get the singleton WhisperService instance.
    
    Returns:
        WhisperService instance
    """
    global _whisper_service
    
    if _whisper_service is None:
        try:
            from config import whisper_config
            _whisper_service = WhisperService(
                model_size=whisper_config.model_size,
                device=whisper_config.device
            )
        except ImportError:
            _whisper_service = WhisperService()
    
    return _whisper_service


def unload_whisper_service():
    """Unload the Whisper service."""
    global _whisper_service
    
    if _whisper_service is not None:
        _whisper_service.unload_model()
        _whisper_service = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def transcribe_audio(
    audio: Union[str, bytes, Path],
    language_hint: Optional[str] = None
) -> Dict:
    """
    Transcribe audio to text.
    
    Args:
        audio: Audio file path or bytes
        language_hint: Optional language hint
        
    Returns:
        Transcription result dict
    """
    return get_whisper_service().transcribe(audio, language_hint=language_hint)


def transcribe_bytes(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language_hint: Optional[str] = None
) -> Dict:
    """
    Transcribe audio bytes to text.
    
    Args:
        audio_bytes: Raw audio data
        filename: Original filename
        language_hint: Optional language hint
        
    Returns:
        Transcription result dict
    """
    return get_whisper_service().transcribe_bytes(
        audio_bytes,
        filename=filename,
        language_hint=language_hint
    )


# =============================================================================
# MAIN - Test the service
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Whisper Service Test")
    print("=" * 60)
    
    # Initialize service
    print("\n1. Initializing WhisperService...")
    service = WhisperService(model_size="base", device="cpu")
    
    # Check model info
    print("\n2. Model info:")
    info = service.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test audio validation
    print("\n3. Testing audio validation...")
    test_cases = [
        ("test.wav", True),
        ("test.mp3", True),
        ("test.mp4", False),
        ("test.txt", False),
    ]
    for filename, expected in test_cases:
        _, ext = os.path.splitext(filename)
        is_supported = ext.lower() in AudioProcessor.SUPPORTED_FORMATS
        status = "✓" if is_supported == expected else "✗"
        print(f"   {status} {filename}: {'Supported' if is_supported else 'Not supported'}")
    
    print("\n4. Supported languages:")
    for code, name in WhisperService.LANGUAGE_HINTS.items():
        print(f"   {code}: {name}")
    
    print("\n" + "=" * 60)
    print("✅ Whisper Service test completed!")
    print("=" * 60)
