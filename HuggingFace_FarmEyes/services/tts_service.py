"""
FarmEyes TTS Service
====================
Text-to-Speech service using Meta MMS-TTS via HuggingFace Transformers (local).

Supports:
- English (eng)
- Hausa (hau)
- Yoruba (yor)
- Igbo (ibo)

Pipeline: N-ATLaS Response → TTS → Audio Playback

NOTE: This uses LOCAL inference with transformers library.
The HuggingFace serverless API no longer reliably supports MMS-TTS models.
"""

import os
import io
import base64
import logging
import time
from typing import Optional, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class TTSConfig:
    """TTS Service Configuration"""
    
    # MMS-TTS Model IDs for each language
    MODELS = {
        "en": "facebook/mms-tts-eng",
        "eng": "facebook/mms-tts-eng",
        "ha": "facebook/mms-tts-hau",
        "hau": "facebook/mms-tts-hau",
        "yo": "facebook/mms-tts-yor",
        "yor": "facebook/mms-tts-yor",
        "ig": "facebook/mms-tts-ibo",
        "ibo": "facebook/mms-tts-ibo",
    }
    
    # Language display names
    LANGUAGE_NAMES = {
        "en": "English",
        "ha": "Hausa",
        "yo": "Yoruba",
        "ig": "Igbo",
    }
    
    # Request settings
    MAX_TEXT_LENGTH = 500  # characters (shorter for local inference)


# =============================================================================
# TTS SERVICE
# =============================================================================

class TTSService:
    """
    Text-to-Speech service using Meta MMS-TTS with local transformers.
    
    Features:
    - Supports English, Hausa, Yoruba, Igbo
    - Uses local transformers library (no API needed)
    - Returns base64 encoded audio
    - Lazy loading of models
    
    Usage:
        service = TTSService()
        result = service.synthesize("Hello world", "en")
        if result["success"]:
            audio_base64 = result["audio_base64"]
    """
    
    def __init__(self):
        """Initialize TTS service."""
        self.config = TTSConfig()
        self.models = {}  # Cache loaded models
        self.tokenizers = {}  # Cache tokenizers
        self._transformers_available = None
        logger.info("TTSService initialized (local transformers mode)")
    
    def _check_transformers(self) -> bool:
        """Check if transformers library is available."""
        if self._transformers_available is None:
            try:
                import transformers
                import torch
                import scipy
                import numpy
                self._transformers_available = True
                logger.info("Transformers library available")
            except ImportError as e:
                self._transformers_available = False
                logger.error(f"Transformers not available: {e}")
        return self._transformers_available
    
    def _load_model(self, language: str):
        """
        Load model and tokenizer for a language.
        Models are cached after first load.
        """
        if language in self.models:
            return self.models[language], self.tokenizers[language]
        
        model_id = self.config.MODELS.get(language.lower())
        if not model_id:
            raise ValueError(f"Unsupported language: {language}")
        
        try:
            from transformers import VitsModel, AutoTokenizer
            import torch
            
            logger.info(f"Loading TTS model: {model_id}")
            start_time = time.time()
            
            model = VitsModel.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Move to MPS if available (Apple Silicon)
            if torch.backends.mps.is_available():
                model = model.to("mps")
                logger.info("TTS model moved to MPS (Apple Silicon)")
            
            load_time = time.time() - start_time
            logger.info(f"TTS model loaded in {load_time:.2f}s")
            
            # Cache the model
            self.models[language] = model
            self.tokenizers[language] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise
    
    def get_model_id(self, language: str) -> Optional[str]:
        """Get the MMS-TTS model ID for a language."""
        return self.config.MODELS.get(language.lower())
    
    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language.lower() in self.config.MODELS
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages."""
        return self.config.LANGUAGE_NAMES.copy()
    
    def synthesize(
        self,
        text: str,
        language: str = "en"
    ) -> Dict:
        """
        Synthesize speech from text using local transformers.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, ha, yo, ig)
            
        Returns:
            Dictionary with:
            - success: bool
            - audio_base64: str (base64 encoded audio)
            - content_type: str (audio MIME type)
            - duration: float (estimated duration in seconds)
            - language: str
            - error: str (if failed)
        """
        start_time = time.time()
        
        # Check transformers availability
        if not self._check_transformers():
            return {
                "success": False,
                "error": "Transformers library not installed. Run: pip install transformers torch scipy",
                "language": language
            }
        
        # Validate input
        if not text or not text.strip():
            return {
                "success": False,
                "error": "No text provided",
                "language": language
            }
        
        # Normalize language code
        lang_key = language.lower()
        if lang_key not in self.config.MODELS:
            return {
                "success": False,
                "error": f"Language '{language}' is not supported for TTS",
                "language": language
            }
        
        # Truncate if too long
        text = text.strip()
        if len(text) > self.config.MAX_TEXT_LENGTH:
            logger.warning(f"Text truncated from {len(text)} to {self.config.MAX_TEXT_LENGTH} chars")
            # Try to break at sentence boundary
            truncated = text[:self.config.MAX_TEXT_LENGTH]
            for sep in [". ", "! ", "? ", ", "]:
                last_sep = truncated.rfind(sep)
                if last_sep > self.config.MAX_TEXT_LENGTH // 2:
                    truncated = truncated[:last_sep + 1]
                    break
            text = truncated.strip()
        
        try:
            import torch
            import scipy.io.wavfile
            import numpy as np
            
            # Load model
            model, tokenizer = self._load_model(lang_key)
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt")
            
            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate audio
            logger.info(f"Generating TTS for: {text[:50]}...")
            with torch.no_grad():
                output = model(**inputs).waveform
            
            # Move to CPU and convert to numpy
            waveform = output.squeeze().cpu().numpy()
            
            # Get sample rate from model config
            sample_rate = model.config.sampling_rate
            
            # Convert to 16-bit PCM
            waveform_int16 = (waveform * 32767).astype(np.int16)
            
            # Write to WAV bytes
            wav_buffer = io.BytesIO()
            scipy.io.wavfile.write(wav_buffer, sample_rate, waveform_int16)
            wav_buffer.seek(0)
            
            # Encode to base64
            audio_base64 = base64.b64encode(wav_buffer.read()).decode("utf-8")
            
            # Calculate duration
            duration = len(waveform) / sample_rate
            
            processing_time = time.time() - start_time
            logger.info(f"TTS success: {duration:.2f}s audio, {processing_time:.2f}s processing")
            
            return {
                "success": True,
                "audio_base64": audio_base64,
                "content_type": "audio/wav",
                "duration": duration,
                "language": language,
                "text_length": len(text),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "language": language
            }
    
    def unload_models(self):
        """Unload all cached models to free memory."""
        self.models.clear()
        self.tokenizers.clear()
        logger.info("TTS models unloaded")


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create the TTS service singleton."""
    global _tts_service
    
    if _tts_service is None:
        _tts_service = TTSService()
    
    return _tts_service


def synthesize_speech(text: str, language: str = "en") -> Dict:
    """Convenience function to synthesize speech."""
    return get_tts_service().synthesize(text, language)


def unload_tts_models():
    """Unload TTS models to free memory."""
    global _tts_service
    if _tts_service:
        _tts_service.unload_models()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TTS Service Test (Local Transformers)")
    print("=" * 60)
    
    service = TTSService()
    
    # Test supported languages
    print("\n1. Supported Languages:")
    for code, name in service.get_supported_languages().items():
        model = service.get_model_id(code)
        print(f"   {code}: {name} -> {model}")
    
    # Test synthesis
    print("\n2. Testing English TTS...")
    result = service.synthesize("Hello, this is a test of the text to speech system.", "en")
    if result["success"]:
        print(f"   ✅ Success! Audio duration: {result['duration']:.2f}s")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Audio size: {len(result['audio_base64'])} chars (base64)")
    else:
        print(f"   ❌ Failed: {result['error']}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
