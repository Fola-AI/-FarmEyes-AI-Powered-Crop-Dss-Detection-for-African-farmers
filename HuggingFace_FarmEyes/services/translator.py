"""
FarmEyes Translator Service – STABLE VERSION
===========================================
This module exposes ALL symbols expected by services/__init__.py.

Public API:
- TranslatorService
- TranslationCache
- get_translator
- translate_text
- translate_to_hausa / yoruba / igbo
- get_ui_text (proxy)
- SUPPORTED_LANGUAGES
- NATIVE_LANGUAGE_NAMES
"""

from typing import List, Dict, Optional
import hashlib

from models.natlas_model import (
    get_natlas_model,
    NATIVE_LANGUAGE_NAMES,
)

# ------------------------------------------------------------------
# LANGUAGE CONSTANTS (PUBLIC)
# ------------------------------------------------------------------

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
}

# ------------------------------------------------------------------
# TRANSLATION CACHE (EXPECTED BY services/__init__.py)
# ------------------------------------------------------------------

class TranslationCache:
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, str] = {}
        self._max_size = max_size

    def _key(self, text: str, lang: str) -> str:
        h = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{lang}:{h}"

    def get(self, text: str, lang: str) -> Optional[str]:
        return self._cache.get(self._key(text, lang))

    def set(self, text: str, lang: str, value: str):
        if len(self._cache) >= self._max_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[self._key(text, lang)] = value

# ------------------------------------------------------------------
# TRANSLATOR SERVICE
# ------------------------------------------------------------------

class TranslatorService:
    def __init__(self, use_cache: bool = True):
        self.model = get_natlas_model()
        self.cache = TranslationCache() if use_cache else None

    def translate(self, text: str, lang: str) -> str:
        if lang == "en":
            return text

        if self.cache:
            cached = self.cache.get(text, lang)
            if cached:
                return cached

        result = self.model.translate(text, lang)

        if self.cache:
            self.cache.set(text, lang, result)

        return result

    def translate_batch(self, texts: List[str], lang: str) -> List[str]:
        return [self.translate(t, lang) for t in texts]

# ------------------------------------------------------------------
# SINGLETON
# ------------------------------------------------------------------

_service: Optional[TranslatorService] = None

def get_translator() -> TranslatorService:
    global _service
    if _service is None:
        _service = TranslatorService()
    return _service

# ------------------------------------------------------------------
# CONVENIENCE FUNCTIONS (PUBLIC)
# ------------------------------------------------------------------

def translate_text(text: str, target_language: str) -> str:
    return get_translator().translate(text, target_language)

def translate_to_hausa(text: str) -> str:
    return translate_text(text, "ha")

def translate_to_yoruba(text: str) -> str:
    return translate_text(text, "yo")

def translate_to_igbo(text: str) -> str:
    return translate_text(text, "ig")

def get_ui_text(key_path: str, language: str = "en") -> str:
    # UI text is already handled in app.py; keep proxy for compatibility
    return key_path
