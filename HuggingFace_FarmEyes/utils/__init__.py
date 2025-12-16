"""
FarmEyes Utilities Package
==========================
Utility modules for the FarmEyes application.

Utilities:
- prompt_templates: N-ATLaS prompt templates for various tasks
"""

from utils.prompt_templates import (
    SystemPrompts,
    TranslationPrompts,
    DiagnosisPrompts,
    ConversationalPrompts,
    get_system_prompt,
    format_prompt_for_natlas,
    get_language_name,
    LANGUAGE_NAMES,
    LANGUAGE_NATIVE_NAMES
)

__all__ = [
    "SystemPrompts",
    "TranslationPrompts",
    "DiagnosisPrompts",
    "ConversationalPrompts",
    "get_system_prompt",
    "format_prompt_for_natlas",
    "get_language_name",
    "LANGUAGE_NAMES",
    "LANGUAGE_NATIVE_NAMES"
]
