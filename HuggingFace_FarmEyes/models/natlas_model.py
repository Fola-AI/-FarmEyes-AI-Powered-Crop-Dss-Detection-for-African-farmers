"""
FarmEyes N-ATLaS Model Integration (Transformers Version)
==========================================================
Uses the official N-ATLaS model via HuggingFace Transformers library.
NO llama-cpp-python required - faster builds, official model support.

Model: NCAIR1/N-ATLaS (8B parameters, Llama-3 based)
Size: ~16GB (downloaded at runtime)

Languages: English, Hausa, Yoruba, Igbo

Powered by Awarri Technologies and the Federal Ministry of 
Communications, Innovation and Digital Economy.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

IS_HF_SPACES = os.environ.get("SPACE_ID") is not None

# Check for GPU
HAS_GPU = False
GPU_NAME = "None"
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU:
        GPU_NAME = torch.cuda.get_device_name(0)
        logger.info(f"🎮 GPU detected: {GPU_NAME}")
    else:
        logger.info("🖥️ No GPU detected - using CPU")
except ImportError:
    logger.warning("PyTorch not installed")

if IS_HF_SPACES:
    logger.info("🤗 Running on HuggingFace Spaces")
else:
    logger.info("🖥️ Running locally")


# =============================================================================
# LANGUAGE MAPPINGS
# =============================================================================

LANGUAGE_NAMES = {
    "en": "English",
    "ha": "Hausa", 
    "yo": "Yoruba",
    "ig": "Igbo"
}

NATIVE_LANGUAGE_NAMES = {
    "en": "English",
    "ha": "Yaren Hausa",
    "yo": "Èdè Yorùbá",
    "ig": "Asụsụ Igbo"
}


# =============================================================================
# HUGGINGFACE API CLIENT (COMPATIBILITY STUB)
# =============================================================================

class HuggingFaceAPIClient:
    """
    Compatibility stub for HuggingFaceAPIClient.
    
    The main N-ATLaS model now uses transformers directly, so this class
    is kept for backwards compatibility with other parts of the codebase.
    """
    
    MODEL_ID = "NCAIR1/N-ATLaS"
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self._is_available = False
        
        if self.api_token:
            logger.info("✅ HuggingFace API token found (using transformers backend)")
        else:
            logger.warning("⚠️ No HF_TOKEN set")
    
    def is_available(self) -> bool:
        """API client is deprecated - using transformers instead."""
        return False
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> Optional[str]:
        """Deprecated - use NATLaSModel instead."""
        logger.warning("HuggingFaceAPIClient.generate() is deprecated. Use NATLaSModel instead.")
        return None
    
    def translate(self, text: str, target_language: str) -> Optional[str]:
        """Deprecated - use NATLaSModel instead."""
        logger.warning("HuggingFaceAPIClient.translate() is deprecated. Use NATLaSModel instead.")
        return None


# =============================================================================
# LOCAL GGUF MODEL (COMPATIBILITY STUB)
# =============================================================================

class LocalGGUFModel:
    """
    Compatibility stub for LocalGGUFModel.
    
    The main N-ATLaS model now uses transformers directly, so this class
    is kept for backwards compatibility with other parts of the codebase.
    """
    
    def __init__(self, **kwargs):
        self._is_loaded = False
        logger.info("LocalGGUFModel stub initialized (using transformers backend)")
    
    @property
    def is_loaded(self) -> bool:
        return False
    
    def load_model(self) -> bool:
        logger.warning("LocalGGUFModel is deprecated. Using transformers backend.")
        return False
    
    def unload_model(self):
        pass
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        return None
    
    def translate(self, text: str, target_language: str) -> Optional[str]:
        return None


# =============================================================================
# N-ATLAS MODEL (TRANSFORMERS VERSION)
# =============================================================================

class NATLaSTransformersModel:
    """
    N-ATLaS model using HuggingFace Transformers.
    
    This is the OFFICIAL way to use N-ATLaS as shown in the model documentation.
    No llama-cpp-python compilation required!
    
    Model: NCAIR1/N-ATLaS
    Base: Llama-3 8B
    Size: ~16GB
    """
    
    MODEL_ID = "NCAIR1/N-ATLaS"
    
    def __init__(
        self,
        model_id: str = MODEL_ID,
        torch_dtype: str = "float16",
        device_map: str = "auto",
        load_on_init: bool = True
    ):
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        
        logger.info(f"NATLaS Config: model={model_id}, dtype={torch_dtype}, device_map={device_map}")
        
        if load_on_init:
            self.load_model()
    
    def load_model(self) -> bool:
        """Load N-ATLaS model using transformers."""
        if self._is_loaded:
            return True
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info("=" * 60)
            logger.info("📥 LOADING N-ATLaS MODEL")
            logger.info("=" * 60)
            logger.info(f"   Model: {self.model_id}")
            logger.info(f"   Size: ~16GB")
            logger.info("   This may take 5-15 minutes on first load...")
            logger.info("=" * 60)
            
            # Determine torch dtype
            if self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model
            logger.info("Loading model weights...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=self.device_map,
                trust_remote_code=True
            )
            
            self._is_loaded = True
            
            logger.info("=" * 60)
            logger.info("✅ N-ATLaS MODEL LOADED SUCCESSFULLY!")
            if HAS_GPU:
                logger.info(f"   Running on GPU: {GPU_NAME}")
            else:
                logger.info("   Running on CPU")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load N-ATLaS model: {e}")
            logger.error("   Make sure you have accepted the model license at:")
            logger.error("   https://huggingface.co/NCAIR1/N-ATLaS")
            return False
    
    def unload_model(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._is_loaded = False
        
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages using the tokenizer's chat template."""
        try:
            current_date = datetime.now().strftime('%d %b %Y')
            text = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                date_string=current_date
            )
            return text
        except Exception as e:
            # Fallback formatting if chat template fails
            logger.warning(f"Chat template failed, using fallback: {e}")
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "user":
                    text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "assistant":
                    text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
            text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            return text
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.12
    ) -> Optional[str]:
        """Generate text using N-ATLaS model."""
        if not self._is_loaded:
            if not self.load_model():
                return None
        
        try:
            import torch
            
            # Default system prompt
            if system_prompt is None:
                system_prompt = (
                    "You are a helpful AI assistant for African farmers. "
                    "You help with crop disease diagnosis, treatment advice, and agricultural questions. "
                    "Respond in the same language the user writes in."
                )
            
            # Format messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            text = self._format_messages(messages)
            
            # Tokenize
            input_tokens = self._tokenizer(
                text,
                return_tensors='pt',
                add_special_tokens=False
            )
            
            # Move to device
            if HAS_GPU:
                input_tokens = input_tokens.to('cuda')
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **input_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Decode
            full_response = self._tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract assistant response
            # Look for the last assistant header and get text after it
            assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
            if assistant_marker in full_response:
                response = full_response.split(assistant_marker)[-1]
            else:
                response = full_response
            
            # Clean up special tokens
            for token in ["<|eot_id|>", "<|end_of_text|>", "<|begin_of_text|>", 
                         "<|start_header_id|>", "<|end_header_id|>"]:
                response = response.replace(token, "")
            
            response = response.strip()
            
            if response:
                logger.info(f"✅ Generation successful: {len(response)} chars")
                return response
            else:
                logger.warning("⚠️ Empty response generated")
                return None
                
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return None
    
    def translate(self, text: str, target_language: str) -> Optional[str]:
        """Translate text to target language."""
        if target_language == "en" or not text:
            return text
        
        lang_name = LANGUAGE_NAMES.get(target_language, target_language)
        
        prompt = f"Translate the following text to {lang_name}. Only provide the translation, nothing else.\n\nText: {text}"
        
        system_prompt = f"You are a professional translator. Translate text accurately to {lang_name}. Only output the translation."
        
        result = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=len(text) * 4,
            temperature=0.3,
            repetition_penalty=1.1
        )
        
        if result:
            result = result.strip()
            # Clean up common prefixes
            prefixes_to_remove = [
                f"{lang_name}:",
                f"{lang_name} translation:",
                "Translation:",
                "Here is the translation:",
                "The translation is:",
            ]
            for prefix in prefixes_to_remove:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix):].strip()
            return result
        
        return None
    
    def chat_response(self, message: str, context: Dict, language: str = "en") -> Optional[str]:
        """Generate chat response with diagnosis context."""
        crop = context.get("crop_type", "crop").capitalize()
        disease = context.get("disease_name", "unknown disease")
        severity = context.get("severity_level", "unknown")
        confidence = context.get("confidence", 0)
        if confidence <= 1:
            confidence = int(confidence * 100)
        
        # Language instruction
        lang_instructions = {
            "en": "Respond in English.",
            "ha": "Respond in Hausa language (Yaren Hausa).",
            "yo": "Respond in Yoruba language (Èdè Yorùbá).",
            "ig": "Respond in Igbo language (Asụsụ Igbo)."
        }
        lang_instruction = lang_instructions.get(language, "Respond in English.")
        
        system_prompt = (
            "You are FarmEyes, an AI assistant helping African farmers with crop diseases. "
            "You provide practical, helpful advice about crop diseases and farming. "
            f"{lang_instruction}"
        )
        
        prompt = (
            f"Current diagnosis information:\n"
            f"- Crop: {crop}\n"
            f"- Disease: {disease}\n"
            f"- Severity: {severity}\n"
            f"- Confidence: {confidence}%\n\n"
            f"Farmer's question: {message}\n\n"
            f"Provide a helpful, practical response about this disease or related farming advice. "
            f"Keep your response concise (2-3 paragraphs maximum)."
        )
        
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=500,
            temperature=0.7
        )


# =============================================================================
# HYBRID N-ATLAS MODEL (MAIN CLASS)
# =============================================================================

class NATLaSModel:
    """
    N-ATLaS model wrapper.
    
    Uses the official NCAIR1/N-ATLaS model via HuggingFace Transformers.
    This is the recommended way to use N-ATLaS.
    """
    
    def __init__(
        self,
        api_token: Optional[str] = None,  # Kept for compatibility
        auto_load: bool = True,
        **kwargs
    ):
        # Get HF token from environment
        self.hf_token = api_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        if self.hf_token:
            logger.info("✅ HuggingFace token found")
            # Set token for huggingface_hub
            try:
                from huggingface_hub import login
                login(token=self.hf_token, add_to_git_credential=False)
            except Exception as e:
                logger.warning(f"Could not set HF token: {e}")
        else:
            logger.warning("⚠️ No HF_TOKEN found - model access may fail")
        
        # Initialize the transformers model
        self.model = NATLaSTransformersModel(load_on_init=auto_load)
        
        # Translation cache
        self._cache: Dict[str, str] = {}
        
        logger.info("=" * 60)
        logger.info("✅ NATLaSModel initialized")
        logger.info(f"   Model loaded: {'Yes' if self.model.is_loaded else 'No'}")
        logger.info(f"   GPU available: {'Yes - ' + GPU_NAME if HAS_GPU else 'No'}")
        logger.info(f"   HF Token: {'Yes' if self.hf_token else 'No'}")
        logger.info(f"   Running on: {'HuggingFace Spaces' if IS_HF_SPACES else 'Local'}")
        logger.info("=" * 60)
    
    @property
    def is_loaded(self) -> bool:
        return self.model.is_loaded
    
    def load_model(self) -> bool:
        return self.model.load_model()
    
    def translate(self, text: str, target_language: str, use_cache: bool = True) -> str:
        """Translate text to target language."""
        if target_language == "en" or not text or not text.strip():
            return text
        
        # Check cache
        cache_key = f"{target_language}:{hash(text)}"
        if use_cache and cache_key in self._cache:
            logger.info("📦 Using cached translation")
            return self._cache[cache_key]
        
        logger.info(f"🌍 Translating to {LANGUAGE_NAMES.get(target_language, target_language)}...")
        result = self.model.translate(text, target_language)
        
        if result and result != text:
            # Cache the result
            if use_cache:
                self._cache[cache_key] = result
                # Limit cache size
                if len(self._cache) > 500:
                    keys = list(self._cache.keys())[:100]
                    for k in keys:
                        del self._cache[k]
            logger.info("✅ Translation successful")
            return result
        
        logger.warning("⚠️ Translation failed - returning original")
        return text
    
    def translate_batch(self, texts: List[str], target_language: str) -> List[str]:
        """Translate multiple texts."""
        return [self.translate(text, target_language) for text in texts]
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs) -> str:
        """Generate text."""
        result = self.model.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        return result if result else ""
    
    def chat_response(self, message: str, context: Dict, language: str = "en") -> str:
        """Generate chat response with context."""
        result = self.model.chat_response(message, context, language)
        if result:
            return result
        return "I'm sorry, I couldn't generate a response. Please try again."
    
    def load_local_model(self) -> bool:
        """Compatibility method."""
        return self.load_model()
    
    def unload_local_model(self):
        """Unload model."""
        self.model.unload_model()
    
    def get_status(self) -> Dict:
        return {
            "model_loaded": self.model.is_loaded,
            "model_id": self.model.model_id,
            "gpu_available": HAS_GPU,
            "gpu_name": GPU_NAME if HAS_GPU else None,
            "hf_token_set": bool(self.hf_token),
            "cache_size": len(self._cache),
            "running_on": "HuggingFace Spaces" if IS_HF_SPACES else "Local"
        }
    
    def clear_cache(self):
        self._cache.clear()


# =============================================================================
# SINGLETON
# =============================================================================

_model_instance: Optional[NATLaSModel] = None


def get_natlas_model(
    api_token: Optional[str] = None,
    auto_load_local: bool = True,
    **kwargs
) -> NATLaSModel:
    """Get singleton model instance."""
    global _model_instance
    
    if _model_instance is None:
        _model_instance = NATLaSModel(
            api_token=api_token,
            auto_load=auto_load_local,
            **kwargs
        )
    
    return _model_instance


def unload_natlas_model():
    """Unload model."""
    global _model_instance
    if _model_instance is not None:
        _model_instance.unload_local_model()
        _model_instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def translate_text(text: str, target_language: str) -> str:
    return get_natlas_model().translate(text, target_language)


def translate_batch(texts: List[str], target_language: str) -> List[str]:
    return get_natlas_model().translate_batch(texts, target_language)


def generate_text(prompt: str, max_tokens: int = 512) -> str:
    return get_natlas_model().generate(prompt, max_tokens=max_tokens)
