"""
FarmEyes Configuration File
===========================
Central configuration for the FarmEyes crop disease detection application.
Contains model paths, class mappings, device settings, API configurations,
session management, and Whisper speech-to-text settings.

Device: Apple Silicon M1 Pro with MPS (Metal Performance Shaders) acceleration
Deployment: Local development + HuggingFace Spaces

Model: Custom trained YOLOv11 for 6 disease classes
Crops: Cassava, Cocoa, Tomato
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# =============================================================================
# PATH CONFIGURATIONS
# =============================================================================

# Base project directory
BASE_DIR = Path(__file__).parent.resolve()

# Data directories
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"

# Create directories if they don't exist
for directory in [DATA_DIR, STATIC_DIR, MODELS_DIR, OUTPUTS_DIR, UPLOADS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Knowledge base and UI translations paths
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
UI_TRANSLATIONS_PATH = STATIC_DIR / "ui_translations.json"


# =============================================================================
# API CONFIGURATION
# =============================================================================

@dataclass
class APIConfig:
    """Configuration for FastAPI backend"""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 7860  # Default HuggingFace Spaces port
    
    # API metadata
    title: str = "FarmEyes API"
    description: str = "AI-Powered Crop Disease Detection for Nigerian Farmers"
    version: str = "2.0.0"
    
    # CORS settings (for frontend access)
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        "https://*.hf.space",  # HuggingFace Spaces
        "*"  # Allow all for development - restrict in production
    ])
    
    # Request limits
    max_upload_size: int = 10 * 1024 * 1024  # 10MB max image upload
    request_timeout: int = 60  # seconds
    
    # Rate limiting (basic)
    rate_limit_requests: int = 100  # requests per minute
    rate_limit_window: int = 60  # seconds
    
    # Debug mode
    debug: bool = os.environ.get("DEBUG", "false").lower() == "true"
    
    # Environment detection
    @property
    def is_huggingface(self) -> bool:
        """Check if running on HuggingFace Spaces"""
        return os.environ.get("SPACE_ID") is not None
    
    @property
    def base_url(self) -> str:
        """Get base URL based on environment"""
        if self.is_huggingface:
            space_id = os.environ.get("SPACE_ID", "")
            return f"https://{space_id.replace('/', '-')}.hf.space"
        return f"http://{self.host}:{self.port}"


# =============================================================================
# SESSION CONFIGURATION
# =============================================================================

@dataclass
class SessionConfig:
    """Configuration for session management"""
    
    # Session settings
    session_lifetime: int = 3600  # 1 hour in seconds
    max_sessions: int = 1000  # Maximum concurrent sessions
    
    # Chat history settings
    max_chat_history: int = 50  # Maximum messages per session
    max_message_length: int = 2000  # Maximum characters per message
    
    # Context retention
    retain_diagnosis: bool = True  # Keep diagnosis context in session
    
    # Cleanup settings
    cleanup_interval: int = 300  # 5 minutes - check for expired sessions


# =============================================================================
# WHISPER CONFIGURATION
# =============================================================================

@dataclass
class WhisperConfig:
    """Configuration for Whisper speech-to-text"""
    
    # Model settings
    model_size: str = "base"  # tiny, base, small, medium, large
    
    # Supported model sizes with approximate VRAM/RAM requirements
    # tiny:   ~1GB  - Fastest, least accurate
    # base:   ~1GB  - Good balance (SELECTED)
    # small:  ~2GB  - Better accuracy
    # medium: ~5GB  - High accuracy
    # large:  ~10GB - Best accuracy
    
    # Device settings
    device: str = "cpu"  # Use CPU for broader compatibility
    # Note: On Apple Silicon, Whisper runs well on CPU
    # For GPU: set to "cuda" (NVIDIA) or use mlx-whisper for Apple Silicon
    
    # Audio settings
    sample_rate: int = 16000  # Whisper expects 16kHz audio
    max_audio_duration: int = 30  # Maximum seconds of audio to process
    
    # Language settings - Whisper auto-detects but we can hint
    language_hints: Dict[str, str] = field(default_factory=lambda: {
        "en": "english",
        "ha": "hausa",
        "yo": "yoruba",
        "ig": "igbo"
    })
    
    # Transcription settings
    task: str = "transcribe"  # transcribe or translate
    
    # Performance settings
    fp16: bool = False  # Use FP32 for CPU compatibility
    
    # Supported audio formats
    supported_formats: List[str] = field(default_factory=lambda: [
        ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"
    ])
    
    # Maximum audio file size (5MB)
    max_file_size: int = 5 * 1024 * 1024


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

@dataclass
class YOLOConfig:
    """Configuration for YOLOv11 disease detection model"""
    
    # Path to trained YOLOv11 model weights (.pt file)
    model_path: Path = MODELS_DIR / "farmeyes_yolov11.pt"
    
    # Confidence threshold for detections (0.0 - 1.0)
    confidence_threshold: float = 0.5
    
    # IoU threshold for non-maximum suppression
    iou_threshold: float = 0.45
    
    # Input image size (YOLOv11 default)
    input_size: int = 640
    
    # Maximum number of detections per image
    max_detections: int = 10
    
    # Device for inference ('mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' for CPU)
    device: str = "mps"


@dataclass
class NATLaSConfig:
    """Configuration for N-ATLaS language model (GGUF format)"""
    
    # Hugging Face model repository
    hf_repo: str = "tosinamuda/N-ATLaS-GGUF"
    
    # GGUF model filename (Q4_K_M quantized version - smaller, faster)
    model_filename: str = "N-ATLaS-GGUF-Q4_K_M.gguf"
    
    # Local path where model will be downloaded/cached
    model_path: Path = MODELS_DIR / "natlas"
    
    # Full path to the GGUF file
    @property
    def gguf_path(self) -> Path:
        return self.model_path / self.model_filename
    
    # Context window size (tokens)
    context_length: int = 4096
    
    # Maximum tokens to generate in response
    max_tokens: int = 1024
    
    # Chat-specific max tokens (shorter for responsiveness)
    chat_max_tokens: int = 512
    
    # Temperature for text generation (0.0 = deterministic, 1.0 = creative)
    temperature: float = 0.7
    
    # Chat temperature (slightly lower for more focused responses)
    chat_temperature: float = 0.6
    
    # Top-p (nucleus) sampling
    top_p: float = 0.9
    
    # Number of GPU layers to offload (for MPS acceleration)
    # Set to -1 to offload all layers, 0 for CPU only
    n_gpu_layers: int = -1
    
    # Number of threads for CPU computation
    n_threads: int = 8
    
    # Batch size for prompt processing
    n_batch: int = 512
    
    # Device for inference
    device: str = "mps"


# =============================================================================
# DISEASE CLASS MAPPINGS (6 CLASSES - NO HEALTHY CLASSES)
# =============================================================================

# YOLOv11 class index to disease key mapping
CLASS_INDEX_TO_KEY: Dict[int, str] = {
    0: "cassava_bacterial_blight",
    1: "cassava_mosaic_virus",
    2: "cocoa_monilia_disease",
    3: "cocoa_phytophthora_disease",
    4: "tomato_gray_mold",
    5: "tomato_wilt_disease"
}

# Reverse mapping: disease key to class index
KEY_TO_CLASS_INDEX: Dict[str, int] = {v: k for k, v in CLASS_INDEX_TO_KEY.items()}

# Class names as they appear in YOLO training (6 classes)
CLASS_NAMES: List[str] = [
    "Cassava Bacteria Blight",      # Index 0
    "Cassava Mosaic Virus",         # Index 1
    "Cocoa Monilia Disease",        # Index 2
    "Cocoa Phytophthora Disease",   # Index 3
    "Tomato Gray Mold Disease",     # Index 4
    "Tomato Wilt Disease"           # Index 5
]

# No healthy class indices in 6-class model
HEALTHY_CLASS_INDICES: List[int] = []

# All class indices are disease classes
DISEASE_CLASS_INDICES: List[int] = [0, 1, 2, 3, 4, 5]

# Crop type mapping (6 classes only)
CROP_TYPES: Dict[str, List[int]] = {
    "cassava": [0, 1],
    "cocoa": [2, 3],
    "tomato": [4, 5]
}

# Reverse mapping: class index to crop type
CLASS_TO_CROP: Dict[int, str] = {}
for crop, indices in CROP_TYPES.items():
    for idx in indices:
        CLASS_TO_CROP[idx] = crop


# =============================================================================
# LANGUAGE CONFIGURATIONS
# =============================================================================

@dataclass
class LanguageConfig:
    """Configuration for supported languages"""
    
    # Supported language codes
    supported_languages: List[str] = field(default_factory=lambda: ["en", "ha", "yo", "ig"])
    
    # Default language
    default_language: str = "en"
    
    # Language display names
    language_names: Dict[str, str] = field(default_factory=lambda: {
        "en": "English",
        "ha": "Hausa",
        "yo": "Yorùbá",
        "ig": "Igbo"
    })
    
    # Language codes for N-ATLaS prompts
    language_full_names: Dict[str, str] = field(default_factory=lambda: {
        "en": "English",
        "ha": "Hausa",
        "yo": "Yoruba",
        "ig": "Igbo"
    })
    
    # Native language names (for display in selector)
    native_names: Dict[str, str] = field(default_factory=lambda: {
        "en": "English",
        "ha": "Hausa",
        "yo": "Yorùbá",
        "ig": "Asụsụ Igbo"
    })


# =============================================================================
# CHAT CONFIGURATION
# =============================================================================

@dataclass
class ChatConfig:
    """Configuration for contextual chatbot"""
    
    # System prompt for agricultural chat
    system_prompt: str = """You are FarmEyes, an AI agricultural assistant helping Nigerian farmers.
You are currently discussing a specific crop disease diagnosis with the farmer.
Your role is to:
1. Answer questions ONLY about the diagnosed disease and related agricultural topics
2. Provide practical, actionable advice in simple language
3. Use local context (Nigerian farming practices, costs in Naira)
4. Be respectful, patient, and supportive
5. If asked about unrelated topics, politely redirect to agricultural matters

IMPORTANT: Stay focused on the diagnosis context provided. Do not make up information.
If you don't know something, say so honestly and suggest consulting a local agricultural extension officer."""

    # Context template for chat
    context_template: str = """CURRENT DIAGNOSIS CONTEXT:
- Crop: {crop_type}
- Disease: {disease_name}
- Confidence: {confidence}%
- Severity: {severity}
- Key symptoms: {symptoms}
- Recommended treatment: {treatment_summary}

The farmer may ask follow-up questions about this diagnosis."""

    # Allowed topic keywords (for moderate context restriction)
    allowed_topics: List[str] = field(default_factory=lambda: [
        # Disease-related
        "disease", "infection", "symptom", "treatment", "cure", "prevention",
        "spread", "cause", "severity", "recovery",
        # Crop-related
        "crop", "plant", "leaf", "stem", "root", "fruit", "harvest", "yield",
        "cassava", "cocoa", "tomato", "farming", "agriculture",
        # Treatment-related
        "medicine", "chemical", "organic", "traditional", "spray", "apply",
        "fungicide", "pesticide", "fertilizer", "cost", "price", "naira",
        # General farming
        "farm", "field", "soil", "water", "weather", "season", "planting",
        "seed", "variety", "resistant", "healthy"
    ])
    
    # Response length limits
    max_response_tokens: int = 400
    
    # Welcome message template
    welcome_template: str = """Hello! I'm your FarmEyes assistant. I've analyzed your {crop_type} plant and detected {disease_name} with {confidence}% confidence.

I can help you understand:
• More about this disease and its symptoms
• Treatment options and costs
• Prevention methods
• When to seek expert help

What would you like to know?"""


# =============================================================================
# APPLICATION CONFIGURATIONS
# =============================================================================

@dataclass
class AppConfig:
    """General application configuration"""
    
    # App information
    app_name: str = "FarmEyes"
    app_version: str = "2.0.0"
    app_tagline: str = "AI-Powered Crop Disease Detection for Nigerian Farmers"
    
    # Server settings (legacy Gradio support)
    server_host: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    
    # Debug mode
    debug: bool = True
    
    # Maximum image file size (in bytes) - 10MB
    max_image_size: int = 10 * 1024 * 1024
    
    # Supported image formats
    supported_image_formats: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    ])
    
    # Confidence thresholds for user feedback
    high_confidence_threshold: float = 0.85
    medium_confidence_threshold: float = 0.60
    low_confidence_threshold: float = 0.40
    
    # Enable/disable features
    enable_voice_input: bool = True   # Voice input with Whisper
    enable_chat: bool = True          # Contextual chat
    enable_history: bool = True       # Session history


# =============================================================================
# DEVICE CONFIGURATION (Apple Silicon Specific)
# =============================================================================

@dataclass
class DeviceConfig:
    """Device and hardware configuration for Apple Silicon M1 Pro"""
    
    # Primary compute device
    compute_device: str = "mps"
    
    # Fallback device if primary is unavailable
    fallback_device: str = "cpu"
    
    # Enable MPS (Metal Performance Shaders) for PyTorch
    use_mps: bool = True
    
    # Memory management
    clear_cache_after_inference: bool = True
    
    @staticmethod
    def get_device() -> str:
        """Determine the best available device for computation."""
        import torch
        
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    @staticmethod
    def get_device_info() -> Dict[str, str]:
        """Get information about the current compute device."""
        import torch
        import platform
        
        info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "device": DeviceConfig.get_device()
        }
        
        if torch.backends.mps.is_available():
            info["mps_available"] = "Yes"
            info["mps_built"] = str(torch.backends.mps.is_built())
        
        return info


# =============================================================================
# PROMPT TEMPLATES CONFIGURATION
# =============================================================================

@dataclass
class PromptConfig:
    """Configuration for N-ATLaS prompt templates"""
    
    # System prompt for the N-ATLaS model
    system_prompt: str = """You are FarmEyes, an AI agricultural assistant helping Nigerian farmers. 
You provide advice about crop diseases and treatments in a clear, simple, and helpful manner.
Always be respectful and use language that farmers can easily understand.
When providing treatment costs, use Nigerian Naira (₦).
Focus on practical advice that farmers can implement."""
    
    # Maximum length for translated text
    max_translation_length: int = 500
    
    # Temperature for different tasks
    translation_temperature: float = 0.3
    diagnosis_temperature: float = 0.7
    chat_temperature: float = 0.6


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

@dataclass
class LogConfig:
    """Logging configuration"""
    
    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_level: str = "INFO"
    
    # Log file path
    log_file: Path = BASE_DIR / "logs" / "farmeyes.log"
    
    # Log format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Enable console logging
    console_logging: bool = True
    
    # Enable file logging
    file_logging: bool = True


# =============================================================================
# INSTANTIATE DEFAULT CONFIGURATIONS
# =============================================================================

# Create default configuration instances
api_config = APIConfig()
session_config = SessionConfig()
whisper_config = WhisperConfig()
yolo_config = YOLOConfig()
natlas_config = NATLaSConfig()
language_config = LanguageConfig()
chat_config = ChatConfig()
app_config = AppConfig()
device_config = DeviceConfig()
prompt_config = PromptConfig()
log_config = LogConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_disease_key(class_index: int) -> Optional[str]:
    """Get disease key from class index."""
    return CLASS_INDEX_TO_KEY.get(class_index)


def get_class_index(disease_key: str) -> Optional[int]:
    """Get class index from disease key."""
    return KEY_TO_CLASS_INDEX.get(disease_key)


def get_crop_type(class_index: int) -> Optional[str]:
    """Get crop type from class index."""
    return CLASS_TO_CROP.get(class_index)


def is_healthy(class_index: int) -> bool:
    """Check if class index represents a healthy plant (always False for 6-class)."""
    return class_index in HEALTHY_CLASS_INDICES


def validate_config() -> Dict[str, bool]:
    """Validate that all required configuration files and paths exist."""
    validations = {
        "knowledge_base_exists": KNOWLEDGE_BASE_PATH.exists(),
        "ui_translations_exists": UI_TRANSLATIONS_PATH.exists(),
        "models_dir_exists": MODELS_DIR.exists(),
        "yolo_model_exists": yolo_config.model_path.exists(),
        "natlas_model_exists": natlas_config.gguf_path.exists(),
        "frontend_dir_exists": FRONTEND_DIR.exists(),
    }
    return validations


def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("FarmEyes Configuration Summary v2.0")
    print("=" * 60)
    
    print(f"\n📁 Paths:")
    print(f"   Base Directory: {BASE_DIR}")
    print(f"   Knowledge Base: {KNOWLEDGE_BASE_PATH}")
    print(f"   Frontend: {FRONTEND_DIR}")
    
    print(f"\n🌐 API Configuration:")
    print(f"   Host: {api_config.host}:{api_config.port}")
    print(f"   Debug: {api_config.debug}")
    print(f"   HuggingFace: {api_config.is_huggingface}")
    
    print(f"\n🤖 YOLOv11 Model:")
    print(f"   Model Path: {yolo_config.model_path}")
    print(f"   Confidence: {yolo_config.confidence_threshold}")
    print(f"   Classes: {len(CLASS_NAMES)}")
    
    print(f"\n🗣️ N-ATLaS Model:")
    print(f"   HF Repo: {natlas_config.hf_repo}")
    print(f"   Chat Max Tokens: {natlas_config.chat_max_tokens}")
    
    print(f"\n🎤 Whisper (Voice):")
    print(f"   Model Size: {whisper_config.model_size}")
    print(f"   Device: {whisper_config.device}")
    
    print(f"\n💬 Chat:")
    print(f"   Enabled: {app_config.enable_chat}")
    print(f"   Voice Input: {app_config.enable_voice_input}")
    
    print(f"\n🌍 Languages:")
    print(f"   Supported: {', '.join(language_config.supported_languages)}")
    
    print("\n" + "=" * 60)


# =============================================================================
# MAIN - Run configuration check
# =============================================================================

if __name__ == "__main__":
    print_config_summary()
