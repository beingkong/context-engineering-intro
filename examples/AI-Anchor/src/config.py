"""
Configuration management for AI Anchor system.

Mirrors pattern from RealtimeVoiceChat/code/server.py but uses pydantic-settings
for structured environment variable management.
"""

from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"


class LanguageCode(str, Enum):
    """Supported language codes."""
    EN = "en"
    ES = "es"
    FR = "fr"
    PT = "pt"
    HI = "hi"
    DE = "de"
    NL = "nl"
    IT = "it"


class TTSEngine(str, Enum):
    """TTS engine options."""
    HIGGS_AUDIO = "higgs-audio"


class LLMProvider(str, Enum):
    """LLM provider options."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    LMSTUDIO = "lmstudio"


class AudioChunk(BaseModel):
    """Audio chunk data model."""
    data: bytes = Field(description="Raw audio data")
    sample_rate: int = Field(description="Audio sample rate")
    timestamp: float = Field(description="Timestamp when chunk was recorded")
    duration: float = Field(description="Duration of the audio chunk in seconds")
    format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio format")
    
    class Config:
        arbitrary_types_allowed = True


class TranscriptionResult(BaseModel):
    """Speech-to-text transcription result."""
    text: str = Field(description="Transcribed text")
    language: LanguageCode = Field(description="Detected language")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    start_time: float = Field(description="Start time of transcription")
    end_time: float = Field(description="End time of transcription")
    is_partial: bool = Field(default=False, description="Whether this is a partial result")


class ModelPaths(BaseModel):
    """Model path configurations."""
    higgs_audio_model: str = Field(
        default="bosonai/higgs-audio-v2-generation-3B-base",
        description="Higgs Audio TTS model path"
    )
    higgs_audio_tokenizer: str = Field(
        default="bosonai/higgs-audio-v2-tokenizer",
        description="Higgs Audio tokenizer path"
    )
    voxtral_model: str = Field(
        default="mistralai/Voxtral-Mini-3B-2507",
        description="Voxtral STT model path"
    )
    ollama_model: str = Field(
        default="hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M",
        description="Ollama LLM model path"
    )
    vad_model: str = Field(
        default="distilbert-base-uncased",
        description="VAD model for turn detection"
    )


class GPUConfig(BaseModel):
    """GPU configuration settings."""
    cuda_visible_devices: str = Field(default="0", description="CUDA device visibility")
    max_gpu_memory_gb: float = Field(default=48.0, description="Maximum GPU memory in GB")
    enable_gpu_monitoring: bool = Field(default=True, description="Enable GPU memory monitoring")
    memory_fraction: float = Field(default=0.95, description="GPU memory utilization fraction")
    
    @field_validator('max_gpu_memory_gb')
    @classmethod
    def validate_gpu_memory(cls, v):
        if v <= 0:
            raise ValueError('GPU memory must be positive')
        return v


class AudioConfig(BaseModel):
    """Audio processing configuration."""
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")
    chunk_size: int = Field(default=1024, description="Audio chunk size")
    format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio format")
    max_duration: int = Field(default=30, description="Maximum audio duration in seconds")
    max_queue_size: int = Field(default=50, description="Maximum audio queue size")
    
    # VAD settings
    vad_threshold: float = Field(default=0.5, description="Voice activity detection threshold")
    min_pause_duration: float = Field(default=0.5, description="Minimum pause duration in seconds")
    max_pause_duration: float = Field(default=3.0, description="Maximum pause duration in seconds")
    
    # TTS settings (mirroring RealtimeVoiceChat pattern)
    tts_final_timeout: float = Field(default=1.0, description="TTS final timeout in seconds")
    direct_stream: bool = Field(default=True, description="Enable direct audio streaming")


class ConversationConfig(BaseModel):
    """Conversation management configuration."""
    max_history: int = Field(default=10, description="Maximum conversation history length")
    default_language: LanguageCode = Field(default=LanguageCode.EN, description="Default language")
    supported_languages: List[LanguageCode] = Field(
        default=[LanguageCode.EN, LanguageCode.ES, LanguageCode.FR, LanguageCode.PT, 
                LanguageCode.HI, LanguageCode.DE, LanguageCode.NL, LanguageCode.IT],
        description="Supported languages"
    )
    context_window_size: int = Field(default=4096, description="LLM context window size")
    temperature: float = Field(default=0.3, description="LLM generation temperature")


class PerformanceConfig(BaseModel):
    """Performance and optimization settings."""
    max_concurrent_requests: int = Field(default=5, description="Maximum concurrent requests")
    response_timeout: int = Field(default=30, description="Response timeout in seconds")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_size_mb: int = Field(default=512, description="Cache size in MB")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    
    # Memory management
    enable_model_swapping: bool = Field(default=False, description="Enable model swapping")
    memory_check_interval: int = Field(default=30, description="Memory check interval in seconds")


class SecurityConfig(BaseModel):
    """Security configuration."""
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins"
    )
    max_request_size: int = Field(default=16 * 1024 * 1024, description="Maximum request size in bytes")


class Settings(BaseSettings):
    """
    Main configuration class for AI Anchor system.
    
    Mirrors the configuration pattern from RealtimeVoiceChat/code/server.py
    but uses pydantic-settings for structured environment variable management.
    """
    
    # Server configuration (mirroring RealtimeVoiceChat pattern)
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: LogLevel = Field(default=LogLevel.DEBUG, description="Logging level")
    use_ssl: bool = Field(default=False, description="Use SSL")  # Mirroring USE_SSL
    
    # Engine configuration (mirroring RealtimeVoiceChat pattern)
    tts_engine: TTSEngine = Field(default=TTSEngine.HIGGS_AUDIO, description="TTS engine")
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA, description="LLM provider")
    
    # Development settings
    save_audio_logs: bool = Field(default=False, description="Save audio for debugging")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    # Nested configurations
    models: ModelPaths = Field(default_factory=ModelPaths)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "env_nested_delimiter": "__",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    @field_validator('port')
    @classmethod 
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    def get_model_download_paths(self) -> Dict[str, str]:
        """
        Get model download paths for automatic model downloading.
        
        Returns:
            Dictionary mapping model names to their download paths.
        """
        return {
            "higgs_audio_model": self.models.higgs_audio_model,
            "higgs_audio_tokenizer": self.models.higgs_audio_tokenizer,
            "voxtral_model": self.models.voxtral_model,
            "ollama_model": self.models.ollama_model,
            "vad_model": self.models.vad_model,
        }
    
    def get_memory_allocation(self) -> Dict[str, float]:
        """
        Get estimated memory allocation for each model.
        
        Returns:
            Dictionary mapping model names to estimated memory usage in GB.
        """
        return {
            "stt_model": 9.5,  # Voxtral-Mini-3B-2507
            "tts_model": 24.0,  # higgs-audio v2
            "llm_model": 14.0,  # huihui-ai Mistral-Small-24B Q4_K_M
            "vad_model": 0.5,   # DistilBERT for VAD
            "system_overhead": 1.0,
        }
    
    def validate_memory_requirements(self) -> bool:
        """
        Validate that total memory requirements fit within GPU limits.
        
        Returns:
            True if memory requirements are within limits.
        """
        total_memory = sum(self.get_memory_allocation().values())
        return total_memory <= self.gpu.max_gpu_memory_gb
    
    def get_cors_middleware_config(self) -> Dict[str, Any]:
        """
        Get CORS middleware configuration.
        
        Returns:
            CORS configuration dictionary.
        """
        return {
            "allow_origins": self.security.cors_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }