"""
Pytest configuration and shared fixtures for AI Anchor tests.
"""

import pytest
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock()
    settings.debug = True
    settings.gpu = Mock()
    settings.gpu.max_gpu_memory_gb = 48.0
    settings.gpu.memory_fraction = 0.95
    settings.gpu.enable_gpu_monitoring = False
    settings.performance = Mock()
    settings.performance.memory_check_interval = 1
    settings.audio = Mock()
    settings.audio.sample_rate = 24000
    settings.audio.format = "wav"
    settings.audio.chunk_size = 1024
    settings.conversation = Mock()
    settings.conversation.temperature = 0.3
    settings.conversation.max_tokens = 2048
    settings.web = Mock()
    settings.web.cors_origins = ["*"]
    settings.web.host = "0.0.0.0"
    settings.web.port = 8000
    return settings


@pytest.fixture
def mock_memory_manager():
    """Create mock memory manager for testing."""
    manager = Mock()
    manager.cuda_available = True
    manager.device_count = 1
    manager.max_gpu_memory_gb = 48.0
    manager.models = {}
    manager.monitoring_active = False
    
    # Mock methods
    manager.get_memory_info = Mock(return_value=Mock(
        allocated_gb=5.0,
        reserved_gb=6.0,
        max_allocated_gb=7.0,
        free_gb=41.0,
        total_gb=48.0,
        utilization_percent=12.5
    ))
    manager.get_system_memory_info = Mock(return_value={
        "total_gb": 32.0,
        "available_gb": 16.0,
        "used_gb": 16.0,
        "percent": 50.0
    })
    manager.estimate_memory_usage = Mock(return_value=10.0)
    manager.can_load_model = Mock(return_value=True)
    manager.register_model = Mock()
    manager.unregister_model = Mock()
    manager.update_model_usage = Mock()
    manager.get_model_info = Mock(return_value=None)
    manager.get_all_models_info = Mock(return_value={})
    manager.memory_context = Mock()
    manager.memory_context.return_value.__enter__ = Mock()
    manager.memory_context.return_value.__exit__ = Mock()
    manager.cleanup_memory = Mock(return_value=2.0)
    manager.detect_memory_leaks = Mock(return_value={
        "status": "checked",
        "potential_leak": False
    })
    manager.get_memory_statistics = Mock(return_value={
        "cuda_available": True,
        "device_count": 1,
        "registered_models": {}
    })
    manager.health_check = AsyncMock(return_value={
        "cuda_available": True,
        "registered_models": 0,
        "health_status": "healthy"
    })
    manager.shutdown = Mock()
    
    return manager


@pytest.fixture
def mock_stt_model():
    """Create mock STT model for testing."""
    model = Mock()
    model.model_name = "voxtral-mini-3b-2507"
    model.language = "en"
    model.is_loaded = True
    
    # Mock methods
    model.initialize = AsyncMock()
    model.shutdown = AsyncMock()
    model.load_model = AsyncMock()
    model.unload_model = AsyncMock()
    model.process_audio_chunk = AsyncMock()
    model.process_buffered_audio = AsyncMock()
    model.feed_audio_chunk = Mock()
    model.set_language = Mock()
    model.get_supported_languages = Mock(return_value=["en", "es", "fr", "de"])
    model.health_check = AsyncMock(return_value={"status": "healthy"})
    
    return model


@pytest.fixture
def mock_tts_model():
    """Create mock TTS model for testing."""
    model = Mock()
    model.model_name = "higgs-audio-v2"
    model.current_voice_profile = "default"
    model.is_loaded = True
    
    # Mock methods
    model.initialize = AsyncMock()
    model.shutdown = AsyncMock()
    model.load_model = AsyncMock()
    model.unload_model = AsyncMock()
    model.synthesize = Mock(return_value=True)
    model.synthesize_streaming = AsyncMock()
    model.set_voice_profile = Mock(return_value=True)
    model.get_available_voices = Mock(return_value=["default", "news_anchor", "friendly"])
    model.clone_voice = AsyncMock(return_value=True)
    model.health_check = AsyncMock(return_value={"status": "healthy"})
    model.stop_event = Mock()
    
    return model


@pytest.fixture
def mock_llm_model():
    """Create mock LLM model for testing."""
    model = Mock()
    model.model_name = "huihui-ai/Mistral-Small-24B-Instruct-2501-abliterated"
    model.server_url = "http://localhost:11434"
    model.is_connected = True
    
    # Mock methods
    model.initialize = AsyncMock()
    model.shutdown = AsyncMock()
    model.prewarm = AsyncMock()
    model.generate = Mock(return_value=iter(["Hello", " there", "!"]))
    model.generate_streaming = AsyncMock()
    model.cancel_generation = Mock()
    model.set_system_prompt = Mock()
    model.set_temperature = Mock()
    model.health_check = AsyncMock(return_value={"status": "healthy"})
    
    return model


@pytest.fixture
def mock_vad_model():
    """Create mock VAD model for testing."""
    model = Mock()
    model.model_name = "turndetect-vad"
    model.is_loaded = True
    
    # Mock methods
    model.initialize = Mock()
    model.shutdown = Mock()
    model.calculate_waiting_time = Mock(return_value=1.5)
    model.predict_sentence_completion = Mock(return_value=0.8)
    model.health_check = AsyncMock(return_value={"status": "healthy"})
    model.on_new_waiting_time = Mock()
    
    return model


@pytest.fixture
def mock_audio_pipeline(mock_stt_model, mock_tts_model, mock_llm_model, mock_vad_model):
    """Create mock audio pipeline for testing."""
    pipeline = Mock()
    pipeline.state = "idle"
    pipeline.current_session = None
    pipeline.session_counter = 0
    
    # Assign models
    pipeline.stt_model = mock_stt_model
    pipeline.tts_model = mock_tts_model
    pipeline.llm_model = mock_llm_model
    pipeline.vad_model = mock_vad_model
    
    # Mock methods
    pipeline.initialize = AsyncMock()
    pipeline.shutdown = AsyncMock()
    pipeline.feed_audio = Mock()
    pipeline.process_text_input = Mock()
    pipeline.abort_current_generation = Mock()
    pipeline.change_voice_profile = Mock()
    pipeline.change_language = Mock()
    pipeline.get_pipeline_state = Mock(return_value={
        "state": "idle",
        "current_session": None,
        "session_counter": 0,
        "request_queue_size": 0,
        "audio_queue_size": 0,
        "statistics": {},
        "model_status": {
            "stt_loaded": True,
            "tts_loaded": True,
            "llm_loaded": True,
            "vad_loaded": True
        }
    })
    pipeline.health_check = AsyncMock(return_value={
        "pipeline_state": "idle",
        "workers_active": {"request_worker": True, "audio_worker": True},
        "queue_sizes": {"requests": 0, "audio_input": 0},
        "model_health": {"stt": {"status": "healthy"}},
        "statistics": {},
        "overall_health": "healthy"
    })
    
    # Callbacks
    pipeline.on_transcription = None
    pipeline.on_response_start = None
    pipeline.on_response_chunk = None
    pipeline.on_audio_chunk = None
    pipeline.on_session_complete = None
    pipeline.on_error = None
    
    return pipeline


@pytest.fixture
def mock_anchor_agent(mock_audio_pipeline, mock_memory_manager):
    """Create mock anchor agent for testing."""
    agent = Mock()
    agent.is_active = False
    agent.current_context = None
    agent.session_counter = 0
    agent.current_personality = Mock()
    agent.current_personality.name = "Professional"
    agent.current_personality.language = "en"
    
    # Assign dependencies
    agent.audio_pipeline = mock_audio_pipeline
    agent.memory_manager = mock_memory_manager
    
    # Mock methods
    agent.initialize = AsyncMock()
    agent.shutdown = AsyncMock()
    agent.start_conversation = AsyncMock(return_value="Hello! How can I help you today?")
    agent.end_conversation = AsyncMock()
    agent.switch_personality = AsyncMock(return_value=True)
    agent.switch_language = AsyncMock()
    agent.process_audio_input = Mock()
    agent.process_text_input = Mock()
    agent.abort_current_response = Mock()
    agent.set_conversation_context = Mock()
    agent.get_agent_status = Mock(return_value={
        "is_active": False,
        "current_personality": "Professional",
        "current_language": "en",
        "conversation_active": False,
        "session_id": None,
        "conversation_mode": None,
        "total_exchanges": 0,
        "pipeline_state": "idle",
        "available_personalities": ["professional", "friendly", "energetic", "calm", "authoritative"],
        "statistics": {
            "total_sessions": 0,
            "total_exchanges": 0,
            "average_response_time": 0.0,
            "personality_switches": 0,
            "errors": 0
        }
    })
    agent.health_check = AsyncMock(return_value={
        "agent_active": False,
        "conversation_active": False,
        "current_personality": "Professional",
        "personality_count": 5,
        "pipeline_health": "healthy",
        "statistics": {},
        "overall_health": "healthy"
    })
    agent.get_conversation_history = Mock(return_value=[])
    agent.get_available_personalities = Mock(return_value={
        "professional": {
            "name": "Professional",
            "personality_type": "professional",
            "language": "en",
            "voice_profile": "news_anchor_neutral",
            "description": "Professional news anchor personality"
        },
        "friendly": {
            "name": "Friendly", 
            "personality_type": "friendly",
            "language": "en",
            "voice_profile": "talk_show_warm",
            "description": "Warm, friendly talk show host personality"
        }
    })
    
    # Callbacks
    agent.on_conversation_start = None
    agent.on_conversation_end = None
    agent.on_personality_change = None
    agent.on_response_generated = None
    agent.on_error = None
    
    return agent


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket for testing."""
    websocket = Mock()
    websocket.messages_sent = []
    websocket.closed = False
    websocket.client_state = "CONNECTED"
    
    # Mock methods
    websocket.accept = AsyncMock()
    websocket.send_json = AsyncMock(side_effect=lambda data: websocket.messages_sent.append(data))
    websocket.receive_json = AsyncMock()
    websocket.close = AsyncMock(side_effect=lambda **kwargs: setattr(websocket, 'closed', True))
    
    return websocket


@pytest.fixture
def sample_audio_data():
    """Create sample audio data for testing."""
    import random
    return bytes([random.randint(0, 255) for _ in range(1024)])


@pytest.fixture
def sample_base64_audio():
    """Create sample base64-encoded audio for testing."""
    import base64
    sample_audio = b"test_audio_data_for_websocket_testing"
    return base64.b64encode(sample_audio).decode('utf-8')


@pytest.fixture
def performance_timer():
    """Create performance timer for benchmarking tests."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return PerformanceTimer()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Cleanup code can go here if needed
    pass