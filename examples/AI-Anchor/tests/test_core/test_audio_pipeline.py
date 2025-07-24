"""
Unit tests for Audio Pipeline Manager.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
from queue import Queue, Empty

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.audio_pipeline import (
    AudioPipelineManager,
    PipelineRequest,
    GenerationSession,
    PipelineState,
    RequestType
)
from core.memory_manager import MemoryManager
from config import Settings, LanguageCode, TranscriptionResult, AudioChunk, AudioFormat


class TestPipelineRequest:
    """Test suite for PipelineRequest dataclass."""
    
    def test_pipeline_request_creation(self):
        """Test PipelineRequest creation."""
        request = PipelineRequest(
            request_type=RequestType.PROCESS_AUDIO,
            data=b"audio_data",
            request_id="test-123"
        )
        
        assert request.request_type == RequestType.PROCESS_AUDIO
        assert request.data == b"audio_data"
        assert request.request_id == "test-123"
        assert isinstance(request.timestamp, float)
        assert request.timestamp > 0
    
    def test_pipeline_request_auto_timestamp(self):
        """Test automatic timestamp generation."""
        request = PipelineRequest(request_type=RequestType.ABORT_GENERATION)
        
        assert request.timestamp is not None
        assert isinstance(request.timestamp, float)


class TestGenerationSession:
    """Test suite for GenerationSession dataclass."""
    
    def test_generation_session_creation(self):
        """Test GenerationSession creation."""
        session = GenerationSession(
            session_id="session-1",
            input_text="Hello world",
            language=LanguageCode.EN,
            voice_profile="default"
        )
        
        assert session.session_id == "session-1"
        assert session.input_text == "Hello world"
        assert session.language == LanguageCode.EN
        assert session.voice_profile == "default"
        
        # Check default values
        assert session.stt_completed is False
        assert session.llm_started is False
        assert session.llm_completed is False
        assert session.tts_started is False
        assert session.tts_completed is False
        assert session.aborted is False
        assert session.error is None
        assert isinstance(session.audio_chunks, list)
        assert len(session.audio_chunks) == 0
        
        # Check threading events
        assert session.llm_ready_event is not None
        assert session.tts_ready_event is not None
        assert session.completion_event is not None


class TestAudioPipelineManager:
    """Test suite for AudioPipelineManager class."""
    
    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        return Settings(
            debug=True,
            audio__sample_rate=24000,
            audio__format=AudioFormat.WAV,
            conversation__temperature=0.3,
        )
    
    @pytest.fixture
    def mock_memory_manager(self, test_settings):
        """Create mock memory manager."""
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.memory_context.return_value.__enter__ = Mock()
        memory_manager.memory_context.return_value.__exit__ = Mock()
        return memory_manager
    
    @pytest.fixture
    def mock_models(self):
        """Create mock models."""
        return {
            "stt": Mock(),
            "tts": Mock(),
            "llm": Mock(),
            "vad": Mock(),
        }
    
    @pytest.fixture
    def pipeline_manager(self, test_settings, mock_memory_manager, mock_models):
        """Create pipeline manager instance."""
        manager = AudioPipelineManager(
            settings=test_settings,
            memory_manager=mock_memory_manager,
            stt_model=mock_models["stt"],
            tts_model=mock_models["tts"],
            llm_model=mock_models["llm"],
            vad_model=mock_models["vad"],
        )
        yield manager
        # Cleanup
        asyncio.create_task(manager.shutdown())
    
    def test_initialization(self, test_settings, mock_memory_manager):
        """Test pipeline manager initialization."""
        manager = AudioPipelineManager(
            settings=test_settings,
            memory_manager=mock_memory_manager,
        )
        
        assert manager.settings == test_settings
        assert manager.memory_manager == mock_memory_manager
        assert manager.state == PipelineState.IDLE
        assert manager.current_session is None
        assert manager.session_counter == 0
        assert isinstance(manager.request_queue, Queue)
        assert isinstance(manager.audio_input_queue, Queue)
        assert isinstance(manager.shutdown_event, threading.Event)
        assert manager.request_worker is None
        assert manager.audio_worker is None
    
    @pytest.mark.asyncio
    async def test_initialize_with_existing_models(self, pipeline_manager, mock_models):
        """Test initialization with pre-existing models."""
        # Mock model initialization methods
        for model in mock_models.values():
            if hasattr(model, 'initialize'):
                model.initialize = AsyncMock()
            if hasattr(model, 'prewarm'):
                model.prewarm = AsyncMock()
        
        await pipeline_manager.initialize()
        
        assert pipeline_manager.state == PipelineState.IDLE
        assert pipeline_manager.request_worker is not None
        assert pipeline_manager.audio_worker is not None
        assert pipeline_manager.request_worker.is_alive()
        assert pipeline_manager.audio_worker.is_alive()
    
    @pytest.mark.asyncio
    async def test_initialize_without_models(self, test_settings, mock_memory_manager):
        """Test initialization without pre-existing models."""
        manager = AudioPipelineManager(
            settings=test_settings,
            memory_manager=mock_memory_manager,
        )
        
        # Mock model classes
        with patch('core.audio_pipeline.VoxtralSTT') as mock_stt_class:
            with patch('core.audio_pipeline.HiggsAudioTTS') as mock_tts_class:
                with patch('core.audio_pipeline.HuihuiAILLM') as mock_llm_class:
                    with patch('core.audio_pipeline.TurnDetectionVAD') as mock_vad_class:
                        
                        # Setup mock instances
                        mock_stt = AsyncMock()
                        mock_tts = AsyncMock()
                        mock_llm = AsyncMock()
                        mock_vad = Mock()
                        
                        mock_stt_class.return_value = mock_stt
                        mock_tts_class.return_value = mock_tts
                        mock_llm_class.return_value = mock_llm
                        mock_vad_class.return_value = mock_vad
                        
                        await manager.initialize()
                        
                        # Check models were created
                        assert manager.stt_model == mock_stt
                        assert manager.tts_model == mock_tts
                        assert manager.llm_model == mock_llm
                        assert manager.vad_model == mock_vad
                        
                        # Check initialization was called
                        mock_stt.initialize.assert_called_once()
                        mock_tts.initialize.assert_called_once()
                        mock_llm.prewarm.assert_called_once()
        
        await manager.shutdown()
    
    def test_feed_audio(self, pipeline_manager):
        """Test feeding audio to the pipeline."""
        audio_data = b"test_audio_data"
        
        # Initially queue should be empty
        assert pipeline_manager.audio_input_queue.qsize() == 0
        
        # Feed audio
        pipeline_manager.feed_audio(audio_data)
        
        # Queue should now contain the audio
        assert pipeline_manager.audio_input_queue.qsize() == 1
        
        # Get the audio back
        queued_audio = pipeline_manager.audio_input_queue.get_nowait()
        assert queued_audio == audio_data
    
    def test_process_text_input(self, pipeline_manager):
        """Test processing direct text input."""
        text = "Hello, this is a test"
        language = LanguageCode.EN
        
        # Initially request queue should be empty
        assert pipeline_manager.request_queue.qsize() == 0
        
        # Process text input
        pipeline_manager.process_text_input(text, language)
        
        # Request should be queued
        assert pipeline_manager.request_queue.qsize() == 1
        
        # Get the request
        request = pipeline_manager.request_queue.get_nowait()
        assert request.request_type == RequestType.GENERATE_RESPONSE
        assert isinstance(request.data, TranscriptionResult)
        assert request.data.text == text
        assert request.data.language == language
        assert request.data.is_final is True
    
    def test_abort_current_generation(self, pipeline_manager):
        """Test aborting current generation."""
        # Initially request queue should be empty
        assert pipeline_manager.request_queue.qsize() == 0
        
        # Abort generation
        pipeline_manager.abort_current_generation()
        
        # Abort request should be queued
        assert pipeline_manager.request_queue.qsize() == 1
        
        request = pipeline_manager.request_queue.get_nowait()
        assert request.request_type == RequestType.ABORT_GENERATION
    
    def test_change_voice_profile(self, pipeline_manager):
        """Test changing voice profile."""
        voice_name = "new_voice"
        
        pipeline_manager.change_voice_profile(voice_name)
        
        assert pipeline_manager.request_queue.qsize() == 1
        request = pipeline_manager.request_queue.get_nowait()
        assert request.request_type == RequestType.CHANGE_VOICE
        assert request.data == voice_name
    
    def test_change_language(self, pipeline_manager):
        """Test changing language."""
        language = LanguageCode.ES
        
        pipeline_manager.change_language(language)
        
        assert pipeline_manager.request_queue.qsize() == 1
        request = pipeline_manager.request_queue.get_nowait()
        assert request.request_type == RequestType.CHANGE_LANGUAGE
        assert request.data == language
    
    def test_get_pipeline_state(self, pipeline_manager):
        """Test getting pipeline state."""
        state = pipeline_manager.get_pipeline_state()
        
        assert isinstance(state, dict)
        assert "state" in state
        assert "current_session" in state
        assert "session_counter" in state
        assert "request_queue_size" in state
        assert "audio_queue_size" in state
        assert "statistics" in state
        assert "model_status" in state
        
        assert state["state"] == PipelineState.IDLE.value
        assert state["current_session"] is None
        assert state["session_counter"] == 0
        assert state["model_status"]["stt_loaded"] is True
        assert state["model_status"]["tts_loaded"] is True
        assert state["model_status"]["llm_loaded"] is True
        assert state["model_status"]["vad_loaded"] is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, pipeline_manager):
        """Test pipeline health check."""
        # Mock model health checks
        pipeline_manager.stt_model.health_check = AsyncMock(return_value={"status": "healthy"})
        pipeline_manager.tts_model.health_check = AsyncMock(return_value={"status": "healthy"})
        pipeline_manager.llm_model.health_check = AsyncMock(return_value={"status": "healthy"})
        pipeline_manager.vad_model.health_check = AsyncMock(return_value={"status": "healthy"})
        
        health = await pipeline_manager.health_check()
        
        assert isinstance(health, dict)
        assert "pipeline_state" in health
        assert "workers_active" in health
        assert "queue_sizes" in health
        assert "model_health" in health
        assert "statistics" in health
        assert "overall_health" in health
        
        assert health["pipeline_state"] == PipelineState.IDLE.value
    
    def test_callbacks(self, pipeline_manager):
        """Test callback mechanism."""
        # Set up callbacks
        on_transcription = Mock()
        on_response_start = Mock()
        on_response_chunk = Mock()
        on_audio_chunk = Mock()
        on_session_complete = Mock()
        on_error = Mock()
        
        pipeline_manager.on_transcription = on_transcription
        pipeline_manager.on_response_start = on_response_start
        pipeline_manager.on_response_chunk = on_response_chunk
        pipeline_manager.on_audio_chunk = on_audio_chunk
        pipeline_manager.on_session_complete = on_session_complete
        pipeline_manager.on_error = on_error
        
        # Test transcription callback
        transcription = TranscriptionResult(
            text="test",
            language=LanguageCode.EN,
            confidence=0.95,
            timestamp=time.time(),
            is_final=True
        )
        
        pipeline_manager._on_realtime_transcription(transcription)
        on_transcription.assert_called_once_with(transcription)
        
        # Test error callback
        error = Exception("Test error")
        pipeline_manager._on_stt_error(error)
        on_error.assert_called_once_with(error)
    
    @pytest.mark.asyncio
    async def test_request_processing(self, pipeline_manager):
        """Test request processing workflow."""
        # Mock the process request method
        pipeline_manager._process_request = AsyncMock()
        
        # Create test request
        request = PipelineRequest(request_type=RequestType.PROCESS_AUDIO, data=b"audio")
        pipeline_manager.request_queue.put(request)
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Should have processed the request
        # Note: This is a simplified test - full integration would need actual workers
    
    @pytest.mark.asyncio
    async def test_generation_session_workflow(self, pipeline_manager, mock_models):
        """Test complete generation session workflow."""
        # Setup mocks
        mock_models["llm"].generate.return_value = iter(["Hello", " there", "!"])
        mock_models["tts"].synthesize.return_value = True
        
        # Create transcription result
        transcription = TranscriptionResult(
            text="Hi there",
            language=LanguageCode.EN,
            confidence=0.95,
            timestamp=time.time(),
            is_final=True
        )
        
        # Start generation
        await pipeline_manager._start_response_generation(transcription)
        
        # Check session was created
        assert pipeline_manager.current_session is not None
        assert pipeline_manager.current_session.input_text == "Hi there"
        assert pipeline_manager.current_session.language == LanguageCode.EN
        assert pipeline_manager.state == PipelineState.GENERATING
    
    @pytest.mark.asyncio
    async def test_session_abortion(self, pipeline_manager):
        """Test session abortion mechanism."""
        # Create a session
        session = GenerationSession(
            session_id="test-session",
            input_text="test",
            language=LanguageCode.EN
        )
        session.llm_started = True
        session.tts_started = True
        
        pipeline_manager.current_session = session
        
        # Mock model cancel methods
        pipeline_manager.llm_model.cancel_generation = Mock()
        pipeline_manager.tts_model.stop_event = Mock()
        
        # Create abort request
        request = PipelineRequest(request_type=RequestType.ABORT_GENERATION)
        
        # Process abort request
        await pipeline_manager._handle_abort_generation(request)
        
        # Check session was aborted
        assert session.aborted is True
        assert session.completion_event.is_set()
        pipeline_manager.llm_model.cancel_generation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_voice_and_language_changes(self, pipeline_manager):
        """Test voice profile and language change handling."""
        # Mock TTS model methods
        pipeline_manager.tts_model.set_voice_profile = Mock(return_value=True)
        pipeline_manager.stt_model.set_language = Mock()
        
        # Test voice change
        voice_request = PipelineRequest(
            request_type=RequestType.CHANGE_VOICE,
            data="new_voice"
        )
        await pipeline_manager._handle_voice_change(voice_request)
        pipeline_manager.tts_model.set_voice_profile.assert_called_once_with("new_voice")
        
        # Test language change
        lang_request = PipelineRequest(
            request_type=RequestType.CHANGE_LANGUAGE,
            data=LanguageCode.ES
        )
        await pipeline_manager._handle_language_change(lang_request)
        pipeline_manager.stt_model.set_language.assert_called_once_with(LanguageCode.ES)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, pipeline_manager):
        """Test error handling in pipeline."""
        error_callback = Mock()
        pipeline_manager.on_error = error_callback
        
        # Test STT error
        test_error = Exception("STT failed")
        pipeline_manager._on_stt_error(test_error)
        error_callback.assert_called_with(test_error)
        
        # Test general error handling
        error_callback.reset_mock()
        
        # Mock a request that will cause an error
        with patch.object(pipeline_manager, '_handle_audio_processing', side_effect=test_error):
            request = PipelineRequest(request_type=RequestType.PROCESS_AUDIO, data=b"audio")
            
            # This should trigger error handling
            await pipeline_manager._process_request(request)
            
            error_callback.assert_called_with(test_error)
    
    @pytest.mark.asyncio
    async def test_shutdown(self, pipeline_manager):
        """Test pipeline shutdown process."""
        # Create a current session
        session = GenerationSession(
            session_id="test-session",
            input_text="test",
            language=LanguageCode.EN
        )
        pipeline_manager.current_session = session
        
        # Add some items to queues
        pipeline_manager.request_queue.put(PipelineRequest(RequestType.PROCESS_AUDIO))
        pipeline_manager.audio_input_queue.put(b"audio_data")
        
        # Mock model shutdown methods
        pipeline_manager.stt_model.shutdown = AsyncMock()
        pipeline_manager.tts_model.shutdown = AsyncMock()
        pipeline_manager.llm_model.shutdown = AsyncMock()
        pipeline_manager.vad_model.shutdown = Mock()
        
        # Shutdown
        await pipeline_manager.shutdown()
        
        # Check state
        assert pipeline_manager.state == PipelineState.SHUTDOWN
        assert pipeline_manager.shutdown_event.is_set()
        assert session.aborted is True
        assert session.completion_event.is_set()
        
        # Check models were shutdown
        pipeline_manager.stt_model.shutdown.assert_called_once()
        pipeline_manager.tts_model.shutdown.assert_called_once()
        pipeline_manager.llm_model.shutdown.assert_called_once()
        pipeline_manager.vad_model.shutdown.assert_called_once()
        
        # Check queues were cleared
        assert pipeline_manager.request_queue.empty()
        assert pipeline_manager.audio_input_queue.empty()


class TestAudioPipelineIntegration:
    """Integration tests for AudioPipelineManager."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_workflow_mocked(self):
        """Test full pipeline workflow with mocked components."""
        settings = Settings(debug=True)
        
        # Create mock memory manager
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.memory_context.return_value.__enter__ = Mock()
        memory_manager.memory_context.return_value.__exit__ = Mock()
        
        # Create pipeline manager
        manager = AudioPipelineManager(
            settings=settings,
            memory_manager=memory_manager,
        )
        
        # Mock model classes and their behavior
        with patch('core.audio_pipeline.VoxtralSTT') as mock_stt_class:
            with patch('core.audio_pipeline.HiggsAudioTTS') as mock_tts_class:
                with patch('core.audio_pipeline.HuihuiAILLM') as mock_llm_class:
                    with patch('core.audio_pipeline.TurnDetectionVAD') as mock_vad_class:
                        
                        # Setup mock instances with required methods
                        mock_stt = Mock()
                        mock_stt.initialize = AsyncMock()
                        mock_stt.process_buffered_audio = AsyncMock()
                        mock_stt.shutdown = AsyncMock()
                        mock_stt_class.return_value = mock_stt
                        
                        mock_tts = Mock()
                        mock_tts.initialize = AsyncMock()
                        mock_tts.shutdown = AsyncMock()
                        mock_tts_class.return_value = mock_tts
                        
                        mock_llm = Mock()
                        mock_llm.prewarm = AsyncMock()
                        mock_llm.shutdown = AsyncMock()
                        mock_llm_class.return_value = mock_llm
                        
                        mock_vad = Mock()
                        mock_vad.shutdown = Mock()
                        mock_vad_class.return_value = mock_vad
                        
                        # Initialize pipeline
                        await manager.initialize()
                        
                        # Test text input processing
                        manager.process_text_input("Hello world", LanguageCode.EN)
                        
                        # Test voice change
                        manager.change_voice_profile("test_voice")
                        
                        # Test language change
                        manager.change_language(LanguageCode.ES)
                        
                        # Get pipeline state
                        state = manager.get_pipeline_state()
                        assert state["state"] == PipelineState.IDLE.value
                        
                        # Perform health check
                        mock_stt.health_check = AsyncMock(return_value={"status": "ok"})
                        mock_tts.health_check = AsyncMock(return_value={"status": "ok"})
                        mock_llm.health_check = AsyncMock(return_value={"status": "ok"})
                        mock_vad.health_check = AsyncMock(return_value={"status": "ok"})
                        
                        health = await manager.health_check()
                        assert "overall_health" in health
                        
                        # Shutdown
                        await manager.shutdown()
                        
                        # Verify shutdown was called on all models
                        mock_stt.shutdown.assert_called_once()
                        mock_tts.shutdown.assert_called_once()
                        mock_llm.shutdown.assert_called_once()
                        mock_vad.shutdown.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])