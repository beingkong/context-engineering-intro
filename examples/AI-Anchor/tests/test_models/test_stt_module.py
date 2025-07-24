"""
Unit tests for STT module (Voxtral-Mini-3B-2507).
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.stt_module import VoxtralSTT, TranscriptionMode
from config import Settings, LanguageCode, TranscriptionResult


class TestVoxtralSTT:
    """Test suite for VoxtralSTT class."""
    
    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        return Settings(
            debug=True,
            models__voxtral_model="test-model-path",
            conversation__default_language=LanguageCode.EN,
            conversation__supported_languages=[LanguageCode.EN, LanguageCode.ES],
        )
    
    @pytest.fixture
    def stt_processor(self, test_settings):
        """Create STT processor instance."""
        return VoxtralSTT(
            settings=test_settings,
            realtime_callback=Mock(),
            final_callback=Mock(),
            error_callback=Mock(),
        )
    
    @pytest.fixture
    def sample_audio_bytes(self):
        """Generate sample audio data as bytes."""
        # Generate 1 second of sine wave at 16kHz
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit integer
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    def test_initialization(self, test_settings):
        """Test STT processor initialization."""
        stt = VoxtralSTT(
            settings=test_settings,
            realtime_callback=Mock(),
            final_callback=Mock(),
        )
        
        assert stt.settings == test_settings
        assert stt.model_loaded is False
        assert stt.engine is None
        assert stt.current_language == LanguageCode.EN
        assert len(stt.audio_buffer) == 0
    
    def test_preprocess_audio(self, stt_processor, sample_audio_bytes):
        """Test audio preprocessing."""
        processed = stt_processor.preprocess_audio(sample_audio_bytes, source_sample_rate=24000)
        
        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.float32
        assert len(processed) > 0
        assert np.all(np.abs(processed) <= 1.0)  # Normalized to [-1, 1]
    
    def test_preprocess_audio_resample(self, stt_processor):
        """Test audio resampling during preprocessing."""
        # Create audio at 24kHz
        sample_rate = 24000
        audio_24k = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate))
        audio_bytes = (audio_24k * 32767).astype(np.int16).tobytes()
        
        processed = stt_processor.preprocess_audio(audio_bytes, source_sample_rate=24000)
        
        # Should be resampled to 16kHz
        expected_length = int(len(audio_24k) * 16000 / 24000)
        assert abs(len(processed) - expected_length) <= 1  # Allow for small rounding differences
    
    def test_detect_language(self, stt_processor):
        """Test language detection."""
        audio = np.random.randn(16000).astype(np.float32)
        language = stt_processor.detect_language(audio)
        
        # Should return default language for now
        assert language == LanguageCode.EN
    
    def test_format_prompt_for_voxtral(self, stt_processor):
        """Test Voxtral prompt formatting."""
        audio = np.random.randn(1600).astype(np.float32)  # 0.1 seconds
        prompt = stt_processor.format_prompt_for_voxtral(audio, LanguageCode.EN)
        
        assert "<|begin_of_transcript|>" in prompt
        assert "<|en|>" in prompt
        assert "<|transcribe|>" in prompt
        assert "<|audio|>" in prompt
        assert "<|end_of_transcript|>" in prompt
    
    def test_clean_transcription_text(self, stt_processor):
        """Test transcription text cleaning."""
        dirty_text = "  <|test|>  hello   world  <|end|>  "
        cleaned = stt_processor._clean_transcription_text(dirty_text)
        
        assert cleaned == "Test hello world end"
    
    def test_set_language(self, stt_processor):
        """Test language setting."""
        # Test valid language
        stt_processor.set_language(LanguageCode.ES)
        assert stt_processor.current_language == LanguageCode.ES
        
        # Test invalid language (should warn but not crash)
        original_lang = stt_processor.current_language
        stt_processor.set_language(LanguageCode.FR)  # Not in supported languages
        assert stt_processor.current_language == original_lang  # Should not change
    
    def test_get_supported_languages(self, stt_processor):
        """Test getting supported languages."""
        languages = stt_processor.get_supported_languages()
        
        assert isinstance(languages, list)
        assert LanguageCode.EN in languages
        assert LanguageCode.ES in languages
        assert len(languages) == 2
    
    def test_feed_audio_chunk(self, stt_processor, sample_audio_bytes):
        """Test audio chunk feeding."""
        initial_buffer_size = len(stt_processor.audio_buffer)
        
        stt_processor.feed_audio_chunk(sample_audio_bytes)
        
        assert len(stt_processor.audio_buffer) == initial_buffer_size + 1
    
    def test_feed_audio_chunk_buffer_limit(self, stt_processor, sample_audio_bytes):
        """Test audio buffer size limiting."""
        # Fill buffer beyond limit
        for _ in range(stt_processor.max_buffer_size + 5):
            stt_processor.feed_audio_chunk(sample_audio_bytes)
        
        assert len(stt_processor.audio_buffer) == stt_processor.max_buffer_size
    
    def test_get_statistics(self, stt_processor):
        """Test statistics retrieval."""
        stats = stt_processor.get_statistics()
        
        assert isinstance(stats, dict)
        assert "chunks_processed" in stats
        assert "total_processing_time" in stats
        assert "average_latency" in stats
        assert "errors" in stats
    
    @pytest.mark.asyncio
    async def test_health_check_without_model(self, stt_processor):
        """Test health check when model is not loaded."""
        health = await stt_processor.health_check()
        
        assert health["model_loaded"] is False
        assert health["engine_ready"] is False
        assert health["processing"] is False
        assert health["test_result"] == "model_not_loaded"
    
    @pytest.mark.asyncio
    async def test_shutdown(self, stt_processor):
        """Test shutdown process."""
        # Add some data to buffer
        stt_processor.audio_buffer = [np.random.randn(100)]
        
        await stt_processor.shutdown()
        
        assert stt_processor.shutdown_flag is True
        assert stt_processor.model_loaded is False
        assert stt_processor.engine is None
        assert len(stt_processor.audio_buffer) == 0


class TestVoxtralSTTIntegration:
    """Integration tests for VoxtralSTT (requires mocking vLLM)."""
    
    @pytest.fixture
    def mock_vllm_engine(self):
        """Mock vLLM engine."""
        engine = AsyncMock()
        
        # Mock generation result
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "Hello world"
        
        async def mock_generate(prompt, **kwargs):
            yield mock_output
            
        engine.generate = mock_generate
        return engine
    
    @pytest.mark.asyncio
    async def test_initialize_with_mock(self, test_settings):
        """Test initialization with mocked vLLM."""
        with patch('models.stt_module.AsyncLLMEngine') as mock_engine_class:
            mock_engine_class.from_engine_args.return_value = AsyncMock()
            
            stt = VoxtralSTT(settings=test_settings)
            await stt.initialize()
            
            assert stt.model_loaded is True
            assert stt.engine is not None
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_with_mock(self, test_settings, sample_audio_bytes, mock_vllm_engine):
        """Test audio transcription with mocked engine."""
        stt = VoxtralSTT(settings=test_settings)
        stt.engine = mock_vllm_engine
        stt.model_loaded = True
        
        result = await stt.transcribe_audio(sample_audio_bytes)
        
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == LanguageCode.EN
        assert result.confidence > 0
        assert result.is_final is True
    
    @pytest.mark.asyncio
    async def test_process_buffered_audio_with_mock(self, test_settings, sample_audio_bytes, mock_vllm_engine):
        """Test buffered audio processing with mocked engine."""
        stt = VoxtralSTT(settings=test_settings)
        stt.engine = mock_vllm_engine
        stt.model_loaded = True
        
        # Add audio to buffer
        stt.feed_audio_chunk(sample_audio_bytes)
        stt.feed_audio_chunk(sample_audio_bytes)
        
        result = await stt.process_buffered_audio()
        
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert len(stt.audio_buffer) == 0  # Buffer should be cleared


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in STT module."""
    settings = Settings(models__voxtral_model="invalid-model")
    error_callback = Mock()
    
    stt = VoxtralSTT(settings=settings, error_callback=error_callback)
    
    # Test initialization error
    with pytest.raises(Exception):
        await stt.initialize()
    
    # Error callback should have been called
    assert error_callback.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])