"""
Unit tests for TTS module (higgs-audio v2).
"""

import pytest
import asyncio
import numpy as np
import threading
from unittest.mock import Mock, patch, AsyncMock
from queue import Queue

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.tts_module import HiggsAudioTTS, VoiceProfile, QUICK_ANSWER_STREAM_CHUNK_SIZE
from config import Settings, LanguageCode


class TestVoiceProfile:
    """Test suite for VoiceProfile class."""
    
    def test_voice_profile_initialization(self):
        """Test voice profile initialization."""
        profile = VoiceProfile(
            name="test_voice",
            language=LanguageCode.EN,
            gender="female",
            description="Test voice profile",
        )
        
        assert profile.name == "test_voice"
        assert profile.language == LanguageCode.EN
        assert profile.gender == "female"
        assert profile.description == "Test voice profile"
        assert profile.reference_audio_data is None
    
    def test_voice_profile_with_audio_data(self):
        """Test voice profile with reference audio data."""
        audio_data = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
        
        profile = VoiceProfile(
            name="test_voice",
            reference_audio_data=audio_data,
            language=LanguageCode.EN,
        )
        
        assert profile.reference_audio_data is not None
        assert len(profile.reference_audio_data) == 24000
        assert profile.reference_audio_data.dtype == np.float32
    
    @patch('soundfile.read')
    def test_load_reference_audio(self, mock_sf_read):
        """Test loading reference audio from file."""
        # Mock soundfile.read
        sample_audio = np.random.randn(48000).astype(np.float32)  # 2 seconds at 24kHz
        mock_sf_read.return_value = (sample_audio, 24000)
        
        profile = VoiceProfile(
            name="test_voice",
            reference_audio_path="/fake/path/audio.wav",
        )
        
        # Manually call load_reference_audio since file doesn't exist
        with patch('os.path.exists', return_value=True):
            profile.load_reference_audio()
        
        assert profile.reference_audio_data is not None
        assert len(profile.reference_audio_data) == 48000


class TestHiggsAudioTTS:
    """Test suite for HiggsAudioTTS class."""
    
    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        return Settings(
            debug=True,
            models__higgs_audio_model="test-model-path",
            models__higgs_audio_tokenizer="test-tokenizer-path",
            audio__sample_rate=24000,
            conversation__default_language=LanguageCode.EN,
        )
    
    @pytest.fixture
    def test_voice_profile(self):
        """Create test voice profile."""
        audio_data = np.random.randn(24000).astype(np.float32)
        return VoiceProfile(
            name="test_voice",
            reference_audio_data=audio_data,
            language=LanguageCode.EN,
            gender="neutral",
        )
    
    @pytest.fixture
    def tts_processor(self, test_settings, test_voice_profile):
        """Create TTS processor instance."""
        return HiggsAudioTTS(
            settings=test_settings,
            voice_profile=test_voice_profile,
        )
    
    def test_initialization(self, test_settings, test_voice_profile):
        """Test TTS processor initialization."""
        tts = HiggsAudioTTS(
            settings=test_settings,
            voice_profile=test_voice_profile,
        )
        
        assert tts.settings == test_settings
        assert tts.voice_profile == test_voice_profile
        assert tts.model_loaded is False
        assert tts.serve_engine is None
        assert tts.current_stream_chunk_size == QUICK_ANSWER_STREAM_CHUNK_SIZE
        assert len(tts.voice_profiles) == 1
        assert "test_voice" in tts.voice_profiles
    
    def test_initialization_without_voice_profile(self, test_settings):
        """Test TTS processor initialization without voice profile."""
        tts = HiggsAudioTTS(settings=test_settings)
        
        assert tts.voice_profile is not None
        assert tts.voice_profile.name == "default"
        assert tts.voice_profile.language == LanguageCode.EN
    
    def test_chunk_audio(self, tts_processor):
        """Test audio chunking functionality."""
        # Create 1 second of audio at 24kHz
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        
        chunks = tts_processor._chunk_audio(audio_data, chunk_size_ms=100)
        
        # Should have ~10 chunks for 1 second with 100ms chunks
        assert len(chunks) >= 9 and len(chunks) <= 11
        
        # Each chunk should be bytes
        for chunk in chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) > 0
    
    def test_add_voice_profile(self, tts_processor):
        """Test adding voice profiles."""
        new_profile = VoiceProfile(
            name="new_voice",
            language=LanguageCode.ES,
            gender="male",
        )
        
        initial_count = len(tts_processor.voice_profiles)
        tts_processor.add_voice_profile(new_profile)
        
        assert len(tts_processor.voice_profiles) == initial_count + 1
        assert "new_voice" in tts_processor.voice_profiles
        assert tts_processor.voice_profiles["new_voice"] == new_profile
    
    def test_get_voice_profile(self, tts_processor):
        """Test getting voice profiles."""
        # Test existing profile
        profile = tts_processor.get_voice_profile("test_voice")
        assert profile is not None
        assert profile.name == "test_voice"
        
        # Test non-existent profile
        profile = tts_processor.get_voice_profile("non_existent")
        assert profile is None
    
    def test_list_voice_profiles(self, tts_processor):
        """Test listing voice profiles."""
        profiles = tts_processor.list_voice_profiles()
        
        assert isinstance(profiles, list)
        assert "test_voice" in profiles
        assert len(profiles) == 1
    
    def test_set_voice_profile(self, tts_processor):
        """Test setting active voice profile."""
        # Add another profile
        new_profile = VoiceProfile(name="new_voice", language=LanguageCode.ES)
        tts_processor.add_voice_profile(new_profile)
        
        # Test setting existing profile
        result = tts_processor.set_voice_profile("new_voice")
        assert result is True
        assert tts_processor.voice_profile.name == "new_voice"
        
        # Test setting non-existent profile
        result = tts_processor.set_voice_profile("non_existent")
        assert result is False
        assert tts_processor.voice_profile.name == "new_voice"  # Should not change
    
    def test_get_statistics(self, tts_processor):
        """Test getting statistics."""
        stats = tts_processor.get_statistics()
        
        assert isinstance(stats, dict)
        assert "ttfa_ms" in stats
        assert "chunks_generated" in stats
        assert "total_synthesis_time" in stats
        assert "average_latency" in stats
        assert "errors" in stats
        assert "voice_profiles" in stats
        assert "active_profile" in stats
        
        assert stats["voice_profiles"] == 1
        assert stats["active_profile"] == "test_voice"
    
    def test_on_audio_stream_stop(self, tts_processor):
        """Test audio stream stop callback."""
        assert not tts_processor.finished_event.is_set()
        
        tts_processor.on_audio_stream_stop()
        
        assert tts_processor.finished_event.is_set()
    
    @pytest.mark.asyncio
    async def test_health_check_without_model(self, tts_processor):
        """Test health check when model is not loaded."""
        health = await tts_processor.health_check()
        
        assert health["model_loaded"] is False
        assert health["engine_ready"] is False
        assert health["voice_profiles"] == 1
        assert health["test_result"] == "model_not_loaded"
    
    @pytest.mark.asyncio
    async def test_shutdown(self, tts_processor):
        """Test shutdown process."""
        # Set some initial state
        tts_processor.stop_event.clear()
        tts_processor.model_loaded = True
        
        await tts_processor.shutdown()
        
        assert tts_processor.stop_event.is_set()
        assert tts_processor.model_loaded is False
        assert tts_processor.serve_engine is None


class TestHiggsAudioTTSIntegration:
    """Integration tests for HiggsAudioTTS (requires mocking higgs-audio)."""
    
    @pytest.fixture
    def mock_higgs_engine(self):
        """Mock higgs-audio serving engine."""
        engine = Mock()
        
        # Mock output with audio
        mock_output = Mock()
        mock_output.audio = np.random.randn(24000).astype(np.float32)
        
        engine.generate.return_value = mock_output
        return engine
    
    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        return Settings(
            models__higgs_audio_model="test-model",
            models__higgs_audio_tokenizer="test-tokenizer",
            audio__sample_rate=24000,
        )
    
    @pytest.mark.asyncio
    async def test_initialize_with_mock(self, test_settings):
        """Test initialization with mocked higgs-audio."""
        with patch('models.tts_module.HiggsAudioServeEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            
            tts = HiggsAudioTTS(settings=test_settings)
            
            # Mock the _prewarm_engine and _measure_ttfa methods to avoid actual generation
            with patch.object(tts, '_prewarm_engine', new_callable=AsyncMock):
                with patch.object(tts, '_measure_ttfa', new_callable=AsyncMock):
                    await tts.initialize()
            
            assert tts.model_loaded is True
            assert tts.serve_engine is not None
    
    @pytest.mark.asyncio
    async def test_synthesize_text_with_mock(self, test_settings, mock_higgs_engine):
        """Test text synthesis with mocked engine."""
        tts = HiggsAudioTTS(settings=test_settings)
        tts.serve_engine = mock_higgs_engine
        tts.model_loaded = True
        
        with patch('models.tts_module.ChatMLSample') as mock_sample:
            mock_sample.return_value = Mock()
            
            audio_data = await tts._synthesize_text("Hello world")
            
            assert audio_data is not None
            assert isinstance(audio_data, np.ndarray)
            assert audio_data.dtype == np.float32
    
    def test_synthesize_with_mock(self, test_settings, mock_higgs_engine):
        """Test complete synthesis with mocked engine."""
        tts = HiggsAudioTTS(settings=test_settings)
        tts.serve_engine = mock_higgs_engine
        tts.model_loaded = True
        
        # Create test queue and event
        audio_queue = Queue()
        stop_event = threading.Event()
        
        with patch('models.tts_module.ChatMLSample') as mock_sample:
            mock_sample.return_value = Mock()
            
            # Mock the async synthesis
            with patch.object(tts, '_synthesize_text', return_value=np.random.randn(24000).astype(np.float32)):
                result = tts.synthesize(
                    text="Hello world",
                    audio_chunks=audio_queue,
                    stop_event=stop_event,
                    generation_string="test",
                )
            
            assert result is True
            assert not audio_queue.empty()
    
    def test_synthesize_generator_with_mock(self, test_settings, mock_higgs_engine):
        """Test generator synthesis with mocked engine."""
        tts = HiggsAudioTTS(settings=test_settings)
        tts.serve_engine = mock_higgs_engine
        tts.model_loaded = True
        
        # Create test generator
        def text_generator():
            yield "Hello "
            yield "world "
            yield "from generator"
        
        # Create test queue and event
        audio_queue = Queue()
        stop_event = threading.Event()
        
        with patch('models.tts_module.ChatMLSample') as mock_sample:
            mock_sample.return_value = Mock()
            
            # Mock the synthesis method
            with patch.object(tts, 'synthesize', return_value=True) as mock_synthesize:
                result = tts.synthesize_generator(
                    generator=text_generator(),
                    audio_chunks=audio_queue,
                    stop_event=stop_event,
                    generation_string="test_gen",
                )
                
                assert result is True
                mock_synthesize.assert_called_once()
                # Check that complete text was passed
                args, kwargs = mock_synthesize.call_args
                assert "Hello world from generator" in args


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in TTS module."""
    settings = Settings(
        models__higgs_audio_model="invalid-model",
        models__higgs_audio_tokenizer="invalid-tokenizer",
    )
    
    tts = HiggsAudioTTS(settings=settings)
    
    # Test initialization error
    with pytest.raises(Exception):
        await tts.initialize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])