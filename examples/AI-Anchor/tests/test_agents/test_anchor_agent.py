"""
Unit tests for Anchor Agent.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from agents.anchor_agent import (
    AnchorAgent,
    AnchorPersonality,
    ConversationMode,
    VoicePersonality,
    ConversationContext
)
from core.audio_pipeline import AudioPipelineManager, GenerationSession
from core.memory_manager import MemoryManager
from config import Settings, LanguageCode, TranscriptionResult, AudioChunk, AudioFormat


class TestVoicePersonality:
    """Test suite for VoicePersonality dataclass."""
    
    def test_voice_personality_creation(self):
        """Test VoicePersonality creation."""
        personality = VoicePersonality(
            name="Test Personality",
            personality=AnchorPersonality.PROFESSIONAL,
            voice_profile="test_voice",
            language=LanguageCode.EN,
            temperature=0.5,
            system_prompt="Test prompt",
            greeting_message="Hello test"
        )
        
        assert personality.name == "Test Personality"
        assert personality.personality == AnchorPersonality.PROFESSIONAL
        assert personality.voice_profile == "test_voice"
        assert personality.language == LanguageCode.EN
        assert personality.temperature == 0.5
        assert personality.system_prompt == "Test prompt"
        assert personality.greeting_message == "Hello test"
        
        # Check default values
        assert personality.speaking_rate == 1.0
        assert personality.pitch_variation == 0.5
        assert personality.emotion_intensity == 0.5


class TestConversationContext:
    """Test suite for ConversationContext dataclass."""
    
    def test_conversation_context_creation(self):
        """Test ConversationContext creation."""
        personality = VoicePersonality(
            name="Test",
            personality=AnchorPersonality.FRIENDLY,
            voice_profile="test",
            language=LanguageCode.EN
        )
        
        context = ConversationContext(
            session_id="test-session",
            mode=ConversationMode.CASUAL_CHAT,
            current_personality=personality,
            topic_context="Testing",
        )
        
        assert context.session_id == "test-session"
        assert context.mode == ConversationMode.CASUAL_CHAT
        assert context.current_personality == personality
        assert context.topic_context == "Testing"
        assert context.total_exchanges == 0
        assert len(context.conversation_history) == 0
    
    def test_add_exchange(self):
        """Test adding conversation exchanges."""
        personality = VoicePersonality(
            name="Test",
            personality=AnchorPersonality.FRIENDLY,
            voice_profile="test",
            language=LanguageCode.EN
        )
        
        context = ConversationContext(
            session_id="test-session",
            mode=ConversationMode.CASUAL_CHAT,
            current_personality=personality,
        )
        
        # Add first exchange
        context.add_exchange("Hello", "Hi there!")
        assert context.total_exchanges == 1
        assert len(context.conversation_history) == 1
        
        exchange = context.conversation_history[0]
        assert exchange["user_input"] == "Hello"
        assert exchange["anchor_response"] == "Hi there!"
        assert exchange["exchange_number"] == 1
        assert "timestamp" in exchange
        
        # Add more exchanges
        for i in range(2, 12):  # Add 10 more exchanges
            context.add_exchange(f"Message {i}", f"Response {i}")
        
        # Should keep only last 10 exchanges
        assert context.total_exchanges == 11
        assert len(context.conversation_history) == 10
        assert context.conversation_history[0]["exchange_number"] == 2  # First should be removed


class TestAnchorAgent:
    """Test suite for AnchorAgent class."""
    
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
    def mock_memory_manager(self):
        """Create mock memory manager."""
        return Mock(spec=MemoryManager)
    
    @pytest.fixture
    def mock_audio_pipeline(self):
        """Create mock audio pipeline."""
        pipeline = Mock(spec=AudioPipelineManager)
        pipeline.get_pipeline_state.return_value = {
            "state": "idle",
            "current_session": None,
            "session_counter": 0,
        }
        pipeline.health_check = AsyncMock(return_value={"overall_health": "healthy"})
        return pipeline
    
    @pytest.fixture
    def anchor_agent(self, test_settings, mock_memory_manager, mock_audio_pipeline):
        """Create anchor agent instance."""
        agent = AnchorAgent(
            settings=test_settings,
            memory_manager=mock_memory_manager,
            audio_pipeline=mock_audio_pipeline,
        )
        return agent
    
    def test_initialization(self, test_settings, mock_memory_manager, mock_audio_pipeline):
        """Test anchor agent initialization."""
        agent = AnchorAgent(
            settings=test_settings,
            memory_manager=mock_memory_manager,
            audio_pipeline=mock_audio_pipeline,
        )
        
        assert agent.settings == test_settings
        assert agent.memory_manager == mock_memory_manager
        assert agent.audio_pipeline == mock_audio_pipeline
        assert agent.is_active is False
        assert agent.current_context is None
        assert agent.session_counter == 0
        
        # Check personalities were initialized
        assert len(agent.personalities) == 5
        assert "professional" in agent.personalities
        assert "friendly" in agent.personalities
        assert "energetic" in agent.personalities
        assert "calm" in agent.personalities
        assert "authoritative" in agent.personalities
        
        # Check current personality
        assert agent.current_personality == agent.personalities["professional"]
    
    def test_personality_initialization(self, anchor_agent):
        """Test personality initialization."""
        personalities = anchor_agent.personalities
        
        # Test professional personality
        prof = personalities["professional"]
        assert prof.name == "Professional"
        assert prof.personality == AnchorPersonality.PROFESSIONAL
        assert prof.voice_profile == "news_anchor_neutral"
        assert prof.language == LanguageCode.EN
        assert prof.temperature == 0.3
        assert "professional news anchor" in prof.system_prompt.lower()
        assert prof.speaking_rate == 0.9
        
        # Test friendly personality
        friendly = personalities["friendly"]
        assert friendly.name == "Friendly"
        assert friendly.personality == AnchorPersonality.FRIENDLY
        assert friendly.temperature == 0.6
        assert friendly.speaking_rate == 1.0
        
        # Test energetic personality
        energetic = personalities["energetic"]
        assert energetic.name == "Energetic"
        assert energetic.personality == AnchorPersonality.ENERGETIC
        assert energetic.temperature == 0.8
        assert energetic.speaking_rate == 1.1
    
    @pytest.mark.asyncio
    async def test_initialize(self, anchor_agent):
        """Test agent initialization."""
        # Mock the _apply_personality method
        anchor_agent._apply_personality = AsyncMock()
        
        await anchor_agent.initialize()
        
        assert anchor_agent.is_active is True
        anchor_agent._apply_personality.assert_called_once_with(anchor_agent.current_personality)
    
    @pytest.mark.asyncio
    async def test_initialize_error(self, anchor_agent):
        """Test initialization error handling."""
        error_callback = Mock()
        anchor_agent.on_error = error_callback
        
        # Mock _apply_personality to raise an error
        anchor_agent._apply_personality = AsyncMock(side_effect=Exception("Test error"))
        
        with pytest.raises(Exception, match="Test error"):
            await anchor_agent.initialize()
        
        assert anchor_agent.is_active is False
        error_callback.assert_called_once()
    
    def test_setup_pipeline_callbacks(self, anchor_agent):
        """Test pipeline callback setup."""
        anchor_agent._setup_pipeline_callbacks()
        
        # Check callbacks were set
        assert anchor_agent.audio_pipeline.on_transcription == anchor_agent._on_transcription_received
        assert anchor_agent.audio_pipeline.on_response_start == anchor_agent._on_response_start
        assert anchor_agent.audio_pipeline.on_response_chunk == anchor_agent._on_response_chunk
        assert anchor_agent.audio_pipeline.on_audio_chunk == anchor_agent._on_audio_chunk_generated
        assert anchor_agent.audio_pipeline.on_session_complete == anchor_agent._on_session_complete
        assert anchor_agent.audio_pipeline.on_error == anchor_agent._on_pipeline_error
    
    @pytest.mark.asyncio
    async def test_start_conversation(self, anchor_agent):
        """Test starting a conversation."""
        anchor_agent.is_active = True
        start_callback = Mock()
        anchor_agent.on_conversation_start = start_callback
        
        greeting = await anchor_agent.start_conversation(
            mode=ConversationMode.INTERVIEW,
            personality="friendly",
            language=LanguageCode.ES
        )
        
        # Check session was created
        assert anchor_agent.current_context is not None
        assert anchor_agent.current_context.session_id == "conversation-1"
        assert anchor_agent.current_context.mode == ConversationMode.INTERVIEW
        assert anchor_agent.session_counter == 1
        
        # Check statistics updated
        assert anchor_agent.session_stats["total_sessions"] == 1
        
        # Check callback was called
        start_callback.assert_called_once_with(anchor_agent.current_context)
        
        # Check greeting returned
        assert isinstance(greeting, str)
        assert len(greeting) > 0
    
    @pytest.mark.asyncio
    async def test_start_conversation_not_active(self, anchor_agent):
        """Test starting conversation when agent not active."""
        anchor_agent.is_active = False
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await anchor_agent.start_conversation()
    
    @pytest.mark.asyncio
    async def test_end_conversation(self, anchor_agent):
        """Test ending a conversation."""
        anchor_agent.is_active = True
        end_callback = Mock()
        anchor_agent.on_conversation_end = end_callback
        
        # Start a conversation first
        await anchor_agent.start_conversation()
        assert anchor_agent.current_context is not None
        
        # End the conversation
        await anchor_agent.end_conversation()
        
        assert anchor_agent.current_context is None
        end_callback.assert_called_once()
        anchor_agent.audio_pipeline.abort_current_generation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_switch_personality(self, anchor_agent):
        """Test switching personality."""
        anchor_agent.is_active = True
        personality_callback = Mock()
        anchor_agent.on_personality_change = personality_callback
        anchor_agent._apply_personality = AsyncMock()
        
        # Switch to friendly personality
        success = await anchor_agent.switch_personality("friendly")
        
        assert success is True
        assert anchor_agent.current_personality.name == "Friendly"
        assert anchor_agent.session_stats["personality_switches"] == 1
        
        # Check callbacks
        anchor_agent._apply_personality.assert_called_once()
        personality_callback.assert_called_once_with(anchor_agent.current_personality)
    
    @pytest.mark.asyncio
    async def test_switch_personality_unknown(self, anchor_agent):
        """Test switching to unknown personality."""
        success = await anchor_agent.switch_personality("unknown")
        
        assert success is False
        assert anchor_agent.current_personality.name == "Professional"  # Should remain unchanged
        assert anchor_agent.session_stats["personality_switches"] == 0
    
    @pytest.mark.asyncio
    async def test_switch_personality_error(self, anchor_agent):
        """Test personality switch error handling."""
        anchor_agent.is_active = True
        error_callback = Mock()
        anchor_agent.on_error = error_callback
        anchor_agent._apply_personality = AsyncMock(side_effect=Exception("Test error"))
        
        success = await anchor_agent.switch_personality("friendly")
        
        assert success is False
        error_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_apply_personality(self, anchor_agent):
        """Test applying personality configuration."""
        personality = anchor_agent.personalities["energetic"]
        
        await anchor_agent._apply_personality(personality)
        
        # Check pipeline methods were called
        anchor_agent.audio_pipeline.change_voice_profile.assert_called_once_with(personality.voice_profile)
        anchor_agent.audio_pipeline.change_language.assert_called_once_with(personality.language)
    
    @pytest.mark.asyncio
    async def test_switch_language(self, anchor_agent):
        """Test switching language."""
        await anchor_agent.switch_language(LanguageCode.FR)
        
        anchor_agent.audio_pipeline.change_language.assert_called_once_with(LanguageCode.FR)
        assert anchor_agent.current_personality.language == LanguageCode.FR
    
    def test_process_audio_input(self, anchor_agent):
        """Test processing audio input."""
        anchor_agent.is_active = True
        audio_data = b"test_audio_data"
        
        anchor_agent.process_audio_input(audio_data)
        
        anchor_agent.audio_pipeline.feed_audio.assert_called_once_with(audio_data)
    
    def test_process_audio_input_inactive(self, anchor_agent):
        """Test processing audio input when inactive."""
        anchor_agent.is_active = False
        audio_data = b"test_audio_data"
        
        anchor_agent.process_audio_input(audio_data)
        
        # Should not call pipeline
        anchor_agent.audio_pipeline.feed_audio.assert_not_called()
    
    def test_process_text_input(self, anchor_agent):
        """Test processing text input."""
        anchor_agent.is_active = True
        text = "Hello world"
        
        anchor_agent.process_text_input(text)
        
        anchor_agent.audio_pipeline.process_text_input.assert_called_once_with(
            text, anchor_agent.current_personality.language
        )
    
    def test_abort_current_response(self, anchor_agent):
        """Test aborting current response."""
        anchor_agent.abort_current_response()
        
        anchor_agent.audio_pipeline.abort_current_generation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_conversation_context(self, anchor_agent):
        """Test setting conversation context."""
        anchor_agent.is_active = True
        
        # Start conversation first
        await anchor_agent.start_conversation()
        
        # Set context
        topic = "AI and technology"
        preferences = {"style": "casual", "length": "short"}
        
        anchor_agent.set_conversation_context(topic, preferences)
        
        assert anchor_agent.current_context.topic_context == topic
        assert anchor_agent.current_context.user_preferences == preferences
    
    def test_event_handlers(self, anchor_agent):
        """Test event handler methods."""
        # Test transcription received
        transcription = TranscriptionResult(
            text="Test transcription",
            language=LanguageCode.EN,
            confidence=0.95,
            timestamp=time.time(),
            is_final=True
        )
        
        anchor_agent._on_transcription_received(transcription)
        # Should not raise any errors
        
        # Test response start
        anchor_agent._on_response_start("Test input")
        
        # Test response chunk
        anchor_agent._on_response_chunk("chunk")
        
        # Test audio chunk
        audio_chunk = AudioChunk(
            data=b"audio",
            format=AudioFormat.WAV,
            sample_rate=24000,
            timestamp=time.time(),
            duration=1.0
        )
        anchor_agent._on_audio_chunk_generated(audio_chunk)
        
        # Test pipeline error
        error_callback = Mock()
        anchor_agent.on_error = error_callback
        
        test_error = Exception("Test error")
        anchor_agent._on_pipeline_error(test_error)
        
        error_callback.assert_called_once_with(test_error)
        assert anchor_agent.session_stats["errors"] == 1
    
    @pytest.mark.asyncio
    async def test_session_complete_handler(self, anchor_agent):
        """Test session completion handler."""
        anchor_agent.is_active = True
        response_callback = Mock()
        anchor_agent.on_response_generated = response_callback
        
        # Start conversation
        await anchor_agent.start_conversation()
        
        # Create mock session
        session = GenerationSession(
            session_id="test-session",
            input_text="Hello",
            language=LanguageCode.EN,
        )
        session.llm_response = "Hi there!"
        
        # Handle completion
        anchor_agent._on_session_complete(session)
        
        # Check conversation history updated
        assert anchor_agent.current_context.total_exchanges == 1
        assert len(anchor_agent.current_context.conversation_history) == 1
        
        exchange = anchor_agent.current_context.conversation_history[0]
        assert exchange["user_input"] == "Hello"
        assert exchange["anchor_response"] == "Hi there!"
        
        # Check statistics updated
        assert anchor_agent.session_stats["total_exchanges"] == 1
        
        # Check callback called
        response_callback.assert_called_once_with("Hello", "Hi there!")
    
    def test_get_agent_status(self, anchor_agent):
        """Test getting agent status."""
        status = anchor_agent.get_agent_status()
        
        assert isinstance(status, dict)
        assert "is_active" in status
        assert "current_personality" in status
        assert "current_language" in status
        assert "conversation_active" in status
        assert "available_personalities" in status
        assert "statistics" in status
        
        assert status["is_active"] is False
        assert status["current_personality"] == "Professional"
        assert status["conversation_active"] is False
        assert len(status["available_personalities"]) == 5
    
    @pytest.mark.asyncio
    async def test_health_check(self, anchor_agent):
        """Test health check."""
        health = await anchor_agent.health_check()
        
        assert isinstance(health, dict)
        assert "agent_active" in health
        assert "conversation_active" in health
        assert "current_personality" in health
        assert "personality_count" in health
        assert "pipeline_health" in health
        assert "statistics" in health
        assert "overall_health" in health
        
        assert health["agent_active"] is False
        assert health["personality_count"] == 5
        assert health["pipeline_health"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, anchor_agent):
        """Test getting conversation history."""
        # No conversation active
        history = anchor_agent.get_conversation_history()
        assert history == []
        
        # Start conversation and add exchanges
        anchor_agent.is_active = True
        await anchor_agent.start_conversation()
        
        anchor_agent.current_context.add_exchange("Hello", "Hi")
        anchor_agent.current_context.add_exchange("How are you?", "I'm good!")
        
        history = anchor_agent.get_conversation_history()
        assert len(history) == 2
        assert history[0]["user_input"] == "Hello"
        assert history[1]["user_input"] == "How are you?"
    
    def test_get_available_personalities(self, anchor_agent):
        """Test getting available personalities."""
        personalities = anchor_agent.get_available_personalities()
        
        assert isinstance(personalities, dict)
        assert len(personalities) == 5
        assert "professional" in personalities
        assert "friendly" in personalities
        
        prof_info = personalities["professional"]
        assert prof_info["name"] == "Professional"
        assert prof_info["personality_type"] == "professional"
        assert prof_info["language"] == "en"
        assert "description" in prof_info
    
    @pytest.mark.asyncio
    async def test_shutdown(self, anchor_agent):
        """Test agent shutdown."""
        anchor_agent.is_active = True
        
        # Start conversation
        await anchor_agent.start_conversation()
        assert anchor_agent.current_context is not None
        
        # Shutdown
        await anchor_agent.shutdown()
        
        assert anchor_agent.is_active is False
        assert anchor_agent.current_context is None


class TestAnchorAgentIntegration:
    """Integration tests for AnchorAgent."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test full conversation flow with mocked components."""
        settings = Settings(debug=True)
        memory_manager = Mock(spec=MemoryManager)
        
        # Create mock audio pipeline with more detailed behavior
        audio_pipeline = Mock(spec=AudioPipelineManager)
        audio_pipeline.get_pipeline_state.return_value = {"state": "idle"}
        audio_pipeline.health_check = AsyncMock(return_value={"overall_health": "healthy"})
        
        # Create agent
        agent = AnchorAgent(
            settings=settings,
            memory_manager=memory_manager,
            audio_pipeline=audio_pipeline,
        )
        
        # Initialize
        await agent.initialize()
        assert agent.is_active is True
        
        # Start conversation
        greeting = await agent.start_conversation(
            mode=ConversationMode.CASUAL_CHAT,
            personality="friendly"
        )
        assert isinstance(greeting, str)
        assert agent.current_context is not None
        
        # Process some inputs
        agent.process_text_input("Hello, how are you?")
        agent.process_audio_input(b"audio_data")
        
        # Switch personality mid-conversation
        success = await agent.switch_personality("energetic")
        assert success is True
        
        # Check status
        status = agent.get_agent_status()
        assert status["conversation_active"] is True
        assert status["current_personality"] == "Energetic"
        
        # Health check
        health = await agent.health_check()
        assert health["overall_health"] == "healthy"
        
        # End conversation
        await agent.end_conversation()
        assert agent.current_context is None
        
        # Shutdown
        await agent.shutdown()
        assert agent.is_active is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])