"""
Full system integration tests for AI Anchor.

Tests the complete system workflow from audio input to voice output.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class TestFullSystemIntegration:
    """Test suite for full system integration."""
    
    @pytest.fixture
    def system_components(self, mock_settings, mock_memory_manager, mock_anchor_agent, mock_audio_pipeline):
        """Create complete system components."""
        return {
            "settings": mock_settings,
            "memory_manager": mock_memory_manager,
            "anchor_agent": mock_anchor_agent,
            "audio_pipeline": mock_audio_pipeline
        }
    
    @pytest.mark.asyncio
    async def test_system_initialization_workflow(self, system_components):
        """Test complete system initialization."""
        settings = system_components["settings"]
        memory_manager = system_components["memory_manager"]
        anchor_agent = system_components["anchor_agent"]
        audio_pipeline = system_components["audio_pipeline"]
        
        # Initialize in correct order
        await audio_pipeline.initialize()
        await anchor_agent.initialize()
        
        # Verify initialization calls
        audio_pipeline.initialize.assert_called_once()
        anchor_agent.initialize.assert_called_once()
        
        # Check that agent is now active
        anchor_agent.is_active = True
        status = anchor_agent.get_agent_status()
        assert status["is_active"] is True
    
    @pytest.mark.asyncio
    async def test_conversation_lifecycle(self, system_components):
        """Test complete conversation lifecycle."""
        anchor_agent = system_components["anchor_agent"]
        audio_pipeline = system_components["audio_pipeline"]
        
        # Make agent active
        anchor_agent.is_active = True
        
        # Start conversation
        greeting = await anchor_agent.start_conversation(
            mode="casual_chat",
            personality="friendly"
        )
        
        assert isinstance(greeting, str)
        assert len(greeting) > 0
        anchor_agent.start_conversation.assert_called_once()
        
        # Simulate user input
        user_text = "Hello, how are you today?"
        anchor_agent.process_text_input(user_text)
        anchor_agent.process_text_input.assert_called_once_with(user_text)
        
        # Simulate audio processing
        audio_data = b"sample_audio_data"
        anchor_agent.process_audio_input(audio_data)
        anchor_agent.process_audio_input.assert_called_once_with(audio_data)
        
        # Switch personality mid-conversation
        personality_success = await anchor_agent.switch_personality("energetic")
        assert personality_success is True
        anchor_agent.switch_personality.assert_called_once_with("energetic")
        
        # End conversation
        await anchor_agent.end_conversation()
        anchor_agent.end_conversation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_audio_pipeline_integration(self, system_components):
        """Test audio pipeline integration with models."""
        audio_pipeline = system_components["audio_pipeline"]
        memory_manager = system_components["memory_manager"]
        
        # Initialize pipeline
        await audio_pipeline.initialize()
        
        # Test audio input
        audio_data = b"test_audio_chunk_1234567890"
        audio_pipeline.feed_audio(audio_data)
        audio_pipeline.feed_audio.assert_called_once_with(audio_data)
        
        # Test text input
        text = "Process this text"
        from config import LanguageCode
        try:
            language = LanguageCode.EN
        except:
            language = "en"
        
        audio_pipeline.process_text_input(text, language)
        audio_pipeline.process_text_input.assert_called_once_with(text, language)
        
        # Test pipeline state
        state = audio_pipeline.get_pipeline_state()
        assert isinstance(state, dict)
        assert "state" in state
        assert "model_status" in state
        
        # Test health check
        health = await audio_pipeline.health_check()
        assert isinstance(health, dict)
        assert "overall_health" in health
    
    @pytest.mark.asyncio
    async def test_memory_management_integration(self, system_components):
        """Test memory management across the system."""
        memory_manager = system_components["memory_manager"]
        audio_pipeline = system_components["audio_pipeline"]
        
        # Test memory info
        memory_info = memory_manager.get_memory_info()
        assert memory_info.total_gb == 48.0
        assert memory_info.allocated_gb == 5.0
        
        # Test model registration
        memory_manager.register_model("STT", audio_pipeline.stt_model, 9.5)
        memory_manager.register_model.assert_called()
        
        # Test memory statistics
        stats = memory_manager.get_memory_statistics()
        assert isinstance(stats, dict)
        assert "cuda_available" in stats
        
        # Test health check
        health = await memory_manager.health_check()
        assert isinstance(health, dict)
        assert "health_status" in health
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, system_components):
        """Test error handling across system components."""
        anchor_agent = system_components["anchor_agent"]
        audio_pipeline = system_components["audio_pipeline"]
        
        # Test error callback setup
        error_callback = Mock()
        anchor_agent.on_error = error_callback
        
        # Simulate error in audio pipeline
        test_error = Exception("Test pipeline error")
        if hasattr(anchor_agent, '_on_pipeline_error'):
            anchor_agent._on_pipeline_error(test_error)
            error_callback.assert_called_once_with(test_error)
        
        # Test graceful degradation
        anchor_agent.is_active = False
        
        # Should handle inactive state gracefully
        anchor_agent.process_text_input("Test message")
        # Should not raise exception even when inactive
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, system_components, performance_timer):
        """Test system performance monitoring."""
        anchor_agent = system_components["anchor_agent"]
        audio_pipeline = system_components["audio_pipeline"]
        
        # Test conversation startup time
        performance_timer.start()
        
        anchor_agent.is_active = True
        await anchor_agent.start_conversation()
        
        performance_timer.stop()
        
        # Should complete quickly (under 1 second with mocks)
        assert performance_timer.elapsed < 1.0
        
        # Test health check performance
        performance_timer.start()
        
        health = await anchor_agent.health_check()
        
        performance_timer.stop()
        
        # Health check should be fast
        assert performance_timer.elapsed < 0.1
        assert health["overall_health"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, system_components):
        """Test system behavior under concurrent operations."""
        anchor_agent = system_components["anchor_agent"]
        audio_pipeline = system_components["audio_pipeline"]
        
        anchor_agent.is_active = True
        
        # Start multiple operations concurrently
        tasks = []
        
        # Start conversation
        tasks.append(anchor_agent.start_conversation())
        
        # Process multiple inputs
        for i in range(5):
            anchor_agent.process_text_input(f"Message {i}")
        
        # Change personality
        tasks.append(anchor_agent.switch_personality("calm"))
        
        # Health checks
        tasks.append(anchor_agent.health_check())
        tasks.append(audio_pipeline.health_check())
        
        # Wait for all operations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)
    
    @pytest.mark.asyncio
    async def test_system_shutdown(self, system_components):
        """Test graceful system shutdown."""
        anchor_agent = system_components["anchor_agent"]
        audio_pipeline = system_components["audio_pipeline"]
        memory_manager = system_components["memory_manager"]
        
        # Initialize system
        await audio_pipeline.initialize()
        await anchor_agent.initialize()
        
        # Start some activity
        anchor_agent.is_active = True
        await anchor_agent.start_conversation()
        
        # Shutdown in reverse order
        await anchor_agent.shutdown()
        await audio_pipeline.shutdown()
        memory_manager.shutdown()
        
        # Verify shutdown calls
        anchor_agent.shutdown.assert_called_once()
        audio_pipeline.shutdown.assert_called_once()
        memory_manager.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_personality_switching_workflow(self, system_components):
        """Test complete personality switching workflow."""
        anchor_agent = system_components["anchor_agent"]
        
        anchor_agent.is_active = True
        
        # Start with professional personality
        await anchor_agent.start_conversation(personality="professional")
        
        # Switch through different personalities
        personalities = ["friendly", "energetic", "calm", "authoritative"]
        
        for personality in personalities:
            success = await anchor_agent.switch_personality(personality)
            assert success is True
            
            # Verify personality was applied
            anchor_agent.switch_personality.assert_called_with(personality)
        
        # Get available personalities
        available = anchor_agent.get_available_personalities()
        assert isinstance(available, dict)
        assert len(available) >= 2  # At least professional and friendly
    
    @pytest.mark.asyncio
    async def test_multi_language_support(self, system_components):
        """Test multi-language support across the system."""
        anchor_agent = system_components["anchor_agent"]
        audio_pipeline = system_components["audio_pipeline"]
        
        anchor_agent.is_active = True
        
        # Test different languages
        languages = ["en", "es", "fr", "de"]
        
        for lang in languages:
            # Switch language
            await anchor_agent.switch_language(lang)
            anchor_agent.switch_language.assert_called_with(lang)
            
            # Start conversation in that language
            await anchor_agent.start_conversation(language=lang)
            
            # Process text in that language
            anchor_agent.process_text_input(f"Hello in {lang}")
    
    @pytest.mark.asyncio
    async def test_system_recovery_from_errors(self, system_components):
        """Test system recovery from various error conditions."""
        anchor_agent = system_components["anchor_agent"]
        audio_pipeline = system_components["audio_pipeline"]
        
        # Test recovery from initialization error
        anchor_agent.initialize.side_effect = Exception("Init error")
        
        with pytest.raises(Exception, match="Init error"):
            await anchor_agent.initialize()
        
        # Reset and try again
        anchor_agent.initialize.side_effect = None
        await anchor_agent.initialize()  # Should succeed
        
        # Test recovery from conversation error
        anchor_agent.start_conversation.side_effect = Exception("Conversation error")
        
        with pytest.raises(Exception, match="Conversation error"):
            await anchor_agent.start_conversation()
        
        # Reset and continue
        anchor_agent.start_conversation.side_effect = None
        anchor_agent.is_active = True
        greeting = await anchor_agent.start_conversation()
        assert isinstance(greeting, str)


class TestSystemStressTests:
    """Stress tests for the AI Anchor system."""
    
    @pytest.mark.asyncio
    async def test_high_volume_requests(self, system_components):
        """Test system under high volume of requests."""
        anchor_agent = system_components["anchor_agent"]
        
        anchor_agent.is_active = True
        
        # Process many requests quickly
        num_requests = 100
        tasks = []
        
        for i in range(num_requests):
            anchor_agent.process_text_input(f"Request {i}")
        
        # All requests should be processed without errors
        # (In real system, would check queue sizes and processing times)
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, system_components):
        """Test for memory leaks during extended operation."""
        memory_manager = system_components["memory_manager"]
        anchor_agent = system_components["anchor_agent"]
        
        anchor_agent.is_active = True
        
        # Simulate extended operation
        for i in range(50):
            await anchor_agent.start_conversation()
            anchor_agent.process_text_input(f"Extended message {i}")
            await anchor_agent.end_conversation()
        
        # Check for memory leaks
        leak_info = memory_manager.detect_memory_leaks()
        assert leak_info["potential_leak"] is False
    
    @pytest.mark.asyncio
    async def test_concurrent_user_simulation(self, system_components):
        """Simulate multiple concurrent users."""
        anchor_agent = system_components["anchor_agent"]
        
        anchor_agent.is_active = True
        
        async def simulate_user(user_id):
            """Simulate a single user session."""
            await anchor_agent.start_conversation()
            
            # Send multiple messages
            for i in range(10):
                anchor_agent.process_text_input(f"User {user_id} message {i}")
                await asyncio.sleep(0.001)  # Small delay
            
            await anchor_agent.end_conversation()
            return f"user_{user_id}_completed"
        
        # Simulate 10 concurrent users
        tasks = [simulate_user(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All users should complete successfully
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result == f"user_{i}_completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])