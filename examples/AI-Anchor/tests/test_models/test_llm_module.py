"""
Unit tests for LLM module (Ollama + huihui-ai).
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import ConnectionError, Timeout
from io import StringIO

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.llm_module import HuihuiAILLM, ConversationHistory
from config import Settings, LanguageCode


class TestConversationHistory:
    """Test suite for ConversationHistory class."""
    
    def test_initialization(self):
        """Test conversation history initialization."""
        history = ConversationHistory(max_history=5, context_window=2048)
        
        assert history.max_history == 5
        assert history.context_window == 2048
        assert len(history.messages) == 0
        assert history.total_tokens == 0
    
    def test_add_message(self):
        """Test adding messages to history."""
        history = ConversationHistory()
        
        history.add_message("user", "Hello world")
        history.add_message("assistant", "Hi there!")
        
        assert len(history.messages) == 2
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "Hello world"
        assert history.messages[1]["role"] == "assistant"
        assert history.messages[1]["content"] == "Hi there!"
        assert history.total_tokens > 0
    
    def test_history_trimming_max_messages(self):
        """Test trimming history when max messages exceeded."""
        history = ConversationHistory(max_history=3)
        
        # Add more messages than limit
        for i in range(5):
            history.add_message("user", f"Message {i}")
        
        # Should only keep last 3 messages
        assert len(history.messages) == 3
        assert history.messages[0]["content"] == "Message 2"
        assert history.messages[-1]["content"] == "Message 4"
    
    def test_history_trimming_context_window(self):
        """Test trimming history when context window exceeded."""
        history = ConversationHistory(context_window=50)  # Very small window
        
        # Add large message that exceeds context window
        large_message = "This is a very long message " * 20  # Much larger than 50 token estimate
        history.add_message("user", large_message)
        history.add_message("user", "Short message")
        
        # Should trim to fit context window
        assert len(history.messages) >= 1  # At least the last message should remain
    
    def test_get_messages(self):
        """Test getting messages returns copy."""
        history = ConversationHistory()
        history.add_message("user", "Test message")
        
        messages = history.get_messages()
        messages.append({"role": "system", "content": "Injected"})
        
        # Original should be unchanged
        assert len(history.messages) == 1
        assert history.messages[0]["content"] == "Test message"
    
    def test_clear(self):
        """Test clearing conversation history."""
        history = ConversationHistory()
        history.add_message("user", "Test")
        history.add_message("assistant", "Response")
        
        assert len(history.messages) == 2
        assert history.total_tokens > 0
        
        history.clear()
        
        assert len(history.messages) == 0
        assert history.total_tokens == 0
    
    def test_get_context_info(self):
        """Test getting context information."""
        history = ConversationHistory(max_history=10, context_window=2048)
        history.add_message("user", "Test message")
        
        info = history.get_context_info()
        
        assert isinstance(info, dict)
        assert info["message_count"] == 1
        assert info["estimated_tokens"] > 0
        assert info["max_history"] == 10
        assert info["context_window"] == 2048


class TestHuihuiAILLM:
    """Test suite for HuihuiAILLM class."""
    
    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        return Settings(
            debug=True,
            models__ollama_model="test-model",
            conversation__max_history=5,
            conversation__context_window_size=2048,
        )
    
    @pytest.fixture
    def llm_processor(self, test_settings):
        """Create LLM processor instance."""
        return HuihuiAILLM(
            settings=test_settings,
            system_prompt="You are a test AI assistant.",
        )
    
    def test_initialization(self, test_settings):
        """Test LLM processor initialization."""
        llm = HuihuiAILLM(
            settings=test_settings,
            system_prompt="Test prompt",
        )
        
        assert llm.settings == test_settings
        assert llm.model == "test-model"
        assert llm.system_prompt == "Test prompt"
        assert llm.base_url == "http://127.0.0.1:11434"
        assert llm.connection_ok is False
        assert llm.client_initialized is False
        assert len(llm._active_requests) == 0
        assert isinstance(llm.conversation_history, ConversationHistory)
    
    def test_initialization_with_default_system_prompt(self, test_settings):
        """Test initialization with default system prompt."""
        llm = HuihuiAILLM(settings=test_settings)
        
        assert llm.system_prompt is not None
        assert "AI news anchor" in llm.system_prompt
    
    def test_format_messages(self, llm_processor):
        """Test message formatting."""
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        
        messages = llm_processor._format_messages(
            text="Current question",
            history=history,
            use_system_prompt=True,
        )
        
        assert len(messages) == 4  # system + history (2) + current user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Previous question"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Previous answer"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Current question"
    
    def test_format_messages_no_system_prompt(self, llm_processor):
        """Test message formatting without system prompt."""
        messages = llm_processor._format_messages(
            text="Test question",
            use_system_prompt=False,
        )
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test question"
    
    def test_create_request_payload(self, llm_processor):
        """Test creating request payload."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        
        payload = llm_processor._create_request_payload(
            messages=messages,
            temperature=0.5,
            max_tokens=512,
            stream=True,
        )
        
        assert payload["model"] == "test-model"
        assert payload["stream"] is True
        assert "prompt" in payload
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 512
        
        # Check prompt formatting
        prompt = payload["prompt"]
        assert "[SYSTEM_PROMPT]" in prompt
        assert "[INST]" in prompt
        assert "[/INST]" in prompt
    
    @patch('requests.Session')
    def test_check_ollama_connection_success(self, mock_session_class, llm_processor):
        """Test successful Ollama connection check."""
        # Mock session and response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        
        llm_processor.session = mock_session
        
        result = llm_processor._check_ollama_connection()
        
        assert result is True
        mock_session.get.assert_called_once()
    
    @patch('requests.Session')
    def test_check_ollama_connection_failure(self, mock_session_class, llm_processor):
        """Test failed Ollama connection check."""
        # Mock session to raise connection error
        mock_session = Mock()
        mock_session.get.side_effect = ConnectionError("Connection failed")
        
        llm_processor.session = mock_session
        
        result = llm_processor._check_ollama_connection()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_run_ollama_ps_success(self, mock_run, llm_processor):
        """Test successful ollama ps command."""
        # Mock successful subprocess run
        mock_result = Mock()
        mock_result.stdout = "MODEL    \tSIZE  \tPROCESSOR\ntest-model\t7.0 GB\t100% GPU"
        mock_run.return_value = mock_result
        
        result = llm_processor._run_ollama_ps()
        
        assert result is True
        mock_run.assert_called_once_with(
            ["ollama", "ps"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0
        )
    
    @patch('subprocess.run')
    def test_run_ollama_ps_not_found(self, mock_run, llm_processor):
        """Test ollama ps when command not found."""
        mock_run.side_effect = FileNotFoundError()
        
        result = llm_processor._run_ollama_ps()
        
        assert result is False
    
    def test_cancel_generation_no_requests(self, llm_processor):
        """Test cancelling when no requests active."""
        result = llm_processor.cancel_generation()
        
        assert result is False
    
    def test_cancel_generation_specific_request(self, llm_processor):
        """Test cancelling specific request."""
        # Add a mock request
        request_id = "test-request"
        mock_response = Mock()
        llm_processor._active_requests[request_id] = {
            "start_time": 12345,
            "response": mock_response,
        }
        
        result = llm_processor.cancel_generation(request_id)
        
        assert result is True
        assert request_id not in llm_processor._active_requests
        mock_response.close.assert_called_once()
    
    def test_cancel_generation_all_requests(self, llm_processor):
        """Test cancelling all requests."""
        # Add multiple mock requests
        for i in range(3):
            request_id = f"test-request-{i}"
            mock_response = Mock()
            llm_processor._active_requests[request_id] = {
                "start_time": 12345,
                "response": mock_response,
            }
        
        result = llm_processor.cancel_generation()  # Cancel all
        
        assert result is True
        assert len(llm_processor._active_requests) == 0
    
    def test_conversation_history_management(self, llm_processor):
        """Test conversation history management."""
        # Initially empty
        assert len(llm_processor.get_conversation_history()) == 0
        
        # Add some messages manually to test
        llm_processor.conversation_history.add_message("user", "Hello")
        llm_processor.conversation_history.add_message("assistant", "Hi there")
        
        history = llm_processor.get_conversation_history()
        assert len(history) == 2
        assert history[0]["content"] == "Hello"
        assert history[1]["content"] == "Hi there"
        
        # Clear history
        llm_processor.clear_conversation_history()
        assert len(llm_processor.get_conversation_history()) == 0
    
    def test_set_system_prompt(self, llm_processor):
        """Test setting system prompt."""
        original_prompt = llm_processor.system_prompt
        new_prompt = "You are a different AI assistant."
        
        llm_processor.set_system_prompt(new_prompt)
        
        assert llm_processor.system_prompt == new_prompt
        assert llm_processor.system_prompt != original_prompt
    
    def test_get_statistics(self, llm_processor):
        """Test getting statistics."""
        stats = llm_processor.get_statistics()
        
        assert isinstance(stats, dict)
        assert "requests_completed" in stats
        assert "total_generation_time" in stats
        assert "total_tokens_generated" in stats
        assert "average_tokens_per_second" in stats
        assert "errors" in stats
        assert "conversation_info" in stats
        assert "connection_ok" in stats
        assert "active_requests" in stats
        
        assert stats["connection_ok"] is False
        assert stats["active_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_health_check_no_connection(self, llm_processor):
        """Test health check when not connected."""
        health = await llm_processor.health_check()
        
        assert health["connection_initialized"] is False
        assert health["connection_ok"] is False
        assert health["model"] == "test-model"
        assert health["test_result"] == "connection_failed"
    
    @pytest.mark.asyncio
    async def test_shutdown(self, llm_processor):
        """Test shutdown process."""
        # Create mock session
        mock_session = Mock()
        llm_processor.session = mock_session
        llm_processor.connection_ok = True
        llm_processor.client_initialized = True
        
        # Add mock active request
        llm_processor._active_requests["test"] = {"response": Mock()}
        
        await llm_processor.shutdown()
        
        assert llm_processor.session is None
        assert llm_processor.connection_ok is False
        assert llm_processor.client_initialized is False
        assert len(llm_processor._active_requests) == 0
        mock_session.close.assert_called_once()


class TestHuihuiAILLMIntegration:
    """Integration tests for HuihuiAILLM (requires mocking Ollama API)."""
    
    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        return Settings(
            models__ollama_model="test-model",
            conversation__max_history=5,
        )
    
    @pytest.fixture
    def mock_successful_response(self):
        """Create mock successful streaming response."""
        # Create mock response that simulates Ollama streaming JSON
        response_lines = [
            '{"response": "Hello"}',
            '{"response": " there"}',
            '{"response": "!"}',
            '{"done": true}'
        ]
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [line.encode() for line in response_lines]
        
        return mock_response
    
    @patch('requests.Session')
    def test_lazy_initialize_success(self, mock_session_class, test_settings):
        """Test successful lazy initialization."""
        # Mock session and connection check
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        llm = HuihuiAILLM(settings=test_settings)
        
        # Mock successful connection check
        with patch.object(llm, '_check_ollama_connection', return_value=True):
            result = llm._lazy_initialize()
        
        assert result is True
        assert llm.connection_ok is True
        assert llm.client_initialized is True
        assert llm.session is not None
    
    @patch('requests.Session')
    def test_lazy_initialize_with_ollama_ps_fallback(self, mock_session_class, test_settings):
        """Test lazy initialization with ollama ps fallback."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        llm = HuihuiAILLM(settings=test_settings)
        
        # Mock first connection check fails, then ollama ps succeeds, then second check succeeds
        connection_check_results = [False, True]  # First fails, second succeeds
        with patch.object(llm, '_check_ollama_connection', side_effect=connection_check_results):
            with patch.object(llm, '_run_ollama_ps', return_value=True):
                with patch('time.sleep'):  # Skip actual sleep
                    result = llm._lazy_initialize()
        
        assert result is True
        assert llm.connection_ok is True
    
    @patch('requests.Session')
    def test_generate_success(self, mock_session_class, test_settings, mock_successful_response):
        """Test successful text generation."""
        mock_session = Mock()
        mock_session.post.return_value = mock_successful_response
        mock_session_class.return_value = mock_session
        
        llm = HuihuiAILLM(settings=test_settings)
        llm.session = mock_session
        llm.connection_ok = True
        llm.client_initialized = True
        
        # Generate text
        tokens = list(llm.generate("Hello world", temperature=0.3))
        
        assert tokens == ["Hello", " there", "!"]
        mock_session.post.assert_called_once()
        
        # Check conversation history was updated
        history = llm.get_conversation_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello world"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hello there!"
    
    @pytest.mark.asyncio
    async def test_prewarm_success(self, test_settings):
        """Test successful prewarming."""
        llm = HuihuiAILLM(settings=test_settings)
        
        # Mock successful initialization and generation
        with patch.object(llm, '_lazy_initialize', return_value=True):
            with patch.object(llm, 'generate') as mock_generate:
                mock_generate.return_value = iter(["OK"])
                
                result = await llm.prewarm()
        
        assert result is True
        mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_prewarm_with_retries(self, test_settings):
        """Test prewarming with retries."""
        llm = HuihuiAILLM(settings=test_settings)
        
        # Mock initialization success but first generation fails
        with patch.object(llm, '_lazy_initialize', return_value=True):
            with patch.object(llm, 'generate') as mock_generate:
                # First call raises exception, second succeeds
                mock_generate.side_effect = [ConnectionError("Failed"), iter(["OK"])]
                
                with patch('asyncio.sleep'):  # Skip actual sleep
                    result = await llm.prewarm(max_retries=1)
        
        assert result is True
        assert mock_generate.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])