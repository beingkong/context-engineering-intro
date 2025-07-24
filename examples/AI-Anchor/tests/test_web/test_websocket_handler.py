"""
Unit tests for WebSocket Handler.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from web.websocket_handler import (
    WebSocketHandler,
    MessageType
)


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.messages_sent = []
        self.closed = False
        self.client_state = "CONNECTED"
    
    async def accept(self):
        pass
    
    async def send_json(self, data):
        self.messages_sent.append(data)
    
    async def close(self, code=None, reason=None):
        self.closed = True


class TestMessageType:
    """Test suite for MessageType enum."""
    
    def test_message_types(self):
        """Test MessageType enum values."""
        # Client to server messages
        assert MessageType.AUDIO_CHUNK == "audio_chunk"
        assert MessageType.TEXT_INPUT == "text_input"
        assert MessageType.START_CONVERSATION == "start_conversation"
        assert MessageType.END_CONVERSATION == "end_conversation"
        assert MessageType.CHANGE_PERSONALITY == "change_personality"
        assert MessageType.CHANGE_LANGUAGE == "change_language"
        assert MessageType.GET_STATUS == "get_status"
        assert MessageType.GET_PERSONALITIES == "get_personalities"
        assert MessageType.ABORT_RESPONSE == "abort_response"
        
        # Server to client messages
        assert MessageType.TRANSCRIPTION == "transcription"
        assert MessageType.RESPONSE_START == "response_start"
        assert MessageType.RESPONSE_CHUNK == "response_chunk"
        assert MessageType.AUDIO_OUTPUT == "audio_output"
        assert MessageType.SESSION_COMPLETE == "session_complete"
        assert MessageType.STATUS_UPDATE == "status_update"
        assert MessageType.ERROR == "error"
        assert MessageType.GREETING == "greeting"
        assert MessageType.PERSONALITY_CHANGED == "personality_changed"
        assert MessageType.CONVERSATION_ENDED == "conversation_ended"


class TestWebSocketHandler:
    """Test suite for WebSocketHandler class."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.debug = True
        return settings
    
    @pytest.fixture
    def mock_anchor_agent(self):
        """Create mock anchor agent."""
        agent = Mock()
        agent.get_agent_status.return_value = {"is_active": True}
        agent.get_available_personalities.return_value = {"professional": {}}
        agent.start_conversation = AsyncMock(return_value="Hello!")
        agent.end_conversation = AsyncMock()
        agent.switch_personality = AsyncMock(return_value=True)
        agent.switch_language = AsyncMock()
        agent.process_audio_input = Mock()
        agent.process_text_input = Mock()
        agent.abort_current_response = Mock()
        agent.get_conversation_history.return_value = []
        agent.current_context = None
        return agent
    
    @pytest.fixture
    def websocket_handler(self, mock_settings, mock_anchor_agent):
        """Create WebSocket handler instance."""
        return WebSocketHandler(mock_settings, mock_anchor_agent)
    
    def test_initialization(self, mock_settings, mock_anchor_agent):
        """Test WebSocket handler initialization."""
        handler = WebSocketHandler(mock_settings, mock_anchor_agent)
        
        assert handler.settings == mock_settings
        assert handler.anchor_agent == mock_anchor_agent
        assert isinstance(handler.active_connections, dict)
        assert isinstance(handler.connection_metadata, dict)
        assert isinstance(handler.message_queues, dict)
        assert len(handler.active_connections) == 0
    
    def test_setup_anchor_callbacks(self, websocket_handler):
        """Test anchor agent callback setup."""
        # Check that callbacks were set on the anchor agent
        agent = websocket_handler.anchor_agent
        assert agent.on_transcription == websocket_handler._on_transcription
        assert agent.on_response_start == websocket_handler._on_response_start
        assert agent.on_response_chunk == websocket_handler._on_response_chunk
        assert agent.on_audio_chunk == websocket_handler._on_audio_chunk
        assert agent.on_session_complete == websocket_handler._on_session_complete
        assert agent.on_error == websocket_handler._on_error
        assert agent.on_personality_change == websocket_handler._on_personality_change
        assert agent.on_conversation_start == websocket_handler._on_conversation_start
        assert agent.on_conversation_end == websocket_handler._on_conversation_end
    
    @pytest.mark.asyncio
    async def test_connect(self, websocket_handler):
        """Test WebSocket connection."""
        websocket = MockWebSocket()
        client_id = "test-client"
        
        # Mock the send method
        websocket_handler._send_to_client = AsyncMock()
        
        await websocket_handler.connect(websocket, client_id)
        
        # Check connection was stored
        assert client_id in websocket_handler.active_connections
        assert websocket_handler.active_connections[client_id] == websocket
        assert client_id in websocket_handler.connection_metadata
        assert client_id in websocket_handler.message_queues
        
        # Check metadata
        metadata = websocket_handler.connection_metadata[client_id]
        assert "connected_at" in metadata
        assert metadata["conversation_active"] is False
        assert metadata["personality"] == "professional"
        assert metadata["language"] == "en"
        assert metadata["message_count"] == 0
        
        # Check status message was sent
        websocket_handler._send_to_client.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, websocket_handler):
        """Test WebSocket disconnection."""
        websocket = MockWebSocket()
        client_id = "test-client"
        
        # Connect first
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        # Set up active conversation
        websocket_handler.connection_metadata[client_id]["conversation_active"] = True
        
        # Disconnect
        await websocket_handler.disconnect(client_id)
        
        # Check cleanup
        assert client_id not in websocket_handler.active_connections
        assert client_id not in websocket_handler.connection_metadata
        assert client_id not in websocket_handler.message_queues
        
        # Check anchor agent end_conversation was called
        websocket_handler.anchor_agent.end_conversation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_text_input(self, websocket_handler):
        """Test text input handling."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        data = {"text": "Hello world"}
        await websocket_handler._handle_text_input(client_id, data)
        
        # Check anchor agent was called
        websocket_handler.anchor_agent.process_text_input.assert_called_once_with("Hello world")
    
    @pytest.mark.asyncio
    async def test_handle_text_input_empty(self, websocket_handler):
        """Test text input handling with empty text."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        websocket_handler._send_error = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        data = {"text": "   "}  # Only whitespace
        await websocket_handler._handle_text_input(client_id, data)
        
        # Check error was sent
        websocket_handler._send_error.assert_called_once_with(client_id, "Empty text input")
        
        # Check anchor agent was not called
        websocket_handler.anchor_agent.process_text_input.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_handle_audio_chunk(self, websocket_handler):
        """Test audio chunk handling."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        # Base64 encoded test data
        import base64
        test_audio = b"test audio data"
        audio_base64 = base64.b64encode(test_audio).decode('utf-8')
        
        data = {"audio_data": audio_base64}
        await websocket_handler._handle_audio_chunk(client_id, data)
        
        # Check anchor agent was called with decoded audio
        websocket_handler.anchor_agent.process_audio_input.assert_called_once_with(test_audio)
    
    @pytest.mark.asyncio
    async def test_handle_start_conversation(self, websocket_handler):
        """Test start conversation handling."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        data = {
            "mode": "interview",
            "personality": "friendly",
            "language": "es"
        }
        await websocket_handler._handle_start_conversation(client_id, data)
        
        # Check anchor agent was called
        websocket_handler.anchor_agent.start_conversation.assert_called_once()
        
        # Check metadata was updated
        metadata = websocket_handler.connection_metadata[client_id]
        assert metadata["conversation_active"] is True
        assert metadata["personality"] == "friendly"
        assert metadata["language"] == "es"
        
        # Check greeting was sent
        websocket_handler._send_to_client.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_end_conversation(self, websocket_handler):
        """Test end conversation handling."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        # Set conversation as active
        websocket_handler.connection_metadata[client_id]["conversation_active"] = True
        
        await websocket_handler._handle_end_conversation(client_id, {})
        
        # Check anchor agent was called
        websocket_handler.anchor_agent.end_conversation.assert_called_once()
        
        # Check metadata was updated
        metadata = websocket_handler.connection_metadata[client_id]
        assert metadata["conversation_active"] is False
        
        # Check confirmation was sent
        websocket_handler._send_to_client.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_change_personality(self, websocket_handler):
        """Test personality change handling."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        data = {"personality": "energetic"}
        await websocket_handler._handle_change_personality(client_id, data)
        
        # Check anchor agent was called
        websocket_handler.anchor_agent.switch_personality.assert_called_once_with("energetic")
        
        # Check metadata was updated
        metadata = websocket_handler.connection_metadata[client_id]
        assert metadata["personality"] == "energetic"
    
    @pytest.mark.asyncio
    async def test_handle_change_personality_failure(self, websocket_handler):
        """Test personality change handling with failure."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        websocket_handler._send_error = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        # Configure anchor agent to return failure
        websocket_handler.anchor_agent.switch_personality.return_value = False
        
        data = {"personality": "unknown"}
        await websocket_handler._handle_change_personality(client_id, data)
        
        # Check error was sent
        websocket_handler._send_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_change_language(self, websocket_handler):
        """Test language change handling."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        data = {"language": "fr"}
        await websocket_handler._handle_change_language(client_id, data)
        
        # Check anchor agent was called
        websocket_handler.anchor_agent.switch_language.assert_called_once_with("fr")
        
        # Check metadata was updated
        metadata = websocket_handler.connection_metadata[client_id]
        assert metadata["language"] == "fr"
        
        # Check confirmation was sent
        websocket_handler._send_to_client.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_get_status(self, websocket_handler):
        """Test get status handling."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        await websocket_handler._handle_get_status(client_id, {})
        
        # Check status message was sent
        websocket_handler._send_to_client.assert_called()
        call_args = websocket_handler._send_to_client.call_args[0]
        message = call_args[1]
        
        assert message["type"] == MessageType.STATUS_UPDATE
        assert "agent_status" in message["data"]
        assert "connection_status" in message["data"]
        assert "active_connections" in message["data"]
    
    @pytest.mark.asyncio
    async def test_handle_get_personalities(self, websocket_handler):
        """Test get personalities handling."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        await websocket_handler._handle_get_personalities(client_id, {})
        
        # Check personalities message was sent
        websocket_handler._send_to_client.assert_called()
        call_args = websocket_handler._send_to_client.call_args[0]
        message = call_args[1]
        
        assert message["type"] == MessageType.STATUS_UPDATE
        assert "personalities" in message["data"]
    
    @pytest.mark.asyncio
    async def test_handle_abort_response(self, websocket_handler):
        """Test abort response handling."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        await websocket_handler._handle_abort_response(client_id, {})
        
        # Check anchor agent was called
        websocket_handler.anchor_agent.abort_current_response.assert_called_once()
        
        # Check confirmation was sent
        websocket_handler._send_to_client.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_message_routing(self, websocket_handler):
        """Test message routing to appropriate handlers."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        # Mock individual handlers
        websocket_handler._handle_text_input = AsyncMock()
        websocket_handler._handle_audio_chunk = AsyncMock()
        websocket_handler._handle_start_conversation = AsyncMock()
        
        # Test text input routing
        message = {"type": MessageType.TEXT_INPUT, "data": {"text": "test"}}
        await websocket_handler.handle_message(client_id, message)
        websocket_handler._handle_text_input.assert_called_once_with(client_id, {"text": "test"})
        
        # Test audio chunk routing
        message = {"type": MessageType.AUDIO_CHUNK, "data": {"audio_data": "test"}}
        await websocket_handler.handle_message(client_id, message)
        websocket_handler._handle_audio_chunk.assert_called_once_with(client_id, {"audio_data": "test"})
        
        # Test start conversation routing
        message = {"type": MessageType.START_CONVERSATION, "data": {"mode": "test"}}
        await websocket_handler.handle_message(client_id, message)
        websocket_handler._handle_start_conversation.assert_called_once_with(client_id, {"mode": "test"})
    
    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self, websocket_handler):
        """Test handling of unknown message types."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        websocket_handler._send_to_client = AsyncMock()
        websocket_handler._send_error = AsyncMock()
        await websocket_handler.connect(websocket, client_id)
        
        message = {"type": "unknown_type", "data": {}}
        await websocket_handler.handle_message(client_id, message)
        
        # Check error was sent
        websocket_handler._send_error.assert_called_once_with(client_id, "Unknown message type: unknown_type")
    
    def test_anchor_agent_event_handlers(self, websocket_handler):
        """Test anchor agent event handlers."""
        websocket_handler._broadcast_to_all = AsyncMock()
        
        # Test transcription handler
        result = Mock()
        result.text = "Test transcription"
        result.language = "en"
        result.confidence = 0.95
        result.is_final = True
        result.timestamp = time.time()
        
        websocket_handler._on_transcription(result)
        # Event handler creates asyncio task, so we can't easily test the broadcast
        
        # Test response start handler
        websocket_handler._on_response_start("Test input")
        
        # Test response chunk handler
        websocket_handler._on_response_chunk("Test chunk")
        
        # Test session complete handler
        session = Mock()
        session.session_id = "test-session"
        session.input_text = "Hello"
        session.llm_response = "Hi there"
        
        websocket_handler._on_session_complete(session)
        
        # Test error handler
        error = Exception("Test error")
        websocket_handler._on_error(error)
    
    @pytest.mark.asyncio
    async def test_send_to_client(self, websocket_handler):
        """Test sending message to specific client."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        # Connect client
        await websocket_handler.connect(websocket, client_id)
        
        # Send message
        message = {"type": "test", "data": "test data"}
        await websocket_handler._send_to_client(client_id, message)
        
        # Check message was sent
        assert len(websocket.messages_sent) == 2  # 1 initial status + 1 test message
        assert websocket.messages_sent[-1] == message
    
    @pytest.mark.asyncio
    async def test_send_to_unknown_client(self, websocket_handler):
        """Test sending message to unknown client."""
        # Should not raise an error
        await websocket_handler._send_to_client("unknown-client", {"type": "test"})
    
    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, websocket_handler):
        """Test broadcasting message to all clients."""
        # Connect multiple clients
        clients = []
        for i in range(3):
            client_id = f"client-{i}"
            websocket = MockWebSocket()
            await websocket_handler.connect(websocket, client_id)
            clients.append((client_id, websocket))
        
        # Broadcast message
        message = {"type": "broadcast", "data": "test"}
        await websocket_handler._broadcast_to_all(message)
        
        # Check all clients received the message
        for client_id, websocket in clients:
            assert len(websocket.messages_sent) >= 2  # At least status + broadcast
            assert message in websocket.messages_sent
    
    @pytest.mark.asyncio
    async def test_send_error(self, websocket_handler):
        """Test sending error message."""
        client_id = "test-client"
        websocket = MockWebSocket()
        
        await websocket_handler.connect(websocket, client_id)
        
        await websocket_handler._send_error(client_id, "Test error message")
        
        # Check error message was sent
        error_message = websocket.messages_sent[-1]
        assert error_message["type"] == MessageType.ERROR
        assert error_message["data"]["error"] == "Test error message"
    
    def test_get_connection_stats(self, websocket_handler):
        """Test getting connection statistics."""
        # Initially no connections
        stats = websocket_handler.get_connection_stats()
        assert stats["active_connections"] == 0
        assert len(stats["connections"]) == 0
        
        # Add some connections manually
        client_id = "test-client"
        websocket_handler.connection_metadata[client_id] = {
            "connected_at": time.time() - 60,  # 1 minute ago
            "conversation_active": True,
            "personality": "friendly",
            "language": "en",
            "message_count": 5
        }
        
        stats = websocket_handler.get_connection_stats()
        assert stats["active_connections"] == 0  # No actual websocket, just metadata
        assert len(stats["connections"]) == 1
        assert stats["connections"][client_id]["personality"] == "friendly"
        assert stats["connections"][client_id]["message_count"] == 5
    
    @pytest.mark.asyncio
    async def test_health_check(self, websocket_handler):
        """Test health check."""
        # Mock anchor agent health check
        websocket_handler.anchor_agent.health_check = AsyncMock(return_value={"overall_health": "healthy"})
        
        health = await websocket_handler.health_check()
        
        assert isinstance(health, dict)
        assert "websocket_handler" in health
        assert "active_connections" in health
        assert "anchor_agent_health" in health
        assert "overall_health" in health
        
        assert health["websocket_handler"] == "healthy"
        assert health["anchor_agent_health"] == "healthy"


class TestWebSocketHandlerIntegration:
    """Integration tests for WebSocketHandler."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test full conversation flow through WebSocket."""
        # Create mocks
        settings = Mock()
        settings.debug = True
        
        anchor_agent = Mock()
        anchor_agent.get_agent_status.return_value = {"is_active": True}
        anchor_agent.get_available_personalities.return_value = {"professional": {}}
        anchor_agent.start_conversation = AsyncMock(return_value="Hello!")
        anchor_agent.end_conversation = AsyncMock()
        anchor_agent.switch_personality = AsyncMock(return_value=True)
        anchor_agent.process_text_input = Mock()
        anchor_agent.current_context = Mock()
        anchor_agent.current_context.session_id = "test-session"
        
        # Create handler
        handler = WebSocketHandler(settings, anchor_agent)
        
        # Connect client
        client_id = "test-client"
        websocket = MockWebSocket()
        handler._send_to_client = AsyncMock()
        await handler.connect(websocket, client_id)
        
        # Start conversation
        await handler.handle_message(client_id, {
            "type": MessageType.START_CONVERSATION,
            "data": {"mode": "casual_chat", "personality": "friendly"}
        })
        
        # Send text input
        await handler.handle_message(client_id, {
            "type": MessageType.TEXT_INPUT,
            "data": {"text": "Hello world"}
        })
        
        # Change personality
        await handler.handle_message(client_id, {
            "type": MessageType.CHANGE_PERSONALITY,
            "data": {"personality": "energetic"}
        })
        
        # End conversation
        await handler.handle_message(client_id, {
            "type": MessageType.END_CONVERSATION,
            "data": {}
        })
        
        # Disconnect
        await handler.disconnect(client_id)
        
        # Verify calls were made
        anchor_agent.start_conversation.assert_called_once()
        anchor_agent.process_text_input.assert_called_once_with("Hello world")
        anchor_agent.switch_personality.assert_called_once_with("energetic")
        anchor_agent.end_conversation.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])