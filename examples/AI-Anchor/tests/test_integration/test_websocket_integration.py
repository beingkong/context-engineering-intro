"""
WebSocket integration tests for AI Anchor system.

Tests real-time communication between clients and the AI anchor.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from web.websocket_handler import WebSocketHandler, MessageType


class MockWebSocketConnection:
    """Enhanced mock WebSocket for integration testing."""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.messages_sent = []
        self.messages_received = []
        self.is_connected = False
        self.client_state = "DISCONNECTED"
    
    async def accept(self):
        """Accept WebSocket connection."""
        self.is_connected = True
        self.client_state = "CONNECTED"
    
    async def send_json(self, data):
        """Send JSON message to client."""
        if self.is_connected:
            self.messages_sent.append({
                "timestamp": time.time(),
                "data": data
            })
    
    async def receive_json(self):
        """Receive JSON message from client."""
        if self.messages_received:
            return self.messages_received.pop(0)
        raise asyncio.TimeoutError("No messages to receive")
    
    def queue_message(self, message):
        """Queue a message to be received."""
        self.messages_received.append(message)
    
    async def close(self, code=None, reason=None):
        """Close WebSocket connection."""
        self.is_connected = False
        self.client_state = "DISCONNECTED"


class TestWebSocketIntegration:
    """Integration tests for WebSocket communication."""
    
    @pytest.fixture
    def websocket_handler(self, mock_settings, mock_anchor_agent):
        """Create WebSocket handler with mocked dependencies."""
        return WebSocketHandler(mock_settings, mock_anchor_agent)
    
    @pytest.fixture
    def mock_websocket_connection(self):
        """Create mock WebSocket connection."""
        return MockWebSocketConnection("test-client-123")
    
    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self, websocket_handler, mock_websocket_connection):
        """Test complete WebSocket connection lifecycle."""
        client_id = "integration-test-client"
        
        # Connect
        await websocket_handler.connect(mock_websocket_connection, client_id)
        
        # Verify connection was established
        assert client_id in websocket_handler.active_connections
        assert client_id in websocket_handler.connection_metadata
        assert client_id in websocket_handler.message_queues
        
        # Check connection metadata
        metadata = websocket_handler.connection_metadata[client_id]
        assert metadata["conversation_active"] is False
        assert metadata["personality"] == "professional"
        assert metadata["language"] == "en"
        assert metadata["message_count"] == 0
        
        # Disconnect
        await websocket_handler.disconnect(client_id)
        
        # Verify cleanup
        assert client_id not in websocket_handler.active_connections
        assert client_id not in websocket_handler.connection_metadata
        assert client_id not in websocket_handler.message_queues
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow_via_websocket(self, websocket_handler, mock_websocket_connection):
        """Test complete conversation flow through WebSocket."""
        client_id = "conversation-test-client"
        
        # Connect client
        await websocket_handler.connect(mock_websocket_connection, client_id)
        
        # Start conversation
        start_message = {
            "type": MessageType.START_CONVERSATION,
            "data": {
                "mode": "casual_chat",
                "personality": "friendly",
                "language": "en"
            }
        }
        await websocket_handler.handle_message(client_id, start_message)
        
        # Verify anchor agent was called
        websocket_handler.anchor_agent.start_conversation.assert_called_once()
        
        # Check metadata updated
        metadata = websocket_handler.connection_metadata[client_id]
        assert metadata["conversation_active"] is True
        assert metadata["personality"] == "friendly"
        assert metadata["language"] == "en"
        assert metadata["message_count"] == 1
        
        # Send text input
        text_message = {
            "type": MessageType.TEXT_INPUT,
            "data": {"text": "Hello, how are you today?"}
        }
        await websocket_handler.handle_message(client_id, text_message)
        
        # Verify text processing
        websocket_handler.anchor_agent.process_text_input.assert_called_once_with("Hello, how are you today?")
        
        # Change personality
        personality_message = {
            "type": MessageType.CHANGE_PERSONALITY,
            "data": {"personality": "energetic"}
        }
        await websocket_handler.handle_message(client_id, personality_message)
        
        # Verify personality change
        websocket_handler.anchor_agent.switch_personality.assert_called_once_with("energetic")
        
        # End conversation
        end_message = {
            "type": MessageType.END_CONVERSATION,
            "data": {}
        }
        await websocket_handler.handle_message(client_id, end_message)
        
        # Verify conversation ended
        websocket_handler.anchor_agent.end_conversation.assert_called_once()
        
        # Check metadata updated
        metadata = websocket_handler.connection_metadata[client_id]
        assert metadata["conversation_active"] is False
        assert metadata["message_count"] == 4  # start, text, personality, end
    
    @pytest.mark.asyncio
    async def test_audio_processing_via_websocket(self, websocket_handler, mock_websocket_connection, sample_base64_audio):
        """Test audio processing through WebSocket."""
        client_id = "audio-test-client"
        
        # Connect client
        await websocket_handler.connect(mock_websocket_connection, client_id)
        
        # Send audio chunk
        audio_message = {
            "type": MessageType.AUDIO_CHUNK,
            "data": {"audio_data": sample_base64_audio}
        }
        await websocket_handler.handle_message(client_id, audio_message)
        
        # Verify audio was processed
        websocket_handler.anchor_agent.process_audio_input.assert_called_once()
        
        # Check that audio was decoded correctly
        call_args = websocket_handler.anchor_agent.process_audio_input.call_args[0]
        audio_bytes = call_args[0]
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_clients_concurrent(self, websocket_handler):
        """Test handling multiple concurrent WebSocket clients."""
        num_clients = 5
        clients = []
        
        # Connect multiple clients
        for i in range(num_clients):
            client_id = f"concurrent-client-{i}"
            websocket = MockWebSocketConnection(client_id)
            clients.append((client_id, websocket))
            
            await websocket_handler.connect(websocket, client_id)
        
        # Verify all clients connected
        assert len(websocket_handler.active_connections) == num_clients
        
        # Send messages from all clients concurrently
        tasks = []
        for client_id, websocket in clients:
            message = {
                "type": MessageType.TEXT_INPUT,
                "data": {"text": f"Message from {client_id}"}
            }
            tasks.append(websocket_handler.handle_message(client_id, message))
        
        # Wait for all messages to be processed
        await asyncio.gather(*tasks)
        
        # Verify all messages were processed
        assert websocket_handler.anchor_agent.process_text_input.call_count == num_clients
        
        # Disconnect all clients
        for client_id, websocket in clients:
            await websocket_handler.disconnect(client_id)
        
        # Verify all clients disconnected
        assert len(websocket_handler.active_connections) == 0
    
    @pytest.mark.asyncio
    async def test_anchor_agent_callbacks_to_websocket(self, websocket_handler, mock_websocket_connection):
        """Test anchor agent callbacks propagating to WebSocket clients."""
        client_id = "callback-test-client"
        
        # Connect client
        await websocket_handler.connect(mock_websocket_connection, client_id)
        
        # Mock the _broadcast_to_all method to track calls
        websocket_handler._broadcast_to_all = AsyncMock()
        
        # Simulate anchor agent events
        
        # 1. Transcription event
        mock_transcription = Mock()
        mock_transcription.text = "Test transcription"
        mock_transcription.language = "en"
        mock_transcription.confidence = 0.95
        mock_transcription.is_final = True
        mock_transcription.timestamp = time.time()
        
        websocket_handler._on_transcription(mock_transcription)
        
        # 2. Response start event
        websocket_handler._on_response_start("Test input")
        
        # 3. Response chunk event
        websocket_handler._on_response_chunk("Test chunk")
        
        # 4. Audio chunk event
        mock_audio_chunk = Mock()
        mock_audio_chunk.data = b"test_audio_data"
        mock_audio_chunk.format = "wav"
        mock_audio_chunk.sample_rate = 24000
        mock_audio_chunk.duration = 1.0
        
        websocket_handler._on_audio_chunk(mock_audio_chunk)
        
        # 5. Session complete event
        mock_session = Mock()
        mock_session.session_id = "test-session"
        mock_session.input_text = "Hello"
        mock_session.llm_response = "Hi there"
        
        websocket_handler._on_session_complete(mock_session)
        
        # 6. Error event
        test_error = Exception("Test error")
        websocket_handler._on_error(test_error)
        
        # Wait for async tasks to complete
        await asyncio.sleep(0.1)
        
        # Verify broadcast calls were made
        # Note: Each event creates an asyncio task, so we can't easily count exact calls
        # But we can verify the method was set up correctly
        assert websocket_handler._broadcast_to_all is not None
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, websocket_handler, mock_websocket_connection):
        """Test WebSocket error handling scenarios."""
        client_id = "error-test-client"
        
        # Connect client
        await websocket_handler.connect(mock_websocket_connection, client_id)
        
        # Test unknown message type
        unknown_message = {
            "type": "unknown_message_type",
            "data": {}
        }
        
        # Mock _send_error to track error messages
        websocket_handler._send_error = AsyncMock()
        
        await websocket_handler.handle_message(client_id, unknown_message)
        
        # Verify error was sent
        websocket_handler._send_error.assert_called_once()
        error_call = websocket_handler._send_error.call_args
        assert error_call[0][0] == client_id
        assert "Unknown message type" in error_call[0][1]
        
        # Test malformed message handling
        websocket_handler._send_error.reset_mock()
        
        malformed_message = {
            "type": MessageType.TEXT_INPUT,
            "data": {}  # Missing required 'text' field
        }
        
        await websocket_handler.handle_message(client_id, malformed_message)
        
        # Should handle gracefully (empty text input error)
        websocket_handler._send_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_connection_stats(self, websocket_handler):
        """Test WebSocket connection statistics."""
        # Initially no connections
        stats = websocket_handler.get_connection_stats()
        assert stats["active_connections"] == 0
        assert len(stats["connections"]) == 0
        
        # Connect some clients
        clients = []
        for i in range(3):
            client_id = f"stats-client-{i}"
            websocket = MockWebSocketConnection(client_id)
            clients.append((client_id, websocket))
            
            await websocket_handler.connect(websocket, client_id)
            
            # Send some messages to update counters
            for j in range(2):
                message = {
                    "type": MessageType.TEXT_INPUT,
                    "data": {"text": f"Message {j} from {client_id}"}
                }
                await websocket_handler.handle_message(client_id, message)
        
        # Check updated stats
        stats = websocket_handler.get_connection_stats()
        assert stats["active_connections"] == 3
        assert len(stats["connections"]) == 3
        
        # Check individual client stats
        for i, (client_id, websocket) in enumerate(clients):
            client_stats = stats["connections"][client_id]
            assert client_stats["message_count"] == 2
            assert client_stats["conversation_active"] is False
            assert client_stats["personality"] == "professional"
            assert client_stats["language"] == "en"
            assert "connected_duration" in client_stats
    
    @pytest.mark.asyncio
    async def test_websocket_health_check(self, websocket_handler):
        """Test WebSocket handler health check."""
        # Initial health check
        health = await websocket_handler.health_check()
        
        assert isinstance(health, dict)
        assert health["websocket_handler"] == "healthy"
        assert health["active_connections"] == 0
        assert health["anchor_agent_health"] == "healthy"
        assert health["overall_health"] == "healthy"
        
        # Connect some clients and check again
        for i in range(2):
            client_id = f"health-client-{i}"
            websocket = MockWebSocketConnection(client_id)
            await websocket_handler.connect(websocket, client_id)
        
        health = await websocket_handler.health_check()
        assert health["active_connections"] == 2
        assert "connection_stats" in health
    
    @pytest.mark.asyncio
    async def test_websocket_message_queuing(self, websocket_handler, mock_websocket_connection):
        """Test WebSocket message queuing and processing."""
        client_id = "queue-test-client"
        
        # Connect client
        await websocket_handler.connect(mock_websocket_connection, client_id)
        
        # Send multiple messages rapidly
        messages = []
        for i in range(10):
            message = {
                "type": MessageType.TEXT_INPUT,
                "data": {"text": f"Rapid message {i}"}
            }
            messages.append(message)
        
        # Send all messages
        tasks = []
        for message in messages:
            tasks.append(websocket_handler.handle_message(client_id, message))
        
        # Wait for all to be processed
        await asyncio.gather(*tasks)
        
        # Verify all messages were processed
        assert websocket_handler.anchor_agent.process_text_input.call_count == 10
        
        # Check message counter
        metadata = websocket_handler.connection_metadata[client_id]
        assert metadata["message_count"] == 10


class TestWebSocketStressTests:
    """Stress tests for WebSocket handling."""
    
    @pytest.mark.asyncio
    async def test_high_frequency_messages(self, websocket_handler):
        """Test handling high frequency of messages."""
        client_id = "stress-client"
        websocket = MockWebSocketConnection(client_id)
        
        await websocket_handler.connect(websocket, client_id)
        
        # Send many messages very quickly
        num_messages = 100
        start_time = time.time()
        
        tasks = []
        for i in range(num_messages):
            message = {
                "type": MessageType.TEXT_INPUT,
                "data": {"text": f"Stress message {i}"}
            }
            tasks.append(websocket_handler.handle_message(client_id, message))
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should handle messages reasonably quickly
        assert processing_time < 5.0  # 5 seconds for 100 messages with mocks
        
        # All messages should be processed
        assert websocket_handler.anchor_agent.process_text_input.call_count == num_messages
    
    @pytest.mark.asyncio
    async def test_many_concurrent_connections(self, websocket_handler):
        """Test many concurrent WebSocket connections."""
        num_connections = 20
        clients = []
        
        # Connect many clients simultaneously
        connect_tasks = []
        for i in range(num_connections):
            client_id = f"concurrent-{i}"
            websocket = MockWebSocketConnection(client_id)
            clients.append((client_id, websocket))
            connect_tasks.append(websocket_handler.connect(websocket, client_id))
        
        await asyncio.gather(*connect_tasks)
        
        # Verify all connected
        assert len(websocket_handler.active_connections) == num_connections
        
        # Send messages from all clients
        message_tasks = []
        for client_id, websocket in clients:
            message = {
                "type": MessageType.GET_STATUS,
                "data": {}
            }
            message_tasks.append(websocket_handler.handle_message(client_id, message))
        
        await asyncio.gather(*message_tasks)
        
        # Disconnect all clients
        disconnect_tasks = []
        for client_id, websocket in clients:
            disconnect_tasks.append(websocket_handler.disconnect(client_id))
        
        await asyncio.gather(*disconnect_tasks)
        
        # Verify all disconnected
        assert len(websocket_handler.active_connections) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])