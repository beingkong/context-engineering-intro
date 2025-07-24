"""
Unit tests for API Routes.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from web.api_routes import AIAnchorAPI, create_app


class TestAIAnchorAPI:
    """Test suite for AIAnchorAPI class."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.debug = True
        settings.web = Mock()
        settings.web.cors_origins = ["*"]
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
        agent.get_conversation_history.return_value = []
        agent.current_context = None
        agent.health_check = AsyncMock(return_value={"overall_health": "healthy"})
        return agent
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager."""
        manager = Mock()
        manager.get_memory_statistics.return_value = {"memory_usage": "normal"}
        manager.health_check = AsyncMock(return_value={"overall_health": "healthy"})
        return manager
    
    @pytest.fixture
    def mock_audio_pipeline(self):
        """Create mock audio pipeline."""
        pipeline = Mock()
        pipeline.get_pipeline_state.return_value = {"state": "idle"}
        pipeline.health_check = AsyncMock(return_value={"overall_health": "healthy"})
        return pipeline
    
    def test_initialization_without_fastapi(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test initialization when FastAPI is not available."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', False):
            api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
            
            assert api.app is None
            assert api.websocket_handler is None
    
    def test_initialization_with_fastapi(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test initialization when FastAPI is available."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI') as mock_fastapi:
                with patch('web.api_routes.WebSocketHandler') as mock_ws_handler_class:
                    mock_app = Mock()
                    mock_fastapi.return_value = mock_app
                    mock_ws_handler = Mock()
                    mock_ws_handler_class.return_value = mock_ws_handler
                    
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    assert api.app == mock_app
                    assert api.websocket_handler == mock_ws_handler
                    
                    # Check FastAPI was configured
                    mock_fastapi.assert_called_once()
                    mock_app.add_middleware.assert_called_once()
    
    def test_get_default_html(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test default HTML generation."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    html = api._get_default_html()
                    
                    assert isinstance(html, str)
                    assert "<!DOCTYPE html>" in html
                    assert "AI Anchor" in html
                    assert "WebSocket" in html
                    assert "personality" in html.lower()
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test health check endpoint."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler') as mock_ws_handler_class:
                    mock_ws_handler = Mock()
                    mock_ws_handler.health_check = AsyncMock(return_value={"overall_health": "healthy"})
                    mock_ws_handler_class.return_value = mock_ws_handler
                    
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    # Mock JSONResponse
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._health_check()
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check response data
                        content = call_args[1]["content"]
                        assert "timestamp" in content
                        assert "api_status" in content
                        assert "anchor_agent" in content
                        assert "memory_manager" in content
                        assert "audio_pipeline" in content
                        assert "websocket_handler" in content
                        assert "overall_health" in content
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_health_check_error(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test health check endpoint with error."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    # Make anchor agent health check raise an error
                    mock_anchor_agent.health_check.side_effect = Exception("Test error")
                    
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._health_check()
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check error response
                        content = call_args[1]["content"]
                        assert "error" in content
                        assert "overall_health" in content
                        assert content["overall_health"] == "error"
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 500
    
    @pytest.mark.asyncio
    async def test_get_status(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test get status endpoint."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler') as mock_ws_handler_class:
                    mock_ws_handler = Mock()
                    mock_ws_handler.get_connection_stats.return_value = {"active_connections": 1}
                    mock_ws_handler_class.return_value = mock_ws_handler
                    
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._get_status()
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check response data
                        content = call_args[1]["content"]
                        assert "timestamp" in content
                        assert "anchor_agent" in content
                        assert "memory_manager" in content
                        assert "audio_pipeline" in content
                        assert "websocket_connections" in content
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_get_personalities(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test get personalities endpoint."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._get_personalities()
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check response data
                        content = call_args[1]["content"]
                        assert "personalities" in content
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_start_conversation(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test start conversation endpoint."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    # Mock current context
                    mock_context = Mock()
                    mock_context.session_id = "test-session"
                    mock_anchor_agent.current_context = mock_context
                    
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    request = {
                        "mode": "interview",
                        "personality": "friendly",
                        "language": "es"
                    }
                    
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._start_conversation(request)
                        
                        # Check anchor agent was called
                        mock_anchor_agent.start_conversation.assert_called_once()
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check response data
                        content = call_args[1]["content"]
                        assert "greeting" in content
                        assert "session_id" in content
                        assert "mode" in content
                        assert "personality" in content
                        assert "language" in content
                        
                        assert content["greeting"] == "Hello!"
                        assert content["session_id"] == "test-session"
                        assert content["personality"] == "friendly"
                        assert content["language"] == "es"
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_start_conversation_no_agent(self, mock_settings, mock_memory_manager, mock_audio_pipeline):
        """Test start conversation endpoint without agent."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    api = AIAnchorAPI(mock_settings, None, mock_memory_manager, mock_audio_pipeline)
                    
                    request = {"mode": "casual_chat"}
                    
                    with patch('web.api_routes.HTTPException') as mock_http_exception:
                        with pytest.raises(Exception):  # HTTPException should be raised
                            await api._start_conversation(request)
                        
                        mock_http_exception.assert_called_once_with(
                            status_code=503, 
                            detail="Anchor agent not available"
                        )
    
    @pytest.mark.asyncio
    async def test_end_conversation(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test end conversation endpoint."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._end_conversation()
                        
                        # Check anchor agent was called
                        mock_anchor_agent.end_conversation.assert_called_once()
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check response data
                        content = call_args[1]["content"]
                        assert "ended" in content
                        assert "timestamp" in content
                        assert content["ended"] is True
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_change_personality(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test change personality endpoint."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    request = {"personality": "energetic"}
                    
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._change_personality(request)
                        
                        # Check anchor agent was called
                        mock_anchor_agent.switch_personality.assert_called_once_with("energetic")
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check response data
                        content = call_args[1]["content"]
                        assert "success" in content
                        assert "personality" in content
                        assert "timestamp" in content
                        assert content["success"] is True
                        assert content["personality"] == "energetic"
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_change_personality_empty(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test change personality endpoint with empty personality."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    request = {"personality": "   "}  # Only whitespace
                    
                    with patch('web.api_routes.HTTPException') as mock_http_exception:
                        with pytest.raises(Exception):
                            await api._change_personality(request)
                        
                        mock_http_exception.assert_called_once_with(
                            status_code=400,
                            detail="Personality name required"
                        )
    
    @pytest.mark.asyncio
    async def test_change_language(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test change language endpoint."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    request = {"language": "fr"}
                    
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._change_language(request)
                        
                        # Check anchor agent was called
                        mock_anchor_agent.switch_language.assert_called_once_with("fr")
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check response data
                        content = call_args[1]["content"]
                        assert "language" in content
                        assert "timestamp" in content
                        assert content["language"] == "fr"
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test get conversation history endpoint."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    # Set up conversation history
                    mock_history = [
                        {"user_input": "Hello", "anchor_response": "Hi there!"},
                        {"user_input": "How are you?", "anchor_response": "I'm doing well!"}
                    ]
                    mock_anchor_agent.get_conversation_history.return_value = mock_history
                    
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._get_conversation_history()
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check response data
                        content = call_args[1]["content"]
                        assert "history" in content
                        assert content["history"] == mock_history
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_get_connections(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test get connections endpoint."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler') as mock_ws_handler_class:
                    mock_ws_handler = Mock()
                    mock_stats = {"active_connections": 2, "connections": {}}
                    mock_ws_handler.get_connection_stats.return_value = mock_stats
                    mock_ws_handler_class.return_value = mock_ws_handler
                    
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._get_connections()
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check response data
                        content = call_args[1]["content"]
                        assert content == mock_stats
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test get memory stats endpoint."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI'):
                with patch('web.api_routes.WebSocketHandler'):
                    mock_stats = {"gpu_memory": "10GB", "cpu_memory": "5GB"}
                    mock_memory_manager.get_memory_statistics.return_value = mock_stats
                    
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        response = await api._get_memory_stats()
                        
                        mock_json_response.assert_called_once()
                        call_args = mock_json_response.call_args
                        
                        # Check response data
                        content = call_args[1]["content"]
                        assert content == mock_stats
                        
                        # Check status code
                        assert call_args[1]["status_code"] == 200
    
    def test_get_app(self, mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline):
        """Test getting FastAPI app instance."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI') as mock_fastapi:
                with patch('web.api_routes.WebSocketHandler'):
                    mock_app = Mock()
                    mock_fastapi.return_value = mock_app
                    
                    api = AIAnchorAPI(mock_settings, mock_anchor_agent, mock_memory_manager, mock_audio_pipeline)
                    
                    assert api.get_app() == mock_app


class TestCreateApp:
    """Test suite for create_app function."""
    
    def test_create_app_with_fastapi(self):
        """Test create_app with FastAPI available."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.AIAnchorAPI') as mock_api_class:
                mock_api = Mock()
                mock_app = Mock()
                mock_api.get_app.return_value = mock_app
                mock_api_class.return_value = mock_api
                
                settings = Mock()
                anchor_agent = Mock()
                memory_manager = Mock()
                audio_pipeline = Mock()
                
                app = create_app(settings, anchor_agent, memory_manager, audio_pipeline)
                
                assert app == mock_app
                mock_api_class.assert_called_once_with(settings, anchor_agent, memory_manager, audio_pipeline)
    
    def test_create_app_without_fastapi(self):
        """Test create_app without FastAPI available."""
        with patch('web.api_routes.FASTAPI_AVAILABLE', False):
            settings = Mock()
            anchor_agent = Mock()
            memory_manager = Mock()
            audio_pipeline = Mock()
            
            app = create_app(settings, anchor_agent, memory_manager, audio_pipeline)
            
            assert app is None


class TestAPIRoutesIntegration:
    """Integration tests for API routes."""
    
    @pytest.mark.asyncio
    async def test_full_api_workflow(self):
        """Test full API workflow with mocked components."""
        # Create comprehensive mocks
        settings = Mock()
        settings.debug = True
        settings.web = Mock()
        settings.web.cors_origins = ["*"]
        
        anchor_agent = Mock()
        anchor_agent.get_agent_status.return_value = {"is_active": True}
        anchor_agent.get_available_personalities.return_value = {
            "professional": {"name": "Professional"},
            "friendly": {"name": "Friendly"}
        }
        anchor_agent.start_conversation = AsyncMock(return_value="Welcome!")
        anchor_agent.end_conversation = AsyncMock()
        anchor_agent.switch_personality = AsyncMock(return_value=True)
        anchor_agent.switch_language = AsyncMock()
        anchor_agent.get_conversation_history.return_value = []
        anchor_agent.health_check = AsyncMock(return_value={"overall_health": "healthy"})
        
        memory_manager = Mock()
        memory_manager.get_memory_statistics.return_value = {"memory_usage": "normal"}
        memory_manager.health_check = AsyncMock(return_value={"overall_health": "healthy"})
        
        audio_pipeline = Mock()
        audio_pipeline.get_pipeline_state.return_value = {"state": "idle"}
        audio_pipeline.health_check = AsyncMock(return_value={"overall_health": "healthy"})
        
        # Test with FastAPI available
        with patch('web.api_routes.FASTAPI_AVAILABLE', True):
            with patch('web.api_routes.FastAPI') as mock_fastapi:
                with patch('web.api_routes.WebSocketHandler') as mock_ws_handler_class:
                    mock_app = Mock()
                    mock_fastapi.return_value = mock_app
                    
                    mock_ws_handler = Mock()
                    mock_ws_handler.get_connection_stats.return_value = {"active_connections": 1}
                    mock_ws_handler.health_check = AsyncMock(return_value={"overall_health": "healthy"})
                    mock_ws_handler_class.return_value = mock_ws_handler
                    
                    # Create API
                    api = AIAnchorAPI(settings, anchor_agent, memory_manager, audio_pipeline)
                    
                    # Test health check
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        await api._health_check()
                        mock_json_response.assert_called_once()
                        
                        # Check that all components were health checked
                        anchor_agent.health_check.assert_called_once()
                        memory_manager.health_check.assert_called_once()
                        audio_pipeline.health_check.assert_called_once()
                        mock_ws_handler.health_check.assert_called_once()
                    
                    # Test start conversation
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        mock_context = Mock()
                        mock_context.session_id = "test-session"
                        anchor_agent.current_context = mock_context
                        
                        await api._start_conversation({
                            "mode": "casual_chat",
                            "personality": "friendly",
                            "language": "en"
                        })
                        
                        anchor_agent.start_conversation.assert_called_once()
                        mock_json_response.assert_called_once()
                    
                    # Test change personality
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        await api._change_personality({"personality": "energetic"})
                        
                        anchor_agent.switch_personality.assert_called_once_with("energetic")
                        mock_json_response.assert_called_once()
                    
                    # Test end conversation
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        await api._end_conversation()
                        
                        anchor_agent.end_conversation.assert_called_once()
                        mock_json_response.assert_called_once()
                    
                    # Test get status
                    with patch('web.api_routes.JSONResponse') as mock_json_response:
                        await api._get_status()
                        
                        # Check all components were queried
                        anchor_agent.get_agent_status.assert_called()
                        memory_manager.get_memory_statistics.assert_called()
                        audio_pipeline.get_pipeline_state.assert_called()
                        mock_ws_handler.get_connection_stats.assert_called()
                        mock_json_response.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])