"""
FastAPI routes for AI Anchor system.

Provides REST API endpoints and WebSocket connections for the web interface.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = Exception
    StaticFiles = None
    HTMLResponse = None
    JSONResponse = None
    CORSMiddleware = None
    FASTAPI_AVAILABLE = False

try:
    from utils.logger import get_logger
    from config import Settings
    from agents.anchor_agent import AnchorAgent, ConversationMode, AnchorPersonality
    from core.memory_manager import MemoryManager
    from core.audio_pipeline import AudioPipelineManager
    from .websocket_handler import WebSocketHandler
    logger = get_logger(__name__)
    FULL_IMPORTS_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    Settings = None
    AnchorAgent = None
    ConversationMode = None
    AnchorPersonality = None
    MemoryManager = None
    AudioPipelineManager = None
    WebSocketHandler = None
    FULL_IMPORTS_AVAILABLE = False


class AIAnchorAPI:
    """
    FastAPI application for AI Anchor system.
    
    Provides REST API endpoints and WebSocket connections for real-time
    communication with the AI anchor agent.
    """
    
    def __init__(
        self,
        settings: Settings,
        anchor_agent: AnchorAgent,
        memory_manager: MemoryManager,
        audio_pipeline: AudioPipelineManager
    ):
        """
        Initialize the API application.
        
        Args:
            settings: Application settings
            anchor_agent: AI anchor agent instance
            memory_manager: Memory manager instance
            audio_pipeline: Audio pipeline manager instance
        """
        self.settings = settings
        self.anchor_agent = anchor_agent
        self.memory_manager = memory_manager
        self.audio_pipeline = audio_pipeline
        
        # Create FastAPI app
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="AI Anchor API",
                description="Real-time AI anchor system with voice interaction",
                version="1.0.0",
                docs_url="/docs" if settings.debug else None,
                redoc_url="/redoc" if settings.debug else None,
            )
            
            # Setup CORS
            self._setup_cors()
            
            # Setup routes
            self._setup_routes()
            
            # Initialize WebSocket handler
            self.websocket_handler = WebSocketHandler(settings, anchor_agent)
            
            logger.info("🌐 AI Anchor API initialized")
        else:
            self.app = None
            self.websocket_handler = None
            logger.error("❌ FastAPI not available - API disabled")
    
    def _setup_cors(self) -> None:
        """Setup CORS middleware."""
        if not self.app:
            return
            
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.security.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        if not self.app:
            return
        
        # Static files
        static_path = Path(__file__).parent / "static"
        if static_path.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        
        # WebSocket endpoint
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self._handle_websocket_connection(websocket, client_id)
        
        # REST API endpoints
        @self.app.get("/")
        async def read_root():
            """Serve the main web interface."""
            return await self._serve_index()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return await self._health_check()
        
        @self.app.get("/api/status")
        async def get_status():
            """Get system status."""
            return await self._get_status()
        
        @self.app.get("/api/personalities")
        async def get_personalities():
            """Get available personalities."""
            return await self._get_personalities()
        
        @self.app.post("/api/conversation/start")
        async def start_conversation(request: Dict[str, Any]):
            """Start a new conversation."""
            return await self._start_conversation(request)
        
        @self.app.post("/api/conversation/end")
        async def end_conversation():
            """End the current conversation."""
            return await self._end_conversation()
        
        @self.app.post("/api/personality/change")
        async def change_personality(request: Dict[str, Any]):
            """Change personality."""
            return await self._change_personality(request)
        
        @self.app.post("/api/language/change")
        async def change_language(request: Dict[str, Any]):
            """Change language."""
            return await self._change_language(request)
        
        @self.app.get("/api/conversation/history")
        async def get_conversation_history():
            """Get conversation history."""
            return await self._get_conversation_history()
        
        @self.app.get("/api/connections")
        async def get_connections():
            """Get WebSocket connection stats."""
            return await self._get_connections()
        
        @self.app.get("/api/memory")
        async def get_memory_stats():
            """Get memory statistics."""
            return await self._get_memory_stats()
    
    async def _handle_websocket_connection(self, websocket: WebSocket, client_id: str) -> None:
        """Handle WebSocket connection."""
        if not self.websocket_handler:
            await websocket.close(code=1011, reason="WebSocket handler not available")
            return
        
        await self.websocket_handler.connect(websocket, client_id)
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                await self.websocket_handler.handle_message(client_id, data)
                
        except WebSocketDisconnect:
            logger.info(f"🌐 Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"🌐 WebSocket error for client {client_id}: {e}")
        finally:
            await self.websocket_handler.disconnect(client_id)
    
    async def _serve_index(self) -> HTMLResponse:
        """Serve the main web interface."""
        html_path = Path(__file__).parent / "static" / "index.html"
        
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(), status_code=200)
        else:
            # Return a basic HTML page if static files not available
            return HTMLResponse(content=self._get_default_html(), status_code=200)
    
    def _get_default_html(self) -> str:
        """Get default HTML page."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Anchor</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .healthy { background: #d4edda; color: #155724; }
                .error { background: #f8d7da; color: #721c24; }
                button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
                .primary { background: #007bff; color: white; }
                .success { background: #28a745; color: white; }
                .danger { background: #dc3545; color: white; }
                #messages { height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
                input, select { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎙️ AI Anchor System</h1>
                
                <div id="status" class="status healthy">
                    System Status: Loading...
                </div>
                
                <div>
                    <h3>Controls</h3>
                    <button id="connectBtn" class="primary" onclick="connect()">Connect</button>
                    <button id="disconnectBtn" class="danger" onclick="disconnect()" disabled>Disconnect</button>
                    <br>
                    
                    <select id="personality">
                        <option value="professional">Professional</option>
                        <option value="friendly">Friendly</option>
                        <option value="energetic">Energetic</option>
                        <option value="calm">Calm</option>
                        <option value="authoritative">Authoritative</option>
                    </select>
                    <button class="success" onclick="changePersonality()">Change Personality</button>
                    <br>
                    
                    <select id="language">
                        <option value="en">English</option>
                        <option value="es">Spanish</option>
                        <option value="fr">French</option>
                        <option value="de">German</option>
                    </select>
                    <button class="success" onclick="changeLanguage()">Change Language</button>
                    <br>
                    
                    <button class="primary" onclick="startConversation()">Start Conversation</button>
                    <button class="danger" onclick="endConversation()">End Conversation</button>
                </div>
                
                <div>
                    <h3>Text Input</h3>
                    <input type="text" id="textInput" placeholder="Type your message..." style="width: 60%;">
                    <button class="primary" onclick="sendText()">Send</button>
                </div>
                
                <div>
                    <h3>Messages</h3>
                    <div id="messages"></div>
                </div>
            </div>
            
            <script>
                let ws = null;
                let clientId = 'web-client-' + Math.random().toString(36).substr(2, 9);
                
                function updateStatus(message, isError = false) {
                    const statusDiv = document.getElementById('status');
                    statusDiv.textContent = 'System Status: ' + message;
                    statusDiv.className = 'status ' + (isError ? 'error' : 'healthy');
                }
                
                function addMessage(type, data) {
                    const messages = document.getElementById('messages');
                    const div = document.createElement('div');
                    div.innerHTML = `<strong>[${new Date().toLocaleTimeString()}] ${type}:</strong> ${JSON.stringify(data)}`;
                    messages.appendChild(div);
                    messages.scrollTop = messages.scrollHeight;
                }
                
                function connect() {
                    if (ws) return;
                    
                    ws = new WebSocket(\`ws://\${window.location.host}/ws/\${clientId}\`);
                    
                    ws.onopen = function() {
                        updateStatus('Connected');
                        document.getElementById('connectBtn').disabled = true;
                        document.getElementById('disconnectBtn').disabled = false;
                        addMessage('SYSTEM', 'Connected to AI Anchor');
                    };
                    
                    ws.onmessage = function(event) {
                        const message = JSON.parse(event.data);
                        addMessage(message.type, message.data);
                    };
                    
                    ws.onclose = function() {
                        updateStatus('Disconnected', true);
                        document.getElementById('connectBtn').disabled = false;
                        document.getElementById('disconnectBtn').disabled = true;
                        ws = null;
                        addMessage('SYSTEM', 'Disconnected from AI Anchor');
                    };
                    
                    ws.onerror = function(error) {
                        updateStatus('Connection Error', true);
                        addMessage('ERROR', 'WebSocket error occurred');
                    };
                }
                
                function disconnect() {
                    if (ws) {
                        ws.close();
                    }
                }
                
                function sendMessage(type, data) {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: type, data: data }));
                    } else {
                        addMessage('ERROR', 'Not connected');
                    }
                }
                
                function changePersonality() {
                    const personality = document.getElementById('personality').value;
                    sendMessage('change_personality', { personality: personality });
                }
                
                function changeLanguage() {
                    const language = document.getElementById('language').value;
                    sendMessage('change_language', { language: language });
                }
                
                function startConversation() {
                    const mode = 'casual_chat';
                    const personality = document.getElementById('personality').value;
                    const language = document.getElementById('language').value;
                    sendMessage('start_conversation', { 
                        mode: mode, 
                        personality: personality,
                        language: language 
                    });
                }
                
                function endConversation() {
                    sendMessage('end_conversation', {});
                }
                
                function sendText() {
                    const input = document.getElementById('textInput');
                    const text = input.value.trim();
                    if (text) {
                        sendMessage('text_input', { text: text });
                        input.value = '';
                    }
                }
                
                // Enter key support for text input
                document.getElementById('textInput').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendText();
                    }
                });
                
                // Load initial status
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        updateStatus('Ready - Click Connect to start');
                    })
                    .catch(error => {
                        updateStatus('API Error', true);
                    });
            </script>
        </body>
        </html>
        """
    
    async def _health_check(self) -> JSONResponse:
        """Health check endpoint."""
        try:
            health_data = {
                "timestamp": time.time(),
                "api_status": "healthy",
                "fastapi_available": FASTAPI_AVAILABLE,
            }
            
            if self.anchor_agent:
                agent_health = await self.anchor_agent.health_check()
                health_data["anchor_agent"] = agent_health
            
            if self.memory_manager:
                memory_health = await self.memory_manager.health_check()
                health_data["memory_manager"] = memory_health
            
            if self.audio_pipeline:
                pipeline_health = await self.audio_pipeline.health_check()
                health_data["audio_pipeline"] = pipeline_health
            
            if self.websocket_handler:
                ws_health = await self.websocket_handler.health_check()
                health_data["websocket_handler"] = ws_health
            
            # Determine overall health
            all_healthy = all(
                component.get("overall_health") == "healthy" 
                for component in [
                    health_data.get("anchor_agent", {}),
                    health_data.get("memory_manager", {}),
                    health_data.get("audio_pipeline", {}),
                    health_data.get("websocket_handler", {})
                ]
                if "overall_health" in component
            )
            
            health_data["overall_health"] = "healthy" if all_healthy else "degraded"
            
            return JSONResponse(content=health_data, status_code=200)
            
        except Exception as e:
            logger.error(f"🌐 Health check error: {e}")
            return JSONResponse(
                content={"error": str(e), "overall_health": "error"}, 
                status_code=500
            )
    
    async def _get_status(self) -> JSONResponse:
        """Get system status."""
        try:
            status_data = {
                "timestamp": time.time(),
                "anchor_agent": self.anchor_agent.get_agent_status() if self.anchor_agent else {},
                "memory_manager": self.memory_manager.get_memory_statistics() if self.memory_manager else {},
                "audio_pipeline": self.audio_pipeline.get_pipeline_state() if self.audio_pipeline else {},
                "websocket_connections": self.websocket_handler.get_connection_stats() if self.websocket_handler else {},
            }
            
            return JSONResponse(content=status_data, status_code=200)
            
        except Exception as e:
            logger.error(f"🌐 Status error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    async def _get_personalities(self) -> JSONResponse:
        """Get available personalities."""
        try:
            personalities = self.anchor_agent.get_available_personalities() if self.anchor_agent else {}
            
            return JSONResponse(content={"personalities": personalities}, status_code=200)
            
        except Exception as e:
            logger.error(f"🌐 Personalities error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    async def _start_conversation(self, request: Dict[str, Any]) -> JSONResponse:
        """Start a new conversation."""
        try:
            if not self.anchor_agent:
                raise HTTPException(status_code=503, detail="Anchor agent not available")
            
            mode = request.get("mode", "casual_chat")
            personality = request.get("personality")
            language = request.get("language")
            
            # Convert string values to appropriate types if needed
            if FULL_IMPORTS_AVAILABLE and ConversationMode:
                try:
                    mode = ConversationMode(mode)
                except ValueError:
                    mode = ConversationMode.CASUAL_CHAT
            
            # Convert language string to LanguageCode enum
            if language and FULL_IMPORTS_AVAILABLE:
                try:
                    from config import LanguageCode
                    language = LanguageCode(language)
                except (ValueError, AttributeError):
                    language = None  # Let the agent use default
            
            greeting = await self.anchor_agent.start_conversation(
                mode=mode,
                personality=personality,
                language=language
            )
            
            return JSONResponse(content={
                "greeting": greeting,
                "session_id": self.anchor_agent.current_context.session_id if self.anchor_agent.current_context else None,
                "mode": str(mode) if hasattr(mode, 'value') else mode,
                "personality": personality or "professional",
                "language": language.value if hasattr(language, 'value') else (language or "en")
            }, status_code=200)
            
        except Exception as e:
            logger.error(f"🌐 Start conversation error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    async def _end_conversation(self) -> JSONResponse:
        """End the current conversation."""
        try:
            if not self.anchor_agent:
                raise HTTPException(status_code=503, detail="Anchor agent not available")
            
            await self.anchor_agent.end_conversation()
            
            return JSONResponse(content={"ended": True, "timestamp": time.time()}, status_code=200)
            
        except Exception as e:
            logger.error(f"🌐 End conversation error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    async def _change_personality(self, request: Dict[str, Any]) -> JSONResponse:
        """Change personality."""
        try:
            if not self.anchor_agent:
                raise HTTPException(status_code=503, detail="Anchor agent not available")
            
            personality = request.get("personality", "").strip()
            if not personality:
                raise HTTPException(status_code=400, detail="Personality name required")
            
            success = await self.anchor_agent.switch_personality(personality)
            
            return JSONResponse(content={
                "success": success,
                "personality": personality,
                "timestamp": time.time()
            }, status_code=200)
            
        except Exception as e:
            logger.error(f"🌐 Change personality error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    async def _change_language(self, request: Dict[str, Any]) -> JSONResponse:
        """Change language."""
        try:
            if not self.anchor_agent:
                raise HTTPException(status_code=503, detail="Anchor agent not available")
            
            language_str = request.get("language", "").strip()
            if not language_str:
                raise HTTPException(status_code=400, detail="Language code required")
            
            # Convert to LanguageCode if available
            if FULL_IMPORTS_AVAILABLE:
                try:
                    from config import LanguageCode
                    language = LanguageCode(language_str)
                except (ValueError, AttributeError):
                    raise HTTPException(status_code=400, detail=f"Unsupported language: {language_str}")
            else:
                language = language_str
            
            await self.anchor_agent.switch_language(language)
            
            return JSONResponse(content={
                "language": language_str,
                "timestamp": time.time()
            }, status_code=200)
            
        except Exception as e:
            logger.error(f"🌐 Change language error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    async def _get_conversation_history(self) -> JSONResponse:
        """Get conversation history."""
        try:
            if not self.anchor_agent:
                raise HTTPException(status_code=503, detail="Anchor agent not available")
            
            history = self.anchor_agent.get_conversation_history()
            
            return JSONResponse(content={"history": history}, status_code=200)
            
        except Exception as e:
            logger.error(f"🌐 Conversation history error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    async def _get_connections(self) -> JSONResponse:
        """Get WebSocket connection stats."""
        try:
            if not self.websocket_handler:
                return JSONResponse(content={"connections": {}}, status_code=200)
            
            stats = self.websocket_handler.get_connection_stats()
            
            return JSONResponse(content=stats, status_code=200)
            
        except Exception as e:
            logger.error(f"🌐 Connections error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    async def _get_memory_stats(self) -> JSONResponse:
        """Get memory statistics."""
        try:
            if not self.memory_manager:
                return JSONResponse(content={"memory": "unavailable"}, status_code=200)
            
            stats = self.memory_manager.get_memory_statistics()
            
            return JSONResponse(content=stats, status_code=200)
            
        except Exception as e:
            logger.error(f"🌐 Memory stats error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    def get_app(self) -> Optional[FastAPI]:
        """Get the FastAPI application instance."""
        return self.app


def create_app(
    settings: Settings,
    anchor_agent: AnchorAgent,
    memory_manager: MemoryManager,
    audio_pipeline: AudioPipelineManager
) -> Optional[FastAPI]:
    """
    Create and configure the FastAPI application.
    
    Args:
        settings: Application settings
        anchor_agent: AI anchor agent instance
        memory_manager: Memory manager instance
        audio_pipeline: Audio pipeline manager instance
        
    Returns:
        FastAPI application instance or None if not available
    """
    if not FASTAPI_AVAILABLE:
        logger.error("❌ FastAPI not available - cannot create web application")
        return None
    
    api = AIAnchorAPI(settings, anchor_agent, memory_manager, audio_pipeline)
    return api.get_app()