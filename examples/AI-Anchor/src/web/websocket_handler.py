"""
WebSocket handler for AI Anchor system.

Manages real-time communication between clients and the AI anchor agent,
handling audio streaming, text input, and control messages.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import asdict
from enum import Enum

try:
    from fastapi import WebSocket, WebSocketDisconnect, status
    from fastapi.websockets import WebSocketState
    FASTAPI_AVAILABLE = True
except ImportError:
    WebSocket = None
    WebSocketDisconnect = Exception
    WebSocketState = None
    FASTAPI_AVAILABLE = False

try:
    from utils.logger import get_logger
    from agents.anchor_agent import AnchorAgent, ConversationMode, AnchorPersonality
    from config import Settings, LanguageCode, AudioFormat
    logger = get_logger(__name__)
    FULL_IMPORTS_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    AnchorAgent = None
    ConversationMode = None
    AnchorPersonality = None
    Settings = None
    LanguageCode = str
    AudioFormat = None
    FULL_IMPORTS_AVAILABLE = False


class MessageType(str, Enum):
    """WebSocket message types."""
    # Client to Server
    AUDIO_CHUNK = "audio_chunk"
    TEXT_INPUT = "text_input"
    START_CONVERSATION = "start_conversation"
    END_CONVERSATION = "end_conversation"
    CHANGE_PERSONALITY = "change_personality"
    CHANGE_LANGUAGE = "change_language"
    GET_STATUS = "get_status"
    GET_PERSONALITIES = "get_personalities"
    ABORT_RESPONSE = "abort_response"
    
    # Server to Client
    TRANSCRIPTION = "transcription"
    RESPONSE_START = "response_start"
    RESPONSE_CHUNK = "response_chunk"
    AUDIO_OUTPUT = "audio_output"
    SESSION_COMPLETE = "session_complete"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    GREETING = "greeting"
    PERSONALITY_CHANGED = "personality_changed"
    CONVERSATION_ENDED = "conversation_ended"


class WebSocketHandler:
    """
    Handles WebSocket connections for the AI Anchor system.
    
    Manages real-time communication between web clients and the anchor agent,
    including audio streaming, text processing, and control commands.
    """
    
    def __init__(self, settings: Settings, anchor_agent: AnchorAgent):
        """
        Initialize WebSocket handler.
        
        Args:
            settings: Application settings
            anchor_agent: AI anchor agent instance
        """
        self.settings = settings
        self.anchor_agent = anchor_agent
        
        # Active connections
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Message queues for each connection
        self.message_queues: Dict[str, asyncio.Queue] = {}
        
        # Setup anchor agent callbacks
        self._setup_anchor_callbacks()
        
        logger.info("🌐 WebSocket handler initialized")
    
    def _setup_anchor_callbacks(self) -> None:
        """Setup callbacks from the anchor agent."""
        if not self.anchor_agent:
            return
            
        # Set up callbacks to relay events to connected clients
        self.anchor_agent.on_transcription = self._on_transcription
        self.anchor_agent.on_response_start = self._on_response_start
        self.anchor_agent.on_response_chunk = self._on_response_chunk
        self.anchor_agent.on_audio_chunk = self._on_audio_chunk
        self.anchor_agent.on_session_complete = self._on_session_complete
        self.anchor_agent.on_error = self._on_error
        self.anchor_agent.on_personality_change = self._on_personality_change
        self.anchor_agent.on_conversation_start = self._on_conversation_start
        self.anchor_agent.on_conversation_end = self._on_conversation_end
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
        """
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available for WebSocket connections")
            return
            
        await websocket.accept()
        
        # Store connection
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": time.time(),
            "conversation_active": False,
            "personality": "professional",
            "language": "en",
            "message_count": 0,
        }
        
        # Create message queue for this connection
        self.message_queues[client_id] = asyncio.Queue()
        
        logger.info(f"🌐 Client {client_id} connected")
        
        # Send initial status
        await self._send_to_client(client_id, {
            "type": MessageType.STATUS_UPDATE,
            "data": {
                "connected": True,
                "agent_status": self.anchor_agent.get_agent_status() if self.anchor_agent else {},
                "available_personalities": self.anchor_agent.get_available_personalities() if self.anchor_agent else {},
            }
        })
    
    async def disconnect(self, client_id: str) -> None:
        """
        Handle client disconnection.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.active_connections:
            # End any active conversation
            if self.connection_metadata[client_id].get("conversation_active"):
                if self.anchor_agent:
                    await self.anchor_agent.end_conversation()
            
            # Clean up
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            if client_id in self.message_queues:
                del self.message_queues[client_id]
            
            logger.info(f"🌐 Client {client_id} disconnected")
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]) -> None:
        """
        Handle incoming message from client.
        
        Args:
            client_id: Client identifier
            message: Incoming message
        """
        if client_id not in self.active_connections:
            logger.warning(f"🌐 Message from unknown client: {client_id}")
            return
        
        try:
            message_type = message.get("type")
            data = message.get("data", {})
            
            # Update message count
            self.connection_metadata[client_id]["message_count"] += 1
            
            logger.debug(f"🌐 Received {message_type} from {client_id}")
            
            # Route message to appropriate handler
            if message_type == MessageType.AUDIO_CHUNK:
                await self._handle_audio_chunk(client_id, data)
            elif message_type == MessageType.TEXT_INPUT:
                await self._handle_text_input(client_id, data)
            elif message_type == MessageType.START_CONVERSATION:
                await self._handle_start_conversation(client_id, data)
            elif message_type == MessageType.END_CONVERSATION:
                await self._handle_end_conversation(client_id, data)
            elif message_type == MessageType.CHANGE_PERSONALITY:
                await self._handle_change_personality(client_id, data)
            elif message_type == MessageType.CHANGE_LANGUAGE:
                await self._handle_change_language(client_id, data)
            elif message_type == MessageType.GET_STATUS:
                await self._handle_get_status(client_id, data)
            elif message_type == MessageType.GET_PERSONALITIES:
                await self._handle_get_personalities(client_id, data)
            elif message_type == MessageType.ABORT_RESPONSE:
                await self._handle_abort_response(client_id, data)
            else:
                await self._send_error(client_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"🌐 Error handling message from {client_id}: {e}")
            await self._send_error(client_id, f"Message handling error: {str(e)}")
    
    async def _handle_audio_chunk(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle audio chunk from client."""
        if not self.anchor_agent:
            await self._send_error(client_id, "Anchor agent not available")
            return
            
        try:
            # Extract audio data (base64 encoded)
            import base64
            audio_base64 = data.get("audio_data", "")
            audio_bytes = base64.b64decode(audio_base64)
            
            # Feed to anchor agent
            self.anchor_agent.process_audio_input(audio_bytes)
            
        except Exception as e:
            logger.error(f"🌐 Error processing audio chunk: {e}")
            await self._send_error(client_id, f"Audio processing error: {str(e)}")
    
    async def _handle_text_input(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle text input from client."""
        if not self.anchor_agent:
            await self._send_error(client_id, "Anchor agent not available")
            return
            
        try:
            text = data.get("text", "").strip()
            if not text:
                await self._send_error(client_id, "Empty text input")
                return
            
            # Process through anchor agent
            self.anchor_agent.process_text_input(text)
            
        except Exception as e:
            logger.error(f"🌐 Error processing text input: {e}")
            await self._send_error(client_id, f"Text processing error: {str(e)}")
    
    async def _handle_start_conversation(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle start conversation request."""
        if not self.anchor_agent:
            await self._send_error(client_id, "Anchor agent not available")
            return
            
        try:
            # Extract parameters
            mode_str = data.get("mode", "casual_chat")
            personality = data.get("personality")
            language = data.get("language")
            
            # Convert mode string to enum if available
            if FULL_IMPORTS_AVAILABLE and ConversationMode:
                try:
                    mode = ConversationMode(mode_str)
                except ValueError:
                    mode = ConversationMode.CASUAL_CHAT
            else:
                mode = mode_str
            
            # Convert language string to enum if available
            if FULL_IMPORTS_AVAILABLE and language:
                try:
                    language = LanguageCode(language)
                except (ValueError, AttributeError):
                    language = None
            
            # Start conversation
            greeting = await self.anchor_agent.start_conversation(
                mode=mode,
                personality=personality,
                language=language
            )
            
            # Update connection metadata
            self.connection_metadata[client_id]["conversation_active"] = True
            if personality:
                self.connection_metadata[client_id]["personality"] = personality
            if language:
                self.connection_metadata[client_id]["language"] = str(language)
            
            # Send greeting
            await self._send_to_client(client_id, {
                "type": MessageType.GREETING,
                "data": {
                    "greeting": greeting,
                    "session_id": self.anchor_agent.current_context.session_id if self.anchor_agent.current_context else None,
                    "mode": mode_str,
                    "personality": personality or "professional",
                    "language": str(language) if language else "en"
                }
            })
            
        except Exception as e:
            logger.error(f"🌐 Error starting conversation: {e}")
            await self._send_error(client_id, f"Failed to start conversation: {str(e)}")
    
    async def _handle_end_conversation(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle end conversation request."""
        if not self.anchor_agent:
            return
            
        try:
            await self.anchor_agent.end_conversation()
            
            # Update connection metadata
            self.connection_metadata[client_id]["conversation_active"] = False
            
            # Send confirmation
            await self._send_to_client(client_id, {
                "type": MessageType.CONVERSATION_ENDED,
                "data": {
                    "ended_at": time.time(),
                }
            })
            
        except Exception as e:
            logger.error(f"🌐 Error ending conversation: {e}")
            await self._send_error(client_id, f"Failed to end conversation: {str(e)}")
    
    async def _handle_change_personality(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle personality change request."""
        if not self.anchor_agent:
            await self._send_error(client_id, "Anchor agent not available")
            return
            
        try:
            personality_name = data.get("personality", "").strip()
            if not personality_name:
                await self._send_error(client_id, "Personality name required")
                return
            
            success = await self.anchor_agent.switch_personality(personality_name)
            
            if success:
                self.connection_metadata[client_id]["personality"] = personality_name
            else:
                await self._send_error(client_id, f"Failed to switch to personality: {personality_name}")
                
        except Exception as e:
            logger.error(f"🌐 Error changing personality: {e}")
            await self._send_error(client_id, f"Personality change error: {str(e)}")
    
    async def _handle_change_language(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle language change request."""
        if not self.anchor_agent:
            await self._send_error(client_id, "Anchor agent not available")
            return
            
        try:
            language_str = data.get("language", "").strip()
            if not language_str:
                await self._send_error(client_id, "Language code required")
                return
            
            # Convert to LanguageCode if available
            if FULL_IMPORTS_AVAILABLE:
                try:
                    language = LanguageCode(language_str)
                except (ValueError, AttributeError):
                    await self._send_error(client_id, f"Unsupported language: {language_str}")
                    return
            else:
                language = language_str
            
            await self.anchor_agent.switch_language(language)
            self.connection_metadata[client_id]["language"] = language_str
            
            # Send confirmation
            await self._send_to_client(client_id, {
                "type": MessageType.STATUS_UPDATE,
                "data": {
                    "language_changed": language_str,
                    "timestamp": time.time()
                }
            })
            
        except Exception as e:
            logger.error(f"🌐 Error changing language: {e}")
            await self._send_error(client_id, f"Language change error: {str(e)}")
    
    async def _handle_get_status(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle status request."""
        try:
            status = {
                "agent_status": self.anchor_agent.get_agent_status() if self.anchor_agent else {},
                "connection_status": self.connection_metadata.get(client_id, {}),
                "active_connections": len(self.active_connections),
                "timestamp": time.time()
            }
            
            await self._send_to_client(client_id, {
                "type": MessageType.STATUS_UPDATE,
                "data": status
            })
            
        except Exception as e:
            logger.error(f"🌐 Error getting status: {e}")
            await self._send_error(client_id, f"Status error: {str(e)}")
    
    async def _handle_get_personalities(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle get personalities request."""
        try:
            personalities = self.anchor_agent.get_available_personalities() if self.anchor_agent else {}
            
            await self._send_to_client(client_id, {
                "type": MessageType.STATUS_UPDATE,
                "data": {
                    "personalities": personalities,
                    "timestamp": time.time()
                }
            })
            
        except Exception as e:
            logger.error(f"🌐 Error getting personalities: {e}")
            await self._send_error(client_id, f"Personalities error: {str(e)}")
    
    async def _handle_abort_response(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle abort response request."""
        if not self.anchor_agent:
            return
            
        try:
            self.anchor_agent.abort_current_response()
            
            await self._send_to_client(client_id, {
                "type": MessageType.STATUS_UPDATE,
                "data": {
                    "response_aborted": True,
                    "timestamp": time.time()
                }
            })
            
        except Exception as e:
            logger.error(f"🌐 Error aborting response: {e}")
            await self._send_error(client_id, f"Abort error: {str(e)}")
    
    # Anchor agent event handlers
    
    def _on_transcription(self, result) -> None:
        """Handle transcription from anchor agent."""
        if not hasattr(result, 'text'):
            return
            
        asyncio.create_task(self._broadcast_to_all({
            "type": MessageType.TRANSCRIPTION,
            "data": {
                "text": result.text,
                "language": str(result.language) if hasattr(result, 'language') else "en",
                "confidence": getattr(result, 'confidence', 1.0),
                "is_final": getattr(result, 'is_final', False),
                "timestamp": getattr(result, 'timestamp', time.time())
            }
        }))
    
    def _on_response_start(self, input_text: str) -> None:
        """Handle response start from anchor agent."""
        asyncio.create_task(self._broadcast_to_all({
            "type": MessageType.RESPONSE_START,
            "data": {
                "input_text": input_text,
                "timestamp": time.time()
            }
        }))
    
    def _on_response_chunk(self, chunk: str) -> None:
        """Handle response chunk from anchor agent."""
        asyncio.create_task(self._broadcast_to_all({
            "type": MessageType.RESPONSE_CHUNK,
            "data": {
                "chunk": chunk,
                "timestamp": time.time()
            }
        }))
    
    def _on_audio_chunk(self, audio_chunk) -> None:
        """Handle audio chunk from anchor agent."""
        import base64
        
        try:
            # Convert audio data to base64 for transmission
            audio_base64 = base64.b64encode(audio_chunk.data).decode('utf-8')
            
            asyncio.create_task(self._broadcast_to_all({
                "type": MessageType.AUDIO_OUTPUT,
                "data": {
                    "audio_data": audio_base64,
                    "format": str(audio_chunk.format) if hasattr(audio_chunk, 'format') else "wav",
                    "sample_rate": getattr(audio_chunk, 'sample_rate', 24000),
                    "duration": getattr(audio_chunk, 'duration', 0.0),
                    "timestamp": time.time()
                }
            }))
            
        except Exception as e:
            logger.error(f"🌐 Error processing audio chunk: {e}")
    
    def _on_session_complete(self, session) -> None:
        """Handle session completion from anchor agent."""
        asyncio.create_task(self._broadcast_to_all({
            "type": MessageType.SESSION_COMPLETE,
            "data": {
                "session_id": getattr(session, 'session_id', 'unknown'),
                "input_text": getattr(session, 'input_text', ''),
                "response": getattr(session, 'llm_response', ''),
                "timestamp": time.time()
            }
        }))
    
    def _on_error(self, error: Exception) -> None:
        """Handle error from anchor agent."""
        asyncio.create_task(self._broadcast_to_all({
            "type": MessageType.ERROR,
            "data": {
                "error": str(error),
                "timestamp": time.time()
            }
        }))
    
    def _on_personality_change(self, personality) -> None:
        """Handle personality change from anchor agent."""
        asyncio.create_task(self._broadcast_to_all({
            "type": MessageType.PERSONALITY_CHANGED,
            "data": {
                "personality": getattr(personality, 'name', 'Unknown'),
                "personality_type": str(getattr(personality, 'personality', 'unknown')),
                "voice_profile": getattr(personality, 'voice_profile', 'default'),
                "language": str(getattr(personality, 'language', 'en')),
                "timestamp": time.time()
            }
        }))
    
    def _on_conversation_start(self, context) -> None:
        """Handle conversation start from anchor agent."""
        asyncio.create_task(self._broadcast_to_all({
            "type": MessageType.STATUS_UPDATE,
            "data": {
                "conversation_started": True,
                "session_id": getattr(context, 'session_id', 'unknown'),
                "mode": str(getattr(context, 'mode', 'unknown')),
                "timestamp": time.time()
            }
        }))
    
    def _on_conversation_end(self, context) -> None:
        """Handle conversation end from anchor agent."""
        asyncio.create_task(self._broadcast_to_all({
            "type": MessageType.CONVERSATION_ENDED,
            "data": {
                "session_id": getattr(context, 'session_id', 'unknown'),
                "total_exchanges": getattr(context, 'total_exchanges', 0),
                "timestamp": time.time()
            }
        }))
    
    # Utility methods
    
    async def _send_to_client(self, client_id: str, message: Dict[str, Any]) -> None:
        """Send message to specific client."""
        if client_id not in self.active_connections:
            return
        
        websocket = self.active_connections[client_id]
        
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
            else:
                # Connection is closed, clean up
                await self.disconnect(client_id)
                
        except Exception as e:
            logger.error(f"🌐 Error sending to client {client_id}: {e}")
            await self.disconnect(client_id)
    
    async def _broadcast_to_all(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        # Send to all clients concurrently
        tasks = []
        for client_id in list(self.active_connections.keys()):
            tasks.append(self._send_to_client(client_id, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_error(self, client_id: str, error_message: str) -> None:
        """Send error message to client."""
        await self._send_to_client(client_id, {
            "type": MessageType.ERROR,
            "data": {
                "error": error_message,
                "timestamp": time.time()
            }
        })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "active_connections": len(self.active_connections),
            "connections": {
                client_id: {
                    "connected_at": metadata["connected_at"],
                    "conversation_active": metadata["conversation_active"],
                    "personality": metadata["personality"],
                    "language": metadata["language"],
                    "message_count": metadata["message_count"],
                    "connected_duration": time.time() - metadata["connected_at"]
                }
                for client_id, metadata in self.connection_metadata.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on WebSocket handler."""
        agent_health = await self.anchor_agent.health_check() if self.anchor_agent else {"overall_health": "unavailable"}
        
        return {
            "websocket_handler": "healthy",
            "active_connections": len(self.active_connections),
            "fastapi_available": FASTAPI_AVAILABLE,
            "anchor_agent_health": agent_health["overall_health"],
            "connection_stats": self.get_connection_stats(),
            "overall_health": "healthy" if FASTAPI_AVAILABLE and agent_health["overall_health"] == "healthy" else "degraded"
        }