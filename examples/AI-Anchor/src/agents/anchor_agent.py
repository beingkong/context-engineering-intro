"""
AI Anchor Agent - Main orchestrator for the AI anchor system.

Manages conversation flow, personality switching, and high-level coordination
of the audio pipeline components.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from contextlib import asynccontextmanager

try:
    from utils.logger import get_logger, log_performance
    logger = get_logger(__name__)
    
    from config import Settings, LanguageCode, AudioChunk, TranscriptionResult
    from core.audio_pipeline import AudioPipelineManager, GenerationSession, PipelineState
    from core.memory_manager import MemoryManager
    FULL_IMPORTS_AVAILABLE = True
except ImportError:
    # Fallback for testing without full dependencies
    import logging
    logger = logging.getLogger(__name__)
    
    # Define stub types for testing
    Settings = None
    LanguageCode = str
    AudioChunk = None
    TranscriptionResult = None
    AudioPipelineManager = None
    GenerationSession = None
    PipelineState = None
    MemoryManager = None
    FULL_IMPORTS_AVAILABLE = False
    
    def log_performance(operation: str):
        def decorator(func):
            return func
        return decorator


class AnchorPersonality(str, Enum):
    """Available anchor personalities."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    ENERGETIC = "energetic"
    CALM = "calm"
    AUTHORITATIVE = "authoritative"


class ConversationMode(str, Enum):
    """Conversation modes for the anchor."""
    INTERVIEW = "interview"
    NEWS_READING = "news_reading"
    CASUAL_CHAT = "casual_chat"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"


@dataclass
class VoicePersonality:
    """
    Defines a voice personality configuration.
    """
    name: str
    personality: AnchorPersonality
    voice_profile: str
    language: LanguageCode
    temperature: float = 0.7
    system_prompt: str = ""
    greeting_message: str = ""
    
    # Voice characteristics
    speaking_rate: float = 1.0
    pitch_variation: float = 0.5
    emotion_intensity: float = 0.5


@dataclass
class ConversationContext:
    """
    Maintains context for ongoing conversations.
    """
    session_id: str
    mode: ConversationMode
    current_personality: VoicePersonality
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    topic_context: str = ""
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    total_exchanges: int = 0
    
    def add_exchange(self, user_input: str, anchor_response: str) -> None:
        """Add a conversation exchange to history."""
        self.conversation_history.append({
            "timestamp": time.time(),
            "user_input": user_input,
            "anchor_response": anchor_response,
            "exchange_number": self.total_exchanges + 1
        })
        self.total_exchanges += 1
        
        # Keep only last 10 exchanges to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]


class AnchorAgent:
    """
    Main AI Anchor agent that orchestrates the entire conversation system.
    
    Manages personality switching, conversation flow, and coordinates
    the audio pipeline for real-time voice-to-voice interaction.
    """
    
    def __init__(
        self,
        settings: Settings,
        memory_manager: MemoryManager,
        audio_pipeline: AudioPipelineManager,
    ):
        """
        Initialize the anchor agent.
        
        Args:
            settings: Application settings
            memory_manager: Memory management system
            audio_pipeline: Audio processing pipeline
        """
        self.settings = settings
        self.memory_manager = memory_manager
        self.audio_pipeline = audio_pipeline
        
        # Agent state
        self.is_active = False
        self.current_context: Optional[ConversationContext] = None
        self.session_counter = 0
        
        # Voice personalities
        self.personalities = self._initialize_personalities()
        self.current_personality = self.personalities["professional"]
        
        # Callbacks
        self.on_conversation_start: Optional[Callable[[ConversationContext], None]] = None
        self.on_conversation_end: Optional[Callable[[ConversationContext], None]] = None
        self.on_personality_change: Optional[Callable[[VoicePersonality], None]] = None
        self.on_response_generated: Optional[Callable[[str, str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Performance tracking
        self.session_stats = {
            "total_sessions": 0,
            "total_exchanges": 0,
            "average_response_time": 0.0,
            "personality_switches": 0,
            "errors": 0,
        }
        
        logger.info("🎙️ AnchorAgent initialized")
    
    def _initialize_personalities(self) -> Dict[str, VoicePersonality]:
        """Initialize available voice personalities."""
        personalities = {}
        
        # Professional news anchor
        personalities["professional"] = VoicePersonality(
            name="Professional",
            personality=AnchorPersonality.PROFESSIONAL,
            voice_profile="news_anchor_neutral",
            language=LanguageCode.EN,
            temperature=0.3,
            system_prompt="""You are a professional news anchor. Speak clearly, 
            authoritatively, and maintain journalistic objectivity. Use formal 
            language and structure your responses with clear, concise information.""",
            greeting_message="Good evening, I'm your AI news anchor. How may I assist you today?",
            speaking_rate=0.9,
            pitch_variation=0.3,
            emotion_intensity=0.2
        )
        
        # Friendly talk show host
        personalities["friendly"] = VoicePersonality(
            name="Friendly",
            personality=AnchorPersonality.FRIENDLY,
            voice_profile="talk_show_warm",
            language=LanguageCode.EN,
            temperature=0.6,
            system_prompt="""You are a warm, friendly talk show host. Be conversational,
            engaging, and show genuine interest in topics. Use a casual but polished tone
            and ask follow-up questions to keep the conversation flowing.""",
            greeting_message="Hello there! Welcome to the show. I'm excited to chat with you!",
            speaking_rate=1.0,
            pitch_variation=0.6,
            emotion_intensity=0.7
        )
        
        # Energetic entertainment host
        personalities["energetic"] = VoicePersonality(
            name="Energetic",
            personality=AnchorPersonality.ENERGETIC,
            voice_profile="entertainment_upbeat",
            language=LanguageCode.EN,
            temperature=0.8,
            system_prompt="""You are an energetic entertainment host. Be enthusiastic,
            dynamic, and engaging. Use expressive language, vary your tone, and create
            excitement around topics. Keep the energy high and the mood positive.""",
            greeting_message="Hey everyone! Welcome to the show! I'm absolutely thrilled to be here with you!",
            speaking_rate=1.1,
            pitch_variation=0.8,
            emotion_intensity=0.9
        )
        
        # Calm educational presenter
        personalities["calm"] = VoicePersonality(
            name="Calm",
            personality=AnchorPersonality.CALM,
            voice_profile="educational_soothing",
            language=LanguageCode.EN,
            temperature=0.4,
            system_prompt="""You are a calm, educational presenter. Speak in a soothing,
            measured tone. Explain concepts clearly and patiently. Use a gentle approach
            and provide thoughtful, well-structured information.""",
            greeting_message="Welcome. I'm here to help you learn and explore topics together in a relaxed setting.",
            speaking_rate=0.8,
            pitch_variation=0.4,
            emotion_intensity=0.3
        )
        
        # Authoritative expert
        personalities["authoritative"] = VoicePersonality(
            name="Authoritative",
            personality=AnchorPersonality.AUTHORITATIVE,
            voice_profile="expert_confident",
            language=LanguageCode.EN,
            temperature=0.2,
            system_prompt="""You are an authoritative expert and presenter. Speak with
            confidence and authority on topics. Use precise language, cite facts when
            relevant, and maintain a commanding presence while being respectful.""",
            greeting_message="Greetings. I'm here to provide authoritative insights and analysis on your topics of interest.",
            speaking_rate=0.85,
            pitch_variation=0.3,
            emotion_intensity=0.4
        )
        
        return personalities
    
    async def initialize(self) -> None:
        """Initialize the anchor agent and its components."""
        logger.info("🎙️ Initializing anchor agent...")
        
        try:
            # Setup audio pipeline callbacks
            self._setup_pipeline_callbacks()
            
            # Set initial personality
            await self._apply_personality(self.current_personality)
            
            self.is_active = True
            logger.info("✅ Anchor agent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize anchor agent: {e}")
            if self.on_error:
                self.on_error(e)
            raise
    
    def _setup_pipeline_callbacks(self) -> None:
        """Setup callbacks for the audio pipeline."""
        # Transcription callback
        self.audio_pipeline.on_transcription = self._on_transcription_received
        
        # Response callbacks
        self.audio_pipeline.on_response_start = self._on_response_start
        self.audio_pipeline.on_response_chunk = self._on_response_chunk
        
        # Audio output callback
        self.audio_pipeline.on_audio_chunk = self._on_audio_chunk_generated
        
        # Session completion callback
        self.audio_pipeline.on_session_complete = self._on_session_complete
        
        # Error callback
        self.audio_pipeline.on_error = self._on_pipeline_error
    
    async def start_conversation(
        self,
        mode: ConversationMode = ConversationMode.CASUAL_CHAT,
        personality: Optional[str] = None,
        language: Optional[LanguageCode] = None
    ) -> str:
        """
        Start a new conversation session.
        
        Args:
            mode: Conversation mode
            personality: Personality name (optional)
            language: Language for conversation (optional)
            
        Returns:
            Session ID
        """
        if not self.is_active:
            raise RuntimeError("Anchor agent not initialized")
        
        self.session_counter += 1
        session_id = f"conversation-{self.session_counter}"
        
        # Switch personality if requested
        if personality and personality in self.personalities:
            await self.switch_personality(personality)
        
        # Switch language if requested
        if language:
            await self.switch_language(language)
        
        # Create conversation context
        self.current_context = ConversationContext(
            session_id=session_id,
            mode=mode,
            current_personality=self.current_personality,
        )
        
        # Update statistics
        self.session_stats["total_sessions"] += 1
        
        # Trigger callback
        if self.on_conversation_start:
            self.on_conversation_start(self.current_context)
        
        logger.info(f"🎙️ Started conversation session: {session_id} (mode: {mode.value})")
        
        # Return greeting
        return self.current_personality.greeting_message
    
    async def end_conversation(self) -> None:
        """End the current conversation session."""
        if self.current_context:
            # Trigger callback
            if self.on_conversation_end:
                self.on_conversation_end(self.current_context)
            
            logger.info(f"🎙️ Ended conversation session: {self.current_context.session_id}")
            self.current_context = None
        
        # Abort any ongoing generation
        self.audio_pipeline.abort_current_generation()
    
    async def switch_personality(self, personality_name: str) -> bool:
        """
        Switch to a different personality.
        
        Args:
            personality_name: Name of the personality to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if personality_name not in self.personalities:
            logger.warning(f"🎙️ Unknown personality: {personality_name}")
            return False
        
        try:
            new_personality = self.personalities[personality_name]
            await self._apply_personality(new_personality)
            
            self.current_personality = new_personality
            self.session_stats["personality_switches"] += 1
            
            # Update current context if active
            if self.current_context:
                self.current_context.current_personality = new_personality
            
            # Trigger callback
            if self.on_personality_change:
                self.on_personality_change(new_personality)
            
            logger.info(f"🎙️ Switched to personality: {personality_name}")
            return True
            
        except Exception as e:
            logger.error(f"🎙️ Failed to switch personality: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def _apply_personality(self, personality: VoicePersonality) -> None:
        """Apply a personality configuration to the pipeline."""
        # Change voice profile
        self.audio_pipeline.change_voice_profile(personality.voice_profile)
        
        # Change language
        self.audio_pipeline.change_language(personality.language)
        
        # Update LLM system prompt (if the model supports it)
        if hasattr(self.audio_pipeline.llm_model, 'set_system_prompt'):
            self.audio_pipeline.llm_model.set_system_prompt(personality.system_prompt)
        
        # Update LLM temperature
        if hasattr(self.audio_pipeline.llm_model, 'set_temperature'):
            self.audio_pipeline.llm_model.set_temperature(personality.temperature)
    
    async def switch_language(self, language: LanguageCode) -> None:
        """
        Switch the conversation language.
        
        Args:
            language: Target language
        """
        self.audio_pipeline.change_language(language)
        
        # Update current personality language
        self.current_personality.language = language
        
        logger.info(f"🎙️ Switched language to: {language.value}")
    
    def process_audio_input(self, audio_data: bytes) -> None:
        """
        Process incoming audio data.
        
        Args:
            audio_data: Raw audio bytes
        """
        if self.is_active:
            self.audio_pipeline.feed_audio(audio_data)
    
    def process_text_input(self, text: str) -> None:
        """
        Process direct text input (bypass STT).
        
        Args:
            text: Input text
        """
        if self.is_active:
            language = self.current_personality.language
            self.audio_pipeline.process_text_input(text, language)
    
    def abort_current_response(self) -> None:
        """Abort the current response generation."""
        self.audio_pipeline.abort_current_generation()
    
    def set_conversation_context(self, topic: str, user_preferences: Dict[str, Any] = None) -> None:
        """
        Set conversation context and user preferences.
        
        Args:
            topic: Topic context for the conversation
            user_preferences: User preference settings
        """
        if self.current_context:
            self.current_context.topic_context = topic
            if user_preferences:
                self.current_context.user_preferences.update(user_preferences)
    
    # Pipeline event handlers
    
    def _on_transcription_received(self, result: TranscriptionResult) -> None:
        """Handle transcription results from the pipeline."""
        if result.is_final and self.current_context:
            logger.debug(f"🎙️ Received transcription: {result.text}")
    
    def _on_response_start(self, input_text: str) -> None:
        """Handle response generation start."""
        logger.debug(f"🎙️ Starting response for: {input_text[:50]}...")
    
    def _on_response_chunk(self, chunk: str) -> None:
        """Handle response chunk generation."""
        # Could be used for real-time response display
        pass
    
    def _on_audio_chunk_generated(self, audio_chunk: AudioChunk) -> None:
        """Handle generated audio chunks."""
        # Audio chunks are ready for playback
        pass
    
    def _on_session_complete(self, session: GenerationSession) -> None:
        """Handle completed generation session."""
        if self.current_context and session.input_text and session.llm_response:
            # Add to conversation history
            self.current_context.add_exchange(
                session.input_text,
                session.llm_response
            )
            
            # Update statistics
            self.session_stats["total_exchanges"] += 1
            
            # Trigger callback
            if self.on_response_generated:
                self.on_response_generated(session.input_text, session.llm_response)
            
            logger.debug(f"🎙️ Completed exchange #{self.current_context.total_exchanges}")
    
    def _on_pipeline_error(self, error: Exception) -> None:
        """Handle pipeline errors."""
        logger.error(f"🎙️ Pipeline error: {error}")
        self.session_stats["errors"] += 1
        
        if self.on_error:
            self.on_error(error)
    
    # Public status and control methods
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        
        Returns:
            Status information dictionary
        """
        pipeline_state = self.audio_pipeline.get_pipeline_state()
        
        return {
            "is_active": self.is_active,
            "current_personality": self.current_personality.name,
            "current_language": self.current_personality.language.value,
            "conversation_active": self.current_context is not None,
            "session_id": self.current_context.session_id if self.current_context else None,
            "conversation_mode": self.current_context.mode.value if self.current_context else None,
            "total_exchanges": self.current_context.total_exchanges if self.current_context else 0,
            "pipeline_state": pipeline_state["state"],
            "available_personalities": list(self.personalities.keys()),
            "statistics": self.session_stats.copy(),
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the anchor agent.
        
        Returns:
            Health status dictionary
        """
        pipeline_health = await self.audio_pipeline.health_check()
        
        return {
            "agent_active": self.is_active,
            "conversation_active": self.current_context is not None,
            "current_personality": self.current_personality.name,
            "personality_count": len(self.personalities),
            "pipeline_health": pipeline_health["overall_health"],
            "statistics": self.session_stats.copy(),
            "overall_health": "healthy" if self.is_active and pipeline_health["overall_health"] == "healthy" else "degraded"
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get current conversation history.
        
        Returns:
            List of conversation exchanges
        """
        if self.current_context:
            return self.current_context.conversation_history.copy()
        return []
    
    def get_available_personalities(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available personalities.
        
        Returns:
            Dictionary of personality information
        """
        return {
            name: {
                "name": personality.name,
                "personality_type": personality.personality.value,
                "language": personality.language.value,
                "voice_profile": personality.voice_profile,
                "description": personality.system_prompt[:100] + "..." if len(personality.system_prompt) > 100 else personality.system_prompt
            }
            for name, personality in self.personalities.items()
        }
    
    async def shutdown(self) -> None:
        """Shutdown the anchor agent and cleanup resources."""
        logger.info("🎙️ Shutting down anchor agent...")
        
        # End current conversation
        if self.current_context:
            await self.end_conversation()
        
        self.is_active = False
        
        logger.info("✅ Anchor agent shutdown complete")