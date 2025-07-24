"""
Audio Pipeline Manager for AI Anchor system.

Orchestrates STT, LLM, TTS, and VAD models in a real-time audio processing pipeline.
Adapted from RealtimeVoiceChat/code/speech_pipeline_manager.py architecture.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty
from typing import Optional, Callable, Dict, Any, List, AsyncIterator
from contextlib import asynccontextmanager

from loguru import logger

from config import Settings, AudioChunk, TranscriptionResult, LanguageCode
from models.stt_module import VoxtralSTT
from models.tts_module import HiggsAudioTTS, VoiceProfile
from models.llm_module import HuihuiAILLM
from models.vad_module import TurnDetectionVAD
from core.memory_manager import MemoryManager, ModelType
from utils.logger import log_performance, log_gpu_memory


class PipelineState(str, Enum):
    """States of the audio pipeline."""
    IDLE = "idle"
    PROCESSING = "processing"
    GENERATING = "generating"
    SPEAKING = "speaking"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class RequestType(str, Enum):
    """Types of pipeline requests."""
    PROCESS_AUDIO = "process_audio"
    GENERATE_RESPONSE = "generate_response"
    ABORT_GENERATION = "abort_generation"
    CHANGE_VOICE = "change_voice"
    CHANGE_LANGUAGE = "change_language"


@dataclass
class PipelineRequest:
    """
    Represents a request to be processed by the audio pipeline.
    
    Adapted from RealtimeVoiceChat PipelineRequest pattern.
    """
    request_type: RequestType
    data: Optional[Any] = None
    timestamp: float = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class GenerationSession:
    """
    Manages state for an ongoing conversation generation.
    
    Adapted from RealtimeVoiceChat RunningGeneration pattern.
    """
    session_id: str
    input_text: str
    language: LanguageCode
    voice_profile: Optional[str] = None
    
    # Processing state
    stt_completed: bool = False
    llm_started: bool = False
    llm_completed: bool = False
    tts_started: bool = False
    tts_completed: bool = False
    
    # Generated content
    transcription_result: Optional[TranscriptionResult] = None
    llm_response: str = ""
    audio_chunks: List[AudioChunk] = None
    
    # Threading events
    llm_ready_event: threading.Event = None
    tts_ready_event: threading.Event = None
    completion_event: threading.Event = None
    
    # Error handling
    error: Optional[Exception] = None
    aborted: bool = False
    
    def __post_init__(self):
        if self.audio_chunks is None:
            self.audio_chunks = []
        if self.llm_ready_event is None:
            self.llm_ready_event = threading.Event()
        if self.tts_ready_event is None:
            self.tts_ready_event = threading.Event()
        if self.completion_event is None:
            self.completion_event = threading.Event()


class AudioPipelineManager:
    """
    Orchestrates the complete AI Anchor audio processing pipeline.
    
    Manages STT, LLM, TTS, and VAD models in coordination for real-time
    voice-to-voice conversation. Adapted from RealtimeVoiceChat architecture
    but integrated with AI Anchor's specific models and requirements.
    """
    
    def __init__(
        self,
        settings: Settings,
        memory_manager: MemoryManager,
        stt_model: Optional[VoxtralSTT] = None,
        tts_model: Optional[HiggsAudioTTS] = None,
        llm_model: Optional[HuihuiAILLM] = None,
        vad_model: Optional[TurnDetectionVAD] = None,
    ):
        """
        Initialize the audio pipeline manager.
        
        Args:
            settings: Application settings
            memory_manager: Memory management system
            stt_model: Speech-to-text model instance
            tts_model: Text-to-speech model instance
            llm_model: Large language model instance
            vad_model: Voice activity detection model instance
        """
        self.settings = settings
        self.memory_manager = memory_manager
        
        # Model instances (will be initialized if not provided)
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.llm_model = llm_model
        self.vad_model = vad_model
        
        # Pipeline state
        self.state = PipelineState.IDLE
        self.current_session: Optional[GenerationSession] = None
        self.session_counter = 0
        
        # Request processing
        self.request_queue: Queue[PipelineRequest] = Queue()
        self.audio_input_queue: Queue[bytes] = Queue()
        
        # Threading and synchronization
        self.shutdown_event = threading.Event()
        self.state_lock = threading.RLock()
        
        # Worker threads
        self.request_worker: Optional[threading.Thread] = None
        self.audio_worker: Optional[threading.Thread] = None
        self.stt_worker: Optional[threading.Thread] = None
        self.llm_worker: Optional[threading.Thread] = None
        self.tts_worker: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_transcription: Optional[Callable[[TranscriptionResult], None]] = None
        self.on_response_start: Optional[Callable[[str], None]] = None
        self.on_response_chunk: Optional[Callable[[str], None]] = None
        self.on_audio_chunk: Optional[Callable[[AudioChunk], None]] = None
        self.on_session_complete: Optional[Callable[[GenerationSession], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Performance tracking
        self.pipeline_stats = {
            "sessions_processed": 0,
            "total_processing_time": 0.0,
            "average_latency": 0.0,
            "stt_latency": 0.0,
            "llm_latency": 0.0,
            "tts_latency": 0.0,
            "errors": 0,
        }
        
        logger.info("🎵 AudioPipelineManager initialized")
    
    async def initialize(self) -> None:
        """
        Initialize all models and start worker threads.
        """
        logger.info("🎵 Initializing audio pipeline...")
        
        try:
            # Initialize models if not provided
            await self._initialize_models()
            
            # Start worker threads
            self._start_workers()
            
            # Set up VAD callback if available
            if self.vad_model:
                self.vad_model.on_new_waiting_time = self._on_vad_waiting_time
            
            self.state = PipelineState.IDLE
            logger.info("✅ Audio pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize audio pipeline: {e}")
            self.state = PipelineState.ERROR
            raise
    
    async def _initialize_models(self) -> None:
        """Initialize AI models if not already provided."""
        # Initialize STT model (optional - skip if fails)
        if self.stt_model is None:
            try:
                logger.info("🎤 Initializing STT model...")
                self.stt_model = VoxtralSTT(
                    settings=self.settings,
                    realtime_callback=self._on_realtime_transcription,
                    final_callback=self._on_final_transcription,
                    error_callback=self._on_stt_error,
                )
                await self.stt_model.initialize()  
                self.memory_manager.register_model(ModelType.STT, self.stt_model)
                logger.info("✅ STT model initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ STT model initialization failed: {e}")
                logger.info("🎤 Continuing without STT model - text input only mode")
                self.stt_model = None
        
        # Initialize TTS model (optional - skip if fails)
        if self.tts_model is None:
            try:
                logger.info("🗣️ Initializing TTS model...")
                self.tts_model = HiggsAudioTTS(settings=self.settings)
                await self.tts_model.initialize()
                self.memory_manager.register_model(ModelType.TTS, self.tts_model)
                logger.info("✅ TTS model initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ TTS model initialization failed: {e}")
                logger.info("🗣️ Continuing without TTS model - text output only mode")
                self.tts_model = None
        
        # Initialize LLM model
        if self.llm_model is None:
            logger.info("🧠 Initializing LLM model...")
            self.llm_model = HuihuiAILLM(settings=self.settings)
            await self.llm_model.prewarm()
            self.memory_manager.register_model(ModelType.LLM, self.llm_model)
        
        # Initialize VAD model (optional - skip if fails)
        if self.vad_model is None:
            try:
                logger.info("👂 Initializing VAD model...")
                self.vad_model = TurnDetectionVAD(
                    settings=self.settings,
                    on_new_waiting_time=self._on_vad_waiting_time,
                    pipeline_latency=0.5,
                )
                self.memory_manager.register_model(ModelType.VAD, self.vad_model)
                logger.info("✅ VAD model initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ VAD model initialization failed: {e}")
                logger.info("👂 Continuing without VAD model - no voice activity detection")
                self.vad_model = None
    
    def _start_workers(self) -> None:
        """Start all worker threads."""
        logger.info("🎵 Starting pipeline worker threads...")
        
        # Request processing worker
        self.request_worker = threading.Thread(
            target=self._request_processing_worker,
            name="PipelineRequestWorker",
            daemon=True
        )
        self.request_worker.start()
        
        # Audio input worker
        self.audio_worker = threading.Thread(
            target=self._audio_input_worker,
            name="AudioInputWorker", 
            daemon=True
        )
        self.audio_worker.start()
        
        logger.info("✅ Pipeline workers started")
    
    def _request_processing_worker(self) -> None:
        """
        Worker thread for processing pipeline requests.
        
        Adapted from RealtimeVoiceChat request processing pattern.
        """
        logger.info("🎵 Request processing worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get request with timeout
                request = self.request_queue.get(timeout=1.0)
                
                # Drain queue to get most recent request
                while not self.request_queue.empty():
                    try:
                        older_request = self.request_queue.get_nowait()
                        logger.debug(f"🎵 Skipping older request: {older_request.request_type}")
                        request = older_request
                    except Empty:
                        break
                
                # Process the request
                asyncio.run(self._process_request(request))
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"🎵 Error in request processing: {e}")
                if self.on_error:
                    self.on_error(e)
        
        logger.info("🎵 Request processing worker stopped")
    
    def _audio_input_worker(self) -> None:
        """
        Worker thread for processing incoming audio chunks.
        """
        logger.info("🎵 Audio input worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_input_queue.get(timeout=1.0)
                logger.debug(f"🎵 Audio input worker got {len(audio_chunk)} bytes from queue")
                
                # Feed to STT model if available
                if self.stt_model:
                    self.stt_model.feed_audio_chunk(audio_chunk)
                    self.memory_manager.update_model_usage(ModelType.STT)
                    logger.debug("🎵 Audio chunk fed to STT model")
                else:
                    logger.warning("🎵 No STT model available to process audio")
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"🎵 Error in audio processing: {e}")
        
        logger.info("🎵 Audio input worker stopped")
    
    async def _process_request(self, request: PipelineRequest) -> None:
        """
        Process a pipeline request.
        
        Args:
            request: Request to process
        """
        logger.debug(f"🎵 Processing request: {request.request_type}")
        
        try:
            if request.request_type == RequestType.PROCESS_AUDIO:
                await self._handle_audio_processing(request)
            elif request.request_type == RequestType.GENERATE_RESPONSE:
                await self._handle_response_generation(request)
            elif request.request_type == RequestType.ABORT_GENERATION:
                await self._handle_abort_generation(request)
            elif request.request_type == RequestType.CHANGE_VOICE:
                await self._handle_voice_change(request)
            elif request.request_type == RequestType.CHANGE_LANGUAGE:
                await self._handle_language_change(request)
            else:
                logger.warning(f"🎵 Unknown request type: {request.request_type}")
                
        except Exception as e:
            logger.error(f"🎵 Error processing request {request.request_type}: {e}")
            if self.on_error:
                self.on_error(e)
    
    async def _handle_audio_processing(self, request: PipelineRequest) -> None:
        """Handle audio processing request."""
        audio_data = request.data
        if not audio_data:
            return
        
        # Process buffered audio through STT
        if self.stt_model:
            logger.info("🎵 Processing buffered audio through STT...")
            with self.memory_manager.memory_context(ModelType.STT):
                result = await self.stt_model.process_buffered_audio()
                if result and result.text.strip():
                    logger.info(f"🎵 STT result: '{result.text}' (confidence: {result.confidence:.2f}, partial: {result.is_partial})")
                    
                    # Trigger VAD analysis
                    if self.vad_model:
                        self.vad_model.calculate_waiting_time(result.text)
                    
                    # Start LLM generation if this is a final transcription
                    if not result.is_partial:  # is_partial=False means final
                        logger.info("🎵 Starting response generation for final transcription")
                        await self._start_response_generation(result)
                    else:
                        logger.debug("🎵 Partial transcription, waiting for more audio")
                else:
                    logger.debug("🎵 No valid transcription result from STT")
        else:
            logger.warning("🎵 STT model not available")
    
    async def _handle_response_generation(self, request: PipelineRequest) -> None:
        """Handle response generation request."""
        transcription_result = request.data
        if not isinstance(transcription_result, TranscriptionResult):
            return
        
        await self._start_response_generation(transcription_result)
    
    async def _handle_abort_generation(self, request: PipelineRequest) -> None:
        """Handle generation abort request."""
        with self.state_lock:
            if self.current_session:
                self.current_session.aborted = True
                logger.info(f"🎵 Aborting session: {self.current_session.session_id}")
                
                # Cancel LLM generation
                if self.llm_model and self.current_session.llm_started:
                    self.llm_model.cancel_generation()
                
                # Cancel TTS synthesis  
                if self.tts_model and self.current_session.tts_started:
                    self.tts_model.stop_event.set()
                
                self.current_session.completion_event.set()
    
    async def _handle_voice_change(self, request: PipelineRequest) -> None:
        """Handle voice profile change request."""
        voice_name = request.data
        if self.tts_model and voice_name:
            success = self.tts_model.set_voice_profile(voice_name)
            if success:
                logger.info(f"🎵 Changed voice profile to: {voice_name}")
            else:
                logger.warning(f"🎵 Failed to change voice profile to: {voice_name}")
    
    async def _handle_language_change(self, request: PipelineRequest) -> None:
        """Handle language change request."""
        language = request.data
        if isinstance(language, LanguageCode):
            if self.stt_model:
                self.stt_model.set_language(language)
            logger.info(f"🎵 Changed language to: {language.value}")
    
    @log_performance("Response generation")
    async def _start_response_generation(self, transcription: TranscriptionResult) -> None:
        """
        Start generating a response to transcribed text.
        
        Args:
            transcription: Final transcription result
        """
        with self.state_lock:
            # Create new generation session
            self.session_counter += 1
            session = GenerationSession(
                session_id=f"session-{self.session_counter}",
                input_text=transcription.text,
                language=transcription.language,
                transcription_result=transcription,
            )
            
            # Abort any current session
            if self.current_session and not self.current_session.aborted:
                self.current_session.aborted = True
                self.current_session.completion_event.set()
            
            self.current_session = session
            self.state = PipelineState.GENERATING
        
        logger.info(f"🎵 Starting response generation for: '{transcription.text[:50]}...'")
        
        try:
            # Start LLM generation in background
            asyncio.create_task(self._generate_llm_response(session))
            
        except Exception as e:
            logger.error(f"🎵 Error starting response generation: {e}")
            session.error = e
            session.completion_event.set()
            if self.on_error:
                self.on_error(e)
    
    async def _generate_llm_response(self, session: GenerationSession) -> None:
        """
        Generate LLM response for a session.
        
        Args:
            session: Generation session
        """
        if session.aborted:
            return
        
        try:
            with self.state_lock:
                session.llm_started = True
            
            # Update memory usage
            self.memory_manager.update_model_usage(ModelType.LLM)
            
            # Generate response using streaming
            with self.memory_manager.memory_context(ModelType.LLM):
                generator = self.llm_model.generate(
                    text=session.input_text,
                    use_system_prompt=True,
                    request_id=session.session_id,
                    temperature=self.settings.conversation.temperature,
                )
                
                # Trigger response start callback
                if self.on_response_start:
                    self.on_response_start(session.input_text)
                
                # Process streaming tokens
                response_parts = []
                for token in generator:
                    if session.aborted:
                        break
                    
                    response_parts.append(token)
                    session.llm_response = "".join(response_parts)
                    
                    # Trigger chunk callback
                    if self.on_response_chunk:
                        self.on_response_chunk(token)
                
                # Complete LLM generation
                if not session.aborted:
                    with self.state_lock:
                        session.llm_completed = True
                        session.llm_ready_event.set()
                    
                    # Start TTS synthesis
                    await self._synthesize_response(session)
                
        except Exception as e:
            logger.error(f"🎵 Error in LLM generation: {e}")
            session.error = e
            session.completion_event.set()
            if self.on_error:
                self.on_error(e)
    
    async def _synthesize_response(self, session: GenerationSession) -> None:
        """
        Synthesize TTS audio for the generated response.
        
        Args:
            session: Generation session
        """
        if session.aborted or not session.llm_response:
            return
        
        try:
            with self.state_lock:
                session.tts_started = True
                self.state = PipelineState.SPEAKING
            
            # Update memory usage
            self.memory_manager.update_model_usage(ModelType.TTS)
            
            # Synthesize audio
            with self.memory_manager.memory_context(ModelType.TTS):
                audio_queue = Queue()
                stop_event = threading.Event()
                
                # Run synthesis in thread
                def synthesis_worker():
                    try:
                        success = self.tts_model.synthesize(
                            text=session.llm_response,
                            audio_chunks=audio_queue,
                            stop_event=stop_event,
                            generation_string=f"Session-{session.session_id}",
                        )
                        
                        if success:
                            logger.info(f"🎵 TTS synthesis completed for session: {session.session_id}")
                        else:
                            logger.warning(f"🎵 TTS synthesis failed for session: {session.session_id}")
                            
                    except Exception as e:
                        logger.error(f"🎵 Error in TTS synthesis: {e}")
                        session.error = e
                    finally:
                        session.tts_ready_event.set()
                
                # Start synthesis
                synthesis_thread = threading.Thread(target=synthesis_worker, daemon=True)
                synthesis_thread.start()
                
                # Process audio chunks as they're generated
                while not session.tts_ready_event.is_set() or not audio_queue.empty():
                    try:
                        audio_chunk_bytes = audio_queue.get(timeout=0.1)
                        
                        # Create AudioChunk object
                        audio_chunk = AudioChunk(
                            data=audio_chunk_bytes,
                            format=self.settings.audio.format,
                            sample_rate=self.settings.audio.sample_rate,
                            timestamp=time.time(),
                            duration=len(audio_chunk_bytes) / (self.settings.audio.sample_rate * 2)
                        )
                        
                        session.audio_chunks.append(audio_chunk)
                        
                        # Trigger audio chunk callback
                        if self.on_audio_chunk:
                            self.on_audio_chunk(audio_chunk)
                            
                        if session.aborted:
                            stop_event.set()
                            break
                            
                    except Empty:
                        continue
                
                # Wait for synthesis to complete
                synthesis_thread.join(timeout=5.0)
                
                # Complete session
                if not session.aborted:
                    with self.state_lock:
                        session.tts_completed = True
                        self.state = PipelineState.IDLE
                    
                    # Update statistics
                    self.pipeline_stats["sessions_processed"] += 1
                    
                    # Trigger completion callback
                    if self.on_session_complete:
                        self.on_session_complete(session)
                
                session.completion_event.set()
                
        except Exception as e:
            logger.error(f"🎵 Error in TTS synthesis: {e}")
            session.error = e
            session.completion_event.set()
            if self.on_error:
                self.on_error(e)
    
    def _on_realtime_transcription(self, result: TranscriptionResult) -> None:
        """Handle real-time transcription updates."""
        if self.on_transcription:
            self.on_transcription(result)
    
    def _on_final_transcription(self, result: TranscriptionResult) -> None:
        """Handle final transcription results."""
        if self.on_transcription:
            self.on_transcription(result)
        
        # Queue response generation
        request = PipelineRequest(
            request_type=RequestType.GENERATE_RESPONSE,
            data=result,
        )
        self.request_queue.put(request)
    
    def _on_stt_error(self, error: Exception) -> None:
        """Handle STT errors."""
        logger.error(f"🎵 STT error: {error}")
        if self.on_error:
            self.on_error(error)
    
    def _on_vad_waiting_time(self, waiting_time: float, text: Optional[str] = None) -> None:
        """Handle VAD waiting time updates."""
        logger.debug(f"🎵 VAD waiting time: {waiting_time:.2f}s for text: {text}")
        # Could be used to adjust pipeline timing
    
    # Public interface methods
    
    def feed_audio(self, audio_data: bytes) -> None:
        """
        Feed audio data to the pipeline.
        
        Args:
            audio_data: Raw audio bytes
        """
        logger.debug(f"🎵 Received {len(audio_data)} bytes of audio data")
        
        # Log pipeline state for debugging
        logger.debug(f"🎵 Pipeline state: {self.state}, shutdown: {self.shutdown_event.is_set()}")
        logger.debug(f"🎵 Audio queue size: {self.audio_input_queue.qsize()}/{self.audio_input_queue.maxsize if hasattr(self.audio_input_queue, 'maxsize') else 'unlimited'}")
        logger.debug(f"🎵 STT model available: {self.stt_model is not None}")
        
        if not self.shutdown_event.is_set():
            try:
                # Feed audio directly to STT for immediate buffering
                if self.stt_model:
                    logger.debug("🎵 Feeding audio directly to STT buffer...")
                    self.stt_model.feed_audio_chunk(audio_data)
                    logger.debug("🎵 Audio fed to STT buffer successfully")
                    
                    # Check buffer status after feeding
                    buffer_size = len(self.stt_model.audio_buffer)
                    logger.debug(f"🎵 STT buffer now has {buffer_size} chunks")
                    
                    # Trigger processing when we have enough buffered audio
                    if buffer_size >= 3:  # Process when we have 3+ chunks for better context
                        logger.debug("🎵 Buffer threshold reached, triggering audio processing...")
                        request = PipelineRequest(
                            request_type=RequestType.PROCESS_AUDIO,
                            data=audio_data,
                        )
                        self.request_queue.put_nowait(request)
                        logger.debug("🎵 Audio processing request queued")
                else:
                    logger.warning("🎵 No STT model available to process audio")
                
                # Also queue for background processing via worker
                self.audio_input_queue.put_nowait(audio_data)
                logger.debug("🎵 Audio data queued for background processing")
                
            except Exception as e:
                logger.error(f"🎵 Failed to process audio data: {e}")
                logger.warning("🎵 Audio processing failed, dropping audio chunk")
    
    def process_text_input(self, text: str, language: LanguageCode = LanguageCode.EN) -> None:
        """
        Process direct text input (bypass STT).
        
        Args:
            text: Input text
            language: Language of the text
        """
        # Create synthetic transcription result
        transcription = TranscriptionResult(
            text=text,
            language=language,
            confidence=1.0,
            start_time=time.time(),
            end_time=time.time(),
            is_partial=False,  # This is a complete text input
        )
        
        request = PipelineRequest(
            request_type=RequestType.GENERATE_RESPONSE,
            data=transcription,
        )
        self.request_queue.put(request)
    
    def abort_current_generation(self) -> None:
        """Abort the current generation session."""
        request = PipelineRequest(request_type=RequestType.ABORT_GENERATION)
        self.request_queue.put(request)
    
    def change_voice_profile(self, voice_name: str) -> None:
        """
        Change the active voice profile.
        
        Args:
            voice_name: Name of the voice profile
        """
        request = PipelineRequest(
            request_type=RequestType.CHANGE_VOICE,
            data=voice_name,
        )
        self.request_queue.put(request)
    
    def change_language(self, language: LanguageCode) -> None:
        """
        Change the active language.
        
        Args:
            language: Language code
        """
        request = PipelineRequest(
            request_type=RequestType.CHANGE_LANGUAGE,
            data=language,
        )
        self.request_queue.put(request)
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """
        Get current pipeline state information.
        
        Returns:
            State information dictionary
        """
        with self.state_lock:
            return {
                "state": self.state.value,
                "current_session": self.current_session.session_id if self.current_session else None,
                "session_counter": self.session_counter,
                "request_queue_size": self.request_queue.qsize(),
                "audio_queue_size": self.audio_input_queue.qsize(),
                "statistics": self.pipeline_stats.copy(),
                "model_status": {
                    "stt_loaded": self.stt_model is not None,
                    "tts_loaded": self.tts_model is not None,
                    "llm_loaded": self.llm_model is not None,
                    "vad_loaded": self.vad_model is not None,
                }
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the audio pipeline.
        
        Returns:
            Health status dictionary
        """
        status = {
            "pipeline_state": self.state.value,
            "workers_active": {
                "request_worker": self.request_worker.is_alive() if self.request_worker else False,
                "audio_worker": self.audio_worker.is_alive() if self.audio_worker else False,
            },
            "queue_sizes": {
                "requests": self.request_queue.qsize(),
                "audio_input": self.audio_input_queue.qsize(),
            },
            "model_health": {},
            "statistics": self.pipeline_stats.copy(),
        }
        
        # Check model health
        try:
            if self.stt_model:
                status["model_health"]["stt"] = await self.stt_model.health_check()
            if self.tts_model:
                status["model_health"]["tts"] = await self.tts_model.health_check()
            if self.llm_model:
                status["model_health"]["llm"] = await self.llm_model.health_check()
            if self.vad_model:
                status["model_health"]["vad"] = await self.vad_model.health_check()
        except Exception as e:
            status["model_health"]["error"] = str(e)
        
        # Overall health assessment
        if self.state == PipelineState.ERROR:
            status["overall_health"] = "error"
        elif self.state == PipelineState.SHUTDOWN:
            status["overall_health"] = "shutdown"
        elif all(worker["request_worker"] and worker["audio_worker"] 
                for worker in [status["workers_active"]]):
            status["overall_health"] = "healthy"
        else:
            status["overall_health"] = "degraded"
        
        return status
    
    async def shutdown(self) -> None:
        """Shutdown the audio pipeline and cleanup resources."""
        logger.info("🎵 Shutting down audio pipeline...")
        
        self.state = PipelineState.SHUTDOWN
        self.shutdown_event.set()
        
        # Abort current session
        if self.current_session:
            self.current_session.aborted = True
            self.current_session.completion_event.set()
        
        # Wait for workers to finish
        workers = [self.request_worker, self.audio_worker]
        for worker in workers:
            if worker and worker.is_alive():
                worker.join(timeout=2.0)
        
        # Shutdown models
        if self.stt_model:
            await self.stt_model.shutdown()
        if self.tts_model:
            await self.tts_model.shutdown()
        if self.llm_model:
            await self.llm_model.shutdown()
        if self.vad_model:
            self.vad_model.shutdown()
        
        # Clear queues
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except Empty:
                break
        
        while not self.audio_input_queue.empty():
            try:
                self.audio_input_queue.get_nowait()
            except Empty:
                break
        
        logger.info("✅ Audio pipeline shutdown complete")