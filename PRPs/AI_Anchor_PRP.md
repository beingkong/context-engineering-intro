# AI Anchor Project PRP
name: "AI Anchor - Real-time Voice Broadcasting System"
description: |

## Purpose
Build a comprehensive AI anchor system for real-time voice broadcasting with advanced AI models, leveraging existing RealtimeVoiceChat architecture while integrating cutting-edge TTS, STT, and LLM technologies optimized for RTX6000 48GB GPU.

## Core Principles
1. **Hardware Optimization**: Leverage RTX6000 48GB for concurrent model operation
2. **Real-time Performance**: Minimize latency in audio processing pipeline
3. **Model Integration**: Seamlessly integrate higgs-audio, Voxtral, and Ollama
4. **Environment Management**: Strict conda+uv dependency management
5. **Code Reusability**: Adapt proven patterns from RealtimeVoiceChat

---

## Goal
Create a production-ready AI anchor system capable of:
- Real-time voice-to-voice conversation with <2 second latency
- Zero-shot voice cloning for dynamic anchor personalities
- Multilingual support (8 languages via Voxtral)
- Robust interruption handling and conversation flow management
- Scalable architecture supporting concurrent model operation

## Why
- **Broadcasting Innovation**: Enable AI-powered broadcasting with human-like interaction
- **Resource Optimization**: Maximize RTX6000 48GB GPU utilization for concurrent AI models
- **Market Demand**: Real-time AI anchoring for news, entertainment, and educational content
- **Technical Advancement**: Showcase integration of latest AI models in production environment

## What
Real-time AI anchor system with web interface supporting:
- Voice input processing (STT via Voxtral-Mini-3B-2507)
- Intelligent response generation (LLM via Ollama/huihui-ai)
- Dynamic voice synthesis (TTS via higgs-audio v2)
- Smart conversation management with VAD
- Web-based interaction interface

### Success Criteria
- [ ] Complete voice-to-voice pipeline with <2s end-to-end latency
- [ ] Support for 8 languages (EN, ES, FR, PT, HI, DE, NL, IT)
- [ ] Zero-shot voice cloning capability operational
- [ ] All models running concurrently within 48GB GPU memory limit
- [ ] Robust error handling and graceful degradation
- [ ] Comprehensive test coverage (>80%)
- [ ] Production-ready deployment with Docker support

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://github.com/boson-ai/higgs-audio/blob/main/README.md
  why: higgs-audio v2 TTS model setup, API usage, and Docker deployment
  
- url: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
  why: Voxtral STT model capabilities, vLLM integration, multilingual support
  
- url: https://huggingface.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF
  why: LLM model specifications, memory requirements, prompt format
  
- file: /root/context-engineering-intro/examples/RealtimeVoiceChat/code/turndetect.py
  why: VAD implementation pattern, TurnDetection class architecture
  
- file: /root/context-engineering-intro/examples/RealtimeVoiceChat/code/speech_pipeline_manager.py
  why: Pipeline orchestration, worker thread management, audio streaming
  
- file: /root/context-engineering-intro/examples/RealtimeVoiceChat/code/audio_module.py
  why: TTS integration patterns, buffering logic, WebSocket streaming
  
- file: /root/context-engineering-intro/CLAUDE.md
  why: Project development guidelines, coding standards, testing requirements
  
- file: /root/context-engineering-intro/INITIAL.md
  why: Model specifications, hardware requirements, architecture considerations
```

### Current Codebase tree
```bash
/root/context-engineering-intro/
├── examples/
│   └── RealtimeVoiceChat/
│       ├── code/
│       │   ├── audio_module.py          # TTS engine integration
│       │   ├── speech_pipeline_manager.py # Pipeline orchestration
│       │   ├── turndetect.py            # VAD implementation
│       │   ├── transcribe.py            # STT processing
│       │   ├── llm_module.py            # LLM integration
│       │   ├── server.py                # FastAPI server
│       │   └── static/                  # Web interface
│       ├── requirements.txt
│       └── docker-compose.yml
├── CLAUDE.md                           # Development guidelines
└── INITIAL.md                          # Project specifications
```

### Desired Codebase tree with files to be added
```bash
/root/context-engineering-intro/examples/AI-Anchor/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment
├── docker-compose.yml                  # Container orchestration
├── .env.example                        # Environment variables template
├── src/
│   ├── __init__.py
│   ├── main.py                         # Application entry point
│   ├── config.py                       # Configuration management
│   ├── models/                         # AI model integrations
│   │   ├── __init__.py
│   │   ├── stt_module.py              # Voxtral-Mini-3B-2507 integration
│   │   ├── tts_module.py              # higgs-audio v2 integration
│   │   ├── llm_module.py              # Ollama/huihui-ai integration
│   │   └── vad_module.py              # VAD from turndetect.py
│   ├── core/                          # Core functionality
│   │   ├── __init__.py
│   │   ├── anchor_agent.py            # Main anchor logic
│   │   ├── audio_pipeline.py          # Audio processing pipeline
│   │   ├── conversation_manager.py    # Dialog management
│   │   └── memory_manager.py          # GPU memory optimization
│   ├── utils/                         # Utility functions
│   │   ├── __init__.py
│   │   ├── text_processing.py         # Text preprocessing
│   │   ├── audio_utils.py             # Audio utilities
│   │   └── logger.py                  # Logging configuration
│   └── web/                           # Web interface
│       ├── __init__.py
│       ├── server.py                  # FastAPI server
│       ├── websocket_handler.py       # WebSocket management
│       └── static/                    # Frontend assets
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest configuration
│   ├── test_models/                   # Model integration tests
│   ├── test_core/                     # Core functionality tests
│   └── test_utils/                    # Utility tests
└── docs/                              # Documentation
    ├── api.md                         # API documentation
    ├── setup.md                       # Setup instructions
    └── architecture.md                # System architecture
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: conda + uv integration requires specific setup order
# 1. Create conda environment first: conda create -n ai-anchor python=3.10
# 2. Activate environment: conda activate ai-anchor  
# 3. Install uv in conda: conda install uv
# 4. Use uv for packages: uv pip install <package>

# CRITICAL: GPU memory management with concurrent models
# Total: ~47.5GB (STT: 9.5GB + TTS: 24GB + LLM: 14GB)
# Must implement lazy loading and memory monitoring

# CRITICAL: higgs-audio requires specific PyTorch version
# Use PyTorch 2.5.1+cu121 for CUDA 12.1 compatibility
# Install order: PyTorch first, then higgs-audio dependencies

# CRITICAL: Voxtral requires vLLM framework
# vLLM has specific CUDA requirements and memory allocation
# Temperature settings: 0.2 for chat, 0.0 for transcription

# CRITICAL: Ollama model path format
# Use: hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M
# Prompt format: <s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{prompt}[/INST]

# CRITICAL: Audio pipeline threading
# RealtimeVoiceChat uses complex threading - maintain similar architecture
# Use threading.Event for synchronization, Queue for audio chunks
```

## Implementation Blueprint

### Data models and structure
```python
# Core data models for type safety and consistency
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"

class LanguageCode(str, Enum):
    EN = "en"
    ES = "es" 
    FR = "fr"
    PT = "pt"
    HI = "hi"
    DE = "de"
    NL = "nl"
    IT = "it"

class AudioChunk(BaseModel):
    data: bytes
    format: AudioFormat
    sample_rate: int = 24000
    timestamp: float
    duration: float

class TranscriptionResult(BaseModel):
    text: str
    language: LanguageCode
    confidence: float
    timestamp: float
    is_final: bool = False

class AnchorResponse(BaseModel):
    text: str
    audio_chunks: List[AudioChunk]
    processing_time: float
    model_metadata: Dict[str, Any]

class ConversationState(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]
    current_language: LanguageCode
    voice_profile: Optional[str] = None
    is_active: bool = True
```

### List of tasks to be completed in order

```yaml
Task 1 - Environment Setup:
CREATE environment.yml:
  - Base conda environment with Python 3.10
  - CUDA toolkit and PyTorch dependencies
  - Basic project structure
  
CREATE requirements.txt:
  - All Python dependencies with pinned versions
  - higgs-audio, vllm, ollama-python
  - FastAPI, WebSocket, testing frameworks

Task 2 - Configuration Management:
CREATE src/config.py:
  - MIRROR pattern from: examples/RealtimeVoiceChat/code/server.py
  - Add model paths, GPU settings, API configurations
  - Environment variable management with pydantic-settings

Task 3 - STT Module Integration:
CREATE src/models/stt_module.py:
  - Integrate Voxtral-Mini-3B-2507 via vLLM
  - ADAPT pattern from: examples/RealtimeVoiceChat/code/transcribe.py
  - Implement multilingual transcription pipeline
  - Add real-time audio chunk processing

Task 4 - TTS Module Integration:
CREATE src/models/tts_module.py:
  - Integrate higgs-audio v2 generation model
  - REPLACE RealtimeTTS patterns from audio_module.py
  - Implement zero-shot voice cloning
  - Add audio streaming capabilities

Task 5 - LLM Module Integration:
CREATE src/models/llm_module.py:
  - ADAPT from: examples/RealtimeVoiceChat/code/llm_module.py
  - Integrate Ollama with huihui-ai model
  - Implement conversation history management
  - Add streaming response generation

Task 6 - VAD Module Integration:
CREATE src/models/vad_module.py:
  - COPY and ADAPT: examples/RealtimeVoiceChat/code/turndetect.py
  - Port TurnDetection class entirely
  - Modify for integration with Voxtral STT
  - Maintain threading and caching patterns

Task 7 - Memory Management:
CREATE src/core/memory_manager.py:
  - Implement GPU memory monitoring
  - Add model loading/unloading strategies
  - Create memory optimization utilities
  - Add CUDA memory leak detection

Task 8 - Audio Pipeline:
CREATE src/core/audio_pipeline.py:
  - MIRROR architecture from: speech_pipeline_manager.py
  - Integrate all models (STT, TTS, LLM, VAD)
  - Implement worker thread management
  - Add audio chunk streaming and buffering

Task 9 - Anchor Agent:
CREATE src/core/anchor_agent.py:
  - Main orchestration logic
  - Conversation state management  
  - Voice personality switching
  - Error handling and recovery

Task 10 - Web Interface:
CREATE src/web/server.py:
  - ADAPT from: examples/RealtimeVoiceChat/code/server.py
  - FastAPI application with WebSocket support
  - Audio streaming endpoints
  - Real-time conversation interface

MODIFY src/web/static/:
  - COPY and ADAPT frontend from RealtimeVoiceChat/static/
  - Update for new model capabilities
  - Add voice profile selection
  - Language selection interface

Task 11 - Testing Suite:
CREATE comprehensive test suite:
  - Unit tests for each model integration
  - Integration tests for audio pipeline
  - Performance benchmarks
  - Memory usage validation

Task 12 - Documentation & Deployment:
CREATE deployment configurations:
  - Docker containerization
  - Environment setup scripts
  - API documentation
  - Architecture documentation
```

### Per task pseudocode as needed

```python
# Task 3 - STT Module Integration
class VoxtralSTT:
    def __init__(self, model_path: str, device: str = "cuda"):
        # PATTERN: Lazy loading for memory efficiency
        self.model = None  # Load on first use
        self.vllm_engine = None
        
    async def transcribe_stream(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptionResult]:
        # CRITICAL: vLLM requires specific audio format
        async for chunk in audio_chunks:
            # GOTCHA: Voxtral expects specific sample rate
            processed_audio = self.preprocess_audio(chunk, target_sr=16000)
            
            # PATTERN: Batch processing for efficiency
            if len(self.audio_buffer) >= self.batch_size:
                results = await self.vllm_engine.generate(
                    self.audio_buffer,
                    temperature=0.0,  # For transcription
                    max_tokens=512
                )
                yield self.parse_transcription(results)

# Task 4 - TTS Module Integration  
class HiggsAudioTTS:
    def __init__(self, model_path: str, tokenizer_path: str):
        # CRITICAL: higgs-audio requires both model and tokenizer
        self.serve_engine = HiggsAudioServeEngine(
            model_path, tokenizer_path, device='cuda'
        )
        
    async def synthesize_stream(self, text_stream: AsyncIterator[str], voice_profile: str = None) -> AsyncIterator[AudioChunk]:
        # PATTERN: Zero-shot voice cloning setup
        if voice_profile:
            # Load reference audio for voice cloning
            ref_audio = self.load_reference_audio(voice_profile)
            
        async for text_chunk in text_stream:
            # GOTCHA: higgs-audio uses ChatML format
            messages = [{"role": "user", "content": text_chunk}]
            
            output = self.serve_engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=1024,
                temperature=0.3
            )
            
            # Convert to audio chunks
            audio_data = self.extract_audio(output)
            yield AudioChunk(
                data=audio_data,
                format=AudioFormat.WAV,
                sample_rate=24000,
                timestamp=time.time(),
                duration=len(audio_data) / (24000 * 2)  # 16-bit audio
            )
```

### Integration Points
```yaml
GPU_MEMORY:
  - allocation: "Implement CUDA memory pool management"
  - monitoring: "Add GPU memory usage tracking with nvidia-ml-py"
  - optimization: "Lazy loading and model swapping strategies"
  
CONDA_ENV:
  - create: "conda create -n ai-anchor python=3.10"
  - activate: "conda activate ai-anchor"
  - uv_install: "conda install uv && uv pip install -r requirements.txt"
  
MODELS:
  - download: "Automatic model downloading on first run"
  - caching: "Local model caching in ~/.cache/ai-anchor/"
  - validation: "Model integrity checks before loading"
  
WEBSOCKET:
  - add: "Real-time audio streaming endpoints"
  - pattern: "Bidirectional communication for voice chat"
  - error_handling: "Graceful connection management"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/ --fix                    # Auto-fix formatting
mypy src/                               # Type checking
conda-env-check environment.yml         # Validate conda env

# Expected: No errors. If errors, READ and fix before proceeding.
```

### Level 2: Unit Tests
```python
# CREATE comprehensive test suite

# Model Integration Tests
def test_voxtral_stt_basic():
    """Test Voxtral STT basic functionality"""
    stt = VoxtralSTT(model_path="mistralai/Voxtral-Mini-3B-2507")
    audio_data = load_test_audio("test_samples/hello_world.wav")
    result = stt.transcribe(audio_data)
    assert result.text.lower() == "hello world"
    assert result.language == LanguageCode.EN

def test_higgs_audio_tts():
    """Test higgs-audio TTS synthesis"""
    tts = HiggsAudioTTS("bosonai/higgs-audio-v2-generation-3B-base")
    text = "Hello, this is a test message."
    audio_chunks = list(tts.synthesize(text))
    assert len(audio_chunks) > 0
    assert all(chunk.sample_rate == 24000 for chunk in audio_chunks)

def test_memory_management():
    """Test concurrent model loading within memory limits"""
    manager = MemoryManager(max_gpu_memory_gb=48)
    
    # Load all models
    stt = manager.load_stt_model()
    tts = manager.load_tts_model()  
    llm = manager.load_llm_model()
    
    # Check memory usage
    usage = manager.get_gpu_memory_usage()
    assert usage.total_gb < 48.0
    assert usage.stt_gb < 10.0
    assert usage.tts_gb < 25.0
    assert usage.llm_gb < 17.0

def test_vad_integration():
    """Test VAD with new models"""
    vad = VADModule()
    text_chunks = ["Hello", "Hello world", "Hello world, how are you?"]
    
    for chunk in text_chunks:
        pause_time = vad.calculate_waiting_time(chunk)
        assert 0.1 <= pause_time <= 3.0  # Reasonable pause range
```

```bash
# Run and iterate until passing:
uv run pytest tests/ -v --cov=src --cov-report=html
# Target: >80% test coverage
```

### Level 3: Integration Test
```bash
# Test conda environment setup
conda env create -f environment.yml
conda activate ai-anchor
uv pip install -r requirements.txt

# Test model downloads and GPU allocation
python -m src.main --test-models

# Start the service
python -m src.main --dev

# Test real-time conversation
curl -X POST http://localhost:8000/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"language": "en", "voice_profile": "default"}'

# Test WebSocket audio streaming  
wscat -c ws://localhost:8000/ws/audio

# Expected: Successful voice-to-voice interaction with <2s latency
```

## Final validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `ruff check src/`
- [ ] No type errors: `mypy src/`
- [ ] Models load within GPU memory limits (<48GB)
- [ ] End-to-end latency <2 seconds
- [ ] All 8 languages supported
- [ ] Zero-shot voice cloning functional
- [ ] WebSocket real-time streaming works
- [ ] Error handling graceful
- [ ] Documentation complete

---

## Anti-Patterns to Avoid
- ❌ Don't load all models simultaneously without memory management
- ❌ Don't ignore CUDA out-of-memory errors - implement proper handling
- ❌ Don't use sync code in async audio processing pipelines
- ❌ Don't hardcode model paths - use configuration management
- ❌ Don't skip model validation - check integrity before loading
- ❌ Don't forget conda+uv environment isolation
- ❌ Don't copy-paste without understanding threading patterns from RealtimeVoiceChat
- ❌ Don't ignore audio format compatibility between models